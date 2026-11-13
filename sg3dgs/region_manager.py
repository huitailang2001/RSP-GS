#
# SG-3DGS: Region Manager
# Manages semantic region maps and boundary maps from SAM preprocessing
#

import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Region type constants
REGION_LINEAR_STRUCTURE = 0   # Linear structures (towers, wires, railings)
REGION_EDGE_BOUNDARY = 1      # Edge/boundary regions (building outlines, roof edges)
REGION_TEXTURED_SURFACE = 2   # Textured surfaces (brick walls, lawns)
REGION_FLAT_REGION = 3        # Flat regions (sky, roads, plain walls)


class RegionManager:
    """
    Manages per-image semantic region maps and boundary maps.
    Loads precomputed results from preprocess_region_classify.py.
    Provides efficient lookup for training.
    """

    def __init__(self, source_path: str, semantic_folder: str = "semantic"):
        self.source_path = source_path
        self.semantic_path = os.path.join(source_path, semantic_folder)
        self.region_maps = {}       # image_name -> (H, W) tensor on GPU, uint8
        self.boundary_maps = {}     # image_name -> (H, W) tensor on GPU, float32 [0,1]
        self.loaded = False

    @staticmethod
    def _normalize_name(image_name: str) -> str:
        """Strip file extension to match preprocessing output naming."""
        base = os.path.splitext(image_name)[0]
        return base

    def load(self):
        """Load all precomputed region maps and boundary maps."""
        if not os.path.exists(self.semantic_path):
            print(f"[RegionManager] Warning: semantic folder not found: {self.semantic_path}")
            print("[RegionManager] Will generate region maps on-the-fly from SAM masks.")
            self.loaded = False
            return False

        region_dir = os.path.join(self.semantic_path, "regions")
        boundary_dir = os.path.join(self.semantic_path, "boundaries")

        if os.path.exists(region_dir):
            for f in os.listdir(region_dir):
                if f.endswith("_region_map.npy"):
                    name = f.replace("_region_map.npy", "")
                    arr = np.load(os.path.join(region_dir, f))
                    self.region_maps[name] = torch.from_numpy(arr.astype(np.int64)).cuda()
            print(f"[RegionManager] Loaded {len(self.region_maps)} region maps.")

        if os.path.exists(boundary_dir):
            for f in os.listdir(boundary_dir):
                if f.endswith("_boundary.npy"):
                    name = f.replace("_boundary.npy", "")
                    arr = np.load(os.path.join(boundary_dir, f))
                    self.boundary_maps[name] = torch.from_numpy(arr.astype(np.float32)).cuda()
            print(f"[RegionManager] Loaded {len(self.boundary_maps)} boundary maps.")

        self.loaded = len(self.region_maps) > 0
        return self.loaded

    def get_region_map(self, image_name: str) -> Optional[torch.Tensor]:
        """Get region type map for an image. Returns (H, W) int64 tensor."""
        name = self._normalize_name(image_name)
        return self.region_maps.get(name, None)

    def get_boundary_map(self, image_name: str) -> Optional[torch.Tensor]:
        """Get boundary proximity map for an image. Returns (H, W) float32 tensor [0,1]."""
        name = self._normalize_name(image_name)
        return self.boundary_maps.get(name, None)

    def get_region_for_gaussians(self, positions_2d: torch.Tensor,
                                  region_map: torch.Tensor) -> torch.Tensor:
        """
        Query region types for Gaussian 2D positions.

        Args:
            positions_2d: (N, 2) pixel coordinates [x, y]
            region_map: (H, W) region type map

        Returns:
            (N,) region type per Gaussian
        """
        H, W = region_map.shape
        x = positions_2d[:, 0].long().clamp(0, W - 1)
        y = positions_2d[:, 1].long().clamp(0, H - 1)
        return region_map[y, x]

    def get_boundary_for_gaussians(self, positions_2d: torch.Tensor,
                                    boundary_map: torch.Tensor) -> torch.Tensor:
        """
        Query boundary proximity for Gaussian 2D positions.

        Args:
            positions_2d: (N, 2) pixel coordinates [x, y]
            boundary_map: (H, W) boundary proximity map [0,1]

        Returns:
            (N,) boundary proximity per Gaussian
        """
        H, W = boundary_map.shape
        x = positions_2d[:, 0].long().clamp(0, W - 1)
        y = positions_2d[:, 1].long().clamp(0, H - 1)
        return boundary_map[y, x]


def project_gaussians_to_2d(xyz: torch.Tensor, viewpoint_cam) -> torch.Tensor:
    """
    Project 3D Gaussian positions to 2D pixel coordinates.

    Args:
        xyz: (N, 3) world-space positions
        viewpoint_cam: Camera object with full_proj_transform, image_width, image_height

    Returns:
        (N, 2) pixel coordinates [x, y], and (N,) valid mask
    """
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device)
    xyz_hom = torch.cat([xyz, ones], dim=1)  # (N, 4)

    # Project: full_proj_transform is (4, 4)
    proj = xyz_hom @ viewpoint_cam.full_proj_transform  # (N, 4)

    # Perspective divide
    w = proj[:, 3:4].clamp(min=1e-6)
    ndc = proj[:, :3] / w  # (N, 3) in NDC [-1, 1]

    # NDC to pixel
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    px = ((ndc[:, 0] + 1.0) * 0.5) * W
    py = ((ndc[:, 1] + 1.0) * 0.5) * H

    positions_2d = torch.stack([px, py], dim=1)  # (N, 2)

    # Valid mask: in front of camera and within image bounds
    valid = (proj[:, 2] > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    return positions_2d, valid
