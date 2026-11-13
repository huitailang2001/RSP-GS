#
# SG-3DGS: Attention Manager
# Manages DINOv2 feature-gradient attention maps for MFG-Loss
#

import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional


def _normalize_name(image_name: str) -> str:
    """Strip file extension to match preprocessing output naming."""
    return os.path.splitext(image_name)[0]


class AttentionManager:
    """
    Manages per-image DINOv2 feature-gradient attention maps.
    Loads precomputed results from preprocess_dinov2.py.
    """

    def __init__(self, source_path: str, features_folder: str = "features"):
        self.source_path = source_path
        self.features_path = os.path.join(source_path, features_folder)
        self.attention_maps = {}   # image_name -> (H, W) tensor on GPU, float32
        self.loaded = False

    def load(self):
        """Load all precomputed attention maps."""
        if not os.path.exists(self.features_path):
            print(f"[AttentionManager] Warning: features folder not found: {self.features_path}")
            print("[AttentionManager] MFG-Loss will use uniform weights (fallback to standard loss).")
            self.loaded = False
            return False

        count = 0
        for f in os.listdir(self.features_path):
            if f.endswith("_attention.npy"):
                name = f.replace("_attention.npy", "")
                arr = np.load(os.path.join(self.features_path, f))
                self.attention_maps[name] = torch.from_numpy(
                    arr.astype(np.float32)
                ).cuda()
                count += 1

        print(f"[AttentionManager] Loaded {count} attention maps.")
        self.loaded = count > 0
        return self.loaded

    def get_attention_map(self, image_name: str,
                          target_h: int = None,
                          target_w: int = None) -> Optional[torch.Tensor]:
        """
        Get attention map for an image, optionally resized.

        Args:
            image_name: Image filename stem
            target_h: Target height (if resize needed)
            target_w: Target width (if resize needed)

        Returns:
            (H, W) float32 tensor with values in [0.3, 1.0], or None
        """
        attn = self.attention_maps.get(_normalize_name(image_name), None)
        if attn is None:
            return None

        if target_h is not None and target_w is not None:
            if attn.shape[0] != target_h or attn.shape[1] != target_w:
                attn = F.interpolate(
                    attn.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                # Re-normalize after interpolation
                attn = attn.clamp(0.3, 1.0)

        return attn
