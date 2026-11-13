#
# SG-3DGS: Structure-Preserving Gaussian Regularization (SP-Reg)
# Innovation 3: Semantic-driven anisotropy constraints, opacity regularization,
#               and boundary-aware scale constraints
#

import torch
import torch.nn.functional as F
from typing import Optional
from sg3dgs.region_manager import (
    REGION_LINEAR_STRUCTURE,
    REGION_EDGE_BOUNDARY,
    REGION_TEXTURED_SURFACE,
    REGION_FLAT_REGION,
)


class StructurePreservingRegularizer:
    """
    Structure-Preserving Gaussian Regularization.
    
    Components:
    A) Semantic-aware anisotropy regularization
    B) Opacity binary regularization (reduce floaters)
    C) Semantic boundary scale constraint
    """

    def __init__(self,
                 # Anisotropy params
                 lambda_aniso: float = 0.01,
                 aniso_start_iter: int = 5000,
                 linear_aspect_target: float = 3.0,
                 surface_flatness_target: float = 0.3,
                 # Opacity regularization
                 lambda_opacity: float = 0.005,
                 opacity_start_iter: int = 15000,
                 # Boundary constraint
                 lambda_boundary: float = 0.005,
                 boundary_start_iter: int = 5000,
                 # Scale regularization (prevent excessively large Gaussians)
                 lambda_scale: float = 0.001,
                 scale_start_iter: int = 3000,
                 max_scale_ratio: float = 10.0):

        self.lambda_aniso = lambda_aniso
        self.aniso_start_iter = aniso_start_iter
        self.linear_aspect_target = linear_aspect_target
        self.surface_flatness_target = surface_flatness_target

        self.lambda_opacity = lambda_opacity
        self.opacity_start_iter = opacity_start_iter

        self.lambda_boundary = lambda_boundary
        self.boundary_start_iter = boundary_start_iter

        self.lambda_scale = lambda_scale
        self.scale_start_iter = scale_start_iter
        self.max_scale_ratio = max_scale_ratio

    def compute_anisotropy_loss(self, gaussians, region_types: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Semantic-aware anisotropy regularization.
        
        - Linear structures: encourage s_max >> s_mid (elongated)
        - Surface regions: encourage s_min << s_mid (flat disk-like)
        """
        scales = gaussians.get_scaling  # (N, 3)
        sorted_scales, _ = torch.sort(scales, dim=1)
        s_min = sorted_scales[:, 0]
        s_mid = sorted_scales[:, 1]
        s_max = sorted_scales[:, 2]

        loss = torch.tensor(0.0, device="cuda")
        count = 0

        if region_types is not None and region_types.shape[0] == scales.shape[0]:
            # Linear structures: encourage elongation (s_max / s_mid > target)
            linear_mask = (region_types == REGION_LINEAR_STRUCTURE)
            if linear_mask.any():
                ratio_linear = s_max[linear_mask] / (s_mid[linear_mask] + 1e-6)
                loss_linear = F.relu(self.linear_aspect_target - ratio_linear).mean()
                loss = loss + loss_linear
                count += 1

            # Surface regions (textured + flat): encourage flatness (s_min / s_mid < target)
            surface_mask = (region_types == REGION_TEXTURED_SURFACE) | \
                           (region_types == REGION_FLAT_REGION)
            if surface_mask.any():
                ratio_surface = s_min[surface_mask] / (s_mid[surface_mask] + 1e-6)
                loss_surface = F.relu(ratio_surface - self.surface_flatness_target).mean()
                loss = loss + loss_surface
                count += 1

            # Edge/boundary regions: encourage moderate flatness
            edge_mask = (region_types == REGION_EDGE_BOUNDARY)
            if edge_mask.any():
                ratio_edge = s_min[edge_mask] / (s_mid[edge_mask] + 1e-6)
                loss_edge = F.relu(ratio_edge - 0.5).mean()
                loss = loss + loss_edge * 0.5
                count += 1
        else:
            # Fallback: general flatness encouragement for all Gaussians
            ratio_all = s_min / (s_mid + 1e-6)
            loss = F.relu(ratio_all - 0.5).mean()
            count = 1

        return loss

    def compute_opacity_loss(self, gaussians) -> torch.Tensor:
        """
        Opacity binary regularization.
        Encourages opacity to be close to 0 or 1, reducing semi-transparent floaters.
        
        L = -mean(log(max(alpha, 1-alpha)))
        """
        opacity = gaussians.get_opacity.squeeze()  # (N,)
        loss = -torch.log(torch.max(opacity, 1.0 - opacity) + 1e-6).mean()
        return loss

    def compute_boundary_loss(self, gaussians,
                               boundary_proximity: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Semantic boundary scale constraint.
        Gaussians near semantic boundaries should have smaller scales
        to avoid crossing boundaries.
        
        L = mean(boundary_proximity * max_scale)
        """
        if boundary_proximity is None:
            return torch.tensor(0.0, device="cuda")

        N = gaussians.get_scaling.shape[0]
        if boundary_proximity.shape[0] != N:
            return torch.tensor(0.0, device="cuda")

        scale_max = gaussians.get_scaling.max(dim=1).values  # (N,)
        loss = (boundary_proximity * scale_max).mean()
        return loss

    def compute_scale_loss(self, gaussians) -> torch.Tensor:
        """
        Prevent excessively large or disproportionate Gaussians.
        Penalizes max_scale / min_scale ratio exceeding threshold.
        """
        scales = gaussians.get_scaling  # (N, 3)
        sorted_scales, _ = torch.sort(scales, dim=1)
        s_min = sorted_scales[:, 0].clamp(min=1e-7)
        s_max = sorted_scales[:, 2]

        ratio = s_max / s_min
        loss = F.relu(ratio - self.max_scale_ratio).mean()
        return loss

    def compute_total_loss(self, gaussians, iteration: int,
                            region_types: Optional[torch.Tensor] = None,
                            boundary_proximity: Optional[torch.Tensor] = None) -> dict:
        """
        Compute all regularization losses with progressive scheduling.
        
        Returns:
            dict with individual and total regularization losses
        """
        losses = {}
        total = torch.tensor(0.0, device="cuda")

        # Anisotropy regularization
        if iteration >= self.aniso_start_iter:
            aniso_loss = self.compute_anisotropy_loss(gaussians, region_types)
            weight = self._progressive_weight(
                iteration, self.aniso_start_iter, self.lambda_aniso, ramp_iters=2000)
            losses['aniso'] = aniso_loss
            total = total + weight * aniso_loss
        else:
            losses['aniso'] = torch.tensor(0.0, device="cuda")

        # Opacity regularization (after densification ends)
        if iteration >= self.opacity_start_iter:
            opacity_loss = self.compute_opacity_loss(gaussians)
            weight = self._progressive_weight(
                iteration, self.opacity_start_iter, self.lambda_opacity, ramp_iters=3000)
            losses['opacity'] = opacity_loss
            total = total + weight * opacity_loss
        else:
            losses['opacity'] = torch.tensor(0.0, device="cuda")

        # Boundary constraint
        if iteration >= self.boundary_start_iter:
            boundary_loss = self.compute_boundary_loss(gaussians, boundary_proximity)
            weight = self._progressive_weight(
                iteration, self.boundary_start_iter, self.lambda_boundary, ramp_iters=2000)
            losses['boundary'] = boundary_loss
            total = total + weight * boundary_loss
        else:
            losses['boundary'] = torch.tensor(0.0, device="cuda")

        # Scale regularization
        if iteration >= self.scale_start_iter:
            scale_loss = self.compute_scale_loss(gaussians)
            weight = self._progressive_weight(
                iteration, self.scale_start_iter, self.lambda_scale, ramp_iters=2000)
            losses['scale'] = scale_loss
            total = total + weight * scale_loss
        else:
            losses['scale'] = torch.tensor(0.0, device="cuda")

        losses['total'] = total
        return losses

    @staticmethod
    def _progressive_weight(iteration: int, start_iter: int,
                             max_weight: float, ramp_iters: int = 2000) -> float:
        """Linear ramp from 0 to max_weight over ramp_iters after start_iter."""
        if iteration < start_iter:
            return 0.0
        progress = min(1.0, (iteration - start_iter) / ramp_iters)
        return max_weight * progress
