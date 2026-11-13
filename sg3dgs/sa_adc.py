#
# SG-3DGS: Semantic-Aware Adaptive Density Control (SA-ADC)
# Innovation 1: Region-adaptive densification thresholds + boundary-enhanced densification
#

import torch
import torch.nn.functional as F
from sg3dgs.region_manager import (
    REGION_LINEAR_STRUCTURE,
    REGION_EDGE_BOUNDARY,
    REGION_TEXTURED_SURFACE,
    REGION_FLAT_REGION,
    project_gaussians_to_2d,
)


class SemanticAwareDensifier:
    """
    Semantic-Aware Adaptive Density Control.
    
    Modifies the standard densification by:
    1. Using per-region adaptive gradient thresholds
    2. Performing boundary-enhanced densification at semantic boundaries
    """

    def __init__(self,
                 # Region-adaptive threshold multipliers
                 alpha_linear: float = 0.5,
                 alpha_edge: float = 0.7,
                 alpha_textured: float = 1.0,
                 alpha_flat: float = 1.5,
                 # Boundary densification
                 boundary_threshold_mult: float = 0.3,
                 boundary_densify_interval: int = 500,
                 # Activation schedule
                 sa_adc_start_iter: int = 3000,
                 # Cache update interval
                 cache_update_interval: int = 100):

        self.alpha = {
            REGION_LINEAR_STRUCTURE: alpha_linear,
            REGION_EDGE_BOUNDARY: alpha_edge,
            REGION_TEXTURED_SURFACE: alpha_textured,
            REGION_FLAT_REGION: alpha_flat,
        }
        self.boundary_threshold_mult = boundary_threshold_mult
        self.boundary_densify_interval = boundary_densify_interval
        self.sa_adc_start_iter = sa_adc_start_iter
        self.cache_update_interval = cache_update_interval

        # Cached per-gaussian region types (updated every cache_update_interval iters)
        self._cached_region_types = None
        self._cached_boundary_proximity = None
        self._cache_iter = -1

    def is_active(self, iteration: int) -> bool:
        return iteration >= self.sa_adc_start_iter

    def update_cache(self, gaussians, viewpoint_cam, region_map, boundary_map, iteration):
        """
        Update cached region type and boundary proximity per Gaussian.
        Only recomputes every cache_update_interval iterations.
        """
        if (iteration - self._cache_iter) < self.cache_update_interval and self._cached_region_types is not None:
            # Check size consistency
            if self._cached_region_types.shape[0] == gaussians.get_xyz.shape[0]:
                return
        
        with torch.no_grad():
            positions_2d, valid = project_gaussians_to_2d(gaussians.get_xyz, viewpoint_cam)
            N = gaussians.get_xyz.shape[0]

            # Default: textured surface (standard threshold)
            region_types = torch.full((N,), REGION_TEXTURED_SURFACE, device="cuda", dtype=torch.long)
            boundary_prox = torch.zeros(N, device="cuda")

            if region_map is not None and valid.any():
                H, W = region_map.shape
                x = positions_2d[valid, 0].long().clamp(0, W - 1)
                y = positions_2d[valid, 1].long().clamp(0, H - 1)
                region_types[valid] = region_map[y, x]

            if boundary_map is not None and valid.any():
                H, W = boundary_map.shape
                x = positions_2d[valid, 0].long().clamp(0, W - 1)
                y = positions_2d[valid, 1].long().clamp(0, H - 1)
                boundary_prox[valid] = boundary_map[y, x]

            self._cached_region_types = region_types
            self._cached_boundary_proximity = boundary_prox
            self._cache_iter = iteration

    def compute_adaptive_threshold(self, base_threshold: float, N: int) -> torch.Tensor:
        """
        Compute per-Gaussian adaptive gradient threshold based on cached region types.
        
        Returns:
            (N,) tensor of per-Gaussian thresholds
        """
        thresholds = torch.ones(N, device="cuda") * base_threshold

        if self._cached_region_types is not None:
            rt = self._cached_region_types
            # Ensure size matches (Gaussians may have changed due to previous densification)
            if rt.shape[0] == N:
                for region_type, alpha in self.alpha.items():
                    mask = (rt == region_type)
                    thresholds[mask] *= alpha

        return thresholds

    def should_boundary_densify(self, iteration: int) -> bool:
        """Check if boundary-enhanced densification should run this iteration."""
        if iteration < self.sa_adc_start_iter:
            return False
        return (iteration % self.boundary_densify_interval == 0)

    def get_boundary_clone_mask(self, gaussians, base_threshold: float) -> torch.Tensor:
        """
        Get mask for boundary-enhanced cloning.
        Gaussians near semantic boundaries with moderate gradients get cloned.
        
        Returns:
            Boolean mask (N,) for Gaussians to clone at boundaries
        """
        if self._cached_boundary_proximity is None:
            return torch.zeros(gaussians.get_xyz.shape[0], device="cuda", dtype=torch.bool)

        grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze()

        N = gaussians.get_xyz.shape[0]
        bp = self._cached_boundary_proximity
        if bp.shape[0] != N:
            return torch.zeros(N, device="cuda", dtype=torch.bool)

        boundary_threshold = base_threshold * self.boundary_threshold_mult

        # Select Gaussians near boundary with sufficient gradient
        near_boundary = (bp > 0.3)
        sufficient_grad = (grads >= boundary_threshold)
        # Only clone (not split) — small Gaussians near boundaries
        not_too_large = (gaussians.get_scaling.max(dim=1).values <=
                         gaussians.percent_dense * 10.0)  # relaxed size constraint

        mask = near_boundary & sufficient_grad & not_too_large
        return mask
