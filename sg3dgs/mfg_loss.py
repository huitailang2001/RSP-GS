#
# SG-3DGS: Multi-Granularity Feature-Guided Loss (MFG-Loss)
# Innovation 2: DINOv2 feature-gradient attention-weighted reconstruction loss
#

import torch
import torch.nn.functional as F
from typing import Optional


class MFGLoss:
    """
    Multi-Granularity Feature-Guided Loss.
    
    Uses precomputed DINOv2 feature-gradient attention maps to weight
    per-pixel reconstruction loss. Critical regions (edges, structures)
    receive higher weight; flat regions receive lower weight.
    
    Also includes optional high-frequency enhancement via edge-aware loss.
    """

    def __init__(self,
                 lambda_dssim: float = 0.2,
                 edge_loss_weight: float = 0.05,
                 edge_loss_start_iter: int = 3000,
                 # Sobel edge detection
                 use_edge_loss: bool = True,
                 # Frequency-domain loss
                 freq_loss_weight: float = 0.01,
                 freq_loss_start_iter: int = 5000,
                 freq_attention_threshold: float = 0.7):
        self.lambda_dssim = lambda_dssim
        self.edge_loss_weight = edge_loss_weight
        self.edge_loss_start_iter = edge_loss_start_iter
        self.use_edge_loss = use_edge_loss
        self.freq_loss_weight = freq_loss_weight
        self.freq_loss_start_iter = freq_loss_start_iter
        self.freq_attention_threshold = freq_attention_threshold

        # Precompute Sobel kernels (on first use)
        self._sobel_x = None
        self._sobel_y = None

    def _init_sobel(self, device):
        """Initialize Sobel kernels for edge detection."""
        if self._sobel_x is not None:
            return
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32, device=device)
        # Shape: (1, 1, 3, 3) for conv2d
        self._sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        self._sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    def compute_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude map from an image using Sobel operator.
        
        Args:
            image: (3, H, W) or (1, 3, H, W) image tensor
            
        Returns:
            (1, H, W) edge magnitude map
        """
        self._init_sobel(image.device)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to grayscale
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        # Apply Sobel
        edge_x = F.conv2d(gray, self._sobel_x, padding=1)
        edge_y = F.conv2d(gray, self._sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        return edge_mag.squeeze(0)  # (1, H, W)

    def compute_loss(self,
                     rendered: torch.Tensor,
                     gt: torch.Tensor,
                     attention_map: Optional[torch.Tensor],
                     ssim_fn,
                     iteration: int,
                     fused_ssim_fn=None) -> dict:
        """
        Compute the MFG-weighted reconstruction loss.
        
        Args:
            rendered: (3, H, W) rendered image
            gt: (3, H, W) ground truth image
            attention_map: (H, W) DINOv2 attention map [0.3, 1.0], or None for uniform
            ssim_fn: SSIM function (from utils.loss_utils)
            iteration: Current training iteration
            fused_ssim_fn: Optional fused SSIM function for efficiency
            
        Returns:
            dict with 'total', 'weighted_l1', 'ssim', 'edge' loss values
        """
        losses = {}

        # --- Weighted L1 Loss ---
        pixel_error = torch.abs(rendered - gt)  # (3, H, W)

        if attention_map is not None:
            # Ensure attention_map matches image size
            H, W = rendered.shape[1], rendered.shape[2]
            if attention_map.shape[0] != H or attention_map.shape[1] != W:
                attention_map = F.interpolate(
                    attention_map.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='bilinear', align_corners=False
                ).squeeze()

            weighted_l1 = (pixel_error * attention_map.unsqueeze(0)).mean()
        else:
            weighted_l1 = pixel_error.mean()

        losses['weighted_l1'] = weighted_l1

        # --- SSIM Loss ---
        if fused_ssim_fn is not None:
            ssim_value = fused_ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0))
        else:
            ssim_value = ssim_fn(rendered, gt)
        losses['ssim'] = ssim_value

        # --- Combined RGB loss ---
        rgb_loss = (1.0 - self.lambda_dssim) * weighted_l1 + \
                   self.lambda_dssim * (1.0 - ssim_value)
        losses['rgb'] = rgb_loss

        # --- Edge Preservation Loss ---
        edge_loss = torch.tensor(0.0, device=rendered.device)
        if self.use_edge_loss and iteration >= self.edge_loss_start_iter:
            gt_edges = self.compute_edge_map(gt)         # (1, H, W)
            render_edges = self.compute_edge_map(rendered)  # (1, H, W)
            edge_loss = F.l1_loss(render_edges, gt_edges)
        losses['edge'] = edge_loss

        # --- Frequency-Domain Loss (L_freq) ---
        freq_loss = torch.tensor(0.0, device=rendered.device)
        if iteration >= self.freq_loss_start_iter and attention_map is not None:
            freq_loss = self.compute_freq_loss(rendered, gt, attention_map)
        losses['freq'] = freq_loss

        # --- Total Loss ---
        total = rgb_loss
        if iteration >= self.edge_loss_start_iter:
            edge_weight = self._get_edge_weight(iteration)
            total = total + edge_weight * edge_loss
        if iteration >= self.freq_loss_start_iter:
            freq_weight = self._get_freq_weight(iteration)
            total = total + freq_weight * freq_loss
        losses['total'] = total

        return losses

    def compute_freq_loss(self, rendered: torch.Tensor, gt: torch.Tensor,
                           attention_map: torch.Tensor) -> torch.Tensor:
        """
        Frequency-domain loss for high-attention regions.
        Compares FFT magnitude spectra to preserve high-frequency details
        (brick textures, decorative cornices, metalwork patterns).
        
        Args:
            rendered: (3, H, W) rendered image
            gt: (3, H, W) ground truth image
            attention_map: (H, W) attention map [0.3, 1.0]
        """
        # Create binary mask for high-attention regions
        high_attn_mask = (attention_map > self.freq_attention_threshold).float()  # (H, W)
        if high_attn_mask.sum() < 100:
            return torch.tensor(0.0, device=rendered.device)

        # Apply mask to isolate high-attention regions
        mask_3d = high_attn_mask.unsqueeze(0)  # (1, H, W)
        rendered_masked = rendered * mask_3d
        gt_masked = gt * mask_3d

        # Convert to grayscale for FFT
        render_gray = (0.299 * rendered_masked[0] + 0.587 * rendered_masked[1] +
                       0.114 * rendered_masked[2])  # (H, W)
        gt_gray = (0.299 * gt_masked[0] + 0.587 * gt_masked[1] +
                   0.114 * gt_masked[2])  # (H, W)

        # FFT
        render_fft = torch.fft.fft2(render_gray)
        gt_fft = torch.fft.fft2(gt_gray)

        # Compare magnitude spectra (L1 on log-magnitude for stability)
        render_mag = torch.log(torch.abs(render_fft) + 1e-8)
        gt_mag = torch.log(torch.abs(gt_fft) + 1e-8)

        loss = F.l1_loss(render_mag, gt_mag)
        return loss

    def _get_edge_weight(self, iteration: int) -> float:
        """Progressive edge loss weight scheduling."""
        if iteration < self.edge_loss_start_iter:
            return 0.0
        # Linear ramp from 0 to edge_loss_weight over 2000 iterations
        ramp = min(1.0, (iteration - self.edge_loss_start_iter) / 2000.0)
        return self.edge_loss_weight * ramp

    def _get_freq_weight(self, iteration: int) -> float:
        """Progressive frequency loss weight scheduling."""
        if iteration < self.freq_loss_start_iter:
            return 0.0
        ramp = min(1.0, (iteration - self.freq_loss_start_iter) / 3000.0)
        return self.freq_loss_weight * ramp
