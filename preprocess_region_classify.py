#!/usr/bin/env python3
#
# SG-3DGS: Region Classification + Boundary Detection
# Converts SAM masks into per-pixel region type maps and boundary maps
#

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Region type constants (must match sg3dgs/region_manager.py)
REGION_LINEAR_STRUCTURE = 0
REGION_EDGE_BOUNDARY = 1
REGION_TEXTURED_SURFACE = 2
REGION_FLAT_REGION = 3


def compute_perimeter(mask_uint8):
    """Compute perimeter of a binary mask using contour detection."""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return perimeter


def compute_aspect_ratio(mask_uint8):
    """Compute aspect ratio of minimum bounding rectangle."""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    # Use the largest contour
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 1.0
    rect = cv2.minAreaRect(largest)
    w, h = rect[1]
    if min(w, h) < 1e-3:
        return 100.0
    return max(w, h) / min(w, h)


def compute_gradient_intensity(image, mask):
    """Compute average gradient intensity within a mask region."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0.0
    return grad_mag[mask_bool].mean()


def classify_mask(mask, image, gradient_threshold=30.0):
    """
    Classify a single SAM mask into one of four region types.
    
    Classification criteria:
    1. Area/perimeter compactness + aspect ratio -> linear vs block
    2. Region gradient intensity -> textured vs flat
    """
    mask_uint8 = mask.astype(np.uint8)
    area = mask.sum()
    
    if area < 50:
        return REGION_FLAT_REGION
    
    perimeter = compute_perimeter(mask_uint8)
    compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    aspect_ratio = compute_aspect_ratio(mask_uint8)
    gradient_intensity = compute_gradient_intensity(image, mask)
    
    # Linear structure: low compactness or high aspect ratio
    if compactness < 0.15 or aspect_ratio > 5.0:
        return REGION_LINEAR_STRUCTURE
    
    # Check if it's an edge/boundary-like region (thin, elongated, but not as extreme)
    if compactness < 0.3 and aspect_ratio > 3.0:
        return REGION_EDGE_BOUNDARY
    
    # Textured vs flat based on gradient intensity
    if gradient_intensity > gradient_threshold:
        return REGION_TEXTURED_SURFACE
    else:
        return REGION_FLAT_REGION


def detect_semantic_boundaries(region_map, dilation_size=5):
    """
    Detect boundaries between different semantic regions.
    
    Args:
        region_map: (H, W) int array of region types
        dilation_size: Dilate boundary for broader coverage
    
    Returns:
        (H, W) float32 boundary proximity map [0, 1]
    """
    H, W = region_map.shape
    
    # Detect boundaries using gradient of region map
    # Shift in 4 directions and check for label differences
    boundary = np.zeros((H, W), dtype=np.uint8)
    
    # Horizontal neighbors
    boundary[:-1, :] |= (region_map[:-1, :] != region_map[1:, :]).astype(np.uint8)
    boundary[1:, :] |= (region_map[:-1, :] != region_map[1:, :]).astype(np.uint8)
    # Vertical neighbors
    boundary[:, :-1] |= (region_map[:, :-1] != region_map[:, 1:]).astype(np.uint8)
    boundary[:, 1:] |= (region_map[:, :-1] != region_map[:, 1:]).astype(np.uint8)
    
    # Dilate boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    boundary_dilated = cv2.dilate(boundary, kernel, iterations=1)
    
    # Convert to distance-based proximity map [0, 1]
    # Closer to boundary = higher value
    dist_transform = cv2.distanceTransform(1 - boundary_dilated, cv2.DIST_L2, 5)
    max_dist = dilation_size * 2.0
    proximity = 1.0 - np.clip(dist_transform / max_dist, 0, 1)
    proximity = proximity.astype(np.float32)
    
    return proximity


def process_image(image, masks, gradient_threshold=30.0):
    """
    Process a single image: classify all masks and build region map + boundary map.
    
    Args:
        image: (H, W, 3) RGB image
        masks: List of binary masks from SAM
        gradient_threshold: Threshold for textured vs flat classification
    
    Returns:
        region_map: (H, W) int array of region types
        boundary_map: (H, W) float32 boundary proximity [0, 1]
    """
    H, W = image.shape[:2]
    
    # Initialize with flat region (default)
    region_map = np.full((H, W), REGION_FLAT_REGION, dtype=np.int32)
    
    # Classify each mask and paint onto region map
    # Process larger masks first (they form the base), then smaller masks override
    mask_areas = [(i, m.sum()) for i, m in enumerate(masks)]
    mask_areas.sort(key=lambda x: -x[1])  # Largest first
    
    for idx, area in mask_areas:
        mask = masks[idx]
        if isinstance(mask, np.ndarray):
            mask_bool = mask.astype(bool)
        else:
            mask_bool = mask
        
        if mask_bool.sum() < 50:
            continue
        
        region_type = classify_mask(mask_bool.astype(np.uint8), image, gradient_threshold)
        region_map[mask_bool] = region_type
    
    # Detect boundaries between different regions
    boundary_map = detect_semantic_boundaries(region_map)
    
    return region_map, boundary_map


def preprocess_regions(
    source_path: str,
    images_folder: str = "images",
    semantic_folder: str = "semantic",
    gradient_threshold: float = 30.0,
    save_visualizations: bool = True,
):
    """
    Main preprocessing: Convert SAM masks to region maps and boundary maps.
    """
    images_path = os.path.join(source_path, images_folder)
    semantic_path = os.path.join(source_path, semantic_folder)
    
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(f"Semantic folder not found: {semantic_path}")
    
    # Output directories
    region_dir = os.path.join(semantic_path, "regions")
    boundary_dir = os.path.join(semantic_path, "boundaries")
    os.makedirs(region_dir, exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)
    
    if save_visualizations:
        vis_dir = os.path.join(semantic_path, "region_vis")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get mask files
    mask_files = sorted([f for f in os.listdir(semantic_path) if f.endswith("_masks.npy")])
    
    if not mask_files:
        raise RuntimeError(f"No mask files found in {semantic_path}")
    
    print(f"Found {len(mask_files)} mask files")
    
    # Region type colors for visualization
    region_colors = {
        REGION_LINEAR_STRUCTURE: [255, 0, 0],      # Red
        REGION_EDGE_BOUNDARY: [255, 165, 0],        # Orange
        REGION_TEXTURED_SURFACE: [0, 255, 0],       # Green
        REGION_FLAT_REGION: [0, 0, 255],             # Blue
    }
    
    stats = {t: 0 for t in range(4)}
    
    for mask_file in tqdm(mask_files, desc="Classifying regions"):
        name = mask_file.replace("_masks.npy", "")
        
        # Load masks
        masks_data = np.load(os.path.join(semantic_path, mask_file), allow_pickle=True)
        masks = list(masks_data)
        
        # Load corresponding image
        image = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = os.path.join(images_path, name + ext)
            if os.path.exists(img_path):
                image = np.array(Image.open(img_path).convert('RGB'))
                break
        
        if image is None:
            print(f"Warning: Image not found for {name}, skipping")
            continue
        
        # Process
        region_map, boundary_map = process_image(image, masks, gradient_threshold)
        
        # Save
        np.save(os.path.join(region_dir, f"{name}_region_map.npy"), region_map.astype(np.uint8))
        np.save(os.path.join(boundary_dir, f"{name}_boundary.npy"), boundary_map)
        
        # Stats
        for t in range(4):
            stats[t] += (region_map == t).sum()
        
        # Visualization
        if save_visualizations:
            H, W = region_map.shape
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            for region_type, color in region_colors.items():
                vis[region_map == region_type] = color
            
            # Overlay with original
            overlay = (image * 0.5 + vis * 0.5).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(vis_dir, f"{name}_regions.png"))
            
            # Boundary visualization
            boundary_vis = (boundary_map * 255).astype(np.uint8)
            Image.fromarray(boundary_vis).save(os.path.join(vis_dir, f"{name}_boundary.png"))
    
    # Print statistics
    total_pixels = sum(stats.values())
    type_names = {
        REGION_LINEAR_STRUCTURE: "Linear Structure",
        REGION_EDGE_BOUNDARY: "Edge/Boundary",
        REGION_TEXTURED_SURFACE: "Textured Surface",
        REGION_FLAT_REGION: "Flat Region",
    }
    
    print(f"\n{'='*60}")
    print("Region Classification Complete")
    print(f"{'='*60}")
    for t, name in type_names.items():
        pct = stats[t] / max(total_pixels, 1) * 100
        print(f"  {name}: {stats[t]:,} pixels ({pct:.1f}%)")
    print(f"  Total: {total_pixels:,} pixels")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SG-3DGS: Region Classification from SAM masks"
    )
    parser.add_argument("-s", "--source_path", required=True, help="Dataset root directory")
    parser.add_argument("--images", default="images", help="Images folder name")
    parser.add_argument("--semantic", default="semantic", help="Semantic folder name")
    parser.add_argument("--gradient_threshold", type=float, default=30.0,
                        help="Gradient threshold for textured vs flat (default: 30.0)")
    parser.add_argument("--no_vis", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SG-3DGS: Region Classification + Boundary Detection")
    print("=" * 60)
    print(f"Dataset: {args.source_path}")
    print()
    
    preprocess_regions(
        source_path=args.source_path,
        images_folder=args.images,
        semantic_folder=args.semantic,
        gradient_threshold=args.gradient_threshold,
        save_visualizations=not args.no_vis,
    )
    
    print("\nNext step: Run DINOv2 feature extraction")
    print(f"  python preprocess_dinov2.py -s {args.source_path}")


if __name__ == "__main__":
    main()
