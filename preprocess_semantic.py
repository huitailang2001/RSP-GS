#!/usr/bin/env python3
#
# 语义3DGS数据预处理
# 使用SAM生成区域掩码，用于SAC+3DCS训练
#

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_sam_model(checkpoint_path: str = None, device: str = "cuda"):
    """加载SAM模型"""
    
    # 查找checkpoint
    if checkpoint_path is None:
        paths = [
            os.environ.get('SAM2_CHECKPOINT', ''),
            "/mnt/DroneSplat/checkpoints/sam2_hiera_large.pt",
            "/mnt/DroneSplat/checkpoints/sam2.1_hiera_large.pt",
            "checkpoints/sam2_hiera_large.pt",
            "checkpoints/sam_vit_h_4b8939.pth",
        ]
        
        for path in paths:
            if path and os.path.exists(path):
                checkpoint_path = path
                break
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("Warning: SAM checkpoint not found")
        print("Will use simple color-based segmentation")
        return None, None
    
    print(f"Loading SAM from: {checkpoint_path}")
    
    # 尝试加载SAM2
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        model = build_sam2("sam2_hiera_l.yaml", checkpoint_path, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            min_mask_region_area=200,
        )
        print("SAM2 loaded successfully")
        return model, mask_generator
        
    except ImportError:
        pass
    
    # 尝试加载SAM1
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        model_type = "vit_h"
        if "vit_l" in checkpoint_path:
            model_type = "vit_l"
        elif "vit_b" in checkpoint_path:
            model_type = "vit_b"
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=200,
        )
        print("SAM loaded successfully")
        return sam, mask_generator
        
    except ImportError:
        print("Warning: SAM not available")
        return None, None


def generate_color_based_masks(image: np.ndarray, num_clusters: int = 20) -> List[np.ndarray]:
    """
    回退方案：基于颜色聚类生成区域掩码
    """
    import cv2
    from sklearn.cluster import KMeans
    
    h, w = image.shape[:2]
    
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # 添加位置信息
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    features = np.concatenate([
        lab.reshape(-1, 3),
        (xx.reshape(-1, 1) / w * 50),
        (yy.reshape(-1, 1) / h * 50),
    ], axis=1)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    label_map = labels.reshape(h, w)
    
    # 生成掩码
    masks = []
    for i in range(num_clusters):
        mask = (label_map == i)
        if mask.sum() > 100:  # 过滤太小的区域
            masks.append(mask)
    
    return masks


def preprocess_dataset(
    source_path: str,
    images_folder: str = "images",
    output_folder: str = "semantic",
    checkpoint_path: str = None,
    device: str = "cuda",
    save_visualizations: bool = True,
) -> str:
    """
    预处理数据集，生成SAM区域掩码
    """
    images_path = os.path.join(source_path, images_folder)
    output_path = os.path.join(source_path, output_folder)
    
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images folder not found: {images_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # 获取图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = sorted([
        f for f in os.listdir(images_path)
        if os.path.splitext(f)[1] in image_extensions
    ])
    
    if not image_files:
        raise RuntimeError(f"No images found in {images_path}")
    
    print(f"Found {len(image_files)} images in {images_path}")
    
    # 加载SAM
    _, mask_generator = load_sam_model(checkpoint_path, device)
    use_sam = mask_generator is not None
    
    # 统计信息
    total_masks = 0
    total_pixels = 0
    
    print("\nGenerating region masks...")
    for image_file in tqdm(image_files, desc="Processing"):
        image_path = os.path.join(images_path, image_file)
        name = os.path.splitext(image_file)[0]
        
        # 加载图像
        image = np.array(Image.open(image_path).convert('RGB'))
        h, w = image.shape[:2]
        total_pixels += h * w
        
        # 生成掩码
        if use_sam:
            try:
                masks_data = mask_generator.generate(image)
                masks = [m['segmentation'] for m in masks_data]
            except Exception as e:
                print(f"SAM failed for {image_file}: {e}")
                masks = generate_color_based_masks(image)
        else:
            masks = generate_color_based_masks(image)
        
        total_masks += len(masks)
        
        # 保存掩码
        masks_np = [m.astype(np.uint8) if isinstance(m, np.ndarray) else m.cpu().numpy().astype(np.uint8) for m in masks]
        np.save(os.path.join(output_path, f"{name}_masks.npy"), masks_np)
        
        # 保存可视化
        if save_visualizations:
            # 创建彩色可视化
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            np.random.seed(42)
            
            for i, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                color = np.random.randint(50, 255, size=3)
                vis[mask.astype(bool)] = color
            
            Image.fromarray(vis).save(os.path.join(output_path, f"{name}_regions.png"))
            
            # 叠加可视化
            overlay = (image * 0.6 + vis * 0.4).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(output_path, f"{name}_overlay.png"))
    
    # 保存元数据
    metadata = {
        'num_images': len(image_files),
        'total_masks': total_masks,
        'avg_masks_per_image': total_masks / len(image_files),
        'total_pixels': int(total_pixels),
        'use_sam': use_sam,
    }
    
    with open(os.path.join(output_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Output folder: {output_path}")
    print(f"Total images: {len(image_files)}")
    print(f"Total masks: {total_masks}")
    print(f"Average masks per image: {total_masks / len(image_files):.1f}")
    print(f"Method: {'SAM' if use_sam else 'Color clustering'}")
    print(f"{'='*60}")
    
    return output_path


def validate_preprocessing(source_path: str, output_folder: str = "semantic"):
    """验证预处理结果"""
    output_path = os.path.join(source_path, output_folder)
    
    if not os.path.exists(output_path):
        print(f"Error: Output folder not found: {output_path}")
        return False
    
    # 加载元数据
    metadata_file = os.path.join(output_path, 'metadata.json')
    if not os.path.exists(metadata_file):
        print("Warning: metadata.json not found")
        return False
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Validated {metadata['num_images']} images")
    print(f"Total masks: {metadata['total_masks']}")
    print(f"Method used: {'SAM' if metadata['use_sam'] else 'Color clustering'}")
    
    # 检查掩码文件
    mask_files = [f for f in os.listdir(output_path) if f.endswith('_masks.npy')]
    if len(mask_files) != metadata['num_images']:
        print(f"Warning: Expected {metadata['num_images']} mask files, found {len(mask_files)}")
        return False
    
    print("Validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="语义3DGS数据预处理 - 生成SAM区域掩码用于SAC+3DCS训练"
    )
    parser.add_argument(
        "-s", "--source_path",
        required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        "--images",
        default="images",
        help="图像文件夹名（默认: images）"
    )
    parser.add_argument(
        "--output",
        default="semantic",
        help="输出文件夹名（默认: semantic）"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SAM checkpoint路径（默认: 自动查找）"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="计算设备（默认: cuda）"
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存可视化图像"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="仅验证现有预处理结果"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("语义3DGS数据预处理 (SAC + 3DCS)")
    print("=" * 60)
    print(f"数据集: {args.source_path}")
    print(f"输出目录: {args.output}")
    print()
    
    if args.validate:
        validate_preprocessing(args.source_path, args.output)
    else:
        preprocess_dataset(
            source_path=args.source_path,
            images_folder=args.images,
            output_folder=args.output,
            checkpoint_path=args.checkpoint,
            device=args.device,
            save_visualizations=not args.no_vis,
        )
        
        print("\n下一步: 运行语义训练")
        print(f"  python train_semantic.py -s {args.source_path} -m output/semantic_model --eval")


if __name__ == "__main__":
    main()
