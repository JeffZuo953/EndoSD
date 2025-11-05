import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def check_original_mask(image_path, mask_path):
    """
    检查原始mask文件的质量，看是否本身就有锯齿
    """
    print(f"检查原始文件:")
    print(f"  图像: {image_path}")
    print(f"  Mask: {mask_path}")
    
    # 加载原始图像
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
        
    if not os.path.exists(mask_path):
        print(f"Mask文件不存在: {mask_path}")
        return
    
    raw_image = cv2.imread(image_path)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if raw_image is None:
        print(f"无法加载图像: {image_path}")
        return
        
    if raw_mask is None:
        print(f"无法加载mask: {mask_path}")
        return
    
    print(f"原始图像尺寸: {raw_image.shape}")
    print(f"原始mask尺寸: {raw_mask.shape}")
    print(f"原始mask值范围: {raw_mask.min()} - {raw_mask.max()}")
    print(f"原始mask唯一值: {np.unique(raw_mask)}")
    
    # 检测纯黑色区域
    if len(raw_image.shape) == 3:
        black_mask = np.all(raw_image == 0, axis=2)
    else:
        black_mask = (raw_image == 0)
    
    print(f"纯黑色像素数量: {np.sum(black_mask)}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 原始mask
    axes[0, 1].imshow(raw_mask, cmap='gray')
    axes[0, 1].set_title('原始Mask')
    axes[0, 1].axis('off')
    
    # 黑色区域检测
    axes[0, 2].imshow(black_mask, cmap='gray')
    axes[0, 2].set_title('检测到的纯黑色区域')
    axes[0, 2].axis('off')
    
    # 放大显示mask边缘区域
    h, w = raw_mask.shape
    # 选择一个有边缘的区域进行放大
    crop_size = min(200, h//3, w//3)
    center_y, center_x = h//2, w//2
    
    # 寻找有边缘的区域
    edges = cv2.Canny(raw_mask, 50, 150)
    edge_points = np.where(edges > 0)
    
    if len(edge_points[0]) > 0:
        # 选择第一个边缘点附近的区域
        edge_y, edge_x = edge_points[0][0], edge_points[1][0]
        y1 = max(0, edge_y - crop_size//2)
        y2 = min(h, edge_y + crop_size//2)
        x1 = max(0, edge_x - crop_size//2)
        x2 = min(w, edge_x + crop_size//2)
    else:
        # 如果没找到边缘，使用中心区域
        y1 = max(0, center_y - crop_size//2)
        y2 = min(h, center_y + crop_size//2)
        x1 = max(0, center_x - crop_size//2)
        x2 = min(w, center_x + crop_size//2)
    
    cropped_image = raw_image[y1:y2, x1:x2]
    cropped_mask = raw_mask[y1:y2, x1:x2]
    cropped_black = black_mask[y1:y2, x1:x2]
    
    axes[1, 0].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'放大图像 ({y1}:{y2}, {x1}:{x2})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cropped_mask, cmap='gray')
    axes[1, 1].set_title('放大Mask (检查锯齿)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cropped_black, cmap='gray')
    axes[1, 2].set_title('放大黑色区域')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 分析边缘质量
    print(f"\n边缘分析:")
    print(f"检测到的边缘像素数量: {np.sum(edges > 0)}")
    
    # 检查mask边缘的平滑度
    # 计算梯度来评估锯齿程度
    grad_x = cv2.Sobel(raw_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(raw_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    print(f"梯度幅值统计:")
    print(f"  平均值: {gradient_magnitude.mean():.2f}")
    print(f"  标准差: {gradient_magnitude.std():.2f}")
    print(f"  最大值: {gradient_magnitude.max():.2f}")
    
    # 高梯度区域可能表示锯齿
    high_gradient_pixels = np.sum(gradient_magnitude > gradient_magnitude.mean() + 2 * gradient_magnitude.std())
    print(f"  高梯度像素数量: {high_gradient_pixels} ({high_gradient_pixels/gradient_magnitude.size*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='检查原始mask文件质量')
    parser.add_argument('--image', type=str, required=True, help='原始图像路径')
    parser.add_argument('--mask', type=str, required=True, help='原始mask路径')
    
    args = parser.parse_args()
    
    check_original_mask(args.image, args.mask)

if __name__ == "__main__":
    main()