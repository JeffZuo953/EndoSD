import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

from collections import OrderedDict

def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict
    
def visualize_depth(img, pred_depth, output_path):
    """
    可视化深度预测结果
    
    Args:
        img (torch.Tensor): 原始图像
        pred_depth (torch.Tensor): 预测深度图
        output_path (str): 输出图像路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(15, 5))  # 减小图像大小

    # 修复图像颜色通道顺序和归一化
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np[..., ::-1]  # BGR转RGB
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # 归一化
    
    # 第一行：原始图像
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_np)
    plt.colorbar(aspect=30) 
    
    # 预测深度图（归一化前）
    plt.subplot(1, 3, 2)
    plt.title('Predicted Depth (Raw)')
    pred_depth_np = pred_depth.cpu().numpy()
    plt.imshow(pred_depth_np, cmap='gray')
    plt.colorbar(aspect=30)  # 保持颜色条比例与图像一致
    
    # 预测深度图（归一化后）
    plt.subplot(1, 3, 3)
    plt.title('Predicted Depth (Normalized)')
    pred_depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    pred_depth_norm_np = pred_depth_norm.cpu().numpy()
    plt.imshow(pred_depth_norm_np, cmap='gray')
    plt.colorbar(aspect=30)  # 保持颜色条比例与图像一致
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 去除额外空白
    plt.close()

# python run2.py --img-path /root/c3vd/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /root/Depth-Anything-V2/checkpoints/c3vd_train_0.1.pth --save-numpy --max-depth 0.1

# python run2.py --img-path /root/c3vd/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /root/Depth-Anything-V2/checkpoints/c3vd_train_0.1.0.pth --save-numpy --max-depth 0.1

# python run2.py --img-path /root/c3vd/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /root/Depth-Anything-V2/checkpoints/01.pth --max-depth 0.1

# python run2.py --img-path /root/c3vd/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /root/Depth-Anything-V2/checkpoints/c3vd_train_100.0.pth --save-numpy --max-depth 100

# python run2.py --img-path /root/c3vd/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /root/Depth-Anything-V2/checkpoints/c3vd_train_100.pth --save-numpy --max-depth 100
# python run.py --img-path /root/dpt/test_mapping.txt --input-size 518 --outdir ./vis_depth --encoder vitl --load-from /data/train_simcol/best_abs_rel.pth --grayscale --max-depth 0.2
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    loaded_state_dict = torch.load(args.load_from, map_location='cpu')
    if isinstance(loaded_state_dict, dict) and "model" in loaded_state_dict:
        depth_anything.load_state_dict(strip_module_prefix(loaded_state_dict["model"]))
    else:
        depth_anything.load_state_dict(strip_module_prefix(loaded_state_dict))
    
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        # Trim whitespace and split if multiple files are accidentally concatenated
        filename = filename.strip().split()[0]
        
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        # Robust image reading with error handling
        if not os.path.exists(filename):
            print(f"Warning: File does not exist: {filename}")
            continue
        
        raw_image = cv2.imread(filename)
        
        if raw_image is None:
            print(f"Warning: Could not read image file: {filename}")
            continue
        
        try:
            depth = depth_anything.infer_image(raw_image, args.input_size)
            depth = torch.from_numpy(depth).to(DEVICE)
            raw_image = torch.from_numpy(raw_image).permute(2, 0, 1).to(DEVICE)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.txt')
            np.savetxt(output_path, depth.cpu().numpy().flatten(), fmt='%.6f')
            
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        
        visualize_depth(raw_image, depth, output_path)