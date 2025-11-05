#!/usr/bin/env python3
"""
将预测（prediction）图像和真值（ground truth）图像进行配对和水平拼接，
以便于可视化比较。
"""

import os
import argparse
import logging
import cv2
import numpy as np
from tqdm import tqdm

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def combine_images(gt_dir: str, pred_dir: str, output_dir: str, add_text: bool, logger: logging.Logger):
    """
    遍历预测文件夹，寻找匹配的真值文件，然后将它们拼接并保存。

    Args:
        gt_dir (str): 真值图像文件夹路径。
        pred_dir (str): 预测图像文件夹路径。
        output_dir (str): 保存拼接后图像的文件夹路径。
        add_text (bool): 是否在图像上添加文字标签。
        logger (logging.Logger): 日志记录器。
    """
    logger.info(f"开始处理预测文件夹: {pred_dir}")
    logger.info(f"真值文件夹: {gt_dir}")
    logger.info(f"输出文件夹: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pred_files = [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
    if not pred_files:
        logger.warning("预测文件夹中没有找到任何文件。")
        return

    processed_count = 0
    not_found_count = 0

    for filename in tqdm(pred_files, desc="拼接图像"):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(gt_path):
            # logger.warning(f"未找到匹配的真值文件: {gt_path}")
            not_found_count += 1
            continue

        try:
            pred_img = cv2.imread(pred_path)
            gt_img = cv2.imread(gt_path)

            if pred_img is None:
                logger.error(f"无法读取预测图像: {pred_path}")
                continue
            if gt_img is None:
                logger.error(f"无法读取真值图像: {gt_path}")
                continue

            # 统一图像尺寸以便拼接
            h1, w1 = pred_img.shape[:2]
            h2, w2 = gt_img.shape[:2]
            
            if h1 != h2 or w1 != w2:
                logger.warning(f"尺寸不匹配: {filename} (Pred: {h1}x{w1}, GT: {h2}x{w2}). 将GT图像缩放到与Pred一致。")
                gt_img = cv2.resize(gt_img, (w1, h1), interpolation=cv2.INTER_NEAREST)

            if add_text:
                # 添加文字标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                font_thickness = 2
                text_color = (0, 255, 255)  # Yellow in BGR
                text_position = (20, 40)    # (x, y) from top-left

                # 在真值图像上添加 "GT"
                cv2.putText(gt_img, 'GT', text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                # 在预测图像上添加 "Pred"
                cv2.putText(pred_img, 'Pred', text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # 水平拼接 (GT | Pred)
            combined_img = np.hstack((gt_img, pred_img))
            
            cv2.imwrite(output_path, combined_img)
            processed_count += 1

        except Exception as e:
            logger.error(f"处理文件 {filename} 时发生错误: {e}")

    logger.info("处理完成。")
    logger.info(f"成功处理并拼接了 {processed_count} 对图像。")
    if not_found_count > 0:
        logger.warning(f"有 {not_found_count} 个预测文件未在真值文件夹中找到对应的文件。")


def main():
    parser = argparse.ArgumentParser(description="将预测图像和真值图像进行拼接以便比较。")
    parser.add_argument("--gt-dir", type=str, required=True, help="包含真值（Ground Truth）图像的文件夹。")
    parser.add_argument("--pred-dir", type=str, required=True, help="包含预测（Prediction）图像的文件夹。")
    parser.add_argument("--output-dir", type=str, required=True, help="用于保存拼接后图像的输出文件夹。")
    parser.add_argument("--add-text", action='store_true', help="在图像上添加 'GT' 和 'Pred' 文字标签。")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    combine_images(args.gt_dir, args.pred_dir, args.output_dir, args.add_text, logger)


if __name__ == "__main__":
    main()