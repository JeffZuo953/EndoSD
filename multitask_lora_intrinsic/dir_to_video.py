import cv2
import os
import re


def images_to_video(image_folder, video_name, fps=10):
    """
    将指定文件夹中的图片合成为视频。

    参数:
    image_folder (str): 包含图片的文件夹路径。
    video_name (str): 输出视频文件的名称 (例如 'output.mp4')。
    fps (int): 生成视频的帧率 (每秒图片数)。
    """
    # 获取所有图片文件，并过滤掉非图片文件
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(supported_formats)]

    # 如果没有找到图片，则退出
    if not images:
        print(f"在文件夹 '{image_folder}' 中没有找到图片文件。")
        return

    # 定义排序函数
    def sort_key(filename):
        try:
            # 移除文件后缀 (.png, .jpg, etc.)
            name_without_ext = os.path.splitext(filename)[0]
            # 按下划线分割并取最后一个部分
            last_part = name_without_ext.split('_')[-1]
            # 将其转换为整数进行排序
            return int(last_part)
        except (ValueError, IndexError):
            # 如果文件名格式不符 (例如没有下划线或最后一部分不是数字)，
            # 返回一个很大的数，使其排在最后
            return float('inf')

    # 根据新的排序逻辑对图片进行排序
    images.sort(key=sort_key)

    print("找到以下图片将用于合成视频:")
    for img_name in images:
        print(f"- {img_name}")

    # 读取第一张图片以获取视频的宽度和高度
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print(f"错误：无法读取第一张图片 '{images[0]}'")
        return

    height, width, layers = frame.shape
    size = (width, height)
    print(f"\n视频尺寸将为: {width}x{height}")
    print(f"视频帧率: {fps}")

    # 定义视频编码器并创建 VideoWriter 对象
    # 对于 .mp4 文件，'mp4v' 是一个不错的选择
    # 对于 .avi 文件，可以使用 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, size)

    print("\n开始合成视频...")
    # 遍历所有图片并将它们写入视频文件
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # 确保图片尺寸与视频尺寸一致，如果不一致则调整
        if (frame.shape[1], frame.shape[0]) != size:
            print(f"警告: 图片 '{image}' 的尺寸与第一张图片不同。已将其调整为 {width}x{height}。")
            frame = cv2.resize(frame, size)

        out.write(frame)

    # 释放 VideoWriter 对象
    out.release()
    print(f"\n视频 '{video_name}' 已成功创建！")


# --- 主程序 ---
if __name__ == '__main__':
    # --- 请修改以下参数 ---

    # 1. 图片文件夹的路径
    # 请确保路径是正确的，或者将图片文件夹放在与此脚本相同的目录中
    # 示例: image_folder = 'C:/Users/YourUser/Desktop/MyImages'
    # 示例: image_folder = 'C:/Users/YourUser/Desktop/MyImages'
    image_folders = [
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_193909_multitask_dinov3_vits16plus_20250830_071130_checkpoint_epoch_294/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_193924_multitask_vits_20250829_204046_checkpoint_epoch_397/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_193937_multitask_dinov3_vits16plus_20250830_071130_checkpoint_epoch_343/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_194027_multitask_vits_20250829_204046_checkpoint_epoch_350/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_194040_multitask_dinov3_vits16plus_20250830_071130_checkpoint_epoch_343/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_194054_multitask_dinov3_vits16plus_20250830_071130_checkpoint_epoch_294/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_194054_multitask_vits_20250829_204046_checkpoint_epoch_350/seg/image',
        '/media/ExtHDD1/jianfu/data/inference_lesion/20250903_194109_multitask_vits_20250829_204046_checkpoint_epoch_397/seg/image',
        "/media/ExtHDD1/jianfu/data/polyp/ETIS-LaribPolypDB/gt_vis/train_cache_gt_vis_20250903_202513", "/media/ExtHDD1/jianfu/data/polyp/ETIS-LaribPolypDB/gt_vis/origin_images",
        "/media/ExtHDD1/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/gt_vis/origin_images",
        "/media/ExtHDD1/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/gt_vis/train_cache_gt_vis_20250903_202400"
    ]

    # 2. 指定输出视频的文件夹
    output_video_dir = '/media/ExtHDD1/jianfu/data/inference_lesion_video'

    # 3. 自动根据文件夹名称和输出目录生成完整的视频文件路径
    # 例如，路径 '.../parent_dir/seg/image' 将生成 'parent_dir.mp4'
    video_names = [
        os.path.join(output_video_dir,
                     os.path.basename(os.path.dirname(os.path.dirname(p))) + '.mp4') for p in image_folders
    ] + ["ETIS-LaribPolypDB_gt_vis.mp4", "ETIS-LaribPolypDB_origin_images.mp4", "CVC-EndoScene_ValidationDataset_origin_images.mp4", "CVC-EndoScene_ValidationDataset_gt_vis.mp4"]

    # 4. 视频的帧率 (FPS)
    fps = 10

    # --- 参数修改结束 ---

    # 确保输出目录存在
    os.makedirs(output_video_dir, exist_ok=True)
    print(f"视频将保存到: {output_video_dir}")

    # 遍历所有指定的文件夹和视频名称
    for i, image_folder in enumerate(image_folders):
        video_name = video_names[i]
        print(f"\n--- 开始处理任务 {i+1}/{len(image_folders)} ---")
        print(f"  - 输入文件夹: {image_folder }")
        print(f"  - 输出视频: {video_name}")

        # 检查图片文件夹是否存在
        if not os.path.isdir(image_folder):
            print(f"错误: 文件夹 '{image_folder}' 不存在。跳过此任务。")
            continue  # 继续下一个循环

        # 调用函数开始合成
        images_to_video(image_folder, video_name, fps)

    print("\n--- 所有任务处理完毕 ---")
