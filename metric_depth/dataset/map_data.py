import os

def collect_rgb_depth_pairs(root_dir, output_txt, rgb_ext=".png", depth_ext=".exr"):
    pairs = []

    for seq_name in sorted(os.listdir(root_dir)):
        seq_path = os.path.join(root_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue

        rgb_dir = os.path.join(seq_path, 'rgb')
        depth_dir = os.path.join(seq_path, 'depth')

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            continue

        # 只保留指定后缀的文件
        rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(rgb_ext))
        depth_files = sorted(f for f in os.listdir(depth_dir) if f.endswith(depth_ext))

        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb_path = os.path.abspath(os.path.join(rgb_dir, rgb_file))
            depth_path = os.path.abspath(os.path.join(depth_dir, depth_file))
            pairs.append(f"{rgb_path} {depth_path}")

    with open(output_txt, 'w') as f:
        for line in pairs:
            f.write(line + '\n')

if __name__ == "__main__":
    root_dir = "./train"  # 根目录
    output_txt = "file_list.txt"
    collect_rgb_depth_pairs(root_dir, output_txt)
    print(f"输出完成，写入 {output_txt}")
