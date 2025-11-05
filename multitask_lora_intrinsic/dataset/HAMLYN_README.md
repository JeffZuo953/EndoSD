# Hamlyn 数据集使用说明

## 数据集结构

Hamlyn Endo Depth and Motion 数据集的目录结构应该如下：

```
/path/to/hamlyn/dataset/
├── rectified01/
│   ├── color/
│   │   ├── frame000000.jpg
│   │   ├── frame000001.jpg
│   │   └── ...
│   ├── depth/
│   │   ├── frame000000.png
│   │   ├── frame000001.png
│   │   └── ...
│   └── intrinsics.txt
├── rectified02/
│   ├── color/
│   ├── depth/
│   └── intrinsics.txt
└── ...
```

## 文件格式

- **RGB图像**: `.jpg` 格式，uint8 类型
- **深度图**: `.png` 格式，uint16 类型，单位为**毫米 (mm)**
- **深度范围**: 1-300 mm (0.001-0.3 m)
- **内参文件**: `intrinsics.txt`，3x3 矩阵

## 创建文件列表

在运行缓存生成之前，需要创建 `train.txt` 文件，格式如下：

```
/path/to/hamlyn/dataset/rectified01 frame000000
/path/to/hamlyn/dataset/rectified01 frame000001
/path/to/hamlyn/dataset/rectified01 frame000002
/path/to/hamlyn/dataset/rectified02 frame000000
/path/to/hamlyn/dataset/rectified02 frame000001
...
```

每行格式：`<序列路径> <帧ID>`

## 生成缓存

### 步骤 1: 修改配置

编辑 `cache_utils_data.py` 文件：

```python
from .hamlyn import HamlynDataset as Dataset
base_dir = "/path/to/hamlyn/dataset"  # 修改为实际路径
```

### 步骤 2: 运行缓存生成

```bash
cd /path/to/multitask_moe_lora
python -m dataset.cache_utils_data
```

### 步骤 3: 查看生成的文件

缓存生成完成后，会在 `{base_dir}/cache/` 目录下生成：

- `train_cache.txt` - 缓存文件路径列表
- `depth_statistics_report.json` - 深度统计报告 (JSON格式)
- `depth_statistics_report.txt` - 深度统计报告 (文本格式)

## 数据集参数

在 `HamlynDataset` 初始化时，可以设置以下参数：

```python
dataset = HamlynDataset(
    filelist_path="path/to/train.txt",
    rootpath="path/to/hamlyn/dataset",
    mode="train",
    size=(518, 518),              # 输出图像尺寸
    max_depth=0.3,                # 最大深度 (米)，默认 300mm
    depth_scale=1000.0,           # 深度缩放因子，默认 1000 (mm -> m)
    image_ext=".jpg",             # 图像文件扩展名
    depth_ext=".png",             # 深度文件扩展名
    cache_intrinsics=True,        # 是否缓存内参
)
```

## 生成文件列表脚本示例

```python
import os
from pathlib import Path

def generate_hamlyn_filelist(base_dir, output_file):
    """
    自动生成 Hamlyn 数据集的文件列表
    """
    file_list = []

    # 遍历所有 rectified 序列
    for seq_dir in sorted(Path(base_dir).glob("rectified*")):
        if not seq_dir.is_dir():
            continue

        color_dir = seq_dir / "color"
        if not color_dir.exists():
            continue

        # 获取所有图像文件
        for img_file in sorted(color_dir.glob("*.jpg")):
            frame_id = img_file.stem  # 不带扩展名的文件名
            file_list.append(f"{seq_dir} {frame_id}\n")

    # 写入文件
    with open(output_file, 'w') as f:
        f.writelines(file_list)

    print(f"生成的文件列表包含 {len(file_list)} 个样本")
    print(f"已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    base_dir = "/path/to/hamlyn/dataset"
    output_file = f"{base_dir}/train.txt"
    generate_hamlyn_filelist(base_dir, output_file)
```

## 注意事项

1. **深度单位**: Hamlyn 数据集的深度以毫米 (mm) 存储，代码会自动转换为米 (m)
2. **内参文件**: 每个序列应该有一个 `intrinsics.txt` 文件，包含 3x3 的相机内参矩阵
3. **有效深度范围**: 代码会自动将深度值裁剪到 [0, 0.3] 米的范围
4. **文件路径**: 确保 `train.txt` 中的路径是绝对路径或相对于工作目录的正确路径

## 验证数据加载

在生成缓存之前，可以先测试数据加载是否正常：

```python
from dataset.hamlyn import HamlynDataset

dataset = HamlynDataset(
    filelist_path="/path/to/train.txt",
    rootpath="/path/to/hamlyn/dataset",
    mode="train",
)

# 测试加载第一个样本
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Depth shape: {sample['depth'].shape}")
print(f"Max depth: {sample['max_depth']}")
print(f"Intrinsics shape: {sample['intrinsics'].shape}")
```
