# MT Dataset Report

> 统计说明：pt 大小与标签频次基于每个数据集前 10 个缓存样本的快速扫描，仅用于 sanity check。

## NO 数据集

### Kidney3D

- 示例 pt：`/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train/01/noLaser/image/Image0001.pt` （约 1.73 MB）
- 包含字段：image, depth, valid_mask, semseg_mask, image_path, depth_path, mask_path, source_type, dataset_name, max_depth
- Tensor 形状：image: (3, 291, 518) torch.float16, depth: (291, 518) torch.float32, valid_mask: (291, 518) torch.bool, semseg_mask: (291, 518) torch.uint8

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt` | 3000 |
| val | `/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt` | 1000 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,480,396 | 98.21% |
| 255 | ignore | 26,984 | 1.79% |

### RIRS-SegC

- 示例 pt：`/data/ziyi/multitask/data/NO/RIRS-SegC/img/case06_01/0641.pt` （约 2.05 MB）
- 包含字段：image, semseg_mask, image_path, valid_mask
- Tensor 形状：image: (3, 518, 518) torch.float16, semseg_mask: (518, 518) torch.uint8, valid_mask: (518, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/RIRS-SegC/train_split_cache.txt` | 1996 |
| val | `/data/ziyi/multitask/data/NO/RIRS-SegC/val_split_cache.txt` | 556 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,293,253 | 48.20% |
| 1 | stone | 1,361,539 | 50.74% |
| 2 | instrument | 24,230 | 0.90% |
| 255 | ignore | 4,218 | 0.16% |

### RIRS-SegP

- 示例 pt：`/data/ziyi/multitask/data/NO/RIRS-SegP/img/Pig_Basket_Stone/0001.pt` （约 2.05 MB）
- 包含字段：image, image_path, semseg_mask, valid_mask
- Tensor 形状：image: (3, 517, 518) torch.float16, semseg_mask: (517, 518) torch.uint8, valid_mask: (517, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/RIRS-SegP/train_cache.txt` | 499 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 2,245,213 | 83.84% |
| 1 | pig-stone lesion | 432,842 | 16.16% |
| 255 | ignore | 5 | 0.00% |

### clinicDB

- 示例 pt：`/data/ziyi/multitask/data/NO/clinicDB/cache/PNG/Original/1.pt` （约 1.54 MB）
- 包含字段：image, semseg_mask, image_path, valid_mask
- Tensor 形状：image: (3, 388, 518) torch.float16, semseg_mask: (388, 518) torch.uint8, valid_mask: (388, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/clinicDB/cache/train_cache.txt` | 612 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,088,330 | 54.15% |
| 255 | ignore | 566,854 | 28.20% |
| 3 | polyp | 354,656 | 17.65% |

### CVC-EndoScene

- 示例 pt：`/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache/images/1.pt` （约 1.78 MB）
- 包含字段：image, semseg_mask, image_path, valid_mask
- Tensor 形状：image: (3, 451, 518) torch.float16, semseg_mask: (451, 518) torch.uint8, valid_mask: (451, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache/train_cache.txt` | 547 |
| val | `/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache/train_cache.txt` | 183 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,778,433 | 76.13% |
| 255 | ignore | 472,100 | 20.21% |
| 3 | polyp | 85,647 | 3.67% |

### Kvasir-SEG

- 示例 pt：`/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/train/images/cju0qkwl35piu0993l0dewei2.pt` （约 1.53 MB）
- 包含字段：image, semseg_mask, image_path
- Tensor 形状：image: (3, 441, 518) torch.float16, semseg_mask: (441, 518) torch.uint8

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/train_cache.txt` | 880 |
| val | `/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/val_cache.txt` | 120 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,584,103 | 69.19% |
| 255 | ignore | 340,590 | 14.88% |
| 3 | polyp | 364,867 | 15.94% |

### bkai-igh-neopolyp

- 示例 pt：`/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache/train/7979870116d06046f5d8dd18841d318d.pt` （约 1.54 MB）
- 包含字段：image, semseg_mask, image_path, valid_mask
- Tensor 形状：image: (3, 388, 518) torch.float16, semseg_mask: (388, 518) torch.uint8, valid_mask: (388, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache/train_cache.txt` | 1000 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,678,514 | 83.51% |
| 255 | ignore | 266,196 | 13.24% |
| 3 | polyp | 65,130 | 3.24% |

### ETIS-LaribPolypDB

- 示例 pt：`/data/ziyi/multitask/data/NO/ETIS-LaribPolypDB/images/1.pt` （约 1.61 MB）
- 包含字段：image, semseg_mask, image_path, valid_mask
- Tensor 形状：image: (3, 408, 518) torch.float16, semseg_mask: (408, 518) torch.uint8, valid_mask: (408, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| val | `/data/ziyi/multitask/data/NO/ETIS-LaribPolypDB/train_cache.txt` | 196 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,971,146 | 93.27% |
| 255 | ignore | 124,867 | 5.91% |
| 3 | polyp | 17,427 | 0.82% |


## LS 数据集

### EndoSynth

- 示例 pt：`/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train/EndoSynth/samples/000000.pt` （约 1.73 MB）
- 包含字段：image, depth, valid_mask, semseg_mask, image_path, depth_path, mask_path, source_type, max_depth
- Tensor 形状：image: (3, 291, 518) torch.float16, depth: (291, 518) torch.float32, valid_mask: (291, 518) torch.bool, semseg_mask: (291, 518) torch.uint8

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt` | 2052 |
| val | `/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt` | 229 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,062,872 | 70.51% |
| 1 | cadiere forceps | 171,460 | 11.37% |
| 2 | fenestrated bipolar | 52,649 | 3.49% |
| 5 | monopolar curved shears | 187,245 | 12.42% |
| 6 | maryland bipolar | 33,154 | 2.20% |

### EndoVis2017

- 示例 pt：`/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train/instrument_dataset_1/output_rectified/left/frame000.pt` （约 2.46 MB）
- 包含字段：image, depth, valid_mask, semseg_mask, image_path, depth_path, mask_path, source_type, frame_token, sequence, max_depth, intrinsics, intrinsics_key
- Tensor 形状：image: (3, 414, 518) torch.float16, depth: (414, 518) torch.float32, valid_mask: (414, 518) torch.bool, semseg_mask: (414, 518) torch.uint8, intrinsics: (3, 3) torch.float32

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train_cache.txt` | 1800 |
| eval | `/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/eval_cache.txt` | 600 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,775,485 | 82.79% |
| 1 | bipolar forceps | 106,636 | 4.97% |
| 2 | prograsp forceps | 44,801 | 2.09% |
| 3 | large needle driver | 138,302 | 6.45% |
| 7 | other | 79,296 | 3.70% |

### EndoVis2018

- 示例 pt：`/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train/endovis2018_seg_depth/output_rectified/left/seq_10_frame000.pt` （约 2.46 MB）
- 包含字段：image, depth, valid_mask, semseg_mask, image_path, depth_path, source_type, frame_token, sequence, max_depth, intrinsics, intrinsics_key
- Tensor 形状：image: (3, 414, 518) torch.float16, depth: (414, 518) torch.float32, valid_mask: (414, 518) torch.bool, semseg_mask: (414, 518) torch.uint8, intrinsics: (3, 3) torch.float32

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt` | 1639 |
| val | `/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt` | 587 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,948,561 | 90.86% |
| 1 | bipolar forceps | 128,630 | 6.00% |
| 8 | suction instrument | 67,329 | 3.14% |

### EndoNeRF

- 示例 pt：`/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train/endonerf/cutting_tissues_twice/images/000000.pt` （约 2.87 MB）
- 包含字段：image, depth, valid_mask, semseg_mask, depth_valid_mask, seg_valid_mask, image_path, depth_path, source_type, sequence, max_depth, intrinsics_key
- Tensor 形状：image: (3, 414, 518) torch.float16, depth: (414, 518) torch.float32, valid_mask: (414, 518) torch.bool, semseg_mask: (414, 518) torch.uint8, depth_valid_mask: (414, 518) torch.bool, seg_valid_mask: (414, 518) torch.bool

**列表文件**

| Split | 路径 | 条目数 |
|---|---|---|
| train | `/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt` | 141 |
| val | `/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt` | 15 |

**分割标签（采样像素）**

| 标签ID | 含义 | 像素数 | 占比 |
|---|---|---|---|
| 0 | background | 1,775,129 | 82.78% |
| 1 | tissue | 21,733 | 1.01% |
| 255 | ignore | 51,800 | 2.42% |
| 6 | instrument | 295,858 | 13.80% |

