import torch
import sys


def print_pt_structure(pt_path):
    data = torch.load(pt_path, map_location='cpu')  # 加载文件

    def print_structure(obj, indent=0):
        prefix = ' ' * indent
        if isinstance(obj, dict):
            print(f"{prefix}Dict with {len(obj)} keys:")
            for k, v in obj.items():
                print(f"{prefix}  Key: {k} -> ", end='')
                print_structure(v, indent + 4)
        elif isinstance(obj, torch.Tensor):
            summary = f"Tensor, shape={tuple(obj.shape)}, dtype={obj.dtype}"
            try:
                if torch.is_tensor(obj) and obj.numel() > 0:
                    obj_float = obj.float()
                    stats = f", min={obj.min().item():.4f}, max={obj.max().item():.4f}, mean={obj_float.mean().item():.4f}"
                else:
                    stats = ""
                summary += stats
            except Exception as e:
                summary += f", 但无法计算统计信息: {e}"
            print(f"{prefix}{summary}")
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}{type(obj).__name__} of length {len(obj)}:")
            for i, item in enumerate(obj):
                print(f"{prefix}  [{i}]: ", end='')
                print_structure(item, indent + 4)
        else:
            print(f"{prefix}{type(obj).__name__}: {str(obj)[:100]}")  # 简短打印其他类型

    print_structure(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python show.py your_model.pt")
        sys.exit(1)

    pt_file = sys.argv[1]
    print_pt_structure(pt_file)
