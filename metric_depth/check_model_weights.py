import torch
import torch.nn as nn
from collections import OrderedDict  # 用于创建有序字典的模型示例
from depth_anything_v2.dpt_lora import DepthAnythingV2_LoRA
from depth_anything_v2.dpt import DepthAnythingV2

# --- 表格打印函数 (与之前的脚本相同) ---
def print_table(headers, data):
    """
    将收集到的数据格式化并打印为表格。

    Args:
        headers (list): 包含列标题字符串的列表。
        data (list): 包含元组的列表，每个元组是一行数据。
    """
    if not data:
        print("未找到任何参数或缓冲区，或者模型为空。")
        return

    # 计算每列的最大宽度
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            # 确保单元格是字符串，以计算长度
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 创建格式化字符串
    header_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
    row_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(['-' * w for w in col_widths])

    # 打印表格
    print(header_format.format(*headers))
    print(separator)
    for row in data:
        # 确保行中的所有元素都是字符串以便格式化
        formatted_row = [str(cell) for cell in row]
        print(row_format.format(*formatted_row))

# --- 用于提取模型权重信息的函数 ---
def get_model_weights_info(model: nn.Module):
    """
    从模型的 state_dict 中提取参数和缓冲区的信息。

    Args:
        model (nn.Module): PyTorch 模型实例。

    Returns:
        list: 一个元组列表，每个元组代表一个参数或缓冲区：
              (名称, 类型名称, 表示字符串). 按名称排序。
    """
    weights_info = []
    # 获取包含参数和持久缓冲区的状态字典
    # 使用 to('cpu') 确保所有张量都在 CPU 上，以便设备信息一致
    # 注意：如果在 GPU 上检查模型，可以移除 .to('cpu')，但要确保模型本身在 GPU 上
    try:
        # 尝试将模型和状态字典移至 CPU 以获得一致的设备报告
        # 这不会修改原始模型对象
        state_dict = model.to('cpu').state_dict()
        print("注意: 正在检查模型在 CPU 上的状态字典副本。")
    except Exception as e:
        print(f"警告: 无法将模型移动到 CPU 进行检查 ({e})。将使用模型的当前设备。")
        state_dict = model.state_dict()


    # 对键进行排序以获得一致的输出顺序
    # state_dict 默认是 OrderedDict，但显式排序更安全
    sorted_keys = sorted(state_dict.keys())

    for name in sorted_keys:
        tensor = state_dict[name]
        # 确认它是一个张量 (通常 state_dict 只包含张量)
        if isinstance(tensor, torch.Tensor):
            type_name = type(tensor).__name__  # 通常是 'Tensor'
            representation = f"shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
            weights_info.append((name, type_name, representation))
        else:
            # 处理 state_dict 中可能存在的非张量项（虽然不常见）
            type_name = type(tensor).__name__
            value_repr = repr(tensor)
            max_len = 60 # 截断过长的表示
            if len(value_repr) > max_len:
                value_repr = value_repr[:max_len-3] + "..."
            weights_info.append((name, type_name, value_repr))

    return weights_info

# --- 主执行部分 ---
if __name__ == "__main__":
    print("正在创建一个示例 PyTorch 模型...")

    # --- 定义一个示例模型 ---
    # 如果你需要检查自己的模型，请在这里实例化你的模型类
    # 例如: my_model = YourModelClass(*args, **kwargs)
    example_model = DepthAnythingV2(
      encoder='vits',
      features=64,
      out_channels=[48, 96, 192, 384],
      max_depth=1000,
    )

    # example_model = DepthAnythingV2_LoRA(
    #   encoder='vits',
    #   features=64,
    #   out_channels=[48, 96, 192, 384],
    #   max_depth=1000,
    #   lora_r=3,
    #   lora_alpha=3,
    #   lora_dropout=0.0,
    #   lora_bias="none",
    # )

    # 打印模型结构（可选）
    print("模型结构:")
    print(example_model)
    print("-" * 80)

    # 将模型设置为评估模式（这会影响 BatchNorm 等层的行为，例如不更新 running mean/var）
    example_model.eval()

    print(f"正在从模型 '{type(example_model).__name__}' 中提取权重和缓冲区...")
    # 获取权重数据
    weights_data = get_model_weights_info(example_model)

    print("-" * 80)
    print("模型参数和缓冲区 (state_dict 内容):")
    print("-" * 80)

    # 定义表格标题
    headers = ["Parameter/Buffer Name", "Type", "Shape / Info"]
    # 以表格格式打印数据
    print_table(headers, weights_data)

    print("-" * 80)
    print(f"在状态字典中找到 {len(weights_data)} 个项目。")

    # === 如何使用你自己的模型 ===
    # 1. 导入你的模型类: from your_model_file import YourModelClass
    # 2. 实例化你的模型: my_model = YourModelClass(*constructor_args)
    # 3. (可选) 加载预训练权重: state = torch.load('your_weights.pth'); my_model.load_state_dict(state)
    # 4. 调用函数: weights_data = get_model_weights_info(my_model)
    # 5. 打印表格: print_table(headers, weights_data)