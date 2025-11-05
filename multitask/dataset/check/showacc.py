import torch
import sys
from collections import Counter

def flatten_deep_list(nested_list):
    """
    递归地将任意深度的嵌套列表或元组展平为一维列表。
    """
    flat_list = []
    for item in nested_list:
        # 如果元素是列表或元组，则递归调用
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten_deep_list(item))
        # 否则，直接添加元素
        else:
            flat_list.append(item)
    return flat_list

def count_key_values(pt_path, key_name):
    """
    加载一个 .pt 文件，访问指定的 key，并打印其中每个唯一值的出现次数。
    此版本可以处理任意维度的 Tensor 和任意深度的嵌套列表。
    """
    try:
        # 加载文件到 CPU
        data = torch.load(pt_path, map_location='cpu')

        # 检查加载的数据是否为字典
        if not isinstance(data, dict):
            print(f"错误：文件 '{pt_path}' 中的数据不是一个字典。")
            return

        # 检查指定的 key 是否存在
        if key_name not in data:
            print(f"错误：在 .pt 文件中未找到 key '{key_name}'。")
            print(f"可用的 keys 是: {list(data.keys())}")
            return

        # 获取 key 对应的 value
        values = data[key_name]
        values_list = []

        # 根据值的类型进行展平处理
        if isinstance(values, torch.Tensor):
            # Tensor 自带的 flatten 方法可以处理任意维度
            values_list = values.flatten().tolist()
        elif isinstance(values, (list, tuple)):
            # 使用辅助函数处理任意深度的嵌套列表/元组
            values_list = flatten_deep_list(values)
        else:
            # 对于其他单一类型的数据，直接放入列表进行计数
            print(f"提示: Key '{key_name}' 的值类型为 {type(values).__name__}，不是一个多维结构。将直接统计该值。")
            values_list = [values]

        # 使用 Counter 进行计数
        value_counts = Counter(values_list)

        print(f"\n--- Key '{key_name}' 的值统计信息 ---")
        if not value_counts:
             print("  (没有找到可统计的值)")
        else:
            # 为了输出稳定，对结果按值进行排序
            for value, count in sorted(value_counts.items(), key=lambda item: str(item[0])):
                print(f"  值: {str(value):<15} | 出现次数: {count}")
        print("--------------------------------------\n")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{pt_path}'")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # 检查命令行参数数量是否正确
    if len(sys.argv) != 3:
        print("\n用法: python show.py <your_model.pt> <key_name>\n")
        print("示例: python show.py data.pt 'labels'\n")
        sys.exit(1)

    # 从命令行获取文件路径和 key 的名称
    pt_file = sys.argv[1]
    target_key = sys.argv[2]

    # 调用函数执行统计和打印
    count_key_values(pt_file, target_key)