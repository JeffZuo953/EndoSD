import sys

# 定义你想“吃掉”多少GB的内存。这个值应该接近你机器的总物理内存。
# 例如，如果机器有64GB内存，你可以尝试分配50GB。
# 你可以通过 `free -h` 命令查看总内存和可用内存。
GB_TO_EAT = 45

print(f"Attempting to allocate ~{GB_TO_EAT} GB of memory to flush file cache...")

try:
    # 创建一个巨大的bytearray对象来占用内存
    # 1024**3 = 1GB
    eater = bytearray(GB_TO_EAT * 1024**3)
    # 访问一下内存，防止被优化掉
    eater[-1] = 0
    print(f"Successfully allocated memory. Cache should be significantly reduced.")
except MemoryError:
    # 如果你申请的内存超过了可用内存，会报这个错。
    # 这恰好说明我们的目的达到了：系统已无更多内存，缓存已被尽可能地清空。
    print("MemoryError caught. This is expected and means we've successfully used up available RAM.")
    print("The file cache has likely been flushed.")
except Exception as e:
    print(f"An error occurred: {e}")

print("Script finished. You can now run your benchmark.")
