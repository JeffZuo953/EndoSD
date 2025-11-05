import os
import re
import numpy as np
import logging
from datetime import datetime, timezone, timedelta

logs = set()


class BeijingFormatter(logging.Formatter):
    """自定义格式化器，使用北京时间（UTC+8）"""
    def formatTime(self, record, datefmt=None):
        # 将时间戳转换为北京时间
        dt = datetime.fromtimestamp(record.created, tz=timezone(timedelta(hours=8)))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 检查分布式训练的rank，只有rank 0输出日志
    rank = 0
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])

    # 只有主进程（rank 0）输出日志
    if rank != 0:
        logger.addFilter(lambda record: False)
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = BeijingFormatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_new_filename_from_path(path_str: str) -> str:
    """
    从输入文件路径生成新的文件名。
    规则：取后三级目录和文件名（不含扩展名），用双下划线连接。
    例如: /a/b/c/d/e.jpg -> c__d__e
    """
    import os

    parts = path_str.replace('\\', '/').split('/')

    # 至少需要一级目录和文件名
    if len(parts) < 2:
        # 如果路径太短，只返回无扩展名的文件名
        return os.path.splitext(parts[-1])[0]

    # 取最后三个部分（两级目录 + 文件名）
    relevant_parts = parts[-3:]

    # 去掉文件名的扩展名
    relevant_parts[-1] = os.path.splitext(relevant_parts[-1])[0]

    return "__".join(relevant_parts)
