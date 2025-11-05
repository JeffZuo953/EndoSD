#!/usr/bin/env python3
"""
CSV 日志记录模块
"""

import os
import pandas as pd
from typing import Dict, List

class CSVLogger:
    """负责将最佳指标记录到CSV文件"""
    
    def __init__(self, filepath: str, columns: List[str]):
        """
        初始化 CSVLogger
        
        Args:
            filepath (str): CSV文件的保存路径
            columns (List[str]): CSV文件的列名
        """
        self.filepath = filepath
        self.columns = columns
        
        # 如果文件不存在，则创建并写入表头
        if not os.path.exists(self.filepath):
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_csv(self.filepath, index=False)
        else:
            # 如果文件存在，则读取
            self.df = pd.read_csv(self.filepath)

    def update(self, best_type: str, data: Dict[str, any]):
        """
        更新或添加一行记录
        
        Args:
            best_type (str): 最佳指标的类型 (e.g., 'best_absrel_kidney')
            data (Dict[str, any]): 要记录的数据，包含epoch和所有指标
        """
        # 确保 'best_type' 列存在
        if 'best_type' not in self.df.columns:
            self.df['best_type'] = ''

        # 查找是否已存在该类型的记录
        existing_index = self.df.index[self.df['best_type'] == best_type].tolist()
        
        # 将新数据转换为DataFrame
        new_row_df = pd.DataFrame([data])
        new_row_df['best_type'] = best_type

        if existing_index:
            # 更新现有行
            for col in new_row_df.columns:
                 if col in self.df.columns:
                    self.df.loc[existing_index[0], col] = new_row_df.iloc[0][col]
        else:
            # 追加新行
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
            
        # 保存到文件
        self.df.to_csv(self.filepath, index=False)