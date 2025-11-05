import torch
from torch.utils.data import Dataset
import os

class PredPtDataset(Dataset):
    """
    一个从txt文件加载pt文件路径的数据集。
    txt文件中每行都是一个pt文件的完整路径。
    """
    def __init__(self, txt_file: str):
        super().__init__()
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Prediction txt file not found: {txt_file}")
        
        with open(txt_file, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
            
        print(f"Found {len(self.file_paths)} prediction files in {txt_file}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.file_paths[index]
        try:
            # 加载pt文件，它应该包含一个tensor
            data = torch.load(path, map_location='cpu')
            return data
        except Exception as e:
            print(f"Error loading file: {path}")
            raise e
