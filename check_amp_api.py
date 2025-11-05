#!/usr/bin/env python3
"""
验证PyTorch AMP API的脚本
"""
import torch
import sys

def check_amp_api():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print("-" * 50)
    
    # 检查 torch.cuda.amp
    try:
        from torch.cuda.amp import GradScaler, autocast
        print("✅ torch.cuda.amp 可用")
        
        # 测试 autocast 参数
        try:
            with autocast(enabled=True):
                print("✅ torch.cuda.amp.autocast() 无参数工作正常")
        except Exception as e:
            print(f"❌ torch.cuda.amp.autocast() 无参数失败: {e}")
            
        try:
            with autocast(enabled=True, device_type='cuda'):
                print("✅ torch.cuda.amp.autocast(device_type='cuda') 工作正常")
        except Exception as e:
            print(f"❌ torch.cuda.amp.autocast(device_type='cuda') 失败: {e}")
            
    except ImportError as e:
        print(f"❌ torch.cuda.amp 不可用: {e}")
    
    print("-" * 50)
    
    # 检查 torch.amp
    try:
        from torch.amp import GradScaler, autocast
        print("✅ torch.amp 可用")
        
        # 测试 autocast 参数
        try:
            with autocast(device_type='cuda', enabled=True):
                print("✅ torch.amp.autocast(device_type='cuda') 工作正常")
        except Exception as e:
            print(f"❌ torch.amp.autocast(device_type='cuda') 失败: {e}")
            
        try:
            with autocast(enabled=True):
                print("✅ torch.amp.autocast() 无device_type工作正常")
        except Exception as e:
            print(f"❌ torch.amp.autocast() 无device_type失败: {e}")
            
    except ImportError as e:
        print(f"❌ torch.amp 不可用: {e}")

if __name__ == "__main__":
    check_amp_api()