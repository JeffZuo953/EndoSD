#!/usr/bin/env python
"""
Debug script to inspect checkpoint keys and help resolve loading issues
"""
import torch
import argparse
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Load and inspect checkpoint structure"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If checkpoint is a dict with 'state_dict' or 'model' key
        if isinstance(checkpoint, dict):
            print(f"\nCheckpoint type: dict with keys: {list(checkpoint.keys())[:10]}...")
            
            # Try common patterns
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Found 'state_dict' in checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Found 'model' in checkpoint")
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
                print("Using checkpoint dict as state_dict")
        else:
            state_dict = checkpoint
            
        # Analyze keys
        all_keys = list(state_dict.keys()) if state_dict else []
        print(f"\nTotal keys: {len(all_keys)}")
        
        # Group keys by prefix
        prefixes = {}
        for key in all_keys:
            prefix = key.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(key)
        
        print(f"\nKey prefixes found: {list(prefixes.keys())}")
        
        # Show sample keys for each prefix
        for prefix, keys in prefixes.items():
            print(f"\n{prefix}: {len(keys)} keys")
            print(f"  Sample keys: {keys[:5]}")
            
        # Look for backbone/pretrained keys specifically
        print("\n=== Backbone/Pretrained Keys ===")
        backbone_keys = [k for k in all_keys if 'pretrained' in k or 'backbone' in k]
        print(f"Found {len(backbone_keys)} backbone/pretrained keys")
        if backbone_keys:
            print("Sample backbone keys:")
            for k in backbone_keys[:10]:
                print(f"  {k}")
                
        # Look for block/attention keys
        print("\n=== Block/Attention Keys ===")
        block_keys = [k for k in all_keys if 'block' in k or 'attn' in k]
        print(f"Found {len(block_keys)} block/attention keys")
        if block_keys:
            print("Sample block keys:")
            for k in block_keys[:10]:
                print(f"  {k}")
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        

def main():
    parser = argparse.ArgumentParser(description='Debug checkpoint keys')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint_path)
    

if __name__ == '__main__':
    main()
