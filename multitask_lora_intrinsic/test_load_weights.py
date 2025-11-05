import torch
import sys
import os

# Use absolute import from the project root
from multitask_moe_lora.depth_anything_v2.dinov3 import DINOv3

def inspect_checkpoint(checkpoint_path):
    """Loads a checkpoint and prints information about its keys."""
    print(f"--- Inspecting Checkpoint: {checkpoint_path} ---")
    
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at '{checkpoint_path}'")
        return

    try:
        # Load the checkpoint onto CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine the actual state dictionary
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        print(f"Successfully loaded checkpoint. Found {len(state_dict)} keys.")
        
        # --- Print all keys to identify the structure ---
        print("\n--- All Keys in State Dictionary (first 20) ---")
        for i, key in enumerate(state_dict.keys()):
            if i >= 20:
                print("...")
                break
            print(f"  - '{key}'")

        # --- Specifically search for 'pos_embed' ---
        print("\n--- Searching for 'pos_embed' keys ---")
        found = False
        for key in state_dict.keys():
            if 'pos_embed' in key:
                print(f"  - Found potential key: '{key}' with shape {state_dict[key].shape}")
                found = True
        if not found:
            print("  - No keys containing 'pos_embed' were found.")
            
        # --- Attempt to load into our DINOv3 model ---
        print("\n--- Attempting to load weights into DinoVisionTransformer ---")
        model = DINOv3(model_name='dinov3_vits16plus', img_size=518)
        
        # Basic remapping: strip 'backbone.' if present, as our model is the backbone itself
        remapped_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('backbone.', '')
            remapped_dict[new_k] = v

        incompatible_keys = model.load_state_dict(remapped_dict, strict=False)
        
        print("\n--- `load_state_dict` Results ---")
        if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
            print("  ✅ Success! All keys matched perfectly.")
        else:
            if incompatible_keys.missing_keys:
                print(f"  🟡 Missing Keys ({len(incompatible_keys.missing_keys)}): These keys were in the model but not in the checkpoint.")
                for i, key in enumerate(incompatible_keys.missing_keys):
                    if i >= 10:
                        print("     ...")
                        break
                    print(f"     - {key}")
            
            if incompatible_keys.unexpected_keys:
                print(f"  🟡 Unexpected Keys ({len(incompatible_keys.unexpected_keys)}): These keys were in the checkpoint but not in the model.")
                for i, key in enumerate(incompatible_keys.unexpected_keys):
                    if i >= 10:
                        print("     ...")
                        break
                    print(f"     - {key}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # Path from the train_multitask_depth_seg_dinov3_un_frozen.sh script
    # IMPORTANT: You might need to adjust this path based on where you run the script from.
    # This path assumes your data is in /data/ziyi/multitask/
    PRETRAINED_WEIGHTS = "/data/ziyi/multitask/pretained/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    
    inspect_checkpoint(PRETRAINED_WEIGHTS)