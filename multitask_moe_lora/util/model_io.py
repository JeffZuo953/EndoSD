import torch
import logging
import re
from typing import Dict, Any

from .config import TrainingConfig


def remap_checkpoint_keys(state_dict: Dict[str, Any], model: torch.nn.Module, config: TrainingConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Remaps checkpoint keys to match the current model structure, handling LoRA and MoE layers.
    Supports bidirectional mapping:
    - original->lora: adds .linear. wrappers
    - lora->original: removes .linear. wrappers
    """
    # Get the actual model (unwrap DDP if present)
    actual_model = model.module if hasattr(model, 'module') else model
    model_state_dict = actual_model.state_dict()
    new_state_dict = {}

    # Debug: show config mode and detected LoRA usage
    config_mode = getattr(config, 'mode', 'original')
    config_mode_str = str(config_mode).lower() if isinstance(config_mode, str) else ""
    model_uses_lora = bool(getattr(actual_model, "use_lora", False))
    extra_lora_modes = {"mtoat", "endounid"}
    is_lora_mode = model_uses_lora or (config_mode_str in extra_lora_modes) or ("lora" in config_mode_str)
    logger.info(f"Config mode: {config_mode}, model.use_lora={model_uses_lora}, is_lora_mode={is_lora_mode}")

    # Debug: show original checkpoint attention keys
    logger.info("--- Original Checkpoint Attention Keys (blocks.0.attn) ---")
    attn_keys_sample = [k for k in list(state_dict.keys()) if 'blocks.0.attn' in k]
    for k in sorted(attn_keys_sample):
        logger.info(f"  - {k}")
    logger.info("--- Model Attention Keys (blocks.0.attn) ---")
    model_attn_keys = [k for k in list(model_state_dict.keys()) if 'blocks.0.attn' in k]
    for k in sorted(model_attn_keys):
        logger.info(f"  - {k}")
    logger.info("---------------------------------")

    for k, v in state_dict.items():
        # Start with the original key
        original_k = k
        target_k = k

        # --- 1. Strip all known prefixes to get a base key ---
        prefixes_to_strip = [
            'module.pretrained.',
            'module.backbone.',
            'pretrained.',
            'module.',
            'model.',   # common in timm/dinov3 checkpoints
            'backbone.' # allow re-prefixing consistently below
        ]
        for prefix in prefixes_to_strip:
            if target_k.startswith(prefix):
                target_k = target_k[len(prefix):]
                break # Stop after stripping the first matching prefix

        # --- 2. Add 'backbone.' prefix if it's a backbone key ---
        # Heuristic: if it's not a head key, it's likely a backbone key.
        if not (target_k.startswith('depth_head.') or target_k.startswith('seg_head.') or target_k.startswith('camera_head.')):
             # Re-add backbone prefix if it's not already there
            if not target_k.startswith('backbone.'):
                base_key = target_k
                target_k = 'backbone.' + base_key
            else: # a key like backbone.*
                base_key = target_k[len('backbone.'):]
        else:
            base_key = target_k

        # Debug specific keys
        if 'blocks.0.attn' in original_k:
            logger.debug(f"Processing key: {original_k} -> {target_k}")

        # --- 3. Handle LoRA and MoE transformations on the base key ---
        final_keys = []

        # LoRA transformation for attention layers (dinov2 and dinov3)
        # LoRA transformation for attention layers should only apply in LoRA modes
        # Handle attention layer transformations (bidirectional)
        if '.attn.qkv.' in target_k or '.attn.proj.' in target_k:
            if is_lora_mode:
                # Model expects .linear., checkpoint might not have it
                if '.linear.' not in target_k:
                    # Add .linear. wrapper
                    lora_key = target_k.replace('.attn.qkv.weight', '.attn.qkv.linear.weight') \
                                      .replace('.attn.proj.weight', '.attn.proj.linear.weight') \
                                      .replace('.attn.qkv.bias', '.attn.qkv.linear.bias') \
                                      .replace('.attn.proj.bias', '.attn.proj.linear.bias')
                    if 'blocks.0.attn' in original_k:
                        logger.info(f"[LoRA mode] Adding .linear.: {target_k} -> {lora_key}")
                    final_keys.append(lora_key)
                else:
                    # Already has .linear., use as-is
                    if 'blocks.0.attn' in original_k:
                        logger.info(f"[LoRA mode] Already has .linear.: {target_k}")
                    final_keys.append(target_k)
            else:
                # Model is in original mode - remove .linear. if present
                if '.linear.' in target_k:
                    # Remove .linear. wrapper
                    plain_key = target_k.replace('.attn.qkv.linear.weight', '.attn.qkv.weight') \
                                       .replace('.attn.proj.linear.weight', '.attn.proj.weight') \
                                       .replace('.attn.qkv.linear.bias', '.attn.qkv.bias') \
                                       .replace('.attn.proj.linear.bias', '.attn.proj.bias')
                    if 'blocks.0.attn' in original_k:
                        logger.info(f"[Original mode] Removing .linear.: {target_k} -> {plain_key}")
                    final_keys.append(plain_key)
                else:
                    # Already plain, use as-is
                    if 'blocks.0.attn' in original_k:
                        logger.info(f"[Original mode] Already plain: {target_k}")
                    final_keys.append(target_k)

        # FFN transformations (LoRA + MoE) for MLP/SwiGLU layers
        elif '.mlp.' in base_key and '.experts.' not in base_key:
            # Regex to capture dinov2 (fc1/fc2) and dinov3 (w1/w2/w3/fc1/fc2) patterns
            match = re.match(r"(blocks\.\d+\.mlp)\.(fc[12]|w[123])\.(weight|bias)", base_key)
            if match:
                prefix, layer, suffix = match.groups()
                # Decide whether this block is MoE-enabled by checking model keys
                has_moe = any(k.startswith(f"backbone.{prefix}.experts.") for k in model_state_dict.keys())
                if has_moe:
                    # Broadcast to all experts in MoE layer
                    for i in range(config.num_experts):
                        expert_key = f"backbone.{prefix}.experts.{i}.{layer}.linear.{suffix}"
                        final_keys.append(expert_key)
                else:
                    # Non-MoE: support LoRA-wrapped Linear ('.linear.') and plain Linear (fallback)
                    lora_linear_key = f"backbone.{prefix}.{layer}.linear.{suffix}"
                    final_keys.append(lora_linear_key)
                    final_keys.append(target_k)
            else:
                final_keys.append(target_k)  # Fallback for non-matching mlp keys
        else:
            final_keys.append(target_k)

        # --- 4. Add valid keys to the new state dictionary ---
        matched = False
        for final_k in final_keys:
            if final_k in model_state_dict:
                new_state_dict[final_k] = v
                matched = True
                if 'blocks.0.attn' in original_k:
                    logger.info(f"[MATCHED] {original_k} -> {final_k}")
            else:
                # This log can be noisy, enable for deep debugging if needed
                if 'blocks.0.attn' in original_k:
                    logger.warning(f"[NOT MATCHED] {original_k} -> {final_k} not in model.")
                pass

        if not matched and 'blocks.0.attn' in original_k:
            logger.error(f"[FAILED] No match found for {original_k}")

    return new_state_dict
