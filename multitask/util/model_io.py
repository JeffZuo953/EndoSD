import torch
import logging
import pprint

def load_weights(model: torch.nn.Module, pretrained_from: str) -> None:
    """
    Loads pretrained weights into a model with flexible key matching.
    """
    logger = logging.getLogger() # Get root logger
    logger.info(f"Loading pretrained weights from: {pretrained_from}")
    
    checkpoint = torch.load(pretrained_from, map_location="cpu")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    
    filtered_state_dict = {}
    non_matching_keys = []

    for k, v in state_dict.items():
        target_k = k
        if k.startswith('module.pretrained.'):
            target_k = 'backbone.' + k[len('module.pretrained.'):]
        elif k.startswith('module.depth_head.'):
            target_k = k[len('module.'):]
        elif k.startswith('module.'):
            target_k = k[len('module.'):]
            if target_k.startswith('pretrained.'):
                target_k = 'backbone.' + target_k[len('pretrained.'):]
        elif k.startswith('pretrained.'):
            target_k = 'backbone.' + k[len('pretrained.'):]
        else:
            potential_k = 'backbone.' + k
            if potential_k in model_state_dict:
                target_k = potential_k

        if target_k in model_state_dict:
            filtered_state_dict[target_k] = v
        else:
            non_matching_keys.append(k)
            
    model.load_state_dict(filtered_state_dict, strict=False)

    loaded_keys = list(filtered_state_dict.keys())
    new_keys = [k for k in model_keys if k not in loaded_keys]

    logger.info(f"Successfully loaded {len(loaded_keys)} parameters.")
    if non_matching_keys:
        logger.warning(f"Skipped {len(non_matching_keys)} non-matching parameters from checkpoint:")
        logger.warning(pprint.pformat(non_matching_keys))
    if new_keys:
        logger.info(f"Initialized {len(new_keys)} new parameters:")
        logger.info(pprint.pformat(new_keys))