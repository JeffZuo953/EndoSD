# Mask2Former configuration for 2-class segmentation
# Based on DINOv2 + Mask2Former reference implementation

# Dataset configuration
num_things_classes = 1  # Your foreground class
num_stuff_classes = 1   # Your background class  
num_classes = 2         # Total classes (excluding background in loss)

# Model configuration
model_config = {
    'type': 'Mask2FormerHead',
    'in_channels': [384, 384, 384, 384],  # DINOv2 ViT-S features
    'feat_channels': 256,                 # Reduced from 1536 for efficiency
    'out_channels': 256,
    'in_index': [0, 1, 2, 3],
    'num_things_classes': num_things_classes,
    'num_stuff_classes': num_stuff_classes,
    'num_queries': 100,
    'num_transformer_feat_level': 3,
    
    # Pixel decoder configuration
    'pixel_decoder': {
        'type': 'MSDeformAttnPixelDecoder',
        'num_outs': 3,
        'norm_cfg': {'type': 'GN', 'num_groups': 32},
        'act_cfg': {'type': 'ReLU'},
        'encoder': {
            'type': 'DetrTransformerEncoder',
            'num_layers': 6,
            'transformerlayers': {
                'type': 'BaseTransformerLayer',
                'attn_cfgs': {
                    'type': 'MultiScaleDeformableAttention',
                    'embed_dims': 256,
                    'num_heads': 8,
                    'num_levels': 3,
                    'num_points': 4,
                    'im2col_step': 64,
                    'dropout': 0.0,
                    'batch_first': False,
                    'norm_cfg': None,
                    'init_cfg': None
                },
                'ffn_cfgs': {
                    'type': 'FFN',
                    'embed_dims': 256,
                    'feedforward_channels': 1024,
                    'num_fcs': 2,
                    'ffn_drop': 0.0,
                    'act_cfg': {'type': 'ReLU', 'inplace': True},
                    'with_cp': True
                },
                'operation_order': ('self_attn', 'norm', 'ffn', 'norm')
            },
            'init_cfg': None
        },
        'positional_encoding': {
            'type': 'SinePositionalEncoding',
            'num_feats': 128,
            'normalize': True
        },
        'init_cfg': None
    },
    
    'enforce_decoder_input_project': False,
    
    # Positional encoding for transformer decoder
    'positional_encoding': {
        'type': 'SinePositionalEncoding',
        'num_feats': 128,
        'normalize': True
    },
    
    # Transformer decoder configuration
    'transformer_decoder': {
        'type': 'DetrTransformerDecoder',
        'return_intermediate': True,
        'num_layers': 9,
        'transformerlayers': {
            'type': 'DetrTransformerDecoderLayer',
            'attn_cfgs': {
                'type': 'MultiheadAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'attn_drop': 0.0,
                'proj_drop': 0.0,
                'dropout_layer': None,
                'batch_first': False
            },
            'ffn_cfgs': {
                'embed_dims': 256,
                'feedforward_channels': 1024,
                'num_fcs': 2,
                'act_cfg': {'type': 'ReLU', 'inplace': True},
                'ffn_drop': 0.0,
                'dropout_layer': None,
                'add_identity': True,
                'with_cp': True
            },
            'feedforward_channels': 1024,
            'operation_order': ('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')
        },
        'init_cfg': None
    },
    
    # Loss configuration
    'loss_cls': {
        'type': 'CrossEntropyLoss',
        'use_sigmoid': False,
        'loss_weight': 2.0,
        'reduction': 'mean',
        'class_weight': [1.0, 1.0, 0.1]  # [foreground, background, no_object]
    },
    'loss_mask': {
        'type': 'CrossEntropyLoss',
        'use_sigmoid': True,
        'reduction': 'mean',
        'loss_weight': 5.0
    },
    'loss_dice': {
        'type': 'DiceLoss',
        'use_sigmoid': True,
        'activate': True,
        'reduction': 'mean',
        'naive_dice': True,
        'eps': 1.0,
        'loss_weight': 5.0
    },
    
    # Training configuration
    'train_cfg': {
        'num_points': 12544,
        'oversample_ratio': 3.0,
        'importance_sample_ratio': 0.75,
        'assigner': {
            'type': 'MaskHungarianAssigner',
            'cls_cost': {'type': 'ClassificationCost', 'weight': 2.0},
            'mask_cost': {'type': 'CrossEntropyLossCost', 'weight': 5.0, 'use_sigmoid': True},
            'dice_cost': {'type': 'DiceCost', 'weight': 5.0, 'pred_act': True, 'eps': 1.0}
        },
        'sampler': {'type': 'MaskPseudoSampler'}
    },
    
    # Test configuration
    'test_cfg': {
        'panoptic_on': False,
        'semantic_on': True,
        'instance_on': False,
        'max_per_image': 100,
        'iou_thr': 0.8,
        'filter_low_score': True
    },
    
    'init_cfg': None
}

# Optimizer configuration (matching reference)
optimizer_config = {
    'type': 'AdamW',
    'lr': 1.8e-05,
    'betas': (0.9, 0.999),
    'weight_decay': 0.0032
}

# Learning rate schedule
lr_config = {
    'policy': 'poly',
    'warmup': 'linear',
    'warmup_iters': 1500,
    'warmup_ratio': 1e-06,
    'power': 1.0,
    'min_lr': 0.0,
    'by_epoch': False
}
