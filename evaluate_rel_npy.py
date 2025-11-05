import argparse
import logging
import os
import pprint
import json
import numpy as np
import torch
from tqdm import tqdm
import cv2

# python evaluate_rel_npy.py  --infer_path /data/vda/pred_npz_split --benchmark_path /data/c3vd_vda --datasets c3vd

# python evaluate_rel_npy.py  --infer_path /data/vda/pred_npz_split --benchmark_path /data/c3vd_vda --datasets c3vd --save-path /data/vda

import logging

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

parser = argparse.ArgumentParser(description='Evaluate Depth Predictions from NPY files')
parser.add_argument('--infer_path', type=str, required=True, help='Path to prediction npy files')
parser.add_argument('--benchmark_path', type=str, required=True, help='Path to benchmark data')
parser.add_argument('--datasets', type=str, nargs='+', default=['kitti', 'sintel', 'nyu_v2', 'bonn', 'scannet'])
parser.add_argument('--save-path', type=str, required=True, help='Path to save results')

def get_gt(depth_gt_path, gt_factor):
    if depth_gt_path.split('.')[-1] == 'npy':
        depth_gt = np.load(depth_gt_path)
    else:
        depth_gt = cv2.imread(depth_gt_path, -1)
        depth_gt = np.array(depth_gt)
    depth_gt = depth_gt / 65535
    depth_gt = (depth_gt - depth_gt.min()) / (
        depth_gt.max() - depth_gt.min() + 1e-8
    )  # 防止除以零

    eps = 1e-8
    depth_gt = np.clip(depth_gt, a_min=eps, a_max=None)
    depth_gt[depth_gt==0] = -1
    return depth_gt


def get_dataset_params(dataset, benchmark_path):
    params = {}
    params.update({
        'json_file': os.path.join(benchmark_path, "c3vd/c3vd_video.json"),
        'root_path': os.path.join(benchmark_path, "c3vd"),
        'max_depth': 1,
        'min_depth': 0.00001,
        'max_eval_len': 1000,
        'crop': (0, 1350, 0, 1080)
    })
    return params

def main():
    args = parser.parse_args()
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    results_save_path = os.path.join(args.save_path, 'results.txt')
    os.makedirs(args.save_path, exist_ok=True)
    
    for dataset in args.datasets:
        params = get_dataset_params(dataset, args.benchmark_path)
        
        with open(params['json_file'], 'r') as fs:
            path_json = json.load(fs)
        
        json_data = path_json[dataset]
        results_all = []
        
        print(f'Evaluating {dataset}...')
        for data in tqdm(json_data):
            for key in data.keys():
                value = data[key]
                
                for images in value[:params['max_eval_len']]:
                    # 加载预测深度
                    pred_path = (os.path.join(args.infer_path, dataset, images['image'])
                               .replace('.jpg', '.npy')
                               .replace('.png', '.npy'))
                    if not os.path.exists(pred_path):
                        continue
                        
                    pred = np.load(pred_path)
                    gt = get_gt(os.path.join(params['root_path'], images['gt_depth']), 
                              images['factor'])
                    
                    # 裁剪到指定区域
                    a, b, c, d = params['crop']
                    pred = pred[a:b, c:d]
                    gt = gt[a:b, c:d]
                    
                    # 创建有效mask
                    valid_mask = (gt > params['min_depth']) & (gt < params['max_depth'])
                    
                    # 转换为相对深度
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    
                    # 转换为tensor
                    pred = torch.from_numpy(pred).float().cuda()
                    gt = torch.from_numpy(gt).float().cuda()
                    valid_mask = torch.from_numpy(valid_mask).cuda()

                    results = eval_depth(pred[valid_mask], gt[valid_mask])
                    results_all.append(list(results.values()))
        
        # 计算平均结果
        if results_all:
            mean_results = np.mean(np.array(results_all), axis=0)
            
            # 保存结果
            with open(results_save_path, 'a') as f:
                f.write(f'\n=== Results for {dataset} ===\n')
                metrics = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
                for metric, value in zip(metrics, mean_results):
                    f.write(f'{metric}: {value:.4f}\n')
                    logger.info(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()
