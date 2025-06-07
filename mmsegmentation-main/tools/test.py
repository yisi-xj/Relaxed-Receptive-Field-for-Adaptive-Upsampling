# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.registry import DATASETS  # 新增注册器导入


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # 新增 --eval 参数支持 bIoU
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='Evaluation metrics (including bIoU)')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')
    return cfg

def legacy_evaluation(runner, args):
    from tqdm import tqdm
    from boundary_iou import eval_boundary_iou  # 确保 boundary_iou.py 在路径中

    # 第一阶段：数据集初始化
    dataset_cfg = runner.cfg.test_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)  # 显式构建数据集对象
    
    # 动态生成真值路径
    img_dir = osp.join(dataset_cfg.data_root, dataset_cfg.data_prefix['img_path'])
    ann_dir = osp.join(dataset_cfg.data_root, dataset_cfg.data_prefix['seg_map_path'])
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    gt_seg_map_paths = [
    osp.join(ann_dir, f.replace('.jpg', '.png'))  # .jpg → .png
    for f in img_files
    ]
    gt_seg_map_paths =sorted(
    [f for f in gt_seg_map_paths if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # 正确提取数字部分
    )


    # 第三阶段：预测文件匹配
    pred_dir = args.out
    pred_files = sorted(
        [f for f in os.listdir(pred_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # 正确提取数字部分
    )
    
    # 增强数量校验（防止后续越界）
    if len(pred_files) != len(gt_seg_map_paths):
        raise ValueError(
            f"预测文件数({len(pred_files)})与真值数({len(gt_seg_map_paths)})不匹配\n"
            f"预测示例：{pred_files[:3]}\n真值示例：{[osp.basename(p) for p in gt_seg_map_paths[:3]]}"
        )

    # 第四阶段：数据加载与计算
    predictions = []
    ground_truths = []
    for idx in tqdm(range(len(dataset)), desc="加载数据"):
        try:
            # 加载预测
            pred_path = osp.join(pred_dir, pred_files[idx])
            pred_img = Image.open(pred_path)
            pred = np.array(pred_img) - 1
            predictions.append(pred[:, :, 0] if pred.ndim == 3 else pred)
            
            # 加载真值
            gt_path = gt_seg_map_paths[idx]
            gt_img = Image.open(gt_path)
            ground_truths.append(np.array(gt_img) - 1)
            print(pred_path, gt_path)
        except Exception as e:
            raise RuntimeError(f"加载第{idx}个样本失败: {str(e)}")

    # 第五阶段：计算bIoU
    print('\n' + '='*30 + ' Boundary IoU Evaluation ' + '='*30)
    eval_boundary_iou(
            predictions=predictions,
            ground_truths=ground_truths,
            num_classes=len(dataset.metainfo['classes'])
    )

def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 工作目录设置
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 强制配置结果保存路径
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    cfg.load_from = args.checkpoint

    # 可视化配置
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # TTA 配置
    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # 构建 Runner
    runner = Runner.from_cfg(cfg)

    # 执行测试
    runner.test()

    # 安全执行后处理
    if 'bIoU' in args.eval:
        try:
            legacy_evaluation(runner, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError("bIoU 计算流程异常终止") from e

if __name__ == '__main__':
    main()