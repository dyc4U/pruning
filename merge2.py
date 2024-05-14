# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from utils.distill_loss import LogicalLoss, FeatureLoss

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model, Detect, RepConv, Faster_Block, C3_Faster, IFM, TopBasicLayer, SimFusion_3in, \
    SimFusion_4in, InjectionMultiSum_Auto_pool, AdvPoolFusion, PyramidPoolAgg, Detect_DyHead_Prune, \
    replace_c2f_with_c2f_v2
from models.convnextv2 import LayerNorm
from models.EfficientFormerv2 import Attention4D, Attention4DDownsample
from models.dyhead_prune import DyHeadBlock_Prune
# from timm.models.layers import SqueezeExcite
from models.fasternet import MLPBlock
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size,  check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

import val

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

import copy
import torch_pruning as tp
from functools import partial
from thop import clever_format
import matplotlib.pylab as plt


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MyMagnitudeImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = []
        # 2. 迭代分组内的各个层，对Conv层计算重要性
        for dep, idxs in group: # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.target.module # 获取 nn.Module
            prune_fn = dep.handler    # 获取 剪枝函数
            # 3. 这里我们简化问题，仅计算卷积输出通道的重要性
            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                w = layer.weight.data[idxs].flatten(1) # 用索引列表获取耦合通道对应的参数，并展开成2维
                local_norm = w.abs().sum(1) # 计算每个通道参数子矩阵的 L1 Norm
                group_imp.append(local_norm) # 将其保存在列表中

        if len(group_imp)==0: return None # 跳过不包含卷积层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp


def get_pruner(opt, model, example_inputs):
    sparsity_learning = False
    if opt.prune_method == "random":
        imp = tp.importance.RandomImportance()

        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "l1":
        # https://arxiv.org/abs/1608.08710
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "lamp":
        # https://arxiv.org/abs/2010.07611
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "slim":
        # https://arxiv.org/abs/1708.06519
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_slim":
        # https://tibshirani.su.domains/ftp/sparse-grlasso.pdf
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning,
                               group_lasso=True)
    elif opt.prune_method == "group_norm":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_sl":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "growing_reg":
        # https://arxiv.org/abs/2012.09243
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=opt.reg, delta_reg=opt.delta_reg,
                               global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_hessian":
        # https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
        imp = tp.importance.HessianImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_taylor":
        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
        imp = tp.importance.TaylorImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    else:
        raise NotImplementedError

    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    customized_pruners = {}
    round_to = None

    """
    设置重要性依据
    """
    imp = MyMagnitudeImportance()
    """
    不剪枝的层设置
    """
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)
    print(ignored_layers)

    """
    初始化剪枝器，参数为模型；torch_pruning依赖图输入；剪枝迭代次数；剪枝程度；剪枝程度字典；最大稀疏率；不剪枝的层
    """
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=0.25,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=opt.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        customized_pruners=customized_pruners,
        round_to=round_to,
        root_module_types=[nn.Conv2d, nn.Linear]
    )
    """
    iterative_steps = 2  # 迭代式剪枝，重复5次Pruning-Finetuning的循环完成剪枝。
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,  # 用于分析依赖的伪输入
        importance=imp,  # 重要性评估指标a
        iterative_steps=iterative_steps,  # 迭代剪枝，设为1则一次性完成剪枝
        ch_sparsity=0.5,  # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,  # 忽略层
    )
    """
    return sparsity_learning, imp, pruner


linear_trans = lambda epoch, epochs, reg, reg_ratio: (1 - epoch / (epochs - 1)) * (reg - reg_ratio) + reg_ratio


def model_prune(opt, model, imp, prune, example_inputs, val_loader, imgsz_test, prune_loader, train_loader):
    # Hyperparameters
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('distill finetune hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # save hyp and opt
    # yaml_save(save_dir / 'hyp.yaml', hyp)
    # yaml_save(save_dir / 'opt.yaml', vars(opt))

    prune_ans = {}  # 存储每次裁剪的附属蒸馏中最好的结果
    distill_ans = {} #存储所有训练的ap

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None

    base_model = copy.deepcopy(model)
    with HiddenPrints():#防止打印下面变量
        ori_flops, ori_params = tp.utils.count_ops_and_params(base_model, example_inputs)
    ori_flops = ori_flops * 2.0
    ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
    ori_result, _, _ = val.run(data_dict, None, batch_size=opt.batch_size * 2,
                               imgsz=imgsz_test, plots=False, model=base_model, dataloader=val_loader)
    ori_map50, ori_map = ori_result[2], ori_result[3]
    iter_idx, prune_flops = 0, ori_flops
    speed_up = 1.0
    LOGGER.info('begin pruning...')
    for i in range(200):
    #while speed_up < opt.speed_up:
        """
        if len(prune_ans) > 0:
            model = torch.load(best, map_location=device)
            if model['ema']:
                model = model['ema'].float()
            else:
                model = model['model'].float()
            for p in model.parameters():
                p.requires_grad_(True)
            model.info(img_size=opt.imgsz)
            sparsity_learning, imp, prune = get_pruner(opt, model, example_inputs)
        """

        """
        正常剪枝
        """
        iter_idx += 1
        # 剪枝运行
        for group in prune.step(interactive=True):
            print(group.details)
            group.prune()
        # 评估剪枝结果
        prune_result, _, _ = val.run(data_dict, None, batch_size=opt.batch_size * 2,
                                     imgsz=imgsz_test, plots=False, model=copy.deepcopy(model), dataloader=val_loader)
        # 剪枝的mAP结果保存
        prune_map50, prune_map = prune_result[2], prune_result[3]
        with HiddenPrints():
            prune_flops, prune_params = tp.utils.count_ops_and_params(model, example_inputs)
        prune_flops = prune_flops * 2.0
        prune_flops_f, prune_params_f = clever_format([prune_flops, prune_params], "%.3f")
        speed_up = ori_flops / prune_flops
        LOGGER.info(
            f'pruning... iter:{iter_idx} ori model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%) map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f}) map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f}) Speed Up:{ori_flops / prune_flops:.2f}')
        LOGGER.info(
            f'start distill for---prune {iter_idx}---'
            f'******************************'
        )
        """
        finetune，每次调用12个epoch训练
        """
        # finetune,前四个为训练参数，配置参数，设备，回调函数，后四个为学生模型，训练集，测试集，剪枝对应蒸馏结果保存
        #res, res_epochs = finetune(hyp, opt, device, callbacks, model, train_loader, val_loader)

        #prune_ans[iter_idx] = res
        #distill_ans[iter_idx] = res_epochs

        # 最多迭代这么多次
        if prune.current_step == prune.iterative_steps:
            break

    if isinstance(imp, tp.importance.HessianImportance):
        imp.zero_grad()
    model.zero_grad()
    torch.cuda.empty_cache()

    LOGGER.info('work done...')
    LOGGER.info(
        f'model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) Speed Up:{ori_flops / prune_flops:.2f}')
    LOGGER.info(f'model params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%)')
    LOGGER.info(f'model map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f})')
    LOGGER.info(f'model map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f})')
    return prune_ans, distill_ans


"""
输入为当前剪枝迭代次数的模型
"""


def finetune(hyp, opt, device, callbacks, model, train_loader, val_loader):  # hyp is path/to/hyp.yaml or hyp dictionary
    epochs, weights, batch_size, single_cls, data, cfg, resume, noval, nosave, workers, freeze = \
        opt.epochs, opt.weights, opt.batch_size, opt.single_cls, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')
    res_label = ['P', 'R', 'mAP@.5', 'mAP@.5-.95', 'val_loss_box', 'val_loss_obj', 'val_loss_cls']

    # Hyperparameters
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model For Student
    LOGGER.info(f'Loaded {weights}')  # report
    amp = check_amp(model)  # check AMP

    # Model For Teacher
    ckpt = torch.load(opt.teacher_weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    t_model = Model(opt.teacher_cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, t_model.state_dict(), exclude=exclude)  # intersect
    t_model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(
        f'Teacher Transferred {len(csd)}/{len(t_model.state_dict())} items from {opt.teacher_weights}')  # report

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    # Init Distill Loss
    kd_logical_loss, kd_feature_loss = None, None
    if opt.kd_loss_type == 'logical' or opt.kd_loss_type == 'all':
        kd_logical_loss = LogicalLoss(hyp, model, opt.logical_loss_type)
    if opt.kd_loss_type == 'feature' or opt.kd_loss_type == 'all':
        s_feature, t_feature = [], []

        def get_activation(feat, backbone_idx=-1):
            def hook(model, inputs, outputs):
                if backbone_idx != -1:
                    for idx, i in enumerate(outputs): print(idx, i.size())
                    feat.append(outputs[backbone_idx])
                else:
                    feat.append(outputs)

            return hook

        hooks = []
        teacher_kd_layers, student_kd_layers = opt.teacher_kd_layers.split(','), opt.student_kd_layers.split(',')
        for t_layer, s_layer in zip(teacher_kd_layers, student_kd_layers):
            if '-' in t_layer:
                t_layer_first, t_layer_second = t_layer.split('-')
                hooks.append(de_parallel(t_model).model[int(t_layer_first)].register_forward_hook(
                    get_activation(t_feature, backbone_idx=int(t_layer_second))))
            else:
                hooks.append(de_parallel(t_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))

            if '-' in s_layer:
                s_layer_first, s_layer_second = s_layer.split('-')
                hooks.append(de_parallel(model).model[int(s_layer_first)].register_forward_hook(
                    get_activation(s_feature, backbone_idx=int(s_layer_second))))
            else:
                hooks.append(de_parallel(model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
        inputs = torch.randn((2, 3, opt.imgsz, opt.imgsz)).to(device)
        with torch.no_grad():
            _ = t_model(inputs)
            _ = model(inputs)
        kd_feature_loss = FeatureLoss([i.size(1) for i in s_feature], [i.size(1) for i in t_feature],
                                      distiller=opt.feature_loss_type)
        for hook in hooks:
            hook.remove()

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'],
                                kd_feature_loss if isinstance(kd_feature_loss, FeatureLoss) else None)

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    compute_loss_ota = ComputeLossOTA(model)  # init ota loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    res_epochs = {}
    for epoch in range(0, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        if opt.kd_loss_type in ['feature', 'all']:
            kd_feature_loss.train()
            hooks = []
            s_feature, t_feature = [], []
            for t_layer, s_layer in zip(teacher_kd_layers, student_kd_layers):
                if '-' in t_layer:
                    t_layer_first, t_layer_second = t_layer.split('-')
                    hooks.append(de_parallel(t_model).model[int(t_layer_first)].register_forward_hook(
                        get_activation(t_feature, backbone_idx=int(t_layer_second))))
                else:
                    hooks.append(
                        de_parallel(t_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))

                if '-' in s_layer:
                    s_layer_first, s_layer_second = s_layer.split('-')
                    hooks.append(de_parallel(model).model[int(s_layer_first)].register_forward_hook(
                        get_activation(s_feature, backbone_idx=int(s_layer_second))))
                else:
                    hooks.append(
                        de_parallel(model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss, logical_disloss, feature_disloss = torch.zeros(3, device=device), torch.zeros(1,
                                                                                             device=device), torch.zeros(
            1, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%15s' * 9) % (
            'Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'log_loss', 'fea_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            if opt.kd_loss_decay == 'constant':
                distill_decay = 1.0
            elif opt.kd_loss_decay == 'cosine':
                eta_min, base_ratio, T_max = 0.01, 1.0, 10
                distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
            elif opt.kd_loss_decay == 'linear':
                distill_decay = ((1 - math.cos(i * math.pi / len(train_loader))) / 2) * (0.01 - 1) + 1
            elif opt.kd_loss_decay == 'cosine_epoch':
                eta_min, base_ratio, T_max = 0.01, 1.0, 10
                distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * ni / T_max)) / 2
            elif opt.kd_loss_decay == 'linear_epoch':
                distill_decay = ((1 - math.cos(ni * math.pi / (epochs * nb))) / 2) * (0.01 - 1) + 1

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)

                with torch.no_grad():
                    t_pred = t_model(imgs)

                # Loss For Student Model
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                log_distill_loss, fea_distill_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
                if kd_logical_loss is not None:
                    log_distill_loss = kd_logical_loss(pred, t_pred) * opt.logical_loss_ratio
                if kd_feature_loss is not None:
                    fea_distill_loss = kd_feature_loss(s_feature, t_feature) * opt.feature_loss_ratio

                loss += (log_distill_loss + fea_distill_loss) * imgs.size(0) * distill_decay
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                logical_disloss = (logical_disloss * i + log_distill_loss.detach()) / (i + 1)  # update mean losses
                feature_disloss = (feature_disloss * i + fea_distill_loss.detach()) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%15s' * 2 + '%15.4g' * 7) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, logical_disloss, feature_disloss,
                                      targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            if kd_feature_loss is not None:
                s_feature.clear()
                t_feature.clear()
            # end batch ------------------------------------------------------------------------------------------------

        if kd_feature_loss is not None:
            for hook in hooks:
                hook.remove()

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            #这里额外保存下每轮的ap，验证12论训练是不是收敛了
            #ap50, map = results[2], results[3]
            #res_label = ['P', 'R', 'mAP@.5', 'mAP@.5-.95', 'val_loss_box', 'val_loss_obj', 'val_loss_cls']
            ap = {}
            i_key = 0
            for key in res_label:
                if key not in ap:
                    ap[key] = results[i_key]
                    i_key = i_key + 1
            if epoch not in ap:
                ap[epoch] = epoch
            #AP = {'epoch': epoch, 'ap50': ap50, 'map': map}
            if epoch not in res_epochs:
                res_epochs[epoch] = ap#保存12论的AP

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    res = {}
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    j = 0
                    for i in res_label:
                        res[i] = results[j]
                        j = j + 1

                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()



    return res, res_epochs


def parse_opt(known=False):
    # 代码来自:BiliBli 魔鬼面具up主
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs_prune/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    # prune
    parser.add_argument('--prune_method', type=str, default=None, help='prune method')
    parser.add_argument('--speed_up', type=float, default=2.0, help='speed up')
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--max_sparsity", type=float, default=1.0)
    parser.add_argument("--iterative_steps", type=int, default=200)
    parser.add_argument('--prune_model', action='store_true', help='is prune model?')

    parser.add_argument("--reg_decay_type", type=str, default='linear', choices=['constant', 'linear', 'step'],
                        help='reg decay type choice in sparsity learning')
    parser.add_argument("--reg_decay", type=float, default=0.01)
    parser.add_argument("--reg_decay_step", type=int, default=10,
                        help='reg decay step size in sparsity learning and reg_decay_type==step')

    # knowledge distillation arguments
    parser.add_argument('--teacher_weights', type=str, default='yolov5n.pt', help='initial weights path (teacher)')
    parser.add_argument('--teacher_cfg', type=str, default='yolov5n.yaml', help='initial model.yaml path (teacher)')
    parser.add_argument('--kd_loss_type', type=str, default='feature', choices=['logical', 'feature', 'all'],
                        help='kd loss type')
    parser.add_argument('--kd_loss_decay', type=str, default='constant',
                        choices=['cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'], help='kd loss decay')

    # logical distillation arguments
    parser.add_argument('--logical_loss_type', type=str, default='l2', choices=['l2', 'l1', 'AlignSoftTarget'],
                        help='logical loss type in kd_loss')
    parser.add_argument('--logical_loss_ratio', type=float, default=1.0, help='logical loss ratio')

    # feature distillation arguments
    parser.add_argument('--teacher_kd_layers', type=str, default='17,20,23',
                        help='Teahcer Layer for Feature knowledge distillation')
    parser.add_argument('--student_kd_layers', type=str, default='17,20,23',
                        help='Student Layer for Feature knowledge distillation')
    parser.add_argument('--feature_loss_type', type=str, default='cwd', choices=['mimic', 'cwd', 'mgd'],
                        help='feature loss type in kd_loss')
    parser.add_argument('--feature_loss_ratio', type=float, default=1.0, help='feature loss ratio')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    # Init Callback
    callbacks = Callbacks()

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Config
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = torch.load(weights, map_location=device)
        if model['ema']:
            model = model['ema'].float()
        else:
            model = model['model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        model.info(img_size=opt.imgsz)

        # for c2f
        replace_c2f_with_c2f_v2(model.model)
        model.to(device)
        LOGGER.info(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    batch_size, single_cls, workers, data, noval = opt.batch_size, opt.single_cls, opt.workers, opt.data, opt.noval
    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        model.half().float()
        callbacks.run('on_pretrain_routine_end', labels, names)

    # prune dataloader
    prune_loader, _ = create_dataloader(train_path,
                                        imgsz,
                                        opt.batch_size // WORLD_SIZE // 2,
                                        gs,
                                        opt.single_cls,
                                        hyp=hyp,
                                        augment=True,
                                        cache=None,
                                        rect=opt.rect,
                                        rank=LOCAL_RANK,
                                        workers=opt.workers,
                                        image_weights=opt.image_weights,
                                        quad=opt.quad,
                                        prefix=colorstr('prune: '),
                                        shuffle=True,
                                        seed=opt.seed)

    # Resume
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str('runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str('runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    save_dir = Path(opt.save_dir)
    evolve = opt.evolve
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'yolov5n_Lamp_CWD.pt'

    # get prune
    example_inputs = torch.randn((1, 3, imgsz, imgsz)).to(device)
    sparsity_learning, imp, prune = get_pruner(opt, model, example_inputs)

    # pruning
    prune_ans, distill_ans = model_prune(opt, model, imp, prune, example_inputs, val_loader, imgsz, prune_loader, train_loader)
    import pickle


    z = save_dir / 'ans_prune.pkl'
    z2 = save_dir / 'ans_distil.pkl'
    with open(z, mode='wb') as f:
        pickle.dump(prune_ans, f)

    with open(z2, mode='wb') as f:
        pickle.dump(distill_ans, f)


    #import cal
    #cal.plot(save_dir, prune_ans)
    # test fuse
    fuse_model = deepcopy(model)
    for p in fuse_model.parameters():
        p.requires_grad_(False)
    fuse_model.fuse()
    del fuse_model
