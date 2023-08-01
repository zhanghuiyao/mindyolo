import argparse
import ast
import os
import time
from functools import partial

import mindspore as ms

from mindyolo.data import COCODataset, create_loader
from mindyolo.models import create_loss, create_model
from mindyolo.optim import (EMA, create_group_param, create_lr_scheduler,
                            create_optimizer, create_warmup_momentum_scheduler)
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.train_step_factory import get_gradreducer, get_loss_scaler, create_train_step_fn_seg
from mindyolo.utils.trainer_factory import create_trainer
from mindyolo.utils.callback import create_callback
from mindyolo.utils.utils import (freeze_layers, load_pretrain, set_default,
                                  set_seed, Synchronizer)

from mindspore import Profiler


def get_parser_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--save_dir", type=str, default="./runs", help="save dir")
    parser.add_argument("--device_per_servers", type=int, default=8, help="device number on a server")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level to print")
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False, help="Distribute train or not")
    parser.add_argument("--ms_mode", type=int, default=0,
                        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2/O3")
    parser.add_argument("--keep_loss_fp32", type=ast.literal_eval, default=True,
                        help="Whether to maintain loss using fp32/O0-level calculation")
    parser.add_argument("--ms_loss_scaler", type=str, default="static", help="train loss scaler, static/dynamic/none")
    parser.add_argument("--ms_loss_scaler_value", type=float, default=1024.0, help="static loss scale value")
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_enable_graph_kernel", type=ast.literal_eval, default=False,
                        help="use enable_graph_kernel or not")
    parser.add_argument("--ms_datasink", type=ast.literal_eval, default=False, help="Train with datasink.")
    parser.add_argument("--overflow_still_update", type=ast.literal_eval, default=True, help="overflow still update")
    parser.add_argument("--ema", type=ast.literal_eval, default=True, help="ema")
    parser.add_argument("--weight", type=str, default="./yolov8x_seg_torch_init.ckpt", help="initial weight path")
    parser.add_argument("--ema_weight", type=str, default="", help="initial ema weight path")
    parser.add_argument("--freeze", type=list, default=[], help="Freeze layers: backbone of yolov7=50, first3=0 1 2")
    parser.add_argument("--epochs", type=int, default=300, help="total train epochs")
    parser.add_argument("--per_batch_size", type=int, default=32, help="per batch size for each device")
    parser.add_argument("--img_size", type=list, default=640, help="train image sizes")
    parser.add_argument("--nbs", type=list, default=64, help="nbs")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="grad accumulate step, recommended when batch-size is less than 64")
    parser.add_argument("--auto_accumulate", type=ast.literal_eval, default=False, help="auto accumulate")
    parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--single_cls", type=ast.literal_eval, default=False,
                        help="train multi-class data as single-class")
    parser.add_argument("--sync_bn", type=ast.literal_eval, default=False,
                        help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--keep_checkpoint_max", type=int, default=100)
    parser.add_argument("--run_eval", type=ast.literal_eval, default=False, help="Whether to run eval during training")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold for run_eval")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS for run_eval")
    parser.add_argument("--conf_free", type=ast.literal_eval, default=False,
                        help="Whether the prediction result include conf")
    parser.add_argument("--rect", type=ast.literal_eval, default=False, help="rectangular training")
    parser.add_argument("--nms_time_limit", type=float, default=20.0, help="time limit for NMS")
    parser.add_argument("--recompute", type=ast.literal_eval, default=False, help="Recompute")
    parser.add_argument("--recompute_layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--summary", type=ast.literal_eval, default=True, help="collect train loss scaler or not")
    parser.add_argument("--profiler", type=ast.literal_eval, default=False, help="collect profiling data or not")
    parser.add_argument("--profiler_step_num", type=int, default=1, help="collect profiler data for how many steps.")
    parser.add_argument("--opencv_threads_num", type=int, default=2, help="set the number of threads for opencv")

    # zhy_test, clip_grad 1
    parser.add_argument("--clip_grad", type=ast.literal_eval, default=False)
    parser.add_argument("--clip_grad_value", type=float, default=10.0)

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file")
    parser.add_argument("--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders")
    parser.add_argument("--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument("--data_dir", type=str, default="/cache/data/",
                        help="ModelArts: local device path to dataset folder")
    parser.add_argument("--ckpt_dir", type=str, default="/cache/pretrain_ckpt/",
                        help="ModelArts: local device path to checkpoint folder")

    return parser


def train(args):
    # Set Default
    set_seed(args.seed)
    set_default(args)
    main_device = args.rank % args.rank_size == 0

    if args.profiler:
        ms.context.set_context(save_graphs=True, save_graphs_path="./irs")
        profiler = Profiler()

    logger.info(f"parse_args:\n{args}")
    logger.info("Please check the above information for the configurations")

    # Create Network
    args.network.recompute = args.recompute
    args.network.recompute_layers = args.recompute_layers
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=args.sync_bn,
    )

    if args.ema:
        ema_network = create_model(
            model_name=args.network.model_name,
            model_cfg=args.network,
            num_classes=args.data.nc,
        )
        ema = EMA(network, ema_network)
    else:
        ema = None
    load_pretrain(network, args.weight, ema, args.ema_weight)  # load pretrain
    freeze_layers(network, args.freeze)  # freeze Layers
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
    if ema:
        ms.amp.auto_mixed_precision(ema.ema, amp_level=args.ms_amp_level)

    # Create Loss
    loss_fn = create_loss(
        **args.loss, anchors=args.network.get("anchors", 1), stride=args.network.stride, nc=args.data.nc
    )
    ms.amp.auto_mixed_precision(loss_fn, amp_level="O0" if args.keep_loss_fp32 else args.ms_amp_level)

    # Create Optimizer
    # zhy_test, optimizer 1
    # from mindspore import nn
    # optimizer = nn.SGD(network.trainable_params(), learning_rate=0.01)
    args.optimizer.steps_per_epoch = 313
    lr = create_lr_scheduler(**args.optimizer)
    params = create_group_param(params=network.trainable_params(), **args.optimizer)
    optimizer = create_optimizer(params=params, lr=lr, **args.optimizer)
    warmup_momentum = create_warmup_momentum_scheduler(**args.optimizer)

    # Create train_step_fn
    reducer = get_gradreducer(args.is_parallel, optimizer.parameters)
    scaler = get_loss_scaler(args.ms_loss_scaler, scale_value=args.ms_loss_scaler_value)
    train_step_fn = create_train_step_fn_seg(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_ratio=args.rank_size,
        scaler=scaler,
        reducer=reducer,
        ema=ema,
        overflow_still_update=args.overflow_still_update,
        ms_jit=args.ms_jit,
        clip_grad=args.clip_grad,  # zhy_test, clip_grad 2
        clip_grad_value=args.clip_grad_value
    )

    # trainer
    network.set_train(True)
    optimizer.set_train(True)

    # load data
    import numpy as np
    img = np.load("./npy_files/batch_img.npy")[:8]
    gt_mask = np.load("./npy_files/batch_mask.npy")[:8]
    _idx = np.load("./npy_files/batch_idx.npy")[:8]
    _cls = np.load("./npy_files/batch_cls.npy")[:8]
    _box = np.load("./npy_files/batch_bboxes.npy")[:8]
    gt_boxes = np.ones((8, 160, 6), dtype=np.float32) * -1
    for i in range(8):
        _select = (_idx == i)
        _gt = np.concatenate((_idx[_select][:, None], _cls[_select], _box[_select]), axis=1)
        _len = _gt.shape[0]
        gt_boxes[i, :_len] = _gt[:_len]
    # to Tensor
    from mindspore import Tensor
    img = Tensor(img, ms.float32)
    gt_boxes = Tensor(gt_boxes, ms.float32)
    gt_mask = Tensor(gt_mask, ms.float32)

    # train v8-seg
    s_time = time.time()
    for i in range(100):
        # zhy_test, optimizer 2
        _dtype = optimizer.momentum.dtype
        optimizer.momentum = Tensor(warmup_momentum[i], _dtype)
        loss, loss_items, unscaled_grads, grads_finite = train_step_fn(img, gt_boxes, gt_mask)
        print(f"Step: {i}, loss: {loss}, loss_items: {loss_items}, cost time: {time.time()-s_time:.2f} s")
        # import pdb;pdb.set_trace()
        s_time = time.time()

    if args.profiler:
        profiler.analyse()

if __name__ == "__main__":
    parser = get_parser_train()
    args = parse_args(parser)
    train(args)
