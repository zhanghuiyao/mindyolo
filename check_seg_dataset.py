import argparse
import ast
import numpy as np
import cv2

from mindyolo.utils.config import parse_args
from mindyolo.data.dataset_seg import COCODatasetSeg


def get_parser_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="CPU", help="device target, Ascend/GPU/CPU")
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
    parser.add_argument("--weight", type=str, default="", help="initial weight path")
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


def show_img_with_poly(data):
    """
    Image and polygons visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images

    Returns: an image with polygons
    """
    img, labels, polys = data[0], data[1], data[2]
    h, w = img.shape[:2]
    labels = labels[labels[:, 1] >= 0]  # filter invalid label
    category_ids = labels[:, 1]

    real_polys = polys[labels[:, 1] >= 0]
    for poly in real_polys:
        poly = poly.astype(np.int32)
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()
        img = cv2.drawContours(img, [poly], -1, color, 2)
    return img

def show_img_with_bbox(data, classes):
    """
    Image and bboxes visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images
        classes: all categories of the whole dataset

    Returns: an image with detection boxes and categories
    """
    img, labels = data[0], data[1]
    h, w = img.shape[:2]

    labels = labels[labels[:, 0] > 0]  # filter invalid label
    category_ids = labels[:, 0]
    bboxes = labels[:, 1:]
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    bboxes[:, [0, 1]] -= bboxes[:, [2, 3]] / 2
    bboxes[:, [2, 3]] = bboxes[:, [2, 3]] + bboxes[:, [0, 1]]

    categories = [classes[int(category_id)] for category_id in category_ids]
    bboxes = bboxes[category_ids >= 0]
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        categories_size = cv2.getTextSize(category + "0", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()

        if bbox[1] - categories_size[1] - 3 < 0:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] + 2),
                (bbox[0] + categories_size[0], bbox[1] + categories_size[1] + 3),
                color=color,
                thickness=-1,
            )
            cv2.putText(
                img,
                category,
                (bbox[0], bbox[1] + categories_size[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                thickness=1,
            )
        else:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] - categories_size[1] - 3),
                (bbox[0] + categories_size[0], bbox[1] - 3),
                color,
                thickness=-1,
            )
            cv2.putText(img, category, (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)
    return img


if __name__ == '__main__':
    parser = get_parser_train()
    args = parse_args(parser)
    transforms = args.data.train_transforms
    stage_transforms = transforms['trans_list']
    dataset = COCODatasetSeg(
        dataset_path=args.data.train_set,
        img_size=args.img_size,
        transforms_dict=stage_transforms[0],
        is_training=True,
        augment=True,
        rect=args.rect,
        single_cls=args.single_cls,
        batch_size=4,
        stride=max(args.network.stride),
    )
    data = dataset[1]
    classes = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush' ]
    # image = show_img_with_bbox(data, classes)
    image = show_img_with_poly(data)
    show_window = False
    if show_window:
        cv2.namedWindow("image")  # 创建一个image的窗口
        cv2.imshow("image", image)  # 显示图像
        cv2.waitKey()  # 默认为0，无限等待
    else:
        cv2.imwrite("1.jpg", image)
    print('done')