# YOLOv10

## Abstract
Over the past years, YOLOs have emerged as the predominant paradigm in the field
of real-time object detection owing to their effective balance between computational 
cost and detection performance. Researchers have explored the architectural
designs, optimization objectives, data augmentation strategies, and others for 
YOLOs, achieving notable progress. However, the reliance on the non-maximum
suppression (NMS) for post-processing hampers the end-to-end deployment of
YOLOs and adversely impacts the inference latency. Besides, the design of various
components in YOLOs lacks the comprehensive and thorough inspection, resulting
in noticeable computational redundancy and limiting the model’s capability. It 
renders the suboptimal efficiency, along with considerable potential for performance
improvements. In this work, we aim to further advance the performance-efficiency
boundary of YOLOs from both the post-processing and the model architecture. To
this end, we first present the consistent dual assignments for NMS-free training
of YOLOs, which brings the competitive performance and low inference latency
simultaneously. Moreover, we introduce the holistic efficiency-accuracy driven
model design strategy for YOLOs. We comprehensively optimize various components 
of YOLOs from both the efficiency and accuracy perspectives, which greatly
reduces the computational overhead and enhances the capability. The outcome
of our effort is a new generation of YOLO series for real-time end-to-end object
detection, dubbed YOLOv10. Extensive experiments show that YOLOv10 achieves
the state-of-the-art performance and efficiency across various model scales. For
example, our YOLOv10-S is 1.8× faster than RT-DETR-R18 under the similar AP on COCO, 
meanwhile enjoying 2.8× smaller number of parameters andFLOPs. Compared with 
YOLOv9-C, YOLOv10-B has 46% less latency and 25% fewer parameters for the same performance.

![yolov10](https://github.com/user-attachments/assets/0241fe17-ccbf-42db-9e0a-f9e2e8c79a1b)


## Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
|   2.5.0   |    24.1.0     | 7.5.0.3.220  |     8.0.0.beta1     |

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>
<summary><b>View More</b></summary>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov10_log python train.py --config ./configs/yolov10/yolov10n.yaml --device_target Ascend --is_parallel True
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov19/yolov10n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov10/yolov10n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt --exec_nms False
```

## Performance


### Detection


Experiments are tested on Ascend 910* with mindspore 2.5.0 graph mode.

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | ms/step | img/s  |  map  |          recipe              |                                                       weight                                                       |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |  :---:  |  :---: |:-----:|          :---:               |:------------------------------------------------------------------------------------------------------------------:|
|    YOLOv10    |    N   |    8   |     32     |  640x640   |     O2      |    301.44s    | 262.36  | 975.76 | 38.3% |    [yaml](./yolov10n.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10n_500e_mAP383-c973023d.ckpt) |
|    YOLOv10    |    S   |    8   |     32     |  640x640   |     O2      |    388.40s    | 343.66  | 744.92 | 45.7% |    [yaml](./yolov10s.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10s_500e_mAP457-8660fa84.ckpt) |
|    YOLOv10    |    M   |    8   |     32     |  640x640   |     O2      |    558.43s    | 495.95  | 516.18 | 50.7% |    [yaml](./yolov10m.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10m_500e_mAP507-1cc8c5fb.ckpt) |
|    YOLOv10    |    B   |    8   |     32     |  640x640   |     O2      |    468.60s    | 598.43  | 427.79 | 52.0% |    [yaml](./yolov10b.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10b_500e_mAP520-0b560f87.ckpt) |
|    YOLOv10    |    L   |    8   |     32     |  640x640   |     O2      |    607.48s    | 687.72  | 372.24 | 52.6% |    [yaml](./yolov10l.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10l_500e_mAP526-226baf5f.ckpt) |
|    YOLOv10    |    X   |    8   |     20     |  640x640   |     O2      |    834.19s    | 604.49  | 264.69 | 53.7% |    [yaml](./yolov10x.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10x_500e_mAP537-aaaa57bb.ckpt) |




### Notes

- map: Accuracy reported on the validation set.
- We refer to the official [YOLOV10](https://github.com/THU-MIG/yolov10) to reproduce the P5 series model.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang. YOLOv10: Real-Time End-to-End Object Detection.
arXiv preprint arXiv:2405.14458
