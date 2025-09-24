# MindYOLO 套件

<p align="left">
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/README.md">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindcv.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
</p>

MindYOLO 是一个基于[MindSpore](https://www.mindspore.cn/)的YOLO系列的目标检测套件。

版本配套关系如下：

| mindyolo |  mindspore  |
| :------: | :---------: |
|   0.5    |    2.5.0    |

## 1. 模型支持列表

- [x] [YOLOv11](configs/yolov11)
- [x] [YOLOv10](configs/yolov10)
- [x] [YOLOv9](configs/yolov9)
- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOX](configs/yolox)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOv3](configs/yolov3)

## 2. 安装

### 2.1 使用pip安装第三方依赖包

- mindspore == 2.5.0
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (for distributed mode)

可以运行以下命令安装python三方包：

```shell
pip install -r requirements.txt
```

⚠️ 注意：当前版本仅支持昇腾硬件，暂时不支持GPU。

### 2.2 使用pip安装MindYOLO包

```shell
pip install mindyolo
```

更多细节请查看 [INSTALLATION](docs/en/installation.md)

## 3. 快速开始

### 3.1 使用预训练模型进行推理

- 第一步，从[模型仓库](benchmark_results.md)列表中选择一个模型及其配置文件，例如， `./configs/yolov7/yolov7.yaml`.
- 第二步，从[模型仓库](benchmark_results.md)列表中下载相应的预训练模型权重文件。
- 第三步，运行：

    ```shell
    # NPU (默认)
    python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
    ```

*结果将保存在`./detect_results`目录下*

*有关命令行参数的详细信息，请参阅`demo/predict.py -h`，或查看其[源代码](https://github.com/mindspore-lab/mindyolo/blob/master/deploy/predict.py)。*

### 3.2 训练和评估

#### 3.2.1 数据集准备

* 按照**YOLO格式**准备您的数据集。如果在**COCO2017数据集**进行训练，请从[yolov5](https://github.com/ultralytics/yolov5)或darknet准备数据集.

    ```text
    coco/
        train2017.txt
        val2017.txt
        annotations/
        instances_train2017.json
        instances_val2017.json
        images/
            train2017/
                00000001.jpg  # image files that are mentioned in the corresponding train/val2017.txt
                ...
            val2017/
                ...
        labels/
            train2017/
                00000001.txt  # label files that are mentioned in the corresponding train/val2017.txt
                ...
            val2017/
                ...
    ```

### 3.2.2 训练

- 启动训练（单卡）：

  ```shell
  python train.py --config ./configs/yolov7/yolov7.yaml 
  ```

- 多卡分布式训练，以8卡为例:

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
  ```

*注意：默认超参是用于coco2017数据集8卡训练的，单卡或不同数据集的情况需按自己的需要进行调整。*

### 3.2.3 评估

- 评估模型的精度（单卡）：

  ```shell
  python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt
  ```

- 多卡分布式评估模型的精度：

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt --is_parallel True
  ```

### 3.2.4 部署

请在 [MindYOLO部署与推理](./deploy/README.md) 查看.

## 4. 使用自定义数据集训练

在**SHWD**数据集(安全帽检测)上使用MindYOLO进行finetune。

### 4.1 版本配套信息

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
|   2.5.0   |    24.1.0     | 7.5.0.3.220  |     8.0.0.beta1     |

### 4.2 数据集格式转换

#### 4.2.1 数据集格式说明

[SHWD数据集](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/tree/master)采用**VOC格式**的数据标注，其文件目录如下所示：

```txt
ROOT_DIR
├── Annotations
│        ├── 000000.xml
│        └── 000002.xml
├── ImageSets
│       └── Main
│             ├── test.txt
│             ├── train.txt
│             ├── trainval.txt
│             └── val.txt
└── JPEGImages
        ├── 000000.jpg
        └── 000002.jpg
```

`Annotations`文件夹下的`xml`文件为每张图片的标注信息，主要内容如下：

```txt
<annotation>
  <folder>JPEGImages</folder>
  <filename>000377.jpg</filename>
  <path>F:\baidu\VOC2028\JPEGImages\000377.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>750</width>
    <height>558</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>hat</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>142</xmin>
      <ymin>388</ymin>
      <xmax>177</xmax>
      <ymax>426</ymax>
    </bndbox>
  </object>
```

其中包含多个object, object中的name为类别名称，xmin, ymin, xmax, ymax则为检测框左上角和右下角的坐标。

MindYOLO支持的数据集格式为**YOLO格式**，详情请参考[yolov5官方仓库](https://github.com/ultralytics/yolov5)或darknet准备数据集，示例如下：

```text
coco/
    train2017.txt
    val2017.txt
    annotations/
    instances_train2017.json
    instances_val2017.json
    images/
        train2017/
            00000001.jpg  # image files that are mentioned in the corresponding train/val2017.txt
            ...
        val2017/
            ...
    labels/
        train2017/
            00000001.txt  # label files that are mentioned in the corresponding train/val2017.txt
            ...
        val2017/
            ...
```

#### 4.2.2 数据集格式转换

由于**MindYOLO**在验证阶段选用图片名称作为`image_id`，因此图片名称只能为数值类型，而不能为字符串类型，还需要对图片进行改名。对SHWD数据集格式的转换包含如下步骤：

- 将图片复制到相应的路径下并改名
- 在根目录下相应的`txt`文件中写入该图片的相对路径
- 解析`xml`文件，在相应路径下生成对应的`txt`标注文件
- 验证集还需生成最终的`json`文件

详细实现可参考[convert_shwd2yolo.py](./convert_shwd2yolo.py)，运行方式如下：

```shell
python examples/finetune_SHWD/convert_shwd2yolo.py --root_dir /path_to_shwd/SHWD
```

运行以上命令将不改变原数据集，并在同级目录生成**YOLO格式**的SHWD数据集。

#### 4.2.3 编写yaml配置文件

配置文件主要包含`数据集`、`数据增强`、`损失函数`、`优化器`、`模型结构`涉及的相应参数，由于MindYOLO提供`yaml`文件继承机制，可以只将需要调整的参数编写到yolov7-tiny_shwd.yaml，可复用或不需要修改的参数可以继承于已有模型的`yaml`文件，其内容如下：

```yaml
__BASE__: [
  '../../configs/yolov7/yolov7-tiny.yaml',
]

per_batch_size: 16 # 单卡batchsize，总的batchsize=per_batch_size * device_num
img_size: 640 # image sizes
weight: ./yolov7-tiny_pretrain.ckpt
strict_load: False # 是否按严格加载ckpt内参数，默认True，若设成False，当分类数不一致，丢掉最后一层分类器的weight
log_interval: 10 # 每log_interval次迭代打印一次loss结果

data:
  dataset_name: shwd
  train_set: ./SHWD/train.txt # 实际训练数据路径
  val_set: ./SHWD/val.txt
  test_set: ./SHWD/val.txt
  nc: 2 # 分类数
  # class names
  names: [ 'person',  'hat' ] # 每一类的名字

optimizer:
  lr_init: 0.001  # initial learning rate
```

说明：

- ```__BASE__```为一个列表，表示继承的`yaml`文件所在路径，可以继承多个`yaml`文件
- `per_batch_size` 表示每张卡上的批处理大小
- `img_size`表示数据处理图片采用的图片尺寸
- `weight`为上述提到的预训练模型的文件路径
- `strict_load`表示丢弃shape不一致的参数
- `log_interval`表示日志打印间隔
- `data`为数据集相关参数
  - `dataset_name`为自定义数据集名称
  - `train_set` 训练数据集的路径
  - `val_set` 验证数据集的路径
  - `test_set` 测试数据集的路径
  - `nc` 为类别数量
  - `names` 为类别名称
- `optimizer`为优化器相关参数
  - `lr_init`为经过warm_up之后的初始化学习率，此处相比默认参数缩小了10倍

参数继承关系和参数说明可参考[configuration_CN.md](../../tutorials/configuration_CN.md)。

#### 4.2.4 下载预训练模型

可选用MindYOLO提供的[模型仓库](../../benchmark_results.md)列表中的模型，作为自定义数据集的预训练模型，预训练模型在COCO数据集上已经有较好的精度表现，相比从头训练，加载预训练模型一般会拥有更快的收敛速度以及更高的最终精度，并且能在一定程度上避免初始化不当导致的梯度消失、梯度爆炸等问题。

自定义数据集类别数通常与COCO数据集不一致，MindYOLO中各模型的检测头head结构跟数据集类别数有关，直接将预训练模型导入可能会因为shape不一致而导入失败，可以在yaml配置文件中设置strict_load参数为False，MindYOLO将自动舍弃shape不一致的参数，并抛出该module参数并未导入的告警

#### 4.2.5 模型微调

模型微调过程中，可首先按照默认配置进行训练，如效果不佳，可考虑调整以下参数：

- 学习率可调小一些，防止loss难以收敛
- per_batch_size可根据实际显存占用调整，通常per_batch_size越大，梯度计算越精确
- epochs可根据loss是否收敛进行调整
- anchor可根据实际物体大小进行调整

由于SHWD训练集只有约6000张图片，选用yolov7-tiny模型进行训练。

- 在多卡上进行分布式模型训练，以8卡为例:

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7-tiny_log python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --is_parallel True
  ```

- 在单卡上微调模型：

  ```shell
  python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml 
  ```

*实验结果(仅供参考)：直接用yolov7-tiny默认参数在SHWD数据集上训练，可取得AP50 87.0的精度。将lr_init参数由0.01改为0.001，可实现ap50为90.5的精度结果。*

#### 4.2.6 可视化推理

使用`demo/predict.py`进行可视化推理，运行方式如下：

```shell
python demo/predict.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
```

推理效果如下：

<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/00006630.jpg"/>
</div>

#### 4.2.7 更多的例子

- [基于MindYOLO的车辆检测案例](examples/finetune_car_detection/README.md)
- [基于MindYOLO的汽车配件分割](examples/finetune_carparts_seg/README.md)
- [基于MindYOLO的安全帽佩戴检测](examples/finetune_SHWD/README.md)
- [基于MindYOLO的自制巧克力花生豆检测](examples/finetune_single_class_dataset/README.md)
- [基于MindYOLO的无人机航拍图像检测](examples/finetune_visdrone/README.md)

## 5. Benchmark 和 Model Zoo 的精度情况

### 5.1 目标检测任务

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |    37.2     | 3.2M   | [yaml](./configs/yolov8/yolov8n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |    44.6     | 11.2M  | [yaml](./configs/yolov8/yolov8s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt)                 |
| YOLOv8 | M                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 25.9M  | [yaml](./configs/yolov8/yolov8m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt)                 |
| YOLOv8 | L                  |  16 * 8   |    640    | MS COCO 2017 |    52.8     | 43.7M  | [yaml](./configs/yolov8/yolov8l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt)                 |
| YOLOv8 | X                  |  16 * 8   |    640    | MS COCO 2017 |    53.7     | 68.2M  | [yaml](./configs/yolov8/yolov8x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt)                 |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |    37.5     | 6.2M   | [yaml](./configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt)              |
| YOLOv7 | L                  |  16 * 8   |    640    | MS COCO 2017 |    50.8     | 36.9M  | [yaml](./configs/yolov7/yolov7.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt)                   |
| YOLOv7 | X                  |  12 * 8   |    640    | MS COCO 2017 |    52.4     | 71.3M  | [yaml](./configs/yolov7/yolov7-x.yaml)       | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt)                 |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |    27.3     | 1.9M   | [yaml](./configs/yolov5/yolov5n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |    37.6     | 7.2M   | [yaml](./configs/yolov5/yolov5s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)                  |
| YOLOv5 | M                  |  32 * 8   |    640    | MS COCO 2017 |    44.9     | 21.2M  | [yaml](./configs/yolov5/yolov5m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)                  |
| YOLOv5 | L                  |  32 * 8   |    640    | MS COCO 2017 |    48.5     | 46.5M  | [yaml](./configs/yolov5/yolov5l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)                  |
| YOLOv5 | X                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 86.7M  | [yaml](./configs/yolov5/yolov5x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt)                  |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |    45.4     | 27.6M  | [yaml](./configs/yolov4/yolov4.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93.ckpt)      |
| YOLOv4 | CSPDarknet53(silu) |  16 * 8   |    608    | MS COCO 2017 |    45.8     | 27.6M  | [yaml](./configs/yolov4/yolov4-silu.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |    45.5     | 61.9M  | [yaml](./configs/yolov3/yolov3.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)         |
| YOLOX  | N                  |   8 * 8   |    416    | MS COCO 2017 |    24.1     | 0.9M   | [yaml](./configs/yolox/yolox-nano.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3.ckpt)                  |
| YOLOX  | Tiny               |   8 * 8   |    416    | MS COCO 2017 |    33.3     | 5.1M   | [yaml](./configs/yolox/yolox-tiny.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e.ckpt)               |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |    40.7     | 9.0M   | [yaml](./configs/yolox/yolox-s.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f.ckpt)                  |
| YOLOX  | M                  |   8 * 8   |    640    | MS COCO 2017 |    46.7     | 25.3M  | [yaml](./configs/yolox/yolox-m.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee.ckpt)                  |
| YOLOX  | L                  |   8 * 8   |    640    | MS COCO 2017 |    49.2     | 54.2M  | [yaml](./configs/yolox/yolox-l.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80.ckpt)                  |
| YOLOX  | X                  |   8 * 8   |    640    | MS COCO 2017 |    51.6     | 99.1M  | [yaml](./configs/yolox/yolox-x.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90.ckpt)                  |
| YOLOX  | Darknet53          |   8 * 8   |    640    | MS COCO 2017 |    47.7     | 63.7M  | [yaml](./configs/yolox/yolox-darknet53.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9.ckpt)          |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv10 | N                 |  32 * 8   |    640    | MS COCO 2017 |     38.3    | 513.63  | 2.8M   | [yaml](./configs/yolov10/yolov10n.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10n_500e_mAP383-c973023d.ckpt)                 |
| YOLOv10 | S                 |  32 * 8   |    640    | MS COCO 2017 |     45.7    | 503.38  | 8.2M   | [yaml](./configs/yolov10/yolov10s.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10s_500e_mAP457-8660fa84.ckpt)                 |
| YOLOv10 | M                 |  32 * 8   |    640    | MS COCO 2017 |     50.7    | 560.81  | 16.6M  | [yaml](./configs/yolov10/yolov10m.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10m_500e_mAP507-1cc8c5fb.ckpt)                 |
| YOLOv10 | B                 |  32 * 8   |    640    | MS COCO 2017 |     52.0    | 695.69  | 20.6M  | [yaml](./configs/yolov10/yolov10b.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10b_500e_mAP520-0b560f87.ckpt)                 |
| YOLOv10 | L                 |  32 * 8   |    640    | MS COCO 2017 |     52.6    | 782.61  | 25.9M  | [yaml](./configs/yolov10/yolov10l.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10l_500e_mAP526-226baf5f.ckpt)                 |
| YOLOv10 | X                 |  20 * 8   |    640    | MS COCO 2017 |     53.7    | 650.63  | 31.8M  | [yaml](./configs/yolov10/yolov10l.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10x_500e_mAP537-aaaa57bb.ckpt)                 |
| YOLOv9 | T                  |  16 * 8   |    640    | MS COCO 2017 |     37.3    | 350  | 2.0M   | [yaml](./configs/yolov9/yolov9-t.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9t_500e_MAP373-c0ee5cbc.ckpt)                 |
| YOLOv9 | S                  |  16 * 8   |    640    | MS COCO 2017 |     46.3    | 377  | 7.1M   | [yaml](./configs/yolov9/yolov9-s.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9s_500e_MAP463-b3cb691d.ckpt)                 |
| YOLOv9 | M                  |  16 * 8   |    640    | MS COCO 2017 |     51.4    | 499  | 20.0M   | [yaml](./configs/yolov9/yolov9-m.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9m_500e_MAP514-86aa8761.ckpt)                 |
| YOLOv9 | C                  |  16 * 8   |    640    | MS COCO 2017 |     52.6    | 627  | 25.3M   | [yaml](./configs/yolov9/yolov9-c.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9c_500e_MAP526-ff7bdf68.ckpt)                 |
| YOLOv9 | E                  |  16 * 8   |    640    | MS COCO 2017 |     55.1    | 826  | 57.3M   | [yaml](./configs/yolov9/yolov9-e.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9e_500e_MAP551-6b55c121.ckpt)                 |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |     37.3    | 373.55  | 3.2M   | [yaml](./configs/yolov8/yolov8n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-0e737186-910v2.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |     44.7    | 365.53  | 11.2M  | [yaml](./configs/yolov8/yolov8s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-fae4983f-910v2.ckpt)  |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |     37.5    | 496.21  | 6.2M   | [yaml](./configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-1d2ddf4b-910v2.ckpt)              |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |     27.4    | 736.08  | 1.9M   | [yaml](./configs/yolov5/yolov5n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-bedf9a93-910v2.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |     37.6    | 787.34  | 7.2M   | [yaml](./configs/yolov5/yolov5s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-df4a45b6-910v2.ckpt)                  |
| YOLOv5 | N6                 |  32 * 8   |    1280   | MS COCO 2017 |     35.7    | 1543.35 | 3.5M   | [yaml](./configs/yolov5/yolov5n6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n6_300e_mAP357-49d91077.ckpt)                  |
| YOLOv5 | S6                 |  32 * 8   |    1280   | MS COCO 2017 |     44.4    | 1514.98 | 13.6M  | [yaml](./configs/yolov5/yolov5s6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s6_300e_mAP444-aeaffe77.ckpt)                  |
| YOLOv5 | M6                 |  32 * 8   |    1280   | MS COCO 2017 |     51.1    | 1769.17 | 38.5M  | [yaml](./configs/yolov5/yolov5m6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5m6_300e_mAP511-025d9536.ckpt)                  |
| YOLOv5 | L6                 |  16 * 8   |    1280   | MS COCO 2017 |     53.6    | 894.65  | 82.9M  | [yaml](./configs/yolov5/yolov5l6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5l6_300e_mAP536-617a1cc1.ckpt)                  |
| YOLOv5 | X6                 |   8 * 8   |    1280   | MS COCO 2017 |     54.4    | 864.43  | 140.9M | [yaml](./configs/yolov5/yolov5x6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5x6_300e_mAP545-81ebdca9.ckpt)                  |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |     46.1    | 337.25  | 27.6M  | [yaml](./configs/yolov4/yolov4.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-64b8506f-910v2.ckpt)      |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |     46.6    | 396.60  | 61.9M  | [yaml](./configs/yolov3/yolov3.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-81895f09-910v2.ckpt)         |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |     41.0    | 242.15  | 9.0M   | [yaml](./configs/yolox/yolox-s.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-cebd0183-910v2.ckpt)                   |
</details>

### 5.2 分割任务

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name       | Scale | BatchSize | ImageSize | Dataset      | Box mAP (%) | Mask mAP (%) | Params |                Recipe                        | Download                                                                                                       |
|------------| :---: |   :---:   |   :---:   |--------------|    :---:    |     :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8-seg |   X   |  16 * 8   |    640    | MS COCO 2017 |     52.5    |     42.9     |  71.8M | [yaml](./configs/yolov8/seg/yolov8x-seg.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x-seg_300e_mAP_mask_429-b4920557.ckpt) |
</details>

### 注意

Box mAP: 表格中精度报告的是验证集上的结果。

更多结果请查看 [Benchmark Results](benchmark_results.md).

## 6. 注意

⚠️ 当前版本是基于 **MindSpore** 的 **图模式+静态Shape** 特性进行构建的，更多信息请在[MindSpore官方文档](https://mindspore.cn/docs/en/r2.0/note/static_graph_syntax_support.html)中查看。

## 7. 怎么贡献到我们的仓库

We appreciate all contributions including issues and PRs to make MindYOLO better.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## 8. License

MindYOLO is released under the [Apache License 2.0](LICENSE.md).

## 9. Acknowledgement

MindYOLO is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could support the growing research community, reimplement existing methods, and develop their own new real-time object detection methods by providing a flexible and standardized toolkit.

## 10. Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{MindSpore Object Detection YOLO 2023,
    title={{MindSpore Object Detection YOLO}:MindSpore Object Detection YOLO Toolbox and Benchmark},
    author={MindSpore YOLO Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindyolo}},
    year={2023}
}
```
