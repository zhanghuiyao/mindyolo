## Getting Started with MindYOLO

*This document provides a brief introduction to the usage of built-in command-line tools in MindYOLO.*

### 1. Infer a image

- **step1,** pick a model and its config file from the [Model Zoo](benchmark_results.md), such as `./configs/yolov7/yolov7.yaml`
  
- **step2,** download the corresponding pre-trained checkpoint from the [Model Zoo](benchmark_results.md) of each model

- **step3,** run YOLO object detection with the built-in configs:

    ```shell
    # Run with Ascend (By default)
    python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
    ```

- *Notes:*

  - The results will be saved in `./detect_results`

  - *For more details of the command line arguments, see `demo/predict.py -h` or look at its [source code](https://github.com/mindspore-lab/mindyolo/blob/master/deploy/predict.py) to understand their behavior.*

### 2. Training & Evaluating

- **First, Prepare Dataset**
  - Prepare your dataset in YOLO format. If trained with COCO (YOLO format), prepare it from [yolov5](https://github.com/ultralytics/yolov5) or the darknet.
  
    <details onclose>
    <summary>More Details</summary>

    ```text
      coco/
        {train,val}2017.txt
        annotations/
          instances_{train,val}2017.json
        images/
          {train,val}2017/
              00000001.jpg
              ...
              # image files that are mentioned in the corresponding train/val2017.txt
        labels/
          {train,val}2017/
              00000001.txt
              ...
              # label files that are mentioned in the corresponding train/val2017.txt
    ```

    </details>

- **Second, Training**

  - 1 NPU/CPU:

    ```shell
    python train.py --config ./configs/yolov7/yolov7.yaml 
    ```

  - To train a model on 8 NPUs:

    ```shell
    msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
    ```

- **Finally, Evaluating**

  - To evaluate a model's performance on 1 NPU/CPU:

    ```shell
    python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt
    ```

  - To evaluate a model's performance 8 NPUs:

    ```shell
    msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt --is_parallel True
    ```

- *Notes:*
  
  - The default hyper-parameter is used for 8-card training, and some parameters need to be adjusted in the case of a single card.
  
  - The default device is Ascend, and you can modify it by specifying 'device_target' as Ascend/CPU, as these are currently supported.

  - For more options, see `train/test.py -h`.

  - Notice that if you are using `msrun` startup with 2 devices, please add `--bind_core=True` to improve performance. For example:

    ```shell
    msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=8118 \
          --log_dir=msrun_log --join=True --cluster_time_out=300 \
          python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
    ```

    For more usage of `msrun`, please reference to [MindSpore Docs](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/startup_method.html).

### Deployment

See [depoly readme](./deploy/README.md).

### To use MindYOLO APIs in Your Code

To be supplemented.
