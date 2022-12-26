# 目录

<!-- TOC -->

- [目录](#目录)
- [BiRealNet描述](#BiRealNet描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的BiRealNet训练](#ImageNet-1k上的BiRealNet训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [BiRealNet描述](#目录)

深度卷积神经网络（CNN）由于精度高在视觉任务中已经有非常广泛的应用，但是 CNN 的模型过大限制了它在很多移动端的部署。模型压缩也因此变得尤为重要。在模型压缩方法中，将网络中的权重和激活都只用+1 或者-1 来表示将可以达到理论上的 32 倍的存储空间的节省和 64 倍的加速效应。由于它的权重和激活都只需要用 1bit 表示，因此极其有利于硬件上的部署和实现。

然而现有的二值化压缩方法在 imagenet 这样的大数据集上会有较大的精度下降。我们认为，这种精度的下降主要是有两方面造成的。1. 1-bit CNN 的表达能力本身很有限，不如实数值的网络。2. 1-bit CNN 在训练过程中有导数不匹配的问题导致难以收敛到很好的精度。

针对这两个问题，作者分别提出了解决方案。

通过分析作者发现，尽管输入 1-bit 卷积层的参数和激活值都是二值化的，但经过 xnor 和 bitcout 之后，网络内部会产生实数值，但是这个实数值的输出如果经过下一层 1-bit 卷积就又会被二值化，造成极大的信息丢失。



# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─imagenet
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```text
└── BiReal
    ├── eval.py
    ├── README.md
    ├── requriments.txt
    ├── scripts
    │   ├── run_distribute_train_ascend.sh
    │   ├── run_eval_ascend.sh
    │   ├── run_infer_310.sh
    │   └── run_standalone_train_ascend.sh
    ├── src
    │   ├── args.py
    │   ├── configs
    │   │   ├── birealnet34.yaml
    │   │   └── parser.py
    │   ├── data
    │   │   ├── augment
    │   │   │   ├── auto_augment.py
    │   │   │   ├── __init__.py
    │   │   │   ├── mixup.py
    │   │   │   ├── random_erasing.py
    │   │   │   └── transforms.py
    │   │   ├── data_utils
    │   │   │   ├── __init__.py
    │   │   │   └── moxing_adapter.py
    │   │   ├── imagenet.py
    │   │   └── __init__.py
    │   ├── models
    │   │   ├── birealnet_step.py
    │   │   ├── initializer.py
    │   │   ├── __init__.py
    │   │   └── layers
    │   │       ├── attention.py
    │   │       └── identity.py
    │   ├── tools
    │   │   ├── callback.py
    │   │   ├── cell.py
    │   │   ├── criterion.py
    │   │   ├── get_misc.py
    │   │   ├── __init__.py
    │   │   ├── optimizer.py
    │   │   └── schedulers.py
    │   └── trainer
    │       └── train_one_step.py
    └── train.py
					

```

## 脚本参数

在birealnet34.yaml.yaml中可以同时配置训练参数和评估参数。

- 配置BiRealNet和ImageNet-1k数据集。

  ```text
    # Architecture 62.2%
    arch: birealnet34
  
    # ===== Dataset ===== #
    data_url: ../data/imagenet
    set: ImageNet
    num_classes: 1000
    mix_up: 0.0
    cutmix: 0.0
    auto_augment: None
    interpolation: bilinear
    re_prob: 0.
    re_mode: pixel
    re_count: 1
    mixup_prob: 1.0
    switch_prob: 0.5
    mixup_mode: batch
    image_size: 224
    crop_pct: 0.875


    # ===== Learning Rate Policy ======== #
    optimizer: adam
    base_lr: 0.001
    warmup_lr: 0.00001
    min_lr: 0.000001
    lr_scheduler: linear_lr
    warmup_length: 0
    
    # ===== Network training config ===== #
    amp_level: O1
    beta: [ 0.9, 0.999 ]
    clip_global_norm_value: 5.
    is_dynamic_loss_scale: True
    epochs: 300
    cooldown_epochs: 10
    label_smoothing: 0.0
    weight_decay: 0.
    momentum: 0.9
    batch_size: 128
    drop_path_rate: 0.
    
    # ===== Hardware setup ===== #
    num_parallel_workers: 32
    device_target: Ascend
  ```

更多配置细节请参考脚本`birealnet34.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/birealnet34.yaml.yaml \
  > train.log 2>&1 &
  
  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  
  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  
  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/birealnet34.yaml.yaml > ./eval.log 2>&1 &
  
  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  [hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)


# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的BiRealNet训练

| 参数                 | Ascend                          |
| -------------------------- |---------------------------------|
|模型| BiRealNet                            |
| 模型版本              | BiRealNet34                  |
| 资源                   | Ascend 910 8卡                   |
| 上传日期              | 2022-11-04                      |
| MindSpore版本          | 1.5.1                           |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数        | epoch=300, batch_size=1024      |
| 优化器                  | Momentum                 |
| 损失函数              | SoftmaxCrossEntropyWithLogits          |
| 输出                    | 概率                              |
| 分类准确率             | 八卡：top1:62.45%      |
| 速度                      | 8卡：177.446毫秒/步                  |
| 训练耗时          | 19h30min03s（run on OpenI）       |


# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)