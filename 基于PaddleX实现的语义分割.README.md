# 基于PaddleSeg实现语义分割



------


# 一、作业任务

> 本次任务将基于PaddleSeg展开语义分割任务的学习与实践，baseline会提供PaddleSeg套件的基本使用，相关细节如有遗漏，可参考[10分钟上手PaddleSeg
](https://aistudio.baidu.com/aistudio/projectdetail/1672610?channelType=0&channel=0)

1. 选择提供的**五个数据集**中的一个数据集作为本次作业的训练数据，并**根据baseline跑通项目**

2. **可视化1-3张**预测图片与预测结果（本次提供的数据集没有验证与测试集，可将训练数据直接进行预测，展示训练数据的预测结果即可）


**加分项:**

3. 尝试**划分验证集**，再进行训练

4. 选择**其他网络**，调整训练参数进行更好的尝试


**PS**:PaddleSeg相关参考项目:

- [常规赛：PALM病理性近视病灶检测与分割基线方案](https://aistudio.baidu.com/aistudio/projectdetail/1941312)

- [PaddleSeg 2.0动态图：车道线图像分割任务简介](https://aistudio.baidu.com/aistudio/projectdetail/1752986?channelType=0&channel=0)

------

------

# 二、数据集说明

------

本项目使用的数据集是:[AI训练营]语义分割数据集合集，包含马分割，眼底血管分割，车道线分割，场景分割以及人像分割。

该数据集已加载到本环境中，位于:

**data/data103787/segDataset.zip**



```python
# unzip: 解压指令
# -o: 表示解压后进行输出
# -q: 表示静默模式，即不输出解压过程的日志
# -d: 解压到指定目录下，如果该文件夹不存在，会自动创建
!unzip -oq data/data103787/segDataset.zip -d segDataset
```

解压完成后，会在左侧文件目录多出一个**segDataset**的文件夹，该文件夹下有**5**个子文件夹：

- **horse -- 马分割数据**<二分类任务>

![](https://ai-studio-static-online.cdn.bcebos.com/2b12a7fab9ee409587a2aec332a70ba2bce0fcc4a10345a4aa38941db65e8d02)

- **fundusVessels -- 眼底血管分割数据**

> 灰度图，每个像素值对应相应的类别 -- 因此label不宜观察，但符合套件训练需要

![](https://ai-studio-static-online.cdn.bcebos.com/b515662defe548bdaa517b879722059ad53b5d87dd82441c8c4611124f6fdad0)

- **laneline -- 车道线分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/2aeccfe514e24cf98459df7c36421cddf78d9ddfc2cf41ffa4aafc10b13c8802)

- **facade -- 场景分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/57752d86fc5c4a10a3e4b91ae05a3e38d57d174419be4afeba22eb75b699112c)

- **cocome -- 人像分割数据**

> label非直接的图片，为json格式的标注文件，有需要的小伙伴可以看一看PaddleSeg的[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# tree: 查看文件夹树状结构
# -L: 表示遍历深度
!tree segDataset -L 2
```

    segDataset
    ├── cocome
    │   ├── Annotations
    │   └── Images
    ├── facade
    │   ├── Annotations
    │   └── Images
    ├── FundusVessels
    │   ├── Annotations
    │   └── Images
    ├── horse
    │   ├── Annotations
    │   └── Images
    └── laneline
        ├── Annotations
        └── Images
    
    15 directories, 0 files


> 查看数据label的像素分布，可从中得知分割任务的类别数： 脚本位于: **show_segDataset_label_cls_id.py**

> 关于人像分割数据分析，这里不做提示，可参考[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# 查看label中像素分类情况
!python show_segDataset_label_cls_id.py
```

    100%|████████████████████████████████████████| 328/328 [00:00<00:00, 957.31it/s]
    horse-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    horse为90分类
    horse实际应转换为2分类(将非0像素转换为像素值为1)
    
    
    100%|████████████████████████████████████████| 845/845 [00:04<00:00, 193.44it/s]
    facade-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8]
    facade为9分类
    
    
    100%|████████████████████████████████████████| 200/200 [00:01<00:00, 180.63it/s]
    fundusvessels-cls_id:  [0, 1]
    fundusvessels为2分类
    
    
    100%|███████████████████████████████████████| 4878/4878 [01:31<00:00, 53.55it/s]
    laneline-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    laneline为20分类


# 三、数据预处理

> 这里就以horse数据作为示例

-----

- 首先，通过上边的像素值分析以及horse本身的标签表现，我们确定horse数据集为二分类任务

- 然而，实际label中，却包含多个像素值，因此需要将horse中的所有label进行一个预处理

- 预处理内容为: 0值不变，非0值变为1，然后再保存label

- **并且保存文件格式为png，单通道图片为Label图片，最好保存为png——否则可能出现异常像素**

**对应horse的预处理脚本，位于:**

parse_horse_label.py


```python
!python parse_horse_label.py
```

    100%|████████████████████████████████████████| 328/328 [00:00<00:00, 394.56it/s]
    [0, 1]
    100%|███████████████████████████████████████| 328/328 [00:00<00:00, 1127.54it/s]
    horse-cls_id:  [0, 1]
    horse为2分类


- 预处理完成后，配置训练的索引文件txt，方便后边套件读取数据

> txt创建脚本位于: **horse_create_train_list.py**

> 同时，生成的txt位于: **segDataset/horse/train_list.txt**


```python
# 创建训练的数据索引txt
# 格式如下
# line1: train_img1.jpg train_label1.png
# line2: train_img2.jpg train_label2.png
!python horse_create_train_list.py
```

    100%|██████████████████████████████████████| 328/328 [00:00<00:00, 16461.83it/s]


# 四、使用套件开始训练

- 1.解压套件: 已挂载到本项目, 位于:**data/data102250/PaddleSeg-release-2.1.zip**


```python
# 解压套件
!unzip -oq data/data102250/PaddleSeg-release-2.1.zip
# 通过mv指令实现文件夹重命名
!mv PaddleSeg-release-2.1 PaddleSeg
```

- 2.选择模型，baseline选择**bisenet**, 位于: **PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml**

- 3.配置模型文件

> 首先修改训练数据集加载的dataset类型:

![](https://ai-studio-static-online.cdn.bcebos.com/2f5363d71034490290f720ea8bb0d6873d7df2712d4b4e84ae41b0378aed8b89)

> 然后配置训练数据集如下:

![](https://ai-studio-static-online.cdn.bcebos.com/29547856db4b4bfc80aa3732e143f2788589f9316c694f369c9bd1da44b815dc)

> 类似的，配置验证数据集: -- **注意修改train_path为val_path**

![](https://ai-studio-static-online.cdn.bcebos.com/09713aaaed6b4611a525d25aae67e4f0538224f7ac0241eb941d97892bf6c4c1)

<font color="red" size=4>其它模型可能需要到: PaddleSeg/configs/$_base_$  中的数据集文件进行配置，但修改参数与bisenet中的数据集修改参数相同 </font>

![](https://ai-studio-static-online.cdn.bcebos.com/b154dcbf15e14f43aa13455c0ceeaaebe0489c9a09dd439f9d32e8b0a31355ec)


- 4.开始训练

使用PaddleSeg的train.py，传入模型文件即可开启训练


```python
!python PaddleSeg/train.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--batch_size 4\
--iters 2000\
--learning_rate 0.01\
--save_interval 200\
--save_dir PaddleSeg/output\
--seed 2021\
--log_iters 20\
--do_eval\
--use_vdl

# --batch_size 4\  # 批大小
# --iters 2000\    # 迭代次数 -- 根据数据大小，批大小估计迭代次数
# --learning_rate 0.01\ # 学习率
# --save_interval 200\ # 保存周期 -- 迭代次数计算周期
# --save_dir PaddleSeg/output\ # 输出路径
# --seed 2021\ # 训练中使用到的随机数种子
# --log_iters 20\ # 日志频率 -- 迭代次数计算周期
# --do_eval\ # 边训练边验证
# --use_vdl # 使用vdl可视化记录
# 用于断训==即中断后继续上一次状态进行训练
# --resume_model model_dir
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    /home/aistudio/PaddleSeg/paddleseg/cvlibs/param_init.py:89: DeprecationWarning: invalid escape sequence \s
      """
    /home/aistudio/PaddleSeg/paddleseg/models/losses/binary_cross_entropy_loss.py:82: DeprecationWarning: invalid escape sequence \|
      """
    /home/aistudio/PaddleSeg/paddleseg/models/losses/lovasz_loss.py:50: DeprecationWarning: invalid escape sequence \i
      """
    /home/aistudio/PaddleSeg/paddleseg/models/losses/lovasz_loss.py:77: DeprecationWarning: invalid escape sequence \i
      """
    /home/aistudio/PaddleSeg/paddleseg/models/losses/lovasz_loss.py:120: DeprecationWarning: invalid escape sequence \i
      """
    2021-08-11 20:53:20 [INFO]	
    ------------Environment Information-------------
    platform: Linux-4.4.0-150-generic-x86_64-with-debian-stretch-sid
    Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0]
    Paddle compiled with cuda: True
    NVCC: Cuda compilation tools, release 10.1, V10.1.243
    cudnn: 7.6
    GPUs used: 1
    CUDA_VISIBLE_DEVICES: None
    GPU: ['GPU 0: Tesla V100-SXM2-32GB']
    GCC: gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
    PaddlePaddle: 2.0.2
    OpenCV: 4.1.1
    ------------------------------------------------
    Connecting to https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
    Downloading optic_disc_seg.zip
    [==================================================] 100.00%
    Uncompress optic_disc_seg.zip
    [==================================================] 100.00%
    2021-08-11 20:53:20 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 2000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - ignore_index: 255
        type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: data/optic_disc_seg
      mode: train
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: OpticDiscSeg
    val_dataset:
      dataset_root: data/optic_disc_seg
      mode: val
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: OpticDiscSeg
    ------------------------------------------------
    W0811 20:53:20.857307   332 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 20:53:20.857367   332 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    2021-08-11 20:53:44 [INFO]	[TRAIN] epoch: 1, iter: 20/2000, loss: 1.5415, lr: 0.009914, batch_cost: 0.4666, reader_cost: 0.01169, ips: 8.5732 samples/sec | ETA 00:15:23
    2021-08-11 20:53:53 [INFO]	[TRAIN] epoch: 1, iter: 40/2000, loss: 0.3494, lr: 0.009824, batch_cost: 0.4521, reader_cost: 0.00010, ips: 8.8483 samples/sec | ETA 00:14:46
    2021-08-11 20:54:02 [INFO]	[TRAIN] epoch: 1, iter: 60/2000, loss: 0.2826, lr: 0.009734, batch_cost: 0.4486, reader_cost: 0.00010, ips: 8.9167 samples/sec | ETA 00:14:30
    2021-08-11 20:54:11 [INFO]	[TRAIN] epoch: 2, iter: 80/2000, loss: 0.2370, lr: 0.009644, batch_cost: 0.4644, reader_cost: 0.00425, ips: 8.6128 samples/sec | ETA 00:14:51
    2021-08-11 20:54:21 [INFO]	[TRAIN] epoch: 2, iter: 100/2000, loss: 0.2196, lr: 0.009553, batch_cost: 0.4609, reader_cost: 0.00010, ips: 8.6791 samples/sec | ETA 00:14:35
    2021-08-11 20:54:30 [INFO]	[TRAIN] epoch: 2, iter: 120/2000, loss: 0.2557, lr: 0.009463, batch_cost: 0.4559, reader_cost: 0.00011, ips: 8.7746 samples/sec | ETA 00:14:17
    2021-08-11 20:54:39 [INFO]	[TRAIN] epoch: 3, iter: 140/2000, loss: 0.2113, lr: 0.009372, batch_cost: 0.4518, reader_cost: 0.00425, ips: 8.8541 samples/sec | ETA 00:14:00
    2021-08-11 20:54:48 [INFO]	[TRAIN] epoch: 3, iter: 160/2000, loss: 0.1952, lr: 0.009282, batch_cost: 0.4509, reader_cost: 0.00010, ips: 8.8706 samples/sec | ETA 00:13:49
    2021-08-11 20:54:57 [INFO]	[TRAIN] epoch: 3, iter: 180/2000, loss: 0.1919, lr: 0.009191, batch_cost: 0.4524, reader_cost: 0.00011, ips: 8.8425 samples/sec | ETA 00:13:43
    2021-08-11 20:55:06 [INFO]	[TRAIN] epoch: 4, iter: 200/2000, loss: 0.1772, lr: 0.009100, batch_cost: 0.4605, reader_cost: 0.00414, ips: 8.6867 samples/sec | ETA 00:13:48
    2021-08-11 20:55:06 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT32, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    76/76 [==============================] - 2s 32ms/step - batch_cost: 0.0322 - reader cost: 3.3694e-
    2021-08-11 20:55:09 [INFO]	[EVAL] #Images: 76 mIoU: 0.7622 Acc: 0.9908 Kappa: 0.6915 
    2021-08-11 20:55:09 [INFO]	[EVAL] Class IoU: 
    [0.9907 0.5337]
    2021-08-11 20:55:09 [INFO]	[EVAL] Class Acc: 
    [0.9921 0.8828]
    2021-08-11 20:55:09 [INFO]	[EVAL] The model with the best validation mIoU (0.7622) was saved at iter 200.
    2021-08-11 20:55:18 [INFO]	[TRAIN] epoch: 4, iter: 220/2000, loss: 0.1722, lr: 0.009009, batch_cost: 0.4512, reader_cost: 0.00010, ips: 8.8646 samples/sec | ETA 00:13:23
    2021-08-11 20:55:27 [INFO]	[TRAIN] epoch: 4, iter: 240/2000, loss: 0.1624, lr: 0.008918, batch_cost: 0.4480, reader_cost: 0.00011, ips: 8.9293 samples/sec | ETA 00:13:08
    2021-08-11 20:55:36 [INFO]	[TRAIN] epoch: 4, iter: 260/2000, loss: 0.1683, lr: 0.008827, batch_cost: 0.4542, reader_cost: 0.00011, ips: 8.8060 samples/sec | ETA 00:13:10
    2021-08-11 20:55:45 [INFO]	[TRAIN] epoch: 5, iter: 280/2000, loss: 0.1676, lr: 0.008735, batch_cost: 0.4577, reader_cost: 0.00403, ips: 8.7392 samples/sec | ETA 00:13:07
    2021-08-11 20:55:54 [INFO]	[TRAIN] epoch: 5, iter: 300/2000, loss: 0.1608, lr: 0.008644, batch_cost: 0.4558, reader_cost: 0.00010, ips: 8.7751 samples/sec | ETA 00:12:54
    2021-08-11 20:56:03 [INFO]	[TRAIN] epoch: 5, iter: 320/2000, loss: 0.1482, lr: 0.008552, batch_cost: 0.4604, reader_cost: 0.00010, ips: 8.6886 samples/sec | ETA 00:12:53
    2021-08-11 20:56:13 [INFO]	[TRAIN] epoch: 6, iter: 340/2000, loss: 0.1447, lr: 0.008461, batch_cost: 0.4565, reader_cost: 0.00419, ips: 8.7630 samples/sec | ETA 00:12:37
    2021-08-11 20:56:22 [INFO]	[TRAIN] epoch: 6, iter: 360/2000, loss: 0.1456, lr: 0.008369, batch_cost: 0.4625, reader_cost: 0.00010, ips: 8.6494 samples/sec | ETA 00:12:38
    2021-08-11 20:56:31 [INFO]	[TRAIN] epoch: 6, iter: 380/2000, loss: 0.1431, lr: 0.008277, batch_cost: 0.4503, reader_cost: 0.00011, ips: 8.8836 samples/sec | ETA 00:12:09
    2021-08-11 20:56:40 [INFO]	[TRAIN] epoch: 7, iter: 400/2000, loss: 0.1347, lr: 0.008185, batch_cost: 0.4639, reader_cost: 0.00453, ips: 8.6226 samples/sec | ETA 00:12:22
    2021-08-11 20:56:40 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 3s 33ms/step - batch_cost: 0.0330 - reader cost: 3.0301e-0
    2021-08-11 20:56:43 [INFO]	[EVAL] #Images: 76 mIoU: 0.8300 Acc: 0.9933 Kappa: 0.7967 
    2021-08-11 20:56:43 [INFO]	[EVAL] Class IoU: 
    [0.9932 0.6668]
    2021-08-11 20:56:43 [INFO]	[EVAL] Class Acc: 
    [0.995 0.88 ]
    2021-08-11 20:56:43 [INFO]	[EVAL] The model with the best validation mIoU (0.8300) was saved at iter 400.
    2021-08-11 20:56:52 [INFO]	[TRAIN] epoch: 7, iter: 420/2000, loss: 0.1431, lr: 0.008093, batch_cost: 0.4519, reader_cost: 0.00010, ips: 8.8519 samples/sec | ETA 00:11:53
    2021-08-11 20:57:01 [INFO]	[TRAIN] epoch: 7, iter: 440/2000, loss: 0.1190, lr: 0.008001, batch_cost: 0.4550, reader_cost: 0.00010, ips: 8.7903 samples/sec | ETA 00:11:49
    2021-08-11 20:57:10 [INFO]	[TRAIN] epoch: 7, iter: 460/2000, loss: 0.1381, lr: 0.007909, batch_cost: 0.4528, reader_cost: 0.00010, ips: 8.8349 samples/sec | ETA 00:11:37
    2021-08-11 20:57:19 [INFO]	[TRAIN] epoch: 8, iter: 480/2000, loss: 0.1168, lr: 0.007816, batch_cost: 0.4600, reader_cost: 0.00427, ips: 8.6954 samples/sec | ETA 00:11:39
    2021-08-11 20:57:28 [INFO]	[TRAIN] epoch: 8, iter: 500/2000, loss: 0.1137, lr: 0.007724, batch_cost: 0.4533, reader_cost: 0.00011, ips: 8.8243 samples/sec | ETA 00:11:19
    2021-08-11 20:57:37 [INFO]	[TRAIN] epoch: 8, iter: 520/2000, loss: 0.1353, lr: 0.007631, batch_cost: 0.4461, reader_cost: 0.00010, ips: 8.9662 samples/sec | ETA 00:11:00
    2021-08-11 20:57:47 [INFO]	[TRAIN] epoch: 9, iter: 540/2000, loss: 0.1087, lr: 0.007538, batch_cost: 0.4587, reader_cost: 0.00420, ips: 8.7199 samples/sec | ETA 00:11:09
    2021-08-11 20:57:56 [INFO]	[TRAIN] epoch: 9, iter: 560/2000, loss: 0.1217, lr: 0.007445, batch_cost: 0.4543, reader_cost: 0.00010, ips: 8.8038 samples/sec | ETA 00:10:54
    2021-08-11 20:58:05 [INFO]	[TRAIN] epoch: 9, iter: 580/2000, loss: 0.1153, lr: 0.007352, batch_cost: 0.4567, reader_cost: 0.00011, ips: 8.7588 samples/sec | ETA 00:10:48
    2021-08-11 20:58:14 [INFO]	[TRAIN] epoch: 10, iter: 600/2000, loss: 0.1146, lr: 0.007259, batch_cost: 0.4587, reader_cost: 0.00418, ips: 8.7205 samples/sec | ETA 00:10:42
    2021-08-11 20:58:14 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 31ms/step - batch_cost: 0.0311 - reader cost: 3.0030e-0
    2021-08-11 20:58:16 [INFO]	[EVAL] #Images: 76 mIoU: 0.8472 Acc: 0.9939 Kappa: 0.8209 
    2021-08-11 20:58:16 [INFO]	[EVAL] Class IoU: 
    [0.9938 0.7006]
    2021-08-11 20:58:16 [INFO]	[EVAL] Class Acc: 
    [0.9958 0.8815]
    2021-08-11 20:58:17 [INFO]	[EVAL] The model with the best validation mIoU (0.8472) was saved at iter 600.
    2021-08-11 20:58:26 [INFO]	[TRAIN] epoch: 10, iter: 620/2000, loss: 0.1193, lr: 0.007166, batch_cost: 0.4528, reader_cost: 0.00012, ips: 8.8333 samples/sec | ETA 00:10:24
    2021-08-11 20:58:35 [INFO]	[TRAIN] epoch: 10, iter: 640/2000, loss: 0.1112, lr: 0.007072, batch_cost: 0.4545, reader_cost: 0.00012, ips: 8.8018 samples/sec | ETA 00:10:18
    2021-08-11 20:58:44 [INFO]	[TRAIN] epoch: 10, iter: 660/2000, loss: 0.1001, lr: 0.006978, batch_cost: 0.4520, reader_cost: 0.00011, ips: 8.8503 samples/sec | ETA 00:10:05
    2021-08-11 20:58:53 [INFO]	[TRAIN] epoch: 11, iter: 680/2000, loss: 0.1023, lr: 0.006885, batch_cost: 0.4570, reader_cost: 0.00418, ips: 8.7518 samples/sec | ETA 00:10:03
    2021-08-11 20:59:02 [INFO]	[TRAIN] epoch: 11, iter: 700/2000, loss: 0.1132, lr: 0.006791, batch_cost: 0.4543, reader_cost: 0.00010, ips: 8.8038 samples/sec | ETA 00:09:50
    2021-08-11 20:59:11 [INFO]	[TRAIN] epoch: 11, iter: 720/2000, loss: 0.1011, lr: 0.006697, batch_cost: 0.4525, reader_cost: 0.00010, ips: 8.8405 samples/sec | ETA 00:09:39
    2021-08-11 20:59:20 [INFO]	[TRAIN] epoch: 12, iter: 740/2000, loss: 0.1183, lr: 0.006603, batch_cost: 0.4548, reader_cost: 0.00420, ips: 8.7948 samples/sec | ETA 00:09:33
    2021-08-11 20:59:29 [INFO]	[TRAIN] epoch: 12, iter: 760/2000, loss: 0.1099, lr: 0.006508, batch_cost: 0.4498, reader_cost: 0.00011, ips: 8.8922 samples/sec | ETA 00:09:17
    2021-08-11 20:59:38 [INFO]	[TRAIN] epoch: 12, iter: 780/2000, loss: 0.1076, lr: 0.006414, batch_cost: 0.4505, reader_cost: 0.00010, ips: 8.8790 samples/sec | ETA 00:09:09
    2021-08-11 20:59:47 [INFO]	[TRAIN] epoch: 13, iter: 800/2000, loss: 0.1000, lr: 0.006319, batch_cost: 0.4574, reader_cost: 0.00414, ips: 8.7457 samples/sec | ETA 00:09:08
    2021-08-11 20:59:47 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 33ms/step - batch_cost: 0.0324 - reader cost: 2.9994e-0
    2021-08-11 20:59:50 [INFO]	[EVAL] #Images: 76 mIoU: 0.8647 Acc: 0.9944 Kappa: 0.8444 
    2021-08-11 20:59:50 [INFO]	[EVAL] Class IoU: 
    [0.9944 0.7349]
    2021-08-11 20:59:50 [INFO]	[EVAL] Class Acc: 
    [0.997  0.8556]
    2021-08-11 20:59:50 [INFO]	[EVAL] The model with the best validation mIoU (0.8647) was saved at iter 800.
    2021-08-11 20:59:59 [INFO]	[TRAIN] epoch: 13, iter: 820/2000, loss: 0.1002, lr: 0.006224, batch_cost: 0.4532, reader_cost: 0.00011, ips: 8.8258 samples/sec | ETA 00:08:54
    2021-08-11 21:00:08 [INFO]	[TRAIN] epoch: 13, iter: 840/2000, loss: 0.1037, lr: 0.006129, batch_cost: 0.4556, reader_cost: 0.00010, ips: 8.7794 samples/sec | ETA 00:08:48
    2021-08-11 21:00:17 [INFO]	[TRAIN] epoch: 14, iter: 860/2000, loss: 0.0919, lr: 0.006034, batch_cost: 0.4527, reader_cost: 0.00404, ips: 8.8358 samples/sec | ETA 00:08:36
    2021-08-11 21:00:27 [INFO]	[TRAIN] epoch: 14, iter: 880/2000, loss: 0.1063, lr: 0.005939, batch_cost: 0.4539, reader_cost: 0.00011, ips: 8.8117 samples/sec | ETA 00:08:28
    2021-08-11 21:00:36 [INFO]	[TRAIN] epoch: 14, iter: 900/2000, loss: 0.1047, lr: 0.005844, batch_cost: 0.4548, reader_cost: 0.00011, ips: 8.7942 samples/sec | ETA 00:08:20
    2021-08-11 21:00:45 [INFO]	[TRAIN] epoch: 14, iter: 920/2000, loss: 0.0849, lr: 0.005748, batch_cost: 0.4493, reader_cost: 0.00010, ips: 8.9032 samples/sec | ETA 00:08:05
    2021-08-11 21:00:54 [INFO]	[TRAIN] epoch: 15, iter: 940/2000, loss: 0.1015, lr: 0.005652, batch_cost: 0.4585, reader_cost: 0.00432, ips: 8.7250 samples/sec | ETA 00:08:05
    2021-08-11 21:01:03 [INFO]	[TRAIN] epoch: 15, iter: 960/2000, loss: 0.1026, lr: 0.005556, batch_cost: 0.4571, reader_cost: 0.00010, ips: 8.7506 samples/sec | ETA 00:07:55
    2021-08-11 21:01:12 [INFO]	[TRAIN] epoch: 15, iter: 980/2000, loss: 0.0915, lr: 0.005460, batch_cost: 0.4499, reader_cost: 0.00010, ips: 8.8909 samples/sec | ETA 00:07:38
    2021-08-11 21:01:21 [INFO]	[TRAIN] epoch: 16, iter: 1000/2000, loss: 0.0906, lr: 0.005364, batch_cost: 0.4602, reader_cost: 0.00388, ips: 8.6927 samples/sec | ETA 00:07:40
    2021-08-11 21:01:21 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 32ms/step - batch_cost: 0.0317 - reader cost: 2.9603e-0
    2021-08-11 21:01:24 [INFO]	[EVAL] #Images: 76 mIoU: 0.8720 Acc: 0.9948 Kappa: 0.8540 
    2021-08-11 21:01:24 [INFO]	[EVAL] Class IoU: 
    [0.9947 0.7493]
    2021-08-11 21:01:24 [INFO]	[EVAL] Class Acc: 
    [0.9973 0.8597]
    2021-08-11 21:01:24 [INFO]	[EVAL] The model with the best validation mIoU (0.8720) was saved at iter 1000.
    2021-08-11 21:01:33 [INFO]	[TRAIN] epoch: 16, iter: 1020/2000, loss: 0.0892, lr: 0.005267, batch_cost: 0.4515, reader_cost: 0.00011, ips: 8.8601 samples/sec | ETA 00:07:22
    2021-08-11 21:01:42 [INFO]	[TRAIN] epoch: 16, iter: 1040/2000, loss: 0.0961, lr: 0.005170, batch_cost: 0.4497, reader_cost: 0.00010, ips: 8.8950 samples/sec | ETA 00:07:11
    2021-08-11 21:01:51 [INFO]	[TRAIN] epoch: 17, iter: 1060/2000, loss: 0.1012, lr: 0.005073, batch_cost: 0.4572, reader_cost: 0.00419, ips: 8.7487 samples/sec | ETA 00:07:09
    2021-08-11 21:02:00 [INFO]	[TRAIN] epoch: 17, iter: 1080/2000, loss: 0.0998, lr: 0.004976, batch_cost: 0.4536, reader_cost: 0.00011, ips: 8.8189 samples/sec | ETA 00:06:57
    2021-08-11 21:02:09 [INFO]	[TRAIN] epoch: 17, iter: 1100/2000, loss: 0.0908, lr: 0.004879, batch_cost: 0.4556, reader_cost: 0.00010, ips: 8.7806 samples/sec | ETA 00:06:49
    2021-08-11 21:02:18 [INFO]	[TRAIN] epoch: 17, iter: 1120/2000, loss: 0.0919, lr: 0.004781, batch_cost: 0.4527, reader_cost: 0.00010, ips: 8.8357 samples/sec | ETA 00:06:38
    2021-08-11 21:02:27 [INFO]	[TRAIN] epoch: 18, iter: 1140/2000, loss: 0.0884, lr: 0.004684, batch_cost: 0.4553, reader_cost: 0.00451, ips: 8.7854 samples/sec | ETA 00:06:31
    2021-08-11 21:02:37 [INFO]	[TRAIN] epoch: 18, iter: 1160/2000, loss: 0.0998, lr: 0.004586, batch_cost: 0.4580, reader_cost: 0.00011, ips: 8.7329 samples/sec | ETA 00:06:24
    2021-08-11 21:02:46 [INFO]	[TRAIN] epoch: 18, iter: 1180/2000, loss: 0.0866, lr: 0.004487, batch_cost: 0.4519, reader_cost: 0.00010, ips: 8.8523 samples/sec | ETA 00:06:10
    2021-08-11 21:02:55 [INFO]	[TRAIN] epoch: 19, iter: 1200/2000, loss: 0.0955, lr: 0.004389, batch_cost: 0.4576, reader_cost: 0.00429, ips: 8.7416 samples/sec | ETA 00:06:06
    2021-08-11 21:02:55 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 33ms/step - batch_cost: 0.0324 - reader cost: 3.0603e-
    2021-08-11 21:02:57 [INFO]	[EVAL] #Images: 76 mIoU: 0.8793 Acc: 0.9953 Kappa: 0.8634 
    2021-08-11 21:02:57 [INFO]	[EVAL] Class IoU: 
    [0.9952 0.7634]
    2021-08-11 21:02:57 [INFO]	[EVAL] Class Acc: 
    [0.9967 0.9114]
    2021-08-11 21:02:58 [INFO]	[EVAL] The model with the best validation mIoU (0.8793) was saved at iter 1200.
    2021-08-11 21:03:07 [INFO]	[TRAIN] epoch: 19, iter: 1220/2000, loss: 0.0905, lr: 0.004290, batch_cost: 0.4538, reader_cost: 0.00010, ips: 8.8140 samples/sec | ETA 00:05:53
    2021-08-11 21:03:16 [INFO]	[TRAIN] epoch: 19, iter: 1240/2000, loss: 0.0818, lr: 0.004191, batch_cost: 0.4551, reader_cost: 0.00010, ips: 8.7886 samples/sec | ETA 00:05:45
    2021-08-11 21:03:25 [INFO]	[TRAIN] epoch: 20, iter: 1260/2000, loss: 0.0852, lr: 0.004092, batch_cost: 0.4595, reader_cost: 0.00397, ips: 8.7043 samples/sec | ETA 00:05:40
    2021-08-11 21:03:34 [INFO]	[TRAIN] epoch: 20, iter: 1280/2000, loss: 0.0918, lr: 0.003992, batch_cost: 0.4598, reader_cost: 0.00012, ips: 8.6986 samples/sec | ETA 00:05:31
    2021-08-11 21:03:43 [INFO]	[TRAIN] epoch: 20, iter: 1300/2000, loss: 0.1038, lr: 0.003892, batch_cost: 0.4452, reader_cost: 0.00010, ips: 8.9844 samples/sec | ETA 00:05:11
    2021-08-11 21:03:52 [INFO]	[TRAIN] epoch: 20, iter: 1320/2000, loss: 0.0810, lr: 0.003792, batch_cost: 0.4501, reader_cost: 0.00010, ips: 8.8867 samples/sec | ETA 00:05:06
    2021-08-11 21:04:01 [INFO]	[TRAIN] epoch: 21, iter: 1340/2000, loss: 0.0854, lr: 0.003692, batch_cost: 0.4574, reader_cost: 0.00416, ips: 8.7443 samples/sec | ETA 00:05:01
    2021-08-11 21:04:10 [INFO]	[TRAIN] epoch: 21, iter: 1360/2000, loss: 0.0857, lr: 0.003591, batch_cost: 0.4566, reader_cost: 0.00010, ips: 8.7601 samples/sec | ETA 00:04:52
    2021-08-11 21:04:19 [INFO]	[TRAIN] epoch: 21, iter: 1380/2000, loss: 0.0964, lr: 0.003490, batch_cost: 0.4512, reader_cost: 0.00011, ips: 8.8656 samples/sec | ETA 00:04:39
    2021-08-11 21:04:28 [INFO]	[TRAIN] epoch: 22, iter: 1400/2000, loss: 0.0845, lr: 0.003389, batch_cost: 0.4510, reader_cost: 0.00426, ips: 8.8687 samples/sec | ETA 00:04:30
    2021-08-11 21:04:28 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 32ms/step - batch_cost: 0.0320 - reader cost: 3.2155e-0
    2021-08-11 21:04:31 [INFO]	[EVAL] #Images: 76 mIoU: 0.8765 Acc: 0.9952 Kappa: 0.8598 
    2021-08-11 21:04:31 [INFO]	[EVAL] Class IoU: 
    [0.9951 0.7579]
    2021-08-11 21:04:31 [INFO]	[EVAL] Class Acc: 
    [0.9966 0.9097]
    2021-08-11 21:04:31 [INFO]	[EVAL] The model with the best validation mIoU (0.8793) was saved at iter 1200.
    2021-08-11 21:04:40 [INFO]	[TRAIN] epoch: 22, iter: 1420/2000, loss: 0.0929, lr: 0.003287, batch_cost: 0.4566, reader_cost: 0.00010, ips: 8.7595 samples/sec | ETA 00:04:24
    2021-08-11 21:04:49 [INFO]	[TRAIN] epoch: 22, iter: 1440/2000, loss: 0.0810, lr: 0.003185, batch_cost: 0.4517, reader_cost: 0.00010, ips: 8.8545 samples/sec | ETA 00:04:12
    2021-08-11 21:04:58 [INFO]	[TRAIN] epoch: 23, iter: 1460/2000, loss: 0.0876, lr: 0.003083, batch_cost: 0.4611, reader_cost: 0.00431, ips: 8.6757 samples/sec | ETA 00:04:08
    2021-08-11 21:05:08 [INFO]	[TRAIN] epoch: 23, iter: 1480/2000, loss: 0.0801, lr: 0.002980, batch_cost: 0.4566, reader_cost: 0.00011, ips: 8.7601 samples/sec | ETA 00:03:57
    2021-08-11 21:05:17 [INFO]	[TRAIN] epoch: 23, iter: 1500/2000, loss: 0.0808, lr: 0.002877, batch_cost: 0.4653, reader_cost: 0.00010, ips: 8.5972 samples/sec | ETA 00:03:52
    2021-08-11 21:05:26 [INFO]	[TRAIN] epoch: 24, iter: 1520/2000, loss: 0.0872, lr: 0.002773, batch_cost: 0.4594, reader_cost: 0.00416, ips: 8.7066 samples/sec | ETA 00:03:40
    2021-08-11 21:05:35 [INFO]	[TRAIN] epoch: 24, iter: 1540/2000, loss: 0.0805, lr: 0.002669, batch_cost: 0.4525, reader_cost: 0.00010, ips: 8.8399 samples/sec | ETA 00:03:28
    2021-08-11 21:05:44 [INFO]	[TRAIN] epoch: 24, iter: 1560/2000, loss: 0.0767, lr: 0.002565, batch_cost: 0.4475, reader_cost: 0.00010, ips: 8.9386 samples/sec | ETA 00:03:16
    2021-08-11 21:05:53 [INFO]	[TRAIN] epoch: 24, iter: 1580/2000, loss: 0.0981, lr: 0.002460, batch_cost: 0.4558, reader_cost: 0.00010, ips: 8.7756 samples/sec | ETA 00:03:11
    2021-08-11 21:06:02 [INFO]	[TRAIN] epoch: 25, iter: 1600/2000, loss: 0.0902, lr: 0.002355, batch_cost: 0.4597, reader_cost: 0.00402, ips: 8.7015 samples/sec | ETA 00:03:03
    2021-08-11 21:06:02 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 32ms/step - batch_cost: 0.0321 - reader cost: 2.9852e-0
    2021-08-11 21:06:05 [INFO]	[EVAL] #Images: 76 mIoU: 0.8814 Acc: 0.9953 Kappa: 0.8661 
    2021-08-11 21:06:05 [INFO]	[EVAL] Class IoU: 
    [0.9952 0.7676]
    2021-08-11 21:06:05 [INFO]	[EVAL] Class Acc: 
    [0.997  0.8977]
    2021-08-11 21:06:05 [INFO]	[EVAL] The model with the best validation mIoU (0.8814) was saved at iter 1600.
    2021-08-11 21:06:14 [INFO]	[TRAIN] epoch: 25, iter: 1620/2000, loss: 0.0830, lr: 0.002249, batch_cost: 0.4555, reader_cost: 0.00010, ips: 8.7825 samples/sec | ETA 00:02:53
    2021-08-11 21:06:23 [INFO]	[TRAIN] epoch: 25, iter: 1640/2000, loss: 0.0858, lr: 0.002142, batch_cost: 0.4523, reader_cost: 0.00010, ips: 8.8446 samples/sec | ETA 00:02:42
    2021-08-11 21:06:33 [INFO]	[TRAIN] epoch: 26, iter: 1660/2000, loss: 0.0960, lr: 0.002035, batch_cost: 0.4572, reader_cost: 0.00409, ips: 8.7491 samples/sec | ETA 00:02:35
    2021-08-11 21:06:42 [INFO]	[TRAIN] epoch: 26, iter: 1680/2000, loss: 0.0787, lr: 0.001927, batch_cost: 0.4577, reader_cost: 0.00010, ips: 8.7395 samples/sec | ETA 00:02:26
    2021-08-11 21:06:51 [INFO]	[TRAIN] epoch: 26, iter: 1700/2000, loss: 0.0802, lr: 0.001819, batch_cost: 0.4564, reader_cost: 0.00010, ips: 8.7641 samples/sec | ETA 00:02:16
    2021-08-11 21:07:00 [INFO]	[TRAIN] epoch: 27, iter: 1720/2000, loss: 0.0797, lr: 0.001710, batch_cost: 0.4655, reader_cost: 0.00407, ips: 8.5936 samples/sec | ETA 00:02:10
    2021-08-11 21:07:09 [INFO]	[TRAIN] epoch: 27, iter: 1740/2000, loss: 0.0814, lr: 0.001600, batch_cost: 0.4527, reader_cost: 0.00010, ips: 8.8354 samples/sec | ETA 00:01:57
    2021-08-11 21:07:18 [INFO]	[TRAIN] epoch: 27, iter: 1760/2000, loss: 0.0830, lr: 0.001489, batch_cost: 0.4554, reader_cost: 0.00010, ips: 8.7837 samples/sec | ETA 00:01:49
    2021-08-11 21:07:27 [INFO]	[TRAIN] epoch: 27, iter: 1780/2000, loss: 0.0943, lr: 0.001377, batch_cost: 0.4532, reader_cost: 0.00012, ips: 8.8259 samples/sec | ETA 00:01:39
    2021-08-11 21:07:37 [INFO]	[TRAIN] epoch: 28, iter: 1800/2000, loss: 0.0834, lr: 0.001265, batch_cost: 0.4588, reader_cost: 0.00428, ips: 8.7177 samples/sec | ETA 00:01:31
    2021-08-11 21:07:37 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 3s 33ms/step - batch_cost: 0.0328 - reader cost: 2.9793e-0
    2021-08-11 21:07:39 [INFO]	[EVAL] #Images: 76 mIoU: 0.8818 Acc: 0.9953 Kappa: 0.8666 
    2021-08-11 21:07:39 [INFO]	[EVAL] Class IoU: 
    [0.9953 0.7682]
    2021-08-11 21:07:39 [INFO]	[EVAL] Class Acc: 
    [0.997  0.9004]
    2021-08-11 21:07:39 [INFO]	[EVAL] The model with the best validation mIoU (0.8818) was saved at iter 1800.
    2021-08-11 21:07:48 [INFO]	[TRAIN] epoch: 28, iter: 1820/2000, loss: 0.0853, lr: 0.001151, batch_cost: 0.4529, reader_cost: 0.00010, ips: 8.8325 samples/sec | ETA 00:01:21
    2021-08-11 21:07:57 [INFO]	[TRAIN] epoch: 28, iter: 1840/2000, loss: 0.0798, lr: 0.001036, batch_cost: 0.4490, reader_cost: 0.00010, ips: 8.9083 samples/sec | ETA 00:01:11
    2021-08-11 21:08:07 [INFO]	[TRAIN] epoch: 29, iter: 1860/2000, loss: 0.0786, lr: 0.000919, batch_cost: 0.4661, reader_cost: 0.00431, ips: 8.5825 samples/sec | ETA 00:01:05
    2021-08-11 21:08:16 [INFO]	[TRAIN] epoch: 29, iter: 1880/2000, loss: 0.0828, lr: 0.000801, batch_cost: 0.4522, reader_cost: 0.00010, ips: 8.8458 samples/sec | ETA 00:00:54
    2021-08-11 21:08:25 [INFO]	[TRAIN] epoch: 29, iter: 1900/2000, loss: 0.0818, lr: 0.000681, batch_cost: 0.4555, reader_cost: 0.00011, ips: 8.7816 samples/sec | ETA 00:00:45
    2021-08-11 21:08:34 [INFO]	[TRAIN] epoch: 30, iter: 1920/2000, loss: 0.0851, lr: 0.000558, batch_cost: 0.4635, reader_cost: 0.00448, ips: 8.6300 samples/sec | ETA 00:00:37
    2021-08-11 21:08:43 [INFO]	[TRAIN] epoch: 30, iter: 1940/2000, loss: 0.0794, lr: 0.000432, batch_cost: 0.4609, reader_cost: 0.00011, ips: 8.6781 samples/sec | ETA 00:00:27
    2021-08-11 21:08:53 [INFO]	[TRAIN] epoch: 30, iter: 1960/2000, loss: 0.0847, lr: 0.000302, batch_cost: 0.4559, reader_cost: 0.00011, ips: 8.7743 samples/sec | ETA 00:00:18
    2021-08-11 21:09:02 [INFO]	[TRAIN] epoch: 30, iter: 1980/2000, loss: 0.0774, lr: 0.000166, batch_cost: 0.4550, reader_cost: 0.00010, ips: 8.7907 samples/sec | ETA 00:00:09
    2021-08-11 21:09:11 [INFO]	[TRAIN] epoch: 31, iter: 2000/2000, loss: 0.0817, lr: 0.000011, batch_cost: 0.4565, reader_cost: 0.00422, ips: 8.7629 samples/sec | ETA 00:00:00
    2021-08-11 21:09:11 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    76/76 [==============================] - 2s 33ms/step - batch_cost: 0.0325 - reader cost: 3.0266e-0
    2021-08-11 21:09:13 [INFO]	[EVAL] #Images: 76 mIoU: 0.8855 Acc: 0.9955 Kappa: 0.8713 
    2021-08-11 21:09:13 [INFO]	[EVAL] Class IoU: 
    [0.9955 0.7755]
    2021-08-11 21:09:13 [INFO]	[EVAL] Class Acc: 
    [0.997  0.9098]
    2021-08-11 21:09:14 [INFO]	[EVAL] The model with the best validation mIoU (0.8855) was saved at iter 2000.
    <class 'paddle.nn.layer.conv.Conv2D'>'s flops has been counted
    Customize Function has been applied to <class 'paddle.nn.layer.norm.SyncBatchNorm'>
    Cannot find suitable count function for <class 'paddle.nn.layer.pooling.MaxPool2D'>. Treat it as zero FLOPs.
    <class 'paddle.nn.layer.pooling.AdaptiveAvgPool2D'>'s flops has been counted
    <class 'paddle.nn.layer.pooling.AvgPool2D'>'s flops has been counted
    Cannot find suitable count function for <class 'paddle.nn.layer.activation.Sigmoid'>. Treat it as zero FLOPs.
    <class 'paddle.nn.layer.common.Dropout'>'s flops has been counted
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.FP32, but right dtype is VarType.INT32, the right dtype will convert to VarType.FP32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    Total Flops: 8061050880     Total Params: 2328346



```python
# 单独进行评估 -- 上边do_eval就是这个工作
!python PaddleSeg/val.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams
# model_path： 模型参数路径
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-08-11 21:26:53 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 1000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: data/optic_disc_seg
      mode: train
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: OpticDiscSeg
    val_dataset:
      dataset_root: data/optic_disc_seg
      mode: val
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: OpticDiscSeg
    ------------------------------------------------
    W0811 21:26:53.681525  1599 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 21:26:53.681571  1599 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-08-11 21:27:06 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-11 21:27:08 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-11 21:27:08 [INFO]	Loaded trained params of model successfully
    2021-08-11 21:27:08 [INFO]	Start evaluating (total_samples: 76, total_iters: 76)...
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT32, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    76/76 [==============================] - 3s 33ms/step - batch_cost: 0.0329 - reader cost: 0.00
    2021-08-11 21:27:11 [INFO]	[EVAL] #Images: 76 mIoU: 0.8855 Acc: 0.9955 Kappa: 0.8713 
    2021-08-11 21:27:11 [INFO]	[EVAL] Class IoU: 
    [0.9955 0.7755]
    2021-08-11 21:27:11 [INFO]	[EVAL] Class Acc: 
    [0.997  0.9098]


- 5.开始预测


```python
# 进行预测
!python PaddleSeg/predict.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams\
--image_path segDataset/horse/Images\
--save_dir PaddleSeg/output/horse
# image_path: 预测图片路径/文件夹 -- 这里直接对训练数据进行预测，得到预测结果
# save_dir： 保存预测结果的路径 -- 保存的预测结果为图片
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-08-11 21:27:13 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 1000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: data/optic_disc_seg
      mode: train
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: OpticDiscSeg
    val_dataset:
      dataset_root: data/optic_disc_seg
      mode: val
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: OpticDiscSeg
    ------------------------------------------------
    W0811 21:27:13.663784  1705 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 21:27:13.663830  1705 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-08-11 21:27:27 [INFO]	Number of predict images = 328
    2021-08-11 21:27:27 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-11 21:27:29 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-11 21:27:29 [INFO]	Start to predict...
    328/328 [==============================] - 17s 51ms/st


# 五、可视化预测结果

通过PaddleSeg预测输出的结果为图片，对应位于:PaddleSeg/output/horse

其中包含两种结果：

- 一种为掩膜图像，即叠加预测结果与原始结果的图像 -- 位于: **PaddleSeg/output/horse/added_prediction**
- 另一种为预测结果的伪彩色图像，即预测的结果图像 -- 位于: **PaddleSeg/output/horse/pseudo_color_prediction**


```python
# 查看预测结果文件夹分布
!tree PaddleSeg/output/horse -L 1
```

    PaddleSeg/output/horse
    ├── added_prediction
    └── pseudo_color_prediction
    
    2 directories, 0 files




> 以下为展示结果

<font color='red' size=5> ---数据集 horse 的预测结果展示--- </font>

<font color='black' size=5> 掩膜图像： </font>

![](https://ai-studio-static-online.cdn.bcebos.com/1b0ca9c2f44d493fa5745031c700b39cd4ec5413682c429690f390c568e144b9)
![](https://ai-studio-static-online.cdn.bcebos.com/6d4f0073764e48cfa129f86013a5a5f9c64079152eab4cb3ae5dfcd73bbf6e52)


<font color='black' size=5> 伪彩色图像： </font>
![](https://ai-studio-static-online.cdn.bcebos.com/c1cda0c11a3d46e7899e999ccaa822f453808b547c214975abfb7afdfa598dc8)
![](https://ai-studio-static-online.cdn.bcebos.com/e54238bef7f84e38b52a6194aa366642553135b81c184401ad88367cab162054)






# 六、提交作业流程

1. 生成项目版本

![](https://ai-studio-static-online.cdn.bcebos.com/1c19ac6cfb314353b5377421c74bc5d777dcb5724fad47c1a096df758198f625)

2. (有能力的同学可以多捣鼓一下)根据需要可将notebook转为markdown，自行提交到github上

![](https://ai-studio-static-online.cdn.bcebos.com/0363ab3eb0da4242844cc8b918d38588bb17af73c2e14ecd92831b89db8e1c46)

3. (一定要先生成项目版本哦)公开项目

![](https://ai-studio-static-online.cdn.bcebos.com/8a6a2352f11c4f3e967bdd061f881efc5106827c55c445eabb060b882bf6c827)

# 七、寄语

<font size=4>

最后祝愿大家都能完成本次作业，圆满结业，拿到属于自己独一无二的结业证书——这一次训练营，将是你们AI之路成长的开始！

希望未来的你们能够坚持在自己热爱的方向上继续自己的梦想！

同时也期待你们能够在社区中创造更多有创意有价值基于飞桨系列的AI项目，发扬开源精神！

<br>

最后，再次祝愿大家都能顺利结业！
 </font>

『飞桨领航团AI达人创造营』大作业
作业说明

官方命题作业分成图像分类、图像分割、目标检测三个方向，每个方向有五个数据集。同学任选其中一个方向，任选其中一个数据集。

图像分类：[AI训练营]PaddleClas实现图像分类Baseline

图像分割：[AI训练营]PaddleSeg实现语义分割Baseline

目标检测：[AI训练营]PaddleX实现目标检测Baseline
请在此处附上您的项目链接

AI Studio 公开状态的项目链接：https://aistudio.baidu.com/aistudio/projectdetail/2275318?forkThirdPart=1

Github 或 Gitee 的项目链接：https://github.com/Extreme-lxh/AI-

