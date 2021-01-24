# ⚕️Covid-19 Medical Image Recognition

This repository is about the course project for the digital image processing(DIP) course in the School of Software (Media Arts and Sciences Direction) of Tongji University in the fall and winter semester of the 2020-2021. This project ranked first 🏆  in the grade with **92.45%** accuracy rate and achieved a perfect score💯 in the defense. The instructor of the course is Prof. Qingjiang Shi, and the project members are Jiajie Li, Lai Ye, Kaibin Zhou and Xiaoyu Jia (in order by student number)

本仓库是同济大学软件学院（18级数字媒体方向）数字图像处理的课程项目。本项目以 **92.45%** 的准确率排名年级第一 🏆 ，并在答辩中取得了满分💯的成绩。本课程的指导教师是史清江教授，项目成员是李嘉杰、叶莱、周楷彬、贾小玉（按学号排序）。



## ⚛️Model Architecture 模型架构
更清晰的版本请参考 Report 的附录
For a clearer version, please refer to the appendix of the REPORT

### 🔭Overview 总览

![overview1](https://github.com/LeeJAJA/DIP_Final_Project/blob/main/img/overview1.png)

#### Self-Trans

![overview3](https://github.com/LeeJAJA/DIP_Final_Project/blob/main/img/overview3.png)

#### COV-CSR

![overview1](https://github.com/LeeJAJA/DIP_Final_Project/blob/main/img/overview2.png)

### 🧠Methods 用到的技术

| METHOD                                                       | Year | TYPE                    |
| ------------------------------------------------------------ | ---- | ----------------------- |
| [1x1 Convolution](https://paperswithcode.com/method/1x1-convolution) | 2013 | Convolutions            |
| [Convolution](https://paperswithcode.com/method/convolution) | 1980 | Convolutions            |
| [Dilated Convolution](https://paperswithcode.com/method/dilated-convolution) | 2015 | Convolutions            |
| [1D CNN](https://paperswithcode.com/method/1d-cnn)           | 2020 | Convolutions            |
| [Max Pooling](https://paperswithcode.com/method/max-pooling) | 2000 | Pooling Operations      |
| [Average Pooling](https://paperswithcode.com/method/average-pooling) | 2000 | Pooling Operations      |
| [Spatial Pyramid Pooling](https://paperswithcode.com/method/spatial-pyramid-pooling) | 2014 | Convolutions            |
| [Pointwise Convolution](https://paperswithcode.com/method/pointwise-convolution) | 2016 | Convolutions            |
| [Depthwise Convolution](https://paperswithcode.com/method/depthwise-convolution) | 2016 | Convolutions            |
| [ReLU](https://paperswithcode.com/method/relu)               | 2000 | Activation Functions    |
| [Adam](https://paperswithcode.com/method/adam)               | 2014 | Stochastic Optimization |
| [Residual Block](https://paperswithcode.com/method/residual-block) | 2015 | Image Model Blocks      |
| [Dense Block](https://paperswithcode.com/method/dense-block) | 2016 | Image Model Blocks      |
| [Bottleneck Residual Block](https://paperswithcode.com/method/bottleneck-residual-block) | 2015 | Image Model Blocks      |
| [Squeeze-and-Excitation Block](https://paperswithcode.com/method/squeeze-and-excitation-block) | 2017 | Image Model Blocks      |
| [Spatial Attention Module](https://paperswithcode.com/method/spatial-attention-module) | 2018 | Image Model Blocks      |
| [CBAM](https://paperswithcode.com/method/cbam)               | 2018 | Image Model Blocks      |
| [Dropout](https://paperswithcode.com/method/dropout)        | 2014 | Regularization          |
| [Weight Decay](https://paperswithcode.com/method/weight-decay) | 1943 | Regularization          |
| [ResNet](https://paperswithcode.com/method/resnet)           | 2015 | Image Models            |
| [VGG](https://paperswithcode.com/method/vgg)                | 2014 | Image Models            |
| [DenseNet](https://paperswithcode.com/method/densenet)       | 2016 | Image Models            |
| [SENet](https://paperswithcode.com/method/senet)             | 2017 | Image Models            |
| [Step Decay](https://paperswithcode.com/method/step-decay)   | 2000 | Learning Rate Schedules |
| [RPN](https://paperswithcode.com/method/rpn)                 | 2015 | Region Proposal         |
| [RoIPool](https://paperswithcode.com/method/roi-pooling)    | 2013 | RoI Feature Extractors  |
| [RoIAlign](https://paperswithcode.com/method/roi-align)      | 2017 | RoI Feature Extractors  |
| [Dense Connections](https://paperswithcode.com/method/dense-connections) | 2000 | Feedforward Networks    |
| [Softmax](https://paperswithcode.com/method/softmax)         | 2000 | Output Functions        |
| [Heatmap](https://paperswithcode.com/method/heatmap)         | 2014 | Output Functions        |
| [k-Means Clustering](https://paperswithcode.com/method/k-means-clustering) | 2000 | Clustering              |
| [Focal Loss](https://paperswithcode.com/method/focal-loss)   | 2017 | Loss Functions          |



### 💽Pretrained Models 预训练模型

| MODEL   | Description                         | LINK                                     |
| ------- | ----------------------------------- | ---------------------------------------- |
| COV-CSR | gate(high contrast part)+focal loss | https://cowtransfer.com/s/9804270e970b47 |
| COV-CSR | gate(high contrast part)            | https://cowtransfer.com/s/93c3e6af339c4e |
| COV-CSR | HE processed                        | https://cowtransfer.com/s/7f61a15f1f2743 |
| COV-CSR | focal loss                          | https://cowtransfer.com/s/e7164e06ec7743 |



## Technology 技术选型
- **前端**

  - React
  - Ant Design
  - Uppy
  - Tus Client

- **后端**

  - Go(Tus Server) 

  - Flask

    


##  👨🏻‍🔧Contribution 团队分工情况

- **Jiajie Li 李嘉杰**

  - 模型

    1. 复现 COVnet，在 COVNet 的基础上上实现 SPP 和 FCN，在 COVNet 上比较 ResNet18，ResNet34，ResNet50 作为 backbone 的结果，将 COVNet 的 backbone 更改为魔改了的 Vgg 并引入空洞卷积。
       Reproduced COVnet, implement SPP and FCN on the basis of COVNet, compare the results of ResNet18, ResNet34, ResNet50 as backbone on COVNet, change the backbone of COVNet to the magic modified Vgg and introduce the hole convolution.
    2. 提出一二级分类器的思路，并使用 Sklearn 实现二级分类器。
       Proposed the idea of first and second level classifier and implemented the two level classifier using Sklearn.
    3. 发现原始数据集中的噪声并提出了清晰数据集噪声的思路。
       Found the noise in the original dataset and proposed the idea of clarifying the noise in the dataset.
    4. 提出了使用 OTDD 评估数据集分布之间的距离，用以衡量同数据集下 label 之间的距离以及不同数据集之间的距离，并以此为依据引导模型训练过程中的预训练权重选择和数据增广。
       Proposed to use OTDD to evaluate the distance between dataset distributions to measure the distance between labels under the same dataset and between different datasets, and use it to guide the pre-training weight selection and data augmentation during model training.
    5. 提出了使用 gate 的思路，并提出分辨两类数据集的方法。
       Proposed the idea of using gate and proposed a method to distinguish between two types of datasets.
    6. 完成了图像增强的工作，对HE，CLAHE，LACE 和 HE+CLAHE 四种增强方式进行实验对比，并将其应用在整个数据集上。
       Completed the work on image enhancement by experimentally comparing four types of enhancement, HE, CLAHE, LACE and HE+CLAHE, and applied the image enhancement to the whole dataset.
    7. 尝试解决类别不均衡，在端到端模型中实现了 Focal Loss，在二级分类器中实现了 Smote + ENN。
       Tried to solve the category imbalance by implementing Focal Loss in the end-to-end model and Smote + ENN in the secondary classifier.
    8. 尝试解决了序列不等长的问题，实现了二级分类器中的上采样，并对四种插值方式（nearest，linear，quadratic，cubic）进行了对比。
       Tried to solve the problem of unequal length of sequences, implemented upsampling in the secondary classifier, and compared four interpolation methods (nearest, linear, quadratic, cubic).
    9. 实现了二级分类器，使用 Xgboost 和 SVM 进行了集成学习（软投票）。
       Implemented a secondary classifier with integrated learning (soft voting) using Xgboost and SVM.
    10. 使用 GradCAM 对模型进行可解释性分析。
        utilized GradCAM to perform interpretable analysis of the model.
    11. 在数据集上使用语义分割进行 ROI 预处理，对比了 Unet 和传统的分割方式的效果，并使用和 基于阈值和形态学操作（膨胀和腐蚀）的方法进行了肺部区域的分割。
        Used semantic segmentation for ROI preprocessing on the dataset, comparing the effects of Unet and traditional segmentation approaches, and performed segmentation of lung regions using and methods based on thresholding and morphological operations (inflation and erosion).
    12. 对网络结构进行了可视化。
        Visualized the network structure.
    13. 实现了图像数据集的划分。
        Implemented the partitioning of image datasets.

  - 前端

    1. 使用 React + Ant Design 实现可交互的前端。
       Used React + Ant Design to implement an interactive front-end.
    2. 使用 Uppy + tus 实现了前端的断点续传组件。
       Used Uppy + tus to implement a front-end breakpoint component.
    3. 完成服务器的部署。
       Completes the deployment of the server.

  - 后端

    1. 使用 Go 编写的后端作为 tus 断点续传的 Server 端。
       Used a backend written in Go as the Server side of the tus breakpoint transfer.
    2. 使用 Flask 编写的后端作为模型的部署。
       Used a backend written in Flask for the deployment of the model.
    3. 完成服务器的部署。
       Completed the deployment of the server.

  - 答辩

    1. 准备答辩材料。
       Prepared the defense materials.
    2. 作为答辩人完成答辩。
       Completed his defense as a respondent.

    

- **Lai Ye 叶莱**

  - 模型

    1. 完成了探索性数据分析（EDA）。
       Completed Exploratory Data Analysis (EDA).
    2. 使用 ImageHash 实现了图像数据集中噪声的清洗。
       Implemented noise cleaning in image dataset using ImageHash.
    3. 实现了图像数据集的划分。
       Implemented the partitioning of image datasets.

  - 答辩

    1. 准备答辩材料。
       Prepared the defense materials.

    

- **Kaibin Zhou 周楷彬**

  - 模型

    1. 复现了 《Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans 》的工作，并将其组作为一级分类器的主要模型，并在其中尝试了注意力机制（CBAM）和伪标签训练。
       Reproduced the work of "Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans" and used its group as the main model for the first level classifier in which attention mechanism (CBAM) and pseudo-label training were tried.
    2. 实现了 gate 中分辨两类图片的算法。
       Implemented the algorithm for discriminating between two classes of images in gate module.
    3. 实现了图像序列层面的下采样。
       Implemented downsampling at the image sequence level.
    4. 实现了 OTDD 的算法，并在各种数据集上进行测试和比较。
       Implemented the algorithm for OTDD and tested and compared it on various datasets.
    5. 实现了图像数据集的划分。
       Implemented the partitioning of image datasets.

  - 答辩

    1. 准备答辩材料。
       Prepared the defense materials.

    

- **Xiaoyu Jia 贾小玉**

  - 模型
    1. 实现了模型中间层的可视化。
       Implemented the visualization of the middle layer of the model.
    2. 撰写了报告
       Typeset the report
    3. 尝试复现了分割算法
       Tried to reproduce the segmentation algorithm
  - 前端
    1. 使用 React + Ant Design 实现可交互的前端。
       Used React + Ant Design to implement an interactive front-end.

  - 答辩
    1. 准备答辩材料。
       Prepared the defense materials.
