# 样本不足条件下的人脸识别问题

样本不足条件下的人脸识别问题，要研究这个问题，首先应当定义什么是“样本不足”。

> Although recently Sparse Representation-Based Classification (SRC) has represented a breakthrough in the field of face recognition due to its good performance and robustness, there is the critical problem that SRC needs sufficiently large training samples to achieve good performance. [^Sensors]

从上文可知，当今在人脸识别领域已经出现了很多表现很好鲁棒性也很强的算法，但这一类算法往往需要相当大量的训练集（MSPP, multiple samples per person, 多张图片的label都打相同）。而样本不足情况下，即**只有少数几张图片打了相同的label，small sample size (SSS); 甚至一个label只有一张图片,single sample per person (SSPP)**的情况。

SSPP的情况其实很普遍，例如在**身份证，护照，校园卡**等等。这种情况下一个人都只能拿到一张图片，训练样本的不足，导致传统的人脸识别没法构建出完整的特征空间。

[TOC]

## 常用方法
我们要克服的是样本不足的现状，很自然的我们会想从两个方面入手。

第一，既然单人样本不够，那么我们想着怎么通过算法**从一张图片生成多张图片**

- 使用透视规律做DA（the perspective of domain adaptation），生成同一个人不同角度的多张图片，从而达到克服样本不足的问题。 例如 SSPP-DAN: DEEP DOMAIN ADAPTATION NETWORK FOR FACE RECOGNITION WITH SINGLE SAMPLE PER PERSON[^arXiv]

第二，**修改算法流程或结构**使其适应样本不足的情况

- 使用集成学习，类似于boosting, bagging, random forest等方法。核心思想是**使用各种不同的分类器提供互补而全面的信息供以分类，尽量从一张图片中获取更多特征**，如Random Quad-Tree based Ensemble Algorithm[^Quad-Tree]

## 所用方法的具体描述
> 暂定选用【SSPP-DAN: DEEP DOMAIN ADAPTATION NETWORK FOR FACE RECOGNITION WITH SINGLE SAMPLE PER PERSON[^arXiv]】这篇论文提到的使用DA透视规律生成单人多张图片的方法。（还没码代码，所以只能做暂定，后面视算法表现力再做下一步决定）

![](https://i.loli.net/2017/10/11/59dcf08af3241.jpg)

大致流程如下

- 使用DA做Image synthesis通过对原始图片的3D脸部建模生成不同姿势，角度的多张图片，增加作为输入的样本数，克服样本不足的问题。
- 使用Feature extractor和两个classifier连接训练集（stable image，通常清晰，背景简单，光线好）和测试集（unstable image，通常清晰度不高，背景复杂，较为随便）。
- 然后使用Gradient reversal layer训练模型。

<!-- more -->
## References

[^Sensors]: [Jun Cai, Jing Chen * and Xing Liang, Single-Sample Face Recognition Based on Intra-Class Differences in a Variation Model, *Sensors*, 2015, 15, 1071-1087](http://www.mdpi.com/1424-8220/15/1/1071/pdf)

[^Quad-Tree]: [Cuicui Zhang, Xuefeng Liang, Takashi Matsuyama, Small Sample Size Face Recognition using Random Quad-Tree based Ensemble Algorithm, Kyoto University, Kyoto 606-8501]()

[^arXiv]: [Sungeun Hong, Woobin Im, Jongbin Ryu, Hyun S. Yang, SSPP-DAN: DEEP DOMAIN ADAPTATION NETWORK FOR FACE RECOGNITION WITH SINGLE SAMPLE PER PERSO, *arXiv:1702.04069*](https://arxiv.org/abs/1702.04069)

https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

