[[51 池化与经典 CNN(LeNet→AlexNet→VGG→ResNet)|池化]]把特征图的每个局部窗口压成一个数(取 max 或平均),实现**固定规则的下采样**与局部平移鲁棒。本篇只讲经典 CNN 的主线：**LeNet(1998)→AlexNet(2012)→VGG(2014)→ResNet(2015)**，以及池化和 GAP 在这条主线中的作用；后来的视觉架构只作为通往相关原子笔记的桥接，不在此做选型或排行榜式综述。

## 直觉

[[49 卷积运算与卷积核|卷积]]后特征图还很大,既费算力又对位置太敏感。**池化**做两件事:
- **下采样**:把空间尺寸缩小(常 2×2 步幅 2 → 减半),省算力、扩大后层 [[50 特征图、步幅、填充与感受野|感受野]]。
- **平移鲁棒**:**最大池化**取局部最大值,目标在窗口内小幅移动,max 不变 → 对小位移更不敏感(这是把卷积的「等变」推向「近似不变」的一步)。

池化**没有可学习参数**,就是个固定的聚合规则。两种最常见:
- **Max pooling**:取窗口最大值。保留「最强响应」(最显著特征),CNN 中间层主流。
- **Average pooling**:取窗口平均。更平滑;**全局平均池化(GAP)** 把整张特征图压成一个数,常用来替代末端的大全连接层(省参数、抗过拟合)。

**经典网络演进**就是这套零件越搭越深的历史(详见时间线图)。

## 例子

**4×4 特征图,2×2 池化、步幅 2 → 2×2 输出。** 切成 4 个不重叠 2×2 块:

$$X=\begin{bmatrix}1&3&5&2\\2&4&1&0\\0&6&8&3\\7&2&1&4\end{bmatrix}$$

四个块及其结果:
- 左上 $\{1,3,2,4\}$:**max=4**,avg=$\frac{1+3+2+4}{4}=2.5$
- 右上 $\{5,2,1,0\}$:**max=5**,avg=$2.0$
- 左下 $\{0,6,7,2\}$:**max=7**,avg=$3.75$
- 右下 $\{8,3,1,4\}$:**max=8**,avg=$4.0$

$$\text{MaxPool}=\begin{bmatrix}4&5\\7&8\end{bmatrix},\qquad
\text{AvgPool}=\begin{bmatrix}2.5&2.0\\3.75&4.0\end{bmatrix}$$

注意 16 个数 → 4 个数,**空间缩到 1/4**,且没引入任何权重。

**全局平均池化(GAP)手算**。同一张 $4\times4$ 特征图 $X$,GAP 把**整张图压成一个数** = 全部 16 个元素的均值:$\frac{1+3+5+2+2+4+1+0+0+6+8+3+7+2+1+4}{16}=\frac{49}{16}\approx3.06$。若有 512 张特征图,GAP 输出就是 512 维向量(每图一个数),直接接一个 $512\to$ 类别数 的小线性层即可——这就是「GAP 替代末端大全连接」省掉上亿参数的原理。

**最大池化反向传播手算**。左上块 $\{1,3,2,4\}$ max=4(在右下角位置)。若上游传回该块的梯度是 0.7,则梯度**全部给 argmax 那个位置(值 4 的格子)**,其余三个格子梯度为 0:回传 $\begin{bmatrix}0&0\\0&0.7\end{bmatrix}$。平均池化则把 0.7 **均分**给四格:每格 $0.7/4=0.175$。这解释了为何 max 池化的梯度稀疏(只有「赢家」通道学习)、avg 池化的梯度平滑。

**GAP 省了多少参数(VGG 痛点手算)**。VGG 末端特征图 $7\times7\times512$,接两个 4096 维全连接:
- 第一层全连接:$(7\times7\times512)\times4096=25088\times4096\approx1.03\times10^8$ 个权重——**单这一层就一亿参数**,占 VGG-16 总参数(约 1.38 亿)的大头。
- 换成 GAP:$7\times7\times512\to512$(每张图取均值,**0 参数**),再接 $512\to1000$ 分类头仅 $51.2$ 万参数。**省掉约 200 倍**。
这就是 NIN/ResNet 用 GAP 替代末端全连接的直接收益:大幅减参 + 抗过拟合 + 输入尺寸灵活。

![[cnn-池化下采样.png]]

## 原理

**最大池化的反向传播。** 前向 $y=\max(x_1,\dots,x_n)$。梯度只回传给「取到最大的那个位置」,其余为 0(类似 [[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|ReLU]] 的稀疏梯度);需缓存 argmax 索引。**平均池化**则把上游梯度均分给窗口内每个元素($\frac1n$)。

**经典 CNN 主线。** LeNet 是手写数字任务的早期卷积网络；AlexNet、VGG 和 ResNet 才是 ImageNet/ILSVRC 语境下连续可比较的关键节点。因此，下面的具体 ImageNet 数字只用于说明后三者各自论文报告的实验，不把 LeNet 与它们混作同一基准上的排行榜。

- **LeNet-5(LeCun, 1998)**:早期现代 CNN 代表。约 7 层「卷积+下采样(子采样)+全连接」、约 6 万参数，面向 MNIST 手写数字；它确立了「conv→下采样→分类」的基本范式。

- **AlexNet(Krizhevsky, 2012)**:**深度学习引爆点**。8 层(5 conv + 3 FC)、~6000 万参数,ILSVRC-2012 top-5 错误从上一年的 ~26% 砍到 ~16%,震动全场。关键工程:**[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|ReLU]]**(替代 sigmoid,训练快几倍)、**[[42 正则化(L2、Dropout、早停、标签平滑)|Dropout]]**、数据增强、双 GPU 训练。

- **VGG-16/19(Simonyan & Zisserman, 2014)**:把网络做**规整且更深**(16~19 层),核心信条:**全部用 3×3 卷积堆叠**。两层 3×3 = 一层 5×5 的 [[50 特征图、步幅、填充与感受野|感受野]],但参数更少、非线性更多。代价是参数巨大(~1.38 亿,主要在末端全连接),论文在其训练设置下报告 top-5 错误约 7%。它说明「在规整的小核设计、数据和训练配方配合时，增加深度可以有效」，并不单独证明深度必然带来收益。

- **ResNet(He et al., 2015)**:把残差块扩展到 **152 层**（亦有 18/34/50/101 层配置）；论文报告其 ILSVRC-2015 集成 top-5 错误为 **3.57%**，并指出 152 层配置的计算量低于 VGG-19。破局点是 **[[52 残差连接与深度可训练性|残差连接]]** $y=F(x)+x$，用恒等路径缓解「网络加深但训练误差反而变差」的退化问题。

**向后只作桥接，不扩展成架构大全。** 深网训练中的归一化见 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]]；可分离卷积等效率算子回到 [[49 卷积运算与卷积核|卷积]]；残差及其可训练性回到 [[52 残差连接与深度可训练性|残差连接]]；视觉 Transformer 的注意力谱系从 [[60 注意力机制的起源(Bahdanau、Luong)|注意力起源]] 进入。DenseNet、MobileNet、EfficientNet、ConvNeXt、ViT 分别解决特征复用、算力约束或视觉建模范式问题，但不宜仅凭年份或单一基准分数给出全局优劣结论。

**主线中的可比规律(见时间线图):** 从 AlexNet 到 VGG 再到 ResNet，设计重心依次是可训练的深网络、规整的小卷积核堆叠、以及残差连接；它们的论文实验使用的模型、训练配方与集成设置并不相同，不能把 $16\%\to7\%\to3.57\%$ 简化为「层数单独造成的收益」。池化/GAP 是这条线中的空间聚合手段；是否改用 stride 卷积取决于任务、特征图尺寸与计算预算。

![[cnn-经典网络时间线.png]]

## 代码

```python
import numpy as np

def max_pool_2x2(X):
    H, W = X.shape
    out = np.zeros((H//2, W//2))
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            out[i//2, j//2] = X[i:i+2, j:j+2].max()   # 取块内最大
    return out

X = np.array([[1,3,5,2],[2,4,1,0],[0,6,8,3],[7,2,1,4]], float)
print(max_pool_2x2(X))     # [[4,5],[7,8]]  ← 与手算一致 ✅

# ❌ 误区:末端用巨大全连接(VGG 那样),参数爆炸 + 易过拟合
#    7*7*512 -> 4096 全连接 = ~1 亿参数,占全网大头
import torch, torch.nn as nn
feat = torch.randn(1, 512, 7, 7)
fc = nn.Linear(512*7*7, 4096)                 # 102,764,544 个参数
print("FC 参数:", sum(p.numel() for p in fc.parameters()))

# ✅ 全局平均池化(GAP):整张特征图 -> 一个数,0 参数,直接接小分类头
gap = nn.AdaptiveAvgPool2d(1)                  # (1,512,7,7) -> (1,512,1,1)
print("GAP 后:", gap(feat).shape)             # [1,512,1,1],无参数 ✅

# ✅ 现成经典网络
from torchvision import models
resnet = models.resnet18(weights=None)         # 18 层残差网,内部已用 GAP
print(type(resnet).__name__)
```

## 面试高频

- **「池化的作用?max 和 avg 区别?」** 下采样(省算力、扩大感受野)+ 局部平移鲁棒,且**无参数**。max 取最强响应(保显著特征,中间层主流);avg 取平均(更平滑,GAP 常替代末端全连接)。
- **「最大池化怎么反向传播?」** 梯度只回传给 argmax 那个位置,其余为 0(需缓存索引);平均池化把梯度均分。
- **「能按时间线讲讲经典 CNN 吗?」** LeNet(1998,确立 conv-pool-FC)→ AlexNet(2012,ReLU+Dropout+GPU,引爆深度学习)→ VGG(2014,全 3×3、做深)→ ResNet(2015,残差连接,上百层)。可补充：AlexNet、VGG、ResNet 各篇论文在各自模型、训练配方和是否集成的条件下报告了约 16%、7%、3.57% 的 top-5 错误；这些数字不是同一控制实验，不能归因成「层数越深，误差必然按此下降」。
- **「VGG 为什么全用 3×3?」** 堆两层 3×3 ≈ 一层 5×5 感受野,但参数更少、非线性更多、更规整。
- **「AlexNet 相比 LeNet 的关键改进?」** ReLU(缓解梯度消失、训练快)、Dropout(正则)、数据增强、GPU 训练、更大更深;加上 ImageNet 大数据。
- **「GAP(全局平均池化)解决什么?」** 把末端巨大全连接换成无参数聚合,大幅减参、抗过拟合,还让输入尺寸更灵活。
- **「现在还用池化吗?」** 仍用,但很多网络改用 stride2 卷积做下采样(可学习);GAP 几乎是标准末端。
- **「按时间线讲经典 CNN 主线?」** LeNet(1998，卷积+下采样范式)→AlexNet(2012，ReLU、Dropout、GPU 训练)→VGG(2014，全 $3\times3$ 堆叠)→ResNet(2015，残差连接)。要延伸到 DenseNet、MobileNet、EfficientNet、ConvNeXt 或 ViT 时，先说明比较维度（特征复用、端侧效率、训练配方或视觉范式），不要把它们当成这条线的单一排名。
- **「池化和 stride 卷积哪个会丢信息?」** 都做有损下采样;池化是固定规则(max 只留最强、avg 抹平),stride 卷积可学但同样丢空间细节。要保分辨率又扩感受野用空洞卷积。
- **「为什么 GAP 让输入尺寸更灵活?」** GAP 把任意 $H\times W$ 压成 $1\times1$,后接的全连接只依赖通道数 $C$、与空间尺寸无关,所以网络能吃不同分辨率输入。

## 关键事实

- **LeNet-5**:LeCun, Bottou, Bengio & Haffner, *Gradient-Based Learning Applied to Document Recognition*(Proc. IEEE, 1998)。
- **AlexNet**:Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep CNNs*(NeurIPS 2012),ILSVRC-2012 top-5 错误约 16.4%(集成 15.3%),引入 ReLU、Dropout、双 GPU。
- **VGG**:Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*(ICLR 2015,arXiv:1409.1556, 2014),全 3×3 卷积,16/19 层,~1.38 亿参数。
- **ResNet**:He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*(CVPR 2016,arXiv:1512.03385, 2015),最深 152 层,ILSVRC-2015 集成 top-5 错误 **3.57%**,FLOPs 低于 VGG。
- **全局平均池化**:Lin, Chen & Yan, *Network In Network*(ICLR 2014),提出用 GAP 替代末端全连接。
- **后续架构的定位**：DenseNet（Huang et al., CVPR 2017）、MobileNet（Howard et al., 2017）、EfficientNet（Tan & Le, ICML 2019）、ConvNeXt（Liu et al., CVPR 2022）和 ViT（Dosovitskiy et al., ICLR 2021）是延伸阅读的原始论文；本篇不比较其跨论文指标或宣称统一的当前最优。
