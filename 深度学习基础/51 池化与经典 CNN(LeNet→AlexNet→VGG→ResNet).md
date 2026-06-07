[[51 池化与经典 CNN(LeNet→AlexNet→VGG→ResNet)|池化]]把特征图的每个不重叠小块压成一个数(取 max 或平均),实现**无参数下采样**+局部平移鲁棒;沿着「conv→池化→全连接」这条范式,经典 CNN 一路从 **LeNet(1998)→ AlexNet(2012)→ VGG(2014)→ ResNet(2015)** 越堆越深,把 ImageNet 错误率从两位数压到 3.57%。

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

**经典 CNN 演进(均为 ImageNet/ILSVRC 关键节点)。**

- **LeNet-5(LeCun, 1998)**:第一套现代 CNN。约 7 层「卷积+下采样(子采样)+全连接」,~6 万参数,做 MNIST 手写数字。奠定了「conv→pool→FC」范式,但受限于算力与数据,沉寂多年。

- **AlexNet(Krizhevsky, 2012)**:**深度学习引爆点**。8 层(5 conv + 3 FC)、~6000 万参数,ILSVRC-2012 top-5 错误从上一年的 ~26% 砍到 ~16%,震动全场。关键工程:**[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|ReLU]]**(替代 sigmoid,训练快几倍)、**[[42 正则化(L2、Dropout、早停、标签平滑)|Dropout]]**、数据增强、双 GPU 训练。

- **VGG-16/19(Simonyan & Zisserman, 2014)**:把网络做**规整且更深**(16~19 层),核心信条:**全部用 3×3 卷积堆叠**。两层 3×3 = 一层 5×5 的 [[50 特征图、步幅、填充与感受野|感受野]],但参数更少、非线性更多。代价是参数巨大(~1.38 亿,主要在末端全连接),top-5 错误 ~7%。证明了「深度本身」是涨点的关键变量。

- **ResNet(He et al., 2015)**:深度跨越式突破。直接堆到 **152 层**(也有 18/34/50/101 版),ILSVRC-2015 冠军,集成 top-5 错误 **3.57%**(超过人类 ~5%),而 152 层的 FLOPs **还低于 VGG**。破局点是 **[[52 残差连接与深度可训练性|残差连接]]** $y=F(x)+x$,解决了「网络一深反而变差」的退化问题(下一篇详述)。

**继续往后的演进(补全脉络)**:
- **Inception / GoogLeNet(Szegedy, 2014)**:同层并联多种核大小($1{\times}1,3{\times}3,5{\times}5$)+ $1{\times}1$ 降维,用 GAP 替代全连接,22 层却比 AlexNet 参数少 12 倍,top-5 ~6.7%。提出「多尺度 + 瓶颈」思路。
- **BatchNorm(Ioffe, 2015)**:不是网络而是组件,但它让深网训练大幅提速、可用更大学习率,是 ResNet 能稳训上百层的关键拼图(见 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|BatchNorm]])。
- **DenseNet(Huang, 2017)**:每层都连到后面所有层(密集跳连),特征复用、参数更省。
- **MobileNet / EfficientNet(2017–2019)**:用深度可分离卷积(见 [[49 卷积运算与卷积核|卷积]])和复合缩放,在移动端/有限算力下取得最优精度-效率折中。
- **ConvNeXt(2022)**:用 Transformer 时代的训练技巧重新武装纯 CNN,证明 CNN 仍可与 ViT 抗衡;同时 **Vision Transformer(ViT, 2020)** 把注意力搬进视觉,开启 CNN 之外的新路线(见 [[60 注意力机制的起源(Bahdanau、Luong)|注意力起源]] → Transformer)。

**纵向规律(见时间线图):** 层数 $7\to8\to19\to152$,错误率 $26\%\to16\%\to7\%\to3.57\%$。趋势是「更深 + 更小核 + 更聪明的连接(残差)」。现代趋势进一步用 **GAP 替代末端全连接**、用 **stride 卷积替代部分池化**、用 **BN 稳训练**、用**深度可分离卷积省算力**。

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
- **「能按时间线讲讲经典 CNN 吗?」** LeNet(1998,确立 conv-pool-FC)→ AlexNet(2012,ReLU+Dropout+GPU,引爆深度学习)→ VGG(2014,全 3×3、做深)→ ResNet(2015,残差连接,上百层)。错误率 26→16→7→3.57%。
- **「VGG 为什么全用 3×3?」** 堆两层 3×3 ≈ 一层 5×5 感受野,但参数更少、非线性更多、更规整。
- **「AlexNet 相比 LeNet 的关键改进?」** ReLU(缓解梯度消失、训练快)、Dropout(正则)、数据增强、GPU 训练、更大更深;加上 ImageNet 大数据。
- **「GAP(全局平均池化)解决什么?」** 把末端巨大全连接换成无参数聚合,大幅减参、抗过拟合,还让输入尺寸更灵活。
- **「现在还用池化吗?」** 仍用,但很多网络改用 stride2 卷积做下采样(可学习);GAP 几乎是标准末端。
- **「按时间线把演进讲全?」** LeNet(1998)→ AlexNet(2012, ReLU+Dropout+GPU)→ VGG(2014, 全3×3)→ GoogLeNet/Inception(2014, 多尺度+1×1降维+GAP)→ ResNet(2015, 残差)→ DenseNet(2017, 密集跳连)→ MobileNet/EfficientNet(2017–19, 高效)→ ConvNeXt/ViT(2020–22, 与 Transformer 融合/竞争)。
- **「Inception 的核心创新?」** 同层并联多种核大小做多尺度感知,$1{\times}1$ 卷积降维省算力,GAP 替代全连接;比 AlexNet 更深却参数更少。
- **「池化和 stride 卷积哪个会丢信息?」** 都做有损下采样;池化是固定规则(max 只留最强、avg 抹平),stride 卷积可学但同样丢空间细节。要保分辨率又扩感受野用空洞卷积。
- **「为什么 GAP 让输入尺寸更灵活?」** GAP 把任意 $H\times W$ 压成 $1\times1$,后接的全连接只依赖通道数 $C$、与空间尺寸无关,所以网络能吃不同分辨率输入。

## 关键事实

- **LeNet-5**:LeCun, Bottou, Bengio & Haffner, *Gradient-Based Learning Applied to Document Recognition*(Proc. IEEE, 1998)。
- **AlexNet**:Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep CNNs*(NeurIPS 2012),ILSVRC-2012 top-5 错误约 16.4%(集成 15.3%),引入 ReLU、Dropout、双 GPU。
- **VGG**:Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*(ICLR 2015,arXiv:1409.1556, 2014),全 3×3 卷积,16/19 层,~1.38 亿参数。
- **ResNet**:He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*(CVPR 2016,arXiv:1512.03385, 2015),最深 152 层,ILSVRC-2015 集成 top-5 错误 **3.57%**,FLOPs 低于 VGG。
- **全局平均池化**:Lin, Chen & Yan, *Network In Network*(ICLR 2014),提出用 GAP 替代末端全连接。
- **Inception / GoogLeNet**:Szegedy et al., *Going Deeper with Convolutions*(CVPR 2015),22 层、多尺度并联、$1{\times}1$ 降维,ILSVRC-2014 冠军 top-5 ~6.7%。
- **DenseNet**:Huang et al.(CVPR 2017);**MobileNet**:Howard et al.(2017);**EfficientNet**:Tan & Le(ICML 2019,复合缩放)。
- **ConvNeXt**:Liu et al.(CVPR 2022);**ViT**:Dosovitskiy et al., *An Image Is Worth 16×16 Words*(ICLR 2021)——视觉 Transformer,挑战 CNN 主导地位。
