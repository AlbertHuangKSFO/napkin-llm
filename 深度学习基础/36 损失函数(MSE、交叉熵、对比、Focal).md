[[36 损失函数(MSE、交叉熵、对比、Focal)|损失函数]]是一把"尺子",把模型当前的预测和真实答案之间的差距量化成一个标量;训练就是 [[37 梯度下降：BGD、SGD、Mini-batch|梯度下降]] 不断把这个标量往小压。选错损失,模型就在优化"错的东西"。

## 直觉

模型每次预测完,我们要回答一个问题:**这次错得有多离谱?** 损失函数就是这个"离谱程度"的打分器。

- 它必须是**一个数**(标量),因为只有标量才能求梯度、才能比较"这一步比上一步好还是坏"。
- 它必须**可导**(几乎处处),否则 [[20 反向传播的数学推导|反向传播]] 没法把梯度传回去。
- 不同任务要用不同的尺子:**回归**(预测房价)关心"差了多少",用 **MSE**;**分类**(猫还是狗)关心"概率对不对",用 **交叉熵**;**检索/对比学习**关心"相似的近、不相似的远",用**对比损失**;**类别极度不平衡**(99% 背景 1% 目标)时交叉熵会被海量简单样本淹没,用 **Focal Loss** 把注意力压到难样本上。

一句话:**损失函数 = 任务目标的数学翻译**。它决定了"对模型而言什么叫好"。

## 例子

**MSE(均方误差)**。真值 $y=3$,预测 $\hat y=5$,单样本损失 $(\hat y-y)^2=(5-3)^2=4$。误差翻倍($\hat y=7$)损失变 $16$ —— **平方让大错误被加倍惩罚**,这也是 MSE 对离群点敏感的原因。

**交叉熵(二分类)**。真标签 $y=1$,模型给正类概率 $p=0.9$:损失 $=-\ln 0.9\approx 0.105$(预测很准,损失小)。若 $p=0.1$(预测严重错):$-\ln 0.1\approx 2.303$(损失大 22 倍)。**交叉熵对"自信地错"惩罚极重**——这正是我们想要的。

**Focal Loss**。同样 $y=1$。一个简单样本 $p=0.9$,$\gamma=2$:调制因子 $(1-0.9)^2=0.01$,损失被压到原来的 $1/100$;一个难样本 $p=0.2$:$(1-0.2)^2=0.64$,损失几乎原样保留。**简单样本被"调小音量",训练自动聚焦难样本**。

**MAE vs MSE vs Huber(对离群点的鲁棒性,手算对比)**。真值 $y=3$,看一个离群预测 $\hat y=13$(误差 $r=10$):
- MSE:$r^2=100$(被平方放大,单个离群点就主导损失)。
- MAE(L1):$|r|=10$(线性,不放大)。
- Huber($\delta=1$):误差大于 $\delta$ 时走线性段 $\delta(|r|-\tfrac12\delta)=1\times(10-0.5)=9.5$(几乎和 MAE 一样不被放大)。
而小误差 $r=0.5$ 时:MSE $=0.25$,Huber $=\tfrac12 r^2=0.125$(走二次段,平滑可导)。**Huber = 小误差像 MSE(平滑)、大误差像 MAE(抗离群)**,$\delta$ 是切换阈值(δ≈1.345 对应正态下 5% 离群的稳健选择)。PyTorch 的 `SmoothL1Loss` 就是 $\delta=1$ 的 Huber。

![[nn-损失函数对比.png]]

## 原理

**MSE**。$N$ 个样本,
$$\mathcal L_{\text{MSE}}=\frac1N\sum_{i=1}^N (\hat y_i-y_i)^2$$
它来自高斯噪声假设下的 [[25 最大似然估计 MLE|最大似然]]:若 $y=\hat y+\varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma^2)$,最大化似然 $\iff$ 最小化 MSE。对单样本求梯度 $\partial\mathcal L/\partial\hat y=2(\hat y-y)$,**梯度正比于误差**,简单干净。MSE 关于 $\hat y$ 是凸的(二次),线性回归下有闭式解(正规方程)。

**MAE / L1 与 Huber(回归鲁棒三件套)**。
$$\mathcal L_{\text{MAE}}=\frac1N\sum|\hat y_i-y_i|,\qquad
\mathcal L_{\text{Huber}}=\begin{cases}\tfrac12 r^2&|r|\le\delta\\\delta(|r|-\tfrac12\delta)&|r|>\delta\end{cases}$$
- **MAE**:对应拉普拉斯噪声假设的 MLE,梯度恒为 $\pm1$(不随误差放大),对离群点鲁棒;但在 0 点不可导、收尾梯度不衰减。最优解是**中位数**(MSE 是均值)。
- **Huber / SmoothL1**:两段拼接,$|r|=\delta$ 处值和斜率都连续,兼得 MSE 的平滑可导和 MAE 的抗离群。Huber 由 Peter Huber(1964)提出;PyTorch `SmoothL1Loss = HuberLoss(δ=1)`。

**交叉熵 / 负对数似然**。多分类,真标签 one-hot $y$,模型经 [[27 Softmax 与温度|Softmax]] 输出概率 $p$:
$$\mathcal L_{\text{CE}}=-\sum_{k} y_k\ln p_k = -\ln p_{c}$$
($c$ 是真类)。它就是 [[30 交叉熵与负对数似然|交叉熵]],等价于最小化预测分布与真实分布的 [[31 KL 散度与 JS 散度|KL 散度]]。**关键漂亮性质**:Softmax + 交叉熵对 logit $z$ 的梯度极其简洁
$$\frac{\partial \mathcal L_{\text{CE}}}{\partial z_k}=p_k-y_k$$
"预测概率减真实概率"——这是 [[38 反向传播在网络中的实现|反向传播]] 里分类网络的标准起点(softmax 的雅可比和 log 的导数刚好约掉)。

**对比损失(InfoNCE 风格)**。给定锚点 $a$、正样本 $a^+$、一批负样本 $\{a^-_j\}$,温度 $\tau$:
$$\mathcal L=-\ln\frac{\exp(\text{sim}(a,a^+)/\tau)}{\exp(\text{sim}(a,a^+)/\tau)+\sum_j\exp(\text{sim}(a,a^-_j)/\tau)}$$
本质是"以正样本为真类的交叉熵",拉近正对、推远负对。是表示学习/检索/嵌入训练的核心目标。温度 $\tau$ 越小越强调最难的负样本(分布更尖,见 [[27 Softmax 与温度|温度]]);它是 InfoNCE 的关键超参。

**Triplet Loss(对比的另一形态)**。$\mathcal L=\max(0,\ d(a,a^+)-d(a,a^-)+m)$:让锚点到正样本的距离比到负样本至少小一个间隔 $m$,否则受罚。人脸识别(FaceNet)的经典损失;与 InfoNCE 的区别是它一次只用一个负样本、用 margin 而非 softmax。

**Hinge Loss(SVM,补全分类损失谱)**。$\mathcal L=\max(0,\ 1-y\cdot\hat z)$($y\in\{-1,+1\}$):分对且间隔 $\ge1$ 则 0 损失,否则线性受罚。最大化间隔、产生稀疏支持向量;与交叉熵的区别是它不输出概率、对"分对且远离边界"的样本梯度为 0。

**Focal Loss**。记 $p_t$ 为模型对**真类**给的概率(预测越准 $p_t$ 越大)。在交叉熵 $-\ln p_t$ 前乘一个**调制因子** $(1-p_t)^\gamma$,再配类别权重 $\alpha_t$:
$$\mathcal L_{\text{focal}}=-\alpha_t(1-p_t)^{\gamma}\ln p_t$$
- $\gamma=0$ 时退化成普通(加权)交叉熵;$\gamma>0$ 时简单样本($p_t\to1$)的因子 $\to0$,被压制。
- 论文默认 $\gamma=2,\ \alpha=0.25$。
直觉:**让一万个简单背景样本不再主导梯度**,模型把力气花在少数难样本上。

![[nn-focal调制因子.png]]

## 代码

```python
import numpy as np

# ❌ 错误:分类任务用 MSE 配 softmax —— 梯度在饱和区接近 0,训练极慢
def bad_mse_for_classification(p, y_onehot):
    return np.mean((p - y_onehot) ** 2)   # 能跑,但收敛慢、易卡

# ✅ 正确:分类用交叉熵(数值稳定版,先减 max 防 exp 溢出)
def cross_entropy(logits, y_idx):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # log-softmax
    return -logp[np.arange(len(y_idx)), y_idx].mean()

logits = np.array([[2.0, 0.5, 0.1]])     # 模型对 3 类的打分
print("CE(真类=0):", cross_entropy(logits, np.array([0])))  # 小,预测对了

# Focal Loss(二分类):验证简单样本被压制
def focal(p_t, gamma=2.0, alpha=0.25):
    return -alpha * (1 - p_t) ** gamma * np.log(p_t)

easy, hard = 0.9, 0.2
print("交叉熵  easy/hard:", -np.log(easy), -np.log(hard))      # 0.105 / 1.609
print("Focal   easy/hard:", focal(easy), focal(hard))         # easy 被压到极小
print("难/易 损失比 —— CE:", np.log(hard)/np.log(easy),
      " Focal:", focal(hard)/focal(easy))                      # Focal 比值更大=更聚焦难样本

# MAE / MSE / Huber 对离群点的鲁棒性对比
def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5*r**2, delta*(a - 0.5*delta))
for r in (0.5, 10.0):
    print(f"误差 r={r:4.1f}:  MSE={r**2:7.2f}  MAE={abs(r):5.2f}  Huber={huber(r):6.2f}")
# r=10 时 MSE=100(爆),MAE/Huber≈10/9.5(抗离群);r=0.5 时 Huber 走二次段更平滑
```

手算对照:交叉熵下 `-ln0.9=0.105`、`-ln0.2=1.609`;Focal 下简单样本因子 $0.1^2=0.01$ 把它压到 $0.25\times0.01\times0.105\approx2.6\times10^{-4}$,而难样本因子 $0.8^2=0.64$ 基本保留,**难/易损失比被显著放大**,印证"聚焦难样本"。

## 面试高频

- **"分类为什么不用 MSE 而用交叉熵?"** ① 交叉熵来自分类的 MLE(伯努利/类别分布),MSE 来自高斯回归假设,任务不匹配;② MSE 配 sigmoid/softmax 时,在输出饱和区梯度 $\to0$,学习极慢;交叉熵的梯度是干净的 $p-y$,不饱和。
- **"Softmax+交叉熵对 logit 的梯度是多少?"** $p_k-y_k$。能秒答说明真懂 backprop 的起点。这也是为什么框架把 softmax 和 CE 融成一个 `CrossEntropyLoss`(数值更稳、梯度更简)。
- **"Focal Loss 解决什么问题?$\gamma$ 和 $\alpha$ 各管什么?"** 解决**类别极度不平衡 + 难易样本失衡**;$\gamma$(focusing,默认 2)压制简单样本、聚焦难样本,$\alpha$(默认 0.25)平衡正负类频率。$\gamma=0$ 退化为加权 CE。
- **"对比损失和交叉熵什么关系?"** InfoNCE 本质就是"把正样本当真类、一批负样本当其它类"的交叉熵,温度 $\tau$ 控制分布尖锐度(见 [[27 Softmax 与温度|温度]])。
- **"MSE 对离群点为什么敏感?MAE / Huber 怎么救?"** 平方放大大误差,单个离群点就主导损失;MAE 线性、对应拉普拉斯假设、最优解是中位数;Huber 小误差走二次(平滑)、大误差走线性(抗离群),`SmoothL1`= Huber(δ=1)。
- **"MSE 的最优常数是均值,MAE 是什么?"** MAE 的最优常数预测是**中位数**(更抗离群),这也是它鲁棒的统计根。
- **"Triplet Loss 和 InfoNCE 区别?"** Triplet 用一个负样本 + margin;InfoNCE 用一批负样本 + softmax 形式(= 以正样本为真类的交叉熵),负样本越多表示学习越强。
- **"Hinge Loss 和交叉熵区别?"** Hinge(SVM)最大化间隔、不输出概率、对分对且远离边界的样本零梯度;交叉熵输出概率、有 MLE 解释、永远有梯度。
- **陷阱:logits vs 概率。** 交叉熵实现要直接吃 logits 做 log-softmax(数值稳定),先 softmax 再取 log 会溢出/丢精度。

## 关键事实

- 交叉熵作为分类标准损失、与 MLE/KL 的等价关系,见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 5、6 章。
- **Focal Loss**:Lin et al., *Focal Loss for Dense Object Detection*(ICCV 2017,arXiv:1708.02002),提出 RetinaNet,默认 $\gamma=2,\ \alpha=0.25$,解决密集检测中前景/背景极度不平衡。
- **对比学习 InfoNCE**:van den Oord et al., *Representation Learning with Contrastive Predictive Coding*(2018);SimCLR(Chen et al., 2020)是其代表性应用;Triplet Loss 见 FaceNet(Schroff et al., 2015)。
- **Huber Loss**:Peter J. Huber, *Robust Estimation of a Location Parameter*(1964);兼具二次段平滑与线性段抗离群,$\delta\approx1.345$ 对应正态下 5% 污染的稳健阈值;PyTorch `SmoothL1Loss` = Huber(δ=1)。
- Softmax+交叉熵融合层(`CrossEntropyLoss`)的数值稳定实现,见 PyTorch 官方文档。
