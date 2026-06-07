[[42 正则化(L2、Dropout、早停、标签平滑)|正则化]]是一切**牺牲一点训练拟合、换取更好泛化**的手段的统称:防止模型把训练集背下来(过拟合)。四个最常用招式——L2(权重衰减)、Dropout、早停、标签平滑——各从不同角度给模型"加约束"。

## 直觉

过拟合 = 模型在训练集上近乎完美,在新数据上却差。原因通常是模型太"自由":权重可以取极端值、可以记住每个训练样本的噪声。正则化就是**给这份自由加限制**:

- **L2(权重衰减)**:在损失里加一项"惩罚大权重",逼模型用**小而分散**的权重 → 决策面更平滑、不极端。
- **Dropout**:训练时随机"关掉"一部分神经元,逼网络**不能依赖任何单个神经元**,被迫学冗余、鲁棒的特征 → 相当于训练并集成了指数多个子网络。
- **早停(Early Stopping)**:盯着验证集损失,一旦它开始回升(过拟合的信号),就停训。**用"训练时间"当正则强度**。
- **标签平滑**:别让模型对正确类有 100% 的自信。把 one-hot 标签 $[0,1,0]$ 软化成 $[0.03,0.94,0.03]$,逼模型**别过度自信** → 校准更好、泛化更稳。

一句话:**L2 限权重、Dropout 断依赖、早停掐时间、标签平滑压自信**。

**为什么"小权重"就泛化好(零基础直觉)**。权重越大,输入扰动一点点,输出就跳很多(函数很"陡")——这种模型对训练集每个点的位置都极敏感,等于在记噪声。小权重让函数平滑,输入小变动只引起输出小变动,**对没见过的数据更稳**。这也是 L1/L2 为什么管用的共同根源:压住权重幅度 = 限制函数的"陡峭程度"。

**L1 vs L2 顺带一提**。L2 惩罚 $\sum w_i^2$,把权重整体往 0 拉但很少压到恰好 0(平方惩罚在 0 附近梯度趋 0);L1 惩罚 $\sum|w_i|$,在 0 处有恒定梯度,会把一批权重**精确压到 0**,产生**稀疏解**(自带特征选择)。深度学习里以 L2 为主,L1 多用于要稀疏的场景。

## 例子

**L2 等价权重衰减**。损失 $\mathcal L=\mathcal L_0+\frac\lambda2\|w\|^2$。梯度多出一项 $\lambda w$,更新变成:
$$w\leftarrow w-\eta(\nabla\mathcal L_0+\lambda w)=(1-\eta\lambda)\,w-\eta\nabla\mathcal L_0$$
每步先把 $w$ **乘以收缩因子** $(1-\eta\lambda)<1$ 再走梯度——这就是"权重衰减"这个名字的由来。取 $\eta=0.1,\lambda=0.01$,每步权重先缩到 $0.999$ 倍,持续把权重往 0 拉。

**Dropout 的训练/推理缩放**。dropout 率 $p=0.5$,训练时每个神经元以 0.5 概率被置 0,**存活的那些除以 $1-p=0.5$**(即 ×2)保持期望不变(inverted dropout)。推理时**不丢弃、不缩放**,直接用全部神经元。验证期望:某神经元输出 $h$,训练时期望 $=0.5\times\frac{h}{0.5}+0.5\times0=h$,与推理一致 ✓。

**标签平滑**。3 类,真类 $c=1$,平滑系数 $\varepsilon=0.1$。硬标签 $[0,1,0]$ → 软标签:
$$y_k^{LS}=(1-\varepsilon)\,y_k+\frac{\varepsilon}{K}=\Big[\tfrac{0.1}{3},\ 0.9+\tfrac{0.1}{3},\ \tfrac{0.1}{3}\Big]=[0.033,\ 0.933,\ 0.033]$$
模型最优目标不再是 logit $\to\infty$(对 one-hot 才需要),而是收敛到有限 logit,**抑制过度自信**。

**L2 等价收缩的多步追踪**。继续上面的 $\eta=0.1,\lambda=0.01$,收缩因子 $0.999$。假设某权重 $w_0=2$,且暂时让 $\nabla\mathcal L_0=0`(只看衰减项):每步 $w\leftarrow0.999\,w$。10 步后 $w=2\times0.999^{10}\approx2\times0.99004=1.980$;1000 步后 $2\times0.999^{1000}\approx2\times0.368=0.736$。可见衰减是**温和的指数收缩**,持续把权重往 0 拉,但不会一步压死——这正是它"平滑约束"而非"硬截断"的特点。

**Dropout 集成视角的小数字**。一层 10 个神经元、$p=0.5$,每次前向随机保留一个子集,可能的子网络共 $2^{10}=1024$ 个。训练 1 万步 = 在这 1024 个共享权重的子网络间随机采样训练;推理时用完整网络 = **对全部子网络的近似几何平均**。这就是"训练一个网络 ≈ 集成指数多个网络"的由来,也解释了为何 dropout 兼具正则与"免费集成"的效果。

![[nn-dropout掩码.png]]

## 原理

**L2 / 权重衰减**。在损失加 $\frac\lambda2\|w\|^2$:
$$\mathcal L=\mathcal L_0+\frac\lambda2\sum_i w_i^2$$
从 [[26 最大后验 MAP 与正则化的概率解释|MAP]] 看,这等价于给权重加**零均值高斯先验**(相信权重该小)。它把 [[09 特征值与特征向量|损失曲率]] 大的方向收缩得多、平方损失下相当于岭回归。**注意**:L2 正则 $\iff$ 权重衰减仅在 SGD 下严格成立;在 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]] 里两者不等价,要用 **AdamW** 把衰减解耦才正确(Loshchilov 2017)。

**Dropout(Srivastava 2014)**。训练时对每个神经元独立以概率 $p$ 置 0,掩码 $m\sim\text{Bernoulli}(1-p)$:
$$\tilde h=\frac{m\odot h}{1-p}\quad(\text{inverted dropout,训练})$$
$$h_{\text{infer}}=h\quad(\text{推理,不丢不缩放})$$
除以 $1-p$ 让训练期望等于推理,这样推理时无需任何缩放。直觉解释:每次 forward 用的是一个随机子网络,训练完相当于**对指数多个共享权重的子网络做集成**(类似 bagging);又因为不能依赖特定神经元,**打破了神经元间的协同适应(co-adaptation)**,逼出冗余鲁棒的特征。常见 $p=0.5$(全连接层)、$0.1$(Transformer)。

**早停(Early Stopping)**。训练损失单调下降,但验证损失先降后升(开始过拟合)。规则:记录验证损失最低点对应的参数,若连续 `patience` 个 epoch 没刷新最低,就回退到最优点停止。它隐式地把优化限制在"参数还没跑太远"的区域,**效果近似 L2**(在二次损失下两者可证等价):训练步数越多 ≈ 正则越弱。

**标签平滑(Szegedy 2016)**。把 one-hot 目标软化:
$$y^{LS}=(1-\varepsilon)\,y_{\text{onehot}}+\frac{\varepsilon}{K}\mathbf 1$$
再算 [[30 交叉熵与负对数似然|交叉熵]]。为什么有用:对 one-hot 目标,[[36 损失函数(MSE、交叉熵、对比、Focal)|交叉熵]]+ [[27 Softmax 与温度|softmax]] 会驱使正类 logit $\to+\infty$、其它 $\to-\infty$,导致**过度自信、校准差、易受噪声标签干扰**;软标签给一个有限的最优 logit,模型更"谦虚",泛化和置信度校准都更好。$\varepsilon$ 常取 0.1。

![[nn-早停曲线.png]]

**正则全家桶(把视野铺满)**。除了这四招,实践中还有一长串带正则效应的手段,面试能报得越全越好:
- **数据侧**:[[48 数据增强与类别不平衡|数据增强]](翻转/裁剪/Mixup/CutMix)、更多数据 —— 通常是**最有效**的正则,因为直接增大有效样本量。
- **结构侧**:[[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|BatchNorm/LayerNorm]](附带噪声/平滑作用)、DropPath / Stochastic Depth(随机丢整层,深残差网常用)、DropConnect(丢的是权重连接而非神经元)、参数共享(如 CNN 卷积核、RNN 跨时间共享)。
- **优化侧**:**SGD 噪声本身**(小 batch 的梯度噪声有正则效应)、weight decay 的解耦版 AdamW、梯度噪声注入、SAM(Sharpness-Aware Minimization,显式找平坦极小点)。
- **概率/约束侧**:从 [[26 最大后验 MAP 与正则化的概率解释|MAP]] 看,任何先验都是一种正则;最大范数约束(max-norm,限制每个神经元权重范数 $\le c$)常与 dropout 搭配。
- **集成侧**:模型集成、权重平均(EMA / SWA)、知识蒸馏。

**它们正交、可叠加**:实践中常同时用 weight decay + dropout + 早停 + (分类时)标签平滑 + 数据增强。注意"叠太多"也会过度正则导致欠拟合,要看验证集调强度。

## 代码

```python
import numpy as np
np.random.seed(0)

# L2 / 权重衰减:更新式里多一个收缩因子 (1-ηλ)
def sgd_step(w, grad, eta=0.1, wd=0.01):
    return (1 - eta * wd) * w - eta * grad          # 每步先把 w 缩小再走梯度

# Inverted dropout:训练丢弃+缩放,推理原样
def dropout(h, p=0.5, train=True):
    if not train: return h                          # ✅ 推理:不丢不缩放
    mask = (np.random.rand(*h.shape) > p) / (1 - p) # 存活的除以 (1-p)
    return h * mask

h = np.ones(8)
out = dropout(h, p=0.5, train=True)
print("dropout 后:", out, " 期望≈", out.mean(), "(理论=1)")   # 期望保持≈1

# 标签平滑
def label_smooth(y_idx, K, eps=0.1):
    soft = np.full(K, eps / K)
    soft[y_idx] += 1 - eps
    return soft
print("软标签:", label_smooth(1, 3, 0.1))            # [0.033 0.933 0.033]

# 早停:跟踪验证损失,patience 内不刷新最优就停
def early_stop(val_losses, patience=3):
    best, best_t, bad = np.inf, 0, 0
    for t, v in enumerate(val_losses):
        if v < best: best, best_t, bad = v, t, 0
        else:
            bad += 1
            if bad >= patience: return best_t        # 回到最优 epoch
    return best_t
print("最优 epoch:", early_stop([1.0,.7,.5,.45,.46,.48,.5]))  # 3(之后回升→停)

# ❌ 错误:推理时仍做 dropout / 仍除以(1-p) —— 输出分布错乱、结果不稳定
# ✅ 正确:dropout 只在训练开;推理用完整网络(框架的 model.eval() 自动处理)
```

手算对照:dropout $p=0.5$ 存活神经元 ×2,8 个里约 4 个存活、每个值为 2,均值 $\approx1$,与"期望不变"一致;标签平滑 $\varepsilon=0.1,K=3$ 得 $[0.033,0.933,0.033]$;早停在验证损失第 3 个 epoch(0.45)触底后连续回升,返回最优 epoch 3——全部与公式吻合。

```python
# PyTorch 里这四招分别长什么样(工程写法)
import torch, torch.nn as nn

# ① L2 / 权重衰减:Adam 用必须走 AdamW(解耦衰减),否则 ≠ 真 L2
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # ✅
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2) # ❌ 衰减被自适应步长扭曲

# ② Dropout:框架自动处理训练/推理切换,你只要切对 model.train()/eval()
net = nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,10))
net.train()   # ✅ 训练:dropout 生效
# net.eval()  # ✅ 推理:dropout 关闭,用完整网络;忘了切 → 结果随机抖动

# ④ 标签平滑:CrossEntropyLoss 自带参数,一行搞定
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # ✅ 无需手动软化标签

# ③ 早停:框架无内置,手写或用 Lightning 的 EarlyStopping 回调
# 关键:监控验证指标、patience 内不刷新最优就停、并保存最优权重
```

## 面试高频

- **"L2 正则和权重衰减是一回事吗?"** 在 **SGD** 下等价(L2 梯度项 $\lambda w$ 正好产生收缩因子 $1-\eta\lambda$);但在 **Adam** 下**不等价**,因为自适应步长会扭曲 $\lambda w$ 的缩放,必须用 **AdamW** 把衰减解耦(见 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|AdamW]])。这是高频陷阱题。
- **"Dropout 训练和推理为什么不一样?inverted dropout 缩放因子?"** 训练随机丢弃并把存活的除以 $1-p$(保期望);推理用完整网络、不丢不缩放。`model.eval()` 自动切换——忘了切会导致结果随机。
- **"Dropout 为什么能正则?"** ① 打破神经元协同适应,逼出冗余鲁棒特征;② 等价于对指数多个共享权重子网络做集成(类似 bagging)。
- **"早停算正则吗?它和 L2 什么关系?"** 算。它限制参数偏离初始点的距离,在二次损失下可证近似等价于 L2;优点是几乎零成本、自动选"正则强度"(= 训练步数)。
- **"标签平滑解决什么?$\varepsilon$ 怎么取?"** 防止 softmax+CE 把 logit 推到 $\pm\infty$ 造成过度自信、校准差;软化目标让最优 logit 有限。$\varepsilon$ 常取 0.1。代价:可能轻微损害"需要极端置信"的下游(如知识蒸馏的 teacher)。
- **"过拟合时该上哪些正则?"** 组合拳:更多数据/数据增强 > weight decay + dropout + 早停 +(分类)标签平滑;它们正交可叠加。
- **"L1 和 L2 正则有什么区别?"** L2 惩罚平方、把权重整体收缩但少压到 0(解平滑、不稀疏);L1 惩罚绝对值、在 0 处梯度恒定、会把一批权重精确压到 0(产生稀疏、自带特征选择)。深度学习以 L2 为主。
- **"为什么小权重 = 泛化好?"** 大权重让函数陡峭、对输入扰动敏感(易记噪声);小权重让函数平滑、对未见数据更稳。正则就是限制函数的陡峭程度。
- **"dropout 和 BatchNorm 一起用要注意什么?"** 二者都引入训练/推理差异且都有噪声,叠加常互相干扰(著名的 "variance shift");实践中 Transformer 用 dropout、CNN 多用 BN,很少在同一层猛叠,顺序也讲究(一般 BN 在 dropout 前)。
- **"标签平滑会不会有副作用?"** 会:它压制极端置信,做**知识蒸馏的 teacher** 时会抹掉类间细微相似度信息,损害蒸馏效果(Müller 2019);需要精确概率校准或极端置信的场景慎用。
- **"还有哪些 dropout 变体?"** DropConnect(丢权重连接)、Spatial/2D Dropout(整张特征图通道一起丢,适合 CNN)、DropPath/Stochastic Depth(随机丢整个残差块)、Variational Dropout(同一序列各步用同一掩码,RNN 用)。

## 关键事实

- **Dropout**:Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*(JMLR 2014);提出随机丢弃 + 集成解释。
- **标签平滑**:Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*(CVPR 2016,Inception-v3),$\varepsilon=0.1$;"何时有用"分析见 Müller, Kornblith & Hinton(NeurIPS 2019)。
- **L2 ≠ 权重衰减(自适应优化器下)**:Loshchilov & Hutter, *Decoupled Weight Decay Regularization*(AdamW,arXiv:1711.05101,2017)。
- **DropConnect**:Wan et al.(ICML 2013);**Stochastic Depth / DropPath**:Huang et al.(ECCV 2016)随机丢残差块训超深网。
- **SGD 噪声的隐式正则**:Keskar et al.(2017)指出小 batch 倾向平坦极小点、泛化更好;SAM(Foret et al., ICLR 2021)显式优化损失曲面平坦度。
- **SWA / 权重平均**:Izmailov et al.(UAI 2018),平均训练后期权重得到更平坦、更泛化的解。
- 早停、L2、dropout 的正则理论(含早停≈L2 的二次近似),见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 7 章。
