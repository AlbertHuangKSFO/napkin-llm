[[37 梯度下降：BGD、SGD、Mini-batch|梯度下降]]是训练神经网络的"下山"算法:沿着 [[36 损失函数(MSE、交叉熵、对比、Focal)|损失]] 的负梯度方向一小步一小步更新参数,直到走到谷底。区别只在**每一步用多少样本估计梯度**:全量(BGD)、单样本(SGD)、还是一小批(Mini-batch)。

## 直觉

想象你蒙着眼站在山坡上要下到谷底。[[15 梯度、方向导数与等高线|梯度]] 是脚下"最陡上坡"的方向,负梯度就是"最陡下坡"。每次摸一下坡度,朝下坡走一小步(步长 = **学习率** $\eta$),重复。

问题是:**"摸坡度"用多少数据?**

- **BGD(批量)**:把全部样本的梯度平均后再走一步。方向最准,但每步要扫一遍整个数据集——百万样本时一步慢得离谱。
- **SGD(随机)**:每次只用**一个**样本估梯度。方向很抖(噪声大),但便宜、走得快,而且抖动反而能帮你跳出 [[44 梯度消失、爆炸与梯度裁剪|平坦区/鞍点]]。
- **Mini-batch(小批量)**:折中——每次用一小批(如 32、256 个)。既能用 GPU 并行矩阵运算(快),方向又比单样本稳。**这是实践中的默认**,我们口头说的 "SGD" 通常指它。

一句话:**BGD 准但慢,SGD 快但抖,Mini-batch 全都要一点**。

## 例子

最简单的一维:$\mathcal L(w)=w^2$,梯度 $\nabla=2w$,学习率 $\eta=0.1$。从 $w_0=5$ 出发,更新 $w\leftarrow w-\eta\cdot 2w=w(1-0.2)=0.8w$:

| 步 | $w$ | $\mathcal L=w^2$ |
|---|---|---|
| 0 | 5.000 | 25.00 |
| 1 | 4.000 | 16.00 |
| 2 | 3.200 | 10.24 |
| 3 | 2.560 | 6.55 |
| … | … | … |

每步 $w$ 乘 $0.8$,**指数级逼近最优 $w^*=0$**。这就是 BGD(梯度是精确的)。

**步长太大会发散**。若 $\eta=1.1$,更新因子 $1-2\eta=-1.2$,$w$ 变 $5\to-6\to7.2\to\dots$ **越跳越远**。学习率必须 $\eta<1$(此例 $\eta<1/\text{曲率}$)才收敛——这是 [[40 学习率调度与 warmup、cosine|学习率]] 至关重要的根源。

**收敛区间手算(一维二次)**。$\mathcal L=\tfrac12 c w^2$(曲率 $c$),更新 $w\leftarrow w(1-\eta c)$。收敛要 $|1-\eta c|<1$,即 $0<\eta<\frac2c$:
- $\eta<\frac1c$:单调收敛(每步同号靠近);
- $\eta=\frac1c$:一步到位;
- $\frac1c<\eta<\frac2c$:震荡收敛(来回但幅度递减);
- $\eta>\frac2c$:发散。

上例 $\mathcal L=w^2$ 即 $c=2$,故临界 $\eta=1$、发散界 $\eta=1$(取 $\frac2c=1$),$\eta=1.1$ 越界发散,完全吻合。多维下 $c$ 换成 Hessian 最大特征值 $L$,稳定界 $\eta<\frac2L$。

**条件数与"狭长山谷"**。若 Hessian 最大/最小特征值之比(条件数 $\kappa=L/\mu$)很大,损失面是狭长椭圆:学习率受最陡方向($L$)限制必须很小,而最缓方向($\mu$)就爬得极慢——这就是朴素梯度下降在病态问题上慢的根,催生了 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|动量和自适应优化器]]。

**SGD 的抖动**:若梯度被替换成"真梯度 + 随机噪声",$w$ 不会平滑下降,而是大方向朝 0、局部来回抖——在谷底附近永远停不下来,要靠**学习率衰减**才能收住。

![[nn-梯度下降三种.svg]]

## 原理

**通用更新式**。参数 $\theta$,损失 $\mathcal L$,学习率 $\eta$:
$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta \mathcal L(\theta_t)$$
负梯度是损失下降最快的方向(由 [[15 梯度、方向导数与等高线|方向导数]] 保证),$\eta$ 控制步幅。

**三者只差在 $\nabla \mathcal L$ 怎么估**。设数据集 $N$ 个样本:
$$\underbrace{\nabla\mathcal L=\frac1N\sum_{i=1}^N\nabla\ell_i}_{\text{BGD:全量,准}}\quad
\underbrace{\nabla\mathcal L\approx\nabla\ell_i}_{\text{SGD:单样本,抖}}\quad
\underbrace{\nabla\mathcal L\approx\frac1B\sum_{i\in\mathcal B}\nabla\ell_i}_{\text{Mini-batch:批量 }B}$$
三者都是真梯度的**无偏估计**(期望相等),区别在**方差**:批量越大方差越小、方向越稳;批量越小噪声越大、但更新越频繁、计算越省。

**为什么噪声有时是好事**。SGD 的随机性给优化加了"扰动",能帮模型逃离尖锐的局部极小和鞍点,倾向于落到**平坦极小**(泛化通常更好)。这是 SGD 在深度学习里长盛不衰的隐含正则效应。

**收敛速率(凸情形,面试加分)**:
- **BGD**(光滑凸,固定 $\eta$):$O(1/t)$;强凸时线性收敛 $O(\rho^t)$($\rho<1$)。
- **SGD**(凸):因梯度噪声,只能 $O(1/\sqrt t)$,且要**学习率衰减**($\eta_t\to0$ 且 $\sum\eta_t=\infty,\sum\eta_t^2<\infty$,如 $\eta_t\propto1/t$)才能真正收到极小,否则在谷底噪声半径内徘徊。
这解释了为什么训练后期必须退火学习率(见 [[40 学习率调度与 warmup、cosine|学习率调度]])。

**梯度噪声尺度与"临界批量"**。批量 $B$ 越大梯度方差越小,但收益递减:存在一个"临界批量",超过后再加大 $B$ 几乎不再减少达标所需的步数,只是徒增单步算量。这给了"大 batch 不是越大越好"一个量化解释(McCandlish et al. 2018)。

**批量大小的权衡**(面试常考):
- 批量 $\uparrow$:梯度方差 $\downarrow\propto 1/B$、GPU 利用率 $\uparrow$、但每步算量 $\uparrow$、且太大时**泛化变差**(易陷尖锐极小),通常需配合**线性放大学习率** $\eta\propto B$ 来补偿。
- 一个 **epoch** = 遍历完整个数据集一次;BGD 一个 epoch 走 1 步,Mini-batch 走 $N/B$ 步,SGD 走 $N$ 步。

![[nn-梯度下降轨迹.svg]]

**梯度下降找的是局部极小**。损失非凸时不保证全局最优,但深度网络的高维损失面里"差的局部极小"很少,实践中能找到足够好的解。配合 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|动量/自适应优化器]]、[[41 权重初始化(Xavier、He、正交)|好的初始化]]、[[40 学习率调度与 warmup、cosine|学习率调度]],收敛又快又稳。

## 代码

```python
import numpy as np
np.random.seed(0)

# 线性回归 y = 2x + 1,比较 BGD / SGD / Mini-batch
N = 200
X = np.random.randn(N, 1)
y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(N)
Xb = np.hstack([X, np.ones((N, 1))])         # 加偏置列

def loss(w): return np.mean((Xb @ w - y) ** 2)
def grad(w, idx):                            # 在样本子集 idx 上的梯度
    Xi, yi = Xb[idx], y[idx]
    return 2 * Xi.T @ (Xi @ w - yi) / len(idx)

def train(batch, eta=0.05, epochs=50):
    w = np.zeros(2)
    for _ in range(epochs):
        order = np.random.permutation(N)
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            w -= eta * grad(w, idx)          # 通用更新式
    return w, loss(w)

print("BGD       :", train(N))               # 全量:稳,一 epoch 一步
print("SGD       :", train(1))               # 单样本:抖,但更新最频繁
print("Mini-batch:", train(32))              # 折中:实践默认,w≈[2,1]

# ❌ 学习率过大 → 发散
# print(train(32, eta=2.0))                  # loss 变 nan / inf
# ✅ 合理学习率 + mini-batch → 稳定收敛到 w≈[2., 1.]
```

手算对照:一维 $\mathcal L=w^2,\ \eta=0.1$ 时更新因子 $0.8$,$5\to4\to3.2\to2.56$,与上表一致;代码里三种方式都收敛到 $w\approx[2,1]$(真值),但每 epoch 的更新次数 BGD<Mini-batch<SGD,印证"小批量更新更频繁"。

## 面试高频

- **"BGD / SGD / Mini-batch 区别?为什么实践用 Mini-batch?"** 区别在每步用多少样本估梯度。Mini-batch 兼顾:① 能用 GPU 并行矩阵乘(比逐样本快几个数量级);② 梯度方差比 SGD 小、方向更稳;③ 仍保留一定噪声帮助泛化。
- **"SGD 的噪声是缺点还是优点?"** 双面:抖动让收敛不平滑、谷底停不住(需学习率衰减),但能逃离鞍点/尖锐极小,带来隐含正则、泛化更好。
- **"批量越大越好吗?"** 不是。大批量梯度更准、训练更快(每步),但每步算量大、且易陷尖锐极小导致泛化下降;通常要配合**学习率随批量线性放大**和 [[40 学习率调度与 warmup、cosine|warmup]]。
- **"学习率怎么选?太大太小各会怎样?"** 太大震荡/发散(更新因子超出收敛区间),太小收敛极慢、可能卡在平坦区。理论上界约为 $\eta<2/L$($L$ 是损失的曲率/Lipschitz 常数)。
- **"梯度下降能保证全局最优吗?"** 凸问题能;深度网络非凸,只保证局部极小,但高维下好的极小足够多,实践够用。
- **"梯度下降的收敛速率?"** 凸光滑 BGD $O(1/t)$、强凸线性;SGD 因噪声 $O(1/\sqrt t)$ 且需学习率衰减才真收敛。能说出"SGD 要衰减学习率"是关键。
- **"什么是条件数?为什么它让训练变慢?"** Hessian 最大/最小特征值之比;大条件数=狭长山谷,学习率被最陡方向限死、最缓方向爬得慢,催生动量/自适应优化器。
- **"学习率稳定上界怎么来的?"** 二次近似下 $\eta<2/L$($L$=Hessian 最大特征值/Lipschitz 常数);$\eta=1/L$ 单调收敛,$(1/L,2/L)$ 震荡收敛,$>2/L$ 发散。
- **"为什么训练后期要降学习率?"** SGD 在谷底有"噪声球",固定大步长停不下来;衰减学习率把噪声球收紧,参数落进极小。
- **概念辨析:epoch / iteration / batch。** 一个 iteration = 一次参数更新(一个 batch);一个 epoch = 遍历全数据一遍 = $N/B$ 个 iteration。

## 关键事实

- 梯度下降及其随机变体的标准表述,见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 4.3、8.1–8.3 节。
- 小批量随机梯度的无偏性与方差 $\propto 1/B$ 分析,见同书 8.1.3。
- 大批量训练需线性放大学习率 + warmup:Goyal et al., *Accurate, Large Minibatch SGD*(Facebook, 2017)。
- SGD 噪声倾向平坦极小、利于泛化:Keskar et al., *On Large-Batch Training*(ICLR 2017)。
