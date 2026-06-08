[[27 Softmax 与温度|Softmax]] 把一组任意实数(logits)压成一个和为 1、各项为正的概率分布(即 [[24 常见分布(高斯、伯努利、类别)|类别分布]]);**温度** $T$ 是除在指数里的一个旋钮:$T\to0$ 趋贪心(只挑最大)、$T$ 大趋均匀——这正是 LLM 解码"调温度"的数学根。

## 直觉

网络最后一层吐出的 logits 是没界的实数(可正可负、可大可小),不能直接当概率。Softmax 干两件事:**指数化**(把负数变正、放大差距)+ **归一化**(除以总和让它们加起来等于 1)。结果是一个分布:谁的 logit 大,谁的概率大,但每个都 $&gt;0$、总和恰为 1。

**温度** $T$ 控制这个分布"有多尖":
- $T=1$:标准 softmax。
- $T\to0$:把 logits 放得无限大,差距被无限拉开,概率几乎全压到最大那一项——**趋向贪心 / argmax**,输出确定、保守。
- $T$ 很大:把 logits 压得几乎一样,差距被抹平,分布**趋向均匀**——输出随机、有创意但可能胡说。

LLM 采样里"temperature=0.7"就是在这条尺上选一点:低温更稳更重复,高温更野更多样。

## 例子

**手算 softmax**。logits $z=(2,1,0)$,$T=1$。$e^2=7.389,e^1=2.718,e^0=1$,和 $=11.107$:

$$p=\Big(\tfrac{7.389}{11.107},\tfrac{2.718}{11.107},\tfrac{1}{11.107}\Big)=(0.665,\ 0.245,\ 0.090)$$

(三者和为 1,最大 logit 拿走 $66.5\%$。)

**温度的影响**(同一 $z=(2,1,0)$):

- $T=0.5$:用 $z/T=(4,2,0)$。$e^4=54.6,e^2=7.389,e^0=1$,和 $63$ → $p=(0.867,0.117,0.016)$。**更尖**,最大项从 $0.665$ 升到 $0.867$。
- $T=2$:用 $z/T=(1,0.5,0)$。$e^1=2.718,e^{0.5}=1.649,1$,和 $5.367$ → $p=(0.506,0.307,0.186)$。**更平**,差距被抹小。

一眼可见:降温拉尖、升温抹平。$T\to0$ 时第一项 $\to1$(贪心),$T\to\infty$ 时三项 $\to(1/3,1/3,1/3)$(均匀)。

**平移不变性手算(为何要减最大值)**。把 $z=(2,1,0)$ 整体加 $100$ 变 $(102,101,100)$,softmax 结果不变:分子分母同乘 $e^{100}$ 约掉,仍是 $(0.665,0.245,0.090)$。所以减去 $\max$ 不改概率、只防溢出。但**乘以常数**(即改温度)会改变结果——加法无关、乘法有关,这条区别要分清。

**两类 softmax = sigmoid(手算验证)**。$K=2$ 时,$p_1=\frac{e^{z_1}}{e^{z_1}+e^{z_2}}=\frac{1}{1+e^{-(z_1-z_2)}}=\sigma(z_1-z_2)$:**softmax 退化成对 logit 之差的 sigmoid**。例 $z=(2,0)$:$p_1=\sigma(2)=0.881$,与直接 softmax 一致。这解释了"二分类既可用 sigmoid+1 个 logit、也可用 softmax+2 个 logit"。

![[prob-softmax温度.png]]

## 原理

**定义**(带温度):

$$p_i=\frac{e^{z_i/T}}{\sum_{j}e^{z_j/T}}$$

性质:$p_i&gt;0$,$\sum_i p_i=1$,且对单调放大保序(softmax 不改变 logits 的排序)。

**数值稳定:减最大值**。$e^{z_i}$ 在 $z_i$ 大时溢出。利用对所有 logits 同减常数 $c$ 不改结果(分子分母同乘 $e^{-c/T}$ 约掉),取 $c=\max_j z_j$:

$$p_i=\frac{e^{(z_i-\max_j z_j)/T}}{\sum_k e^{(z_k-\max_j z_j)/T}}$$

这样指数里最大为 0,绝不溢出——所有框架的 softmax 都这么实现。

**两个极限**(LLM 温度的根):
- $T\to0^+$:设 $z_m$ 唯一最大,$(z_i-z_m)/T\to-\infty$($i\ne m$),故 $p_m\to1$、其余 $\to0$——**退化为 argmax / 贪心解码**。
- $T\to\infty$:$z_i/T\to0$,所有 $e^{z_i/T}\to1$,故 $p_i\to1/K$——**均匀分布**。

**与交叉熵/MLE 的关系**。softmax 输出正是 [[24 常见分布(高斯、伯努利、类别)|类别分布]] 的参数;接上 $-\sum_k y_k\ln p_k$ 就是分类的交叉熵 = [[25 最大似然估计 MLE|MLE]] 的负对数似然。这条 "logits → softmax → 交叉熵" 是分类网络训练的标准管线;详细在 [[30 交叉熵与负对数似然|交叉熵]] 展开(前向呼应)。

**softmax 的雅可比(自身导数,要会推)**。softmax 不是逐元素函数,$p_i$ 依赖所有 $z_j$,雅可比是
$$\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)=\begin{cases}p_i(1-p_i)&i=j\\-p_i p_j&i\ne j\end{cases}$$
对角项 $p_i(1-p_i)>0$(自己 logit 升则自己概率升),非对角 $-p_ip_j<0$(别人升则自己降,概率"此消彼长")。把它接上交叉熵的 $-1/p$,中间 $p_i$ 约掉,才得到下面那个干净结果。

![[nn-Softmax雅可比.png]]

**softmax 的梯度(为何好训练)**。配交叉熵时,对 logit 的梯度极简洁:$\frac{\partial L}{\partial z_i}=p_i-y_i$(预测概率减真实 one-hot)。这个干净形式让反向传播稳定,是 softmax+交叉熵成为分类标配的关键(完整逐步推导在 [[30 交叉熵与负对数似然|交叉熵]])。

**温度对梯度的影响**。带温度时 $\frac{\partial L}{\partial z_i}=\frac1T(p_i-y_i)$,即温度把梯度整体缩放 $1/T$。知识蒸馏用高温 $T$ 时,损失要乘回 $T^2$ 来补偿这个缩放(Hinton 2015 的实现细节),否则软标签项的梯度被压得太小。

**注意区别**:训练时温度恒为 $T=1$(温度是推理期的采样旋钮,不是可学参数);温度只在**解码/蒸馏**时调。知识蒸馏里高温 softmax 暴露"软标签"(类间相似度),也用到 $T$。讲相似度软分布时这与 [[04 Embedding 与向量数据库|Embedding]] 的相似度归一化思想相通。

**LLM 解码的三个旋钮(温度只是其一)**:
- **温度 $T$**:重塑整个分布的尖锐度($T<1$ 更确定,$T>1$ 更随机)。
- **top-k**:只在概率最高的 $k$ 个 token 里采样,截断长尾(防止偶尔抽到离谱词)。
- **top-p(nucleus)**:取累积概率达 $p$ 的最小 token 集合里采样,集合大小随上下文自适应。
工程上常先 top-k/top-p 截断、再按温度归一化采样。温度只调"软硬",top-k/p 调"候选范围",三者正交配合。

**softmax 在注意力里也出现**。Transformer 的注意力权重 $\text{softmax}(QK^\top/\sqrt{d_k})$ 就是对打分做 softmax,$\sqrt{d_k}$ 起的正是"温度"作用——维度越高点积方差越大、需要降温防止 softmax 过尖、梯度消失。所以"缩放点积注意力"的缩放因子本质是一个固定温度。

![[prob-softmax管线.png]]

## 代码

```python
import numpy as np

def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max()              # ✅ 减最大值,防溢出
    e = np.exp(z)
    return e / e.sum()

z = np.array([2., 1., 0.])
print("T=1.0:", softmax(z, 1.0).round(3))   # [0.665 0.245 0.09 ]
print("T=0.5:", softmax(z, 0.5).round(3))   # [0.867 0.117 0.016] 更尖
print("T=2.0:", softmax(z, 2.0).round(3))   # [0.506 0.307 0.186] 更平
print("T→0  :", softmax(z, 1e-3).round(3))  # [1. 0. 0.] 趋贪心
print("T→大 :", softmax(z, 1e3).round(3))   # [0.333 0.333 0.333] 趋均匀

# ❌ 不减最大值:大 logit 直接溢出成 inf/nan
big = np.array([1000., 999., 998.])
e = np.exp(big)
print("❌ 朴素 exp:", e[:1])                 # inf → 后面全 nan

# ✅ 稳定版正常工作
print("✅ 稳定 softmax:", softmax(big).round(3))  # [0.665 0.245 0.09]
```

`❌` 直接对大 logits 做 `exp` 会溢出成 `inf`、归一化后变 `nan`;`✅` 减最大值后结果与小 logits 情形完全一致([1000,999,998] 与 [2,1,0] 差距相同 → 概率相同)。温度从 0.5→2 分布由尖变平,$T\to0$ 趋贪心、$T\to\infty$ 趋均匀,印证手算。

```python
import numpy as np

# 验证两类 softmax = sigmoid(logit 之差)
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()
def sigmoid(x): return 1 / (1 + np.exp(-x))
z = np.array([2., 0.])
print("softmax[0]:", round(softmax(z)[0], 4), " sigmoid(2-0):", round(sigmoid(2.), 4))  # 相等

# top-k / top-p 截断 + 温度采样(LLM 解码三旋钮)
def sample(logits, T=1.0, top_k=None, top_p=None, rng=np.random.default_rng(0)):
    z = logits / T
    p = softmax(z)
    if top_k:                                   # 只留概率最高的 k 个
        thr = np.sort(p)[-top_k]
        p = np.where(p >= thr, p, 0)
    if top_p:                                   # 留累积到 top_p 的最小集合
        order = np.argsort(p)[::-1]; csum = np.cumsum(p[order])
        keep = order[csum <= top_p]; keep = np.append(keep, order[len(keep)])
        mask = np.zeros_like(p); mask[keep] = 1; p = p * mask
    p = p / p.sum()
    return rng.choice(len(p), p=p)
logits = np.array([3., 2., 1., 0., -1.])
print("贪心(T→0):", logits.argmax())
print("top-k=2 采样:", sample(logits, T=1.0, top_k=2))
```

两类 softmax 与 sigmoid 数值完全一致,印证退化关系;top-k/top-p 在采样前把长尾截断,温度再控制软硬。

## 面试高频

- **"softmax 公式 + 温度怎么加?"** $p_i=e^{z_i/T}/\sum_j e^{z_j/T}$。$T\to0$ 贪心,$T\to\infty$ 均匀。LLM 解码温度就是这个 $T$。
- **"为什么 softmax 要减最大值?"** 防 $e^{z}$ 溢出;同减常数不改结果(分子分母约掉)。工程必答。
- **"温度调高/调低对生成有何影响?"** 低温更确定、重复、保守;高温更随机、多样、易跑题。$T=0$ 等于贪心。
- **"softmax+交叉熵对 logit 的梯度?"** $p_i-y_i$,极简。这是它好训练的核心,务必能写。
- **"softmax 会改变 logits 排序吗?"** 不会(单调)。argmax 在任何 $T&gt;0$ 下不变,变的只是概率的尖锐程度。
- **"两类 softmax 和 sigmoid 什么关系?"** $K=2$ 时 softmax 退化成对 logit 之差的 sigmoid;二分类用 sigmoid(1 logit)或 softmax(2 logit)等价。
- **"softmax 的雅可比是什么?"** $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$;对角正、非对角负,概率此消彼长。接交叉熵后约成 $p-y$。
- **"top-k、top-p、温度区别?"** 温度调分布软硬;top-k 固定保留前 $k$ 个候选;top-p 保留累积概率达 $p$ 的自适应候选集。正交,常组合用。
- **"注意力里的 $\sqrt{d_k}$ 为什么需要?"** 它是固定温度:高维点积方差大,不除会让 softmax 过尖、梯度消失;除以 $\sqrt{d_k}$ 把打分方差拉回 1 量级。
- **陷阱:温度是推理旋钮,训练时 $T=1$。** 把温度当可学参数是常见误解;它在解码/蒸馏时手调(蒸馏高温时损失要乘 $T^2$ 补偿梯度缩放)。

## 关键事实

- "softmax"一词与多分类用法源自 Bridle(1990);它是 logistic/sigmoid 到多类的推广,见 Bishop《PRML》(2006)第 4.3.4 节。
- 带温度的 softmax 用于知识蒸馏的"软标签",见 Hinton, Vinyals & Dean《Distilling the Knowledge in a Neural Network》(2015)。
- LLM 解码中温度采样(temperature sampling)与 top-k / nucleus(top-p)采样的对比,见 Holtzman et al.《The Curious Case of Neural Text Degeneration》(2020)。
