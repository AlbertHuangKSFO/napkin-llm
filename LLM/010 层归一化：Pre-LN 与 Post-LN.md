[[010 层归一化：Pre-LN 与 Post-LN]]:Transformer 用 **LayerNorm**(对每个 token 沿特征维归一)而非 BatchNorm 来稳定训练;把它放在残差相加**之后**是 Post-LN(2017 原版,需 warmup、难训深),放在子层**之前**是 Pre-LN(GPT-2 之后主流,梯度稳、可少/免 warmup)。

## ① 直觉:给每个 token 的特征"做标准化",防止数值乱飘

深网络里,每层输出的数值分布会随训练不断漂移(尺度忽大忽小),让后面的层很难学——这叫内部协变量偏移。归一化([[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]])的办法是:把每层的激活重新拉回"均值 0、方差 1",再用可学习的缩放 $\gamma$、平移 $\beta$ 还原表达力。

**关键在沿哪个方向归一**:

- **BatchNorm**:对每个特征维,跨整个 batch 的样本求 $\mu,\sigma$(纵向)。依赖 batch 统计。
- **LayerNorm**:对**每个 token 自己**,沿它的特征维求 $\mu,\sigma$(横向)。**与 batch、序列长度完全无关**。

LLM 用 LN 不用 BN,正因为 NLP 序列**变长、batch 常常很小、还有 `<pad>` 污染**——BatchNorm 的 batch 统计会很噪、不稳定,而 LayerNorm 只看单个 token,训练和推理行为一致。

## ② 例子:对一个 token 做 LayerNorm

某 token 特征向量 $x=[2,\,4,\,4,\,4,\,5,\,5,\,7,\,9]$($d=8$)。

均值:$\mu=\dfrac{2+4+4+4+5+5+7+9}{8}=\dfrac{40}{8}=5$。

方差:$\sigma^2=\dfrac{1}{8}\big[(2-5)^2+(4-5)^2\cdot3+(5-5)^2\cdot2+(7-5)^2+(9-5)^2\big]=\dfrac{9+3+0+4+16}{8}=4$,故 $\sigma=2$。

归一化(取 $\gamma=1,\beta=0$):

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\approx\frac{x_i-5}{2}
$$

得 $\hat{x}=[-1.5,\,-0.5,\,-0.5,\,-0.5,\,0,\,0,\,1,\,2]$,均值 0、方差 1。**注意:这步只用到这一个 token 自己的 8 个数,和 batch 里其他样本、和句子多长都无关**。这正是它在变长序列里稳定的原因。

![[tf-LN沿特征归一.png]]

## ③ 原理:LayerNorm 公式 + Pre-LN/Post-LN 位置之争

LayerNorm 对单个样本的特征向量 $x\in\mathbb{R}^{d}$:

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\quad \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,\quad \text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

$\gamma,\beta\in\mathbb{R}^{d}$ 是可学习参数,$\epsilon$ 防除零。

**放哪里?两种接法**(配合 [[009 残差连接与梯度流|残差]]):

| | 公式 | 特点 |
|---|---|---|
| **Post-LN**(原版) | $x_{\ell+1}=\text{LN}\big(x_\ell+\text{Sublayer}(x_\ell)\big)$ | LN 卡在主干残差路上 |
| **Pre-LN**(现代) | $x_{\ell+1}=x_\ell+\text{Sublayer}\big(\text{LN}(x_\ell)\big)$ | 残差路是干净恒等 |

Xiong et al.(2020)用平均场理论证明:**Post-LN 在初始化时靠近输出层的梯度很大**,直接上大学习率会爆,所以必须有 learning-rate **warmup**(慢慢热身)才能稳;层数一深更难训。而 **Pre-LN 把 LN 移进残差块内部,初始化时梯度就很规整**,残差通路不被 LN 打断(那个 "+1" 高速路保持干净),因此**可以去掉或弱化 warmup**、更容易堆深、收敛更快。代价是 Pre-LN 最终精度有时略低于调好的 Post-LN,但工程稳定性的收益让它成为 GPT-2 之后的默认选择。

![[tf-preLN-postLN.png]]

**现代变体**:很多 LLM(LLaMA 等)用 **RMSNorm**——省掉减均值,只按均方根缩放:$\text{RMSNorm}(x)=\gamma\odot\dfrac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}$,更省算力、效果相当。详见 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]]。

**RMSNorm 省了哪一步(用上面的例子)**。还是 $x=[2,4,4,4,5,5,7,9]$。RMSNorm **不减均值**,直接算均方根:
$$\text{RMS}=\sqrt{\tfrac18(4+16+16+16+25+25+49+81)}=\sqrt{\tfrac{232}{8}}=\sqrt{29}\approx5.39$$
$$\text{RMSNorm}(x)=x/5.39=[0.371,0.742,0.742,0.742,0.928,0.928,1.30,1.67]$$
对比 LayerNorm 先减 $\mu=5$ 再除 $\sigma=2$。RMSNorm 省掉"求均值、减均值"两步,只保留缩放;Zhang & Sennrich(2019)发现减均值这步对效果贡献很小,去掉后更快(少一次归约),LLaMA 等遂采用。代价:RMSNorm 不做中心化,对均值漂移敏感些,但实践中无碍。

**LayerNorm vs BatchNorm:同一批数据,沿不同方向归一**。设 batch 有 2 个 token,各 4 维:$x_1=[1,2,3,4]$,$x_2=[10,20,30,40]$。
- **LayerNorm**(每个 token 沿自己 4 维):$x_1$ 用自己的 $\mu=2.5,\sigma\approx1.12$ 归一,$x_2$ 用自己的 $\mu=25,\sigma\approx11.2$ 归一——**两个 token 各算各的,互不影响**,推理时来一个算一个,行为确定。
- **BatchNorm**(每个特征维沿 batch):第 1 维用 $\{1,10\}$ 的统计,第 2 维用 $\{2,20\}$……**依赖 batch 里有谁**。batch 小、变长、有 pad 时统计噪声大;且训练用 batch 统计、推理用滑动平均,两者不一致。这就是 LLM 坚决用 LN 不用 BN 的具体原因。

## ④ 代码:LayerNorm 手算 + Pre/Post 接法对比

```python
import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    # x: (..., d),沿最后一维(特征)归一,与 batch 无关
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    if gamma is None: gamma = np.ones(x.shape[-1])
    if beta  is None: beta  = np.zeros(x.shape[-1])
    return gamma * xhat + beta

x = np.array([2.,4,4,4,5,5,7,9])
print(layer_norm(x).round(2))   # [-1.5 -0.5 -0.5 -0.5 0. 0. 1. 2.]

# ❌ Post-LN:LN 在相加之后,卡在残差主干上 → 需 warmup,难训深
def post_ln_block(x, sub, ln):
    return ln(x + sub(x))

# ✅ Pre-LN:LN 在子层之前,残差是干净恒等 → 梯度稳,易堆深
def pre_ln_block(x, sub, ln):
    return x + sub(ln(x))
```

```python
# PyTorch:注意 normalized_shape 是「特征维 d」,不是 batch
import torch.nn as nn
ln = nn.LayerNorm(768)          # 对每个 token 的 768 维归一
# ✅ LLM 用 LayerNorm;❌ 不要在 Transformer 里用 nn.BatchNorm1d(变长+小batch 不稳)
```

## 面试高频

- **Q:为什么 LLM 用 LayerNorm 而不是 BatchNorm?** NLP 序列变长、batch 小、有 padding,BN 的 batch 统计噪声大且训练/推理不一致;LN 只对单 token 沿特征归一,与 batch 和序列长度解耦,稳定。
- **Q:Pre-LN 和 Post-LN 区别与取舍?** Post-LN(原版)LN 在残差相加后,需 warmup、深了难训;Pre-LN 把 LN 放子层前,残差通路干净、梯度稳、可免/弱 warmup、易堆深,是现代主流;Post-LN 调好后峰值精度有时略高。
- **Q:LayerNorm 沿哪个维度算均值方差?** 沿特征维(最后一维 $d$),每个 token 一组 $\mu,\sigma$。
- **Q:RMSNorm 和 LayerNorm 差在哪?** RMSNorm 不减均值、只用均方根缩放,少一步、更快,效果相当,LLaMA 等采用。
- **Q:为什么 Post-LN 需要 warmup?** Xiong et al. 证明其初始化时输出层附近梯度过大,大学习率会不稳,warmup 用小学习率热身规避;Pre-LN 梯度天然规整故可省。
- **Q:γ、β 的作用?** 归一化会抹掉尺度信息,$\gamma,\beta$ 让网络可学习地"还原"必要的缩放和平移,保留表达力。
- **Q:RMSNorm 省了哪一步,为什么能省?** 省掉"求均值、减均值",只按均方根缩放;Zhang & Sennrich(2019)发现中心化对效果贡献小,去掉后少一次归约、更快,LLaMA 采用。
- **Q:LayerNorm 有几个可学习参数?** $2d$ 个($\gamma,\beta$ 各 $d$);RMSNorm 只有 $\gamma$ 共 $d$ 个(无 $\beta$)。
- **Q:还有哪些归一化位置变体?** 除 Pre/Post-LN,还有 sandwich norm(子层前后都加,如 Gemma)、DeepNorm(给残差乘放大系数 + 缩小初始化,让 Post-LN 也能训到上千层,微软 2022)。核心都是平衡"残差路干净"与"输出方差受控"。
- **Q:为什么 Post-LN 调好后峰值精度有时反而更高?** Post-LN 的 LN 在残差后,对每层输出强制归一,正则更强;代价是难训(需 warmup)。是稳定性 vs 峰值精度的权衡。

## 关键事实

- LayerNorm 出自《Layer Normalization》(Ba, Kiros & Hinton, 2016)。
- 原版 Transformer 用 Post-LN(Vaswani et al., 2017,$\text{LayerNorm}(x+\text{Sublayer}(x))$)。
- Pre-LN 的理论分析出自《On Layer Normalization in the Transformer Architecture》(Xiong et al., ICML 2020):证明 Post-LN 初始化时输出层梯度大、需 warmup,Pre-LN 梯度规整、可去 warmup;GPT-2(2019)起 Pre-LN 成主流。
- RMSNorm 出自《Root Mean Square Layer Normalization》(Zhang & Sennrich, 2019),被 LLaMA(2023)等采用。
