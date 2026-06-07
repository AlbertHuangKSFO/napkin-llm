[[033 ALiBi 与 NoPE]]:两种「不用位置嵌入向量」的位置方案——ALiBi(Attention with Linear Biases,Press 2021,arXiv 2108.12409)给注意力分数减去与距离成正比的惩罚;NoPE(No Positional Encoding)干脆什么都不加,靠因果掩码隐式获得位置感。

## 直觉:位置信息可以「不进向量」

[[029 正弦余弦位置编码(推导)|正弦编码]]、[[030 可学习与相对位置编码|可学习编码]] 都是往词向量里**加**一段位置向量;[[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 是**旋转** q、k。ALiBi 和 NoPE 走得更极端:**位置信息根本不进任何向量**。

- **ALiBi**:在 [[002 自注意力 Self-Attention|注意力]] 打分后、softmax 前,直接给分数**减去一个正比于「query 与 key 距离」的数**。离得越远,惩罚越大,注意力越弱——一个写死的「近大远小」先验。没有任何位置嵌入。
- **NoPE**:decoder-only 模型在因果掩码下,即使**完全不加位置编码**,也能从「能看到几个 token」这一结构里隐式推断位置。第 1 个 token 只能看到自己,第 5 个能看到 5 个——token 数本身就泄露了绝对位置。

两者的共同卖点都是**长度外推**:ALiBi 训 1024、测 2048 困惑度不升;NoPE 在某些研究里外推甚至优于显式编码。

![[pos-alibi-bias.png]]

## 例子:ALiBi 在 5 个 token 上手算

设斜率 $m=1$(实际每个头用不同斜率)。query 在位置 $i$、key 在位置 $j$,偏置 $= -m\,(i-j)$(只看过去,$i\ge j$)。

某个头的原始分数 $q_2\cdot k_j$ 假设是 $[2.0,\ 1.5,\ 1.0]$($j=0,1,2$)。加 ALiBi 偏置 $-1\times(2-j)=[-2,-1,0]$:

$$\text{score}=[2.0-2,\ 1.5-1,\ 1.0-0]=[0.0,\ 0.5,\ 1.0]$$

softmax 后:$e^{0},e^{0.5},e^{1.0}=[1,1.65,2.72]$,归一化 $\approx[0.19,\ 0.31,\ 0.50]$。

对比**不加** ALiBi:softmax$([2,1.5,1])\approx[0.51,0.31,0.19]$——本来最远的 $j=0$ 权重最高,加偏置后**最近的 $j=2$ 反而最高**。ALiBi 把「注意力天然偏向近邻」这个先验硬编进了分数。

**斜率怎么选**:8 个头时,斜率取几何序列 $m_h=2^{-8h/H}$,即 $\frac1{2^1},\frac1{2^2},\dots,\frac1{2^8}$。斜率小的头惩罚轻、能看远;斜率大的头惩罚重、只盯近邻——不同头分工不同距离尺度。

**8 个头的斜率逐个列出(几何序列)。** $H=8$ 时 $m_h=2^{-8h/8}=2^{-h}$,$h=1,\dots,8$:
$$\tfrac12,\ \tfrac14,\ \tfrac18,\ \tfrac1{16},\ \tfrac1{32},\ \tfrac1{64},\ \tfrac1{128},\ \tfrac1{256}$$
对距离 $i-j=100$ 的一对 token,各头的惩罚是 $-m_h\times100$:斜率最大的头惩罚 $-50$(几乎彻底屏蔽 100 远的 token,**专注近邻**),斜率最小的头惩罚 $-0.39$(几乎不衰减,**能看到很远**)。**一组头从"只看眼前"到"放眼全局"连续覆盖各种距离尺度**——这就是为什么固定一组几何斜率就够用、不需要学。$H$ 不是 2 的幂时,论文给了用 $2^{-8/H}$ 起步、必要时插值补足的构造。

## 原理

**ALiBi**。标准注意力 $\text{softmax}(q_i^\top k_j/\sqrt d)$ 改成:

$$\text{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt d}-m\,(i-j)\right)$$

- $m>0$ 是该头固定(非学习)的斜率;$(i-j)\ge0$ 是相对距离。
- 这是一个**静态的相对位置偏置**:不依赖内容、不引入参数、不进 value。
- 之所以能外推:训练时模型见过「距离 $d$ 惩罚 $-md$」的规律,测试时距离更大,惩罚按同一线性规律外推,**分布形状不变**,所以困惑度稳定。论文里 13 亿参数模型训 1024、测 2048,困惑度追平「训 2048 的正弦编码模型」,且快 11%、省 11% 显存。

ALiBi 与 [[032 RoPE 外推：NTK-aware、位置插值、YaRN|RoPE 外推]] 的区别:RoPE 需要插值/调 base 才能外推,ALiBi 天生外推。但 ALiBi 是**纯距离衰减**,表达力弱于 RoPE,长上下文精细检索任务上常不如 RoPE,所以主流大模型最终还是选了 RoPE(BLOOM、MPT 用过 ALiBi)。

**NoPE**。decoder-only + 因果掩码下,第 $t$ 个位置的隐藏状态只聚合了前 $t$ 个输入。理论上(Kazemnejad et al., 2023)证明:无位置编码的因果 Transformer **能表示绝对位置**——注意力可以通过「能看到多少个 token」来计数。直觉:把每个 token 当成「我前面有几个人」,第一层就能数出绝对位置,后续层据此构造相对位置。结论:小模型上 NoPE 外推甚至优于显式编码;但它只对 decoder-only 有效([[035 BERT：双向编码与 MLM|双向编码器]] 没有因果掩码,NoPE 会彻底丢位置),且大规模下是否够用仍有争议。

**NoPE 怎么"数出"位置(构造性直觉)。** 设每个 token 的 value 里有一维恒为常数 1。第 1 层某个头若把注意力做成"对可见范围内均匀平均",那么位置 $t$ 处该维的输出 $=\frac1t\sum_{s\le t}1=1$——这步给不出 $t$;但若另一个头用"全 1 的 key"让 softmax 退化、再结合未归一化的计数信号,网络可在前几层构造出与 $t$(或 $1/t$)相关的量,等价于把"我前面有几个人"算出来。**因果掩码提供的"可见 token 数随位置单调增"就是隐式的位置信号**,NoPE 让网络自己去读它。代价:这种隐式位置较脆弱,依赖训练学到的特定机制,规模化与鲁棒性不如显式编码,所以主流大模型仍显式加 RoPE/ALiBi。

**ALiBi 为什么不需要"训练学斜率"。** 斜率是写死的几何先验:它编码的是"注意力随距离单调衰减"这一**几乎普适的语言归纳偏置**,无需数据去拟合;而且写死才保证外推时"距离 $d$ 的惩罚恒为 $-md$"这条线性规律对任意大 $d$ 成立、分布形状不变。若把斜率设成可学习,反而可能学出对训练长度过拟合的非线性衰减,外推变差。**"零参数 + 线性"正是 ALiBi 外推稳的两个支点**。

## 代码:ALiBi 偏置矩阵

```python
import torch

def alibi_slopes(n_heads):
    # 论文：斜率取几何序列 2^(-8/n), 2^(-16/n), ... （n_heads 为 2 的幂时最简洁）
    import math
    start = 2 ** (-8 / n_heads)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])

# ❌ 错误：把 ALiBi 当成「加进词向量的位置嵌入」——它根本不进向量，而是进注意力分数
def alibi_wrong(x, pos_emb):
    return x + pos_emb          # 这是正弦/可学习编码的做法，不是 ALiBi

# ✅ 正确：构造 (n_heads, L, L) 的偏置矩阵，加到 softmax 之前的分数上
def alibi_bias(n_heads, L):
    slopes = alibi_slopes(n_heads)              # (H,)
    i = torch.arange(L)[:, None]                # 行=query 位置
    j = torch.arange(L)[None, :]                # 列=key 位置
    rel = -(i - j).clamp(min=0).float()         # -(i-j)，未来位 = 0（再叠因果掩码）
    bias = slopes[:, None, None] * rel          # (H, L, L)
    causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    return bias + causal                        # 上三角设 -inf，未来不可见

scores = torch.randn(4, 5, 5)                    # (H, L, L) 原始注意力分数
scores = scores + alibi_bias(4, 5)              # 直接相加
attn = scores.softmax(dim=-1)
# 关键：零额外参数、不碰 q/k/v 向量、训短测长可直接外推
```

## 面试高频

- **ALiBi 和 RoPE 谁更好?** 各有取舍。ALiBi 天生外推(训短测长不退化)、零参数、实现简单;但表达力弱(纯距离衰减),长上下文精检索弱于 RoPE。RoPE 表达力强、是主流标配,但外推需 [[032 RoPE 外推：NTK-aware、位置插值、YaRN|插值技巧]]。BLOOM、MPT 选 ALiBi,LLaMA 系选 RoPE。
- **ALiBi 加在哪?** softmax 前的注意力分数上,作为静态相对偏置;**不进**词向量、不进 q/k/v、零参数。
- **ALiBi 为什么能外推?** 偏置是「距离的线性函数」,训练学到的规律对更大距离按同一斜率外推,注意力分布形状不变 → 困惑度稳定。
- **NoPE 凭什么不加位置也能工作?** 仅限 decoder-only:因果掩码下「能看到几个 token」隐式编码了绝对位置,网络能据此计数。双向编码器无因果掩码,NoPE 会丢位置。
- **斜率 m 是学的吗?** 否,ALiBi 的斜率是写死的几何序列,每个头一个,不参与训练。
- **为什么不同头用不同斜率?** 让一组头从"只盯近邻"(大斜率)到"放眼全局"(小斜率)覆盖各种距离尺度,模型自动分工;若所有头同斜率,长程依赖与局部依赖就无法并存。
- **ALiBi 和 [[030 可学习与相对位置编码|T5 相对偏置]]都加在打分上,区别?** T5 的偏置按距离**分桶查表**(可学习、非单调、桶数有限);ALiBi 是**写死的线性** $-m(i-j)$(零参数、严格单调、任意距离可外推)。ALiBi 更简单、外推更强,T5 更灵活。
- **NoPE 在 encoder(双向)能用吗?** 不能。双向无因果掩码 → 失去"可见 token 数随位置增"的隐式信号 → 彻底置换等变、丢位置。NoPE 只对 decoder-only 成立。
- **既然 ALiBi/NoPE 外推好,为什么主流选 RoPE?** RoPE 表达力强(旋转编码相对相位、保范数、可配各种 kernel)、长上下文精检索更好,配 YaRN 后外推也够用;ALiBi 纯距离衰减表达力弱、NoPE 鲁棒性存疑。综合质量上限,RoPE 胜出成为事实标准。

## 关键事实

- ALiBi:Press、Smith、Lewis,2021,arXiv 2108.12409,标题《Train Short, Test Long》;1.3B 模型训 1024、测 2048 困惑度追平正弦编码训 2048 的模型,且快 11%、省 11% 显存。
- ALiBi 偏置 $-m(i-j)$,斜率 $m$ 写死(几何序列,每头不同),零参数,不进向量,作用于 softmax 前分数。
- 采用 ALiBi 的代表:BLOOM(176B)、MPT;但 LLaMA/Qwen/Mistral 等主流最终选了 RoPE。
- NoPE 理论分析:Kazemnejad et al., 2023(《The Impact of Positional Encoding on Length Generalization》),证明因果 decoder-only 无位置编码也能表示绝对位置,小模型外推可优于显式编码,但仅对 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|decoder-only]] 成立。
- 两者都属「相对位置 / 隐式位置」思路,延续 [[030 可学习与相对位置编码|相对位置编码]] 的脉络。
