[[115 手写多头注意力与 Transformer Block(numpy)|手写多头注意力与 Transformer Block]]:只用 numpy 实现**缩放点积注意力 → 多头拆分合并 → 一个完整 Transformer Block(Pre-LN + 多头注意力 + FFN + 两条残差)**,逐行打印张量形状,让每个矩阵乘都肉眼可见。这是把 [[002 自注意力 Self-Attention|自注意力]]、[[005 多头注意力 Multi-Head|多头]]、[[013 Transformer 整体数据流(逐张量形状)|数据流]] 从公式变成可运行代码的一步——不为训练(numpy 手写反传太慢),只为彻底看清 GPT 的核心积木。

## 直觉

注意力一句话:**每个 token 拿自己的 Query 去和所有 token 的 Key 算相似度,得到一组权重,再用权重对所有 token 的 Value 加权平均**——于是每个位置都「看了一眼」相关的上下文(见 [[003 Query、Key、Value 的设计|QKV 设计]])。

落到 numpy 就五个矩阵运算:

1. `Q = x @ Wq`,`K = x @ Wk`,`V = x @ Wv`——把输入投影成 Query/Key/Value。
2. `scores = Q @ K.T / sqrt(d)`——所有 token 两两的相似度,**除以 $\sqrt{d}$** 防止 softmax 饱和(见 [[004 缩放点积注意力(为何除以根号 dk)|缩放点积]])。
3. 加**因果掩码**:位置 $t$ 不许看 $>t$(自回归必需,见 [[007 因果掩码与 padding 掩码|因果掩码]])。
4. `attn = softmax(scores)`——每行归一成概率(行和=1)。
5. `out = attn @ V`——加权平均,回到维度 $d$。

**多头** = 把 $d$ 切成 $h$ 份,各自独立做上面这套,再拼回来过一个输出投影 $W_o$(见 [[005 多头注意力 Multi-Head|多头]])。一个 **Block** = 注意力子层 + FFN 子层,每个子层外包一层 LN 和一条残差(见 [[013 Transformer 整体数据流(逐张量形状)|数据流]])。

![[impl-注意力数据流形状.png]]

## 例子

**走一遍最小张量(B=1, T=3, d=4, 2 头)。** 三个 token,每个 4 维。

- `x`:`(1,3,4)`。
- `Q,K,V`:各 `(1,3,4)`(线性投影不变形)。
- 拆 2 头:`(1, 2, 3, 2)` —— `(B, n_head, T, head_size)`,$head\_size=d/h=2$。
- `scores = Q@K.T/sqrt(2)`:`(1,2,3,3)` —— 每个头一张 $3\times3$ 的「谁看谁」表。
- 加因果掩码后 softmax,head0 的注意力矩阵实测:

```
[[1.000  0     0   ]      ← 位置0 只能看自己
 [0.512  0.488 0   ]      ← 位置1 看 0、1，行和=1，上三角=0
 [0.342  0.343 0.315]]    ← 位置2 看 0、1、2
```
**严格下三角**——这就是因果掩码起作用的证据;每行和恰为 1。

- `out = attn@V`:`(1,2,3,2)` → 合头 `(1,3,4)` → 过 $W_o$ → `(1,3,4)`。**进去什么形状,出来什么形状**——注意力是「形状保持」的算子,这让它能无限堆叠。

**为什么除以 $\sqrt{d_k}$ 不能省(一个数值直觉)。** 设 Q、K 各维独立、均值 0 方差 1,则点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 的方差是 $d_k$(独立项方差相加),标准差 $\sqrt{d_k}$。$d_k=64$ 时点积量级约 $\pm 8$,$d_k=128$ 时约 $\pm 11$。这么大的 logits 喂进 softmax,会让分布**极度尖锐**(几乎 one-hot),反传时梯度趋近 0(softmax 饱和区)→ 训练停滞。除以 $\sqrt{d_k}$ 把点积方差**归一回 1**,logits 量级稳定在 $\pm 1$ 附近,softmax 不饱和、梯度健康。这就是为什么是 $\sqrt{d_k}$ 而不是别的常数(见 [[004 缩放点积注意力(为何除以根号 dk)|为何除以根号 dk]])。

**注意力矩阵的每一行在算什么(读懂热图)。** `attn[b,h,t,:]` 是位置 $t$ 的 query「分配给所有位置的注意力权重」(行和=1)。因果掩码让第 $t$ 行只有前 $t+1$ 列非零(严格下三角)。在训好的模型里,这些权重有可解释结构:有的头是「上一个 token 头」(注意力集中在对角线下方一格)、有的是「分隔符头」「指代头」——不同头学不同关系,这正是多头的价值。tiny 版权重随机,所以行内近似均匀,但**下三角 + 行和 1** 的结构必然成立——这是验证实现对错的硬指标。

**Block 走一遍。** 同样的 `(1,3,4)` 进 Block:Pre-LN → 多头注意力 → 加残差 → Pre-LN → FFN(`4→16→4`)→ 加残差 → 出 `(1,3,4)`。实测输出形状不变。**正因形状不变,GPT 才能把 N 个 Block 直接 `for` 循环串起来。**

![[impl-TransformerBlock数据流.png]]

## 原理

**1. 缩放点积注意力(单头)。** 设 $X\in\mathbb{R}^{T\times d}$,投影得 $Q=XW_q,\ K=XW_k,\ V=XW_v\in\mathbb{R}^{T\times d}$。注意力(见 [[006 注意力的矩阵形式与张量形状|矩阵形式]]):

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$QK^\top\in\mathbb{R}^{T\times T}$ 是相似度矩阵;除以 $\sqrt{d_k}$ 让点积方差归一(否则 $d$ 大时点积量级 $\propto d$,softmax 进饱和区、梯度消失,见 [[004 缩放点积注意力(为何除以根号 dk)|为何除以根号 dk]])。$M$ 是因果掩码:上三角填 $-\infty$($-10^9$),softmax 后变 0。

**2. 多头。** 把 $d$ 维切成 $h$ 个 $d_k=d/h$ 维子空间,每头独立算注意力,再拼接过输出投影(见 [[005 多头注意力 Multi-Head|多头]]):

$$\mathrm{MHA}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_o,\quad \text{head}_i=\mathrm{Attn}(XW_q^{(i)},XW_k^{(i)},XW_v^{(i)})$$

实现上不开 $h$ 套权重,而是用整块 $W_q,W_k,W_v\in\mathbb{R}^{d\times d}$ 投影后,**reshape 成 $(B,T,h,d_k)$ 再 transpose 成 $(B,h,T,d_k)$**,让批量矩阵乘一次算完所有头——这就是「拆头/合头」全部内容。多头的价值:不同头学不同关系(语法/指代/位置),表达力更强(见 [[005 多头注意力 Multi-Head|多头]])。

**3. Transformer Block(Pre-LN)。** 一个 Block 是两个残差子层(见 [[009 残差连接与梯度流|残差]]、[[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]):

$$\begin{aligned}
x &\leftarrow x+\mathrm{MHA}(\mathrm{LN}_1(x))\\
x &\leftarrow x+\mathrm{FFN}(\mathrm{LN}_2(x))
\end{aligned}$$

**Pre-LN**(归一化放在子层前)让残差成为干净的恒等路径,深层训练更稳——现代 GPT 的标配(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN 与 Post-LN]]、[[015 Transformer 训练稳定性|训练稳定性]])。FFN 是「升维 4 倍 → 非线性 → 降回」:$\mathrm{FFN}(h)=\mathrm{GELU}(hW_1+b_1)W_2+b_2$,$W_1\in\mathbb{R}^{d\times4d}$(见 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]])。

**3'. LayerNorm 在做什么、为什么对最后一维。** LN 对每个 token 的特征向量(最后一维 $d$)做标准化:$\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta$,$\mu,\sigma$ 是这个 token 自己 $d$ 维的均值方差。注意是**对特征维、不是对 batch 或序列维**——这是和 BatchNorm 的关键区别:LN 不跨样本、不依赖 batch 统计,所以推理时行为和训练一致(无需存 running stats),且对变长序列友好。$\gamma,\beta$ 是可学习的缩放和平移(让模型能「撤销」归一化)。Pre-LN 把它放在子层前,使残差路径 $x+f(\text{LN}(x))$ 里的 $x$ 是干净的恒等流,梯度直达,深层稳(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]、[[009 残差连接与梯度流|残差]])。

**4. 为什么用 numpy 而不训练它。** numpy 没有 autograd,手写反传一个完整 Block 极繁琐;它的价值是**前向透明**:每个 `.shape` 都能打印,每个掩码都能可视化。理解后,117 把这套换成 PyTorch 的 `nn.Linear` + autograd,数学完全一致,只是反传和加速由框架代劳。

**5. 这套 numpy Block 与生产注意力的差距。** 数学一致,被省的是**效率**:① **Flash-Attention**(见 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]])——本实现显式构造 $T\times T$ 的 `scores` 矩阵(占 $O(T^2)$ 显存),Flash 用分块 + online-softmax 不落地这张大矩阵,显存降到 $O(T)$、还更快;② **KV-Cache**(见 [[102 KV-Cache|KV-Cache]])——生成时不重算历史 K、V;③ **GQA/MQA**(见 [[019 GQA 分组查询注意力|GQA]])——KV 头共享省显存;④ **融合 kernel**——把投影、缩放、softmax、加权合成一个 CUDA kernel 减访存。**但这些都不改 `softmax(QK^T/√d + M)V` 这个核心公式**,理解 numpy 版就理解了本质。

## 代码

完整可运行(纯 numpy)。已验证:注意力严格因果(上三角为 0)、每行和为 1、Block 形状保持。

```python
# model_numpy.py —— 用 numpy 看清多头注意力与一个 Transformer Block（前向理解版）
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # 减最大值防溢出（数值稳定 softmax）
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_self_attention(x, Wq, Wk, Wv, Wo, n_head):
    """x:(B,T,d) -> out:(B,T,d), attn:(B,n_head,T,T)"""
    B, T, d = x.shape
    hs = d // n_head                                  # head_size
    Q, K, V = x @ Wq, x @ Wk, x @ Wv                 # 各 (B,T,d)
    split = lambda z: z.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)  # ->(B,nh,T,hs)
    Q, K, V = split(Q), split(K), split(V)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(hs)   # (B,nh,T,T) 相似度，缩放
    mask = np.tril(np.ones((T, T)))                      # 下三角=1（可见），上三角=0
    scores = np.where(mask == 0, -1e9, scores)           # 因果掩码：未来位置 -inf
    attn = softmax(scores, axis=-1)                      # 行归一，行和=1
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, d)  # 合头回 (B,T,d)
    return out @ Wo, attn                                # 输出投影

def layernorm(x, g, b, eps=1e-5):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def gelu(x):                                              # GPT 用的近似 GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class Block:
    """一个 Transformer Block：Pre-LN + 多头注意力 + FFN + 两条残差。"""
    def __init__(self, d, n_head, seed=0):
        rng = np.random.default_rng(seed); s = 0.3
        self.nh = n_head
        self.Wq, self.Wk = rng.standard_normal((d,d))*s, rng.standard_normal((d,d))*s
        self.Wv, self.Wo = rng.standard_normal((d,d))*s, rng.standard_normal((d,d))*s
        self.g1, self.b1 = np.ones(d), np.zeros(d)       # LN1 仿射参数
        self.g2, self.b2 = np.ones(d), np.zeros(d)       # LN2
        self.W1, self.b_1 = rng.standard_normal((d,4*d))*s, np.zeros(4*d)  # FFN 升维
        self.W2, self.b_2 = rng.standard_normal((4*d,d))*s, np.zeros(d)    # FFN 降维
    def __call__(self, x):
        a, _ = causal_self_attention(layernorm(x, self.g1, self.b1),
                                     self.Wq, self.Wk, self.Wv, self.Wo, self.nh)
        x = x + a                                        # 残差① (注意力子层)
        h = layernorm(x, self.g2, self.b2)
        h = gelu(h @ self.W1 + self.b_1) @ self.W2 + self.b_2   # FFN: d->4d->d
        x = x + h                                        # 残差② (FFN 子层)
        return x
```

```python
# —— 运行验证：形状、因果、行和、Block 保形 ——
rng = np.random.default_rng(0)
B, T, d, nh = 1, 3, 4, 2
x = rng.standard_normal((B, T, d)) * 0.5
Wq, Wk, Wv, Wo = [rng.standard_normal((d, d)) * 0.3 for _ in range(4)]

out, attn = causal_self_attention(x, Wq, Wk, Wv, Wo, nh)
print("out:", out.shape, " attn:", attn.shape)   # out: (1,3,4)  attn: (1,2,3,3)
print("行和(应全为1):", attn.sum(-1).round(3))    # 全 1.0
print("head0 注意力(应严格下三角):\n", attn[0,0].round(3))
#  [[1.    0.    0.  ]
#   [0.51  0.49  0.  ]    ← 上三角为 0 = 因果掩码生效
#   [0.34  0.34  0.32]]

blk = Block(d, nh)
print("Block 输出形状(应与输入相同):", blk(x).shape)   # (1,3,4) —— 保形，可堆叠
```

```python
# —— 堆叠 N 个 Block：因为保形，直接 for 循环就是 GPT 主干 ——
blocks = [Block(d, nh, seed=i) for i in range(4)]   # 4 层
h = x
for blk in blocks:
    h = blk(h)                                      # 每层 (1,3,4)->(1,3,4)
print("4 层后形状:", h.shape)   # (1,3,4) —— 这就是 117 PyTorch 版 GPT 主干的雏形
```

```python
# —— 验证「为何除以 √d_k」：看缩放前后点积的方差 ——
rng = np.random.default_rng(1)
for dk in [16, 64, 256]:
    q = rng.standard_normal((1000, dk)); k = rng.standard_normal((1000, dk))
    dots = (q * k).sum(-1)                           # 逐对点积
    print(f"d_k={dk:3d}  点积 std={dots.std():6.2f} (≈√d_k={np.sqrt(dk):5.2f}) "
          f" 除以√d_k 后 std={ (dots/np.sqrt(dk)).std():.2f}")
# d_k=16  点积 std≈ 4.0 (≈√16=4)   除以√d_k 后 std≈1.0
# d_k=64  点积 std≈ 8.0 (≈√64=8)   除以√d_k 后 std≈1.0
# d_k=256 点积 std≈16.0 (≈√256=16) 除以√d_k 后 std≈1.0
# 结论：不缩放，d 越大 logits 越大、softmax 越饱和、梯度越小；缩放把方差归一回 1

# —— 验证因果掩码：漏掉 mask，位置 0 会「偷看」未来，行不再下三角 ——
def attn_no_mask(x, Wq, Wk, Wv, Wo, n_head):         # ❌ 故意漏掉因果掩码
    B, T, d = x.shape; hs = d // n_head
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    split = lambda z: z.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)
    Q, K, V = split(Q), split(K), split(V)
    return softmax(Q @ K.transpose(0,1,3,2) / np.sqrt(hs), axis=-1)
a_bad = attn_no_mask(x, Wq, Wk, Wv, Wo, nh)
print("漏掉掩码后 head0(上三角不再为 0 = 偷看未来):\n", a_bad[0,0].round(3))
# 训练时这会导致「未来信息泄漏」：train loss 极低，但推理时没有未来可看 → 崩
```

```python
# ============ 扩展练习 ============
# 1. 把 n_head 改成 1（单头），对比注意力矩阵和多头的差异。
# 2. 加位置编码（正弦或 RoPE），让模型「知道」token 顺序(纯注意力是置换等变的)。
# 3. 给 Block 加 dropout（前向时随机置零），理解为何推理要关掉它。
# 4. 用 114 的 Value 张量化版给这个 Block 写反传，在玩具数据上训练一步验证。
# 5. 把 scores 的显式构造改成「分块 online-softmax」，体会 Flash-Attention 的省显存思路。
```

## 面试高频

- **Q:手写一下缩放点积注意力。** A:`Q@K.T/sqrt(d_k)` 得相似度 → 加因果掩码(上三角 $-\infty$)→ `softmax`(行归一)→ `@V` 加权平均。形状 $(T,d)\to(T,d)$ 保持。
- **Q:多头注意力怎么实现「拆头」?** A:不开 $h$ 套权重;用整块 $W\in\mathbb{R}^{d\times d}$ 投影后 reshape 成 $(B,T,h,d_k)$ 再 transpose 成 $(B,h,T,d_k)$,批量矩阵乘一次算完所有头,最后合头过 $W_o$。
- **Q:因果掩码怎么写?为什么必需?** A:`np.tril` 取下三角,上三角位置在 softmax 前填 $-10^9$(≈ $-\infty$),softmax 后变 0。自回归训练里位置 $t$ 不能看未来,否则训练时偷看答案、推理时崩(信息泄漏)。
- **Q:Pre-LN 和 Post-LN Block 区别?** A:Pre-LN 把 LN 放子层「前」($x+f(\mathrm{LN}(x))$),残差是干净恒等路径,深层稳、是现代标配;Post-LN 放子层后,需谨慎 warmup(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN 与 Post-LN]])。
- **Q:为什么注意力是「形状保持」的?这有什么用?** A:输入输出都是 $(T,d)$,所以能无限堆叠 Block——GPT 就是 N 个同构 Block 的 `for` 循环。
- **Q:FFN 为什么升维 4 倍?** A:经验配方,给注意力后的特征一个高维非线性变换空间;$d\to4d\to d$,中间 GELU(见 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]])。
- **Q:为什么是除以 $\sqrt{d_k}$,不是别的?** A:Q、K 各维独立方差 1 时,点积方差 = $d_k$、标准差 = $\sqrt{d_k}$;除以它把 logits 方差归一回 1,避免 $d$ 大时 softmax 饱和、梯度消失。可代码验证:点积 std ≈ √d_k。
- **Q:LayerNorm 对哪一维归一?和 BatchNorm 区别?** A:对每个 token 的特征维($d$)归一,不跨样本、不依赖 batch 统计,所以推理与训练行为一致、对变长序列友好;BatchNorm 对 batch 维归一、要存 running stats。
- **Q:纯注意力能感知顺序吗?** A:不能,自注意力是置换等变的(打乱 token 顺序输出也跟着打乱、不变内容);必须靠位置编码注入顺序信息(正弦/可学习/RoPE)。
- **Q:numpy 版和生产注意力差在哪?** A:数学一致,差效率:Flash-Attention(不落地 $T\times T$ 矩阵、$O(T)$ 显存)、KV-Cache(不重算历史)、GQA/MQA(KV 头共享)、融合 kernel。核心公式 $\text{softmax}(QK^T/\sqrt{d}+M)V$ 不变。

## 关键事实

- 缩放点积注意力与多头来自原始 Transformer(Vaswani et al. 2017, "Attention Is All You Need", arXiv:1706.03762);$\sqrt{d_k}$ 缩放、Concat+$W_o$ 多头均出自该文。
- 本实现对应 nanoGPT/minGPT 的 `CausalSelfAttention`(Karpathy),numpy 版仅去掉 autograd 用于理解;真正训练见 [[117 训练一个 tiny GPT(PyTorch,可跑)|PyTorch 版]]。
- 数学依据:自注意力 [[002 自注意力 Self-Attention|自注意力]]、QKV [[003 Query、Key、Value 的设计|QKV]]、缩放 [[004 缩放点积注意力(为何除以根号 dk)|缩放]]、多头 [[005 多头注意力 Multi-Head|多头]]、矩阵形式与形状 [[006 注意力的矩阵形式与张量形状|矩阵形式]]、整体数据流 [[013 Transformer 整体数据流(逐张量形状)|数据流]]。
- Pre-LN 是现代 decoder-only 标配,深层训练更稳(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]、[[015 Transformer 训练稳定性|训练稳定性]]);GELU 近似式 $0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ 是 GPT-2 用法。
- softmax 减最大值是数值稳定标配(防 `exp` 上溢);因果掩码用 $-10^9$ 而非真 $-\infty$ 以免 `exp(-inf)` 产生 NaN 风险。
- $\sqrt{d_k}$ 缩放的理由:Q、K 独立方差 1 时点积方差为 $d_k$,缩放把 logits 方差归一回 1、避免 softmax 饱和与梯度消失(代码可验证点积 std≈√d_k);LayerNorm 对特征维归一、不依赖 batch 统计,故推理训练一致、对变长序列友好。
- 自注意力置换等变,顺序信息必须靠位置编码注入(正弦/可学习/RoPE,见 [[029 正弦余弦位置编码(推导)|正弦编码]]、[[031 RoPE 旋转位置编码(推导与实现)|RoPE]]);本 numpy 版未含位置编码,仅演示注意力本体。
- numpy 版与生产差在效率:Flash-Attention 不落地 $T\times T$ 矩阵($O(T)$ 显存)、KV-Cache 不重算历史、GQA/MQA 共享 KV 头、融合 kernel,核心公式不变。
- 关联:注意力 $O(n^2)$ 瓶颈 [[014 注意力复杂度 O(n²) 与瓶颈|复杂度]];效率变体 MQA/GQA/MLA [[018 MQA 多查询注意力|MQA]]、[[019 GQA 分组查询注意力|GQA]];总入口 [[113 从零实现总览：课程地图到代码|从零实现总览]];反传机制 [[114 手写自动微分引擎(micrograd 级)|自动微分]]。
