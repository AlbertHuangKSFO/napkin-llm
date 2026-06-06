[[007 因果掩码与 padding 掩码]]:两种注意力掩码——**因果掩码**用下三角把"未来"设为 -∞(自回归生成的核心),**padding 掩码**屏蔽补齐用的 `<pad>`,二者本质都是"在 softmax 之前把不该看的位置加上 -∞"。

## ① 直觉:掩码就是"蒙住眼睛不让看"

[[002 自注意力 Self-Attention|自注意力]]里,每个词都会对句中所有词打分、加权求和。但有些位置**不该被看到**:

- **因果掩码(causal / look-ahead mask)**:训练时整句并行喂进去,如果第 2 个词能看到第 3、4 个词,就等于提前看了答案——这叫数据泄露。我们要让"生成第 t 个词时只能依赖前 t 个词",于是把每一行里"右边(未来)"的位置全部蒙住。形状固定为**下三角放行**。这正是 [[036 GPT 系列：自回归与规模化|GPT 自回归]]能成立的根。
- **padding 掩码**:一个 batch 里句子长短不一,短句要右侧补 `<pad>` 凑到同一长度。这些 `<pad>` 是凑数的,不该分走注意力,于是把它们对应的列蒙住。形状随每个样本的句长变化。

"蒙住"的技术手段是统一的:在打分矩阵上,把要屏蔽的位置加上 `-∞`,过完 softmax 它们自然变成概率 0。

## ② 例子:4 个词的因果掩码手算

句子"我 / 爱 / 吃 / 面",[[004 缩放点积注意力(为何除以根号 dk)|缩放点积]]后得到 4×4 分数矩阵(行=当前 query 词,列=被看的 key 词)。掩码矩阵 $M$ 是:

$$
M=\begin{bmatrix}0 & -\infty & -\infty & -\infty\\ 0 & 0 & -\infty & -\infty\\ 0 & 0 & 0 & -\infty\\ 0 & 0 & 0 & 0\end{bmatrix}
$$

下三角(含对角线)= 0(放行),上三角 = $-\infty$(屏蔽)。把它加到分数上,以第 2 行(query="爱")为例,原始分数 $[0.2,\,0.8,\,0.5,\,0.2]$ 变成:

$$
[0.2,\;0.8,\;0.5+(-\infty),\;0.2+(-\infty)] = [0.2,\;0.8,\;-\infty,\;-\infty]
$$

过 [[27 Softmax 与温度|softmax]]:

$$
\frac{e^{0.2}}{e^{0.2}+e^{0.8}}\approx 0.35,\quad \frac{e^{0.8}}{e^{0.2}+e^{0.8}}\approx 0.65,\quad e^{-\infty}\to 0,\quad e^{-\infty}\to 0
$$

结果 $[0.35,\,0.65,\,0,\,0]$:"爱"只把注意力分给"我"和它自己,完全看不到右边的"吃""面"。每一行都这样处理,得到上三角恒为 0 的权重矩阵。

![[tf-因果掩码下三角.svg]]

**padding 掩码**:句 A="我 爱 面"补两个 `<pad>` 到长度 5,掩码向量是 `[1,1,1,0,0]`,值为 0 的列(第 4、5 个 key)在打分时被加 $-\infty$。解码"爱"时,看真实词正常算,看 `<pad>` 得 $-\infty$,softmax 后真实词权重重新归一到和为 1,不被 `<pad>` 稀释。

![[tf-padding掩码.svg]]

## ③ 原理:为什么在 softmax 之前加 -∞

注意力权重是 $\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}+M\right)$。把掩码 $M$ 加在 softmax **内部**(分数上),而不是事后把权重置 0,有两个好处:

1. **自动归一化**。softmax 第 $i$ 行第 $j$ 列为 $\dfrac{e^{s_{ij}+M_{ij}}}{\sum_k e^{s_{ik}+M_{ik}}}$。当 $M_{ij}=-\infty$ 时分子 $e^{-\infty}=0$,该列权重精确为 0;同时它从**分母里消失**,剩下的合法位置自动重新归一,行和仍为 1。若事后置 0 再不重新归一,行和就不再是 1,加权平均会被破坏。
2. **数值上**用一个很大的负数(如 `-1e9`)代替 $-\infty$,避免 `inf` 参与运算产生 `NaN`。

**两种掩码可叠加**:解码器训练时既不能看未来、也不能看 `<pad>`,把两个 $M$ 直接相加(逻辑上取"允许位置"的交集)即可。

**训练 vs 推理的一致性**:推理时逐词生成,模型本就看不到尚未生成的未来词,理论上不需要因果掩码;但训练时为了并行必须用它,从而保证训练分布与推理分布一致——这是因果掩码真正的意义。

**更多掩码变体(理解了"加 -∞"就能举一反三)**:

| 掩码类型 | 形状/规则 | 用在哪 |
|---|---|---|
| 因果(causal) | 下三角放行,上三角 $-\infty$ | GPT 解码器自注意力 |
| 双向(无掩码) | 全 0,谁都能看谁 | BERT 编码器(见 011) |
| Prefix-LM | 前缀部分双向、生成部分因果(L 形) | UniLM、GLM、T5 的 prefix 设定 |
| 滑动窗口(sliding window) | 只放行 $\lvert i-j\rvert\le w$ 的对角带 | Mistral、Longformer 局部注意力 |
| 块对角(block-diagonal) | 同一文档内放行、跨文档 $-\infty$ | 多文档拼包训练(避免串味) |
| padding | 屏蔽 `<pad>` 列 | 所有变长 batch |

它们本质都是"在 $S$ 上某些位置加 $-\infty$",只是放行集合不同。**Prefix-LM** 尤其常考:它让 prompt(前缀)内部双向互看、只对要生成的部分用因果——结合了理解和生成的长处。

**滑动窗口掩码的代价**:把全连接的 $O(L^2)$ 降到 $O(L\cdot w)$($w$=窗口宽),是稀疏注意力的一种(见 [[022 稀疏注意力(BigBird、块稀疏)|稀疏注意力]]);但远端 token 要靠多层逐步"接力"才能间接相连,牺牲了部分长依赖。

**因果掩码 + KV-Cache 的配合**:推理时已生成的 token 的 K、V 被缓存(见 [[102 KV-Cache|KV-Cache]]),新 token 的 query 只对"自己 + 缓存里的历史"打分——天然只看左侧,所以**增量推理时其实不需要显式建因果掩码矩阵**(query 长度为 1,看全部缓存即合法)。因果掩码主要是**训练并行**时才需要的东西。这是个高频追问点。

## ④ 代码:numpy 手写因果掩码 + softmax

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

scores = np.array([[0.9, 0.3, 0.1, 0.4],
                   [0.2, 0.8, 0.5, 0.2],
                   [0.4, 0.3, 0.7, 0.1],
                   [0.1, 0.5, 0.6, 0.9]])   # (seq, seq)

# ❌ 错:不加掩码,每个词都能看到未来 → 训练时偷看答案
w_bad = softmax(scores)

# ✅ 对:上三角(不含对角线)填 -∞,再 softmax
seq = scores.shape[0]
causal = np.triu(np.ones((seq, seq)), k=1).astype(bool)   # 上三角 True
masked = np.where(causal, -1e9, scores)                   # 用 -1e9 代替 -inf
w_good = softmax(masked)

print(w_good.round(2))
# [[1.   0.   0.   0.  ]   第1词只看自己
#  [0.35 0.65 0.   0.  ]   第2词看前2个
#  [0.3  0.28 0.42 0.  ]
#  [0.18 0.27 0.3  0.25]]  上三角恒为 0
```

PyTorch 里 `nn.MultiheadAttention` 用 `attn_mask`(因果)+ `key_padding_mask`(padding)两个参数分别承担这两件事;`F.scaled_dot_product_attention(..., is_causal=True)` 可一行开启因果掩码。

**两种掩码叠加 + padding 行的坑**:

```python
# 因果掩码(下三角放行)+ padding 掩码(屏蔽 <pad> 列),直接相加
def build_mask(seq, pad_mask):           # pad_mask: (seq,) 1=真实 0=pad
    causal = np.triu(np.ones((seq, seq)), k=1).astype(bool)   # 上三角 True=屏蔽
    pad = (pad_mask[None, :] == 0)        # 广播成 (1, seq),pad 列 True=屏蔽
    return causal | pad                   # 任一为真即屏蔽(取交集放行)

# ⚠️ 全 <pad> 行的陷阱:某个 query 位置本身是 pad,它那一行可能整行被屏蔽,
#    softmax(全 -inf) = NaN(分母为 0)。实现里要么把 pad query 行的 loss 屏蔽掉,
#    要么给对角线留一个合法位防止整行 -inf。
```

这个 "全屏蔽行 → softmax NaN" 是工程上真实会爆的 bug:padding 位置作为 query 时若整行 $-\infty$,$e^{-\infty}/\sum e^{-\infty}=0/0=$ NaN。生产代码要么在 loss 里 mask 掉 pad 位置(它们的输出本就不计入损失),要么强制对角线可见。

## 面试高频

- **Q:为什么 GPT 训练要因果掩码,而 BERT 不要?** GPT 是[[036 GPT 系列：自回归与规模化|自回归]]语言模型,预测下一个词,必须禁止看未来;[[035 BERT：双向编码与 MLM|BERT]]做掩码语言建模(MLM),目标是用双向上下文还原被 `[MASK]` 的词,**需要**看到左右,所以不加因果掩码。
- **Q:为什么把屏蔽位置设成 -∞ 而不是直接把权重置 0?** 因为要在 softmax 内部屏蔽,这样合法位置能自动重新归一(行和=1);事后置 0 会破坏归一化。
- **Q:因果掩码和 padding 掩码能不能合并?** 能,直接相加(取允许位置交集)。解码器训练时两者同时存在。
- **Q:推理生成时还需要因果掩码吗?** 严格说不需要(看不到未来),但为了实现统一、且配合 [[102 KV-Cache|KV-Cache]] 增量计算,通常仍保留逻辑。增量步里 query 长度为 1、看全部已缓存 KV 即合法,无需显式建掩码矩阵。
- **Q:掩码加在哪一步?** 加在 $QK^\top/\sqrt{d_k}$ 之后、softmax 之前。
- **Q:什么是 Prefix-LM 掩码?和纯因果有何不同?** 前缀部分(prompt)内部双向互看,只对要生成的部分用因果,呈 L 形。兼顾理解(前缀充分双向)与生成,用于 UniLM、GLM、T5。
- **Q:滑动窗口注意力是怎么用掩码实现的?复杂度?** 只放行 $\lvert i-j\rvert\le w$ 的对角带,其余 $-\infty$,把 $O(L^2)$ 降到 $O(Lw)$;远端靠多层接力间接相连,牺牲部分长依赖。Mistral、Longformer 用。
- **Q:一个 batch 里拼了多个文档,怎么防止跨文档"串味"?** 用块对角掩码:同文档内放行,跨文档边界 $-\infty$,让注意力不越过文档边界。
- **Q:padding query 行整行被屏蔽会怎样?** softmax(全 $-\infty$)= 0/0 = NaN;需在 loss 里 mask 掉 pad 位置或保留对角线可见。常见工程 bug。

## 关键事实

- 因果(look-ahead)掩码与 padding 掩码均出自《Attention Is All You Need》(Vaswani et al., 2017),解码器的"masked multi-head self-attention"即指因果掩码。
- 工程上 `-∞` 用大负数(如 `-1e9` 或 `float('-inf')`)实现;PyTorch `scaled_dot_product_attention` 2022 年起内置 `is_causal` 与 FlashAttention 后端,把掩码与计算融合以省显存。
- 因果掩码是下三角(含对角线)放行、上三角屏蔽;softmax 后注意力权重矩阵的上三角恒为 0,这是判断模型是否自回归的直观特征。
