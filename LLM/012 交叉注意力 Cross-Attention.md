[[012 交叉注意力 Cross-Attention]]:Encoder-Decoder 里连接两段序列的注意力——**Query 来自解码器(目标序列)、Key 和 Value 来自编码器(源序列)**,让解码器在生成每个词时去"查"源句、实现源↔目标对齐;数学公式和自注意力完全一样,只是 Q、K、V 的**来源不同**。

## ① 直觉:解码时回头"查"源句

[[002 自注意力 Self-Attention|自注意力]]里,Q、K、V 都来自**同一句**,每个词在自己句子内部建立关系。但在翻译这种任务里,解码器一边生成译文,一边需要不断回头看**原文**——"我现在要生成的这个词,对应原文的哪一部分?"

交叉注意力(cross-attention,也叫 encoder-decoder attention)就是干这个的:

- **Query**:解码器当前位置的状态(它在"提问":我该关注源句的哪里?)
- **Key / Value**:编码器输出的源句表示(每个源词的"招牌 K"和"内容 V")
- 解码器用自己的 Q 去和源句的 K 打分,得到对源句各位置的注意力权重,再加权 V,把源句信息**路由**进当前生成步。

翻译 "I love noodles → 我爱面",解码 "爱" 时,交叉注意力会把大部分权重放在源句的 "love" 上——这就是经典的"软对齐"。

![[tf-交叉注意力连线.png]]

## ② 例子:解码"爱"时的交叉注意力小算一遍

编码器输出 3 个源词向量(K=V,简化为 2 维):

$$
k_{\text{I}}=v_{\text{I}}=[1,0],\quad k_{\text{love}}=v_{\text{love}}=[0,1],\quad k_{\text{noodles}}=v_{\text{noodles}}=[1,1]
$$

解码器当前 "爱" 的 Query $q=[0,\,2]$(它"想找"和第二维相关的源词)。

打分([[004 缩放点积注意力(为何除以根号 dk)|缩放点积]],$\sqrt{d_k}=\sqrt2\approx1.41$):

$$
\frac{q\cdot k_{\text{I}}}{\sqrt2}=\frac{0}{1.41}=0,\quad \frac{q\cdot k_{\text{love}}}{\sqrt2}=\frac{2}{1.41}\approx1.41,\quad \frac{q\cdot k_{\text{noodles}}}{\sqrt2}=\frac{2}{1.41}\approx1.41
$$

[[27 Softmax 与温度|softmax]] $([0,\,1.41,\,1.41])$:

$$
\frac{e^{0}}{e^{0}+2e^{1.41}}\approx0.11,\quad \frac{e^{1.41}}{\dots}\approx0.45,\quad \frac{e^{1.41}}{\dots}\approx0.45
$$

权重 $[0.11,\,0.45,\,0.45]$,加权 V:

$$
0.11[1,0]+0.45[0,1]+0.45[1,1]=[0.56,\,0.90]
$$

解码器拿到的这个向量,主要由 "love" 和 "noodles" 贡献——它成功从源句里"取"到了和"爱"相关的语义。

## ③ 原理:和自注意力同公式,只换 Q/K/V 来源

两者的核心公式都是缩放点积注意力(见 [[006 注意力的矩阵形式与张量形状|矩阵形式]]):

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V
$$

差别**只在投影来源**。设编码器输出 $H_{enc}\in\mathbb{R}^{m\times d}$(源句长 $m$),解码器当前层输入 $H_{dec}\in\mathbb{R}^{n\times d}$(目标长 $n$):

| | 自注意力(解码器内) | **交叉注意力** |
|---|---|---|
| $Q$ | $H_{dec}W^Q$ | $H_{dec}W^Q$ |
| $K$ | $H_{dec}W^K$ | $H_{enc}W^K$ |
| $V$ | $H_{dec}W^V$ | $H_{enc}W^V$ |
| 形状 | $QK^\top\in\mathbb{R}^{n\times n}$ | $QK^\top\in\mathbb{R}^{n\times m}$ |

注意交叉注意力的分数矩阵是 $n\times m$(目标 × 源)的**长方形**,不是方阵。

**解码器一个 block 的三明治结构**(原版 Encoder-Decoder):

1. **因果自注意力**:目标序列内部,带 [[007 因果掩码与 padding 掩码|因果掩码]],只看已生成的左侧;
2. **交叉注意力**:Q 来自上一步输出,K/V 来自编码器,去"查"源句;**这里不需要因果掩码**(源句整句早已编码完,可全部看到),但可能有源句的 padding 掩码;
3. **FFN**。每个子层都配 [[009 残差连接与梯度流|残差]] + [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]]。

![[tf-decoder三明治.png]]

**Decoder-Only 模型(GPT)没有交叉注意力**:它把源内容直接拼进 prompt,用因果自注意力扫左侧来"读"源句,因此整个模型只有一种注意力。交叉注意力是 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|Encoder-Decoder]] 架构(T5、BART)的标志。

**交叉注意力的推理优化:编码器 KV 只算一次**。生成一句话要解码很多步,但**源句的 K、V 在每一步都一样**(源句没变)。所以实现里把编码器输出的 $K_{enc},V_{enc}$ **算一次、缓存住**,后续每个解码步直接复用,不重算编码器。这和解码器自注意力的 [[102 KV-Cache|KV-Cache]] 是两套缓存:自注意力缓存"已生成目标 token"的 KV(逐步增长),交叉注意力缓存"源句"的 KV(固定不变)。这是个高频追问点。

**为什么交叉注意力分数矩阵是长方形而非方阵**。自注意力 Q、K 同源,长度都是 $L$,故 $L\times L$ 方阵;交叉注意力 Q 来自目标($n$ 个)、K 来自源($m$ 个),故 $n\times m$ 长方形——目标每个位置对源每个位置打一分。softmax 仍沿 key 维(源维 $m$)做,每个目标位置对源的注意力和为 1。

**和 RAG / 检索的关系(联系现代系统)**。交叉注意力本质是"用 query 去一堆 key-value 里检索并融合"——这和 RAG(检索增强生成)里"用问题去文档库检索、再把检索结果喂进生成"是同构思想,只是粒度不同:交叉注意力在 token 级、可微、端到端训;RAG 在文档级、用向量检索(见 [[04 Embedding 与向量数据库|向量检索]])。理解交叉注意力有助于理解为什么"检索 + 融合"是处理外部知识的通用范式。

## ④ 代码:自注意力 vs 交叉注意力,只差来源

```python
import numpy as np

def attention(Q, K, V):
    dk = Q.shape[-1]
    s = Q @ K.T / np.sqrt(dk)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return w @ V, w

d = 2
H_dec = np.array([[0.,2],[1,1]])        # 解码器 2 个 token
H_enc = np.array([[1.,0],[0,1],[1,1]])  # 编码器 3 个源 token
Wq = Wk = Wv = np.eye(d)                 # 简化:投影设为单位阵

# ❌ 误把 K/V 也取自解码器 → 那是自注意力,根本没看源句
out_self, _ = attention(H_dec@Wq, H_dec@Wk, H_dec@Wv)   # (2,2),只在目标内部混

# ✅ 交叉注意力:Q 来自 decoder,K/V 来自 encoder
out_cross, w = attention(H_dec@Wq, H_enc@Wk, H_enc@Wv)   # (2,2)
print("交叉注意力权重形状:", w.shape)   # (2, 3) = 目标 × 源,长方形!
print("第2个目标词对3个源词的权重:", w[1].round(2))
```

```python
# PyTorch:同一个 MultiheadAttention,传不同的 query/key/value 即可
import torch, torch.nn as nn
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
# 自注意力:三者同源
self_out, _  = mha(dec, dec, dec)
# ✅ 交叉注意力:query=解码器,key=value=编码器输出
cross_out, _ = mha(dec, enc_out, enc_out)
```

## 面试高频

- **Q:交叉注意力的 Q、K、V 各来自哪?** Q 来自解码器(目标序列),K 和 V 来自编码器输出(源序列)。
- **Q:它和自注意力公式上有什么区别?** 公式完全相同($\text{softmax}(QK^\top/\sqrt{d_k})V$),只是 K/V 换了来源;因此分数矩阵是 $n\times m$(目标×源)的长方形而非方阵。
- **Q:交叉注意力需要因果掩码吗?** 不需要——源句整句已编码完毕,解码器可以看到源句全部位置;因果掩码只用在解码器的**自注意力**上。
- **Q:Decoder-Only 模型有交叉注意力吗?** 没有,它用 prompt 拼接 + 因果自注意力替代;交叉注意力是 Encoder-Decoder(T5/BART)的标志。
- **Q:交叉注意力在直觉上对应什么?** 神经机器翻译里的"软对齐",可视化权重能看到目标词主要对齐到哪些源词,是可解释性的经典例子。
- **Q:解码器 block 里注意力出现几次、顺序如何?** 两次:先因果自注意力(看已生成的目标),再交叉注意力(看源句),最后 FFN,各配残差+LN。
- **Q:交叉注意力推理时编码器要每步重算吗?** 不用——源句的 K、V 算一次缓存住,每个解码步复用;这与解码器自注意力缓存"已生成 token KV"是两套独立缓存。
- **Q:交叉注意力为什么分数是长方形?softmax 沿哪维?** Q 来自目标($n$)、K 来自源($m$),故 $n\times m$;softmax 沿源维 $m$,每个目标位置对源的注意力和为 1。
- **Q:Decoder-Only 把源拼进 prompt,和真正的交叉注意力有什么差别?** 拼接靠因果自注意力扫左侧源句,源和目标共用一套注意力、混在同一序列里;交叉注意力是独立模块、源被双向编码后再被查,信息更充分但模块更多。
- **Q:交叉注意力和 RAG 是什么关系?** 同构思想"用 query 检索 key-value 并融合":交叉注意力在 token 级、可微、端到端;RAG 在文档级、用向量检索。

## 关键事实

- 交叉注意力(encoder-decoder attention)出自《Attention Is All You Need》(Vaswani et al., 2017):解码器每层在自注意力之后插入一层 Q 来自解码器、K/V 来自编码器输出的注意力。
- 其思想源自神经机器翻译的对齐注意力(Bahdanau et al., 2014;Luong et al., 2015),即 [[60 注意力机制的起源(Bahdanau、Luong)]]。
- T5、BART、mBART 等 Encoder-Decoder 模型以交叉注意力为核心的信息路由机制;Decoder-Only 模型(GPT 系)不含交叉注意力。
- 交叉注意力的分数矩阵形状为 (目标长 $n$ × 源长 $m$),与自注意力的方阵 ($n\times n$) 不同。
