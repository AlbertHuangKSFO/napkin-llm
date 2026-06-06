[[001 Transformer 总览：为何抛弃 RNN|Transformer 总览]] —— Transformer 是 2017 年提出的、**完全靠注意力**(不用循环、不用卷积)的序列模型;它用 [[002 自注意力 Self-Attention|自注意力]] 让序列里所有 token 一次互相"看到",从而既能**并行训练**又能**直接建模长距离依赖**,解决了 [[54 RNN 原理与 BPTT|RNN]] 的两大死穴。

## 直觉
把"读一句话"想象成开会。

- **RNN 模式**:大家排成一队,只能耳语传话——第 1 个人把消息小声传给第 2 个,第 2 个再传第 3 个……到第 50 个人时,最初的消息早被传歪、传丢了(**长依赖衰减**);而且后面的人必须等前面的人说完才能开口(**不能并行**)。
- **Transformer 模式**:所有人坐在一张圆桌旁,每个人想了解谁就**直接转头问谁**,一轮之内人人都能听到所有人。没有"传话链",任意两人距离都是"一步";而且所有人可以**同时**发问——这正好对上 GPU 喜欢的"大批量矩阵乘"。

代价是:n 个人两两都要看一眼,得看 $n^2$ 次(见 014 注意力复杂度)。

再补一个反直觉点:**Transformer 一开始对"谁先谁后"完全无感**。圆桌没有座次,删掉循环后模型不知道"猫"在"狗"前面还是后面。RNN 自带顺序(传话链有方向),Transformer 必须靠**位置编码**把顺序信息显式塞回去——这是"用并行换来的债",后面要专门还(位置编码笔记)。

## 三种序列建模范式的总账

| 维度 | RNN/LSTM | CNN(1D 卷积) | Transformer |
|---|---|---|---|
| 任意两 token 路径长度 | $O(n)$ | $O(n/k)$(k=核宽,需多层堆) | $O(1)$ |
| 每层时间复杂度 | $O(n\cdot d^2)$ | $O(k\cdot n\cdot d^2)$ | $O(n^2\cdot d)$ |
| 序列方向可并行 | ❌ 串行 $n$ 步 | ✅ | ✅ |
| 长依赖 | 难(梯度连乘衰减) | 中(感受野受限) | 易(直接一跳) |
| 顺序感知 | 自带 | 自带(局部) | 需位置编码 |

这张表正是《Attention Is All You Need》表 1 的精简版:Transformer 用"每层 $O(n^2 d)$ 的算力"换"路径长度 $O(1)$ + 完全并行"。当 $n<d$(短序列、宽模型)时,$O(n^2 d)$ 甚至比 RNN 的 $O(n d^2)$ 还便宜;只有长序列才暴露 $n^2$ 的痛(见 014)。

## 例子
一句话:`猫 累了 它 睡 了`。要判断"它"指代谁。

- RNN 处理"它"时,关于"猫"的信息已经过了 2 个时间步,被反复乘以权重矩阵后衰减,可能记不清。
- 自注意力让"它"这个 token 直接对"猫"打一个高分(它们语义相关),**一步**就把"猫"的信息加权拿过来。距离 5 还是距离 500,代价一样是"一跳"。

再看并行:RNN 算第 5 个词的隐状态 $h_5$ 必须先算完 $h_1,h_2,h_3,h_4$;Transformer 把整句话堆成一个矩阵 $X\in\mathbb{R}^{L\times d}$,一次矩阵乘就同时算出全部 token 的表示。

![[tf-RNN对比并行.svg]]

## 原理
**RNN 的递推**:$h_t = f(W_h h_{t-1} + W_x x_t)$。两个结构性问题:

1. **顺序依赖,无法并行**:$h_t$ 显式依赖 $h_{t-1}$,时间维上必须串行走 $L$ 步。训练时这条链阻止了在序列长度方向上的并行,GPU 利用率低。
2. **长依赖衰减 / 梯度问题**:把 $h_t$ 对早期输入 $x_1$ 求导,会出现连乘 $\prod \frac{\partial h_{t}}{\partial h_{t-1}}$(见 [[54 RNN 原理与 BPTT|BPTT]]),特征值 <1 则梯度指数消失、>1 则爆炸,远距离信号难以传递。LSTM/GRU 缓解但没根除。[[58 Seq2Seq 编码器与解码器|Seq2Seq]] 还有个额外瓶颈:整句被压成**一个**定长向量,信息过载。

**注意力的破解**:Bahdanau / Luong 的注意力起源(见 [[60 注意力机制的起源(Bahdanau、Luong)|注意力起源]])让解码器每步回看整个输入序列、按需取信息,缓解了定长瓶颈。Transformer 把这一招推到极致——**索性把循环全删掉**,encoder/decoder 内部也用注意力:

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **路径长度**:任意两 token 间的信息传播路径从 RNN 的 $O(n)$ 降到 $O(1)$,长依赖天然好学。
- **并行度**:每层的计算是大矩阵乘,序列方向完全并行(对照见 [[13 张量、广播与维度对齐|张量与维度对齐]])。
- **位置信息**:删掉循环后模型本身对顺序"无感",必须额外注入**位置编码**(否则 `猫追狗` 和 `狗追猫` 在模型眼里一样)。

整体一层 Transformer block = 多头自注意力 + [[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|FFN]] 前馈网络,各自外面套 [[52 残差连接与深度可训练性|残差连接]] 与 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|层归一化]]。完整逐张量数据流见 013。

权衡一句话:**用 $O(n^2)$ 的算力换 $O(1)$ 的路径长度和完全并行**。在 GPU 时代这笔买卖极划算,这是 Transformer 取代 RNN 的根本原因。

**原论文那张"复杂度表"怎么读**(论文 Table 1,逐列拆给零基础看):
- **Complexity per layer(每层算力)**:Self-Attention 是 $O(n^2\cdot d)$,Recurrent 是 $O(n\cdot d^2)$。当 $n<d$(短句、宽模型)时注意力反而更省;$n>d$ 才贵。
- **Sequential Operations(必须串行几步)**:Self-Attention 是 $O(1)$(一层内一把算完),Recurrent 是 $O(n)$(逐步走)。这一列才是并行优势的真正出处——GPU 怕的是"必须等上一步"。
- **Maximum Path Length(最长信息路径)**:Self-Attention $O(1)$,Recurrent $O(n)$,卷积 $O(\log_k n)$。路径越短,长依赖越好学。

**为什么不是 CNN?** CNN 也能并行,但每层只有 $k$(核宽)大小的感受野,要让远端两 token 相连得堆 $O(n/k)$ 层;Transformer 一层就让任意两 token 直连。这是"为何抛弃 RNN"的同一逻辑延伸到 CNN:都输在"远距交互要么串行、要么堆很多层"。

**一个常被忽略的红利:可解释性**。注意力矩阵 $A$ 是显式的、可可视化的"谁看了谁",能画出"它→猫"的指代连线;RNN 的隐状态是黑箱向量,看不出这种对齐。这让 Transformer 在调试和分析上也占便宜。

## 代码
对比"RNN 必须串行" vs "自注意力一次并行"(只示意前向传播形状):

```python
import numpy as np

L, d = 5, 8                      # 5 个 token,维度 8
X = np.random.randn(L, d)

# ❌ 朴素:RNN 风格,序列方向必须 for 循环,无法并行
def rnn_forward(X, Wx, Wh):
    h = np.zeros(Wh.shape[0])
    outs = []
    for t in range(len(X)):      # 第 t 步必须等第 t-1 步
        h = np.tanh(X[t] @ Wx + h @ Wh)
        outs.append(h)
    return np.stack(outs)

# ✅ 高效:自注意力,一次矩阵乘搞定全部 token(无 for over L)
def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (L, dk) 一次算完
    scores = Q @ K.T / np.sqrt(Wk.shape[1])   # (L, L) 全 token 互相打分
    scores -= scores.max(axis=1, keepdims=True)
    A = np.exp(scores); A /= A.sum(axis=1, keepdims=True)
    return A @ V                              # (L, dv) 加权求和

Wq = Wk = Wv = np.random.randn(d, d)
print(self_attention(X, Wq, Wk, Wv).shape)    # (5, 8),序列方向零循环
```

关键差异:RNN 的 `for t` 是**数据依赖**循环(去不掉);自注意力里没有沿序列的循环,整块交给 BLAS 矩阵乘,GPU 满载。

**再加一段:为什么"训练快"不等于"推理快"。** 自回归生成时,第 $t$ 个 token 要等第 $t-1$ 个生成完才能开始——推理重新变回串行!这是 Transformer 没躲掉的命:训练时整句已知,可并行算所有位置的 loss;推理时下一词未知,只能一个个吐。所以推理慢的锅不在架构并行性,而在自回归本身。缓解办法是 [[102 KV-Cache|KV-Cache]](缓存已算过的 K、V,每步只算新 token)和投机解码(speculative decoding)。

```python
# 训练:整句并行,一次前向算出所有位置的"下一词"预测(配因果掩码,见 007)
logits = model(full_sequence)        # (L, V),一把出 L 个预测,可并行
# 推理:自回归,逐 token,无法并行(每步依赖上一步的输出)
tokens = [bos]
for _ in range(max_len):             # ← 串行循环又回来了
    next_tok = model(tokens)[-1].argmax()
    tokens.append(next_tok)
```

## 面试高频
- **Q:Transformer 为什么比 RNN 快?** A:RNN 在序列长度方向有数据依赖,必须串行 $O(L)$ 步;自注意力把整段序列一次矩阵乘,序列方向完全并行,GPU 利用率高。注意"快"指**训练并行**;自回归**推理**时仍是逐 token 串行(这正是 [[102 KV-Cache|KV-Cache]] 要优化的)。
- **Q:Transformer 怎么解决长依赖?** A:任意两 token 间路径长度 $O(1)$,不经过递推连乘,信号不衰减;对照 RNN 的 $O(n)$ 路径。
- **Q:删了循环带来什么新问题?** A:模型失去顺序感,必须加位置编码;同时复杂度从 RNN 的 $O(n\cdot d^2)$ 变成注意力的 $O(n^2 d)$(见 014),长序列变贵。
- **Q:Transformer 比 RNN 多了哪些超参数/设计选择?** A:头数 $h$、模型维 $d$、层数 $N$、FFN 扩张比、位置编码方案(绝对/相对/RoPE)、Pre/Post-LN——RNN 没有"头"和"位置编码"这两组概念。
- **Q:既然注意力是 $O(n^2)$,为什么短序列上反而可能比 RNN 快?** A:RNN 每层 $O(n d^2)$ 但要串行 $n$ 步,GPU 利用率低;注意力 $O(n^2 d)$ 但一层并行算完。当 $n<d$ 时连算力都更少,加上并行,实际墙钟时间(wall-clock)远快于 RNN。
- **Q:LSTM 不是已经解决长依赖了吗,为什么还要 Transformer?** A:LSTM 用门控**缓解**了梯度消失,但没消除串行瓶颈,路径仍是 $O(n)$,远距依赖仍随距离衰减;且无法在序列方向并行。Transformer 把路径压到 $O(1)$ 且全并行,是质变不是量变。
- **陷阱**:别说"Transformer 没有顺序信息所以更好"——它**需要**位置编码补回顺序,否则是个词袋(`猫追狗`=`狗追猫`)。也别把"训练并行"和"推理并行"混为一谈;更别把"$O(1)$ 路径长度"误说成"$O(1)$ 算力"——算力是 $O(n^2 d)$,是路径长度才 $O(1)$。

## 关键事实
- Transformer 出自 Vaswani et al., *Attention Is All You Need*, NeurIPS 2017,arXiv:1706.03762。原文用 6 层 encoder + 6 层 decoder,$d_{model}=512$,8 个注意力头。
- 论文核心主张:"dispensing with recurrence and convolutions entirely"(彻底抛弃循环与卷积),只靠注意力。
- 注意力机制本身更早,源自 Bahdanau et al. 2015(arXiv:1409.0473)与 Luong et al. 2015,Transformer 是把它做成唯一构件。
