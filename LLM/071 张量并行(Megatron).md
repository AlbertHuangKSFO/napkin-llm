[[071 张量并行(Megatron)|张量并行(Megatron)]] 把**单层内部的权重矩阵**横/纵切到多卡:Megatron 的关键技巧是 FFN **先按列切、再按行切**,使两次矩阵乘之间无需通信,每个块只在末尾做一次 **all-reduce**;attention 则**按头分卡**。每个 transformer 层前向 2 次、反向 2 次 all-reduce——通信重且频繁,所以 TP 必须**限在机内**。

## 直觉

[[070 ZeRO 与 FSDP|ZeRO]] 只切存储,算的时候每卡仍要临时凑齐完整权重。但如果**单层的权重矩阵大到一张卡连临时凑齐都放不下**(隐藏维巨大),就得把**计算本身**切开——这就是**张量并行(Tensor Parallelism, TP)**,也叫层内并行。

矩阵乘 $Y=XW$ 怎么切?有两种自然方式:
- **列切(column parallel)**:把 $W$ 按**列**分块 $W=[W_1\,W_2]$,则 $Y=[XW_1\,XW_2]$。每卡算输出的一部分列,**互不依赖**,算完输出天然按卡分块。
- **行切(row parallel)**:把 $W$ 按**行**分块 $W=\begin{bmatrix}W_1\\W_2\end{bmatrix}$,要求输入也按列分块 $X=[X_1\,X_2]$,则 $Y=X_1W_1+X_2W_2$。每卡算一个**部分和**,最后 all-reduce 相加。

Megatron 的妙处:FFN 是两层 $Z=\text{GeLU}(XA)\cdot B$,第一层 $A$ **列切**、第二层 $B$ **行切**。列切的输出正好已按卡分块,直接喂给行切的对应分片——**中间不必通信**,只在末尾 all-reduce 一次把部分和拼回。attention 更自然:多头本就独立,**按头分卡**,每个头的注意力完全在本卡内算完,输出投影行切后 all-reduce。

![[dist-Megatron行列切分.svg]]

## 例子

**2 卡切一个 FFN(隐藏维 h=8、中间维 4h=32)**:

- 第一层 $A\in\mathbb R^{8\times32}$ **列切**成 $A_1,A_2\in\mathbb R^{8\times16}$。GPU0 算 $Y_0=\text{GeLU}(XA_1)$(输出左 16 列),GPU1 算 $Y_1=\text{GeLU}(XA_2)$(右 16 列)。**无通信**——GeLU 逐元素,不跨列。
- 第二层 $B\in\mathbb R^{32\times8}$ **行切**成 $B_1,B_2\in\mathbb R^{16\times8}$。GPU0 算 $Z_0=Y_0B_1$、GPU1 算 $Z_1=Y_1B_2$,各是部分和。
- **all-reduce**:$Z=Z_0+Z_1$。前向就这**一次**通信(整个 FFN)。

**attention 按头切(8 头、2 卡)**:GPU0 管头 1-4、GPU1 管头 5-8。每卡只投影自己 4 个头的 Q/K/V(QKV 列切),本地算完 $\text{softmax}(QK^\top/\sqrt d)V$——**头内自洽,不跨卡**,这正是多头注意力天然可并行的地方。输出投影 $W_o$ 行切,末尾 all-reduce。

**一个 transformer 层的通信账**:attention 块 1 次 + FFN 块 1 次 = **前向 2 次 all-reduce**;反向对称,**再 2 次**。一个 96 层模型,光 TP 就 $96\times4\approx 384$ 次 all-reduce/步,每次同步**完整激活**——这就是 TP 必须走机内 NVLink 的原因。

![[dist-Megatron注意力切分.svg]]

## 原理

**1. 列切与行切的代数。** 矩阵乘 $Y=XW$,$X\in\mathbb R^{b\times h}$,$W\in\mathbb R^{h\times h'}$。

- **列并行**:$W=[W_1\,\cdots\,W_p]$(按输出维 $h'$ 切)。每卡 $Y_i=XW_i$,$Y=[Y_1\,\cdots\,Y_p]$。**前向无通信**;反向对 $X$ 的梯度需 all-reduce(各卡贡献相加)。
- **行并行**:$W=\begin{bmatrix}W_1\\\vdots\\W_p\end{bmatrix}$(按输入维 $h$ 切),要求 $X=[X_1\,\cdots\,X_p]$。每卡 $Y^{(i)}=X_iW_i$(部分和),$Y=\sum_i Y^{(i)}$。**前向末尾 all-reduce**;反向无通信。

**2. FFN 块:列切→行切,只一次通信。** FFN $=\text{GeLU}(XA)B$。$A$ 列切 → $\text{GeLU}(XA_i)=Y_i$ 已按卡分块(GeLU 逐元素不破坏分块);$B$ 行切恰好吃 $Y_i$ → $Z=\sum_i Y_iB_i$,末尾 all-reduce。**中间天然对齐,省掉一次通信**。Megatron 把这对操作封装成 `ColumnParallelLinear` + `RowParallelLinear`。

**3. attention 块:按头切。** 多头注意力 $\text{head}_j=\text{softmax}(Q_jK_j^\top/\sqrt{d})V_j$,各头**独立**。把头分到各卡:Q/K/V 投影按头**列切**(每卡只投影自己的头),本地算完注意力,输出投影 $W_o$ **行切**,末尾 all-reduce。
$$\text{Attn}=\Big[\text{head}_1\,\cdots\,\text{head}_a\Big]W_o,\quad W_o=\begin{bmatrix}W_o^{(1)}\\\vdots\\W_o^{(a)}\end{bmatrix}\Rightarrow \text{Attn}=\sum_j \text{head}_j\,W_o^{(j)}.$$
所以一个块也只 1 次 all-reduce。

**4. 通信算符 $f$ 与 $g$。** Megatron 用一对共轭算符:$f$ 前向恒等、反向 all-reduce;$g$ 前向 all-reduce、反向恒等。每个 transformer 层=1 个 attention + 1 个 FFN,各含一个 $g$ → **前向 2 次、反向 2 次 all-reduce**(论文核心结论)。

**5. 为什么限机内。** 每次 all-reduce 同步的是**完整激活**(大小 $\sim b\cdot s\cdot h$),且频率=层数,通信量巨大。机内 NVLink 带宽数百 GB/s 才扛得住;跨机以太网/IB 几十 GB/s 会被掐死。故 **TP 度数 ≤ 单机 GPU 数**(常 8),跨机靠通信轻的 [[072 流水线并行与气泡|PP]] 和 [[069 数据并行与 AllReduce|DP]]。TP 的好处是**单层权重和激活都被切**,最省单层显存;还能配 [[073 序列并行与上下文并行|序列并行]] 把 LayerNorm/Dropout 的激活也切掉。

## TP 到底切了多少显存(逐项)

TP 度数 $t$,一个 transformer 层里**被切**和**没被切**的东西要分清:
- **被切($1/t$)**:QKVO 权重(4d² 切成 $4d^2/t$)、FFN 权重(8d² → $8d^2/t$)、注意力/FFN 的**中间激活**(QKV 投影输出、注意力分数、FFN 中间维 4d 的激活——这些都按头/列分到各卡)。
- **没被切(每卡仍存完整)**:LayerNorm 的 γ/β(量小可忽略)、**LayerNorm 和 Dropout 的输入/输出激活**(形状 $b\cdot s\cdot d$,TP 切不动)、残差流上的激活。

所以 TP **省了权重和大头计算激活,但 LN/Dropout 激活每卡仍存完整**——这正是 [[073 序列并行与上下文并行|序列并行]] 要补的洞:把这部分也按序列维切到 $1/t$。**TP + SP 一起上,激活才真正降到 $1/t$**。

数值例:d=4096、b=8、s=2048、$t=8$。每层 QKVO+FFN 权重从 $12d^2\times2\text{B}=402$ MB 降到 $50$ MB/卡;但 LN/Dropout 激活 $b\cdot s\cdot d\times2\text{B}=134$ MB 不切,8 卡各存一份纯浪费——SP 把它切成 $17$ MB/卡。

## 嵌入层与输出层也要并行(vocab parallel)

大词表(V=128k)的 embedding 和 lm_head 是 $V\times d$ 的巨阵,单卡也吃力,Megatron 把它们按**词表维**切:
- **输入 embedding**:词表切成 $t$ 段,每卡只持 $V/t$ 行。某 token 若不在本卡负责的词表段,本卡查出 0,末尾 all-reduce 把各卡结果加起来(只有负责它的卡非零)。
- **输出 lm_head + softmax**:logits 是 $b\cdot s\cdot V$,V 很大时这个张量本身就巨大。Megatron 用 **vocab-parallel cross-entropy**:各卡只算自己词表段的 logits 和局部 $\sum e^{z}$,通过一次小 all-reduce 拿到全局归一化分母,**避免把完整 $b\cdot s\cdot V$ 的 logits 物化到单卡**(否则 V=128k、b·s=16k 时光 logits 就 $16384\times128000\times4\text{B}\approx 8$ GB)。这是长上下文/大词表训练的重要省显存技巧。

## 通信量逐项:TP 每步搬多少

每层 4 次 all-reduce(前 2 反 2),每次同步**完整激活** $b\cdot s\cdot d$(fp16,$2bsd$ 字节)。环形 all-reduce 单卡搬 $2\frac{t-1}{t}\cdot 2bsd\approx 4bsd$ 字节/次。L 层一步前向+反向:
$$\text{TP 通信量}\approx 4L\times 4bsd = 16\,L\,b\,s\,d\ \text{字节(量级)}.$$
代入 L=80、b=8、s=2048、d=8192:每步约 $16\times80\times8\times2048\times8192\times2\text{B}\approx 5.6$ TB 的聚合搬运——**全压在机内 NVLink**(900 GB/s 级)才扛得住,跨机以太网(几十 GB/s)直接崩。这就是"TP ≤ 单机卡数"的硬约束的定量来源。对比 PP 每步只在 $p-1$ 处边界各传一个 $bsd$ 激活,通信量小两三个数量级——所以 PP 跨机、TP 机内。

## GQA/MQA、MoE 下的 TP 怎么切

- **GQA/MQA**(见 [[019 GQA 分组查询注意力]]):K、V 头数远少于 Q 头(MQA 只 1 组 KV)。TP 按 Q 头切没问题,但 KV 头不够分时,要么**复制** KV 到各卡(MQA + 高 TP 度),要么让 TP 度 ≤ KV 组数。面试常问:"MQA 模型 TP=8 但只有 1 个 KV 头怎么办?"——答:KV 投影在各卡复制,Q 投影按头切。
- **MoE**:专家走 [[049 专家并行 EP 与 MoE 部署|专家并行(EP)]] 用 all-to-all,**不走 TP**;每个专家内部的 FFN 若仍太大,可在专家内再叠 TP(EP×TP)。
- **TP 度必须整除**:头数、隐藏维、FFN 中间维都要能被 $t$ 整除,否则切不均。

## 代码

```python
import torch, torch.nn as nn

# ✅ 列并行：W 按输出维切，前向无通信，反向 all-reduce 对 X 的梯度
class ColumnParallelLinear(nn.Module):
    def __init__(self, h_in, h_out, tp, rank):
        super().__init__()
        assert h_out % tp == 0
        self.w = nn.Parameter(torch.randn(h_in, h_out // tp))  # 本卡只持有 1/tp 列
    def forward(self, x):
        return x @ self.w                  # 输出已按卡分块（左半/右半），不通信

# ✅ 行并行：W 按输入维切，吃列并行的分块输出，前向末尾 all-reduce
class RowParallelLinear(nn.Module):
    def __init__(self, h_in, h_out, tp, rank):
        super().__init__()
        assert h_in % tp == 0
        self.w = nn.Parameter(torch.randn(h_in // tp, h_out))  # 本卡只持有 1/tp 行
    def forward(self, x_shard):            # x_shard 是上游列并行的本卡分片
        y_partial = x_shard @ self.w       # 部分和
        torch.distributed.all_reduce(y_partial)   # 唯一一次通信，拼成完整输出
        return y_partial

# Megatron FFN：列切 → GeLU → 行切，整段只 1 次 all-reduce
class MegatronFFN(nn.Module):
    def __init__(self, h, tp, rank):
        super().__init__()
        self.A = ColumnParallelLinear(h, 4*h, tp, rank)   # 列切
        self.B = RowParallelLinear(4*h, h, tp, rank)      # 行切
    def forward(self, x):
        return self.B(torch.nn.functional.gelu(self.A(x)))
```

```python
# ❌ 错：两层都列切 → 第二层输入维没对齐分片，中间被迫 all-gather，多一次通信
#   self.A = Column(...); self.B = Column(...)   # 中间要凑齐 Y → 白白多通信
# ✅ 对：列切→行切，列切输出正好是行切输入，中间零通信，仅末尾 1 次 all-reduce

# ❌ 错：把 TP 度数设成跨机（TP=16 跨两机）→ 每层 all-reduce 走慢网，吞吐崩
# ✅ 对：TP ≤ 单机卡数（走 NVLink），跨机维度交给 PP / DP
```

## 面试高频

- **Q:张量并行怎么切 FFN?为什么是列切→行切?** A:第一层权重 A 按列切、第二层 B 按行切。列切的输出已按卡分块(GeLU 逐元素不破坏分块),正好喂给行切的对应分片,中间无需通信;只在末尾 all-reduce 把部分和相加。若两层都列切,中间得多一次 all-gather。
- **Q:张量并行怎么切注意力?** A:按头分卡。Q/K/V 投影列切(每卡只投影自己负责的头),每个头的注意力 softmax(QKᵀ/√d)V 在本卡内自洽算完(头独立),输出投影 Wo 行切,末尾 all-reduce。
- **Q:一个 transformer 层 TP 要几次 all-reduce?** A:前向 2 次(attention 块 1 + FFN 块 1),反向对称 2 次。每次同步完整激活,所以通信重且频繁。
- **Q:为什么 TP 要限在机内?** A:每层都 all-reduce 完整激活,通信量大、频率=层数,需机内 NVLink(数百 GB/s)。跨机带宽(几十 GB/s)会被掐死。所以 TP 度数一般≤单机 GPU 数(如 8),跨机用通信轻的 PP/DP。
- **Q:TP 和 ZeRO 区别?** A:ZeRO 只切存储、每卡仍跑完整计算(实现简单、不改模型);TP 切单层矩阵乘的计算本身(要改模型为 Column/RowParallel、限机内)。模型放不下先 ZeRO,单层都超卡再加 TP。
- **Q:序列并行和 TP 什么关系?** A:Megatron-SP 在 TP 基础上,把 LayerNorm/Dropout 这些非矩阵乘部分的激活也按序列维切(TP 切不了它们),进一步省激活显存,见 [[073 序列并行与上下文并行|序列并行]]。
- **Q:TP 切了显存,LayerNorm 的激活切了吗?** A:没有。TP 切权重和注意力/FFN 的中间激活,但 LN/Dropout 的激活($b\cdot s\cdot d$)每卡仍存完整——这正是序列并行(SP)要补的,TP+SP 才让激活真正降到 $1/t$。
- **Q:大词表的 embedding/lm_head 怎么并行?** A:按词表维切(vocab parallel)。embedding 各卡查自己词表段、末尾 all-reduce;输出用 vocab-parallel cross-entropy,各卡算局部 logits 与归一化分母,小 all-reduce 拼全局,避免把完整 $b\cdot s\cdot V$ logits 物化到单卡。
- **Q:MQA 模型只有 1 个 KV 头,TP=8 怎么切?** A:Q 投影按头切到 8 卡,KV 投影在各卡复制(KV 头不够分)。GQA 则让 TP 度 ≤ KV 组数,或组内复制。
- **Q:定量说说 TP 为什么非机内不可?** A:每层 4 次 all-reduce 同步完整激活 $bsd$,L 层一步聚合搬运约 $16Lbsd$ 字节(80 层模型可达 TB 级/步),只有 NVLink(数百 GB/s)扛得住,跨机慢网会崩。
- **陷阱**:① 两层都列切会多通信;② TP 度数要整除头数/隐藏维/FFN 中间维;③ TP 跨机性能暴跌;④ TP 省单层显存但每步通信开销大,小模型不划算;⑤ MQA/GQA 下 KV 头不够分要复制;⑥ 忘了 vocab-parallel CE,大词表 logits 撑爆单卡。

## 关键事实

- **Megatron-LM**:Shoeybi 等(NVIDIA,2019,arXiv:1909.08053)提出层内张量并行,FFN 列切→行切、attention 按头切;**每个 transformer 层前向 2 次、反向 2 次 all-reduce**,无需改优化器即可训百亿参数模型。
- 实现为 `ColumnParallelLinear` / `RowParallelLinear`,封装通信算符 $f$(前向恒等/反向 all-reduce)与 $g$(前向 all-reduce/反向恒等)。
- **TP 限机内**:all-reduce 同步完整激活、频率=层数,需 NVLink 高带宽;TP 度数通常≤单机 GPU 数(典型 8)。
- **序列并行(Megatron-SP)**:Korthikanti 等(NVIDIA,2022,arXiv:2205.05198)在 TP 上把 LayerNorm/Dropout 激活按序列维切,用 all-gather/reduce-scatter 替部分 all-reduce,大幅降激活显存(见 [[073 序列并行与上下文并行|序列并行]])。
- **3D 并行落地**:Megatron + DeepSpeed 把 TP(机内)×PP(跨机)×DP 组合训练 530B(Megatron-Turing NLG,2022,arXiv:2201.11990)。
- 关联:多头注意力 [[005 多头注意力 Multi-Head|多头]];FFN 结构 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]];切层的 [[072 流水线并行与气泡|流水线并行]];切存储的 [[070 ZeRO 与 FSDP|ZeRO/FSDP]];底层 [[074 通信原语与计算通信重叠|通信原语]];全谱 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]]。
