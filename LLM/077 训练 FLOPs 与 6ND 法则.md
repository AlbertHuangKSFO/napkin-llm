[[077 训练 FLOPs 与 6ND 法则]]:训练一个 Transformer 的总计算量约 $C\approx 6ND$(N=参数量、D=训练 token 数);拆开就是前向 2N + 反向 4N = 每参数每 token 6 次浮点运算,而推理每 token 仅约 2N——这把"训练要多少卡、多少天"变成一道乘法题。

## ① 直觉:一个 token 流过一个参数,大约 6 次运算

参数量 N(见 [[075 参数量逐层手算(GPT 全拆)]])和数据量 D 都好理解,关键是"每个参数处理每个 token 要算几次"。答案是约 **6 次**,来自一个朴素观察:

神经网络的核心运算是矩阵乘,而矩阵乘的基本单元是**乘加(MAC, multiply-accumulate)**——一次乘法 + 一次加法 = **2 个浮点运算(FLOPs)**。每个参数(权重矩阵里的一个数)在前向时,对每个流过的 token 恰好参与**一次乘加**,即 2 FLOPs。

训练要前向 + 反向。反向比前向贵一倍(要算两个梯度:对输入的 + 对权重的),约 4 FLOPs。于是:

$$
\underbrace{2}_{\text{前向}} + \underbrace{4}_{\text{反向}} = 6\ \text{FLOPs / 参数 / token}
$$

乘上 N 个参数、D 个 token,就是训练总算力 **C ≈ 6ND**。这个公式简单到能心算,却是估算训练成本、设计训练规模、读 [[079 Scaling Law 与 Chinchilla 最优|Scaling Law]] 的基石。

## ② 例子:训练 7B 模型喂 2T token 要多少卡、多少天

**算总算力**:N=7e9,D=2e12

$$
C = 6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}\ \text{FLOPs}
$$

**换算成时间**:一张 H100 峰值约 1e15 FLOP/s(BF16,1 PFLOP/s),但实际利用率(MFU, model FLOPs utilization)通常 40–50%,取 5e14 FLOP/s:

$$
t_{\text{单卡}} = \frac{8.4\times10^{22}}{5\times10^{14}} = 1.68\times10^{8}\ \text{秒} \approx 5.3\ \text{年}
$$

**用 1000 张卡**(假设近线性扩展):$5.3\ \text{年}/1000 \approx 1.9\ \text{天}$。这就是为什么大模型训练动辄上千卡——单卡要算几年,只能靠堆卡 + 分布式(见 [[070 ZeRO 与 FSDP]]、[[073 序列并行与上下文并行]])。

**反过来估成本**:GPT-3 175B 训练约 3e23 FLOPs(N=175e9, D≈300e9, $6\times175e9\times300e9\approx3.15\times10^{23}$),与论文公布的算力一致。

![[param-训练FLOPs6ND.png]]

**推理对比**:生成 1 个 token 只前向,约 **2N FLOPs**。7B 模型生成 1 token ≈ $1.4\times10^{10}$ FLOPs;生成 1000 token ≈ $1.4\times10^{13}$ FLOPs,**比训练便宜得多**——但推理要做无数次,长期累积的总算力可能超过训练。

## ③ 原理:6ND 怎么严格拆出来,以及它忽略了什么

**前向 2N 的推导**。考虑一个权重矩阵 $W\in\mathbb{R}^{m\times n}$,作用在一个 token 向量上做 $y=Wx$:要算 $m\times n$ 次乘加 = $2mn$ FLOPs。而 $W$ 的参数量就是 $mn$。所以**每参数每 token 前向 = 2 FLOPs**,整模型前向 = $2N$(对每个 token)。

**反向 4N 的推导**。反向传播对每一层要算两类梯度:

- 对**输入的梯度** $\frac{\partial L}{\partial x}=W^\top\frac{\partial L}{\partial y}$:又一次矩阵乘,$2mn$ FLOPs。
- 对**权重的梯度** $\frac{\partial L}{\partial W}=\frac{\partial L}{\partial y}x^\top$:再一次,$2mn$ FLOPs。

两者合计 $4mn$ = $4N$ /token。前向 2N + 反向 4N = **6N** /token,乘 D 个 token:

$$
\boxed{C \approx 6\,N\,D}
$$

**它忽略了什么**(面试加分点):

1. **注意力的 $O(s^2)$ 项**。$QK^\top$ 和 $\cdot V$ 的计算量正比于序列长度平方,不含在"参数 × token"里。当序列很长($s$ 接近 $d$ 量级)时,这项不可忽略,精确公式是 $C\approx 6ND + \text{(注意力项)}$。短序列下注意力项相对小,故 6ND 是好近似。
2. 这里 N 通常指**非嵌入参数**($\approx 12Ld^2$),因为嵌入查表不做乘加。
3. 激活函数、LayerNorm、softmax 等逐元素运算量是 $O(N)$ 级,相对矩阵乘可忽略。

**与显存的联动**:6ND 决定"算多久",[[076 显存占用估算(参数、梯度、优化器、激活、KV)|16N 字节]]决定"装不装得下",两者共同框定一次训练的可行性。而"给定算力预算,N 和 D 怎么配"由 Chinchilla 回答:$D\approx 20N$ 最优,详见 [[079 Scaling Law 与 Chinchilla 最优|Scaling Law]]。

## ⑤ 把注意力项补回去(完整公式)

6ND 只数了"参数 × token"的矩阵乘,漏了注意力分数 $QK^\top$ 和 $\cdot V$——这两步不含权重参数,但计算量 $\propto s^2$。完整的单 token(实为单序列摊到 token)训练 FLOPs:
$$C\approx 6ND + 6\,L\,s\,D\cdot(\text{每层注意力的 }s\text{ 相关项}),$$
更常写成每层每序列注意力约 $2\times(2\cdot s^2\cdot d_{\text{head}}\cdot n_{\text{head}})\times3$(前向 + 反向)。**关键比值**:注意力项 / 6ND $\approx \frac{s}{6d}$(数量级)。所以:
- **短序列**($s\ll 6d$,如 s=2048、d=4096 → $s/6d\approx0.08$):注意力项 <10%,6ND 是好近似。
- **长序列**($s$ 接近或超过 $d$,如 128K 上下文):注意力项与 6ND 同量级甚至主导,**必须修正**,否则严重低估算力。这也解释了为什么长上下文训练这么贵。

## ⑥ MoE 的 6ND 与 MFU vs HFU

- **MoE 的 6ND**:用**激活参数** $N_{\text{act}}$ 而非总参数(见 [[075 参数量逐层手算(GPT 全拆)|参数量]])。Mixtral-8×7B 总参 46.7B,但训练/推理 FLOPs 按激活 ≈12.9B 算——这是 MoE"参数多但算得快"的来源。但显存仍按总参,且 all-to-all 路由有额外通信开销。
- **MFU vs HFU**(面试易混):
  - **MFU(Model FLOPs Utilization)**:有效模型 FLOPs(6ND)/ 峰值算力。**不含**梯度检查点的重算开销——衡量"对训练真正有用的算力占比"。
  - **HFU(Hardware FLOPs Utilization)**:硬件实际跑的 FLOPs(含重算)/ 峰值。HFU ≥ MFU(重算让硬件多干活但不产生新的有效进度)。
  - 报训练效率一般用 **MFU**;PaLM 报 46.2% MFU(Chowdhery 2022)。开了激活重计算,HFU 比 MFU 高约 33%(多一遍前向)。

## ⑦ 训练成本与碳:从 FLOPs 到美元

把 6ND 接到钱上(面试"训这个要多少钱"):
$$\text{GPU·小时}=\frac{6ND}{\text{单卡 FLOP/s}\times\text{MFU}\times3600},\quad \text{成本}=\text{GPU·小时}\times\text{单价}.$$
例:7B/2T token,$C=8.4\times10^{22}$;H100 BF16 峰值 1e15、MFU 0.45 → 有效 4.5e14 FLOP/s:
- GPU·秒 $=8.4\times10^{22}/4.5\times10^{14}=1.87\times10^8$ → GPU·小时 ≈ **5.2万**。
- 按 $2/GPU·小时(云价粗估)→ **约 10 万美元**;1000 卡跑 ≈2.2 天。
- GPT-3(3e23 FLOPs)同口径约 **百万美元级**,与公开估算一致。

记忆:**成本 ∝ 6ND / MFU**——省钱要么减 N·D(小模型/少数据),要么提 MFU(更好的 kernel、重叠、并行配比)。

## ⑧ 推理总算力为何可能反超训练

单次推理 2N/token 远比训练 6ND 便宜,但**推理要做亿万次**。设模型部署后总共生成 $D_{\text{infer}}$ 个 token:
$$\frac{\text{推理总算力}}{\text{训练算力}}=\frac{2N\cdot D_{\text{infer}}}{6N\cdot D_{\text{train}}}=\frac{D_{\text{infer}}}{3D_{\text{train}}}.$$
当**累计推理 token 超过 3× 训练 token** 时,推理总算力就反超训练。热门模型上线后日活亿级、每天生成万亿 token,几个月就能让推理算力远超一次训练——这是工业界宁可"过训小模型"(偏离 [[079 Scaling Law 与 Chinchilla 最优|Chinchilla]] 最优)来永久省推理成本的根本经济动因。

![[flops-推理反超训练.png]]

## ④ 代码:6ND 估算器(训练时间、成本、推理对比)

```python
def train_flops(N, D):
    """训练总 FLOPs ≈ 6·N·D。"""
    return 6 * N * D

def infer_flops_per_token(N):
    """推理每生成 1 token ≈ 2N(只前向,无反向)。"""
    return 2 * N

def train_days(N, D, n_gpus, gpu_flops=1e15, mfu=0.45):
    """近线性扩展下的训练天数。gpu_flops=峰值, mfu=实际利用率。"""
    total = train_flops(N, D)
    eff = n_gpus * gpu_flops * mfu          # 集群有效算力
    return total / eff / 86400

# ❌ 错:用峰值算力、忽略反向,严重低估训练时间
wrong = (2 * 7e9 * 2e12) / (1000 * 1e15) / 86400   # 只前向 + 满利用率
print(f"乐观估计(错): {wrong:.2f} 天")            # 太短,不可信

# ✅ 对:6ND + 真实 MFU
print(f"7B/2T token, 1000卡: {train_days(7e9, 2e12, 1000):.1f} 天")  # ≈4.3 天
print(f"GPT-3 训练总算力: {train_flops(175e9, 300e9):.2e} FLOPs")    # ≈3.15e23

# 训练 vs 推理单位算力之比
print(f"训练/推理 = {6/2}x")   # 训练每 token 是推理的 3 倍(6N vs 2N)
```

## 面试高频

- **Q:训练一个 Transformer 大约要多少算力?** $C\approx 6ND$,N=参数量、D=token 数。每参数每 token 6 FLOPs。
- **Q:6 是怎么来的?** 一次乘加 = 2 FLOPs;前向每参数每 token 2,反向 4(对输入梯度 2 + 对权重梯度 2),共 6。
- **Q:推理每 token 多少 FLOPs?** 约 2N(只前向),是训练单位算力的 1/3。
- **Q:6ND 忽略了什么?** 注意力的 $O(s^2)$ 项(长序列下显著)、逐元素运算;N 一般取非嵌入参数。
- **Q:已知 N、D 和集群算力,怎么估训练时间?** 时间 = 6ND /(卡数 × 单卡峰值 × MFU),MFU 实际约 40–50%,别用峰值。
- **Q:6ND 和 Chinchilla 什么关系?** 6ND 给出算力 C 与 N、D 的关系,Chinchilla 在 C 固定下求最优 N、D 配比(D≈20N),见 [[079 Scaling Law 与 Chinchilla 最优]]。
- **Q:MFU 和 HFU 区别?** MFU = 有效模型 FLOPs(6ND)/峰值,不含重算;HFU = 硬件实跑 FLOPs(含重算)/峰值,HFU≥MFU。报效率用 MFU,开激活重计算时 HFU 比 MFU 高约 33%。
- **Q:注意力项什么时候不能忽略?** 注意力项/6ND ≈ s/(6d)。短序列(s≪6d)<10% 可忽略;长上下文(s 接近 d,如 128K)同量级甚至主导,必须修正,这是长序列训练贵的原因。
- **Q:MoE 的 6ND 按总参还是激活参?** 激活参数。Mixtral-8×7B 按 ≈12.9B 算 FLOPs(不是 46.7B);但显存按总参,还有 all-to-all 路由开销。
- **Q:推理总算力会超过训练吗?** 会。推理总算力/训练 = $D_{\text{infer}}/(3D_{\text{train}})$,累计推理 token 超 3× 训练 token 就反超;热门模型几个月即达,这是工业界"过训小模型省推理"的经济动因。
- **Q:训练 7B/2T token 大概多少钱?** $C=8.4e22$,H100 MFU 0.45 → 约 5.2 万 GPU·小时,按 $2/小时约 10 万美元;1000 卡 ≈2.2 天。

## 关键事实

- 训练 $C\approx 6ND$、推理 $\approx 2N$/token 的近似出自《Scaling Laws for Neural Language Models》(Kaplan et al., 2020, arXiv:2001.08361)附录 F,N 为非嵌入参数。
- 拆解:前向 2N(每参数每 token 一次乘加 = 2 FLOPs)、反向 4N(对输入梯度 + 对权重梯度各 2N),是社区标准推导(如 Chinchilla 论文 Hoffmann et al., 2022, arXiv:2203.15556 附录)。
- 完整公式含注意力项:$C\approx 6ND + 6\,L\,s\,D\,\cdot(\text{常数})$ 的二次项;短序列下 6ND 主导,长序列需修正(EleutherAI Transformer Math 101)。
- MFU(model FLOPs utilization)实际常在 30–55%,PaLM 报告 46.2%(Chowdhery et al., 2022, arXiv:2204.02311);估算训练时间须乘 MFU 而非用峰值。
- GPT-3 175B 训练约 3.14e23 FLOPs,与 6ND 估算(6×175e9×300e9)吻合(Brown et al., 2020, arXiv:2005.14165)。
- MFU(有效模型 FLOPs/峰值,不含重算)vs HFU(硬件实跑/峰值,含重算)区分见《Reducing Activation Recomputation》(Korthikanti et al., 2022, arXiv:2205.05198);开激活重计算 HFU 比 MFU 高约 33%。
- 注意力项 / 6ND ≈ s/(6d):短序列可忽略、长序列(128K 上下文)同量级甚至主导,长上下文训练算力须含二次项修正(EleutherAI Transformer Math 101)。
- MoE 训练/推理 FLOPs 按激活参数算(Mixtral 按 ≈12.9B 而非 46.7B);显存按总参,路由有 all-to-all 开销(Jiang et al., 2024, arXiv:2401.04088)。
- 推理总算力 / 训练 = $D_{\text{infer}}/(3D_{\text{train}})$:累计推理 token 超 3× 训练即反超,驱动工业界"过训小模型"省长期推理成本(对照 [[079 Scaling Law 与 Chinchilla 最优|Chinchilla]] 的 LLaMA D/N≈140)。
