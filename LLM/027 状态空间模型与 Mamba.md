[[027 状态空间模型与 Mamba|状态空间模型与 Mamba]]:把整段历史压进一个**固定大小的隐状态** $h$、沿时间**扫描**更新——训练像卷积可并行、推理像 [[54 RNN 原理与 BPTT|RNN]] 常数显存,$O(n)$ 线性;Mamba 让 SSM 参数**随输入变化(选择性)**,补回表达力,成为注意力的替代路线。

## 直觉:用"一个状态"代替"看全部历史"
[[002 自注意力 Self-Attention|自注意力]] 每生成一个 token 都要回看**全部历史**(KV-Cache 随长度线性膨胀,算力 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]])。另一条思路是 [[54 RNN 原理与 BPTT|RNN]]:把历史**压进一个隐状态**,每步只看上一时刻的状态——$O(n)$、推理常数显存,但 RNN 训练要串行、且易梯度消失。

**状态空间模型(SSM)** 是"加强版线性 RNN":隐状态用一个线性动力系统 $h_t=\bar A h_{t-1}+\bar B x_t$ 演化,$y_t=C h_t$。因为是线性递推,它有两副面孔——**能写成卷积**(训练时整段并行)、**也能写成递推**(推理时一步一步、常数显存)。早期的 S4 用精心设计的 $A$(HiPPO 初始化)能记很长的历史。

S4 的死穴:$A,B,C$ 是**固定**的、对所有 token 一视同仁,无法"按内容决定记什么"——做不了内容相关的推理(比如"只记住名字,忽略停顿词")。**Mamba 的突破:让 $B,C,\Delta$ 成为输入 $x_t$ 的函数(选择性 selective)**,模型于是能根据当前 token 动态决定**记什么、忘什么**——把注意力那种内容相关的能力补回到 SSM 里。

类比:RNN/SSM 像边读边在一张固定大小的便签(状态 $h$)上记要点;注意力像把整本书摊开每次全文检索。Mamba 让"记便签"这件事变聪明——遇到关键信息浓墨重写,遇到废话直接略过。

**⚠️ 常见误区**:Mamba 不是「注意力的近似/线性化」(那是 Performer、线性注意力做的事),它是**另起炉灶**的序列建模——把历史压进固定状态、沿时间扫描,根本不算 $QK^\top$;省显存的代价是没有全量 KV,精确长程检索弱于注意力。

## 例子:为什么 $O(n)$ 且推理常数显存
设序列长 $n=100000$、状态维 $N=16$、特征维 $d=1024$。
- **Transformer 解码**:第 $t$ 步要对前 $t$ 个 token 做注意力,KV-Cache 占 $O(t)$ 显存且越来越大;生成全程算力 $\sum_t t = O(n^2)$。
- **Mamba 解码**:每步只更新固定的 $h\in\mathbb{R}^{N}$(每通道),显存 **不随 $n$ 增长**,每步 $O(N\cdot d)$ 常数,全程 $O(n)$。

$n=10$ 万这种长度下,Transformer 的 KV-Cache 会撑爆显存,Mamba 的状态始终是那么大——**长序列推理是它的主场**。代价:固定状态容量有限,精确"大海捞针"式长程检索不如注意力。

**显存量化对比($n=100\text{K}$, fp16)。** Transformer(假设 GQA, g=8, d_head=128, 32 层)每 token KV ≈ 0.13 MB → 100K token ≈ **13 GB**(还随 $n$ 线性涨);Mamba 每层只存一个 $h\in\mathbb{R}^{N}$ 每通道($N=16$),全部状态 ≈ 层数 × $d$ × $N$ × 2B ≈ 32 × 1024 × 16 × 2 ≈ **1 MB**,**且不随 $n$ 增长**。差出 4 个数量级——这就是 Mamba 在百万 token 推理上的硬优势。但反过来:固定 1 MB 状态的"记忆带宽"有限,要精确复述 100K 之前的某个具体 token,远不如把所有 KV 都留着的注意力。**这是"压缩状态 vs 全量缓存"的根本取舍**。

## 原理:连续 SSM → 离散化 → 选择性 → 并行扫描
**① 连续状态空间。** 经典线性系统:
$$h'(t)=A\,h(t)+B\,x(t),\qquad y(t)=C\,h(t)$$
$h$ 是隐状态,$A\in\mathbb{R}^{N\times N}$ 控制状态如何随时间演化。

**② 离散化。** 序列是离散的,用步长 $\Delta$ 把连续系统离散成递推:
$$\bar A=\exp(\Delta A),\quad \bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\,\Delta B$$
$$h_t=\bar A\,h_{t-1}+\bar B\,x_t,\qquad y_t=C\,h_t$$
这就是个线性 RNN。因为线性,可展开成**卷积**:$y=\bar K * x$,$\bar K=(C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots)$——训练时用 FFT 卷积整段并行算。

**③ 选择性(Mamba 的核心)。** 让 $\bar B,C,\Delta$ 变成 $x_t$ 的函数:$\Delta=\text{softplus}(\text{Linear}(x_t))$ 等。$\Delta$ 大 → 多吸收当前输入(记);$\Delta$ 小 → 状态几乎不变(略过)。**代价**:参数随时间变 → 不再是固定卷积核 → 不能用 FFT。

**离散化数值小例(看 $\Delta$ 如何调"记/忘")。** 设对角 $A=-1$(单通道),$\bar A=\exp(\Delta\cdot(-1))=e^{-\Delta}$:
- $\Delta=2$(大):$\bar A=e^{-2}=0.135$ → 旧状态几乎被冲掉、$\bar B$ 大 → **多吸收当前输入(浓墨重写)**;
- $\Delta=0.05$(小):$\bar A=e^{-0.05}=0.951$ → 旧状态保留 95% → **几乎略过当前输入(状态不变)**。
所以"选择性"就体现在 $\Delta=\text{softplus}(\text{Linear}(x_t))$:**遇到关键 token 网络输出大 $\Delta$(记),遇到废话输出小 $\Delta$(忘)**。这正是 S4(固定 $\Delta$,对所有 token 一视同仁)做不到的内容相关门控。

**④ 硬件感知并行扫描。** 选择性破坏了卷积形式,Mamba 改用**并行前缀扫描(parallel scan)**:递推 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$ 满足结合律,可用 scan 算法在 $O(\log n)$ 深度并行。且**硬件感知**——状态留在 SRAM,不把所有 $h_t$ 物化到 HBM(思路与 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 一脉相承)。

**为什么递推能并行(结合律 → 前缀扫描)。** 把每步写成 $h_t=a_t h_{t-1}+b_t$(标量示意),定义二元操作 $(a_2,b_2)\bullet(a_1,b_1)=(a_2a_1,\ a_2b_1+b_2)$——这个操作**满足结合律**。满足结合律的序列归约可用**并行前缀扫描(parallel/Blelloch scan)**在 $O(\log n)$ 深度、$O(n)$ 工作量内算完,而非朴素 RNN 的 $O(n)$ 串行深度。于是 Mamba 训练时虽不能用 FFT 卷积(参数随时变),仍能靠 scan 在 GPU 上并行——这是"选择性"与"可并行训练"得以兼得的关键。

![[attn-并行前缀扫描.png]]

![[attn-Mamba状态空间扫描.png]]

**与 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 的血缘**:线性注意力的因果形式 $S_t=S_{t-1}+\varphi(k_t)v_t^\top$ 本就是个线性 RNN 隐状态递推(见 023 的"Transformers are RNNs")。Mamba 同属"可并行训练 + 类 RNN 推理"的状态递推家族(还有 RWKV、RetNet),区别是用**选择性 + 离散化的 SSM** 把表达力做强。

## 代码:选择性 SSM 的递推骨架
```python
import torch, torch.nn.functional as F

# ❌ 朴素 RNN:训练严格串行,且 A 固定、无选择性
def vanilla_rnn(x, A, B, C, h0):                # x:(T,d)
    h = h0
    ys = []
    for t in range(x.size(0)):
        h = A @ h + B @ x[t]                     # 固定 A/B,一视同仁
        ys.append(C @ h)
    return torch.stack(ys)

# ✅ 选择性 SSM(Mamba 思想,教学递推版):Δ、B、C 由输入决定
def selective_ssm(x, A, W_dt, W_B, W_C, h):     # x:(T,d_in); A:(N,) 对角化
    ys = []
    for t in range(x.size(0)):
        dt = F.softplus(W_dt @ x[t])            # (1,) 步长 = 输入的函数 ← 选择性!
        Ab = torch.exp(dt * A)                  # (N,) 离散 Ā = exp(ΔA)
        Bx = (W_B @ x[t]) * dt                  # (N,) 输入相关的 B̄·x
        h = Ab * h + Bx                         # 状态递推(逐通道、固定大小 N)
        C = W_C @ x[t]                          # (N,) 输出投影也随输入
        ys.append((C * h).sum())
    return torch.stack(ys)
# 训练时真实实现用并行前缀扫描 + 融合 CUDA kernel(状态留 SRAM),此处只展示语义。
# ❌ 易错:把 Δ/B/C 设成固定常数 → 退化回 S4,丢掉"内容选择"能力,长程推理变差。
# ❌ 易错:推理时仍缓存所有 h_t → 浪费;只需保留当前 h(常数显存)即可逐 token 解码。
```

## 面试高频
- **Mamba 凭什么 $O(n)$?** 它是 SSM(线性 RNN):历史压进固定大小状态 $h$,每步常数更新,训练可并行扫描、推理常数显存,全程线性——不像注意力要 $O(n^2)$ 回看全历史。
- **Mamba 比早期 SSM(S4)强在哪?"选择性"是什么?** 让 $\Delta,B,C$ 随输入 $x_t$ 变化,模型能按内容动态决定**记什么、忘什么**;S4 的参数固定、做不了内容相关推理。
- **选择性带来什么代价、怎么解决?** 参数随时间变 → 不能用 FFT 卷积 → 改用**硬件感知并行扫描**(prefix scan + 状态留 SRAM)恢复并行训练。
- **Mamba 和 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]]/RWKV/RetNet 什么关系?** 同属"状态递推 = 可并行训练的现代 RNN"家族;Mamba 用选择性 SSM,表达力更强、长序列建模更好。
- **它能完全取代 Transformer 吗?** 推理快、长序列省显存,但固定状态容量有限,**精确长程检索(in-context 复制、大海捞针)弱于注意力**;故出现 Jamba 这类 Mamba+注意力**混合**架构取长补短。
- **和 [[54 RNN 原理与 BPTT|RNN]] 比?** 同是状态递推,但 SSM 用线性动力 + 离散化,可写成卷积并行训练、避开 RNN 的串行与梯度消失。
- **离散化里 $\bar A=\exp(\Delta A)$ 的 $\Delta$ 大小代表什么?** $\Delta$ 大 → $\bar A$ 小(旧状态被冲掉)、多吸收当前输入(记);$\Delta$ 小 → $\bar A\to1$(状态几乎不变)、略过输入(忘)。选择性就靠 $\Delta=\text{softplus}(\text{Linear}(x))$ 按内容调它。
- **选择性破坏卷积后,为什么还能并行训练?** 递推满足结合律($(a,b)$ 组合操作可结合),用并行前缀扫描在 $O(\log n)$ 深度并行,替代被破坏的 FFT 卷积。
- **Mamba-2 / SSD 是什么?** Mamba-2(Dao & Gu 2024)提出 SSD(state space duality),把选择性 SSM 与(掩码)注意力在数学上**统一**起来,说明二者是同一对偶的两面,且让 SSM 能复用矩阵乘高效硬件 → 更快。考点:SSM 与注意力并非对立,存在对偶联系。

![[attn-SSD对偶.png]]
- **它和 KV-Cache 的关系?** Mamba **没有 KV-Cache**——历史压进固定状态 $h$,推理只带一个常数大小的状态向量,不像注意力随 $n$ 膨胀。这既是省显存的来源,也是精确长程检索弱的原因。

## 关键事实
- Albert Gu、Tri Dao,*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*,2023,arXiv:2312.00752。核心:**选择性 SSM**(参数随输入变)+ **硬件感知并行扫描**;架构去掉注意力与 MLP,纯 SSM 块堆叠。
- 前身:S4(Gu et al. 2021,*Structured State Spaces*),用 HiPPO 初始化的结构化 $A$ 建模长程,但参数固定、无选择性。
- 复杂度:训练 $O(n)$(并行扫描)、推理 $O(1)$/token 且**常数显存**(无 [[102 KV-Cache|KV-Cache]] 膨胀);较同规模 Transformer 推理吞吐高约 5×。
- 离散化:$\bar A=\exp(\Delta A)$;选择性参数 $\Delta=\text{softplus}(\text{Linear}(x))$ 等由输入生成。
- 后续:Mamba-2(Dao & Gu 2024,arXiv:2405.21060)提出 SSD,统一 SSM 与注意力、复用矩阵乘硬件;混合架构成主流取舍——Jamba(AI21,2024,Mamba+注意力+MoE)、Zamba、Hymba 等用"少量注意力层补检索 + 大量 SSM 层省显存"。
- 定位:[[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的**替代路线**(不是注意力的近似,而是另起炉灶的序列建模);与 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 同源(状态递推),后续催生混合架构 Jamba(Mamba+注意力,2024)。
