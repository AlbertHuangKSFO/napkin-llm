[[025 FlashAttention(IO 感知精确注意力)|FlashAttention]]:把注意力**分块**搬进高速 SRAM 计算、用 **online softmax** 流式归一、反向**重计算**而非缓存,全程**绝不把 $n\times n$ 注意力矩阵写回显存**——既省显存又提速,且结果**精确不是近似**。

## 直觉:先分清算量与数据搬运
GPU 有 HBM(容量大、带宽较低)和片上 SRAM(容量小、带宽较高)两级存储。标准实现若先算 $n\times n$ 分数 $S$、再 softmax 得 $P$，会把这些二次中间量写回 HBM 再读出。FlashAttention 关注的正是这些层级间读写；具体硬件的容量/带宽与哪一项成为瓶颈应查询目标 GPU 规格并 profile，不把某一代 A100 的数字当通用事实。

关键认知:FlashAttention **不减少注意力的二次 FLOPs 主项**，而是减少 HBM 访问。当目标 shape/kernel 受 HBM 往返限制时，这会带来墙钟收益；在计算受限或不同融合条件下，收益大小会不同。这就是“IO 感知”。

招数:把 Q、K、V 切成小块,每次只把一对块搬进 SRAM,在片上算完这一块的贡献、累加到输出,**整个 $n\times n$ 矩阵从不在 HBM 上完整出现**。

类比:抄一本千页大书做笔记。笨办法是每查一个字都跑回图书馆(HBM)翻一次;聪明办法是一次借一摞(一个块)放在手边书桌(SRAM)上,这一摞处理完再换下一摞——少跑无数趟。

## 例子:省了几趟 IO
设 $n=4096,d=64$。标准实现会处理 $n^2=16{,}777{,}216$ 个 score/probability 位置；这说明为什么物化二次矩阵昂贵。令 $M$ 表示论文 IO 模型中按标量计的 SRAM 容量，在其适用的 $d\le M\le nd$ 范围内，FlashAttention 的 HBM I/O 上界为 $O(n^2d^2/M)$，而朴素物化会有 $\Theta(n^2)$ 级二次中间量读写。两个式子的变量定义不同，**不能**把它们混写成 $O(n^2d/M)$ 或“$O(n^2/M)$”。

FlashAttention 论文(2022)在 GPT-2(序列 1K)报告约 3×、在 BERT-large(512)相对 MLPerf 1.1 记录报告 15% 的端到端加速；这是模型、长度、硬件与实现绑定的实验卡，不是所有 deployment 的固定速度承诺。

**online softmax 两块手算(看清"rescale 累加"为何精确)。** 设某 query 对 6 个 key 的分数,分两块各 3 个流入。目标:验证流式结果 = 一次性 softmax。
- 一次性:分数 $[1,3,2,\,0,5,4]$,$\max=5$,$\text{softmax}$ 后 $\propto[e^{-4},e^{-2},e^{-3},e^{-5},e^{0},e^{-1}]$。
- 流式块 1 $[1,3,2]$:局部 $m_1=3$,$\ell_1=e^{-2}+e^{0}+e^{-1}=0.135+1+0.368=1.503$,累加器 $O_1=e^{-2}v_1+e^{0}v_2+e^{-1}v_3$(以 $m_1=3$ 为基准)。
- 流式块 2 $[0,5,4]$:局部 $m_2=5$。新全局 $m=\max(3,5)=5$。
  - **rescale 旧的**:旧基准是 3、新基准是 5,旧量都要乘 $\alpha=e^{m_{old}-m_{new}}=e^{3-5}=e^{-2}=0.135$。于是旧 $\ell$ 折算成 $0.135\times1.503=0.203$,旧 $O$ 同乘 $0.135$。
  - **加新块**(以 5 为基准):$\ell_{new\_block}=e^{-5}+e^{0}+e^{-1}=0.0067+1+0.368=1.375$;合并 $\ell=0.203+1.375=1.578$。
  - 最终除以 $\ell$:与一次性 $\sum e^{x_i-5}=e^{-4}+e^{-2}+e^{-3}+e^{-5}+e^0+e^{-1}=0.018+0.135+0.050+0.0067+1+0.368=1.578$ **完全相等**。

关键就是那个 $\alpha=e^{m_{old}-m_{new}}$:每当新块带来更大 max,把旧的 $\ell,O$ 一起按 $\alpha$ 折算到新基准——数学恒等,所以**逐块流式 = 一次性**,这是 FlashAttention "精确" 的来源。

## 原理:三招拆解
**① Tiling 分块。** 把 $Q$ 切成 $T_r$ 个行块 $Q_i$,$K,V$ 切成 $T_c$ 个块 $K_j,V_j$。外层遍历 $K/V$ 块,内层遍历 $Q$ 块;每对 $(Q_i,K_j)$ 在 SRAM 内算局部分数 $S_{ij}=Q_iK_j^\top$(块×块,小到能放下)。$n\times n$ 永不落地 HBM。

**② Online softmax(增量归一)。** softmax 要"减最大值再归一",难点是似乎得见到整行才能定最大值。在线版维护三个量:运行最大值 $m$、归一化和 $\ell$、输出累加器 $O$。新块 $j$ 到来:
$$m^{new}=\max(m,\ \tilde m_j),\quad \ell^{new}=e^{m-m^{new}}\ell+e^{\tilde m_j-m^{new}}\tilde\ell_j$$
$$O^{new}=e^{m-m^{new}}\,O+e^{\tilde m_j-m^{new}}\,\tilde P_j V_j$$
即把"旧结果"乘 $e^{m_{old}-m_{new}}$ **rescale 到新基准**再加上新块贡献。逐块流式处理在实数算术上与一次性 softmax 等价；浮点实现仍会有舍入次序差异，所以“精确”指不引入算法近似，而非逐 bit 相同。

![[attn-online-softmax.png]]

**③ Recomputation 重计算。** 反向传播本需要 $n\times n$ 的 $S,P$。FlashAttention **不存它们**(只存 $O$ 和标量 $m,\ell$),反向时在 SRAM 里用 $m,\ell$ **现场重算**注意力。多花一点点算力,换来不必把 $n\times n$ 留在 HBM——再次用 IO 换 FLOPs,净赚。

![[attn-FA重计算.png]]

![[attn-FlashAttention分块IO.png]]

**为何是“精确”?** Linformer/线性/稀疏注意力会改变所计算的注意力；FlashAttention 只改变块化、归一与存储顺序，在实数算术中仍计算 $\text{softmax}(QK^\top)V$。浮点舍入可能使不同 kernel 的低位不同。它直击 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的训练激活/带宽维度，时间级别仍是 $O(n^2)$。

**IO 量化(变量要一致)。** 在 Dao 等(2022)的两级内存模型中，$M$ 是 fast memory 可容纳的标量数；在给定条件下 FlashAttention 的 HBM I/O 为 $O(n^2d^2/M)$，并在一段 $M$ 范围内满足论文的最优性结论。标准实现物化 $S,P$ 会导致二次中间量 HBM 往返。不要将 $M$ 误称为“block size”，也不要从渐进式直接承诺固定“9×”；应在目标 GPU、dtype、head dimension、序列长度与 kernel 版本上测 HBM 读写和端到端时间。

**反向重计算的代价收支。** 反向需要相关的注意力量。标准做法可缓存二次中间量；FlashAttention 保存 $O,m,\ell$ 等线性量并在反向重算块。代价是额外计算，收益是避免二次 HBM 缓存与读写。哪个占优要以实际 profile 判断，不从“recompute”一词直接推导固定倍数。

## 代码:核心是 online softmax 累加
```python
import torch, torch.nn.functional as F

# ❌ 标准:物化 n×n 的 S、P,反复读写 HBM,显存 O(n²)
def standard_attn(Q, K, V):                     # (n, d)
    S = Q @ K.T / Q.size(-1) ** 0.5             # (n, n) ← 整张落地显存
    P = F.softmax(S, dim=-1)                     # (n, n) ← 又一张
    return P @ V

# ✅ FlashAttention 思想(教学版,纯 PyTorch 演示分块 + online softmax)
def flash_attn(Q, K, V, block=128):             # (n, d)
    n, d = Q.shape
    scale = d ** 0.5
    O = torch.zeros_like(Q)
    m = torch.full((n, 1), -float('inf'), device=Q.device, dtype=Q.dtype)
    l = torch.zeros((n, 1), device=Q.device, dtype=Q.dtype)
    for j in range(0, n, block):                # 遍历 K/V 块,n×n 永不成型
        Kj, Vj = K[j:j+block], V[j:j+block]
        Sij = Q @ Kj.T / scale                  # (n, block) ← 只这么大,进 SRAM
        m_j = Sij.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, m_j)
        P = torch.exp(Sij - m_new)              # (n, block)
        alpha = torch.exp(m - m_new)            # rescale 旧结果的系数
        l = alpha * l + P.sum(-1, keepdim=True) # 更新归一化和
        O = alpha * O + P @ Vj                  # 累加新块贡献(关键!)
        m = m_new
    return O / l                                # 最后统一除归一化和
# 真实 CUDA kernel 会把这段融合进 SRAM；此处只验证数学语义。

def broken_without_rescale(Q, K, V, block=3):
    """❌ 对照：忘记 alpha 后，不同块的 softmax 基准不一致。"""
    n, d = Q.shape
    O = torch.zeros_like(Q)
    m = torch.full((n, 1), -float("inf"), device=Q.device, dtype=Q.dtype)
    l = torch.zeros((n, 1), device=Q.device, dtype=Q.dtype)
    for j in range(0, n, block):
        Sij = Q @ K[j:j + block].T / d ** 0.5
        m_new = torch.maximum(m, Sij.max(dim=-1, keepdim=True).values)
        P = torch.exp(Sij - m_new)
        l, O, m = l + P.sum(-1, keepdim=True), O + P @ V[j:j + block], m_new  # 漏 alpha
    return O / l

# 同一组 Q/K/V 才能检验等价；刻意让第二块有更大的最大值以暴露漏 rescale 的 bug。
q = torch.ones(6, 1)
k = torch.tensor([[1.], [3.], [2.], [0.], [5.], [4.]])
v = torch.arange(6., dtype=torch.float32).unsqueeze(-1)
ref = standard_attn(q, k, v)
assert torch.allclose(flash_attn(q, k, v, block=3), ref, atol=1e-6, rtol=1e-6)
assert not torch.allclose(broken_without_rescale(q, k, v), ref, atol=1e-6, rtol=1e-6)
```

## 面试高频
- **FlashAttention 是近似吗?** **不是,精确**。它只改计算/存储顺序,算的还是 $\text{softmax}(QK^\top)V$,误差仅浮点舍入。这是它和 Linformer/线性/稀疏注意力的根本区别。
- **它快在哪?省的是 FLOPs 还是 IO?** 主要减少 **IO(HBM 读写)**，不降低二次 FLOPs 主项，反向还会重算。论文 IO 模型的量为 $O(n^2d^2/M)$（$M$ 是 SRAM 标量容量），实际速度收益须看目标 profile。
- **三个核心招?** ① Tiling 分块进 SRAM;② online softmax 流式归一(维护 $m,\ell$,rescale 累加);③ 反向重计算注意力而非缓存 $n\times n$。
- **online softmax 为何能不丢算法精度?** 新块带来更大 max 时把旧 $\ell,O$ 乘 $e^{m_{old}-m_{new}}$ 折算到新基准；实数算术与一次性 softmax 等价，浮点核之间仍可有舍入差异。
- **它解决了 [[014 注意力复杂度 O(n²) 与瓶颈|哪一维瓶颈]]?** 主要是**显存**($O(n^2)\to O(n)$)与**带宽**;算力仍 $O(n^2)$ 量级。要降算力级别得靠 [[024 Linformer 与低秩近似|Linformer]]/[[023 线性注意力(Linear Transformer、Performer)|线性]]/[[022 稀疏注意力(BigBird、块稀疏)|稀疏]] 这些近似法。
- **它和 [[019 GQA 分组查询注意力|GQA]]、[[102 KV-Cache|KV-Cache]] 是一回事吗?** 不是。FlashAttention 优化**单次注意力前向/反向**的 IO;GQA/KV-Cache 优化**自回归解码**时 KV 的存储与读取。常叠加使用。
- **FlashAttention-2/3 的结论如何使用?** 将其各自论文给出的 GPU、dtype、head dimension、序列长度、kernel 版本与指标视为实验卡。不要把某个 H100 TFLOPs/利用率或“比上一代快几倍”移植到不同硬件、精度或服务 shape；需要时查原论文和目标库 release。
- **online softmax 里那个 $\alpha$ 是什么、漏了会怎样?** $\alpha=e^{m_{old}-m_{new}}$,用来把旧的 $\ell,O$ 折算到新 max 基准;漏乘 → 各块基准不一致、结果全错。这是手写 FlashAttention 最常见 bug。
- **块大小怎么定?** 受 SRAM 容量、head dimension、寄存器压力、occupancy、mask 与 GPU 架构共同限制。应让实际 kernel/autotuner 选择并 benchmark；不要把“常 64/128”当作所有后端的固定配置。
- **为什么 FlashAttention 有利于长上下文?** 它避免物化 $n\times n$ 注意力中间量，训练激活由二次降为线性量级；但长上下文仍受二次算量、KV cache、模型激活与硬件限制，不能仅凭 FA 宣称任意 128K workload 都可运行。

## 关键事实
- Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré,*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*,2022,arXiv:2205.14135(NeurIPS 2022)。
- 核心:IO 感知 + tiling,避免物化 $n\times n$；在论文两级内存模型的条件下，HBM I/O 为 $O(n^2 d^2/M)$($M$=SRAM 可容纳的标量数)，并在论文注明的 $M$ 范围内满足最优性结论。
- 效果卡:**无算法近似**；论文报告 GPT-2(1K)约 3×、BERT-large(512)较 MLPerf 1.1 快约 15%、LRA(1K–4K)约 2.4×。这些数字的模型、长度、硬件和实现均以 Dao 等(2022)为准，不能泛化成承诺。
- 反向用重计算(recomputation):只存 $O,m,\ell$,反向现算 $S,P$,以少量额外 FLOPs 换 IO。
- 后续:FlashAttention-2(Dao 等,2023,arXiv:2307.08691)与 FlashAttention-3(Shah 等,2024,arXiv:2407.08608)分别报告其版本与硬件设置下的改进；使用时按原论文/目标库 release 核对。原论文也将思想扩展到 [[022 稀疏注意力(BigBird、块稀疏)|块稀疏注意力]]。
- 定位:与 [[024 Linformer 与低秩近似|Linformer]]/[[023 线性注意力(Linear Transformer、Performer)|线性]]/[[022 稀疏注意力(BigBird、块稀疏)|稀疏]] 同为破 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的路线,但唯一一条**精确**——这是它成为工业默认实现的关键。
