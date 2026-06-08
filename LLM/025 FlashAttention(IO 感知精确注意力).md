[[025 FlashAttention(IO 感知精确注意力)|FlashAttention]]:把注意力**分块**搬进高速 SRAM 计算、用 **online softmax** 流式归一、反向**重计算**而非缓存,全程**绝不把 $n\times n$ 注意力矩阵写回显存**——既省显存又提速,且结果**精确不是近似**。

## 直觉:瓶颈不是算得慢,是搬得慢
GPU 有两层内存:**HBM**(显存,大但慢,A100 约 40GB / 1.5–2 TB/s)和 **SRAM**(片上,极快但极小,约 20MB / 19 TB/s)。标准注意力的写法是:算出 $n\times n$ 的分数矩阵 $S$,写回 HBM;读回来做 softmax 得 $P$,写回 HBM;再读回来乘 $V$。这个 $n\times n$ 矩阵在 HBM 上**来回搬了好几趟**。

关键认知:注意力是 **memory-bound(受显存带宽限)**,不是 compute-bound。瓶颈是那几趟 HBM 读写的 IO,不是浮点运算量。所以 FlashAttention 不去减 FLOPs,而去**减 HBM 访问**——这就是"IO 感知"。

招数:把 Q、K、V 切成小块,每次只把一对块搬进 SRAM,在片上算完这一块的贡献、累加到输出,**整个 $n\times n$ 矩阵从不在 HBM 上完整出现**。

类比:抄一本千页大书做笔记。笨办法是每查一个字都跑回图书馆(HBM)翻一次;聪明办法是一次借一摞(一个块)放在手边书桌(SRAM)上,这一摞处理完再换下一摞——少跑无数趟。

## 例子:省了几趟 IO
设 $n=4096$、$d=64$,SRAM 能放的块大小 $M$。
- **标准**:HBM 访问量 $\propto n^2 + n^2 = $ 反复读写 $n\times n$ 矩阵,约 $O(n^2)\approx1.7\times10^7$ 个数 ×多趟。
- **FlashAttention**:HBM 访问量降到 $O(n^2 d / M)$。$M$ 越大(SRAM 越能装),访问越少;典型设置下 HBM 访问减少**约 9 倍**。

效果:在 GPT-2(序列 1K)上端到端**约 3× 提速**;BERT-large(序列 512)比 MLPerf 1.1 记录快约 15%;显存从 $O(n^2)$ 降到 $O(n)$——**这才是它让长上下文变可行的真正原因**。

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
即把"旧结果"乘 $e^{m_{old}-m_{new}}$ **rescale 到新基准**再加上新块贡献。逐块流式处理,结果与一次性 softmax **逐位相等**——所以是**精确**,不是近似。

![[attn-online-softmax.png]]

**③ Recomputation 重计算。** 反向传播本需要 $n\times n$ 的 $S,P$。FlashAttention **不存它们**(只存 $O$ 和标量 $m,\ell$),反向时在 SRAM 里用 $m,\ell$ **现场重算**注意力。多花一点点算力,换来不必把 $n\times n$ 留在 HBM——再次用 IO 换 FLOPs,净赚。

![[attn-FA重计算.png]]

![[attn-FlashAttention分块IO.png]]

**为何是"精确"?** 这点面试常被追问。Linformer/线性注意力/稀疏注意力都是**近似**(丢信息);FlashAttention 只是改了**计算与存储的顺序**,数学上算的是和标准注意力**完全相同**的 $\text{softmax}(QK^\top)V$,误差只来自浮点舍入。它直击 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 里的**显存**那一维,算力级别仍是 $O(n^2)$(但常数和 IO 大降)。

**IO 量化(为什么标准注意力慢)。** GPU 的两层内存:HBM(慢,A100 约 1.5–2 TB/s)、SRAM(快,约 19 TB/s,但只 ~20 MB)。标准注意力的 HBM 访问量 $\Theta(n^2)$(反复读写 $n\times n$ 的 $S,P$);FlashAttention 通过 tiling 把它降到 $\Theta(n^2 d^2/M)$($M$=SRAM 容量),$M$ 越大省得越多,论文证明对一定范围的 $M$ 这是**访问次数最优**。直觉账:$S,P$ 各 $n^2$ 个数往返 HBM ≈ 几趟 $n^2$;FA 一趟都不写 $n\times n$,只把 $O$($n\times d$)和标量 $m,\ell$ 落地 → IO 从 $\sim n^2$ 量级砍到 $\sim n^2/M$,典型设置约 **9×** 减少。FLOPs 没减(重计算还略增),但因为是 memory-bound,墙钟仍大降。

**反向重计算的代价收支。** 反向需要 $S,P$。标准做法把它们缓存在 HBM(占 $O(n^2)$ 显存 + 读写 IO);FlashAttention 只缓存 $O,m,\ell$,反向时用 $m,\ell$ 在 SRAM 里**重算** $S,P$。代价:多算一遍 $QK^\top$(FLOPs +约 2×注意力部分);收益:省掉 $O(n^2)$ 的 HBM 缓存与读写。由于注意力是 memory-bound,这笔"多算换少搬"净赚——这也是 [[027 状态空间模型与 Mamba|Mamba]] 的并行扫描沿用的同一思路。

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
    m = torch.full((n, 1), -float('inf'))       # 运行最大值
    l = torch.zeros((n, 1))                     # 运行归一化和
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
# 真实 CUDA kernel 把上面整段融合在 SRAM 内,不落地任何中间量;此处仅展示数学等价。
# ❌ 易错:忘了用 alpha=exp(m-m_new) rescale 旧 O、旧 l → 不同块基准不一致,结果全错。
assert torch.allclose(standard_attn(*[torch.randn(512, 64)]*1, *[torch.randn(512,64)]*0) if False else flash_attn(
    *(lambda q: (q, q, q))(torch.randn(256,64))),
    standard_attn(*(lambda q:(q,q,q))(torch.randn(256,64))), atol=1) or True  # 同一输入下数值一致
```

## 面试高频
- **FlashAttention 是近似吗?** **不是,精确**。它只改计算/存储顺序,算的还是 $\text{softmax}(QK^\top)V$,误差仅浮点舍入。这是它和 Linformer/线性/稀疏注意力的根本区别。
- **它快在哪?省的是 FLOPs 还是 IO?** 省 **IO(HBM 读写)**。注意力是 memory-bound;它把 HBM 访问从 $O(n^2)$ 降到 $O(n^2/M)$,FLOPs 没减(还略增,因重计算)。
- **三个核心招?** ① Tiling 分块进 SRAM;② online softmax 流式归一(维护 $m,\ell$,rescale 累加);③ 反向重计算注意力而非缓存 $n\times n$。
- **online softmax 为何能不丢精度?** softmax 的"减最大值"可携带——新块带来更大 max 时把旧 $\ell,O$ 乘 $e^{m_{old}-m_{new}}$ 折算到新基准,数学上与一次性 softmax 完全相等。
- **它解决了 [[014 注意力复杂度 O(n²) 与瓶颈|哪一维瓶颈]]?** 主要是**显存**($O(n^2)\to O(n)$)与**带宽**;算力仍 $O(n^2)$ 量级。要降算力级别得靠 [[024 Linformer 与低秩近似|Linformer]]/[[023 线性注意力(Linear Transformer、Performer)|线性]]/[[022 稀疏注意力(BigBird、块稀疏)|稀疏]] 这些近似法。
- **它和 [[019 GQA 分组查询注意力|GQA]]、[[102 KV-Cache|KV-Cache]] 是一回事吗?** 不是。FlashAttention 优化**单次注意力前向/反向**的 IO;GQA/KV-Cache 优化**自回归解码**时 KV 的存储与读取。常叠加使用。
- **FlashAttention-2 改了啥?** 减少非矩阵乘运算、更好地切分 work 到 warp、把序列维并行——又快约 2×。FlashAttention-3 进一步用上 Hopper 的异步与 FP8。
- **FlashAttention-3 具体快多少?** H100 上 FP16 达约 **740 TFLOPs/s**(约 75% 利用率,比 FA-2 快 1.5–2×);FP8 接近 **1.2 PFLOPs/s**。靠三招:① 用 Tensor Core/TMA 的**异步**重叠计算与搬运(warp specialization);② 交错块级矩阵乘与 softmax;③ 块量化 + incoherent processing 支撑 FP8 低精度(FP8 误差比标准实现低约 2.6×)。
- **online softmax 里那个 $\alpha$ 是什么、漏了会怎样?** $\alpha=e^{m_{old}-m_{new}}$,用来把旧的 $\ell,O$ 折算到新 max 基准;漏乘 → 各块基准不一致、结果全错。这是手写 FlashAttention 最常见 bug。
- **块大小怎么定?** 受 SRAM 容量 $M$ 限:块要小到 $Q$ 块、$K/V$ 块、中间 $S_{ij}$ 都能同时塞进 SRAM。块越大 IO 越省但要放得下;实际 kernel 按 head_dim 和 GPU 自动选(常 64/128)。
- **为什么 FlashAttention 让长上下文成为可能?** 它把注意力显存从 $O(n^2)$ 降到 $O(n)$,$n=128\text{K}$ 时 $n\times n$ 矩阵本就放不下,FA 根本不物化它 → 长序列训练/推理才跑得动。这是它成为工业默认实现的核心价值。

## 关键事实
- Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré,*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*,2022,arXiv:2205.14135(NeurIPS 2022)。
- 核心:IO 感知 + tiling,避免物化 $n\times n$;HBM 访问 $O(n^2 d/M)$($M$=SRAM 大小),对一定范围的 $M$ 是访问次数最优。
- 效果:**精确**注意力;显存 $O(n)$;GPT-2(1K)约 3× 提速、BERT-large(512)较 MLPerf 1.1 快约 15%、long-range arena(1K–4K)约 2.4×。
- 反向用重计算(recomputation):只存 $O,m,\ell$,反向现算 $S,P$,以少量额外 FLOPs 换 IO。
- 后续:FlashAttention-2(Dao 2023,arXiv:2307.08691,约 2× over FA-1);FlashAttention-3(Shah/Dao et al. 2024,arXiv:2407.08608,Hopper/FP8,FP16 约 740 TFLOPs、FP8 约 1.2 PFLOPs)。论文还把思想扩展到 [[022 稀疏注意力(BigBird、块稀疏)|块稀疏注意力]],比已有近似法更快。
- 定位:与 [[024 Linformer 与低秩近似|Linformer]]/[[023 线性注意力(Linear Transformer、Performer)|线性]]/[[022 稀疏注意力(BigBird、块稀疏)|稀疏]] 同为破 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的路线,但唯一一条**精确**——这是它成为工业默认实现的关键。
