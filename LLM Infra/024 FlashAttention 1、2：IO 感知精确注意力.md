[[024 FlashAttention 1、2：IO 感知精确注意力]] 是一种 **IO 感知(IO-aware)的精确注意力** kernel:用分块 tiling + [[023 online softmax 与数值稳定|online softmax]] 把 $n\times n$ 的注意力矩阵**从不写回 HBM**,只在 [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|SRAM]] 里逐块算完即弃。它把 [[LLM/014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]] 从显存里彻底拿掉、显存降到 $O(n)$,且结果**精确无近似**。详见跨域笔记 [[LLM/025 FlashAttention(IO 感知精确注意力)|FlashAttention]];本篇聚焦 Infra 视角的 IO 复杂度与 FA1→FA2 工程改进。

## 直觉类比
朴素注意力像把所有人两两握手记成一张巨大的 $n\times n$ 名册,写满整个仓库(HBM)再读回来算——搬运名册的时间远超握手本身。FlashAttention 改成:每次只取一小队 Q 和一小队 K/V 到手边白板(SRAM),算完这小块的得分、当场用 online softmax 更新累计输出,白板擦掉换下一队。名册从未落地,仓库的搬进搬出省到极致。注意力本是 memory-bound,省 IO 就是省时间(见 [[021 kernel 融合：为何能省带宽|kernel 融合]])。

**生活类比**:在图书馆查资料。书库(HBM)容量大但离桌子远,来回搬一趟很慢;桌上的白板(SRAM)又快又近但很小。**朴素做法**:把整整一箱 4096×4096 的大名册从书库搬到桌上,写满、读回、又写……桌子早被占满,人坐着干等搬运——这就是反复进出 HBM,访问量 Θ(n²)、显存 O(n²),搬书时间远超真正"握手计算"的时间。**Flash 做法**:每次只从书库取一小叠 Q 和一小叠 K/V 放白板上,当场算完这块得分、用 online softmax 把结果累加进总账,然后擦掉白板换下一叠——那张大名册**从头到尾没落过地**。HBM 访问降到 Θ(n²d²/M);取 d=128、白板 M≈100KB 时,搬运量约少 6 倍,长序列墙钟快 2–4×。关键:省的是"搬书"不是"握手",计算量一点没变,结果精确无近似。

![[flash-024类比图书馆取书.png]]

## 小数字例子
序列 $n=4096$、头维 $d=128$、SRAM 约 $M=100\text{KB}$:
- 朴素:物化 $S=QK^\top$ 与 $P=\text{softmax}(S)$ 各 $4096^2\approx1.7\times10^7$ 个元素,反复进出 HBM,HBM 访问 $\Theta(n^2)$,显存 $O(n^2)$。
- Flash:Q 分块 $B_r$、K/V 分块 $B_c$(约 $\sqrt{M}$ 量级),外层遍历 K/V 块、内层在 SRAM 算小块并滚动累加 $O$。HBM 访问降到 $\Theta(n^2 d^2/M)$,显存只 $O(n)$。
- 取 $d=128,M=10^5$:IO 减少因子 $\approx M/d^2 = 10^5/16384 \approx 6\times$,长序列上墙钟加速 2–4×。

## 原理:IO 复杂度
设 SRAM 大小 $M$、头维 $d$、序列 $n$。标准注意力的 HBM 访问量是

$$\Theta(n d + n^2)$$

(那个 $n^2$ 来自读写注意力矩阵)。FlashAttention 经分块,HBM 访问量降为

$$\Theta\!\left(\frac{n^2 d^2}{M}\right)$$

当 $M \gg d^2$ 时远小于 $n^2$。配合 online softmax 的递推 $\ell^\text{new}=e^{m^\text{old}-m^\text{new}}\ell^\text{old}+\sum e^{x_k-m^\text{new}}$,输出同步重缩放累加,无需物化 $P$。**FA2 的改进**:① 把并行维度从 batch×head 扩到 **序列维**,长序列也能占满更多 [[019 CUDA 执行模型：grid、block、warp|block]],提高 occupancy;② 重排循环、减少非 matmul 的 rescale 运算(GPU 上非 matmul FLOP 比 matmul 慢得多);③ 优化 warp 间 work 划分,减少 SRAM 读写争用——综合带来约 **2×** 提速。

## 图
![[flash-分块数据流SRAM.png]]

把标准与 Flash 的 HBM IO 与显存逐项对比,IO 减少因子 ≈ M/d²:

![[flash-024IO复杂度对比.png]]

FA1 到 FA2 这 ~2× 提速,具体来自三处工程改进(算法没动):

![[flash-024FA1到FA2改进.png]]

## 代码:物化矩阵 vs 分块融合
```python
# ❌ 朴素:物化整个 n×n 注意力矩阵,反复进出 HBM,显存 O(n²)
S = Q @ K.transpose(-1, -2) / math.sqrt(d)   # [n, n] 落 HBM
P = S.softmax(dim=-1)                          # [n, n] 落 HBM
O = P @ V                                      # 又一次读回

# ✅ FlashAttention:分块 + online softmax,n×n 矩阵从不落 HBM
# (伪码)外层遍历 K/V 块,内层在 SRAM 算并滚动累加
for j in range(num_kv_blocks):
    Kj, Vj = load_to_sram(K, V, j)             # 只载小块
    Sij = Qi @ Kj.T                            # SRAM 内,小块
    m_new = max(m_i, Sij.max(-1))              # online softmax: running max
    Pij = (Sij - m_new).exp()
    l_i = exp(m_i - m_new)*l_i + Pij.sum(-1)   # running sum 重缩放
    O_i = exp(m_i - m_new)*O_i + Pij @ Vj      # 输出同步缩放累加
    m_i = m_new
O_i = O_i / l_i                                # 写回时才落 HBM,O(n) 显存
```

## 面试高频
- **FlashAttention 为什么快?** 它不省 FLOP,而省 HBM IO:把 $n^2$ 注意力矩阵留在 SRAM 不物化,注意力本是 memory-bound,故墙钟变快。
- **它是近似吗?** 不是,精确(exact),靠 online softmax 保证与标准注意力数学等价。
- **IO 复杂度多少?** 标准 $\Theta(n^2)$,Flash $\Theta(n^2 d^2/M)$;显存从 $O(n^2)$ 降到 $O(n)$。
- **FA2 相比 FA1 改了什么?** 增加序列维并行提升 occupancy、减少 rescale 等非 matmul 运算、改进 warp 间 work 划分,约 2× 提速。
- **后续?** FA3 针对 Hopper 用 TMA 异步搬运、WGMMA、FP8,见 [[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FlashAttention-3]]。

## 关键事实
- FlashAttention(FA1):Dao, Fu, Ermon, Rudra, Ré,"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",arXiv:2205.14135(2022)。
- FlashAttention-2:Tri Dao,"Faster Attention with Better Parallelism and Work Partitioning",arXiv:2307.08691(2023),约 2× over FA1。
- HBM IO 复杂度 $\Theta(n^2d^2/M)$,显存 $O(n)$,精确(无近似),依赖 online softmax(Milakov & Gimelshein 2018)。
