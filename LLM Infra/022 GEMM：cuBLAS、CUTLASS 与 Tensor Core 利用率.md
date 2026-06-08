[[022 GEMM：cuBLAS、CUTLASS 与 Tensor Core 利用率]] 讲 GPU 上通用矩阵乘(GEMM,$C=A\cdot B$)如何靠**分块 tiling** 把数据逐级搬进 [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|SRAM/寄存器]] 喂给 **Tensor Core**,以及为何实际利用率常打不满。GEMM 是 Transformer 几乎所有重活(QKV 投影、FFN)的底层,理解它直接关系到 [[深度学习基础/06 矩阵乘法的几何意义|矩阵乘法]] 在硬件上的真实成本。

## 直觉类比
两个大矩阵相乘,数据太大塞不进快内存。tiling 就像拼大墙:不可能一次端整面墙的砖,而是一格一格(tile)地搬砖到手边(SRAM)、砌完一格再搬下一格,且每块砖反复砌进多处(数据复用)。Tensor Core 则是"一次砌 16×16 一片"的超级砌墙机——但如果砖供不上(带宽不够),机器只能空转。

## 小数字例子
$C[4096\times4096]=A[4096\times4096]\times B[4096\times4096]$,典型三级 tile:
- **block tile 128×128**:每个 [[019 CUDA 执行模型：grid、block、warp|block]] 负责一块 C,把对应 A 行带、B 列带载入 SRAM。
- **warp tile 64×64**:block 内每个 warp 取一块到寄存器复用。
- **MMA tile 16×16**:warp 协作发一条 `mma.sync`,一次算 $D=A\cdot B+C$。
- 一块 128×128 的 C 需沿 K 维循环 4096/K_tile 次累加;每个元素被复用 ~128 次 → 算术强度高,才喂得饱 Tensor Core。
- 朴素未分块写法每个 C 元素都从 HBM 重读整行整列,流量爆炸、利用率个位数。

**三级 tile 各落哪层内存、占多大,逐级算一遍。** 取 K_tile=32、FP16(2 字节):
- **block tile 128×128 → SMEM**:一轮要驻留 A 子块 $128\times32$ + B 子块 $32\times128$ = $2\times(128\times32)\times2\,\text{B}=16\,\text{KB}$(双缓冲再翻倍到 32 KB),正好压在一台 SM 的 ~228 KB SMEM 里,这是「搬一格砖到手边」的那只手。
- **warp tile 64×64 → 寄存器**:一个 block 256 线程切 8 个 warp,每 warp 啃 $64\times64$ 的 C 累加器,均摊每线程 $64\times64/32=128$ 个 FP32 累加值,直接占寄存器堆——离 ALU 最近、零延迟复用。
- **MMA tile 16×16 → Tensor Core**:warp 内 32 线程协作发一条 `mma.sync`,把 $16\times16\times16$ 的小块喂进 Tensor Core 阵列一拍算完。

层层缩小不是为了好看:**SMEM 装得下一片砖、寄存器攥得住一小撮、Tensor Core 一口吞 16×16**——尺寸跟着「从慢到快、从大到小」的内存层级走,谁也别越界把上一层撑爆。

## 原理:roofline 与利用率上限
算术强度 $I=\dfrac{\text{FLOP}}{\text{访存字节}}$。GEMM 的 FLOP $=2MNK$,理想下 $I$ 随 tile 增大而升:

$$\text{峰值利用率} \le \min\!\left(1,\; \frac{I \times \text{带宽}}{\text{峰值算力}}\right)$$

- $M$ 或 $N$ 太小(如 decode 阶段 batch=1 的 GEMV)→ $I$ 低 → **memory-bound**,Tensor Core 饿肚子。
- 维度非 tile 整数倍 → **tail effect**,边角 block 算力浪费。
- 即便 compute-bound,寄存器/SRAM 压力、流水线气泡也让利用率难到 100%。

**GEMM vs GEMV 一句话**:prefill/大 batch 是矩阵×矩阵(GEMM,$I$ 高、吃满 Tensor Core);decode 的 batch=1 退化成矩阵×向量(GEMV,$I≈1$、Tensor Core 几乎空转)——同一份权重,瓶颈一个在算力、一个在带宽。

## 图
![[kern-GEMM-tiling层级.png]]

为什么实际利用率常打不满?三个杀手(瘦长矩阵、tail effect、流水线气泡)配 roofline 一图看清:

![[kern-022GEMM利用率决策.png]]

## 代码:从直接调库到自定义
```cpp
// ✅ 默认首选:cuBLAS,闭源但高度调优,直接拿走最稳
cublasGemmEx(handle, OP_N, OP_N, m, n, k,
             &alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
             &beta,  C, CUDA_R_16F, ldc,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); // 走 Tensor Core
```
```cpp
// ✅ 需融合/定制 tile/特殊数据类型时:CUTLASS(开源 C++ 模板)
//    暴露 thread/warp/block/device 各级原语,可调 tile 尺寸与 epilogue
using Gemm = cutlass::gemm::device::Gemm<half, RowMajor, half, RowMajor,
                                         float, RowMajor, float,
                                         cutlass::arch::OpClassTensorOp, // MMA
                                         cutlass::arch::Sm80>;
```
```cpp
// ❌ 朴素三重循环 kernel:无 tiling,每元素重读 HBM,利用率个位数
for (int kk = 0; kk < K; ++kk) acc += A[row*K+kk] * B[kk*N+col];
```

## 一次 Tensor Core 矩阵乘:逐数手算

公式里的 `2MNK`、tiling 这些词很抽象。我们取一个小到能用纸笔算的例子,把「Tensor Core 在干什么」从头到尾走一遍。

取两个 2×2 矩阵:

$$A=\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix},\quad B=\begin{bmatrix}5 & 6\\ 7 & 8\end{bmatrix}$$

矩阵乘的规则就一句话:**输出 D 里第 i 行第 j 列的元素 = A 的第 i 行 与 B 的第 j 列做点积**(对应位置相乘再相加)。逐个手算:

- D[0,0] = 第0行(1,2) · 第0列(5,7) = 1·5 + 2·7 = **19**
- D[0,1] = 第0行(1,2) · 第1列(6,8) = 1·6 + 2·8 = **22**
- D[1,0] = 第1行(3,4) · 第0列(5,7) = 3·5 + 4·7 = **43**
- D[1,1] = 第1行(3,4) · 第1列(6,8) = 3·6 + 4·8 = **50**

$$D = A\cdot B = \begin{bmatrix}19 & 22\\ 43 & 50\end{bmatrix}$$

CPU 会一个接一个地算这 4 个点积(串行 8 次乘加)。**Tensor Core 的不同之处:它把这 4 个点积塞进 MMA 阵列,一条指令同时做完**——这就是它快的全部秘密。真实硬件的一条指令处理的块更大(Hopper 上一条 WGMMA 是 64×256×16 的块,由一个 warpgroup=4 warp=128 线程协作发出),这里用 2×2 只为讲清「一行点一列、并行一次出」的原理。

![[kern-022逐数手算2x2矩阵乘.png]]

**再讲 tiling(瓷砖)。** 上面是 2×2,实际 LLM 里的矩阵动辄 4096×4096——这么大的数据,根本塞不进又快又小的寄存器和 SRAM。办法是「切瓷砖」:把大矩阵切成一片片小瓷砖(比如 16×16),**一次只搬一片到手边的快内存、喂给 Tensor Core 算,算完再搬下一片**,沿 K 维循环逐块累加。边搬边算、每块数据反复复用(一块 128×128 的瓷砖里每个元素被复用上百次),才能让冲压机不停地有砖可压。这「分块搬运 + 逐块累加」就是 GEMM 的核心套路,对应三级 tile:block tile→SRAM、warp tile→寄存器、MMA tile→Tensor Core。

![[kern-022大矩阵切瓷砖喂冲压机.png]]

**利用率为什么总打不满?** 三个原因,全是「冲压机有空档」:
- **瓷砖边角(tail effect):** 矩阵维度不是瓷砖尺寸的整数倍时,边上的块只填了一半,那部分算力浪费。
- **搬运赶不上算(memory-bound):** 砖供不上,冲压机只能空转等砖。decode 阶段 batch=1 退化成 GEMV(矩阵乘向量),数据复用极低,几乎必然卡在带宽上。
- **K 维太小:** 每个点积只有寥寥几次乘加,算术强度太低,喂一次算一点点,机器大部分时间在等数据而非计算。

## 面试高频
- **cuBLAS vs CUTLASS?** cuBLAS 闭源调好的库,直接用最稳;CUTLASS 开源模板,可定制 tile/数据类型、能与其他算子融合(FlashAttention 等基于它)。
- **Tensor Core 怎么用?** 通过 WMMA API(可移植但稍慢)或 PTX 的 `mma.sync` 指令族(更快更灵活);Volta 起每周期做 4×4 MMA,Hopper 引入 warpgroup 级 WGMMA(见 [[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FlashAttention-3]])。
- **利用率为何打不满?** 小矩阵/瘦长矩阵 → memory-bound;维度非整除 → tail effect;decode 的 GEMV 几乎必 memory-bound。
- **三级 tiling 各对应哪层内存?** block tile→SRAM、warp tile→寄存器、MMA tile→Tensor Core。

## 关键事实
- CUTLASS 是 NVIDIA 开源的 CUDA C++ GEMM 模板库,提供 thread/warp/block/device 各级原语(NVIDIA, 2017 起)。
- Tensor Core:Volta(2017)起,每周期 4×4 MMA;Hopper(2022)引入 WGMMA,warpgroup=4 warp=128 线程。
- decode 阶段 batch=1 的注意力/投影退化为 GEMV,算术强度低,是 LLM 推理 memory-bound 的根因。
