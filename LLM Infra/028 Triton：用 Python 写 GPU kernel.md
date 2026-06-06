[[028 Triton：用 Python 写 GPU kernel]] 是 OpenAI 开源的 GPU 内核编程语言:用 **Python 写"块级(block-level)"内核**,把"每个线程干什么、SMEM 怎么排、怎么合并访存"这些苦活交给编译器,产出能与手写 CUDA 竞争的 PTX。它是 [[027 量化内核：W4A16、W8A8 GEMM|量化内核]]、融合算子、[[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FlashAttention]] 类工作的主力工具——vLLM 的多数 attention 后端、PyTorch 2.x 的 `torch.compile` 默认都降级到 Triton。

## 直觉类比
写 CUDA C++ 像**指挥一支千人军队**:你得告诉每个士兵(线程)站哪、拿什么、什么时候动,谁踩谁的脚(bank 冲突)、谁掉队(非合并访存)全靠你盯。写 Triton 像**给一个排长下达"块"级命令**:"把这 1024 个元素加起来"——排长(编译器)自己去安排手下士兵怎么分工、怎么取数。你用接近 NumPy 的思维写"对一整块数据做什么",线程级细节编译器包办。

## 小数字例子
向量加 `z = x + y`,长度 98304。设块大小 `BLOCK=1024`:
- **网格(grid)**:$\lceil 98304 / 1024 \rceil = 96$ 个程序实例(program),每个 `program_id` 处理一块。
- 第 `pid` 个程序处理下标 `pid*1024 : pid*1024+1024`,一次 `tl.load` 把这 1024 个元素**整块**搬进来,向量化加,再 `tl.store`。
- 边界:最后一块若不满 1024,用 `mask = offsets < N` 防越界。
你写的是"块对块",96 个块怎么映射到 SM、每块内部多少线程,Triton 编译器自动定;`@triton.autotune` 还能在多组 `BLOCK`/`num_warps` 里实测选最快的。

## 原理:块级抽象与编译流程
CUDA 是 **SIMT**(线程级):你显式写 `threadIdx`,每线程处理一个标量,SMEM、同步、coalescing 全手动。Triton 抽象到**块级**:程序处理的是张量块,核心三件套——

1. **`tl.program_id(axis)`**:当前块在网格中的编号,等价 CUDA 的 `blockIdx`,但**没有 `threadIdx`**——块内并行由编译器接管。
2. **块指针 / 偏移**:`offs = pid*BLOCK + tl.arange(0, BLOCK)`,`ptr + offs` 得到一块地址,`tl.load(ptr+offs, mask=...)` 整块取数。
3. **块级算子**:`tl.dot`(矩阵乘,自动用 Tensor Core)、`tl.sum`、`tl.max` 等,编译器负责向量化、寄存器分配、SMEM 排布与 warp 调度。

编译流程:`@triton.jit` 触发即时编译,Python → **Triton IR**(块级)→ 优化/调度 → **LLVM IR** → **PTX/cubin**,结果按 (shape, dtype, 常量) 缓存,后续调用直接复用。`@triton.autotune` 在候选配置里跑基准择优,实现**跨 GPU 的性能可移植性**(同一份 Triton 代码在不同架构重新调参即可)。

![[cuda-Triton编译流程.svg]]

![[cuda-028Triton对比CUDA抽象.svg]]

## 真实 Triton kernel:向量加 + softmax
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 当前块编号
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 这一块负责的下标
    mask = offs < N                              # 越界掩码
    x = tl.load(x_ptr + offs, mask=mask)         # 整块加载
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)     # 整块写回

# 启动:grid = 块数;块内线程数由编译器决定
def add(x, y):
    z = torch.empty_like(x); N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    add_kernel[grid](x, y, z, N, BLOCK=1024)
    return z

@triton.jit
def softmax_row_kernel(in_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                        # 一个块算一整行
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)                     # 数值稳定:减行最大
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)               # 块级归约,无需手写同步
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)
```

```text
❌ 为一个融合算子(如 bias+GELU+dropout)手写 CUDA:几百行、调 bank 冲突、跨架构重写
✅ Triton 几十行块级代码,融合成单 kernel(省去中间张量的 HBM 往返),torch.compile 自动生成
```

## 面试高频
- **Triton 和 CUDA 的核心区别?** Triton 是块级(写"对一块数据做什么"),CUDA 是线程级(写"每个线程做什么")。Triton 把线程分配、SMEM、coalescing、向量化交给编译器,开发快、可移植;CUDA 控制力满但慢且易错。
- **Triton 里有 threadIdx 吗?** 没有。只有 `program_id`(块编号),块内并行编译器接管。
- **为什么 vLLM/PyTorch 大量用 Triton?** `torch.compile` 默认降级到 Triton 生成融合算子;vLLM 的 attention、RMSNorm、RoPE 等多用 Triton 写,跨 GPU 可移植且性能接近手写。
- **autotune 做什么?** 在多组 `BLOCK_SIZE`/`num_warps`/`num_stages` 配置里实测选最快,实现性能可移植性。
- **Triton 一定比 CUDA 快吗?** 不一定。多数算子能逼近手写,但极致内核(FA3、CUTLASS 的 WGMMA/TMA 编排)仍常用 CUDA/C++ 榨取最后几个百分点。
- **mask 的作用?** 处理边界块(N 不整除 BLOCK),`tl.load(..., mask=offs<N)` 防越界读写。

## 关键事实
- **Triton** 由 **OpenAI** 开发,开源;`@triton.jit` 即时编译,`@triton.autotune` 自动调参。
- 编译链:Python → Triton IR → LLVM → **PTX/cubin**,按 shape/dtype 缓存。
- 是 **PyTorch 2.x `torch.compile` 的默认内核后端**;**vLLM** 的多数 attention/norm 算子用 Triton 实现(2024–2025)。
- 编程模型核心:`tl.program_id`(块编号,无 threadIdx)、块指针 + `mask`、`tl.load/dot/store` 块级算子。
- 用途:实现 [[027 量化内核：W4A16、W8A8 GEMM|量化 GEMM]]、融合算子、attention;极致内核仍可能用 [[002 GPU 架构：SM、CUDA Core 与 Tensor Core|CUDA/CUTLASS]]。配 [[029 何时写自定义 kernel：Nsight 性能分析|Nsight]] 验证收益。
