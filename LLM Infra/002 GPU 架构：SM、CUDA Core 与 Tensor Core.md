[[002 GPU 架构：SM、CUDA Core 与 Tensor Core|GPU 架构]] 是理解 LLM 为何快的硬件底座:GPU 由上百个 **SM(流多处理器)** 组成,每个 SM 内有通用的 **CUDA Core** 和专做矩阵乘的 **Tensor Core**,而 Transformer 几乎所有计算都是矩阵乘,所以**吃 Tensor Core**。

类比工厂:整张 GPU 是一座厂区,**SM 是车间**(H100 有 132 个),车间里 **CUDA Core 是通用工人**(什么活都能干、但慢),**Tensor Core 是专用冲压机**(只会"一拍冲一整块矩阵乘",但快一个数量级)。LLM 的活 99% 是大矩阵乘,所以冲压机决定产能,通用工人只打杂(采样、归一化)。

具体数字感受差距(H100 SXM):
- CUDA Core 通用 FP32 峰值 ~67 TFLOPS。
- Tensor Core FP16 峰值 ~989 TFLOPS(dense),约 **15×**;切到 [[005 数值格式：FP32、TF32、BF16、FP8、FP4|FP8]] 再翻倍到 ~1979 TFLOPS。
- 线程组织:**warp = 32 个线程锁步执行同一条指令**(SIMT)。一个 7B 模型的 FFN 第一层是 $[B, 4096] \times [4096, 11008]$ 的矩阵乘——这种规整大 GEMM 正是 Tensor Core 的主场,而逐元素的 GELU 才轮到 CUDA Core。

原理:Tensor Core 一条指令完成一个小块矩阵乘加 $D = A \cdot B + C$(如 $16\times16$ 块),把本来要 CUDA Core 多个周期、多条指令的乘加合并成**一拍**。设矩阵乘总计算量
$$\text{FLOPs} = 2 \cdot M \cdot N \cdot K$$
($M\times K$ 乘 $K\times N$,每个输出元素 $K$ 次乘加 = $2K$ 次浮点)。Tensor Core 把这堆乘加并行塞进专用阵列,吞吐 = 单位时间能完成多少这样的块。衡量实际用得好不好的指标是 **MFU(Model FLOPs Utilization)**:
$$\text{MFU} = \frac{\text{模型实际有效 FLOPs/s}}{\text{Tensor Core 峰值 FLOPs/s}}$$
[[001 LLM 推理的系统视角：从一次请求到一张卡|prefill]] 大 batch 能到 40–60%,[[004 算力 vs 带宽：Roofline 与算术强度|decode]] 因带宽瓶颈常 <5%(算力被晾着)。底层数学见 [[深度学习基础/06 矩阵乘法的几何意义|矩阵乘法]]。

![[hw-002MFU手算对比.png]]

![[hw-GPU架构SM层级.png]]

用 PyTorch 直接看 Tensor Core 是否被启用,以及对吞吐的影响:

```python
import torch, time

A = torch.randn(8192, 8192, device='cuda')
B = torch.randn(8192, 8192, device='cuda')

def bench(x, y, tag):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): z = x @ y
    torch.cuda.synchronize()
    dt = (time.time()-t)/50
    flops = 2 * 8192**3
    print(f"{tag}: {dt*1e3:.2f} ms, {flops/dt/1e12:.0f} TFLOPS")

# ❌ FP32:走 CUDA Core(或 TF32),吞吐低
bench(A, B, "FP32")
# ✅ FP16/BF16:走 Tensor Core,吞吐高一个数量级
bench(A.half(), B.half(), "FP16")
# 经验:同一块卡 FP16 的 TFLOPS 通常是 FP32 的 8~15×,差距即来自 Tensor Core
```

查 SM 数等架构信息:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; p=torch.cuda.get_device_properties(0); \
print(p.name, 'SM=', p.multi_processor_count)"   # H100 → SM= 132
```


![[hw-002SM内部结构.png]]

## 看懂 Tensor Core:从「单个计算器」到「整块冲压机」

如果前面的 TFLOPS 数字还是看不懂,先忘掉术语,用最朴素的画面理解这两种核心。

**CUDA Core = 一个普通计算器。** 它一次只算一个「乘加」:`a×b+c`。你给它两个数相乘再加一个数,它吐出一个结果——就这么点本事。GPU 厉害在于「人海战术」:一个 SM 里塞了上万个这样的计算器(H100 一个 SM 有 **128** 个 FP32 CUDA Core,全卡 132 个 SM 加起来上万),让它们同时各算各的。计算器很灵活,加减乘除、比较大小、走分支都能干,所以采样、归一化、GELU 这类「逐元素杂活」都归它。但用它做一个大矩阵乘,等于要一条一条地发上亿条指令,慢。

**Tensor Core = 一台矩阵冲压机。** 它不接受「单个数」,只接受「整块小矩阵」:一条指令直接吞下两个小矩阵 A、B,猛地一压,**一次就盖出整块乘加结果** `D = A·B + C`。用最小的例子感受:一台 4×4×4 的冲压机,一条指令同时完成 **64 个乘加**(输出 4×4=16 个元素,每个元素含 4 次乘加,16×4=64)。同样这 64 个乘加,交给 CUDA Core 要 64 步,冲压机只要 1 拍。

![[hw-002类比计算器vs冲压机.png]]

![[hw-002TensorCore一次吞两矩阵.png]]

**为什么 LLM 几乎全靠 Tensor Core?** 因为 Transformer 的算力 99% 是矩阵乘(QKV 投影、FFN、注意力打分),全是冲压机的主场;只有那 1% 的逐元素操作才轮到计算器。冲压机比单个计算器快几十倍,谁决定 LLM 跑多快,一目了然。

**用具体数字钉死这个反差:** H100 一个 SM 里,CUDA Core 有 **128 个**(FP32),Tensor Core 却**只有 4 个**——数量上冲压机少得可怜。可就这 4 台冲压机,贡献了全卡 FP16 ~**989 TFLOPS**(dense)峰值里的绝大部分;而上万个 CUDA Core 的 FP32 峰值合起来才 ~**67 TFLOPS**。差距约 **15×**,切到 FP8 还能再翻倍到 ~1979 TFLOPS。一句话:**算力不在「核多」,在「有没有冲压机」。** 这也解释了为什么上面代码里 FP16(走 Tensor Core)比 FP32 快一个数量级——不是 GPU 变快了,是活儿换了机器干。

## 面试高频
- **Q:CUDA Core 和 Tensor Core 有什么区别?** A:CUDA Core 是通用标量/向量单元,做逐元素、控制流、归约等任意运算;Tensor Core 是专用矩阵乘加阵列,一条指令算一个小块 $D=A\cdot B+C$,吞吐高一个数量级但只擅长 GEMM。LLM 99% FLOPs 是矩阵乘,所以主要吃 Tensor Core。陷阱:说 Tensor Core 是"更多的 CUDA Core"——它是结构不同的专用单元。
- **Q:什么是 warp?为什么重要?** A:warp 是 32 个线程的锁步执行组(SIMT),同一 warp 内所有线程执行同一指令;若分支让线程走不同路径会"分化"(warp divergence)串行执行各分支,降效。陷阱:把 warp 和 SM 混淆——SM 是硬件车间,warp 是软件线程调度的最小单位。
- **Q:什么是 MFU,为什么 decode 的 MFU 很低?** A:MFU = 有效 FLOPs/s ÷ 峰值 FLOPs/s。decode 每步只算 1 个 token 却要读全部权重,被 HBM 带宽卡住,Tensor Core 大量空闲,故 MFU 常 <5%。陷阱:把 MFU 和 GPU 利用率(`nvidia-smi` 的 util%)混为一谈——后者只表示 SM 是否在忙,不反映算力是否被有效利用。

## 关键事实
- H100 SXM5 含 132 个 SM;FP16 Tensor Core ~989 TFLOPS(dense)、FP8 ~1979 TFLOPS(dense),分别约为 FP32 CUDA Core 的 15×、30×。来源:NVIDIA H100 Datasheet / Hopper Architecture In-Depth(NVIDIA Technical Blog, 2022)。
- A100(Ampere)有 108 个 SM、6912 个 CUDA Core、432 个第 3 代 Tensor Core。来源:NVIDIA Ampere Architecture Whitepaper(2020)。
- Tensor Core 自 Volta(2017)引入,Hopper 为第 4 代并新增 FP8 支持。来源:NVIDIA Hopper Architecture Whitepaper(2022)。
- warp 固定为 32 线程(SIMT 模型),自早期 CUDA 架构沿用至今。来源:NVIDIA CUDA C Programming Guide。
