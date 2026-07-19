[[004 算力 vs 带宽：Roofline 与算术强度|Roofline]] 是判断一段计算到底卡在**算力**还是**带宽**的一页纸模型:横轴是**算术强度** $I$(每搬一字节数据做多少次浮点),纵轴是可达性能;落在脊点左边就是 memory-bound、右边就是 compute-bound。它是 [[001 LLM 推理的系统视角：从一次请求到一张卡|prefill/decode 受限分析]]的理论支柱,与 [[LLM/078 推理算力、吞吐与延迟、Roofline|推理算力与 Roofline]] 同源。

类比:一辆货车(GPU)既有**马力**(算力 = Tensor Core 峰值 FLOPS),又有**进货通道宽度**(带宽 = [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|HBM 带宽]])。如果你的活"每进一箱货只加工一下"(强度低),再大马力也闲着——卡在进货通道(memory-bound);如果"每箱货反复深加工"(强度高),通道再宽也得等马力(compute-bound)。**算术强度**就是"每箱货加工几下"。

**生活类比**:想象用一台超猛水泵抽水池。泵的马力强得吓人,水管却有限。decode 单请求就是"每来 1 桶水只搅一下就放走"(算术强度约为 1),泵根本没活干——真正干成的活被水管锁死。这就是 memory-bound:要改善它,要么**换更粗的水管**(加带宽),要么**一次抽很多桶共用一趟进水**(batch 抬高算术强度)。屋顶线就是把这两条上限画成两道屋檐——你的活落在哪道屋檐下,就是被谁卡住。

![[mem-004类比水管水泵.png]]

H100 数字手算脊点(理想参考值):
$$I^{*} = \frac{\text{峰值算力}}{\text{带宽}} = \frac{989\text{ TFLOPS}}{3.35\text{ TB/s}} = \frac{9.89\times10^{14}}{3.35\times10^{12}} \approx 295 \text{ FLOP/Byte}$$
- **Decode** 单步算术强度:读 1 个权重(FP16,2 字节)做 1 次乘加(2 FLOP),$I \approx 2/2 = 1 \ll 295$ → 死贴斜屋顶,**memory-bound**。要提速只能加带宽,或靠 batching 让多请求共用一次权重搬运(把 $I$ 抬高)。
- **理想化大 batch**:令 $I\approx B$。$B=256$ 时 $I=256<295$,仍是 memory-bound,$P=256\times3.35\approx858$ TFLOPS;仅在 $B=512$ 这类 $I>295$ 的**理想示例**中才会落到 compute-bound 平屋顶,$P=989$ TFLOPS。真实模型还会受 KV、attention、通信、有效带宽与 kernel 效率影响,不能把 295 当作生产 batch 阈值。

![[mem-004batch爬过脊点.png]]

原理:任一 kernel 的可达性能被两条屋顶夹住——
$$P = \min\big(\underbrace{P_{\text{peak}}}_{\text{平屋顶,算力}},\ \underbrace{I \times \text{BW}}_{\text{斜屋顶,带宽}}\big)$$
当 $I < I^{*}$,$I \times \text{BW} < P_{\text{peak}}$,性能被带宽锁住(memory-bound);当 $I > I^{*}$,被算力锁住(compute-bound)。脊点 $I^{*} = P_{\text{peak}}/\text{BW}$ 正是两条线的交点。注意:换更低位宽会同时改变两条屋顶——FP8 让 $P_{\text{peak}}$ 翻倍**且**每字节搬运的有效数据更多,等效右移又抬高屋顶。

![[mem-Roofline屋顶线.png]]

把 Roofline 写成判定器:

```python
def roofline(I, peak_TFLOPS, bw_TBs):
    peak = peak_TFLOPS*1e12
    bw   = bw_TBs*1e12
    I_star = peak / bw
    P = min(peak, I*bw)                 # 可达性能
    bound = "memory-bound" if I < I_star else "compute-bound"
    return P/1e12, I_star, bound

# H100 SXM 的理想参考屋顶:989 TFLOPS / 3.35 TB/s。
# 注意:989 是官方列出的 TF32 Tensor Core(带稀疏性)峰值之一；
# 真实 kernel 必须用同一数值格式、稀疏设定和实测有效带宽重算。
for I, tag in [(1, "decode 单请求"), (256, "理想 batch=256"), (512, "理想 batch=512")]:
    P, Istar, b = roofline(I, 989, 3.35)
    print(f"{tag}: I={I}, 脊点={Istar:.0f}, 可达 {P:.0f} TFLOPS → {b}")
# decode 单请求: I=1, 脊点=295, 可达 3 TFLOPS → memory-bound  ← 算力 0.3% 利用
# 理想 batch=256: I=256, 脊点=295, 可达 858 TFLOPS → memory-bound
# 理想 batch=512: I=512, 脊点=295, 可达 989 TFLOPS → compute-bound

# ❌ 误区:看到 decode 慢就想"换算力更强的卡"
# ✅ 正解:decode 在斜屋顶上,只有加带宽或加 batch(抬 I)才有用
```

可达 3 TFLOPS 对峰值 989 —— decode 时算力利用率仅 ~0.3%,数字直观说明"为什么 decode 该优化带宽不该堆算力"。


![[mem-004算术强度数轴.png]]

## 面试高频
- **Q:什么是算术强度,怎么算?** A:算术强度 $I$ = 计算量(FLOPs) ÷ 访存量(Bytes),衡量"每搬一字节做多少浮点运算"。它与脊点 $I^{*}=P_{\text{peak}}/\text{BW}$ 比较即可判定瓶颈。陷阱:把访存量只算输入、漏掉权重读取——decode 的访存大头是权重。
- **Q:怎么判断一个 kernel 是 compute-bound 还是 memory-bound?** A:算它的 $I$,和硬件脊点 $I^{*}$ 比:$I<I^{*}$ 为 memory-bound(加带宽/合并访存),$I>I^{*}$ 为 compute-bound(加算力/降精度)。陷阱:脱离具体硬件谈 bound——同一 kernel 在带宽更高的卡上脊点更低,bound 性质可能变。
- **Q:为什么 $I=256$ 不能因"接近 295"就叫 compute-bound?** A:Roofline 的判据是 $I\ge I^*$ 而不是"看起来接近"。在这一理想参考下,$256\times3.35=858<989$,所以仍由带宽屋顶限制；$I=512$ 才给出越过脊点的理想例。追问时要说明 295 不是任何 BF16 服务的实测阈值。
- **Q:LLM 的 prefill 和 decode 在 Roofline 上各落在哪?优化方向?** A:decode 单请求常在斜屋顶($I\approx1$),可用连续批处理、量化、减少 KV 访存改善；较大的 prefill / batch 往往有更高 $I$,但是否 compute-bound 必须以该 kernel、dtype、形状、通信和实测带宽判定。陷阱:把硬件峰值或 batch 数直接当成生产结论。

## 关键事实
- Roofline 模型用 $P=\min(P_{\text{peak}}, I\times\text{BW})$ 刻画可达性能,脊点 $I^{*}=P_{\text{peak}}/\text{BW}$ 区分两类瓶颈。来源:Roofline: An Insightful Visual Performance Model(Williams, Waterman, Patterson, CACM 2009)。
- 截至 2026-07-18,NVIDIA H100 SXM 官方规格列出 TF32 Tensor Core 989 TFLOPS(带稀疏性脚注)与 HBM 3.35 TB/s。此处 $989/3.35\approx295$ 仅作理想 Roofline 手算参考；实际脊点必须匹配 kernel 的 dtype、稀疏设定、峰值口径和有效带宽。来源:NVIDIA H100 产品规格页(核验于 2026-07-18)。
- continuous batching / 提高 batch 通过复用权重搬运提升有效算术强度,是 decode 吞吐优化的核心。来源:Orca(Yu et al., OSDI 2022)、vLLM(Kwon et al., SOSP 2023, arXiv:2309.06180)。
