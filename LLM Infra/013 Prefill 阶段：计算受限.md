[[013 Prefill 阶段：计算受限|Prefill]] 是[[012 自回归推理全流程：一个 token 的旅程|自回归推理]]的第一阶段:把整段 prompt 的全部 $S$ 个 token **一次性并行**送过 N 层网络,算出它们的隐状态并填满 [[LLM/102 KV-Cache|KV-Cache]],最后从最后一个位置采出第 1 个输出 token。由于所有 token 同时参与,每层是又高又宽的大 [[004 算力 vs 带宽：Roofline 与算术强度|GEMM]],算术强度高、能把算力吃满,因此 prefill 是 **compute-bound(计算受限)**——它直接决定 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT]](首 token 延迟)。与之相对的是访存受限的 [[014 Decode 阶段：访存受限|Decode]]。

## 直觉

Prefill 像**一次性把整篇文章读完做笔记**:你可以一目十行并行扫,GPU 的几千个核同时处理 prompt 的几千个 token,每个矩阵乘都是大方阵——算力利用率轻松冲到 80–90%。

工业类比:工厂一次上一整托盘零件,装配线满负荷运转,单位时间产出最大化。瓶颈在"机床转多快"(算力 FLOP/s),而不是"搬料快不快"(带宽)。所以给 prefill 加算力(更强的 GPU、更高 tensor core 利用)直接见效。

代价:prompt 越长,这块大 GEMM 越大,prefill 耗时近似随 prompt 长度线性增长,TTFT 也随之拉长——长文档/长系统提示场景下,首字延迟主要被 prefill 吃掉。

## 例子

Llama 7B(70 亿参数),prompt $S=2048$ token,H100(BF16 算力约 990 TFLOP/s):

- **FLOPs**:前向 ≈ $2\times N_\text{params}\times S = 2\times 7\times10^9\times 2048 \approx 2.9\times10^{13}$ FLOP ≈ 29 TFLOP。
- **理论 prefill 时间**:$29\text{ TFLOP} \div 990\text{ TFLOP/s} \approx 29$ ms(理想满算力)。实际含开销约 40–80 ms。
- **若 prompt 翻倍到 4096**:FLOPs 翻倍 →prefill 时间 ≈ 翻倍 →TTFT 翻倍。

对比同模型 decode 一步只算 $2\times 7\times10^9\times 1 \approx 1.4\times10^{10}$ FLOP(14 GFLOP),不到 prefill 的两千分之一,却因访存受限并不快两千倍——这正是两阶段的非对称性。

## 原理

每层主算子是 $Y = X W$,$X\in\mathbb{R}^{S\times d}$,$W\in\mathbb{R}^{d\times d}$。算术强度(每搬 1 字节做多少 FLOP):

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2\,S\,d^2}{(\,S d + d^2 + S d\,)\cdot b}\ \xrightarrow{\ S \text{ 较大}\ }\ \approx \frac{2S}{b}\ \ (\text{权重被 } S \text{ 行复用})
$$

$S$ 越大,算术强度越高,落在 Roofline 的**计算屋顶**下方斜坡之外的水平段 → compute-bound。$b$ 为字节/元素(BF16 取 2)。

机器平衡点 $\text{AI}^* = \dfrac{\text{峰值 FLOP/s}}{\text{HBM 带宽}}$(H100 BF16 约 $990\text{T}/3.35\text{T}\approx 295$)。prefill 时 $\text{AI}\gg \text{AI}^*$,故计算受限。

$$
\text{TTFT} \approx t_\text{queue} + t_\text{prefill},\qquad t_\text{prefill}\approx \frac{2\,N_\text{params}\,S}{\text{峰值 FLOP/s}\times u}
$$

$u$ 为算力利用率(prefill 下高,常 0.7–0.9)。

## 图

![[pd-prefill计算图.png]]

![[pd-013prefill随长度线性.png]]

## 代码

测量 prefill 时间(=近似 TTFT 的核心项),并对比"逐 token 喂 prompt"的错误做法:

```python
import torch, time

@torch.no_grad()
def measure_ttft(model, prompt_ids):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(prompt_ids, use_cache=True)         # ✅ 整段 prompt 一次并行 → 大 GEMM
    first = out.logits[:, -1].argmax(-1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, first  # ms

# ❌ 错误:把 prompt 一个一个喂进去当成 prefill
@torch.no_grad()
def prefill_bad(model, prompt_ids):
    past = None
    for i in range(prompt_ids.shape[1]):            # 退化成 S 次 GEMV，丢掉并行
        out = model(prompt_ids[:, i:i+1], past_key_values=past, use_cache=True)
        past = out.past_key_values
    return out.logits[:, -1].argmax(-1)             # 慢得多，且无任何收益
```

`❌` 把 prompt 拆成单 token 串行喂,放弃了 prefill 唯一的优势——并行大 GEMM,把 compute-bound 退化成 memory-bound;`✅` 整段一次 forward 才能吃满算力。

## 面试高频

- **Q:prefill 为什么是 compute-bound?** 整段 prompt 并行 →大方阵 GEMM →权重被 $S$ 行复用 →算术强度高、远超机器平衡点 →算力先饱和。
- **Q:TTFT 由什么决定?** 排队时间 + prefill 时间;prefill 时间近似随 prompt 长度线性,所以长 prompt 显著拉高 TTFT。
- **Q:怎么降 TTFT?** 加算力 / 更高 GPU 利用、**chunked prefill**(切块避免长 prompt 阻塞)、prefix caching 复用公共前缀、PD 分离让 prefill 不被 decode 拖慢。
- **Q:prefill 和 decode 能共享一个 batch 吗?** 可以(混合批),但 prefill 的大 GEMM 会拖慢同批 decode 的 ITL,故有 chunked prefill 与 PD 分离之争。

## 关键事实

- Prefill FLOPs ≈ $2\,N_\text{params}\,S$,随 prompt 长度线性;Llama 70B 在 H100 上 prefill 时算力利用率可达约 **92%**(2025 实测博客)。
- 机器平衡点(H100 BF16)≈ **295 FLOP/byte**;prefill 算术强度远高于此,故计算受限。
- 业界(截至 2025)主流用 **chunked prefill** 把长 prompt 切成小块,与 decode 交错调度,避免长 prompt 造成 TTFT/ITL 尖刺(vLLM 默认开启)。
