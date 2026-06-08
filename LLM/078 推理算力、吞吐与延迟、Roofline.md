[[078 推理算力、吞吐与延迟、Roofline]]:LLM 推理分两阶段——**prefill**(处理整段 prompt)算力受限、**decode**(逐 token 生成)访存受限;用 **Roofline 模型**按"算术强度"判断瓶颈,并理解**吞吐(throughput)与延迟(latency)**为何此消彼长、加大 batch 如何提吞吐却伤单请求延迟。

## ① 直觉:推理不是一种活,是两种性质相反的活

把"推理"拆成两阶段,瓶颈完全不同:

- **Prefill(预填充)**:模型一次性读完整个 prompt(可能几千 token),并行算出它们的 K、V。这是一次**大矩阵乘**,要算大量浮点——**算力受限(compute-bound)**,GPU 的计算单元被喂满。它决定 **TTFT(Time To First Token,首 token 时延)**。
- **Decode(解码)**:之后逐个 token 自回归生成,每步只算 1 个新 token,但要**把整个模型的权重从显存读一遍**(还要读 KV-Cache)。计算量极小、访存量极大——**访存受限(memory-bound)**,GPU 算力大量闲置,卡在带宽上。它决定 **TPOT(Time Per Output Token,每 token 时延)**。

为什么这区分重要?因为**优化手段相反**:prefill 慢就上更强算力的卡、FP8;decode 慢就上更高带宽、量化权重、攒大 batch。判断一个算子到底受限于哪,靠 **Roofline 模型**——一张图把"算术强度"和"可达性能"连起来,一眼看出你撞的是"算力天花板"还是"带宽斜坡"。

## ② 例子:为什么 decode 这么"浪费"算力

设 7B 模型(fp16,权重 14GB),GPU 带宽 3TB/s、算力 1000 TFLOP/s。

**decode 一步**:生成 1 个 token,要读一遍全部权重 14GB。

- 访存时间:$14\text{GB} / 3\text{TB/s} \approx 4.7\text{ ms}$
- 计算量:约 $2N = 1.4\times10^{10}$ FLOPs(见 [[077 训练 FLOPs 与 6ND 法则]]),计算时间:$1.4\times10^{10}/10^{15} = 0.014\text{ ms}$
- **算术强度** I = FLOPs/字节 = $1.4\times10^{10} / 1.4\times10^{10}\text{ B} \approx 1$ FLOP/Byte —— 极低!

访存 4.7ms ≫ 计算 0.014ms,GPU 算力利用率不到 1%。这就是 decode 访存受限的本质:**每读一字节权重只做约 1 次运算**,大部分时间在等数据从显存搬过来。

**怎么救?攒 batch**。同时给 8 条请求 decode,权重只读一遍(还是 14GB),却算了 8 个 token——计算量 ×8,访存不变,算术强度 ×8,吞吐近乎 ×8 而单步延迟几乎不变。这就是 **continuous batching(连续批处理,vLLM 等)** 的威力。

**prefill 对比**:读 2048 个 token 的 prompt,一次大 GEMM,计算量 $\approx 2N\times2048$,但权重还是读一遍 14GB → 算术强度高达约 2048,**远在脊点右侧,吃满算力**。

![[param-Roofline模型.png]]

![[param-吞吐与延迟权衡.png]]

## ③ 原理:Roofline、算术强度与吞吐延迟权衡

**Roofline 模型**把硬件性能画成两段"屋顶":

$$
\text{可达性能} = \min\big(\underbrace{\text{峰值算力 } \pi}_{\text{算力屋顶}},\ \underbrace{I \times \beta}_{\text{带宽斜坡}}\big)
$$

其中 $I$ 是**算术强度(arithmetic / operational intensity)= FLOPs / 访存字节数**,$\beta$ 是内存带宽。两段的交点叫**脊点(ridge point)**:

$$
I^\* = \frac{\pi}{\beta}\quad(\text{FLOP/Byte})
$$

- $I < I^\*$:落在斜坡上,**访存受限**,性能 = $I\times\beta$,提速靠提带宽或提高 $I$(攒 batch)。decode 在这里。
- $I > I^\*$:落在平台上,**算力受限**,性能 = $\pi$,提速靠提算力(更强 GPU、FP8)。prefill 在这里。

例:H100 算力 ≈ 1000 TFLOP/s(BF16)、带宽 ≈ 3.35 TB/s,脊点 $I^\*\approx 300$ FLOP/Byte。decode 的 $I\approx 1\ll 300$ → 死死钉在带宽斜坡;prefill 的 $I\approx$ 序列长 → 轻松越过脊点。

**吞吐 vs 延迟的权衡**,两个常被混淆的指标:

- **延迟(latency)**:单条请求多快出结果。看 TTFT(首 token)和 TPOT(每后续 token)。在线聊天要低延迟。
- **吞吐(throughput)**:单位时间总共生成多少 token(跨所有并发请求)。离线批处理要高吞吐。

二者此消彼长:加大 batch,decode 的算术强度上升、GPU 利用率上升、**总吞吐线性提高**;但每条请求要等更大的批一起算,**单请求延迟上升**。数学上,decode 受限于带宽 $\beta$,单步时间约 $\approx \frac{N\cdot b_{\text{param}}}{\beta}$(读权重),近乎与 batch 无关——所以小 batch 到中 batch,吞吐几乎免费翻倍,直到算术强度逼近脊点、转为算力受限才饱和。

**与 KV-Cache 的联动**:batch 越大、序列越长,KV-Cache 显存越大(见 [[102 KV-Cache]]),最终受显存容量约束 batch 上限。这就是 [[026 PagedAttention 与 KV 分页|PagedAttention]] 通过分页提高 KV 利用率、从而支持更大 batch、提吞吐的原因。

## ⑤ TTFT 与 TPOT 的定量公式

把两个延迟指标算死(面试"端到端延迟怎么估"):
- **TTFT(首 token,prefill 主导,算力受限)**:处理 $s_p$ 个 prompt token 的大 GEMM,$\approx 2N s_p$ FLOPs。
$$\text{TTFT}\approx\frac{2N\,s_p}{\pi\cdot\text{MFU}}+\text{(网络/排队)}.$$
prompt 越长、TTFT 越长(线性)。这是长 prompt 场景"半天不出第一个字"的原因。
- **TPOT(每后续 token,decode 主导,访存受限)**:每步读一遍权重(+KV-Cache)。
$$\text{TPOT}\approx\frac{N\,b_{\text{param}}+\text{KV 字节}}{\beta}.$$
7B fp16、$\beta$=3.35TB/s → 读权重 14GB/3.35TB/s ≈ **4.2 ms/token**(≈238 token/s 单请求上限,纯权重)。长上下文时 KV 字节加进分子,TPOT 随上下文变长而上升。
- **端到端**:生成 $g$ 个 token 的总时延 $\approx\text{TTFT}+g\times\text{TPOT}$。

## ⑥ batch 上限受 KV-Cache 卡死

② 说"攒 batch 几乎免费提吞吐",但有天花板:**batch 越大,KV-Cache 显存越大**(见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|KV 显存]])。
$$\text{可用显存}=\text{总显存}-\text{权重},\quad b_{\max}=\frac{\text{可用显存}}{2Lsd_{kv}\cdot b_{\text{param}}}.$$
例:H100 80GB、7B 权重 14GB → 剩 66GB;单条 s=4096 的 KV ≈4.3GB(MHA) → $b_{\max}\approx15$。**这就是为什么 GQA/MQA(KV 砍 4-32×)、PagedAttention(减碎片)、KV 量化对吞吐这么关键**——它们直接放大 $b_{\max}$,而吞吐 ∝ batch。换 GQA 后 $b_{\max}$ 涨到 ~60,吞吐随之 4×。

![[roof-batchKV卡死.png]]

## ⑦ 投机解码:用算力换访存,绕开 decode 的带宽墙

decode 访存受限 → 算力大量闲置。**投机解码(speculative decoding,Leviathan 2023)**用一个小"草稿模型"一次猜 $k$ 个 token,再用大模型**一次前向并行验证**这 $k$ 个:
- 大模型本来 decode 是 batch=1、算术强度 ≈1;验证 $k$ 个 token 时算术强度 ≈$k$,**把闲置算力用起来**。
- 草稿命中则一次推进多 token,TPOT 摊薄;不命中回退,保证输出分布**严格等于**大模型(无损)。
- 本质:**把 decode 从"逐 token 访存受限"变成"小批量、更高算术强度"**,用便宜的草稿算力换大模型的访存次数。典型 2-3× 加速。同理 Medusa(多头并行预测)、EAGLE 等。

## ⑧ prefill/decode 分离 + chunked prefill

两阶段性质相反,放一起会互相干扰(长 prompt 的 prefill 会阻塞其他请求的 decode,拉高 TPOT 抖动)。现代推理系统两招:
- **PD 分离(disaggregation)**:prefill 和 decode 跑在**不同 GPU 池**,各自优化(prefill 用强算力卡、decode 用大带宽卡),KV-Cache 在两池间传。代表:DistServe、Mooncake。
- **chunked prefill**:把长 prompt 的 prefill **切成小块**,和正在 decode 的请求**混批**调度,既填满算力(prefill 块)又不让 decode 饿死(vLLM、Sarathi)。

**MoE 推理**:decode 时每 token 只激活 top-k 专家,但**不同 token 可能路由到不同专家**,batch 内要 all-to-all 派发 token,访存模式更复杂;且要把所有专家权重都常驻显存(按总参),是 MoE 推理显存大的原因。

## A100 vs H100 脊点对比(换硬件结论变吗)

脊点 $I^*=\pi/\beta$ 随卡变,但 decode 的 $I\approx1$ 远低于任何主流卡的脊点,**结论不变**:
- **H100**:990 TFLOP/s BF16、3.35 TB/s → $I^*\approx296$。
- **A100 80GB**:312 TFLOP/s BF16、2.0 TB/s → $I^*\approx156$。
两卡脊点差近 2×,但 decode 的 $I\approx1$ 在两者上都是"死死钉在带宽斜坡"——所以 **decode 永远访存受限,换卡只改带宽(改 TPOT 绝对值),不改"访存受限"这个定性结论**。A100 带宽低 → decode 更慢,这是 H100 推理吞吐显著高的主因之一。

## ④ 代码:Roofline 判定 + 吞吐延迟估算

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved                      # FLOP / Byte

def roofline_perf(I, peak_flops, bandwidth):
    """可达性能 = min(算力屋顶, 带宽斜坡)。"""
    return min(peak_flops, I * bandwidth)

def is_memory_bound(I, peak_flops, bandwidth):
    ridge = peak_flops / bandwidth                  # 脊点 I*
    return I < ridge, ridge

# H100 近似:1000 TFLOP/s,3.35 TB/s
PEAK, BW = 1e15, 3.35e12

# decode 一步:7B fp16,batch=1,读 14GB 权重,算 ~2N FLOPs
N = 7e9
flops_decode = 2 * N                                # ≈1.4e10
bytes_decode = N * 2                                # 读一遍 fp16 权重
I_dec = arithmetic_intensity(flops_decode, bytes_decode)
mb, ridge = is_memory_bound(I_dec, PEAK, BW)
print(f"decode I={I_dec:.1f}, 脊点={ridge:.0f} → 访存受限={mb}")  # I≈1, 访存受限

# ❌ 错:以为加大 batch 会让单请求更快 → 反了,单请求延迟会升
# ✅ 对:加大 batch 提的是吞吐(权重只读一遍却算多个 token)
for b in [1, 8, 32]:
    flops = 2 * N * b                               # 计算量 ×b
    bytes_ = N * 2                                  # 权重访存不变!
    I = arithmetic_intensity(flops, bytes_)
    perf = roofline_perf(I, PEAK, BW)
    print(f"batch={b:2d}: I={I:5.1f}, 可达={perf/1e12:6.1f} TFLOP/s")
# batch 越大 I 越高、越接近算力屋顶 → 吞吐提升,直到越过脊点饱和
```

## 面试高频

- **Q:LLM 推理两阶段、各受什么限制?** prefill(处理 prompt)算力受限,决定 TTFT;decode(逐 token 生成)访存受限,决定 TPOT。
- **Q:为什么 decode 访存受限?** 每步只生成 1 token、计算量约 2N 很小,却要把整个模型权重从显存读一遍,算术强度约 1 FLOP/Byte,卡在带宽上。
- **Q:Roofline 模型怎么用?** 算出算子的算术强度 I=FLOPs/字节,与脊点 I*=峰值算力/带宽 比;I<I* 访存受限,I>I* 算力受限。
- **Q:吞吐和延迟为什么矛盾?** 加大 batch 提高算术强度和总吞吐,但单请求要等更大批一起算,延迟上升;在线服务要低延迟(小 batch),离线要高吞吐(大 batch)。
- **Q:加大 batch 为什么几乎免费提吞吐(到一定程度)?** decode 受带宽限,权重只读一遍却能算多个 token,batch 内分摊访存,吞吐近线性升,直到算术强度逼近脊点转算力受限。
- **Q:怎么分别优化 prefill 和 decode?** prefill 上更强算力/FP8;decode 上更高带宽、权重量化、攒大 batch、PagedAttention + continuous batching;还可 prefill/decode 分离部署。
- **Q:TTFT 和 TPOT 怎么估?** TTFT ≈ $2Ns_p/(\pi\cdot\text{MFU})$(随 prompt 长线性);TPOT ≈ (权重字节+KV字节)/带宽,7B fp16 约 4.2ms/token(≈238 tok/s 单请求上限);端到端 ≈ TTFT + 生成数×TPOT。
- **Q:batch 能无限大吗?** 不能,KV-Cache 显存卡死:$b_{\max}$=(总显存−权重)/单条KV。H100 80GB 7B MHA s=4096 约 15;换 GQA 涨到 ~60,吞吐随之 4×——这是 GQA/PagedAttention/KV 量化对吞吐关键的原因。
- **Q:投机解码为什么能加速访存受限的 decode?** 小草稿模型一次猜 k 个 token,大模型一次前向并行验证(算术强度从 1 升到 k,用闲置算力换访存次数),命中则多推进、不命中回退,输出无损。典型 2-3×。
- **Q:为什么要 prefill/decode 分离?** 两阶段性质相反,长 prompt 的 prefill 会阻塞 decode、拉高 TPOT 抖动。PD 分离让两者跑不同 GPU 池各自优化;chunked prefill 则把长 prefill 切块与 decode 混批。
- **Q:换 A100 还是 H100,decode 的访存受限结论变吗?** 不变。脊点 A100≈156、H100≈296,但 decode 的 I≈1 在两者都远低于脊点,永远访存受限;换卡只改带宽(改 TPOT 绝对值),H100 带宽高故推理更快。

## 关键事实

- LLM 推理 prefill 算力受限、decode 访存受限,是 Roofline 分析的标准结论(《LLM Inference Unveiled: Survey and Roofline Model Insights》, Yuan et al., 2024, arXiv:2402.16363)。
- Roofline 模型(可达性能 = min(峰值算力, 算术强度 × 带宽),脊点 = 算力/带宽)出自《Roofline: An Insightful Visual Performance Model》(Williams, Waterman, Patterson, 2009, CACM)。
- TTFT(首 token 时延,prefill 主导)与 TPOT/ITL(每 token 时延,decode 主导)是 LLM 服务的两大延迟指标;吞吐以总 token/s 衡量(MLPerf Inference、vLLM 文档)。
- Continuous batching(连续批处理)让解码完的请求即时被新请求替换,大幅提升 GPU 利用率与吞吐,由 Orca(Yu et al., OSDI 2022)提出、vLLM 普及。
- PagedAttention 用分页管理 KV-Cache、减少碎片,支持更大 batch 从而提吞吐(Kwon et al., 2023, arXiv:2309.06180,见 [[026 PagedAttention 与 KV 分页]]、[[102 KV-Cache]])。
- H100(SXM)BF16 峰值约 990 TFLOP/s、HBM3 带宽约 3.35 TB/s,脊点算术强度约 300 FLOP/Byte(NVIDIA H100 数据手册,2022);decode 的 I≈1 远低于此。
- A100 80GB BF16 峰值 312 TFLOP/s、HBM2e 带宽约 2.0 TB/s,脊点约 156 FLOP/Byte(NVIDIA A100 数据手册);decode 在 A100/H100 上均访存受限,换卡只改带宽不改结论。
- 投机解码(Leviathan et al., 2023, arXiv:2211.17192;Chen et al., 2023, arXiv:2302.01318)用草稿模型并行验证,无损加速 decode 2-3×;变体 Medusa(Cai et al., 2024)、EAGLE(Li et al., 2024)。
- batch 上限受 KV-Cache 显存约束:$b_{\max}$=(总显存−权重)/单条KV;GQA/MQA、PagedAttention、KV 量化通过缩小 KV 放大 $b_{\max}$ 从而提吞吐。
- prefill/decode 分离(DistServe, Zhong et al., 2024, arXiv:2401.09670;Mooncake, 2024)与 chunked prefill(Sarathi, Agrawal et al., 2023;vLLM)是降低 TPOT 抖动、提升整体吞吐的部署技术。
