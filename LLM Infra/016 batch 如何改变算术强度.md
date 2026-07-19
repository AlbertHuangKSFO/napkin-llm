[[016 batch 如何改变算术强度|batch 改变算术强度]]揭示推理调优的核心杠杆:[[014 Decode 阶段：访存受限|Decode]] 在 batch=1 时常是访存受限(算术强度约 1 FLOP/byte),但每提高一档 batch,**一次权重搬运可被多行激活复用**,在权重主导的简化模型中算术强度近似随 batch 线性上升。它常先提高系统吞吐；是否、何时跨过计算脊点则取决于 KV、attention、通信、dtype、形状和有效带宽，必须以目标负载实测。这正是 [[LLM/105 连续批处理 continuous batching|连续批处理]] 想利用的复用，也是 [[017 吞吐与延迟：根本性张力|吞吐与延迟张力]]的物理根源。

## 直觉

权重搬运常是 decode 的固定大开销。batch=1 时,一份权重只服务 1 个请求的 1 个 token；batch 增大后,同一轮调度可让一份权重服务多行激活，搬运成本被摊到多份上。这里的"同一次"是 Roofline 层的复用模型，不保证真实服务的 wall-clock 不变：KV 读取、collective、调度和 kernel 效率都会改变单步时间。

工业类比:把整套模具吊上机床(搬权重)很贵。一次只冲 1 件 → 行车忙、机床可能闲(memory-bound)。一次冲 32 件(batch)→ 同一次吊装被摊薄，算术强度上升；是否已经算力饱和，要看这台机床、模具和其他工序。算术强度 = "吊一次模具能冲多少件",batch 直接抬它。

转折点:当 batch 大到算术强度越过机器平衡点,瓶颈可能从带宽切换到算力,再加 batch 的吞吐增益会放缓。在此之前，batch 往往能显著提高吞吐，但不是"免费"：排队、KV 显存、attention 与通信可能先成为约束。

**生活类比**:外卖骑手送餐。去小区的主路像权重搬运，顺路捎单能分摊这段路；但每单自己的楼道、地址核对像 KV/attention，单子越多这些成本越会累积。`batch=1` 是一趟送 1 单；`batch=10` 可显著提高一趟总产出，却不能承诺每位用户仍在同一时间拿到餐。箱子容量、排队和路线绕行决定何时收益变缓——这就是吞吐与延迟张力的来源。

![[pd-016类比骑手捎单.png]]

## 例子

沿用上一节由 **989 TFLOPS / 3.35 TB/s** 构造的**理想参考屋顶示例**：$\text{AI}_{\mathrm{ref}}^*\approx295$ FLOP/byte；在权重主导、每元素 $b=2$ byte 的 GEMM 简化里，对应的**参考转折批量** $B^*_{\mathrm{ref}}\approx295$。下表里的 `295` 只指这个参考 $B^*_{\mathrm{ref}}$，不是生产 BF16 服务的既定拐点：

| batch $B$ | 算术强度 ≈ $2B/b$ | 参考状态(沿用 $B^*_{\mathrm{ref}}=295$) | 权重复用 | 生产判断 |
|---|---|---|---|---|
| 1 | ≈ 1 | memory-bound | 基线 | 实测 |
| 16 | ≈ 16 | memory-bound | 一次权重流量可服务 16 行 | 实测 |
| 128 | ≈ 128 | 仍在参考脊点左侧 | 权重可被 128 行复用 | 需实测 |
| 256 | ≈ 256 | 仍在参考脊点左侧 | 权重可被 256 行复用 | 需实测 |
| 512 | ≈ 512 | 参考跨线例 | 不能据此外推生产服务 | 需实测 |

- 在权重主导且无排队的理想模型里，batch 从 1→16 会把权重搬运摊给 16 个 token；真实吞吐和 TPOT 仍要从 profiler / 服务指标读取。
- $B=256$ 在这个参考示例下仍未跨线；$B=512$ 才是**用于说明判定器的参考跨线例**。真实 BF16 服务必须以相同 dtype、实际 kernel 和有效带宽重算；更大的 batch 还可能被 KV、显存容量、collective 或调度先限制。

![[pd-016batch吞吐延迟此消彼长.png]]

## 原理

decode 主算子 $Y=XW$,$X\in\mathbb{R}^{B\times d}$。搬运字节以权重 $d^2 b$ 为主(被所有 $B$ 行共享),FLOPs $=2Bd^2$:

$$
\text{AI}(B) = \frac{2Bd^2}{d^2 b + 2Bd\,b}\ \xrightarrow{\ d \gg B\ }\ \frac{2B}{b}
$$

线性于 $B$。与机器平衡点 $\text{AI}^* = \dfrac{\text{峰值 FLOP/s}}{\text{HBM 带宽}}$ 比较,转折批量:

$$
B^*_{\mathrm{ref}} \approx \frac{b}{2}\cdot \text{AI}_{\mathrm{ref}}^* = \frac{b}{2}\cdot\frac{\text{参考峰值 FLOP/s}}{\text{参考带宽}}\quad(b=2:\ B^*_{\mathrm{ref}}\approx \tfrac{2}{2}\times 295\approx295)
$$

- $B < B^*_{\mathrm{ref}}$:在这个简化参考模型中处于参考脊点左侧，权重复用会提高系统吞吐；TPOT 是否近似不变要看 $K(S)$、queue 与实际 kernel。
- $B > B^*_{\mathrm{ref}}$:在这个简化参考模型中处于参考脊点右侧，吞吐增益会放缓；单请求延迟通常需要用服务实测确认。

注意 KV-Cache 字节也随 $B$ 涨(见 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]])。因此这个 $B^*$ 只是权重主导 GEMM 的参考值；真实拐点不必更小或更大，应该对给定模型、上下文长度、并行策略、dtype 和 GPU 实测。

## 图

## 代码

扫不同 batch,观察吞吐与单请求 TPOT 的此消彼长,并定位转折点:

```python
import torch, time

@torch.no_grad()
def sweep_batch(model, vocab, d_prompt=512, batches=(1,4,16,64,128,256,512), n=32):
    rows = []
    for B in batches:
        ids = torch.randint(0, vocab, (B, d_prompt), device='cuda')
        out = model(ids, use_cache=True); past = out.past_key_values
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            out = model(tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1e3 / n      # 单步耗时
        tpot = step_ms                                       # 每请求 TPOT
        sys_tok_s = B * 1000 / step_ms                       # 系统吞吐 tok/s
        rows.append((B, round(tpot, 1), round(sys_tok_s)))
    return rows  # 用实测行而非屋顶图，判断吞吐、TPOT 与 OOM/SLO 的共同拐点

def b_star(flops_TFs, bw_TBs, bytes_per=2):
    return bytes_per / 2 * (flops_TFs / bw_TBs)              # 参考值：≈295；真实 BF16 用同 dtype、kernel、有效带宽重算

# ❌ 误区：以为 batch 越大单请求也越快（把"系统吞吐"当成"单请求速度"）
# ✅ 事实：先测 t_step、sys_tok_s、TPOT、显存和 SLO；B* 只解释一个受控 Roofline 近似
```

`❌` 把"系统吞吐随 batch 涨"误读成"单个用户的 token 速度也随 batch 涨";`✅` batch 主要抬的是系统复用，单请求 TPOT / ITL 是否变差需与 queue、KV、attention 和 SLO 一起实测。若要估真实 BF16 服务的转折点，必须按**同 dtype、同 kernel、同有效带宽**重算自己的 $B^*$。

## 面试高频

- **Q:batch 为什么可能把 decode 从 memory-bound 推向 compute-bound?** 权重搬一次被 $B$ 行复用，在权重主导近似下算术强度 ≈ $2B/b$ 随 batch 线性升；是否越过机器平衡点仍须匹配真实 kernel 的访存、计算和通信。
- **Q:转折批量 $B^*$ 怎么估?** 在权重主导的 GEMM 近似中，$B^*\approx \tfrac{b}{2}\cdot\dfrac{\text{峰值 FLOP/s}}{\text{带宽}}$。沿用 989/3.35 的参考屋顶与 $b=2$ 可得 $B^*_{\mathrm{ref}}=295$，因此 $B=256$ 仍在参考脊点左侧、$B=512$ 是参考跨线例；真实 BF16 服务必须用同 dtype、实际 kernel 与有效带宽重算，不能当作 H100 的既定 batch 阈值。
- **Q:加 batch 是不是没有代价?** 不是。KV 显存、KV/attention 访存、collective、排队与 SLO 都会约束它；即便权重复用仍在提升，TPOT / ITL 也可能先变差。
- **Q:prefill 也吃 batch 红利吗?** prefill 通常具有更高数据复用，额外 batch 的边际收益常与 decode 不同；它是否 compute-bound 仍取决于模型、长度、形状与硬件。
- **Q:这和 continuous batching 什么关系?** 连续批处理动态把可运行请求拼进同一 decode step，以提高权重复用；调度器还要用 TTFT/TPOT SLO、KV 容量和队列做约束，而不是盲目逼近一个理论 $B^*$。

## 关键事实

- 在权重主导、$d\gg B$ 的 GEMM 近似中，decode 算术强度为 $\text{AI}(B)\approx2B/b$；KV、attention、通信与非理想访存会改变真实曲线。
- 将 989/3.35 代入得到的 $295$ FLOP/byte 是**参考屋顶示例**，而不是“H100 BF16 的既定脊点/最佳 batch”或可迁移生产阈值；真实 BF16 服务应按同 dtype、实际 kernel 和有效带宽重新测量。
- 截至 2026-07-19，NVIDIA H100 SXM 官方列出 TF32 Tensor Core 989 TFLOPS(带稀疏性脚注)和 HBM 3.35 TB/s；本文用它们构造 $\text{AI}_{\mathrm{ref}}^*\approx295$ FLOP/byte，并在 $b=2$ 的简化里得到 $B^*_{\mathrm{ref}}\approx295$ 这个**理想参考**，不是 BF16 服务的实测脊点。真实 BF16 服务应按同 dtype、同 kernel、同有效带宽重算。来源:NVIDIA H100 产品规格页(核验于 2026-07-19)。
- continuous batching(vLLM 等)动态合批以提高权重复用和吞吐，但 batch 大小须由模型、上下文、并行、显存及 TTFT/TPOT SLO 共同决定。来源:Orca(Yu et al., OSDI 2022)；vLLM(Kwon et al., SOSP 2023)。
