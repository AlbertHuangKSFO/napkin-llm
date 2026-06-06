[[016 batch 如何改变算术强度|batch 改变算术强度]]揭示推理调优的核心杠杆:[[014 Decode 阶段：访存受限|Decode]] 在 batch=1 时是访存受限(算术强度 ≈ 1 FLOP/byte),但每提高一档 batch,**权重只需从 HBM 搬一次却被 batch 行激活复用**,算术强度近似随 batch 线性上升;当它越过机器平衡点,decode 就从 memory-bound 走向 **compute-bound**,系统吞吐随之大涨——但代价是单请求的 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TPOT]] 变长。这正是 [[LLM/105 连续批处理 continuous batching|连续批处理]] 想最大化的东西,也是[[017 吞吐与延迟：根本性张力|吞吐与延迟张力]]的物理根源。

## 直觉

权重搬运是 decode 的固定大开销。batch=1 时,你花 42 ms 把 70B 权重搬上来,只服务 1 个请求的 1 个 token——极度浪费。batch=32 时,**同一次搬运**(还是那 42 ms)服务 32 个请求各 1 个 token,搬运成本被摊到 32 份上,算力终于有活干。

工业类比:把整套模具吊上机床(搬权重)很贵。一次只冲 1 件 → 行车忙死、机床闲死(memory-bound)。一次冲 32 件(batch)→ 同一次吊装摊薄,机床开始吃满(compute-bound)。算术强度 = "吊一次模具能冲多少件",batch 直接抬它。

转折点:当 batch 大到算术强度越过机器平衡点,瓶颈从带宽切换到算力,再加 batch 吞吐就不再线性涨(算力饱和)。在此之前,加 batch 几乎是"免费"涨吞吐。

**生活类比**:外卖骑手送餐。从店到小区那段路是**固定要跑的**(对应每步都得从 HBM 搬一次 140GB 权重,跑一趟 42ms)。`batch=1` = 骑手一趟只送 1 单 → 跑了 42ms 的路只送出 1 个 token,路全白跑(算术强度 ≈ 1,memory-bound,摩托再快也没用)。`batch=10` = 顺路捎 10 单 → **还是同一趟路、还是 42ms**,却一口气送出 10 个 token,路程被摊薄 10 倍,吞吐 ≈ 10×;而每位用户拿到自己那单的时间几乎不变(单请求 TPOT 没变慢)。这就是 continuous batching 的"免费红利区"。但箱子有上限:塞到一定单数后摩托后座装不下、配送本身忙不过来(算术强度越过机器平衡点 B*≈295),再加单吞吐就不再线性涨,且每位用户开始等久一点(TPOT 上升)——这就是吞吐与延迟张力的来源。

![[pd-016类比骑手捎单.svg]]

## 例子

Llama 70B,H100(算力 ≈ 990 TFLOP/s,带宽 ≈ 3.35 TB/s,机器平衡点 $\text{AI}^*\approx 295$ FLOP/byte),BF16:

| batch $B$ | 算术强度 ≈ $2B/b$ | 状态 | 每 token 搬权重 | 系统吞吐(相对) |
|---|---|---|---|---|
| 1 | ≈ 1 | memory-bound | 140 GB | 1× |
| 16 | ≈ 16 | memory-bound | 140 GB(摊 16 份) | ≈ 16× |
| 128 | ≈ 128 | 接近脊线 | 140 GB(摊 128 份) | ≈ 100×+ |
| 256+ | ≳ 295 | compute-bound | — | 增速放缓 |

- batch 从 1→16:权重搬运不变(还是 140 GB/步),但一步出了 16 个 token →**系统吞吐 ≈ 16 倍**,而单请求 TPOT 几乎不变(都 ≈ 42 ms)。这就是 continuous batching 的红利区。
- batch 继续涨到越过脊线:算力饱和,TPOT 开始随 batch 上升(每步要算的 FLOP 多了),单请求变慢。

![[pd-016batch吞吐延迟此消彼长.svg]]

## 原理

decode 主算子 $Y=XW$,$X\in\mathbb{R}^{B\times d}$。搬运字节以权重 $d^2 b$ 为主(被所有 $B$ 行共享),FLOPs $=2Bd^2$:

$$
\text{AI}(B) = \frac{2Bd^2}{d^2 b + 2Bd\,b}\ \xrightarrow{\ d \gg B\ }\ \frac{2B}{b}
$$

线性于 $B$。与机器平衡点 $\text{AI}^* = \dfrac{\text{峰值 FLOP/s}}{\text{HBM 带宽}}$ 比较,转折批量:

$$
B^* \approx \frac{b}{2}\cdot \text{AI}^* = \frac{b}{2}\cdot\frac{\text{峰值 FLOP/s}}{\text{带宽}}\quad(\text{H100, BF16: } B^*\approx \tfrac{2}{2}\times 295 \approx 295)
$$

- $B < B^*$:memory-bound,加 batch **吞吐线性涨、TPOT 几乎不变**(黄金区)。
- $B > B^*$:compute-bound,加 batch 吞吐增速放缓、**TPOT 开始升**(算力饱和)。

注意 KV-Cache 字节也随 $B$ 涨(见 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]]),故实际 $B^*$ 还受显存容量与 KV 搬运拖累,通常比理论值小。

## 图

![[pd-batch改变算术强度曲线.svg]]

## 代码

扫不同 batch,观察吞吐与单请求 TPOT 的此消彼长,并定位转折点:

```python
import torch, time

@torch.no_grad()
def sweep_batch(model, vocab, d_prompt=512, batches=(1,4,16,64,256), n=32):
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
    return rows  # 期望：B↑ → sys_tok_s 先近线性涨、tpot 在 B* 后才明显升

def b_star(flops_TFs, bw_TBs, bytes_per=2):
    return bytes_per / 2 * (flops_TFs / bw_TBs)              # ≈ 295 (H100 BF16)

# ❌ 误区：以为 batch 越大单请求也越快（把"系统吞吐"当成"单请求速度"）
# ✅ 事实：B<B* 时单请求 TPOT 几乎不变、系统吞吐线性涨；B>B* 后单请求反而变慢
```

`❌` 把"系统吞吐随 batch 涨"误读成"单个用户的 token 速度也随 batch 涨";`✅` 真相是 batch 抬的是**系统吞吐**,单请求 TPOT 在越过 $B^*$ 后是上升的——吞吐与延迟的张力由此而生。

## 面试高频

- **Q:batch 为什么能把 decode 从 memory-bound 推向 compute-bound?** 权重搬一次被 $B$ 行复用,算术强度 ≈ $2B/b$ 随 batch 线性升,越过机器平衡点即转计算受限。
- **Q:转折批量 $B^*$ 怎么估?** $B^*\approx \tfrac{b}{2}\cdot\dfrac{\text{峰值 FLOP/s}}{\text{带宽}}$,H100 BF16 约 295(实际因 KV 与显存更小)。
- **Q:加 batch 是不是没有代价?** $B<B^*$ 时近似免费涨吞吐;$B>B^*$ 后单请求 TPOT 上升,且 KV 显存随 $B$ 线性涨可能先爆显存。
- **Q:prefill 也吃 batch 红利吗?** 不多——prefill 本就 compute-bound,batch 收益远小于 decode;红利主要在 decode。
- **Q:这和 continuous batching 什么关系?** 连续批处理就是动态把更多请求拼进同一 decode step、逼近 $B^*$ 以吃满算力,同时不让单请求等太久。

## 关键事实

- decode 算术强度 ≈ $2B/b$,**随 batch 线性**;这是"加 batch 涨吞吐"的物理本质(2024–2025 服务系统共识)。
- 转折批量 $B^*$(H100 BF16)理论 ≈ 295,实际受 KV 显存/搬运拖累常落在数十到一百多;超过后吞吐增速放缓、TPOT 上升。
- continuous batching(vLLM 等,2023 起标配)正是动态逼近 $B^*$、把 batch=1 时 &lt;5% 的算力利用率拉满的工程实现。
