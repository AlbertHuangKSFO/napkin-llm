[[014 Decode 阶段：访存受限|Decode]] 是[[012 自回归推理全流程：一个 token 的旅程|自回归推理]]的第二阶段:[[013 Prefill 阶段：计算受限|Prefill]] 出了第 1 个 token 后,每一步**只输入 1 个新 token**,过一遍 N 层网络、读取并追加 [[LLM/102 KV-Cache|KV-Cache]],采样出下一个 token,如此串行循环到 EOS。问题在于:为了算这 1 个 token,GPU 必须把**全部模型权重 + 全部 KV-Cache 从 HBM 搬一遍**,而真正的计算量极小,于是显存带宽先打满、算力大量闲置——decode 是 **memory-bound(访存受限)**,它直接决定 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TPOT/ITL]]。提 batch 是把它推向计算受限的主要手段(见 [[016 batch 如何改变算术强度|batch 改变算术强度]])。

## 直觉

Decode 像**为了写一个字,把整套百科全书从书库搬到桌上翻一遍,写完一个字再搬一遍**。计算(写一个字)几乎不花时间,瓶颈全在"搬书"(从 HBM 读权重和 KV)。

工业类比:工厂为生产 1 个零件,要把整套几吨重的模具从仓库吊到机床上、装好、只冲压 1 件、再吊回去。机床转得再快也没用,瓶颈是行车(显存带宽)。所以 decode 优化的核心是**少搬运**:量化(权重字节减半)、[[LLM/102 KV-Cache|GQA]](KV 字节减 8 倍)、提 batch(一次搬运服务多个请求)。

后果:单请求 decode 速度有硬上限 ≈ 显存带宽 ÷ 每 token 需搬字节,与算力多强基本无关。这就是为什么单用户跑大模型,token 速度上不去。

## 例子

Llama 70B,BF16,权重 ≈ 140 GB,H100 HBM 带宽 ≈ 3.35 TB/s:

- **每 token 至少搬权重 140 GB**(暂忽略 KV)。
- **理论 TPOT** ≈ $140\text{ GB} \div 3.35\text{ TB/s} \approx 42$ ms/token。
- **单请求吞吐上限** ≈ $1000 / 42 \approx 24$ token/s。再强的算力也突破不了,因为是带宽决定。
- **decode 一步算力** ≈ $2\times 70\times10^9 \approx 1.4\times10^{11}$ FLOP = 140 GFLOP;H100 算力 990 TFLOP/s 本可在 0.14 ms 算完,却被 42 ms 的搬运拖住 → **算力利用率仅约 0.3%**(batch=1)。
- 加上 KV-Cache:32k 上下文单序列 KV ≈ 10 GB,每 token 还要再搬这 10 GB,TPOT 进一步升。

## 原理

decode 时 $S=1$,主算子 $y = xW$ 是**向量×矩阵**(GEMV),$x\in\mathbb{R}^{1\times d}$。算术强度:

$$
\text{AI} = \frac{2\,d^2}{(d + d^2 + d)\,b}\ \approx\ \frac{2}{b}\quad(\text{权重字节远超激活,且只服务 1 行})
$$

$b$=2(BF16)时 $\text{AI}\approx 1$ FLOP/byte,**远低于**机器平衡点 $\text{AI}^*\approx 295$(H100),落在 Roofline 的**带宽斜坡**上 → memory-bound。每步性能 ≈ 带宽 / 每 token 字节,与峰值算力无关:

$$
\text{TPOT} \approx \frac{N_\text{params}\cdot b_w + \text{KV 字节}}{\text{HBM 带宽}\times u},\qquad u<1\ (\text{batch=1 时极低})
$$

提 batch $B$:激活变 $B$ 行,权重搬一次被 $B$ 行复用,$\text{AI}\approx \dfrac{2B}{b}$ 随 $B$ 升;当 $\dfrac{2B}{b}\gtrsim \text{AI}^*$ 时越过脊线、转为计算受限(详见 [[016 batch 如何改变算术强度|batch]])。

## 图

![[mem-decode访存图.svg]]

![[mem-014搬运碾压计算.svg]]

## 代码

测量单步 decode 时间(=TPOT)并估算算力利用率,对比"误以为加算力能加速 decode"的认知:

```python
import torch, time

@torch.no_grad()
def measure_tpot(model, prompt_ids, n=64):
    out = model(prompt_ids, use_cache=True)            # prefill 一次
    past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):                                 # ✅ decode：每步只喂 1 token
        out = model(tok, past_key_values=past, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n        # ms/token = TPOT

# 估算瓶颈：搬字节 / 带宽 vs 算 FLOP / 算力
def decode_bound(n_params, bw_TBs, flops_TFs, bytes_per=2):
    move_ms = (n_params * bytes_per / 1e12) / bw_TBs * 1e3      # 搬权重耗时
    compute_ms = (2 * n_params / 1e12) / flops_TFs * 1e3       # 算耗时
    # ❌ 误区：以为换更强算力(flops↑)能让 decode 变快
    # ✅ 事实：move_ms ≫ compute_ms，decode 由带宽决定，算力几乎无关
    return move_ms, compute_ms   # 例如 70B/H100 ≈ (42 ms, 0.14 ms)
```

`❌` 认为"换算力更猛的卡 decode 就更快"——错,decode 是 memory-bound,搬运时间 $\gg$ 计算时间;`✅` 该比的是 **HBM 带宽**与每 token 搬运字节,优化靠量化/GQA/batch 减少搬运。

## 面试高频

- **Q:decode 为什么是 memory-bound?** $S=1$ 退化成 GEMV,权重搬一次只服务 1 行,算术强度 ≈ 1 FLOP/byte,远低于机器平衡点,带宽先饱和。
- **Q:单请求 token/s 的硬上限怎么估?** ≈ HBM 带宽 ÷ (权重字节 + KV 字节)。70B/H100 约 24 token/s,与算力强弱无关。
- **Q:怎么提升 decode 吞吐?** 提 batch(把 memory-bound 推向 compute-bound)、权重/KV 量化、GQA/MQA 减 KV、投机解码一次产多 token、PD 分离专卡跑 decode。
- **Q:为什么提 batch 能涨吞吐但 prefill 不那么吃 batch?** prefill 本就 compute-bound(已吃满算力),batch 收益小;decode 本是 memory-bound,batch 复用权重搬运,收益巨大。

## 关键事实

- 单请求 decode 速度 ≈ 显存带宽 ÷ 每 token 搬运字节,**与峰值算力基本无关**——这是 decode 最反直觉的事实。
- batch=1 时 decode 的 GPU 算力利用率常 **&lt; 5%**(70B 上约 0.3%),提 batch 是恢复利用率的主要手段(2024–2025 共识)。
- 实测中 decode 算术强度约 **60–80 FLOP/byte 量级**(取决于 batch 与上下文),仍低于 H100 约 295 的机器平衡点,故主流仍属访存受限区。
