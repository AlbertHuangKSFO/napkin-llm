> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[126 写一个 benchmark：测 TTFT、TPOT、吞吐]]:把 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标]] 从定义变成数字——用官方 `vllm bench serve` 跑 ShareGPT,或自写 asyncio 压测脚本,算出 p50/p99 的 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT/TPOT]] 与吞吐。这把「尺子」是 127/128/129 反复复用的基线工具。

## ① 类比:给餐厅做「上菜计时」

光说「我家上菜快」没用,得拿秒表实测:**第一道菜多久上桌(TTFT)**、**之后每道菜间隔多久(TPOT/ITL)**、**一小时能服务多少桌(吞吐)**。而且不能只看平均——要看**最慢的那 1% 客人(p99)**,因为 SLO 是按尾延迟定的。开环(按固定到达率发请求)还是闭环(等上一个回来再发下一个)会测出完全不同的数(见 [[106 压测方法：开环 vs 闭环、并发模型|开环 vs 闭环]])。

## ② 官方 benchmark:一条命令

```bash
# 先下 ShareGPT 数据集
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 对已起的 vllm serve 端点压测
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 10          # 开环:每秒 10 req;去掉=离线打满测最大吞吐
```

输出含 Mean/Median/**P99 TTFT、TPOT**、Request throughput(req/s)、Output token throughput(tok/s)。

## ③ 原理:三个指标怎么算、为什么看 p99

**TTFT** = 请求到达 → 收到第一个 token。受 [[013 Prefill 阶段：计算受限|prefill]]、排队、[[042 chunked prefill：切块融合|chunked prefill]] 影响。**TPOT** = (总时间 − TTFT) / (输出 token 数 − 1),即稳态每 token 间隔,受 [[014 Decode 阶段：访存受限|decode]] 带宽决定。**吞吐**分两种:req/s(请求级)和 output tok/s(token 级)。

**为什么 p99 而非均值:** [[047 准入控制与排队论：队列长度到延迟|排队]]导致延迟分布长尾,均值掩盖尾部;[[105 SLO、SLA 设计：为推理定指标|SLO]] 通常写成「p99 TTFT < Xms」。**为什么开环更真实:** 闭环会自我限流(系统慢→发得慢),掩盖过载;开环固定到达率,能压出真实饱和点和 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]]。

![[lab-压测数据流.png]]

对比开环(固定到达率)与闭环(发一个等一个)为何测出完全不同的数,以及三指标算法与「这把尺子」如何贯穿后续:

![[lab-126开环闭环.png]]

## ④ 自写 asyncio 压测脚本(可直接跑)

```python
# bench.py — 开环并发压测,算 p50/p99 TTFT/TPOT/吞吐
import asyncio, time, json, numpy as np, aiohttp

URL = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = ["解释 PagedAttention 的核心思想。"] * 200
RATE = 10        # req/s,开环到达率
MAX_TOK = 128

async def one(session, prompt, results):
    t0 = time.perf_counter(); ttft = None; ntok = 0
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOK, "stream": True}
    async with session.post(URL, json=payload) as r:
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            if ttft is None: ttft = time.perf_counter() - t0   # 第一个 chunk
            ntok += 1
    total = time.perf_counter() - t0
    tpot = (total - ttft) / max(ntok - 1, 1)
    results.append((ttft, tpot, ntok, total))

async def main():
    results = []
    async with aiohttp.ClientSession() as s:
        tasks = []
        for p in PROMPTS:
            tasks.append(asyncio.create_task(one(s, p, results)))
            await asyncio.sleep(1.0 / RATE)        # 开环:固定间隔发,不等回包
        await asyncio.gather(*tasks)
    ttfts = np.array([r[0] for r in results]) * 1000   # ms
    tpots = np.array([r[1] for r in results]) * 1000
    wall  = max(r[3] for r in results)
    out_tok = sum(r[2] for r in results)
    print(f"requests={len(results)}")
    print(f"TTFT  p50={np.percentile(ttfts,50):.1f}ms  p99={np.percentile(ttfts,99):.1f}ms")
    print(f"TPOT  p50={np.percentile(tpots,50):.1f}ms  p99={np.percentile(tpots,99):.1f}ms")
    print(f"throughput={len(results)/wall:.2f} req/s  {out_tok/wall:.1f} tok/s")

asyncio.run(main())
```

```bash
pip install aiohttp numpy
python bench.py
```

❌ 反模式:闭环(发一个等一个)还号称「压测」,或只报均值不报 p99——掩盖长尾和过载。
✅ 正解:**开环固定到达率**(`asyncio.sleep(1/RATE)` 不等回包)+ 报 p50/**p99**;扫多个 RATE 画出延迟-吞吐曲线,找拐点(饱和 QPS)。

## 面试高频

- **「TTFT、TPOT 怎么算?」** TTFT=到达→首 token;TPOT=(总时长−TTFT)/(输出 token−1)。流式下首个 SSE chunk 时间即 TTFT。
- **「为什么必须看 p99 不看均值?」** 排队致长尾,均值掩盖尾部;SLO 按 p99 定。
- **「开环 vs 闭环差别?」** 闭环自我限流(系统慢则发慢)掩盖过载;开环固定到达率能压出真实饱和点与 goodput。实测要开环。
- **「怎么找系统的最大可服务 QPS?」** 扫多个 request-rate,画延迟-吞吐曲线,p99 越过 SLO 的点就是 goodput 上限。
- **「这把尺子在后续怎么用?」** 量化(127)、调优(128)、PD 分离(129)每一步都用同一脚本对比 Δ,否则无法判断改善。

## 关键事实

- 官方:`vllm bench serve --dataset-name sharegpt --request-rate N`;输出含 P99 TTFT/TPOT、req/s、tok/s。
- 去掉 `--request-rate` = 离线打满测最大吞吐;带上 = 开环测延迟。
- TPOT=(total−TTFT)/(out_tok−1);流式首 chunk = TTFT。
- 自写脚本核心:`asyncio.sleep(1/RATE)` 实现开环、解析 SSE 取首 chunk、`np.percentile` 算 p99。
- 同一脚本贯穿 127/128/129,做基线对比。
