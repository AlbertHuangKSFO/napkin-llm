> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[128 调优：前缀缓存 + chunked prefill + 投机解码]]:把三大「零/低成本提速开关」落成命令——`--enable-prefix-caching`([[032 前缀缓存：RadixAttention 树结构|前缀缓存]])、`--max-num-batched-tokens`([[042 chunked prefill：切块融合|chunked prefill]] token 预算)、`--speculative-config` EAGLE3([[073 投机解码系统：draft-verify 全流程|投机解码]]),并用 126 的尺子逐项压测看收益。

## ① 类比:同一台灶,三种不花钱的提速

不加硬件也能更快:**前缀缓存**=同样的开胃汤底(system prompt、few-shot)只熬一次,后面的客人直接舀(命中则 prefill 几乎免费);**chunked prefill**=把一大锅料分批下,别让一个大单堵住正在出的菜(平衡 prefill/decode);**投机解码**=小厨(draft 模型)先猜几道菜,大厨([[014 Decode 阶段：访存受限|decode]])一次性验收,验对就白赚(访存受限阶段并行验多 token)。三者都**先开后量**,看 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 实际涨多少。

## ② 三个开关:命令

```bash
# 1. 前缀缓存(V1 默认已开,显式确认;命中率 0% 也近零开销)
vllm serve $MODEL --enable-prefix-caching

# 2. chunked prefill 的 token 预算(V1 默认开,调预算大小)
vllm serve $MODEL --max-num-batched-tokens 8192
#   小预算→TTFT 友好(decode 不被长 prefill 挤);大→吞吐友好

# 3. 投机解码 EAGLE3(JSON 传 --speculative-config)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'

# 3'. 无需草稿模型权重:n-gram 投机(prompt 复用场景)
vllm serve $MODEL \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":5,"prompt_lookup_min":2}'
```

## ③ 原理:各自省在哪、何时不灵

**前缀缓存:** 共享前缀(system prompt、RAG 上下文、多轮历史)的 KV 只算一次、命中则复用 → **直接砍 TTFT 和 prefill 算力**。[[088 vLLM V1 架构剖析|V1]] 重写为常数时间驱逐,命中率 0% 也不掉,故可常开。命中率取决于负载共享前缀的比例(见 [[033 自动前缀缓存的命中与失效|命中与失效]])。

**chunked prefill:** 把长 prompt 切块、与 decode token 拼进同一 batch,避免长 prompt 独占一步、卡住正在解码的请求(缓解 [[043 prefill、decode 干扰与 stall|prefill/decode 干扰]])。`--max-num-batched-tokens` 是核心旋钮:小→保 decode 的 [[014 Decode 阶段：访存受限|TPOT]]/TTFT,大→冲吞吐。

**投机解码:** draft 一次猜 k 个 token,target 一步并行验证。decode 是访存受限,验 k 个 token 的算力几乎「白送」,接受率高则 TPOT 大降(见 [[078 接受率、草稿长度与收益分析|接受率与收益]])。[[074 EAGLE-3：工业标准投机解码|EAGLE-3]] 是工业标准。但**高并发下收益缩水**:batch 已打满算力,投机抢算力反而可能拖慢——所以**必须实测**。

**拐点小数字。** 设接受率使一次 target 步平均吐出 2.5 个有效 token(草稿 k=3)。低并发 **QPS=2** 时 batch 小、算力大量闲置,验 k token 近乎白送 → TPOT 约 $1/2.5 \approx 0.4$ 个 target-step/token,实测吞吐**提升约 +30%**。但把负载推到 **QPS=20**:此时 batch 已把 GPU 算力打满(从访存受限转入算力受限),draft 模型前向 + 验证 k 个 token 的那点算力不再「免费」,反而**挤占** target 的算力预算 → 净吞吐**反降约 −10%**。两点连起来:收益随并发单调下滑,在某个 QPS(此例约 10~15 之间)穿过 0,过了拐点就该**果断关投机**。这就是「投机不是越开越好」的量化形态——必须在**目标并发**下测 Δgoodput,而非低并发测完就上线。

![[lab-调优前后对比.png]]

三个开关各提什么指标、何时不灵,以及为何必须 base→+apc→+chunk→+spec 逐项消融而非一起开:

![[lab-128三开关消融.png]]

## ④ 逐项压测:消融实验(用 126 的尺子)

```bash
# 控制变量:每次只开一个,跑同一压测对比 Δ
declare -A CFG=(
  [base]=""
  [+apc]="--enable-prefix-caching"
  [+chunk]="--enable-prefix-caching --max-num-batched-tokens 4096"
  [+spec]="--enable-prefix-caching --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":5}"
)
for name in base +apc +chunk +spec; do
  vllm serve $MODEL ${CFG[$name]} --port 8000 & sleep 60
  vllm bench serve --backend vllm --model $MODEL \
    --dataset-name sharegpt --dataset-path ./sharegpt.json \
    --num-prompts 500 --request-rate 8 | tee tune_$name.txt
  kill %1; sleep 10
done
```

| 开关 | 主要受益指标 | 何时收益大 | 何时不灵 |
|---|---|---|---|
| 前缀缓存 | TTFT↓、prefill 算力↓ | 共享前缀多(RAG/多轮) | 前缀各不相同 |
| chunked prefill 预算 | TPOT/TTFT 平衡 | 长 prompt 混短 decode | — |
| 投机解码 | TPOT↓(低并发) | 低 QPS、接受率高 | 高并发(算力已满) |

❌ 反模式:三个一起开,看总吞吐变好就归功投机;或在高并发下硬上投机解码反而变慢还不自知。
✅ 正解:**控制变量逐项消融**,每个开关单独对比 Δgoodput;投机解码在**目标并发**下实测,接受率低 / 高并发时果断关。

## 面试高频

- **「这三个开关分别提什么指标?」** 前缀缓存→TTFT/prefill 算力;chunked prefill 预算→平衡 TPOT 与 TTFT;投机解码→低并发下 TPOT。
- **「投机解码为什么高并发收益缩水?」** decode 访存受限时验 k token 算力白送;但 batch 打满后算力已是瓶颈,投机抢算力反而拖慢——必须按目标并发实测。
- **「`--max-num-batched-tokens` 调大调小各偏向什么?」** 调小利于 decode 的 TTFT/TPOT(长 prompt 被切碎不堵车);调大利于吞吐。
- **「前缀缓存为什么 V1 敢默认常开?」** 常数时间驱逐、极少对象创建,命中率 0% 也近零开销。
- **「EAGLE3 怎么配?」** `--speculative-config` 传 JSON,`method:eagle3` + draft `model` + `num_speculative_tokens`;无草稿权重可用 `method:ngram`。

## 关键事实

- 前缀缓存:`--enable-prefix-caching`,V1 默认开,共享前缀多则 TTFT 大降。
- chunked prefill:V1 默认开,旋钮是 `--max-num-batched-tokens`(小保延迟、大冲吞吐)。
- 投机解码:`--speculative-config '{"method":"eagle3"|"ngram", "num_speculative_tokens":N, ...}'`(JSON 字符串)。
- 调优铁律:控制变量逐项消融,用 126 压测对比 Δgoodput;投机在目标并发实测。
- 投机高并发可能反而变慢(算力饱和)。
