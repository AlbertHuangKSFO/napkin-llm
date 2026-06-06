[[038 KV-aware 路由与跨引擎复用]] 是在多实例/多节点推理集群里,**按 KV 命中决定把请求送到哪台 worker** 的负载均衡策略:谁已经持有这条请求前缀的 KV,就优先把请求路由给谁,避免重复 prefill。它把 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]] 从「单实例内复用」推广到「跨实例/跨引擎复用」,是 [[037 Mooncake：KVCache 中心的存储池|Mooncake]] 式 KVCache 池能真正发挥作用的前提,也与 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|分层 offload]] 配合(offload 把 KV 存下来,路由负责命中它)。代表实现:**NVIDIA Dynamo** 的 KV-aware router、**SGLang router**、vLLM 等。

## 直觉类比
一群客服(worker),每人脑子里**记着自己刚服务过的几个客户的背景**(KV)。来了个新咨询,如果它和「某客服刚处理过的同一份合同」相关,把它派给那个客服最省事——他不用从头读合同(prefill),张嘴就接着答。但也不能所有活都堆给这一个人:他要是**已经排了一长队**(过载),宁可派给别人重读一遍合同,也比让客户干等强。KV-aware 路由就是在「**谁手上有现成 KV**」和「**谁现在不忙**」之间做权衡。

## 小数字例子
集群 4 台 worker,新请求前缀 = system prompt(512 tok)+ 共享文档(8000 tok),共 8512 token 前缀。
- **Worker A**:已缓存这整段前缀 → overlap ≈ 8512 token,命中。路由到 A:**跳过 8512 token 的 prefill**,只算新增问题部分 → TTFT 可能从数秒降到几百毫秒。
- **Worker B**:无缓存 → overlap = 0,要重算全部 8512 token。
- **Worker C**:也命中(overlap 8512),但当前活跃块负载很高、队列深 → 即便命中,综合分可能不如 A。
路由器对每台算:`score = overlap_weight × 前缀命中块数 − 当前负载`。把权重 `--kv-overlap-score-weight` 调大 → 更偏命中(TTFT 好,ITL 可能差);调成 **0** → 退化为**纯负载均衡**(完全不看前缀)。

## 原理:overlap 重叠分 + 负载权衡
请求到达时,路由器把 prompt **哈希成 token block 序列**,在一棵全局 **RadixTree / 前缀树**(追踪各 worker 缓存了哪些 KV 块)里查重叠,对每个 worker 算 **overlap 重叠分**(命中前缀块越多、可跳过的 prefill 越多),再减去该 worker 当前的活跃负载,综合择优:

$$
\text{worker}^* = \arg\max_{w}\Big(\,w_{\text{ov}}\cdot \text{overlap}(req, w)\;-\;\text{load}(w)\,\Big)
$$

其中 $w_{\text{ov}}$ 即 `kv-overlap-score-weight`。重叠分直接对应**省下的 prefill 代价**:

$$
\text{overlap}(req,w)=\big|\,\text{prefix\_blocks}(req)\cap \text{cached\_blocks}(w)\,\big|
$$

- $w_{\text{ov}}$ 大 → 偏命中 → TTFT 好,但热点 worker 可能积压 → ITL/TPOT 变差。
- $w_{\text{ov}}=0$ → 不看前缀 → 纯按活跃块负载均衡。

**RadixAttention**(SGLang 提出)的前缀树是这套复用的基础:支持高效匹配/插入/淘汰 KV 块,把「KV 复用」普及开来;Dynamo 在分布式环境里用哈希 + RadixTree 可扩展地追踪缓存位置。

![[kv-aware路由.svg]]

![[kv-038路由打分手算.svg]]

## 配置 / 代码
```bash
# NVIDIA Dynamo:启用 KV-aware router,调命中 vs 均衡的权衡
python -m dynamo.frontend \
  --router-mode kv \
  --kv-overlap-score-weight 1.0   # 大→偏前缀命中(TTFT↓);设 0→纯负载均衡

# SGLang router:多副本前缀感知路由(cache-aware load balancing)
python -m sglang_router.launch_router \
  --worker-urls http://w0:30000 http://w1:30000 \
  --policy cache_aware
```

```text
❌ 轮询 / 随机路由:同一份长前缀被分到不同 worker,每台都从头 prefill → 重复重算、TTFT 高、算力浪费
✅ KV-aware 路由:按 overlap 把请求送到已持有前缀 KV 的 worker(再权衡负载)→ 跳过重算、TTFT↓
```

## 面试高频
- **KV-aware 路由解决什么?** 多实例下默认轮询会让同一前缀在多台被反复 prefill;按 KV 命中路由可跨实例复用前缀,最小化重算、降 TTFT。
- **怎么算「该路由到谁」?** 把 prompt 哈希成 block,在全局 RadixTree 查各 worker 的前缀重叠分,综合 `overlap_weight × 重叠 − 负载` 取最优。
- **命中 vs 均衡怎么权衡?** `kv-overlap-score-weight` 大偏命中(TTFT 好、ITL 可能差,热点积压);为 0 退化纯负载均衡。要按 SLO 调。
- **和 RadixAttention 什么关系?** RadixAttention(SGLang)用前缀树在**单实例内**复用 KV;KV-aware 路由把这套追踪扩展到**跨实例/跨集群**,决定请求落哪台。
- **它和 PD 解耦/Mooncake 怎么配合?** Mooncake 把 KV 存进集群池,路由负责让请求命中池中/某 worker 已有的前缀;没有 KV-aware 路由,KVCache 池的复用价值发挥不出来。

## 关键事实
- **NVIDIA Dynamo** KV-aware router(**2025**,`ai-dynamo/dynamo`):跨大规模 GPU 群管理 KV,支持 SGLang / TensorRT-LLM / vLLM 后端;算 overlap 重叠分择 worker。
- 关键参数 **`--kv-overlap-score-weight`**:大→偏前缀命中、TTFT 好、ITL 可能差;设 0→纯负载均衡(不看前缀)。
- 技术基础:把请求哈希进 **RadixTree** 可扩展追踪缓存位置;**RadixAttention**(SGLang)的前缀树普及了 KV 复用。
- **SGLang router** 提供 cache-aware 负载均衡;Baseten 等报告 KV-aware 路由带来约 **2×** 更快推理(配 Dynamo)。
- 把 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]] 从单实例推广到跨引擎,是 [[037 Mooncake：KVCache 中心的存储池|Mooncake]] KVCache 池与 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|offload]] 复用落地的前提。
