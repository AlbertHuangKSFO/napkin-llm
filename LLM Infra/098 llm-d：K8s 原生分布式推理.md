[[098 llm-d：K8s 原生分布式推理]]讲的是:一套 K8s 原生的分布式 LLM 推理栈,它把智能推理网关(cache-aware 路由)、[[048 为何分离 prefill 与 decode|PD 解耦]]、跨节点 KV offload、跨节点 [[057 张量并行推理：延迟换显存|TP]]/[[061 专家并行 EP：大规模 MoE 服务|EP]] 组装成"well-lit path",专门服务那些超出单节点显存的大模型。它是 [[095 从单进程到集群：服务架构演进|架构演进]]第 ④ 阶段的代表数据面,常被 [[097 KServe：模型生命周期与 LLMInferenceService|KServe LLMInferenceService]] 当作底层调度引擎来用。

## 类比
普通负载均衡像不认人的叫号机,谁空叫谁。llm-d 的网关像一个记性极好的前台:它知道"这位客人的开场白(prompt 前缀)我们 2 号工位刚算过、KV 还热着",于是把续单直接派给 2 号,省掉重算([[100 推理网关与智能路由(cache-aware)|cache-aware]])。同时它把"备料"和"上菜"拆成两条专门产线——备料档口([[013 Prefill 阶段：计算受限|prefill]],算力受限)和上菜档口([[014 Decode 阶段：访存受限|decode]],访存受限)各自优化、各自扩缩,备料快了不会卡上菜。当一道菜大到单个厨房装不下,就把好几间厨房连成一条跨楼层流水线(跨节点 TP/EP + KV offload)。

## 小数字例子
服务 DeepSeek-R1(大 MoE,远超单卡):
- 朴素多副本:单节点放不下,根本起不来;就算硬拆,prefill 与 decode 抢同引擎,长 prompt 一来 decode 全 [[043 prefill、decode 干扰与 stall|stall]]。
- llm-d:Prefill 池用多副本、低 TP 吃吞吐;Decode 池高 TP、大显存逐 token;prompt 的 KV 经 [[053 KV 传输：NIXL、点对点与带宽|NIXL]] 从 P 池搬到 D 池;[[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]] 把专家跨多节点 EP+DP 摊开。社区基准里 Wide-EP 路径在 H200 集群上达到约 2.2k tok/s/GPU 量级的持续吞吐。
- 多轮会话:前缀命中率高时,cache-aware 把同会话粘到持有 KV 的 Pod,[[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT]] 显著下降。

## 原理:四大支柱
1. **智能网关 IGW + EPP**:基于 Gateway API 与新的 `InferencePool` CRD。Endpoint Picker(EPP)是 Envoy ext-proc 旁路 sidecar,逐请求经 gRPC 选 Pod,信号包括 [[015 KV-Cache 的显存账(逐层手算)|KV-Cache]] 利用率、队列深度、活跃请求数、前缀缓存命中率。

![[orch-098EPP打分.png]]
2. **cache-aware 路由**:KV Cache Manager 跟踪"哪些 KV block 在哪个节点",把请求路由到已持有相关前缀的 Pod,避免跨副本重算([[038 KV-aware 路由与跨引擎复用|KV-aware 路由]] 的工程化)。
3. **PD 解耦**:Prefill 池与 Decode 池独立部署、独立扩缩。Decode 用高 TP/大显存;Prefill 用多副本/低 TP;两池间用 NIXL 点对点搬 KV。降 TTFT、稳 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TPOT]]。

![[disagg-llmd-nixl搬运.png]]
4. **跨节点扩展**:跨节点 TP、Wide-EP(大 MoE)、分层 KV offload(GPU/CPU/远端,[[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|LMCache 思路]]),让模型大到超单节点也能服务。

![[orch-098PD请求旅程.png]]

**控制面**全程 K8s 原生:CRD + Operator 管理 variant 自动扩缩、scale-to-zero(v0.5 引入),并支持 NVIDIA/AMD/Gaudi 多种加速器。与 KServe 互补:KServe 给生命周期/CRD 抽象,llm-d 给分布式调度数据面。

![[orch-llm-d架构.png]]

## 代码:InferencePool + 对接(声明 PD 解耦与 cache-aware)
```yaml
# llm-d 经 Gateway API Inference Extension 暴露,核心新对象是 InferencePool
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata: { name: deepseek-pool }
spec:
  selector: { app: deepseek-r1 }          # 这组 model-serving Pod
  extensionRef:
    name: endpoint-picker                  # EPP:按 KV 命中/队列深度选 Pod
---
# EPP 调度策略(cache-aware 打分)
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: EndpointPickerConfig
metadata: { name: endpoint-picker }
spec:
  plugins:
    - type: prefix-cache-scorer            # 前缀命中率加分 → 粘住持有 KV 的 Pod
    - type: kv-cache-utilization-scorer    # KV 越满越不优先
    - type: queue-depth-scorer             # 队列越深越不优先
```
```python
# 客户端无感:仍是 OpenAI 兼容,路由/PD 解耦全在网关后面
from openai import OpenAI
c = OpenAI(base_url="http://llm-d-gateway/v1", api_key="x")
c.chat.completions.create(model="deepseek-r1",
    messages=[{"role":"user","content":"接着上文..."}])  # 同会话自动粘到热 KV 的 Pod
```
```yaml
# ❌ 反例:对超单节点大模型只堆无脑多副本 + round-robin LB
#   - 单节点显存放不下 → Pod 起不来
#   - 即便拆开,prefill/decode 同引擎互相 stall,前缀跨副本重算
#   → 正解:PD 解耦池 + cache-aware EPP + 跨节点 TP/EP
```

## 面试高频
- **llm-d 解决什么 KServe 单靠 Deployment 解决不了的问题?** 分布式调度数据面:cache-aware 路由、PD 解耦、跨节点 TP/EP、分层 KV offload——服务超单节点的大模型并提效率。
- **EPP 怎么选 Pod?** Envoy ext-proc 旁路 sidecar,逐请求 gRPC,综合 KV 利用率、队列深度、活跃请求、前缀命中率打分选最优 Pod。
- **cache-aware 路由为何重要?** 多轮/共享前缀场景下把请求粘到已持有 KV 的 Pod,避免跨副本重算 prefill,降 TTFT、提吞吐。
- **PD 解耦在 llm-d 里怎么落地?** Prefill/Decode 独立池独立扩缩,KV 经 NIXL 点对点传输;decode 高 TP、prefill 多副本低 TP。
- **llm-d 和 KServe 是竞争还是互补?** 互补:KServe 管 CRD/生命周期(LLMInferenceService),llm-d 提供分布式调度数据面,常组合使用。
- **InferencePool 是什么?** Gateway API Inference Extension 引入的新 CRD,表示一组 model-serving Pod,配 EPP 做智能路由。

## 关键事实
- llm-d 2025-05 由 Red Hat、Google Cloud、IBM Research、CoreWeave、NVIDIA 联合发起;2026 进入 CNCF(早期为 Sandbox)。
- 核心能力:KV cache-aware 路由、PD 解耦、vLLM 优化调度器、Wide-EP;支持 NVIDIA/AMD/Intel Gaudi 多加速器。
- v0.5(2026-02 前后)引入可复现基准、分层 KV offload、cache-aware LoRA 路由、active-active 高可用、UCCL 传输韧性、scale-to-zero 自动扩缩。
- IGW/EPP 来自 Kubernetes Gateway API Inference Extension(InferencePool CRD);v0.3 起 IGW GA。Wide-EP 在 H200 集群社区基准约 2.2k tok/s/GPU 量级。
- 与 KServe LLMInferenceService 深度集成,是 2025–2026 K8s 原生大模型分布式推理的事实标准之一。
