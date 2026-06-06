[[099 Ray Serve：Python 优先的编排]]讲的是:用 Ray 的分布式 Python 运行时来编排 LLM 服务——Ray Serve LLM 提供 OpenAI 兼容的 LLM deployment,用 deployment graph 把多个独立扩缩的组件(路由、嵌入、多个推理引擎)拼成一张图,并原生支持 [[048 为何分离 prefill 与 decode|PD 解耦]]、data-parallel attention + [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]]。相比 [[097 KServe：模型生命周期与 LLMInferenceService|KServe]]/[[098 llm-d：K8s 原生分布式推理|llm-d]] 的 "K8s/CRD 优先",它走的是 "Python 优先",是 [[095 从单进程到集群：服务架构演进|架构演进]]第 ④ 阶段里另一条主流路线。

## 类比
KServe/llm-d 像用"建筑图纸+施工队"(CRD/YAML + Operator)盖楼:声明每层结构,系统去施工。Ray Serve 像给你一套"乐高积木 + Python 胶水":你直接用代码把"前台路由""嵌入服务""模型 A 引擎""模型 B 引擎"几块积木拼起来,每块自己设几个副本、占几张卡,想加一段 RAG 预处理就在中间插一块新积木——逻辑即代码,不必先翻译成 YAML。底座 Ray Cluster 像一张统一的桌子,所有积木(actor)都摆在上面,autoscaler 按桌上的"在办请求数"自动加减积木。

## 小数字例子
一个服务里要同时跑:Llama-3-8B 主模型 + 一个 reranker + RAG 检索前处理。
- KServe 写法:三个 CRD + 一个 InferenceGraph,调度链路靠 YAML 串。
- Ray Serve:一个 `app.py`,三个 `@serve.deployment`,用 `.bind()` 组成 deployment graph;reranker 设 4 副本(GPU),前处理设 8 副本(CPU),主模型 `tensor-parallel-size=2`。
- 自动扩缩:把 LLMServer 的 `target_ongoing_requests` 设为其最大容量的 ~75%,ingress 的同步设比例;否则 server 满载而 ingress 不扩,请求全堆在网关。
- 大 MoE(DeepSeek-V3):开 data-parallel attention + EP,把专家跨多 worker 摊开,单引擎能力被水平放大。

## 原理:三层
1. **Ray Serve LLM 层**:提供 OpenAI 兼容的 `LLMServer`/`build_openai_app`,底层引擎用 [[088 vLLM V1 架构剖析|vLLM]]。把单引擎横向扩展并叠加分布式策略(PD 解耦、data-parallel attention、EP)。
2. **deployment graph**:每个 `@serve.deployment` 是图里一个节点,**各自独立设副本数与资源**(CPU/GPU),用 `.bind()` 组合;天然适合多模型组合、自定义路由、把 RAG/前后处理写进同一张图。
3. **Ray Cluster 底座**:Head 调度 + Worker(GPU/CPU)跑 actor;Autoscaler 按 `target_ongoing_requests`(在办请求数)扩缩。

**PD 解耦**:Ray Serve LLM 把负载拆到两个 vLLM 引擎——prefill 引擎算 prompt 的 [[015 KV-Cache 的显存账(逐层手算)|KV-Cache]] 并传给 decode 引擎;KV 传输用 [[053 KV 传输：NIXL、点对点与带宽|NIXL]](NIXLConnector)或 LMCacheConnector(多种后端)。目的同 098:防 prefill 拖慢 decode、各自达 SLO。

**与 K8s 优先方案的取舍**:Python 优先 = 编排即代码、灵活组合、迭代快,适合研究/复杂图/重前后处理;K8s 优先(KServe/llm-d)= 声明式、与 K8s 生态(HPA/KEDA/Gateway)无缝、运维标准化。两者不互斥:Ray 可经 KubeRay 跑在 K8s 上。

![[orch-099扩缩比例.svg]]

![[orch-RayServe架构.svg]]

## 代码:Ray Serve LLM deployment graph
```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

# 主模型:vLLM 后端,TP=2,自动扩缩按在办请求数
llm = LLMConfig(
    model_loading_config={"model_id": "llama3-8b",
                          "model_source": "meta-llama/Llama-3-8B"},
    engine_kwargs={"tensor_parallel_size": 2},
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1, "max_replicas": 8,
            "target_ongoing_requests": 12,   # ← server 容量的 ~75%
        }},
    accelerator_type="H100",
)
app = build_openai_app({"llm_configs": [llm]})   # OpenAI 兼容
serve.run(app)
```
```python
# 自定义 deployment graph:RAG 前处理 + reranker + 主模型,各自独立扩缩
@serve.deployment(num_replicas=8)                 # CPU,多副本
class Retriever:
    def __call__(self, q): ...

@serve.deployment(ray_actor_options={"num_gpus": 1}, num_replicas=4)
class Reranker:                                   # GPU
    def __call__(self, docs): ...

@serve.deployment
class Ingress:
    def __init__(self, retriever, reranker, llm):
        self.r, self.rr, self.llm = retriever, reranker, llm
    async def __call__(self, req): ...            # 路由/组合逻辑就是 Python

app = Ingress.bind(Retriever.bind(), Reranker.bind(), llm)  # graph 由代码拼出
```
```python
# ❌ 反例:扩缩比例配错
# ingress target_ongoing_requests 设得和 LLMServer 一样满
#   → server 已满载但 ingress 不扩,请求全堵在网关,p99 暴涨
# ✅ ingress 维持为 server 容量的 ~75%,让两层比例随负载一起扩
```

## 面试高频
- **Ray Serve 与 KServe/llm-d 的根本差异?** Python 优先 vs K8s/CRD 优先:Ray 用 deployment graph 把编排写成代码、灵活组合;KServe/llm-d 用声明式 CRD、与 K8s 生态无缝。不互斥(KubeRay)。
- **deployment graph 是什么?好处?** 每个 deployment 独立设副本/资源,用 `.bind()` 组合成图;天然适合多模型组合、自定义路由、把 RAG/前后处理放进同一服务并各自独立扩缩。
- **Ray Serve 怎么自动扩缩?** 按 `target_ongoing_requests`(在办请求数)扩缩;多层(ingress↔server)要保持比例,ingress 常设为 server 容量 ~75%。
- **Ray Serve 支持 PD 解耦吗?** 支持:拆成 prefill / decode 两个 vLLM 引擎,KV 经 NIXLConnector 或 LMCacheConnector 传输。
- **大 MoE 怎么服务?** data-parallel attention + 专家并行(EP),跨 worker 把专家摊开,横向放大单引擎能力(如 DeepSeek-V3 / GPT-OSS)。
- **什么时候选 Ray Serve?** 编排逻辑复杂、需重前后处理/多模型组合、团队 Python 重、迭代快;要 K8s 标准化运维与 Gateway 生态时偏 KServe/llm-d。

## 关键事实
- Ray Serve LLM 提供 OpenAI 兼容 LLM 部署,底层引擎用 vLLM;把单引擎横向扩展并支持 PD 解耦、data-parallel attention、专家并行(EP)等分布式策略(2025–2026)。
- PD 解耦把 prefill/decode 拆成两个 vLLM 引擎,KV 传输支持 NIXLConnector(NVIDIA NIXL,网络传输)与 LMCacheConnectorV1(多后端缓存)。
- 自动扩缩按 `target_ongoing_requests`;多层部署需维持比例(ingress ≈ server 容量 75%),否则瓶颈堆到网关。
- Ray 既可独立部署,也可经 KubeRay 跑在 K8s 上,与 KServe/llm-d 这类 K8s 优先方案并非二选一。
