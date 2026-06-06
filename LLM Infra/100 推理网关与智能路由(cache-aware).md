[[100 推理网关与智能路由(cache-aware)]]:把请求送到「最该去的那张卡」的那一层。普通 L7 负载均衡只看连接数轮询,对 LLM 的有状态 KV 一无所知;**LLM 专用推理网关**反过来,把 [[038 KV-aware 路由与跨引擎复用|KV-aware 路由]]、[[032 前缀缓存：RadixAttention 树结构|前缀缓存]] 命中、队列深度都当成路由信号,选出能复用 KV、又不过载的副本。2025 年这套能力被标准化进 **Gateway API Inference Extension**(GIE),也是 [[098 llm-d：K8s 原生分布式推理|llm-d]] 与 [[097 KServe：模型生命周期与 LLMInferenceService|KServe]] 的入口层。它和 [[112 负载均衡与会话亲和(prefix affinity)|prefix affinity]]、[[060 数据并行与副本扩展|副本扩展]] 直接相关:扩出多副本后,「怎么分流量」才是决定 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT/goodput]] 的关键。

## ① 类比:懂业务的前台 vs 抽号机

普通 L7 LB 像超市抽号机:谁来都给下一个空窗口,绝对公平、绝对无脑。推理网关像医院**分诊台**:它知道你上次找的是 3 号医生(病历=KV 已在那台机器)、知道 5 号窗口在排长队、知道 2 号刚空出来——于是把你导向「既认识你病历、又不堵」的那个口。对 LLM 而言,「认识病历」= 目标副本的 KV-Cache 里已有你的前缀,直接跳过 [[013 Prefill 阶段：计算受限|prefill]] 重算。

## ② 小数字例子:同前缀打中与打偏

设 system prompt + few-shot 共 **2000 token** 前缀,8B 模型单卡 prefill 约 **2000 tok ÷ ~10000 tok/s ≈ 0.2 s**。

- **cache-aware 命中**:网关把请求路由到已缓存该前缀的副本,prefill 几乎为 0,TTFT 主要剩排队 + 首 token,可能 **~50 ms**。
- **round-robin 打偏**:落到没缓存的副本,要重算 2000 token 前缀,TTFT 多出 **~200 ms**;高并发下命中率从 90% 掉到 ~30%,P99 TTFT 翻几倍,GPU 还在重复算同一段 prefill。

一句话:**前缀命中率是网关的核心 KPI**,直接换算成省下的 prefill FLOPs 与 TTFT。

## ③ 原理:网关凭什么「懂推理」

**1. 路由信号变了。** 普通 LB 的信号是连接数 / 轮询 / 最少连接;推理网关的信号是**实时推理态**:
- 各副本的 **KV-Cache 利用率**(快满的别再塞,见 [[031 KV 显存碎片与 block 管理|KV block]])。
- **等待队列深度 / 运行中请求数**(避开过载副本,呼应 [[047 准入控制与排队论：队列长度到延迟|排队论]])。
- **前缀命中**:维护「哪个 prefix 在哪个副本」的索引,优先路由到命中副本,最大化 [[033 自动前缀缓存的命中与失效|APC]] 复用。

**2. Gateway API Inference Extension(GIE)。** 2025 年 K8s SIG 把它标准化:引入 `InferencePool`(一组跑在共享 GPU 上的模型服务 pod)与 `InferenceObjective`(SLO/优先级),并把选副本的逻辑外置成 **EndpointPicker(EPP)** 这个 external processing 服务。Gateway(Envoy / Istio / NGINX Gateway Fabric 等)收到请求后,把候选 endpoint 列表交给 EPP,EPP 按上面的指标 + 前缀感知挑一个返回。这样「智能路由」与「网关数据面」解耦,可插拔。

**3. 和 prefill/decode 解耦联动。** 在 [[048 为何分离 prefill 与 decode|PD 分离]] 架构里,网关还要区分把 prefill 请求送到 prefill 池、decode 送到 decode 池,并跨池传 KV(NIXL,见 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]])。这正是 [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]]、llm-d router 在做的事。

![[orch-cache-aware路由.svg]]

## ④ 配置:从普通 LB 到 InferencePool

```yaml
# Gateway API Inference Extension:用 InferencePool 替代普通 Service
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm-llama3-pool
spec:
  selector:                       # 选中一组 vLLM pod
    app: vllm-llama3-8b
  targetPortNumber: 8000
  extensionRef:
    name: vllm-epp                # 指向 EndpointPicker(EPP)做智能选副本
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs: [{ name: inference-gateway }]
  rules:
    - backendRefs:                # 后端是 InferencePool,而非裸 Service
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-pool
```

❌ 反模式:把多副本挂在普通 `Service` + kube-proxy round-robin 后面。同一长前缀被随机打散到各副本,前缀缓存命中率塌方,每个副本都在重复 prefill,过载副本照样收新请求 → P99 TTFT 爆炸、GPU 利用率虚高。

✅ 正解:用 **InferencePool + EPP**,以 KV 利用率 / 队列深度 / 前缀命中为路由信号;把同前缀粘到已缓存副本(prefix affinity),过载副本自动避让。换引擎/扩副本时,网关层不动,只改 pool 成员。


![[orch-100命中vs打偏.svg]]

## 面试高频

- **「推理网关和普通 L7 LB 差在哪?」** 信号不同:普通 LB 看连接数/轮询,对 LLM 有状态 KV 无感;推理网关把 KV 利用率、队列深度、前缀命中当路由信号,目标是抬高前缀复用、避开过载副本、优化 TTFT/goodput。
- **「Gateway API Inference Extension 是什么?」** 2025 年 K8s 标准:`InferencePool` 抽象一组模型服务 pod,`InferenceObjective` 表达 SLO,选副本逻辑外置成 `EndpointPicker`(EPP),与 Envoy/Istio 等数据面解耦。
- **「cache-aware 路由为什么能降 TTFT?」** 把同前缀请求粘到已缓存该 prefix 的副本,直接跳过 prefill 重算;命中率每升一档,省下的就是整段 prefill FLOPs 与排队。
- **「网关和 PD 分离怎么配合?」** 网关区分 prefill/decode 请求分别路由到对应池,并触发跨池 KV 传输(NIXL);这是 Dynamo / llm-d router 的核心职责。
- **「prefix affinity 有什么坑?」** 太粘会让热点前缀挤爆单副本造成热点不均;要在「命中率」和「负载均衡」之间权衡,EPP 通常同时看命中与队列深度。

## 关键事实

- **Gateway API Inference Extension(GIE)**:K8s SIG-Network 项目,2025 年随官方博客推广;核心 CRD `InferencePool` + 外置 `EndpointPicker(EPP)`,Istio / Envoy AI Gateway / NGINX Gateway Fabric 已支持。
- 路由信号 = **KV-Cache 利用率 + 队列深度 + 运行请求数 + 前缀命中**,而非连接数轮询。
- **prefix-cache aware routing** 是 llm-d / KServe LLMInferenceService 的入口能力,经 Envoy AI Gateway + GIE 实现。
- 命中前缀 ≈ 省掉整段 prefill;2000 token 前缀在 8B 单卡约省 ~0.2 s TTFT。
- 网关层把「智能路由」与「引擎实现」解耦:换引擎(vLLM↔SGLang)、扩副本都不动网关,只改 pool 成员,呼应 [[094 OpenAI 兼容 API 与引擎抽象|OpenAI 兼容契约]]。
