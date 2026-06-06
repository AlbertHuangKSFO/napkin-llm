[[104 AIBrix 与其他生产栈]]:前面几篇拆了网关、扩缩、调度、发布这些「零件」,这一篇把它们装成**整套生产栈**,并做选型对比。核心三家:**AIBrix**(字节,自带电池的 vLLM 控制面)、[[098 llm-d：K8s 原生分布式推理|llm-d]](K8s 原生分布式调度层)、[[097 KServe：模型生命周期与 LLMInferenceService|KServe]](标准化控制面 / CRD)。它们共享同一底座——都把 [[LLM/108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|vLLM]] 当数据面引擎、跑在 K8s 上、用 Gateway API Inference Extension 做入口——差别在「控制面怎么编排」。本篇是 [[100 推理网关与智能路由(cache-aware)|网关]]、[[038 KV-aware 路由与跨引擎复用|KV-aware 路由]]、[[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]] 等点的收口。

## ① 类比:同一台发动机,不同整车厂

vLLM 像一台高性能发动机(数据面引擎)。AIBrix / llm-d / KServe 是三家不同的整车厂,都用这台发动机,但底盘、变速箱、车机系统(控制面)各有侧重:有的主打「自带全套配置一键上路」(AIBrix),有的主打「最聪明的智能调度大脑」(llm-d),有的主打「标准化的整车平台、随便换发动机」(KServe)。你买哪辆,取决于你要的是开箱即用、极致调度、还是标准可移植。

## ② 小数字例子:分布式 KV 的收益与 LoRA 密度

**分布式 KV 的收益。** AIBrix 把 KV-Cache 做成跨节点共享池(InfiniStore,RDMA),让前缀/KV 在副本间复用:官方数据**吞吐 +50%、推理延迟 -70%**(长上下文、prefill-heavy 场景尤甚)。这正是 [[037 Mooncake：KVCache 中心的存储池|Mooncake]]、[[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|LMCache]] 那条「KV 当一等存储」路线的产品化。

**高密度 LoRA。** 传统每个微调版本占一个完整副本;AIBrix 的高密度 LoRA 管理让**一个 base 模型上挂几十个 LoRA adapter**,按请求动态切换,几十个「模型」共享一份 base 权重的显存 → 多租户微调服务成本大降。

**注意对比维度。** 第三方基准里 vLLM Production Stack 在 prefill-heavy 工作负载上比 AIBrix 更快(AIBrix 早期 PyTorch 版 KV offload 在高 QPS 下 TTFT 抬升);说明「分布式 KV」收益强依赖实现成熟度,选型要看版本与场景。

![[orch-104分布式KV收益.svg]]

## ③ 原理:三栈各管什么、怎么组合

**1. AIBrix(字节,2025-02 在 vLLM 项目下开源)。** 定位:**自带电池的 vLLM 控制面**,面向大规模、成本优先。一套里打包:
- **分布式 KV-Cache**(InfiniStore,RDMA 多级缓存):跨节点 KV 复用,提吞吐降延迟。
- **高密度 LoRA 管理**:一 base + 多 adapter 动态切换。
- **Ray + K8s 混合编排**:Ray 管细粒度应用编排,K8s 管粗粒度资源。
- **cache-aware / 公平路由、异构 GPU 支持**。
字节内部多业务已用,适合「想要一站式、规模大」的团队。坑:早期版本 KV offload 实现在高 QPS 下有 TTFT 抬升。

**2. llm-d。** 定位:**K8s 原生分布式推理的"调度智能"层**。如果说控制面管生命周期,llm-d 管**集群级优化**:前缀缓存感知路由、[[048 为何分离 prefill 与 decode|PD 分离]] 调度、KV-aware 调度,经 Envoy AI Gateway + Gateway API Inference Extension 做入口。它用 vLLM 内建 KV(不另起 InfiniStore 那样的独立 KV server),Red Hat / Google 等推动。

**3. KServe。** 定位:**标准化控制面 / CRD**。提供 `LLMInferenceService`(2025 GenAI-first CRD)统一模型生命周期、版本、[[103 多副本、蓝绿与金丝雀发布|canary/蓝绿切流]]、[[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|scale-to-zero(Knative)]]、多框架后端。2025-11 成为 CNCF 孵化项目,治理稳定。

**4. 怎么组合(关键认知)。** 这三者不全是竞品——**KServe + llm-d 是经典组合**:KServe 当控制面(模型生命周期、API 抽象、发布),llm-d 当底层分布式智能调度层。AIBrix 则更像「自成一体的替代栈」。再加上 [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|NVIDIA Dynamo]](带 SLO Planner 的分布式推理框架),构成 2025-2026 的主流候选。选型轴:① 要不要分布式 KV(长上下文/prefill-heavy → 是);② 要不要标准 CRD 可移植(KServe);③ 要不要开箱即用一站式(AIBrix);④ 多租户 LoRA 密度需求(AIBrix 强)。

![[orch-生产栈对比.svg]]

![[orch-104栈组合选型.svg]]

## ④ 配置:AIBrix 部署形态 vs KServe CRD

```yaml
# AIBrix:声明一个带分布式 KV + 自动扩缩的 vLLM 部署(自带电池)
apiVersion: orchestration.aibrix.ai/v1alpha1
kind: RayClusterFleet            # Ray+K8s 混合编排的分布式 vLLM
metadata: { name: llama3-70b }
spec:
  replicas: 4
  # AIBrix 控制面附带:分布式 KV(InfiniStore)、cache-aware 路由、LoRA 管理
  template:
    spec:
      engine: vllm
      model: meta-llama/Llama-3.1-70B-Instruct
```

```yaml
# KServe:标准 CRD 抽象,底层可挂 llm-d 做分布式调度
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata: { name: llama3 }
spec:
  model: { uri: hf://meta-llama/Llama-3.1-8B-Instruct }
  # KServe 提供生命周期/版本/canary/scale-to-zero;分布式优化交给 llm-d
```

❌ 反模式:① 不分场景盲选"最快的栈"——prefill-heavy 长上下文和短 chat 的最优栈不同,分布式 KV 的收益强依赖实现成熟度与负载。② 把 AIBrix / llm-d / KServe 当三选一互斥——其实 KServe+llm-d 常组合,各管控制面与调度层。③ 多 LoRA 多租户却用「一版一副本」——显存爆炸,该用 AIBrix 式高密度 LoRA。

✅ 正解:先定**底座一致(vLLM + K8s + GIE)**,再按需求选控制面:要标准可移植 + 发布治理→KServe(+llm-d 做分布式调度);要一站式 + 强分布式 KV/LoRA 密度 + 大规模→AIBrix;要带 SLO Planner 的解耦推理→Dynamo。所有选型都按真实负载(上下文长度、prefill/decode 比、多租户)benchmark 后定。

## 面试高频

- **「AIBrix、llm-d、KServe 各是什么定位?」** AIBrix=字节的自带电池 vLLM 控制面(分布式 KV+LoRA+Ray/K8s);llm-d=K8s 原生分布式智能调度层(前缀感知路由、PD 分离);KServe=标准化控制面/CRD(生命周期、发布、scale-to-zero)。三者共享 vLLM + K8s + GIE 底座。
- **「它们是竞品吗?」** 不全是。KServe+llm-d 是经典组合(控制面 + 分布式调度);AIBrix 更像自成一体的替代栈;还有 Dynamo 这类带 SLO Planner 的框架。
- **「分布式 KV 带来什么、有什么代价?」** 跨节点复用 KV/前缀,长上下文/prefill-heavy 下提吞吐降延迟(AIBrix 称 +50%/-70%);代价是实现复杂度,早期版本高 QPS 下可能 TTFT 抬升,收益强依赖成熟度。
- **「高密度 LoRA 解决什么?」** 一 base + 几十 adapter 共享 base 显存、按请求切换,多租户微调服务成本大降,避免「一版一副本」。
- **「生产栈选型看哪些轴?」** 上下文长度/prefill-decode 比(决定是否要分布式 KV)、可移植性(标准 CRD)、开箱即用程度、多租户 LoRA 密度、是否需 SLO 驱动的 PD 解耦——按真实负载 benchmark 定。

## 关键事实

- **AIBrix**(字节):2025-02 在 vLLM 项目下开源;分布式 KV(InfiniStore RDMA)、高密度 LoRA、Ray+K8s 混合编排、cache-aware/公平路由、异构 GPU;官方称分布式 KV +50% 吞吐 / -70% 延迟。
- **llm-d**:K8s 原生分布式推理框架,前缀缓存感知路由 + PD 分离 + KV-aware 调度,经 Envoy AI Gateway + GIE 入口,用 vLLM 内建 KV;Red Hat/Google 等推动。
- **KServe**:`LLMInferenceService`(2025 GenAI-first CRD),生命周期/版本/canary/蓝绿/scale-to-zero;2025-11 CNCF 孵化项目;常与 llm-d 组合。
- 三栈**共享底座**:vLLM 引擎 + Kubernetes + Gateway API Inference Extension;差别在控制面编排。
- 第三方基准:prefill-heavy 下 vLLM Production Stack 曾快于早期 AIBrix → 选型须按真实负载实测,勿盲信单一数字。
