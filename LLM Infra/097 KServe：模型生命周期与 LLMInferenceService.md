[[097 KServe：模型生命周期与 LLMInferenceService]]讲的是:KServe 这个 K8s 上的标准化推理平台,如何用一个声明式 CRD(InferenceService / 新的 LLMInferenceService)托管模型的完整生命周期——部署、Knative 自动扩缩与 scale-to-zero、控制面/数据面分工、金丝雀灰度与版本治理。它把 [[096 Kubernetes 上部署 LLM：基本盘|K8s 部署基本盘]]里那一堆手搓的 Deployment/Service/探针/HPA 收进一个对象,是 [[095 从单进程到集群：服务架构演进|架构演进]]第 ④ 阶段的主流控制面之一,且能向下对接 [[098 llm-d：K8s 原生分布式推理|llm-d]]。

## 类比
KServe 像云厨房的"中央管理系统",你不再一个个雇厨师、排班、装监控。你只递一张菜单卡(CRD):"我要这个模型、这个引擎、忙时扩到 8 口锅、没人点单就关火"。系统(Controller)自己去开 Deployment、装探针、配 HPA、接路由,并随时把现状收敛到你声明的目标——这就是**控制面**。真正炒菜的档口是**数据面**:核心是 Predictor(掌勺),前面可加 Transformer(切配/摆盘),旁边可挂 Explainer(写菜品说明)。Knative 则是"没人来就熄火、来人再点火"的省电开关(scale-to-zero),代价是第一位客人要等灶台重新烧热(冷启动)。

## 小数字例子
内部工具,白天 QPS 20、夜里几乎为 0,跑 Llama-3-8B:
- 常驻 2 副本(各 1×H100):24h 占 2 卡,夜里全浪费。
- KServe + Knative scale-to-zero:夜里缩到 0 副本,GPU 成本接近 0;早上首个请求触发冷启动,拉权重+载入 VRAM ~ 1–3 分钟,该请求 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT]] 被拉到分钟级,之后按并发自动扩到 N。
- 折中:`minReplicas: 1` 常驻一副本兜底首请求延迟,其余按并发弹性扩——省钱与冷启动之间调旋钮。

## 原理:控制面 / 数据面 + 生命周期
**控制面**:KServe Controller 监听 `InferenceService`/`LLMInferenceService` 这类 CRD,reconcile 出底层对象——Deployment+Service、HPA/KEDA、(Serverless 模式下)Knative 路由与 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|扩缩]]。你声明期望状态,它负责收敛。

**数据面**:执行请求。经典 InferenceService 由三部分组成——**Predictor**(核心,托管模型,后端是 ServingRuntime,如 vLLM)、**Transformer**(自定义前/后处理)、**Explainer**(可选,产出可解释性)。

**两种部署模式**:
- *Serverless(Knative)*:按并发/请求自动扩缩,支持 scale-to-zero——省钱但有冷启动。
- *Raw/Deployment*:用原生 HPA/KEDA,不缩到 0,延迟更稳。

![[orch-097两种部署模式.png]]

**LLMInferenceService(新 CRD,专为生成式)**:在 Predictor 之上原生支持 [[057 张量并行推理：延迟换显存|TP]]/[[059 流水线并行推理：micro-batch 与气泡|PP]]/多 GPU 分片,单 CRD 部署 70B+,并对接 llm-d 拿到 [[048 为何分离 prefill 与 decode|PD 解耦]]、前缀缓存、[[100 推理网关与智能路由(cache-aware)|智能调度]]、variant 自动扩缩。

**模型治理**:金丝雀按流量百分比灰度新版本、一键回滚、多模型/多 runtime 统一 CRD 管理。配套实操见 [[130 K8s 上 KServe 部署 + 自动扩缩 + 监控面板|KServe 实操]]。

![[orch-KServe控制面.png]]

## 代码:LLMInferenceService(声明即生命周期)
```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: llama3-8b
spec:
  model:
    uri: hf://meta-llama/Llama-3-8B           # 模型来源
  replicas:
    min: 1                                     # ← 关键旋钮:0=可缩到零(省钱,有冷启动)
    max: 8                                     #    1=常驻兜底首请求延迟
  parallelism:
    tensor: 1                                  # 大模型改 2/4/8 走 TP/PP
  router:
    scheduler:                                 # 对接 llm-d:cache-aware + PD 解耦
      cacheAware: true
  autoscaling:
    metric: concurrency                        # Knative 按并发扩缩
    target: 10
```
```yaml
# 经典 InferenceService(三件套语义更直观),金丝雀灰度示例
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata: { name: my-llm }
spec:
  predictor:
    minReplicas: 1
    model:
      runtime: vllm-runtime
      storageUri: pvc://models/llama3-8b
  # 滚动新版本时,只把 10% 流量切给金丝雀,稳了再放大
  # canaryTrafficPercent: 10
```
```yaml
# ❌ 反例:scale-to-zero 但又有强 SLA 的对外接口
# replicas: { min: 0 }   # 夜间缩零省钱,但每次冷启动首请求分钟级 TTFT
#                        # 对要求 p99<1s 的外部 API 是灾难 → 该用 min:1 或预热
```

## 面试高频
- **KServe 控制面 vs 数据面?** 控制面 = Controller 监听 CRD、reconcile 出 Deployment/HPA/路由(声明期望);数据面 = Predictor(+Transformer/Explainer)真正执行推理。
- **InferenceService 三组件?** Predictor(核心托管模型)、Transformer(前后处理)、Explainer(可选解释)。
- **scale-to-zero 的好处与代价?** 空闲缩到 0 副本省 GPU 成本;代价是首请求吃分钟级冷启动(拉权重+载入 VRAM+编译);缓解:min:1 常驻、权重预拉、激活器缓冲。
- **Serverless 模式靠什么扩缩?为何不缩零用 HPA?** Knative 按并发/请求扩缩可到 0;Raw/Deployment 用 HPA/KEDA 不缩零、延迟更稳。
- **LLMInferenceService 比经典 InferenceService 多了什么?** 原生 TP/PP/多 GPU 分片、单 CRD 部 70B+、对接 llm-d 拿 PD 解耦/前缀缓存/智能路由/variant 扩缩。
- **怎么安全上新模型版本?** 金丝雀按流量百分比灰度 + 一键回滚。

## 关键事实
- KServe 2025-11 成为 CNCF incubating 项目;LLMInferenceService 是面向生成式推理的新 CRD,支持与 llm-d 集成实现 PD 解耦、前缀缓存、智能调度、variant 自动扩缩。
- Serverless 模式基于 Knative,提供并发/请求驱动的自动扩缩与 scale-to-zero;Raw/Deployment 模式用原生 HPA/KEDA,不缩到 0。
- 经典 InferenceService 三组件(Predictor/Transformer/Explainer)中,Predictor 为核心,后端由 ServingRuntime(如 vLLM)实现;ModelMesh 模式下 Transformer/Explainer/金丝雀部分不适用。
- scale-to-zero 省成本但引入冷启动延迟,这是面试与生产里最常被追问的 trade-off。
