> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[130 K8s 上 KServe 部署 + 自动扩缩 + 监控面板]]:把全栈拼齐——用 [[097 KServe：模型生命周期与 LLMInferenceService|KServe]] 的 `LLMInferenceService` 声明式部署 vLLM,配 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|HPA/KEDA]] 自动扩缩与 scale-to-zero,接 Prometheus + Grafana 做 [[108 监控：GPU 利用率、KV 占用、排队、p99|监控]] 面板。这是 123 路线图的终点站。

## ① 类比:从「手开一家店」到「连锁总部」

前面 124–129 都是手动起进程。**K8s 是连锁总部**:你只交一张「开店申请表」(`LLMInferenceService` YAML)写清要哪个模型、几张卡、几个副本,总部自动开店、挂招牌(路由)、生意火了自动加店([[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|HPA]])、半夜没人就关店省钱(scale-to-zero)、墙上装监控大屏(Grafana)盯每家店的 [[108 监控：GPU 利用率、KV 占用、排队、p99|GPU/KV/p99]]。你不再 ssh 进机器手动 `vllm serve`。

## ② KServe LLMInferenceService:声明式部署

```yaml
# llmisvc.yaml — 单卡 8B 最小例
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: llama3-8b
  namespace: default
spec:
  model:
    uri: hf://meta-llama/Llama-3.1-8B-Instruct
    name: meta-llama/Llama-3.1-8B-Instruct
  replicas: 1
  template:
    containers:
      - name: main
        image: vllm/vllm-openai:latest
        args: ["--model", "/mnt/models", "--enable-prefix-caching"]
        resources:
          limits: { nvidia.com/gpu: "1" }
  router:                # 自动建 Gateway + Route + 调度
    gateway: {}
    route: {}
    scheduler: {}
```

```bash
kubectl apply -f llmisvc.yaml
kubectl get llminferenceservice llama3-8b      # READY=True 即可用
```

多卡 70B:加 `spec.parallelism: {tensor: 4}` + `worker:` 段 + GPU limit 改 4(见 [[125 多卡 TP 部署 70B 模型|多卡 TP]])。

## ③ 原理:三层拼图

**1. KServe 声明式编排。** `LLMInferenceService` 把「模型 + 运行时 + 副本 + 路由 + 扩缩」收进一个 CR;controller 调谐成底层 Deployment/Service/Gateway。它与 [[098 llm-d：K8s 原生分布式推理|llm-d]] 集成,可声明 PD 分离与多节点(见 [[097 KServe：模型生命周期与 LLMInferenceService|KServe 生命周期]])。

**2. 自动扩缩。** [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|HPA]] 按指标加减副本。LLM 不能只看 CPU——要按**自定义指标**(队列长度、KV 占用、p99)扩,用 **KEDA** 拉 Prometheus 指标触发。scale-to-zero 省钱但有**冷启动**(拉权重 + 起引擎几十秒~分钟),需预热/保活权衡。

**3. 监控。** vLLM 在 `/metrics` 暴露 Prometheus 指标:`vllm:num_requests_running/waiting`(排队)、`vllm:gpu_cache_usage_perc`([[031 KV 显存碎片与 block 管理|KV]] 占用)、`vllm:time_to_first_token_seconds`(TTFT 直方图)、`vllm:time_per_output_token_seconds`。Grafana 画 p99 TTFT/TPOT、KV 占用、排队、[[002 GPU 架构：SM、CUDA Core 与 Tensor Core|GPU]] 利用率(DCGM exporter)。

![[lab-K8s全栈部署.svg]]

把三层拼图叠起来看——KServe 声明式编排 → KEDA 按业务指标扩缩(含 scale-to-zero)→ Prometheus+Grafana 监控闭环喂回扩缩:

![[lab-130三层拼图.svg]]

## ④ 自动扩缩:KEDA 按 KV 占用扩

```yaml
# scaledobject.yaml — 按 vLLM 排队/KV 指标扩缩,支持 scale-to-zero
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llama3-8b-scaler
spec:
  scaleTargetRef:
    name: llama3-8b-predictor          # 目标 Deployment
  minReplicaCount: 0                    # scale-to-zero
  maxReplicaCount: 8
  cooldownPeriod: 300                   # 缩容前等待,缓冲冷启动抖动
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: avg(vllm:num_requests_waiting)   # 排队长度
        threshold: "5"                          # 平均排队>5 就加副本
```

## ⑤ 监控面板:Prometheus + Grafana

```yaml
# servicemonitor.yaml — 让 Prometheus 抓 vLLM /metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata: { name: vllm }
spec:
  selector: { matchLabels: { app: llama3-8b } }
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

```promql
# Grafana 关键面板查询
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))  # p99 TTFT
histogram_quantile(0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le)) # p99 TPOT
avg(vllm:gpu_cache_usage_perc)            # KV 占用率(接近 1 = 显存吃紧/将抢占)
sum(vllm:num_requests_waiting)            # 排队深度(扩缩信号)
DCGM_FI_DEV_GPU_UTIL                      # GPU 利用率(DCGM exporter)
```

❌ 反模式:HPA 只按 CPU/GPU 利用率扩——LLM 在 decode 阶段 GPU 利用率可能不高但已排队;或 scale-to-zero 不评估冷启动就上,首请求超时。
✅ 正解:**按 `num_requests_waiting`/KV 占用/p99 等业务指标扩**(KEDA + Prometheus);scale-to-zero 配预热或保活权衡冷启动;面板必含 p99 TTFT/TPOT、KV 占用、排队、GPU 利用率四件套。

## 面试高频

- **「为什么 LLM 自动扩缩不能只看 CPU/GPU 利用率?」** decode 访存受限时 GPU 利用率不高但可能已排队;要按排队长度、KV 占用、p99 等业务指标扩,用 KEDA 拉 Prometheus 指标。
- **「scale-to-zero 的代价?」** 冷启动:拉权重 + 起引擎几十秒~分钟,首请求慢;需预热/保活/快照权衡。
- **「LLM 服务监控盯哪几个指标?」** p99 TTFT/TPOT、KV 占用(`gpu_cache_usage_perc`)、排队深度(`num_requests_waiting`)、GPU 利用率(DCGM)。
- **「KServe LLMInferenceService 解决什么?」** 把模型+运行时+副本+路由+扩缩收进一个声明式 CR,controller 调谐;可集成 llm-d 做 PD 分离/多节点。
- **「vLLM 指标从哪来?」** vLLM 自带 `/metrics`(Prometheus 格式),ServiceMonitor 抓取,Grafana 画图。

## 关键事实

- KServe:`apiVersion: serving.kserve.io/v1alpha1`,`kind: LLMInferenceService`;`model.uri: hf://...`、`replicas`、`router: {gateway/route/scheduler}`;多卡加 `parallelism.tensor` + `worker`。
- 镜像 `vllm/vllm-openai:latest`;`kubectl get llminferenceservice` 看 READY。
- 自动扩缩:KEDA `ScaledObject` 按 `vllm:num_requests_waiting` 等 Prometheus 指标,`minReplicaCount: 0` 即 scale-to-zero(有冷启动)。
- 监控:vLLM `/metrics` → ServiceMonitor → Prometheus → Grafana;核心指标 `vllm:time_to_first_token_seconds`、`vllm:gpu_cache_usage_perc`、`vllm:num_requests_waiting`、DCGM GPU util。
- 别用 CPU/GPU 利用率单独驱动 LLM 扩缩。
