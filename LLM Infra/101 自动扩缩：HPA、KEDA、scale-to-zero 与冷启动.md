[[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动]]:流量起落,副本数也该跟着动。但 LLM 的自动扩缩有两个反直觉点——**该看的指标不是 CPU,而是队列/KV/GPU**;**缩到 0 省钱的代价是冷启动慢到分钟级**(几十到上百 GB 权重要搬进显存)。这一篇接着 [[100 推理网关与智能路由(cache-aware)|推理网关]] 之后讲:网关决定「请求往哪分」,扩缩决定「有几个副本可分」。它是 [[060 数据并行与副本扩展|副本扩展]] 的运维落地,也直接关系 [[113 容量规划：从 QPS 反推 GPU 数|容量规划]] 给出的副本数是静态下限,自动扩缩在其上随流量浮动。

## ① 类比:餐厅按排队加灶,而非按服务员忙闲

普通 web 服务的 HPA 像「看服务员忙不忙就加人」——CPU 高就扩。但 LLM 的瓶颈在 GPU(显存/带宽),CPU 常年很闲,看 CPU 永远不会扩,等你发现延迟爆了已经晚了。正确做法是看**门口排队的人数**(等待队列长度)和**厨房灶台占用**(KV-Cache 利用率)。scale-to-zero 则像「打烊就熄灶省燃气」,省钱;但客人深夜来,得等灶台重新烧热——这就是冷启动,LLM 的「烧热」= 把几十 GB 权重搬进显存,要几分钟。

## ② 小数字例子:阈值与冷启动秒数

**扩缩阈值。** 单副本 8B 健康水位:等待队列 ≤ 5、KV 利用率 ≤ 80%。设目标「平均队列长度 = 5」,当前 3 副本实测平均队列 15:
- HPA 算 desired = ceil(15 ÷ 5 × 3) = **9 副本**,自动扩到 9。
- 队列降回 5 以下且持平稳定窗口后,逐步缩回。

**那个「15」从哪来 + 防抖。** HPA 喂的是**跨副本聚合后的平均**:Prometheus 把 3 个副本各自的 `num_requests_waiting` 取均值,$\frac{q_1+q_2+q_3}{3}=15$(等价于全场积压 $45$ 条排队请求 $\div\,3$ 副本)。代入 $\text{desired}=\lceil\frac{15}{5}\times 3\rceil=\lceil\frac{45}{5}\rceil=9$——直觉就是「45 条积压 ÷ 每副本目标 5 条 = 需 9 副本」。防抖靠两个旋钮:HPA 默认带 ±10% 的 **tolerance**(指标在目标值 ±10% 内不动),缩容侧再叠 `cooldownPeriod`(如 300s)与 stabilization window,避免队列一抖就反复扩缩拉起冷启动。

**冷启动秒数(scale-to-zero 后第一个请求)。** 70B、130 GB 权重:
- 从对象存储下载 ≈ **26 s 起**;加载到 8×GPU ≈ **84 s**;加上调度、拉镜像、引擎初始化,**总冷启动常 2–3 分钟**。
- 对比普通 web pod 冷启动 ~秒级。差距全在「搬权重」。

## ③ 原理:三层扩缩 + 冷启动的本质

**1. HPA(Horizontal Pod Autoscaler)= 基础闭环。** `desired = ceil(当前指标 ÷ 目标值 × 当前副本数)`。原生只认 CPU/内存,对 LLM 几乎无用——必须喂**自定义指标**。

**2. KEDA = 自定义指标 + scale-to-zero。** KEDA 是事件驱动扩缩器,内置几十种 scaler(Prometheus、Kafka、SQS…)。它把「队列深度 / KV 利用率 / QPS」这类引擎指标转成 HPA 能消费的外部指标,并且**能缩到 0**(原生 HPA 最小 1)。对 LLM 这是标配:闲时 0 副本不烧 GPU,来流量再从 0 拉起。

**3. Knative / KServe Serverless = 请求级 scale-to-zero + 激活器。** Knative 用 activator 在 0 副本时先 hold 住请求、触发扩容、再放行,提供更顺的 scale-to-zero。KServe 的 Serverless 模式即基于此(见 [[097 KServe：模型生命周期与 LLMInferenceService|KServe]])。

**4. 为什么 LLM 冷启动是分钟级。** 冷启动是复合过程:调度起 pod → 拉镜像/容器启动 → **下载权重 → 加载进显存** → 引擎 warmup。后两步是大头,因为权重是几十~上百 GB,是「内存搬运」而非「跑代码」。缓解:本地 SSD 缓存权重、预拉镜像、**权重流式加载**(如 NVIDIA Run:ai Model Streamer 并发读 + 直流进显存)、保留 1 个热副本(min=1 而非 0)、或预热(prewarm)在低谷期先拉起。这也是为什么很多生产系统宁可留个最小热副本,只对长尾闲置模型用真 scale-to-zero。

![[orch-autoscale指标链.png]]

![[orch-冷启动时间线.png]]

## ④ 配置:KEDA 按队列扩缩 + 缩到零

```yaml
# KEDA:按 vLLM 等待队列长度扩缩,闲时缩到 0
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-llama3-scaler
spec:
  scaleTargetRef:
    name: vllm-llama3            # 目标 Deployment
  minReplicaCount: 0            # 关键:缩到 0 省 GPU(代价=冷启动)
  maxReplicaCount: 12
  cooldownPeriod: 300          # 缩容前的静默期,避免抖动
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: avg(vllm:num_requests_waiting)   # 平均等待队列长度
        threshold: "5"          # 每副本目标队列=5,超了就扩
```

```yaml
# 对照:原生 HPA 用 CPU —— LLM 上几乎是错的
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
  # ❌ LLM 瓶颈在 GPU,CPU 常年低 → 永不扩,直到延迟爆掉
```

❌ 反模式:① 用 CPU/内存利用率给 LLM 扩缩——CPU 永远不高,队列爆了也不扩。② 对所有模型一律 scale-to-zero——长尾模型还好,主力模型第一个请求吃 2–3 min 冷启动,SLO 直接违约。

✅ 正解:用 **KEDA + 引擎指标(队列/KV/SM 利用率)** 做扩缩信号;主力模型留 `min=1` 热副本或预热,真正 scale-to-zero 只用于低频长尾;配权重流式加载 + 本地缓存压缩冷启动;缩容设 cooldown 防抖动。

## 面试高频

- **「LLM 自动扩缩该看什么指标,为什么不是 CPU?」** 看等待队列长度、KV-Cache 利用率、运行请求数、GPU SM 利用率。LLM 瓶颈在 GPU 显存/带宽,CPU 常年闲,用 CPU 扩缩会「队列爆了还不扩」。
- **「HPA、KEDA、Knative 各解决什么?」** HPA 是基础闭环但只认 CPU/内存;KEDA 提供自定义指标 + scale-to-zero(原生 HPA 最小 1);Knative/KServe Serverless 用 activator 做请求级 scale-to-zero。
- **「scale-to-zero 省什么、代价是什么?」** 省闲时 GPU 成本(GPU 按小时计费很贵);代价是第一个请求吃几分钟冷启动。权衡:留最小热副本 vs 真缩零。
- **「LLM 冷启动慢在哪、怎么缓解?」** 慢在下载权重 + 加载进显存(几十~上百 GB 搬运),非跑代码。缓解:本地 SSD 缓存、预拉镜像、权重流式加载(Run:ai Model Streamer)、预热、min=1 热副本。
- **「扩缩和容量规划什么关系?」** 容量规划给静态副本下限(峰值 QPS 反推),自动扩缩在其上随实时流量浮动,二者互补。

## 关键事实

- **HPA 公式**:`desired = ceil(当前指标 ÷ 目标值 × 当前副本数)`;原生只认 CPU/内存,LLM 须接自定义指标。
- **KEDA**:CNCF 毕业项目,事件驱动扩缩,内置 Prometheus 等 scaler,支持 `minReplicaCount: 0`(scale-to-zero),把引擎指标喂给 HPA。
- **冷启动 70B 实测**:下载 130GB 权重 ≈ 26s+,加载 8×GPU ≈ 84s,总冷启动常 2–3 min;普通 web pod 仅秒级。
- **NVIDIA Run:ai Model Streamer**(2025):开源,并发读权重直流进 GPU 显存,显著降冷启动。
- 生产常见折中:主力模型 `min=1` 热副本,长尾模型才真 scale-to-zero;缩容设 cooldown(如 300s)防抖动。
