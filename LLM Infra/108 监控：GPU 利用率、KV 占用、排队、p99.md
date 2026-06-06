[[108 监控：GPU 利用率、KV 占用、排队、p99|监控指标]]是把 [[105 SLO、SLA 设计：为推理定指标|SLO]]从「目标」变成「实时可观测」的眼睛。标准链路是 **DCGM exporter(GPU 硬件层) + 引擎 /metrics(vLLM 等业务层) → Prometheus 抓取 → Grafana 看板 + 告警**。四类核心信号:**GPU 利用率**、**KV cache 占用**、**排队深度**、**p99 延迟**。最大的反直觉陷阱在第一类:**`GPU_UTIL=99%` 不等于真的算满**——它只表示「有 kernel 在跑」,而 [[014 Decode 阶段：访存受限|decode 访存受限]]时 SM 大量空转(等 HBM 数据)也照样显示 99%。要判断是否真用满,得看 **SM 活跃度(DCGM_FI_PROF_SM_ACTIVE)、Tensor Core 活跃度、HBM 带宽利用率**。其余三类各有用途:**KV 占用是吞吐崩塌的先行指标**(见 [[031 KV 显存碎片与 block 管理|KV 显存]]),**排队深度是过载早警**(对应 [[047 准入控制与排队论：队列长度到延迟|排队论]]),**p99 是 SLO 守门人**(永远看百分位不看平均)。NVIDIA **DCGM** 是 GPU 监控事实标准。这与 [[110 成本与 TCO：每百万 token 成本怎么算|成本 TCO]]互为表里——用满才省钱。

## 直觉

把监控想成**给推理服务装四块仪表盘**。

- **GPU 利用率(陷阱表)**:像汽车转速表显示「引擎在转」,但**转速高不代表在做功**——空挡轰油门也高转速。decode 阶段 GPU 在「等内存搬数据」,util 照样 99%,实际算力闲置。别被这块表骗了,要去看「Tensor Core 真在算吗、HBM 带宽喂满了吗」。
- **KV cache 占用(油量表)**:KV 显存就是「服务并发的油箱」。70% 还有余量;90% 快见底,新请求挤不进去,马上要触发抢占/swap,吞吐**先于 p99 告警**就开始掉——它是先行指标。
- **排队深度(候车人数)**:waiting 队列在涨说明「来的比处理的快」,是过载的最早信号,比 p99 破线更早。
- **p99 延迟(到站准点率)**:直接对应用户体验和 SLO/错误预算。看平均会被尾部骗,必须看 p99。

## 例子

某 vLLM 实例稳态采样:

| 指标(Prometheus) | 读数 | 含义 / 动作 |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | 99% | **别信这个**——可能在空转等访存 |
| `DCGM_FI_PROF_SM_ACTIVE` | 0.42 | SM 真实活跃仅 42% → 算力没用满 |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | 0.18 | Tensor Core 多在闲 → 访存受限典型 |
| `vllm:gpu_cache_usage_perc` | 0.88 | KV 88% → 接近饱和,扩 batch 困难 |
| `vllm:num_requests_running` | 32 | 在跑 32 个 |
| `vllm:num_requests_waiting` | 47 | **排队 47 > 在跑** → 吸不下流量,过载早警 |
| `vllm:e2e_request_latency` p99 | 2800ms | p99 破 SLO(目标 800ms) |
| avg 同指标 | 240ms | **平均看着没事** → 印证只看 avg 会漏判 |

- **诊断链**:GPU_UTIL 99% 看似满载,但 SM_ACTIVE=0.42 揭示其实是 decode 访存受限;同时 waiting(47)>running(32),KV=88%,p99=2800ms——这是**典型过载**:队列堆积 + KV 见底,加卡/扩容比优化 kernel 更有效。
- **先行性验证**:KV 从 70%→88% 和 waiting 从 0→47,**早于** p99 从 240ms 爬到 2800ms 发生——它们是 p99 爆炸的前兆,可用于提前扩容/限流。

## 原理

**指标三大类型**(Prometheus):
- **Gauge**(瞬时值):`gpu_cache_usage_perc`、`num_requests_waiting`——可上可下。
- **Counter**(累计):`request_success_total`——只增,算速率用 `rate()`。
- **Histogram**(直方图):`time_to_first_token_seconds_bucket`——按桶累计,**分位由它估**:

$$
p99 = \texttt{histogram\_quantile}\Big(0.99,\ \sum_{le}\texttt{rate}(\text{bucket}[5m])\Big)
$$

**为什么 GPU_UTIL 会骗人**:NVML/DCGM 的 `GPU_UTIL` 定义是「采样窗口内至少有一个 kernel 在执行的时间占比」。decode 是 [[014 Decode 阶段：访存受限|访存受限]]的,kernel 一直在跑(故 util≈100%)但 SM 大量周期在 stall 等 HBM。真实「做功」要看:

$$
\text{真实算力利用} \approx \texttt{SM\_ACTIVE} \times \texttt{(FP/TENSOR\_ACTIVE)} \quad\ll \texttt{GPU\_UTIL}
$$

decode 阶段典型 `GPU_UTIL≈99%` 而 `SM_ACTIVE≈0.4`、`TENSOR_ACTIVE≈0.2`——算术强度低、被带宽卡住(见 [[004 算力 vs 带宽：Roofline 与算术强度|Roofline]])。

**排队作为过载先行指标**:由 [[047 准入控制与排队论：队列长度到延迟|排队论]],队列长度 $L$ 与等待 $W_q$ 通过 $W_q \propto \rho/(1-\rho)$ 关联;`num_requests_waiting` 上升早于 $W_q$ 进入 p99——监控它能在 SLO 破之前动手。

**告警分层**:页面告警挂在 **SLO 燃烧率**(p99/错误预算)上;容量告警挂在**先行指标**(KV 占用、waiting 队列)上,提前于用户受影响。

## 图

![[obs-监控面板.svg]]

![[obs-108假满载与先行指标.svg]]

## 代码

vLLM 自带 Prometheus 指标 + 关键 PromQL 与告警:

```python
# 启动即暴露 /metrics(Prometheus 格式),无需额外埋点
# vllm serve <model> --port 8000   →   GET http://host:8000/metrics

# ❌ 反模式:只盯 GPU_UTIL 和 avg 延迟下结论
def health_BAD(gpu_util, avg_latency_ms):
    return gpu_util > 90 and avg_latency_ms < 500
    # GPU_UTIL=99% 可能在空转;avg<500ms 可能 p99=3000ms → 双重误判

# ✅ 看真实算力 + 先行指标 + 百分位
def health(m):
    overloaded = m["waiting"] > m["running"] or m["kv_usage"] > 0.9
    underused  = m["sm_active"] < 0.5 and m["gpu_util"] > 0.9  # 假满载(访存受限)
    slo_break  = m["ttft_p99_ms"] > 800
    return {"overloaded": overloaded, "underused": underused, "slo_break": slo_break}
```

```yaml
# Prometheus 告警:容量(先行)+ SLO(滞后)分层
groups:
- name: llm-infra
  rules:
  - alert: KVCacheNearFull          # ✅ 先行:吞吐崩前预警
    expr: vllm:gpu_cache_usage_perc > 0.9
    for: 2m
  - alert: QueueBacklog             # ✅ 先行:过载早警
    expr: vllm:num_requests_waiting > vllm:num_requests_running
    for: 3m
  - alert: TTFTp99SLOBreach         # 守门:SLO 已破,页面告警
    expr: histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) > 0.8
    for: 5m
```

`❌` 只看 GPU_UTIL 会把访存受限的「假满载」当真满载、错失优化机会,只看 avg 会漏掉破 SLO 的尾部;`✅` 用 SM/Tensor 活跃度判真实算力、用 KV 占用 + 排队深度做先行告警、用 histogram 分位守 SLO,三层联动才能既不浪费算力又不破合约。

## 面试高频

- **Q:GPU_UTIL=99% 是不是就算满载了?** 不是。它只表示有 kernel 在跑;decode 访存受限时 SM 大量空转等 HBM 也显示 99%。要看 SM_ACTIVE、Tensor Core 活跃度、HBM 带宽才知是否真用满。
- **Q:监控推理服务最该看哪几个指标?** GPU 真实算力(SM/Tensor 活跃)、KV cache 占用、排队深度(waiting/running/swapped)、TTFT/ITL 的 p99。前两类是先行指标,p99 是 SLO 守门。
- **Q:为什么 KV 占用是先行指标?** KV 接近饱和(>90%)时无法再扩 batch、触发抢占/swap,吞吐先于 p99 告警就开始跌;监控它能在用户受影响前扩容。
- **Q:排队深度怎么用于告警?** waiting 增速超过 running、或 waiting 持续 >0 上升,说明吸不下流量(ρ→1 前兆),比 p99 破线更早,可触发扩容/限流。
- **Q:p99 怎么从 Prometheus 算?** 用 histogram 类型指标的桶 + `histogram_quantile(0.99, rate(..._bucket[5m]))`;不能用平均值,延迟右偏重尾会掩盖尾部。
- **Q:监控链路标准组件?** DCGM exporter(GPU 硬件,DCGM 是事实标准,GPU Operator 自带,端口 9400)+ 引擎 /metrics(vLLM 原生)→ Prometheus 抓取 → Grafana 看板/告警;K8s 里常配 HPA 用这些自定义指标扩缩。

## 关键事实

- **DCGM(NVIDIA Data Center GPU Manager)是 GPU 监控事实标准**,DCGM exporter 端口 9400,GPU Operator 自动安装,Prometheus 直接抓;`GPU_UTIL` 会因 decode 访存受限「假满载」,需配 `DCGM_FI_PROF_SM_ACTIVE`、`PIPE_TENSOR_ACTIVE`、HBM 带宽才看真实算力(2025–2026 通行)。
- **vLLM 原生暴露 `/metrics`**:`gpu_cache_usage_perc`(KV 占用)、`num_requests_running/waiting/swapped`(排队)、`time_to_first_token_seconds_bucket` 等;KV 占用与排队深度是吞吐崩塌/过载的先行指标,p99 用 `histogram_quantile` 估算守 SLO。
- **告警分层**:容量/先行指标(KV>90%、waiting>running)提前预警,SLO/燃烧率(p99 破阈)做页面告警;K8s 上常以这些自定义指标驱动 HPA 自动扩缩(GKE/EKS 2025 常见实践)。
