[[093 Triton Inference Server：多框架后端]]:[[087 引擎全景：六大 runtime 选型|六引擎]]里**不在同一层**的那个——Triton 不是「又一个 LLM 引擎」,而是 NVIDIA 的**企业级服务/编排框架**:用**多框架后端**(把 [[090 TensorRT-LLM：编译式极致优化|TensorRT-LLM]]、[[088 vLLM V1 架构剖析|vLLM]]、PyTorch、ONNX 等当插件)、**动态批**、**ensemble 流水编排**,在一套服务器里统管 LLM + embedding + 视觉 + 前后处理的混合模型车队。

## ① 类比:Triton 是「机场塔台」,不是「飞机」

vLLM/TRT-LLM 是飞机(具体引擎),Triton 是**机场塔台 + 调度中心**:它自己不飞,但统一管理跑道(GPU)、排班(动态批)、多机型(多框架后端)、转机流程(ensemble:一次请求内部走 tokenize→LLM→detokenize 多步)。所以问「Triton vs vLLM 谁快」是问错层——正确问法是「**用 Triton 包住哪个后端**」。

![[eng-093塔台分层.png]]

## ② 小数字例子:什么时候非 Triton 不可

- **混合模型车队**:一个服务里既有 LLM,又有 embedding 模型、视觉编码器、分类器、自定义前后处理。Triton 用不同**后端**同时托管它们、各自动态批、并发执行、统一指标——这是单一 LLM 引擎做不到的。
- **动态批提吞吐**:请求突发或单条很小时,动态批把零碎请求**聚合**成大 batch 喂 GPU,显著提利用率。
- **ensemble 流水**:把「前处理 → LLM → 后处理」定义成一条 ensemble,客户端发**一次**请求,Triton 内部串起多模型,省掉多次网络往返。

![[eng-093ensemble流水.png]]
- **企业级设施**:多模型版本管理、负载均衡、并发执行、Prometheus 指标 API——大厂多模型平台常以它为底座。

## ③ 原理:三块能力

**1. 多框架后端(backend)。** Triton 把「怎么跑某类模型」抽象成可插拔后端:
- `tensorrtllm_backend`:服 [[090 TensorRT-LLM：编译式极致优化|TensorRT-LLM]] engine,C++ 实现的 `inflight_batcher_llm` 支持 [[041 连续批处理：迭代级调度内幕|in-flight batching]]、paged attention;两种调度策略 `MAX_UTILIZATION`(吞吐优先)/ `GUARANTEED_NO_EVICT`(不驱逐、稳延迟),由 `batch_scheduler_policy` 指定。
- `vllm_backend`:把 vLLM 当后端,享 PagedAttention + 连续批,同时套上 Triton 的基础设施。
- PyTorch / ONNX / TensorFlow / Python custom:跑 embedding、视觉、分类、自定义逻辑。

**2. 动态批(dynamic batching)。** Triton 核心调度能力:在请求突发或小请求场景把多条聚合成 batch,提 GPU 吞吐(对 LLM 后端,真正的迭代级连续批仍由该后端自身做;Triton 层的动态批更面向非 LLM 模型与请求聚合)。

**3. ensemble + 全生命周期管理。** ensemble 把多模型串成 DAG 流水线(前处理→推理→后处理),对外是一次调用。再加多模型并发执行、负载均衡、版本管理、Prometheus 指标——这是它「企业级」的实体。

![[eng-Triton多框架后端.png]]

一句话定位:**Triton 管「服务/编排」,把 vLLM/TRT-LLM 当「引擎后端」**——和它们是包含关系,不是替代关系。

## ④ 代码/配置:model repo + 后端选择

```bash
# 目录结构(模型仓库):每个模型一个目录,config.pbtxt 声明后端与批策略
# models/
#   llama_trtllm/        config.pbtxt -> backend: tensorrtllm
#   llama_vllm/          config.pbtxt -> backend: vllm
#   embedder/            config.pbtxt -> backend: python / onnxruntime
#   pipeline/            config.pbtxt -> ensemble(串前处理+LLM+后处理)

tritonserver --model-repository=/models
```

```protobuf
# config.pbtxt 片段:用 TRT-LLM 后端 + 动态批 + 调度策略
backend: "tensorrtllm"
parameters: { key: "batch_scheduler_policy" value: { string_value: "MAX_UTILIZATION" } }
dynamic_batching { max_queue_delay_microseconds: 1000 }
```

❌ 反模式:只跑单个 LLM、没有任何多模型/编排需求,却套一层 Triton——白白增加配置与运维复杂度,直接 `vllm serve` 更省。
✅ 正解:**有混合模型车队(LLM+embedding+视觉+前后处理)、要 ensemble 流水、要企业级版本/指标/编排**时上 Triton,后端挂 vLLM 或 TRT-LLM;纯单 LLM 直接用引擎自带 server。

## 面试高频

- **「Triton 和 vLLM/TensorRT-LLM 是竞品吗?」** 不是同层。Triton 是**服务/编排框架**,vLLM/TRT-LLM 是**引擎**;Triton 用 `vllm_backend`/`tensorrtllm_backend` 把它们当后端,负责多模型、动态批、ensemble、指标。
- **「什么时候必须用 Triton?」** 混合模型车队(LLM 旁边还有 embedding/视觉/分类/前后处理)、需要 ensemble 流水、要企业级版本管理与统一指标时。
- **「TRT-LLM backend 的两种调度策略?」** `MAX_UTILIZATION`(榨吞吐)与 `GUARANTEED_NO_EVICT`(不驱逐、稳延迟),由 `batch_scheduler_policy` 配置。
- **「Triton 的动态批 vs LLM 引擎的连续批?」** Triton 动态批面向请求聚合/非 LLM 模型;LLM 的迭代级连续批由其后端(vLLM/TRT-LLM)自身实现。
- **「ensemble 解决什么?」** 把前处理→推理→后处理串成一次调用的 DAG,省多次网络往返、统一编排。

## 关键事实

- NVIDIA **Triton Inference Server**:多框架推理服务/编排平台,非单一 LLM 引擎。
- 后端可插拔:`tensorrtllm_backend`、`vllm_backend`、PyTorch、ONNX、TensorFlow、Python custom。
- `tensorrtllm_backend` 的 `inflight_batcher_llm`(C++)支持 in-flight batching、paged attention;调度策略 `MAX_UTILIZATION` / `GUARANTEED_NO_EVICT`。
- 核心能力:模型仓库管理、**动态批**、并发模型执行、负载均衡、**ensemble** 流水、Prometheus 指标 API。
- 典型用途:LLM + embedding + 视觉 + 前后处理的**混合模型车队**统一编排。
