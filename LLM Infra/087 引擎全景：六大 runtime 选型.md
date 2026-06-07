[[087 引擎全景：六大 runtime 选型]]:把 [[LLM/108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|推理引擎]]的概念升级成一张**选型决策图**——把 **vLLM、SGLang、TensorRT-LLM、TGI、llama.cpp、Triton** 六大 runtime 放进「负载象限 × 硬件绑定 × 运维成本」三轴,回答面试里最常被追问的「这个场景你选谁、为什么」。本篇是 `LLM Infra/` 域的入口 MOC 式总览,后续 [[088 vLLM V1 架构剖析|vLLM V1]]、[[089 SGLang：RadixAttention、HiCache 与前端|SGLang]]、[[090 TensorRT-LLM：编译式极致优化|TensorRT-LLM]]、[[091 TGI 与 Hugging Face 生态|TGI]]、[[092 llama.cpp、GGUF：CPU 与端侧|llama.cpp]]、[[093 Triton Inference Server：多框架后端|Triton]]、[[094 OpenAI 兼容 API 与引擎抽象|OpenAI 兼容 API]] 逐一深挖。

## ① 类比:选引擎像选交通工具

同样是「把人从 A 送到 B」,你不会只问「哪个最快」:**通勤**选地铁(vLLM:通用、班次密、随到随走),**赛道刷圈**选 F1(TensorRT-LLM:为这条赛道这台车专门调校,换赛道得重调),**山区独行**选越野车(llama.cpp:没铺好的路也能走、单人即可),**公司班车**选大巴(TGI:HF 生态接驳顺、稳),**拼车共享路段**选顺风车平台(SGLang:同一段路多人共享、省一次算),**调度整个车队**选交通指挥中心(Triton:本身不是车,而是把各种车统一编排)。选错不是「慢」,是「根本不匹配场景」。

## ② 小数字例子:同一个 Llama-70B,六种场景六种答案

相对差异(同卡型、量级示意,具体数随版本/负载变动,不当绝对值):

- **A100×8,要尽快上线、会频繁换模型的 chat** → **vLLM**。开箱 PagedAttention + 连续批,A100-80G 上 ~3500 tok/s 量级;换模型改一行。
- **固定单模型、长期高并发、全 H100** → **TensorRT-LLM**。编译后高并发吞吐比 vLLM 高约 **20–40%**、p99 尾延迟更低(社区基准:H100 FP8 可达 1e4+ tok/s 量级、TTFT 亚 100ms);代价是模型一改要重编译。
- **RAG:几千问共享同一批文档 + 同一 system prompt** → **SGLang**。RadixAttention 共享前缀只算一次,共享比例高时吞吐显著领先 vLLM。
- **已重度用 HF 全家桶、要稳生产** → **TGI**。Rust router + 连续批 + Prometheus/OTel 开箱。
- **MacBook 无独显跑 70B 4bit** → **llama.cpp**。下个 GGUF 文件 Metal 直跑,无需 CUDA/Python。
- **一套服务里既有 LLM 又有 embedding/视觉/分类多模型要编排** → **Triton**,用 vLLM 或 TRT-LLM 当后端。

## ③ 原理:三轴决策

把六引擎拍进三个问题(见图):

1. **负载象限**:在线低延迟 / 离线高吞吐 / **共享前缀多** / **本地边缘单用户**。共享前缀多 → SGLang;本地单用户 → llama.cpp;通用在线 → vLLM;吞吐至上且模型冻结 → TensorRT-LLM。
2. **硬件绑定**:**TensorRT-LLM 仅 NVIDIA**(编译进 CUDA 图);vLLM/SGLang 支持多后端但 CUDA 一等公民;llama.cpp 跨 CPU/Metal/CUDA/ROCm/Vulkan 最广。
3. **运维 / 工程成本**:llama.cpp 最低(单文件)、vLLM/SGLang 中、TGI 中(生产设施全)、TensorRT-LLM 最高(每模型×每卡型重编译)、Triton 另一维(编排复杂度)。

关键洞察:六家底层概念高度重叠([[LLM/026 PagedAttention 与 KV 分页|PagedAttention]]、[[041 连续批处理：迭代级调度内幕|连续批]]、[[042 chunked prefill：切块融合|chunked prefill]]、前缀缓存、量化都被多家采用),差异在**侧重与实现深度**,而非「有没有」。理解 [[094 OpenAI 兼容 API 与引擎抽象|引擎四层抽象]](API/调度/显存/后端)就知道换引擎换的是哪几层。

![[eng-引擎四层抽象.png]]

![[eng-六引擎选型矩阵.png]]

## ④ 决策矩阵(速查)

| 维度 | vLLM | SGLang | TensorRT-LLM | TGI | llama.cpp | Triton |
|---|---|---|---|---|---|---|
| 招牌 | PagedAttention | RadixAttention/HiCache | 编译式 engine | HF 集成/Rust router | GGUF/可移植 | 多框架后端编排 |
| 最佳场景 | 通用在线、常换模型 | 共享前缀(chat/RAG/Agent) | 固定模型、极致吞吐/延迟 | HF 生态、稳生产 | 端侧、CPU、单用户 | 多模型混合编排 |
| 硬件 | 多(CUDA 优先) | 多(CUDA 优先) | 仅 NVIDIA | 多 | 最广(含 CPU/手机) | 多(取决后端) |
| 运维成本 | 中 | 中 | 高(重编译) | 中 | 低 | 中高(编排) |
| 上手速度 | 快 | 快 | 慢 | 中 | 最快 | 中 |

## ⑤ 代码/配置:六个最小启动命令

```bash
# vLLM —— 一行起 OpenAI 端点
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# SGLang —— 共享前缀负载首选
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

# TensorRT-LLM —— 先编译 engine,再 serve(两步,运维重)
trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine --gemm_plugin auto
trtllm-serve ./engine --port 8000

# TGI —— HF 官方镜像
docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference \
  --model-id meta-llama/Llama-3.1-8B-Instruct

# llama.cpp —— 下个 GGUF 直接跑,无 CUDA/Python
llama-server -m Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8080

# Triton —— 装好 model repo 后启动,后端可挂 vLLM / TRT-LLM
tritonserver --model-repository=/models
```

❌ 反模式:迭代期就上 TensorRT-LLM,每改一次 prompt/LoRA/权重都等几十分钟重编译,把工程时间烧在编译上。
✅ 正解:**vLLM 起步打磨产品**;模型冻结、吞吐直接等于钱、且全 NVIDIA,再评估迁 TensorRT-LLM;共享前缀重就换 SGLang;端侧就 llama.cpp。


![[eng-087运维成本谱.png]]

## 面试高频

- **「给你一个 RAG 在线服务,QPS 高、共享同一批文档,选哪个引擎?」** SGLang(RadixAttention 把共享前缀 KV 算一次复用);若已绑 vLLM 也可开其 Automatic Prefix Caching,但分叉粒度不如 radix 树细。
- **「vLLM 够快了,什么时候才值得上 TensorRT-LLM?」** 决策点不是「快不快」,是「**模型还会不会变**」——TRT-LLM 优势来自编译期专用 engine,模型一改要重编译;模型冻结 + 长期高并发 + 全 NVIDIA 才值得吃这成本。
- **「Triton 和 vLLM 是竞品吗?」** 不是同层。Triton 是**编排/服务框架**,vLLM/TRT-LLM 是**引擎**;Triton 常把它们当后端,负责多模型、动态批、ensemble、指标。
- **「端侧/无 GPU 跑 LLM 选什么?」** llama.cpp + GGUF,CPU/Metal 直跑、单文件分发,Ollama/LM Studio 底层多是它。
- **「六家这么多重叠,本质差异在哪?」** 在[[094 OpenAI 兼容 API 与引擎抽象|四层抽象]]的 ②调度 ③显存 ④后端实现深度,①API 已趋同(都仿 OpenAI),所以引擎可替换。

## 关键事实

- vLLM **V1 架构** 2025-01 发布(见 [[088 vLLM V1 架构剖析|088]]),chunked prefill 默认开、近零开销前缀缓存。
- SGLang 由 LMSYS 主导;**HiCache** 2025-09 公布,分层 KV(GPU/host/分布式)宣称最高 6× 吞吐、80% TTFT 下降(见 [[089 SGLang：RadixAttention、HiCache 与前端|089]])。
- TensorRT-LLM 为 NVIDIA 官方;2025 多版本迭代,encoder-decoder 也支持 in-flight batching。
- TGI 三层架构(Launcher / Rust Router / Server),OpenAI 兼容 + Prometheus + OTel。
- llama.cpp 仓库 `ggml-org/llama.cpp`;GGUF 为本地模型分发事实标准,Q4_K_M ≈ 4.5 bpw、7B ≈ 4.1GB。
- Triton 两种 TRT-LLM 调度策略:`MAX_UTILIZATION` 与 `GUARANTEED_NO_EVICT`。
