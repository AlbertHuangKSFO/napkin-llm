# LLM Infra 学习地图

> 把训练好的模型**跑起来、扛住并发、压低成本**这件事的全栈工程学科。从「一次请求落到一张卡」的硬件视角(GPU 体系结构、HBM/SRAM 带宽、Roofline)→ 推理两阶段本质(prefill 算力受限、decode 访存受限)→ CUDA 高性能 kernel(FlashAttention、FlashMLA)→ KV-Cache 系统(PagedAttention、前缀缓存、分层 offload)→ 批处理与调度(连续批、chunked prefill)→ 解耦式服务(PD 分离、DistServe、Mooncake、Dynamo)→ 推理侧并行(TP/PP/EP、wide-EP)→ 集群通信与网络(NCCL、IB/RoCE、rail-optimized)→ 解码加速(投机解码、EAGLE-3)→ 服务量化 → 推理引擎(vLLM/SGLang/TRT-LLM)→ 云原生编排(K8s、KServe、llm-d)→ 可观测/成本/SLO → 高级服务特性 → **真实可跑的实操 capstone**。学术原理与工业落地并重,面试可直接答。这里只放定位与链接,不写正文。配套地基见 [[LLM]] 域(模型怎么来)与 [[深度学习基础]] 域(数学与神经网络)。

## ① GPU 硬件与计算基础
- [[001 LLM 推理的系统视角：从一次请求到一张卡|推理系统视角]]
- [[002 GPU 架构：SM、CUDA Core 与 Tensor Core|GPU 架构]]
- [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|GPU 内存层级]]
- [[004 算力 vs 带宽：Roofline 与算术强度|Roofline 与算术强度]]
- [[005 数值格式：FP32、TF32、BF16、FP8、FP4|数值格式]]
- [[006 GPU 代际：A100 到 H100、H200、B200、GB200|GPU 代际]]
- [[007 异构加速器：TPU、Trainium、MI300X|异构加速器]]
- [[008 片内互联：NVLink、NVSwitch、NVLink-C2C|片内互联]]
- [[009 节点与机架：PCIe、GB200 NVL72、scale-up 与 scale-out|节点与机架]]
- [[010 显存墙与 LLM 推理的本质约束|显存墙]]
- [[011 单卡能放多大模型：参数与 KV 显存预算|单卡显存预算]]

## ② 推理两阶段本质
- [[012 自回归推理全流程：一个 token 的旅程|自回归推理全流程]]
- [[013 Prefill 阶段：计算受限|Prefill 计算受限]]
- [[014 Decode 阶段：访存受限|Decode 访存受限]]
- [[015 KV-Cache 的显存账(逐层手算)|KV-Cache 显存账]]
- [[016 batch 如何改变算术强度|batch 与算术强度]]
- [[017 吞吐与延迟：根本性张力|吞吐与延迟张力]]
- [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标定义]]

## ③ CUDA 与高性能 kernel
- [[019 CUDA 执行模型：grid、block、warp|CUDA 执行模型]]
- [[020 显存合并访问与 bank conflict|合并访问与 bank conflict]]
- [[021 kernel 融合：为何能省带宽|kernel 融合]]
- [[022 GEMM：cuBLAS、CUTLASS 与 Tensor Core 利用率|GEMM 与利用率]]
- [[023 online softmax 与数值稳定|online softmax]]
- [[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention 1、2]]
- [[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FlashAttention-3]]
- [[026 FlashMLA：DeepSeek 的 MLA 推理内核|FlashMLA]]
- [[027 量化内核：W4A16、W8A8 GEMM|量化内核]]
- [[028 Triton：用 Python 写 GPU kernel|Triton]]
- [[029 何时写自定义 kernel：Nsight 性能分析|自定义 kernel 与性能分析]]

## ④ KV-Cache 系统
- [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention 深入]]
- [[031 KV 显存碎片与 block 管理|KV 碎片与 block 管理]]
- [[032 前缀缓存：RadixAttention 树结构|前缀缓存 RadixAttention]]
- [[033 自动前缀缓存的命中与失效|自动前缀缓存命中]]
- [[034 KV 量化部署：FP8、INT8 KV|KV 量化部署]]
- [[035 KV 驱逐与压缩：H2O 与注意力汇|KV 驱逐与注意力汇]]
- [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|KV 分层 offload]]
- [[037 Mooncake：KVCache 中心的存储池|Mooncake 存储池]]
- [[038 KV-aware 路由与跨引擎复用|KV-aware 路由]]
- [[039 vAttention：不靠 PagedAttention 的动态显存|vAttention]]

## ⑤ 批处理与请求调度
- [[040 静态批、动态批与连续批|静态批与连续批]]
- [[041 连续批处理：迭代级调度内幕|连续批处理内幕]]
- [[042 chunked prefill：切块融合|chunked prefill]]
- [[043 prefill、decode 干扰与 stall|PD 干扰与 stall]]
- [[044 调度器设计：waiting、running 队列与抢占|调度器设计]]
- [[045 抢占：重计算 vs swap|抢占：重计算与 swap]]
- [[046 优先级、公平性与饥饿|优先级与公平性]]
- [[047 准入控制与排队论：队列长度到延迟|准入控制与排队论]]

## ⑥ 解耦式服务 PD 分离
- [[048 为何分离 prefill 与 decode|为何分离 PD]]
- [[049 DistServe：goodput 优先的解耦|DistServe]]
- [[050 Splitwise：异构硬件分工|Splitwise]]
- [[051 Mooncake：KVCache 中心解耦架构|Mooncake 架构]]
- [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|NVIDIA Dynamo]]
- [[053 KV 传输：NIXL、点对点与带宽|KV 传输 NIXL]]
- [[054 PD 配比与独立扩缩|PD 配比与扩缩]]
- [[055 聚合还是解耦：决策与权衡|聚合还是解耦]]

## ⑦ 推理侧并行
- [[056 推理并行总览：TP、PP、EP、DP、SP 各管什么|推理并行总览]]
- [[057 张量并行推理：延迟换显存|张量并行推理]]
- [[058 TP 的通信：每层 all-reduce 与 NVLink 依赖|TP 通信]]
- [[059 流水线并行推理：micro-batch 与气泡|流水线并行推理]]
- [[060 数据并行与副本扩展|数据并行与副本]]
- [[061 专家并行 EP：大规模 MoE 服务|专家并行 EP]]
- [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP 部署]]
- [[063 注意力数据并行(attn-DP)|注意力数据并行]]
- [[064 序列与上下文并行推理(长序列)|序列与上下文并行]]
- [[065 并行策略组合：单卡、多卡、多节点决策树|并行策略组合]]

## ⑧ 集群通信与网络
- [[066 NCCL：集合通信库与原语|NCCL 原语]]
- [[067 all-reduce 算法：ring、tree 与 SHARP|all-reduce 算法]]
- [[068 all-to-all：MoE、EP 的通信瓶颈|all-to-all 瓶颈]]
- [[069 InfiniBand vs RoCE|InfiniBand vs RoCE]]
- [[070 GPU 集群拓扑：fat-tree、rail-optimized、dragonfly|集群拓扑]]
- [[071 计算通信重叠技巧|计算通信重叠]]
- [[072 万卡规模通信：故障、拥塞与扩展极限|万卡规模通信]]

## ⑨ 解码加速
- [[073 投机解码系统：draft-verify 全流程|投机解码全流程]]
- [[074 EAGLE-3：工业标准投机解码|EAGLE-3]]
- [[075 Medusa 与多头草稿|Medusa 多头草稿]]
- [[076 lookahead 解码与 n-gram|lookahead 解码]]
- [[077 多 token 预测 MTP(DeepSeek)|多 token 预测 MTP]]
- [[078 接受率、草稿长度与收益分析|接受率与收益分析]]
- [[079 投机解码与连续批、前缀缓存的兼容|投机解码兼容性]]

## ⑩ 服务侧量化
- [[080 服务量化总览：W、A、KV 三处|服务量化总览]]
- [[081 W8A8 与 W4A16：权重激活精度组合|W8A8 与 W4A16]]
- [[082 FP8 服务：H100、Blackwell 原生|FP8 服务]]
- [[083 KV 量化与服务吞吐|KV 量化与吞吐]]
- [[084 引擎里的量化集成：AWQ、GPTQ、FP8 加载|引擎量化集成]]
- [[085 校准、精度回退与离群值|校准与离群值]]
- [[086 量化叠加投机解码、PD 分离的坑|量化叠加的坑]]

## ⑪ 推理引擎深入
- [[087 引擎全景：六大 runtime 选型|引擎全景选型]]
- [[088 vLLM V1 架构剖析|vLLM V1 架构]]
- [[089 SGLang：RadixAttention、HiCache 与前端|SGLang]]
- [[090 TensorRT-LLM：编译式极致优化|TensorRT-LLM]]
- [[091 TGI 与 Hugging Face 生态|TGI]]
- [[092 llama.cpp、GGUF：CPU 与端侧|llama.cpp 与 GGUF]]
- [[093 Triton Inference Server：多框架后端|Triton Inference Server]]
- [[094 OpenAI 兼容 API 与引擎抽象|OpenAI 兼容 API]]

## ⑫ 服务编排与云原生
- [[095 从单进程到集群：服务架构演进|服务架构演进]]
- [[096 Kubernetes 上部署 LLM：基本盘|K8s 部署基本盘]]
- [[097 KServe：模型生命周期与 LLMInferenceService|KServe]]
- [[098 llm-d：K8s 原生分布式推理|llm-d]]
- [[099 Ray Serve：Python 优先的编排|Ray Serve]]
- [[100 推理网关与智能路由(cache-aware)|推理网关 cache-aware]]
- [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|自动扩缩与冷启动]]
- [[102 GPU 调度：Kueue、MIG、分时与碎片|GPU 调度]]
- [[103 多副本、蓝绿与金丝雀发布|发布策略]]
- [[104 AIBrix 与其他生产栈|AIBrix 与生产栈]]

## ⑬ 生产可观测、可靠与成本
- [[105 SLO、SLA 设计：为推理定指标|SLO 与 SLA 设计]]
- [[106 压测方法：开环 vs 闭环、并发模型|压测方法]]
- [[107 基准：MLPerf Inference 与 InferenceMAX|基准 MLPerf]]
- [[108 监控：GPU 利用率、KV 占用、排队、p99|监控指标]]
- [[109 容错与健康检查：重试与节点失效|容错与健康检查]]
- [[110 成本与 TCO：每百万 token 成本怎么算|成本与 TCO]]
- [[111 多租户隔离、配额与限流|多租户与限流]]
- [[112 负载均衡与会话亲和(prefix affinity)|负载均衡与亲和]]
- [[113 容量规划：从 QPS 反推 GPU 数|容量规划]]

## ⑭ 高级服务特性
- [[114 结构化输出：guided decoding 与 XGrammar|结构化输出 XGrammar]]
- [[115 多 LoRA 服务：S-LoRA、Punica 与热插拔|多 LoRA 服务]]
- [[116 长上下文服务：streaming、attention sink 与 KV 预算|长上下文服务]]
- [[117 多模态推理服务：图像、音频 token 化与缓存|多模态推理服务]]
- [[118 函数调用与工具服务化|函数调用服务化]]
- [[119 流式输出：SSE、token streaming 与背压|流式输出与背压]]
- [[120 语义缓存与响应复用|语义缓存]]
- [[121 机密推理与隐私：TEE、加密简述|机密推理 TEE]]
- [[122 端侧与边缘部署：量化、蒸馏与移动推理|端侧与边缘部署]]

## ⑮ 实操 capstone(真实配置 + 脚本,需 GPU 环境实跑)
- [[123 实操总览：从零搭一套推理服务|实操总览]]
- [[124 部署 vLLM：单卡 OpenAI 端点|部署 vLLM 单卡]]
- [[125 多卡 TP 部署 70B 模型|多卡 TP 部署 70B]]
- [[126 写一个 benchmark：测 TTFT、TPOT、吞吐|写 benchmark]]
- [[127 部署量化模型：AWQ、FP8 实战|部署量化模型]]
- [[128 调优：前缀缓存 + chunked prefill + 投机解码|调优三件套]]
- [[129 PD 分离实验：Dynamo、llm-d 上手|PD 分离实验]]
- [[130 K8s 上 KServe 部署 + 自动扩缩 + 监控面板|K8s 全栈部署]]

## ⑯ 跨域桥接
- 模型地基:整域 [[LLM]] 讲「模型怎么来」(Transformer、注意力、MoE、训练、对齐),本域讲「模型怎么跑」。重叠概念深链互引:[[LLM/068 并行总览：DP、TP、PP、EP、SP|并行总览]]↔[[056 推理并行总览：TP、PP、EP、DP、SP 各管什么|推理并行总览]];[[LLM/102 KV-Cache|KV-Cache 原理]]↔[[030 PagedAttention 深入：KV 当虚拟内存|KV 系统]];[[LLM/092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化原理]]↔[[080 服务量化总览：W、A、KV 三处|服务量化]];[[LLM/104 投机解码与 Medusa、Lookahead|投机解码原理]]↔[[073 投机解码系统：draft-verify 全流程|投机解码系统]];[[LLM/108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|推理引擎]]↔[[088 vLLM V1 架构剖析|vLLM 深剖]]。
- 数学地基:[[深度学习基础/06 矩阵乘法的几何意义|矩阵乘法]] 通向 [[022 GEMM：cuBLAS、CUTLASS 与 Tensor Core 利用率|GEMM]] 与 [[016 batch 如何改变算术强度|算术强度]]。
- 检索加速:[[032 前缀缓存：RadixAttention 树结构|前缀缓存]]、[[120 语义缓存与响应复用|语义缓存]] 服务 [[RAG/01 什么是 RAG|RAG]] 系统。
- 工具与编排:[[114 结构化输出：guided decoding 与 XGrammar|结构化输出]]、[[118 函数调用与工具服务化|工具服务化]] 是 [[Agent]] 的服务侧基座。
- 安全:[[121 机密推理与隐私：TEE、加密简述|机密推理]] 接 [[AI 安全]] 域。
