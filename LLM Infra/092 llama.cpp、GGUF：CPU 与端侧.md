[[092 llama.cpp、GGUF：CPU 与端侧]]:[[087 引擎全景：六大 runtime 选型|六引擎]]里的「本地/边缘派」——纯 C/C++ 写的 llama.cpp 配 **GGUF 单文件量化格式**,把 LLM 跑进 CPU、Apple Metal、手机、树莓派,做到「下个文件就能本地离线跑」。它不追求大规模高并发,而是把**单用户、低门槛、可移植**做到极致;Ollama / LM Studio 底层多是它。

## ① 类比:把整辆车压成一个 U 盘

云端引擎像「机房里的服务器集群」,要 CUDA、要 Python 环境、要 GPU。llama.cpp + GGUF 像把「一辆能开的车」压缩成**一个 U 盘文件**:GGUF 里塞好了量化权重 + tokenizer + 元数据,插到任何机器(哪怕没 GPU)——纯 C/C++ 引擎读它就能跑。代价是它是「家用代步」,不是「赛道/货运」:并发能力弱,但**单人随处可开**。

## ② 小数字例子:GGUF 量化把模型塞进消费级硬件

- **Q4_K_M**:非整数有效位宽约 **4.5 bpw**;7B 模型量化后约 **4.1 GB**(比 FP16 小约 70%),WikiText-2 [[LLM/109 语言模型评估：困惑度与 bits-per-byte|困惑度]]相对 FP16 仅升 ~0.1–0.3,对话/指令任务几乎无感。
- **无 GPU 也能跑**:CPU 路径用 AVX2/AVX-512(x86)、ARM NEON(Apple/Arm)SIMD;还能**部分层 offload 到 GPU**。
- **Apple 一等公民**:M 系列经 NEON + Accelerate + Metal 优化,MacBook 无独显也能跑 4bit 70B(慢但能跑)。
- 这把「Llama-3.3-70B、Qwen 大模型 FP16 装不进消费硬件」的鸿沟用 4/5bit 量化补上。

![[eng-092量化档位.png]]

## ③ 原理:两个核心资产

**1. GGUF 格式(GGML Universal Format)。** llama.cpp 原生格式,**单文件容器**,打包:量化权重(2~8bit)、tokenizer 配置、metadata(模型架构、上下文窗口等超参)。一个文件即可分发整模型,已成**本地模型分发事实标准**。命名里的 `Q4_K_M`:`Q4`=4bit、`_K`=K-quant 分组量化、`_M`=中等档(还有 `_S`/`_L`),平衡体积与质量(量化背景见 [[LLM/092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]、[[LLM/093 PTQ 与 QAT|PTQ]])。

**2. llama.cpp 引擎(纯 C/C++、零重依赖)。** 不依赖 Python/CUDA 运行时,后端覆盖**最广**:CPU(AVX/NEON)、Apple Metal、CUDA、ROCm、Vulkan、SYCL、甚至 WebAssembly。设计取舍很清楚:**不追求大规模高并发吞吐**(连续批等能力弱于 [[088 vLLM V1 架构剖析|vLLM]]),而把「一台普通设备上本地、离线、低门槛跑起来」做到极致——所以它是**单用户/低并发**场景的最优解。

**生态。** Ollama、LM Studio 等「一键本地跑 LLM」工具,底层多是 llama.cpp;它把端侧推理的复杂度封装成「下个 GGUF 文件 + 一行命令」。

![[eng-llama.cpp端侧.png]]

与云端引擎的本质分界:云端 = [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] + [[041 连续批处理：迭代级调度内幕|连续批]] 服大量并发;llama.cpp = 极致可移植 + 量化,服一个人。

## ④ 代码/配置:量化 + 本地起服务

```bash
# 1) 把 HF 模型转 GGUF 并量化到 Q4_K_M(推荐档)
python convert_hf_to_gguf.py ./Llama-3.1-8B-Instruct --outfile model-f16.gguf
llama-quantize model-f16.gguf Llama-3.1-8B-Instruct-Q4_K_M.gguf Q4_K_M

# 2) 起本地 OpenAI 兼容 server(无需 CUDA/Python)
llama-server -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --port 8080 \
  -c 8192 \          # 上下文长度
  -ngl 99            # offload 到 GPU 的层数(Metal/CUDA;纯 CPU 设 0)

# 3) 直接命令行对话
llama-cli -m Llama-3.1-8B-Instruct-Q4_K_M.gguf -p "解释 GGUF 格式"
```

❌ 反模式:拿 llama.cpp 当高并发在线服务,几百路并发压上去——它的批处理/调度不是为此设计,吞吐和延迟都吃亏。
✅ 正解:**端侧/无 GPU/单用户/离线**用 llama.cpp + GGUF(Q4_K_M 是体积-质量甜点);需要规模化在线高并发回到 [[088 vLLM V1 架构剖析|vLLM]] / [[089 SGLang：RadixAttention、HiCache 与前端|SGLang]] / [[090 TensorRT-LLM：编译式极致优化|TensorRT-LLM]]。

## 面试高频

- **「GGUF 是什么、为什么流行?」** llama.cpp 原生**单文件容器**,打包量化权重+tokenizer+metadata,一个文件分发整模型;成本地模型分发事实标准。
- **「Q4_K_M 里的字母什么意思?」** Q4=4bit,_K=K-quant 分组量化,_M=中档(有 S/M/L);约 4.5 bpw,7B≈4.1GB,质量损失很小。
- **「llama.cpp 为什么能在 CPU/手机跑?」** 纯 C/C++ 零重依赖 + 量化 + SIMD(AVX/NEON)+ 后端覆盖最广(含 Metal/Vulkan/WASM),可部分层 GPU offload。
- **「它为什么不适合高并发服务?」** 设计取向是单用户/低并发可移植,连续批等并发调度能力弱于 vLLM。
- **「Ollama 和 llama.cpp 什么关系?」** Ollama/LM Studio 等本地工具底层多基于 llama.cpp,做了易用封装。

## 关键事实

- 仓库 `ggml-org/llama.cpp`,纯 C/C++;**GGUF** = GGML Universal Format,单文件容器(权重+tokenizer+metadata)。
- **Q4_K_M** ≈ 4.5 bpw;7B 模型 ≈ 4.1GB(比 FP16 小约 70%),困惑度相对 FP16 仅升 ~0.1–0.3。
- 后端最广:CPU(AVX2/512、ARM NEON)、Apple Metal(一等公民,经 Accelerate)、CUDA、ROCm、Vulkan、SYCL、WebAssembly;支持部分层 GPU offload。
- 定位:**单用户/低并发/端侧/离线**,不追求大规模高并发吞吐。
- 下游:Ollama、LM Studio 等本地工具底层多基于它;`llama-quantize` 做量化,`llama-server` 起 OpenAI 兼容端点。
