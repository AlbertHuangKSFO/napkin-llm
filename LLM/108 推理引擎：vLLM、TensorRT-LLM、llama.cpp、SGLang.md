[[108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang]]:把前面的优化(KV 分页、连续批、前缀缓存、量化、chunked prefill)打包成生产可用系统的四大开源引擎——**vLLM**(通用首选)、**TensorRT-LLM**(NVIDIA 极致性能)、**llama.cpp**(本地边缘)、**SGLang**(共享前缀)各有招牌与适用象限。

## ① 直觉:同一套优化,不同引擎的侧重不同

前面几篇讲的都是「概念/技术」:[[026 PagedAttention 与 KV 分页|KV 分页]]、[[105 连续批处理 continuous batching|连续批处理]]、[[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|前缀缓存与量化]]、[[106 chunked prefill 与 prefill、decode 解耦|chunked prefill]]。**推理引擎**就是把这些组装成一套能扛真实流量的服务系统(调度器 + 显存管理 + 优化 kernel + API)。

四大引擎用的底层概念高度重叠,差异在**侧重与定位**:

- **vLLM**:PagedAttention 的发源地,通用、易用、生态全 → **要快上线、常换模型,就选它**(业界默认起点)。
- **TensorRT-LLM**:NVIDIA 官方,编译期把模型图深度优化、算子融合、吃满 TensorCore → **单模型长期产线、吞吐至上、全 NVIDIA**。
- **llama.cpp**:C/C++ 写的,GGUF 量化格式,能跑在 CPU/手机/Mac → **本地、边缘、无 GPU**。
- **SGLang**:RadixAttention 把共享前缀的 KV 用前缀树自动复用 → **聊天/RAG/Agent 这类大量共享前缀的负载**。

选型本质是问两件事:**你的负载在「在线低延迟 / 离线高吞吐 / 共享前缀多 / 本地边缘」哪个象限?绑不绑 NVIDIA?**

**引擎到底由哪几块组成(拆开看就不神秘)。** 不管哪家,一个 serving 引擎都是这四层:① **API/前端**(OpenAI 兼容 HTTP 接口、流式 SSE、对话模板渲染);② **调度器**(连续批、准入、抢占、优先级,见 [[105 连续批处理 continuous batching|连续批]]);③ **显存/KV 管理**(分页、前缀缓存、swap/recompute,见 [[026 PagedAttention 与 KV 分页|PagedAttention]]);④ **执行后端**(优化 kernel:Flash-Attention、量化 GEMM、CUDA Graph、张量/流水并行)。四家的差异主要在③④的实现深度和②的策略,①基本趋同(都仿 OpenAI API)。理解这个分层,就知道「换引擎」换的是什么、benchmark 数字差在哪。

![[infer-推理引擎对比.png]]

## ② 例子:同一个 Llama-70B,四种场景四种选择

- **创业公司,A100×8,要尽快上线一个会频繁换的 chat 服务** → **vLLM**。装好即用,PagedAttention + 连续批开箱给到高吞吐(A100-80G 上约 3500 tok/s 量级),换模型只改一行。
- **大厂,单一固定模型,长期高并发产线,全是 H100** → **TensorRT-LLM**。花几小时为这个模型 + 这种卡型编译引擎,换来比 vLLM 高 30–50% 的高并发吞吐 + FP8 深度量化。代价:模型一变就要重编译。
- **个人想在 MacBook(无独显)本地跑 70B 的 4bit 量化版** → **llama.cpp**。下载一个 GGUF 文件,Metal 后端直接跑,无需 Python/CUDA 环境。
- **做一个 RAG 系统,几千条问题共享同一批检索文档 + 同一 system prompt** → **SGLang**。RadixAttention 让共享前缀的 KV 算一次复用,实测比 vLLM 高约 29% 吞吐。

**一个常见的纠结:vLLM 已经够快,什么时候才值得换 TensorRT-LLM?** 决策点不在「快不快」,在「模型还会不会变」。TensorRT-LLM 的吞吐优势来自**编译期**为「这个模型 × 这种卡」生成专用引擎,但模型一改(换权重、换 LoRA、改并行度、换卡型)就要**重新编译**(可能几十分钟到几小时)。所以:迭代期、A/B 实验期、频繁换模型 → vLLM(改一行就跑);模型冻结、长期高并发、吞吐直接等于钱、且全 NVIDIA → 才值得吃编译成本上 TensorRT-LLM。很多团队的现实路径是「vLLM 起步打磨产品,模型稳定后再评估是否迁 TensorRT-LLM」。

## ③ 原理:四家的招牌技术拆解

**vLLM —— PagedAttention 立家。** 核心是 [[026 PagedAttention 与 KV 分页|PagedAttention]]:KV-Cache 按页分配、块表映射,消除碎片、利用率冲到 90%+,从而支持更大 batch。配 [[105 连续批处理 continuous batching|连续批处理]](槽位随进随出)、Automatic Prefix Caching(自动复用公共前缀)、chunked prefill(默认开)。定位是**通用 serving 的事实标准**:模型支持广、社区活跃、上手最快、迭代灵活。多数团队的起点。

**TensorRT-LLM —— 编译期榨干 NVIDIA。** 不是「运行时解释执行」,而是把模型**编译成针对特定 GPU 优化的引擎**:算子融合(把多个小 kernel 合成一个,减少启动开销和访存往返)、深度量化(FP8、INT4 AWQ/GPTQ,见 [[095 GPTQ|GPTQ]])、为 TensorCore 定制 kernel、in-flight batching(它对连续批的叫法)。结果是高并发下吞吐领先 20–50%。代价:**每个模型 × 每种卡型都要单独编译**,迭代慢、工程门槛高——适合「模型定了、长期不换、吞吐就是钱」的产线。

**llama.cpp —— 量化 + 可移植,跑在一切上。** 纯 C/C++ 零重依赖,核心资产是 **GGUF**:一种把权重 2~8bit 量化、打包成单文件的格式,已成本地模型分发的事实标准。后端覆盖 CPU、Apple Metal、CUDA、ROCm、Vulkan、SYCL、甚至 WebAssembly。它**不追求大规模高并发吞吐**(连续批等能力弱于 vLLM),而是把「在一台普通设备(笔记本、手机、树莓派)上本地、离线、低门槛跑起来」做到极致。Ollama、LM Studio 等本地工具底层多是它。

**SGLang —— RadixAttention 复用共享前缀。** 在 PagedAttention 之上,把所有请求的前缀组织成一棵 **radix tree(基数树/前缀树)**:节点存对应 token 前缀的 KV。新请求进来,沿树匹配**最长已缓存前缀**,命中部分**直接复用 KV、跳过 prefill 重算**,只为新增 token 算 KV;树节点用 LRU 驱逐管显存。比普通「整段精确前缀」缓存粒度更细(任意分叉处都能共享)。再加结构化输出、多步 LLM 程序的前端 DSL。共享前缀越多收益越大(聊天多轮、RAG 同文档多问、Agent 同 system prompt 多次调用),吞吐比 vLLM 高约 29%。

**量化格式速查(选型常被问)。** 不同引擎支持不同量化方案,影响显存和质量:
- **GPTQ / AWQ**:训练后权重量化到 INT4(AWQ 按激活重要性保护关键通道),质量损失小,vLLM/TensorRT-LLM/TGI 都支持(见 [[095 GPTQ|GPTQ]])。
- **FP8**:H100/Ada 起硬件原生支持,精度比 INT4 高、速度快,TensorRT-LLM 的招牌。
- **GGUF(Q4_K_M 等)**:llama.cpp 的量化格式,K-quant 系列(Q2_K~Q8_K)在质量/体积间分档,本地部署事实标准。
- **NF4**:QLoRA 用的 4-bit(见 [[097 NF4 与 QLoRA 4-bit|QLoRA]]),主要用于微调而非纯推理。

记法:**产线推理用 GPTQ/AWQ(INT4)或 FP8(新卡),本地用 GGUF,微调用 NF4**。

![[infer-RadixAttention前缀树.png]]

**还有几家值得知道(面试加分)**:
- **TGI(Text Generation Inference)**:HuggingFace 官方,生态整合好、易部署,功能与 vLLM 重叠(连续批、PagedAttention 思路、投机解码),企业里常见。
- **LMDeploy**(商汤)、**MLC-LLM**(跨平台编译,能跑浏览器/手机)、**Ollama**(llama.cpp 之上的易用本地封装,一行拉模型)。
- **DeepSpeed-Inference / FasterTransformer**:早期的高性能推理库,部分能力被 vLLM/TensorRT-LLM 吸收。

**衡量引擎的关键指标(别只看「吞吐」)**:① **吞吐**(tok/s,离线/批处理看这个);② **TTFT**(首 token 时延,交互体验);③ **TPOT/ITL**(每 token 时延及其抖动,流式体验);④ **goodput**(满足 SLA 前提下的有效吞吐——这才是生产真正在乎的:吞吐再高但一半请求超时也没用)。不同引擎在「吞吐 vs 延迟」上有不同甜点,选型要对着自己的 SLA 测,而非看别人的 benchmark 数字。

**共同底座(都是概念,非某家专利)**:KV 分页去碎片、连续/in-flight 批处理提利用率、前缀缓存去重算、量化降显存、chunked prefill 稳 TPOT。四家差的是**实现侧重 + 定位**,不是用了完全不同的原理。绝大多数现代引擎也都内置**投机解码**(见 [[104 投机解码与 Medusa、Lookahead|投机解码]])与 **CUDA Graph / 算子融合**降启动开销。

## ④ 代码:四引擎最小起服务对照(伪代码风格)

```python
# vLLM —— 通用首选,几行起一个高吞吐 OpenAI 兼容服务
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3-8B", enable_prefix_caching=True)
out = llm.generate(["你好"], SamplingParams(max_tokens=128))
# 自动:PagedAttention + 连续批 + 前缀缓存 + chunked prefill

# SGLang —— 共享前缀场景,RadixAttention 自动复用
import sglang as sgl
@sgl.function
def qa(s, doc, question):
    s += sgl.system("你是问答助手")          # 多请求共享 → 前缀树命中,KV 复用
    s += sgl.user(doc + "\n问:" + question)
    s += sgl.assistant(sgl.gen("ans", max_tokens=128))

# llama.cpp —— 本地,GGUF 单文件,CPU/Metal 直接跑
#   $ ./llama-cli -m llama-3-8b-q4_k_m.gguf -p "你好"     # 无需 GPU/Python

# TensorRT-LLM —— 先「编译」再跑(换模型/换卡要重编译)
#   $ trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine --gemm_plugin fp8
#   再用 engine 起 Triton 服务,吃满 TensorCore

# ❌ 选型误区:本地 MacBook 无 GPU 还硬上 TensorRT-LLM(它仅 NVIDIA、要编译)
# ✅ 对症:本地边缘→llama.cpp;通用上线→vLLM;固定产线榨吞吐→TensorRT-LLM;共享前缀→SGLang
```

## 面试高频

- **Q:vLLM 的核心创新是什么?** PagedAttention——KV-Cache 按页(借鉴 OS 虚拟内存)非连续分配,消碎片、利用率 90%+,支撑更大 batch;配连续批处理与前缀缓存,是通用 serving 事实标准。
- **Q:TensorRT-LLM 比 vLLM 强在哪、代价是什么?** 编译期图优化 + 算子融合 + FP8/INT4 深度量化 + 吃满 TensorCore,高并发吞吐高 20–50%;代价是每模型/每卡型都要编译,迭代慢、仅 NVIDIA。
- **Q:llama.cpp 适合什么、GGUF 是什么?** 适合本地/边缘/无 GPU;GGUF 是其 2–8bit 量化的单文件权重格式,已成本地模型分发事实标准,后端覆盖 CPU/Metal/CUDA/Vulkan 等。
- **Q:SGLang 的 RadixAttention 解决什么?** 用前缀树自动复用请求间的共享前缀 KV,命中部分跳过 prefill 重算;聊天/RAG/Agent 等共享前缀多的负载吞吐比 vLLM 高约 29%。
- **Q:给一个负载怎么选引擎?** 看象限:通用快速上线/常换模型→vLLM;固定模型长期产线、吞吐至上、全 NVIDIA→TensorRT-LLM;本地边缘→llama.cpp;共享前缀多→SGLang。
- **Q:这些引擎底层技术是不是各搞一套?** 不是。KV 分页、连续/in-flight 批、前缀缓存、量化、chunked prefill 是共享概念,差别在实现侧重与定位。
- **Q:vLLM 够快了,何时才换 TensorRT-LLM?** 不看快慢,看模型变不变。TensorRT-LLM 靠编译期为「模型×卡型」生成专用引擎,模型一改就要重编译(几十分钟~几小时)。迭代期/常换模型用 vLLM;模型冻结、长期高并发、全 NVIDIA 才上 TensorRT-LLM。
- **Q:衡量推理引擎只看吞吐吗?** 不。要看吞吐、TTFT(首 token 时延)、TPOT/ITL(每 token 时延及抖动),最终看 **goodput**(满足 SLA 的有效吞吐)——吞吐再高但大量请求超时也无意义。要对着自己 SLA 测。
- **Q:还有哪些引擎?** TGI(HuggingFace)、LMDeploy、MLC-LLM(跨平台/浏览器)、Ollama(llama.cpp 易用封装)、DeepSpeed-Inference/FasterTransformer(早期高性能库,能力多被吸收)。

## 关键事实

- vLLM 以 PagedAttention 为核心,是通用 LLM serving 的开源事实标准,默认启用连续批处理、Automatic Prefix Caching、chunked prefill(Kwon et al., 2023, arXiv:2309.06180;vLLM 文档 2024)。
- TensorRT-LLM 是 NVIDIA 官方引擎,通过编译期优化、算子融合、FP8/INT4 量化与 in-flight batching,在高并发下吞吐通常较 vLLM 高 20–50%,但需按模型/卡型编译、仅限 NVIDIA(NVIDIA TensorRT-LLM 文档,2023–2024)。
- llama.cpp 为 C/C++ 实现,GGUF 量化格式(2–8bit)已成本地模型分发事实标准,后端覆盖 CPU、Metal、CUDA、ROCm、Vulkan、SYCL、WebAssembly(ggml-org/llama.cpp,2023–)。
- SGLang 在 PagedAttention 基础上引入 RadixAttention,用基数树自动复用共享前缀 KV、跳过重复 prefill,共享前缀场景吞吐较 vLLM 高约 29%,并支持结构化生成(Zheng et al., 2024, arXiv:2312.07104)。
- 四引擎共享底层概念(KV 分页、连续/in-flight 批、前缀缓存、量化见 [[095 GPTQ]]、chunked prefill),差异在侧重与适用象限;选型取决于负载性质(在线/离线/共享前缀/本地)与硬件绑定。
- 其他常见引擎:TGI(HuggingFace 官方)、LMDeploy(商汤)、MLC-LLM(跨平台编译,可跑浏览器/手机)、Ollama(llama.cpp 易用封装)、DeepSpeed-Inference/FasterTransformer(早期高性能库)。
- 衡量推理服务的指标不止吞吐:还有 TTFT(首 token 时延)、TPOT/ITL(每 token 时延及抖动)与 goodput(满足 SLA 的有效吞吐);不同引擎在吞吐 vs 延迟上甜点不同,需对自身 SLA 实测。
- 现代引擎普遍内置投机解码(见 [[104 投机解码与 Medusa、Lookahead]])与 CUDA Graph / 算子融合以降启动与访存开销。
