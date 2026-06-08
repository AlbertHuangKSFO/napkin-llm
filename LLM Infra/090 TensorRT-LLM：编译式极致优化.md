[[090 TensorRT-LLM：编译式极致优化]]:[[087 引擎全景：六大 runtime 选型|六引擎]]里的「NVIDIA 性能天花板」——不在运行时解释执行,而是**编译期**把「这个模型 × 这种卡 × 这种量化」固化成专用 `.engine`,靠**算子融合、量化、为 SKU 选 kernel、in-flight batching** 把吞吐和尾延迟榨到极限。代价是运维重、仅 NVIDIA、模型一改要重编译。与 [[088 vLLM V1 架构剖析|vLLM]] 的「运行时灵活」形成经典对照。

## ① 类比:量产前先开模具

vLLM 像 3D 打印:改个设计马上重打,灵活但单件慢一点。TensorRT-LLM 像**注塑开模**:先花几小时为「这个零件 × 这台机器」开一套钢模(编译 engine),之后每件又快又一致;但**零件一改就得重新开模**。所以它属于「设计定稿、要大批量量产」的阶段,不属于「天天改图」的迭代期。

## ② 小数字例子:编译换来多少、代价多大

相对量级(同卡、社区基准示意,随版本/负载变动):

- **吞吐**:编译后高并发吞吐比 vLLM 高约 **20–40%**;某基准里 batch=8,vLLM ~85–95 tok/s、p95 ~450ms,TensorRT-LLM ~110–130 tok/s、p95 ~350ms。
- **极限**:2025 报告 H100 + FP8 可达 **1e4+ output tok/s** 量级、**TTFT 亚 100ms**。
- **代价**:每个「模型 × 卡型 × 量化 × 并行度」组合要**单独编译**(几十分钟到几小时);换 LoRA、换权重、换 TP 配置、换卡都触发重编译。

结论:吞吐/尾延迟领先是真,但仅在**模型冻结 + 长期高并发 + 全 NVIDIA**时回得了本。

![[eng-090vLLM对TRT权衡.png]]

## ③ 原理:编译期做了什么

**1. 算子融合(kernel fusion)。** 把 LayerNorm、GEMM、bias、激活等多个小 kernel 合成**单个 CUDA kernel**,减少 kernel 启动开销和 HBM 访存往返(背景见 [[021 kernel 融合：为何能省带宽|kernel 融合]])。

**2. 深度量化。** 编译进 FP8、INT4(AWQ/GPTQ)等,权重与 KV 都可低精度,显存和带宽双省(见 [[027 量化内核：W4A16、W8A8 GEMM|量化 GEMM]]、[[034 KV 量化部署：FP8、INT8 KV|KV 量化]])。

**3. 为具体 SKU 选 kernel / tactic。** 编译器针对**目标 GPU SKU + batch/seq 范围**挑最优 kernel 实现、吃满 TensorCore(见 [[022 GEMM：cuBLAS、CUTLASS 与 Tensor Core 利用率|TensorCore 利用率]])。这是「运行时引擎」做不到的——它不知道你将来跑在哪张卡。

**4. plugin 机制。** plugin 是插进网络图的节点,映射到用户自定义 GPU kernel;执行 engine 时触发其封装的 kernel,允许把特殊算子塞进编译图。

**5. 运行期 in-flight batching。** 它对[[041 连续批处理：迭代级调度内幕|连续批]]的叫法:每个生成 step 把新请求并进运行中的 batch;配 paged KV(含 encoder-decoder 也支持 in-flight batching)。

![[eng-TRT-LLM编译流程.png]]

整条链路分两段:**编译期(离线、慢、专用)** 产出 `.engine`,**运行期(在线、快)** 执行;模型一改回到编译期重来——这正是它和 vLLM 的本质分界。

![[eng-090重编译触发.png]]

## ④ 代码/配置:两步流程(编译 + serve)

```bash
# Step 1：编译 engine（离线，慢；指定量化/并行度/卡型相关参数）
trtllm-build \
  --checkpoint_dir ./Llama-3.1-8B-ckpt \
  --output_dir ./engine \
  --gemm_plugin auto \
  --max_batch_size 64 \
  --max_input_len 4096 --max_seq_len 8192

# Step 2：用编译好的 engine 起 OpenAI 兼容服务
trtllm-serve ./engine --port 8000

# 也常通过 Triton 的 tensorrtllm_backend 部署(见 093),
# 调度策略可选 MAX_UTILIZATION 或 GUARANTEED_NO_EVICT
```

❌ 反模式:产品还在天天改 prompt 模板、换 LoRA、试不同并行度,就上 TensorRT-LLM——每改一次等一次重编译,工程时间全烧在 build 上。
✅ 正解:**vLLM 起步**做迭代;**模型冻结、长期高并发、吞吐=钱、全 NVIDIA**后,再为定稿模型编译 TRT-LLM engine 收割吞吐/尾延迟;通常通过 [[093 Triton Inference Server：多框架后端|Triton]] 的 `tensorrtllm_backend` 进生产。

## 面试高频

- **「TensorRT-LLM 为什么快?」** 编译期算子融合 + 深度量化 + 为目标 SKU 选 kernel 吃满 TensorCore + in-flight batching;把优化前移到离线,运行时只执行固化的专用 engine。
- **「它最大的缺点?」** 每个「模型×卡型×量化×并行度」要单独编译(几十分钟~几小时),迭代慢、工程门槛高,且**仅 NVIDIA**。
- **「in-flight batching 和连续批是一回事吗?」** 是,只是 NVIDIA 的叫法:每 step 把新请求并入运行中 batch。
- **「什么时候从 vLLM 迁到 TensorRT-LLM?」** 模型冻结、长期高并发、尾延迟(p99)关键、全 NVIDIA,且吞吐直接等于成本时。
- **「怎么进生产?」** 直接 `trtllm-serve`,或挂到 Triton 的 `tensorrtllm_backend`(可选 MAX_UTILIZATION / GUARANTEED_NO_EVICT 调度策略)。

## 关键事实

- NVIDIA 官方,**仅 NVIDIA GPU**;核心是把模型**编译成针对特定 SKU 的 engine**。
- 优化:算子融合、FP8/INT4 量化、为 SKU 选 kernel、plugin 自定义 kernel、in-flight batching、paged KV。
- 相对 vLLM:高并发吞吐约高 **20–40%**、尾延迟更低;2025 报告 H100 FP8 可达 1e4+ tok/s、TTFT 亚 100ms(量级示意)。
- 2025 多版本迭代;encoder-decoder 也支持 in-flight batching。
- **最新进展(2025-2026)**:**TensorRT-LLM 1.0** 于 2025-09-24 发布,把**PyTorch-native 架构定为默认且生产就绪**(从早期 C++ 编译式重构而来),LLM API 稳定并给出向后兼容保证——即「编译式」标签正在淡化,主线转向 PyTorch 运行时 + 模块化 Python,降低了"改模型即重编译"的门槛。(来源:NVIDIA Developer Forums "TensorRT LLM 1.0 is here" / GitHub NVIDIA/TensorRT-LLM,2025)
- 部署:`trtllm-build` → `trtllm-serve`,或经 Triton `tensorrtllm_backend`(调度策略 MAX_UTILIZATION / GUARANTEED_NO_EVICT)。
