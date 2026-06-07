[[082 FP8 服务：H100、Blackwell 原生|FP8 服务：H100、Blackwell 原生]]是 2025 高吞吐 LLM 服务的**默认精度**:权重和激活都用 **FP8 E4M3**,跑在 H100/H200/Blackwell 的**原生 FP8 [[002 GPU 架构：SM、CUDA Core 与 Tensor Core|Tensor Core]]** 上,算力约 BF16 的 2×,而精度**近无损**。它是 [[081 W8A8 与 W4A16：权重激活精度组合|W8A8]] 里"浮点版 8-bit"的那条路,数值格式细节回链 [[005 数值格式：FP32、TF32、BF16、FP8、FP4|数值格式]]、算法回链 [[098 FP8 训练推理与 AQLM 极低比特|FP8 训练推理]]。

## 直觉

FP8 比 INT8 香在哪?**浮点有指数位**。INT8 是均匀刻度尺,一旦激活里冒出离群值,整把尺子被拉长、小值全挤成 0;FP8 E4M3(4 位指数、3 位尾数,范围 ±448)像一把"对数刻度尺",大小值都能落上,**天然更耐离群值**——这正是激活量化最头疼的问题(见 [[085 校准、精度回退与离群值|离群值]])。再加上 H100 起 Tensor Core **硬件原生**支持 FP8×FP8→FP32 累加,等于"既好量又好算"。一句话:FP8 把 W8A8 的吞吐收益拿到手,同时把 INT8 的掉点风险压到近乎为零——**前提是你的卡原生支持**。

## 例子

- **吞吐**:H100 FP8 上,TensorRT-LLM 64 并发可达 **>10000 输出 token/s**,首 token ~100ms;相对 A100 是 **~4.6× 峰值吞吐、~4.4× 更快首 token**。
- **精度**:研究测得 **FP8 (W8A8-FP) 在各规模模型上基本无损**,多数生产模型掉点可忽略(<0.5%)。
- **显存**:权重 BF16→FP8 减半;70B 权重 140GB → **70GB**。
- **反例(硬件不原生)**:A100 上强行 FP8 走**软件模拟**,可能比原生 INT8 还**慢 10–20%**——FP8 的收益**绑死硬件**。

## 原理

FP8 服务分两步(见图):**离线**用校准集生成 FP8 checkpoint(per-tensor FP32 scale 把张量缩进 ±448),**在线**激活动态量化后做 FP8 GEMM、FP32 累加:

$$
Y = \big(W_{\text{fp8}}\cdot X_{\text{fp8}}\big)_{\text{FP32 累加}}\cdot s_W s_X,\qquad
\text{E4M3 范围}=\pm 448
$$

关键三点:① **累加在 FP32**,所以乘法虽是 FP8、求和不丢精度,这是"近无损"的来源;② E4M3 用于权重/激活,**E5M2**(范围大精度低)主要用于训练梯度,推理少用;③ scale 粒度:权重常 per-tensor,激活可 per-tensor(静态校准)或 **per-token 动态**(更准,vLLM 默认对部分层用动态)。FP8 不需要 INT8 那套 [[096 AWQ 与 SmoothQuant|SmoothQuant]] 离群值迁移,因为浮点指数位已经覆盖了动态范围——这是它工程上最省心的地方。

## 图

![[sq-FP8服务流程.png]]

## 代码

```bash
# vLLM:在线动态 FP8(无需预量化 checkpoint,加载时即时量化权重)
vllm serve meta-llama/Llama-3-70B --quantization fp8

# 或加载已校准的 FP8 静态 checkpoint(激活 scale 离线定,延迟更低)
vllm serve neuralmagic/Llama-3-70B-FP8

# TensorRT-LLM:构建时开 FP8
trtllm-build --checkpoint_dir ./llama --dtype bfloat16 --use_fp8 ...
```

❌ 误区:在 A100 上开 `--quantization fp8` 期待提速。
✅ 正解:FP8 收益**只在 Hopper(H100/H200)、Ada、Blackwell 原生**;Ampere/A100 无 FP8 Tensor Core,会走软件模拟反而变慢——A100 上要量化激活请用 **INT8 W8A8**。


![[sq-082FP8耐离群.png]]

## 面试高频

- **Q:FP8 相对 INT8 的核心优势?** 浮点指数位 → 动态范围大、天然耐激活离群值,无需 SmoothQuant 这类离群值迁移;且 H100 起原生支持。
- **Q:为什么 FP8 能"近无损"?** GEMM 累加在 FP32,只有乘法操作数是 FP8;E4M3 范围 ±448 配 scale 足以覆盖多数激活分布。
- **Q:E4M3 和 E5M2 怎么选?** 推理权重/激活用 E4M3(精度优先);E5M2 范围大精度低,主要给训练梯度。
- **Q:FP8 在 A100 上能用吗?** 能加载但无原生算力,软件模拟可能更慢——A100 选 INT8。

## 关键事实

- 2025 现状:FP8 是 H100/Blackwell 上**生产级高吞吐服务的默认精度**,近无损、~2× BF16 算力。
- 硬件门槛:FP8 Tensor Core 自 **Hopper(H100, 2022)、Ada(L40S)** 起原生;A100/Ampere **无**。
- Blackwell(B200)在 FP8 之上再加**原生 FP4(NVFP4)**,但 FP8 仍是稳妥默认;FP4 更激进、需 QAT/蒸馏回收(见 [[085 校准、精度回退与离群值|校准与回退]])。
- KV 也可独立用 FP8 E4M3 存(见 [[083 KV 量化与服务吞吐|KV 量化与服务吞吐]]),与权重/激活 FP8 正交。
