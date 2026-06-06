> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[127 部署量化模型：AWQ、FP8 实战]]:把 [[084 引擎里的量化集成：AWQ、GPTQ、FP8 加载|量化集成]] 落成命令——加载 AWQ(W4A16)或 [[082 FP8 服务：H100、Blackwell 原生|FP8]](W8A8)checkpoint,用 `--quantization awq`/`fp8`,实测显存与吞吐对比,看精度回退。

## ① 类比:把菜谱换成「浓缩版」

70B BF16 ~140GB 太占地。**量化是把食材浓缩**:AWQ 把权重压到 4-bit(体积 ~1/4),FP8 压到 8-bit(~1/2)。浓缩后**占地小、出菜可能更快**(显存省出来给更大 batch),但要确认味道没变(精度回退可接受)。关键:**激活精度决定算得快不快**——AWQ 是 W4A16(权重 4-bit、激活 16-bit,省显存为主),FP8 是 W8A8(权重激活都 8-bit,H100 原生 Tensor Core 直接加速,吞吐才真涨,见 [[081 W8A8 与 W4A16：权重激活精度组合|W4A16 vs W8A8]])。

## ② 加载量化模型:两条路

```bash
# A. 加载已量化的 AWQ checkpoint(社区/官方预量化)
vllm serve casperhansen/llama-3-70b-instruct-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92

# B. 加载 FP8 checkpoint(H100/Ada 原生 W8A8)
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --quantization fp8 \
  --tensor-parallel-size 2

# C. 在线动态 FP8(给 BF16 权重打 FP8,无需预量化文件)
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8
```

很多预量化 checkpoint 的 `config.json` 已含量化元数据,`--quantization` 可省;显式写更稳妥。

## ③ 原理:省哪、快哪、坑在哪

**省显存(W):** 权重从 16-bit → 4/8-bit,直接砍模型显存。70B BF16 140GB → FP8 ~70GB(2 卡可)→ AWQ ~35GB(单卡可能装下),省出的显存给更大 KV 池 / 更高并发。

**提吞吐靠激活精度(A):** AWQ 是 **W4A16**——反量化回 16-bit 再算,GEMM 仍走 BF16,**主要省显存、吞吐提升有限**;在 decode([[014 Decode 阶段：访存受限|访存受限]])阶段因权重读取减少也能略快。FP8 是 **W8A8**——H100/Blackwell 有原生 FP8 [[002 GPU 架构：SM、CUDA Core 与 Tensor Core|Tensor Core]],GEMM 直接 8-bit,**吞吐能真涨(官方 ~1.6×)**(见 [[082 FP8 服务：H100、Blackwell 原生|FP8 原生服务]])。

**硬件门槛:** FP8 W8A8 需 Hopper(H100)/Ada;Ampere/Turing 只能走 W8A16(Marlin kernel)。**精度回退:** 离群值大的层可能掉点,AWQ 靠激活感知保护重要通道,FP8 可对敏感层保留高精度(见 [[085 校准、精度回退与离群值|校准与离群值]])。

![[lab-量化部署对比.svg]]

三列并排看 BF16 / FP8(W8A8)/ AWQ(W4A16)的显存、吞吐、精度与硬件门槛——核心是「省显存看 W,提吞吐看 A」:

![[lab-127精度组合.svg]]

## ④ 实测对比(用 126 的尺子)

```bash
# 对每个配置跑同一压测,对比显存 + 吞吐 + 精度
for cfg in bf16 fp8 awq; do
  # ... 起对应 serve ...
  vllm bench serve --backend vllm --model $MODEL \
    --dataset-name sharegpt --dataset-path ./sharegpt.json \
    --num-prompts 500 --request-rate 8 | tee bench_$cfg.txt
  nvidia-smi --query-gpu=memory.used --format=csv >> bench_$cfg.txt
done
```

| 配置 | 70B 显存(/卡) | 相对吞吐 | 精度 | 适用 |
|---|---|---|---|---|
| BF16 | ~70GB(2卡) | 1.0× | 基线 | 卡多、要精度 |
| FP8(W8A8) | ~35GB(2卡) | ~1.6× | 接近无损 | H100/Blackwell 首选 |
| AWQ(W4A16) | ~35GB(单卡可) | ~1.0–1.2× | 略降 | 显存吃紧、卡少 |

```python
# 精度回退校验:量化前后跑同一组 prompt 比对(或上 lm-eval)
# pip install lm-eval; lm_eval --model local-completions \
#   --model_args base_url=http://localhost:8000/v1/completions,model=$MODEL \
#   --tasks gsm8k --num_fewshot 5
```

❌ 反模式:在 Ampere 卡上指望 FP8 W8A8 加速(不支持,退化或报错);或用 AWQ 只为「提吞吐」却不知它主要省显存。
✅ 正解:**H100/Blackwell 选 FP8(显存省一半、吞吐真涨)**;**卡少/显存吃紧选 AWQ(4-bit 省到极致)**;两者都必须用 126 的压测 + 精度评测确认回退可接受。

## 面试高频

- **「AWQ 和 FP8 部署上怎么选?」** FP8(W8A8)在 H100/Blackwell 有原生 Tensor Core,显存省一半且吞吐 ~1.6×,首选;AWQ(W4A16)4-bit 省显存最狠、卡少时用,但主要省显存、吞吐提升有限。
- **「为什么 AWQ 省显存却不一定提吞吐?」** W4A16 反量化回 16-bit 算 GEMM,计算路径仍 BF16;FP8 是 W8A8,GEMM 直接 8-bit 才真加速。
- **「FP8 有硬件门槛吗?」** W8A8 需 Hopper/Ada;Ampere/Turing 只能 W8A16(Marlin)。
- **「怎么验证量化没掉点?」** 量化前后同组 prompt 比对 + lm-eval 跑 gsm8k 等任务,确认精度回退在阈值内。
- **「`--quantization` 必须写吗?」** 预量化 checkpoint 的 config 多含元数据可省,但显式写更稳;在线 FP8 对 BF16 权重需显式 `--quantization fp8`。

## 关键事实

- AWQ:`--quantization awq`,W4A16,主省显存(~1/4),吞吐提升有限。
- FP8:`--quantization fp8`,W8A8,H100/Blackwell 原生,显存 ~1/2、吞吐 ~1.6×;Ampere 不支持 W8A8。
- 在线动态 FP8 可对 BF16 权重直接 `--quantization fp8`,无需预量化文件。
- 70B:BF16 ~140GB → FP8 ~70GB → AWQ ~35GB。
- 必须用 126 压测对比吞吐 + lm-eval 验精度回退。
