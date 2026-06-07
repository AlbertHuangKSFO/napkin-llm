[[084 引擎里的量化集成：AWQ、GPTQ、FP8 加载|引擎里的量化集成：AWQ、GPTQ、FP8 加载]]讲**工程落地的最后一公里**:量化算法产出的 checkpoint 有不同格式,推理引擎(vLLM、TensorRT-LLM)怎么**识别格式、选对 kernel、把它跑起来**。算法本身回链 [[LLM/095 GPTQ|GPTQ]]、[[LLM/096 AWQ 与 SmoothQuant|AWQ]]、[[LLM/098 FP8 训练推理与 AQLM 极低比特|FP8]];这里只关心**部署侧:checkpoint 格式、`--quantization` 开关、kernel 支持矩阵**。

## 直觉

量化不是"模型权重变小了"就完事——引擎得知道**这堆字节是什么格式、用哪个内核去算**。把它想成快递:量化工具(AutoAWQ、GPTQModel、LLM Compressor)把模型"打包"成某种格式(AWQ、GPTQ、`compressed-tensors`、FP8),包裹上贴了标签(`config.json` 里的 `quantization_config`);引擎读标签,**自动**挑选对应的拆包工具([[027 量化内核：W4A16、W8A8 GEMM|量化 GEMM kernel]],如 Marlin、Machete、cutlass FP8)。标签不对、或硬件没这个内核,要么报错要么**退化到慢速反量化路径**。所以"能不能跑得快"= checkpoint 格式 × 引擎 kernel × 硬件,三者要对齐。

## 例子

同一个 70B,部署路径按硬件和并发分流:

- **AWQ W4A16**:`vllm serve model-AWQ --quantization awq`,引擎读到 AWQ config → 走 **Marlin** 内核(Ampere+);单卡放大模型、低并发首选。
- **GPTQ W4A16**:同理 `--quantization gptq`,也走 Marlin;质量略低于 AWQ 但生态广。
- **FP8 W8A8**:`--quantization fp8`,Hopper/Blackwell 走原生 FP8 cutlass 内核;高吞吐近无损。社区常用 **LLM Compressor** 产出 `compressed-tensors` 格式,vLLM 原生加载。
- **踩坑**:把 AWQ checkpoint 丢到没 Marlin 的旧引擎/硬件 → 退化到逐层反量化,慢且白量化。

## 原理

引擎加载量化模型的流程(见图):**checkpoint 自带量化元信息** → 引擎读 `quantization_config`(方法、比特、group_size、scale 位置)→ **匹配 kernel** → 加载量化权重 + scale。`--quantization` 多数时候可省略(引擎从 config 自动推断),显式指定用于覆盖或强制。

$$
\text{跑得快} = \underbrace{\text{格式被识别}}_{\text{config 标签}} \times \underbrace{\text{有对应 kernel}}_{\text{Marlin/cutlass}} \times \underbrace{\text{硬件原生}}_{\text{FP8 需 Hopper+}}
$$

格式速记:**AWQ / GPTQ**(W4A16,group-wise INT4 + per-group scale)、**compressed-tensors**(vLLM/LLM Compressor 通用容器,可装 W4A16/W8A8/FP8/KV scale)、**FP8**(权重激活 E4M3 + per-tensor scale)。TensorRT-LLM 走**构建期(build-time)**量化:`trtllm-build` 把量化烘进引擎,支持 per-group scale 实现 GPTQ/AWQ、以及 FP8。kernel 层:**Marlin/Machete** 是 vLLM 的高效 W4A16 GEMM,**cutlass FP8** 给 Hopper/Blackwell。

## 图

![[sq-引擎量化集成.png]]

## 代码

```bash
# vLLM:多数情况 --quantization 可省略,引擎从 checkpoint config 自动识别
vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w4a16   # 自动 AWQ/GPTQ
vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-FP8               # 自动 FP8

# 显式指定(覆盖/强制 kernel)
vllm serve model-AWQ --quantization awq

# 用 LLM Compressor 离线产出 compressed-tensors(W8A8 INT8 示例)
# from llmcompressor.transformers import oneshot
# oneshot(model=..., recipe="W8A8", dataset=calib_ds)  → 保存即可被 vLLM 加载
```

❌ 误区:以为"量化好的模型放进任何引擎/卡都能享受加速"。
✅ 正解:加速绑定 **格式 × kernel × 硬件**。AWQ 需 Marlin 内核、FP8 需 Hopper+;格式或硬件不匹配会退化到慢速反量化路径,等于白量化。部署前确认引擎版本支持该格式的 kernel。


![[sq-084格式核硬件矩阵.png]]

## 面试高频

- **Q:引擎怎么知道一个 checkpoint 是量化的、用什么 kernel?** 读 `config.json` 的 `quantization_config`(方法/比特/group_size),自动匹配 kernel;`--quantization` 用于显式覆盖。
- **Q:AWQ 和 GPTQ 在引擎里跑有区别吗?** 格式不同但都是 W4A16,vLLM 多走 Marlin/Machete;AWQ 质量通常略优,生态都很广。
- **Q:vLLM 和 TensorRT-LLM 加载量化的差别?** vLLM **运行期**加载量化 checkpoint(灵活);TensorRT-LLM **构建期**把量化烘进引擎(`trtllm-build`,延迟更优但需重建)。
- **Q:compressed-tensors 是什么?** vLLM/LLM Compressor 的通用量化容器格式,可统一表达 W4A16/W8A8/FP8/KV scale。

## 关键事实

- 2025:**LLM Compressor + compressed-tensors** 是 vLLM 生态的主流量化产出链;AWQ/GPTQ 仍是社区 W4A16 主力,FP8 是高吞吐默认。
- vLLM 高效 W4A16 内核:**Marlin、Machete**(Ampere+);FP8 走 cutlass(Hopper+)。
- TensorRT-LLM 通过 per-group scale 实现 GPTQ/AWQ,并原生支持 FP8(Hopper/Blackwell),量化在**构建期**完成。
- 格式与硬件不匹配会**静默退化**到慢路径——部署前务必验证 kernel 命中。
