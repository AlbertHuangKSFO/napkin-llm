[[034 KV 量化部署：FP8、INT8 KV|KV 量化部署：FP8、INT8 KV]]把 [[LLM/102 KV-Cache|KV-Cache]] 从 BF16(2 字节)**降精度存成 FP8 或 INT8(1 字节)**,每 token KV 字节**减半**,同显存能塞约 2× 的上下文/并发;注意力计算时再反量化回高精度,所以**只动存储、不动数学**。它是 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]]里"降字节 $b$"那个旋钮的工程落地,与 [[LLM/103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV 优化]]的 GQA、[[030 PagedAttention 深入：KV 当虚拟内存|分页]]、[[032 前缀缓存：RadixAttention 树结构|前缀缓存]]**正交可叠加**。

## 直觉

KV 显存账 $M_\text{KV}=2 L H_\text{kv} d_h S B \cdot b$ 里,$b$ 是每元素字节。BF16 的 $b=2$,FP8/INT8 的 $b=1$ → KV **直接减半**。工业类比:仓库里的货(KV)本用高清照片(BF16)存档,改用压缩图(FP8)——分辨率略降但仓库省一半,要用时再放大(反量化)做计算。因为 [[LLM/103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|decode 是访存受限]],搬的字节少一半理论上还能更快;但实测收益看引擎(见下)。关键:量化只压**存储**,$q\cdot k$ 仍在反量化后的高精度域算,**注意力数学不变**(近似来自精度损失,不来自算法改动)。

## 例子

[[015 KV-Cache 的显存账(逐层手算)|Llama-3 70B]](80 层、8 KV 头、head_dim 128):
- **BF16**:每 token $2\cdot80\cdot8\cdot128\cdot2=320$ KB;32k 上下文单序列 **10 GB**。
- **FP8**($b=1$):每 token **160 KB**;32k 单序列 **5 GB** → 同一张 80GB 卡能塞**约 2× 并发或上下文**。

代价:FP8 E4M3 动态范围 ±240,需配 FP32 scale;长上下文任务可能掉一点精度,要评测。INT8 同样减半,但需更细的 scale 才稳。

## 原理

对称量化:$x_q=\text{round}(x/s)$,反量化 $\hat x = x_q\cdot s$,$s$ 是 scale。scale 粒度决定精度:

$$
s_{\text{per-tensor}}=\frac{\max|x|}{q_\max}\quad\text{(1 个 scale)}\qquad
s_{\text{per-channel/head}}=\frac{\max_c|x_{:,c}|}{q_\max}\quad\text{(每通道/头一个)}
$$

**粒度越细越准、开销越大**:per-tensor 最省但易被**离群通道**拉爆;per-token / per-channel / per-head 抓住离群值、精度更好。**校准(calibration)**在目标数据上统计求 scale,进一步降损。格式:**FP8 E4M3**(精度优先,vLLM 默认 KV 格式)、**FP8 E5M2**(范围大精度低,少用);**INT8** 定点,vLLM 当前**不支持 KV INT8**(只支持 FP8),TensorRT-LLM 支持 FP8/INT8 KV。长上下文掉精度可用 **FP32 两级累加**缓解。

## 图

![[kv-量化对比.png]]

![[kv-034字节减半手算.png]]

## 代码

```python
from vllm import LLM
# ✅ FP8 E4M3 KV：每元素 1 字节，KV 显存减半；calibrate 出的 scale 随权重加载
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",          # 或 "fp8_e4m3"；存储 FP8，计算反量化回高精度
    # 经校准的 checkpoint 会带 per-tensor/per-head 的 KV scale，长上下文更稳
)

# 量化只动存储，不动注意力数学（示意）
def quant_fp8(x, s):          # s = scale（per-tensor 或 per-head）
    return (x / s).to(torch.float8_e4m3fn)        # ✅ 存 1 字节
def attn(q, k_fp8, v_fp8, sk, sv):
    k = k_fp8.to(torch.bfloat16) * sk             # ✅ 反量化回高精度再算
    v = v_fp8.to(torch.bfloat16) * sv
    return softmax(q @ k.transpose(-1, -2)) @ v   # 数学与 BF16 KV 相同，仅有量化误差

# ❌ 反例 1：vLLM 里指定 kv_cache_dtype="int8" —— 当前不支持 KV INT8（只 FP8）；INT8 KV 走 TRT-LLM
# ❌ 反例 2：长上下文 + per-tensor scale 不校准 —— 离群通道掉精度，必须评测质量回归
# ❌ 反例 3：以为 FP8 KV 必然提吞吐 —— vLLM 主要省显存，prefill 重时吞吐可能略降；TRT-LLM 才明显提速
```

`❌` 三个部署坑:**(1)** vLLM KV 量化目前只有 **FP8**,要 INT8 KV 用 TensorRT-LLM;**(2)** 长上下文 + 粗粒度 scale 不校准会掉精度,**必须评测**;**(3)** FP8 KV 在 vLLM 主要是**省显存**(从而更大 batch),吞吐未必直接涨。

## 选型卡:KV 量化格式与引擎怎么选

| 场景 | 选什么 | 为什么 |
|---|---|---|
| vLLM 上想省 KV 显存、塞更大 batch | **FP8 E4M3**(`kv_cache_dtype="fp8"`) | vLLM 默认且只支持 FP8;精度优先、配 FP32 scale,主省显存 |
| 要 INT8 KV 或要 KV 量化直接提吞吐 | **TensorRT-LLM**(FP8/INT8 KV) | vLLM 不支持 INT8 KV;TRT-LLM 的 FP8/INT8 KV 才明显提吞吐 |
| 长上下文、怕掉精度 | **细粒度 scale + 校准**(per-head/per-channel) | per-tensor 易被离群通道拉爆;校准 + FP32 两级累加缓解回归 |
| 短上下文 / 一般任务 | **per-tensor FP8**(最省开销) | 精度影响小,粗粒度 scale 够用、开销最低 |
| 已用 GQA/分页/前缀缓存 | **正交叠加 KV 量化** | 量化降字节、GQA 降头数、分页去碎片、前缀复用,互不冲突 |

## 面试高频

- **KV 量化省什么、原理?** 把 K/V 存成 FP8/INT8($b$ 从 2→1),KV 显存**减半**;只压存储,注意力在反量化后高精度算,**数学不变**。
- **FP8 E4M3 vs E5M2?** E4M3 精度高、范围 ±240(配 FP32 scale),KV 默认用它;E5M2 范围大精度低,少用。
- **scale 粒度怎么权衡?** per-tensor 最省但易被离群通道拉爆;per-token/per-channel/per-head 更准但开销大;**校准**进一步降损。
- **vLLM 支持 INT8 KV 吗?** 当前**只 FP8**(E4M3/E5M2);INT8 KV 走 **TensorRT-LLM**。
- **FP8 KV 一定提吞吐吗?** **不一定**。vLLM 主要省显存(→ 更大 batch);TRT-LLM 的 FP8/INT8 KV 才有明显吞吐提升。收益本质来自 [[LLM/103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|decode 访存受限]]的访存减半。
- **会掉精度吗?** 长上下文任务可能,需 FP32 两级累加 + 校准 + 质量评测;一般任务影响小。
- **和 GQA/分页/前缀缓存关系?** 全**正交可叠加**:量化降字节、GQA 降 KV 头、分页去碎片、前缀缓存复用。

## 关键事实

- KV 量化把每元素从 BF16(2 B)降到 **FP8/INT8(1 B)**,KV 显存**减半**,同显存约 **2×** 上下文/并发;只动存储,注意力反量化后高精度计算。
- vLLM 支持 **FP8 E4M3 / E5M2** KV(`kv_cache_dtype="fp8"`),**当前不支持 INT8 KV**;**per-tensor 默认**,正发展 per-channel/per-head,支持校准。
- TensorRT-LLM 支持 **FP8 与 INT8** KV,且 FP8/INT8 KV 吞吐有明显提升;vLLM 的 FP8 KV 主要省显存(prefill 重时吞吐可能略降)。
- 精度:长上下文可能回归,需 **FP32 两级累加** + 校准缓解(vLLM 文档/博客 2024–2026);与 [[LLM/019 GQA 分组查询注意力|GQA]]、[[030 PagedAttention 深入：KV 当虚拟内存|分页]]、[[032 前缀缓存：RadixAttention 树结构|前缀缓存]]正交叠加。
