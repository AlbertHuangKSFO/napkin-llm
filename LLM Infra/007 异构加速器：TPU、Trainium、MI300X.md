[[007 异构加速器：TPU、Trainium、MI300X|异构加速器]] 指 NVIDIA GPU 之外、专为大模型训练/推理设计的 AI 芯片——Google TPU、AWS Trainium、AMD MI300X——它们用不同的计算范式(脉动阵列 / NeuronCore / CDNA chiplet)、显存与互联,去抢同一块「搬权重」的蛋糕。

## 直觉:三种「工厂」造同一种货
把矩阵乘想成流水线生产。**NVIDIA GPU** 像一座全能代工厂:啥订单都接(通用 SIMT + Tensor Core),工人多、工具(CUDA)最全,但管理开销大。**Google TPU 脉动阵列**像一条专用冲压线:模具(权重)先固定在机床网格里,板材(激活)像传送带一样脉冲式流过,每个工位冲一下传给下一个——产线简单、单位能耗低,但只能冲规整件。**AWS Trainium** 是亚马逊为自家云定制的产线,绑 EC2 卖性价比。**AMD MI300X** 则是「超大仓库车间」:192GB HBM 让单台机器一次性塞下别人要两台才装得下的料(见 [[011 单卡能放多大模型：参数与 KV 显存预算|显存预算]])。

## 小数字例子:单卡能装多少 70B?
70B 模型 fp16 权重 ≈ 140GB。
- **H100 80GB / TPU v6e 32GB**:单卡装不下,必须张量并行切到多卡(靠 [[008 片内互联：NVLink、NVSwitch、NVLink-C2C|NVLink]])。
- **MI300X 192GB**:140GB 权重 + KV-Cache 还能单卡塞下 → 省掉跨卡通信,这是 AMD 的卖点。
- decode 阶段是带宽限:每生成 1 token 要把 140GB 权重整读一遍。MI300X 5.3TB/s → 理论上限 5300/140 ≈ **37.8 tok/s**(单流);H200 4.8TB/s、B200 8TB/s 同理换算。带宽直接定吞吐(见 [[004 算力 vs 带宽：Roofline 与算术强度|Roofline]])。

## 原理:脉动阵列的复用率
脉动阵列把 $C = A \times B$ 摊到 $n \times n$ 个 PE 上。权重驻留后,每个权重元素被复用 $O(n)$ 次而非每次从 HBM 重读,算术强度被人为拉高:

$$\text{算术强度} = \frac{\text{FLOPs}}{\text{Bytes from HBM}} \uparrow \quad\Rightarrow\quad \text{越靠近 compute-bound,能效越高}$$

TPU v6e 的 MXU 是 $256\times256$(前代 $128\times128$),一拍吞 $256^2$ 个 MAC;但矩阵维度凑不满 256 时,边缘 PE 空转,利用率掉。

## 图
![[hw-异构加速器架构对比.svg]]

![[hw-脉动阵列数据流.svg]]

![[hw-007加速器显存带宽对比.svg]]

## 已验证规格表(2025–2026)

| 芯片 | 计算范式 | HBM 容量 | HBM 带宽 | 片内/片间互联 | 软件栈 |
|---|---|---|---|---|---|
| NVIDIA B200 | SIMT + Tensor Core | 192GB HBM3e | ~8 TB/s | NVLink 5 = 1.8 TB/s/卡 | CUDA |
| Google TPU v6e (Trillium) | 脉动阵列 256×256 MXU | 32GB | ~1.6 TB/s | ICI 3.2 Tbps,2D 环面,256 卡/pod | JAX / XLA |
| AWS Trainium2 | 8× NeuronCore-V3 | 96GB HBM3 | 2.9 TB/s | NeuronLink(16 卡/实例,64 卡 UltraServer) | Neuron SDK |
| AMD MI300X | CDNA3,8× XCD chiplet,304 CU | 192GB HBM3 | 5.3 TB/s | Infinity Fabric 128 GB/s/卡 | ROCm 6 |

## 代码/配置:框架可移植性的现实

```python
# ❌ 以为换芯片只是改 device 字符串——CUDA-only 代码到 TPU/Trainium 直接挂
model.to("cuda")               # MI300X 上 PyTorch+ROCm 可跑;TPU 不认 CUDA
torch.cuda.amp.autocast()      # Trainium 没有 cuda 命名空间

# ✅ TPU:走 JAX/XLA,设备无关,由 XLA 编译器决定布局
import jax, jax.numpy as jnp
@jax.jit                       # XLA 把计算图编译成脉动阵列指令
def fwd(w, x): return x @ w

# ✅ Trainium:torch-neuronx 编译,显式 trace
import torch_neuronx
neuron_model = torch_neuronx.trace(model, example_inputs)

# ✅ MI300X:ROCm 几乎是 CUDA 的 drop-in,但 kernel 性能/算子覆盖需实测
# HIP 把 cuda* API 映射成 hip*,大部分 PyTorch 代码无改动跑通
```

## 面试高频
- **TPU 脉动阵列 vs GPU Tensor Core,本质差别?** 脉动阵列权重驻留、数据流过固定 PE 网格、控制逻辑极简 → 能效高但只擅长规整 matmul;Tensor Core 是通用 SIMT 流水线里的矩阵加速单元,灵活性高、生态全(CUDA)。
- **为什么大家追着 NVIDIA 跑还有人买 TPU/Trainium/MI300X?** 性价比 + 供给。CUDA 是护城河,但 MI300X 192GB 单卡装更大模型、TPU/Trainium 绑自家云垂直整合(XLA/Neuron 编译器)压成本、避开 NVIDIA 产能瓶颈。
- **推理选型先看什么?** 先看 HBM 带宽(定 decode 吞吐),再看显存容量(定能否单卡装下、并发上限),最后看互联(定能否高效张量并行)。算力 FLOPS 在 decode 阶段往往不是瓶颈(见 [[010 显存墙与 LLM 推理的本质约束|显存墙]])。
- **ROCm 能无痛替代 CUDA 吗?** HIP 提供 drop-in API 映射,PyTorch 多数模型直接跑;但自定义 kernel、最新算子、性能调优仍有缺口,生态成熟度是最大变量。

## 关键事实(2025–2026)
- **TPU v6e Trillium**:32GB HBM、~1.6 TB/s、256×256 MXU、ICI 3.2 Tbps、2D 环面 256 卡/pod;较 v5e 算力 4.7×、能效 +67%。
- **AWS Trainium2**:8× NeuronCore-V3,96GB HBM3、2.9 TB/s;Trn2 实例 16 卡(1.5TB HBM、46 TB/s),Trn2 UltraServer 64 卡(83.2 PFLOPS FP8)。Trainium3(2025 末)144GB HBM3e、4.9 TB/s、NeuronLink-v4 2 TB/s。
- **AMD MI300X**:192GB HBM3、5.325 TB/s、304 CU、8× XCD(CDNA3)、Infinity Fabric 128 GB/s/卡、ROCm 6。
- **范式总结**:脉动阵列(TPU)= 能效/规整 matmul;chiplet 大显存(MI300X)= 单卡装得多;通用 GPU(NVIDIA)= 生态王者。三者都在抢「带宽限的搬权重」生意。
