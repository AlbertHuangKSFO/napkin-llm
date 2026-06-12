[[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]:把高精度浮点(FP16/BF16)的权重或激活,用 $q=\text{round}(x/s)+z$ 映射到低位宽整数(常 int8/int4)以省显存、提吞吐;两个核心参数是 **scale $s$**(每个整数格子代表多宽的浮点)和 **zero-point $z$**(浮点 0 落在哪个整数)。三个关键设计维度:**对称 / 非对称**($z$ 是否为 0)、**位宽**(int8/int4)、**粒度**(per-tensor / per-channel / per-group,即多少个数共享一组 $s,z$)。它是 [[093 PTQ 与 QAT|PTQ/QAT]]、[[094 LLM.int8 与离群值|LLM.int8]]、[[095 GPTQ|GPTQ]]、[[096 AWQ 与 SmoothQuant|AWQ]] 的共同地基。

## 直觉:把连续的浮点轴吸附到等间距的整数格点

一个 FP16 权重是连续值(几乎能取任意小数)。int8 只有 256 个格子($-128\sim127$)。量化就是给浮点轴画一把**等间距的尺子**:每格宽 $s$(scale),把每个浮点数**四舍五入到最近的格点**,记下它的整数编号 $q$。用的时候再 $\hat x=s(q-z)$ **反量化**近似还原。

为什么省:存储从 16 位降到 8 位(省一半)甚至 4 位(省 3/4),[[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存]]和访存带宽直接下降;而 LLM 推理是**访存受限**([[078 推理算力、吞吐与延迟、Roofline|Roofline]] 的 memory-bound 区),少搬数据就直接快。

代价是**量化误差**:四舍五入丢掉的零头 $|x-\hat x|$。设计量化方案,本质就是**在固定位宽下把误差压到最小**。三个旋钮:

- **对称 / 非对称**:尺子的零点要不要偏移,去贴合数据的真实分布。
- **位宽**:格子越多(int8 比 int4)越准但越占。
- **粒度**:一把尺子量多少个数。一个离群值会把尺子撑得很宽、让其余数挤在几个格子里——所以**分得越细**(每列、每组各用一把尺子),越能隔离离群值、越准,但要存的 $s,z$ 越多。

![[quant-浮点到定点格点.png]]

## 例子:把 [-3.2, 1.7] 量化到 int8(手算)

取一组权重 $x=[-3.2,\,-0.6,\,0.9,\,1.7]$,量化到 int8。

**(A)对称量化**($z=0$,范围 $[-127,127]$)。scale 由最大绝对值定:

$$
s=\frac{\max|x|}{127}=\frac{3.2}{127}\approx0.0252
$$

逐个 $q=\text{round}(x/s)$:

| $x$ | $x/s$ | $q$(round,clip) | $\hat x=s\cdot q$ | 误差 $x-\hat x$ |
|---|---|---|---|---|
| $-3.2$ | $-127.0$ | $-127$ | $-3.200$ | $0.000$ |
| $-0.6$ | $-23.8$ | $-24$ | $-0.605$ | $+0.005$ |
| $0.9$ | $35.7$ | $36$ | $0.907$ | $-0.007$ |
| $1.7$ | $67.5$ | $67$ | $1.688$ | $+0.012$ |

注意 $1.7$ 只映射到 $67$,远没到 $127$——因为正侧最大才 $1.7$,而对称量化按 $\pm3.2$ 撑开尺子,**正半轴一半格子被浪费**了。这正是非对称的动机。

**(B)非对称量化**(uint8,范围 $[0,255]$,贴合真实 $[\min,\max]=[-3.2,1.7]$):

$$
s=\frac{\max-\min}{255}=\frac{1.7-(-3.2)}{255}\approx0.01922,\quad z=\text{round}\Big(\frac{-\min}{s}\Big)=\text{round}(166.5)=167
$$

$q=\text{round}(x/s)+z$ 得 $[0,\,136,\,214,\,255]$,反量化 $\hat x=s(q-z)$ 得 $[-3.209,\,-0.596,\,0.903,\,1.691]$。整段 $[-3.2,1.7]$ **铺满了全部 256 个格子**,平均误差比对称更小。

**(C)同一组量化到 int4(只有 16 个格子,手算)**。位宽降到 4-bit,误差会明显放大,正好看清「为什么低比特更需要细粒度/补偿」。对称 int4 范围 $[-7,7]$:

$$
s=\frac{\max|x|}{7}=\frac{3.2}{7}\approx0.457
$$

| $x$ | $x/s$ | $q$(round,clip 到 $[-7,7]$) | $\hat x=s\cdot q$ | 误差 |
|---|---|---|---|---|
| $-3.2$ | $-7.00$ | $-7$ | $-3.200$ | $0.000$ |
| $-0.6$ | $-1.31$ | $-1$ | $-0.457$ | $+0.143$ |
| $0.9$ | $1.97$ | $2$ | $0.914$ | $-0.014$ |
| $1.7$ | $3.72$ | $4$ | $1.829$ | $-0.129$ |

对比 int8 时 $-0.6$ 的误差只有 $0.005$,int4 下飙到 $0.143$——**位宽每降 1 bit,格距翻倍、误差大致翻倍**。这就是为什么 int8 朴素量化几乎无损,而 int4 必须靠 [[095 GPTQ|GPTQ]] 误差补偿或 [[096 AWQ 与 SmoothQuant|AWQ]] 关键权重保护才能保精度。

![[quant-对称非对称.png]]

## 原理:量化 / 反量化公式与三个维度

**量化与反量化**:

$$
q=\text{clip}\big(\text{round}(x/s)+z,\ q_{\min},\ q_{\max}\big),\qquad \hat x=s\,(q-z)
$$

- **scale** $s$:每个整数格代表的浮点宽度,$s=\dfrac{\text{浮点范围}}{\text{整数范围}}$。
- **zero-point** $z$:让浮点 $0$ 精确映到某个整数(对量化 padding、ReLU 的 0 很重要),它必须是整数。

**1)对称 vs 非对称**。

- **对称**:$z=0$,范围取 $\pm\alpha$($\alpha=\max|x|$),$s=\alpha/(2^{b-1}-1)$。只存一个 $s$,且矩阵乘里不用处理 $z$ 的交叉项,**计算更省**。适合**零附近对称分布**的数据——典型是**权重**。
- **非对称**:$z\neq0$,范围取真实 $[\min,\max]$,$s=\dfrac{\max-\min}{2^b-1}$,$z=q_{\min}-\text{round}(\min/s)$。能贴合**偏斜分布**(如 ReLU/GELU 后非负的**激活**),不浪费格子;代价是矩阵乘展开时多出含 $z$ 的修正项。

**2)粒度**(共享一组 $s,z$ 的范围,见下图):

- **per-tensor**:整个张量一个 $s,z$。最省、最快,但**一个离群值就把尺子撑宽**,其余值挤在少数格子里 → 误差大。
- **per-channel**(逐通道,常是权重矩阵的逐列/逐输出通道):每个通道一组 $s,z$。各通道独立、能隔离某列的大值,**权重量化的标配**。
- **per-group**(逐组,如每连续 128 个元素一组):比 per-channel 更细,int4 低比特量化常用(如 [[095 GPTQ|GPTQ]]、[[096 AWQ 与 SmoothQuant|AWQ]] 默认 group size 128),精度/开销折中。

粒度越细,量化误差越小,但要存的 $s,z$ 元数据越多(per-tensor 存 1 组,per-channel 存「通道数」组)。

**元数据开销算一笔账(为什么 group size 128 是甜点)**。设权重 int4(0.5 字节/参),per-group 每组 128 个权重共享 1 个 FP16 scale(2 字节)。scale 平摊位宽 $=\frac{16\,\text{bit}}{128}=0.125$ bit/参,相对 4 bit 主体只多 3%——几乎免费却换来明显更准。若 group size 取 32(更细),scale 开销升到 $0.5$ bit/参(主体的 12.5%),收益递减;取 per-channel(一行共享一个 scale,行很长时近似 per-tensor)则元数据几乎为零但抗离群差。所以 **128 是「精度↑ vs 元数据↑」的经典折中**,GPTQ/AWQ 都默认它。

![[quant-粒度.png]]

**3)为何 LLM 激活比权重难量化**:权重分布平稳、近似 [[24 常见分布(高斯、伯努利、类别)|高斯]],per-channel 对称量化就很好;而激活里存在**少数维度的离群值**(数值比其他大几十倍),一个 per-tensor 尺子被它撑爆 → 这就是 [[094 LLM.int8 与离群值|LLM.int8]]、[[096 AWQ 与 SmoothQuant|SmoothQuant]] 要专门处理的难点。

**4)量化误差的理论估计(均匀量化噪声)**。把舍入误差近似看成在 $[-s/2,s/2]$ 上均匀分布的噪声,其方差 $=\frac{s^2}{12}$(均匀分布方差公式)。所以**量化噪声功率 $\propto s^2$**,而 $s=\frac{\text{范围}}{2^b-1}$ —— 位宽 $b$ 每加 1,$s$ 减半,噪声功率降 4 倍,即 **每多 1 bit,信噪比约 +6 dB**(经典 ADC 结论)。这定量解释了:① int8→int4 误差大致 $\times16$(差 4 bit,$s\times16$);② 离群值把范围撑大 → $s$ 变大 → 噪声功率平方级恶化,所以隔离离群值收益巨大。

**5)静态 vs 动态量化(激活)**。权重离线可见,scale 总是**静态**(量化时定死)。激活依赖输入,有两种:**静态**——用校准集预先统计每层激活范围、定死 $s,z$,推理时不再算(快,但校准分布偏了就不准);**动态**——推理时**按每个 batch 现算**激活的 $\min/\max$ 定 $s,z$(更准、自适应输入,略慢)。LLM 推理常用「权重静态 per-channel/group + 激活动态 per-token」的组合(per-token 动态对激活离群更鲁棒)。

**离群激活下静态 vs 动态 per-token(int8 对称手算)**。设某 token 的激活向量 $x=[0.5,\,1.2,\,40.0]$,第三维是离群值。校准集只见过常规激活、$\max|x|=4.0$,于是**静态**定死 $s_{\text{stat}}=4.0/127\approx0.0315$:运行时 $40/s_{\text{stat}}\approx1270$ 被 **clip 到 127**,反量化回 $127\times0.0315=4.0$,离群维误差高达 $|40-4|=36$,信息全毁(常规维 $0.5,1.2$ 倒很准,误差 $\sim0.004$)。改用 **动态 per-token**,scale 现算自本 token 的 $\max=40$:$s_{\text{dyn}}=40/127\approx0.315$,$40\to127\to40.0$,离群维**误差 0**;代价是常规维变粗($0.5\to q{=}2\to0.63$,误差 $0.13$)。这就是「激活离群 → 静态会饱和丢大值,动态 per-token 用本 token 范围兜住离群」的定量来由,也呼应 [[094 LLM.int8 与离群值|LLM.int8]]/[[096 AWQ 与 SmoothQuant|SmoothQuant]] 为何还要进一步把离群维单独拎出来。

## 代码:对称 / 非对称 int8 量化(❌ vs ✅)

```python
import torch

x = torch.tensor([-3.2, -0.6, 0.9, 1.7])

# ❌ 错:对「分布偏斜」的数据硬用对称量化,正半轴格子大量浪费,误差偏大
def quant_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1                 # 127
    s = x.abs().max() / qmax                   # 只看 max|x|,z=0
    q = torch.clamp(torch.round(x / s), -qmax, qmax)
    x_hat = s * q                              # 反量化
    return q.int(), x_hat, s

# ✅ 对:非对称量化贴合真实 [min,max],格子铺满,误差更小
def quant_asymmetric(x, bits=8):
    qmin, qmax = 0, 2 ** bits - 1              # uint8: [0,255]
    mn, mx = x.min(), x.max()
    s = (mx - mn) / (qmax - qmin)              # scale
    z = torch.round(qmin - mn / s)             # zero-point(整数)
    q = torch.clamp(torch.round(x / s) + z, qmin, qmax)
    x_hat = s * (q - z)                        # 反量化
    return q.int(), x_hat, s, z

qs, xs, _ = quant_symmetric(x)
qa, xa, *_ = quant_asymmetric(x)
print("对称  q:", qs.tolist(), " 误差:", (x - xs).abs().mean().item())
print("非对称 q:", qa.tolist(), " 误差:", (x - xa).abs().mean().item())  # 通常更小

# per-channel(逐列)对称量化:权重矩阵的标配,每列单独一个 scale
def quant_per_channel(W, bits=8):              # W: [out, in],逐 out 通道
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True) / qmax   # 每行(输出通道)一个 scale
    q = torch.clamp(torch.round(W / s), -qmax, qmax)
    return q.int(), s                          # 比 per-tensor 抗离群值
```

## 选型卡:量化粒度与对称性怎么挑

| 场景 | 选什么 | 为什么 |
|---|---|---|
| 权重(分布零对称、平稳) | 对称 + per-channel | $z=0$ 省参、乘法简单;逐列 scale 抗离群 |
| int4 权重(要更准) | per-group(group=128) | 隔离离群更细,scale 仅占 ~0.125 bit/参(GPTQ/AWQ 默认) |
| 激活(有离群、偏斜) | 非对称 + per-token 动态 | 贴合 $[\min,\max]$、动态 scale 抗逐 token 离群 |
| 追求最省/硬件最简 | per-tensor | 元数据最少,但对离群最脆,易掉点 |
| 激活离群严重、要 W8A8 | 先 [[096 AWQ 与 SmoothQuant\|SmoothQuant]] 迁移再量化 | 把激活难度搬给权重,两边都能 int8 |

## 面试高频

- **scale 和 zero-point 各是什么?** scale 是每个整数格代表的浮点宽度($s=$ 浮点范围/整数范围);zero-point 是「浮点 0 对应哪个整数」,保证 0 精确可表示,**必须是整数**。
- **对称和非对称怎么选?** 数据零对称(权重)用对称($z=0$,省一个参数、乘法更简单);数据偏斜(ReLU/GELU 后的激活)用非对称(贴合 $[\min,\max]$、不浪费格子)。
- **per-tensor / per-channel / per-group 区别?为什么 LLM 偏好细粒度?** 越细 = 共享一组 $s,z$ 的数越少 → 越能隔离离群值、误差越小,但元数据越多。权重常 per-channel;int4 常 per-group(128);per-tensor 最省但对离群值最脆。
- **量化误差从哪来?如何减小?** 来自 $\text{round}$ 的舍入和 $\text{clip}$ 的截断。减小:提位宽(int8>int4)、细粒度、非对称贴合分布、或专门处理离群值([[094 LLM.int8 与离群值|混合精度]]、[[096 AWQ 与 SmoothQuant|SmoothQuant]])。
- **为什么量化能加速 LLM 推理?** LLM 解码是**访存受限**([[078 推理算力、吞吐与延迟、Roofline|memory-bound]]),权重位宽减半即搬运量减半,直接提吞吐;int8 还能用更快的整数 Tensor Core。
- **激活为什么比权重难量化?** 权重分布平稳;激活有少数**离群维度**值极大,会把 per-tensor 尺子撑爆,需 LLM.int8 / SmoothQuant 等手段。
- **静态 vs 动态量化(激活)?** 静态:用校准集预先定好激活的 $s,z$(快,推理时不算);动态:推理时按每个 batch 现算激活的 $s,z$(更准,略慢)。权重一般是静态的;LLM 常用激活 per-token 动态以抗离群。
- **每多 1 bit 误差降多少?** 量化噪声功率 $\approx s^2/12\propto s^2$,位宽加 1 → $s$ 减半 → 噪声功率降 4 倍(信噪比约 +6 dB)。所以 int8→int4(差 4 bit)误差大致 $\times16$;离群值撑大范围则平方级恶化误差,故隔离离群值收益极大。
- **group size 选多大?为什么是 128?** 越小越准但 scale 元数据越多。int4 + group 128 时 scale 只占 0.125 bit/参(主体的 3%),精度/开销最优;group 32 升到 0.5 bit/参收益递减。GPTQ/AWQ 默认 128。
- **量化用于推理 vs 混合精度用于训练,有何区别?** [[064 混合精度 FP16、BF16 与 FP8|混合精度]] 是训练时用 FP16/BF16 算、加速收敛、保 FP32 主权重;量化是推理时把已训好的权重压成低比特整数省显存/带宽。一个保训练数值稳定,一个压部署体积,目标不同。
- **int8 对称为什么常取 $[-127,127]$ 而非 $[-128,127]$?** 留 $-128$ 不用,使正负范围严格对称($\pm127$),便于硬件对称处理与避免 $-128$ 的特殊符号问题;牺牲一个格子换对称性。

## 关键事实

- 通用量化公式 $q=\text{round}(x/s)+z$、对称/非对称、per-tensor/channel 的系统综述见 Gholami et al.《A Survey of Quantization Methods for Efficient Neural Network Inference》(2021,arXiv:2103.13630)与 Nagel et al.《A White Paper on Neural Network Quantization》(Qualcomm,2021,arXiv:2106.08295)。
- int8 推理与硬件支持的奠基工作:Jacob et al.《Quantization and Training of Neural Networks for Integer-Arithmetic-Only Inference》(Google,2018,arXiv:1712.05877),提出对称/非对称、per-tensor/per-channel 的工程框架。
- LLM 权重常用 per-channel 或 per-group(group size 128 是 [[095 GPTQ|GPTQ]]/[[096 AWQ 与 SmoothQuant|AWQ]] 的常见默认);int8 权重 + int8 激活的难点正是激活离群值(见 [[094 LLM.int8 与离群值|LLM.int8]],Dettmers 2022,arXiv:2208.07339)。
- 反量化恒等式:$\hat x=s(q-z)$;对称时 $z=0$ ⇒ $\hat x=sq$;非对称 uint8 范围 $[0,255]$,对称 int8 范围常取 $[-127,127]$(留 $-128$ 不用以保持对称)。
- 关联:[[064 混合精度 FP16、BF16 与 FP8|混合精度(训练用)]] vs 量化(推理用)、[[093 PTQ 与 QAT|PTQ/QAT]]、[[078 推理算力、吞吐与延迟、Roofline|Roofline]]、[[24 常见分布(高斯、伯努利、类别)|分布]]。
