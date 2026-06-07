[[097 NF4 与 QLoRA 4-bit|NF4 与 QLoRA 4-bit]] 是把基座权重压到 **4 位**再微调的省显存方案:用专为正态分布权重设计的 **NF4(4-bit NormalFloat)**数据类型存冻结基座,再叠一份 16-bit 的 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|QLoRA]] LoRA 旁路训练,配合**双重量化**与分页优化器,让 65B 模型能在单张 48GB GPU 上微调,精度接近 16-bit 全微调。

## 直觉

普通 4-bit 量化(INT4)把权重区间**均匀切 16 段**,每段一样宽。可是预训练后的权重**集中在 0 附近、呈钟形正态分布**——绝大多数值挤在中间,均匀分桶就浪费了:中间一个宽桶里塞了一大堆不同的值(精度不够),两侧的桶却几乎没人用(码字白给)。

NF4 换个思路:**按正态分布的分位数(quantile)来切桶**,让每个桶覆盖**等量的权重**。结果是中间密集区桶又多又窄(精度集中在最需要的地方),两侧稀疏区桶少而宽。对"零中心正态分布的数据",这是**信息论意义上最优**的 4-bit 编码。

**为什么「每桶等概率」就是最优?** 量化误差是「桶内的值被代表点替代」造成的;一个桶里落的权重越多、桶越宽,误差贡献越大。若按概率等分桶(等量数据/桶),则**每个桶承担的期望误差大致相等**,没有哪个桶「人满为患」拖累整体——这在该分布下最小化期望量化误差(即信息论最优)。均匀分桶相反:中间高频区一个宽桶塞了一大堆值(误差大),两侧低频区的桶几乎没人用(码字浪费)。**把码字预算花在数据密集的地方**,正是分位量化的核心思想。

QLoRA 把它和 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]] 拼起来:**基座权重用 NF4 冻结存储**(只占 1/4 显存,前向时临时反量化成 BF16 参与计算),**只训练那条 16-bit 的低秩 LoRA 旁路**。一句话:**用 NF4 把大模型"压扁"放进显存,再用一根细小的可训练旁路去微调它**。

## 例子

设某层 4 个权重 `[-0.9, -0.1, 0.05, 0.8]`(典型钟形,中间密)。

- **INT4 均匀量化**:在 `[-1, 1]` 均匀切 16 格,每格宽 `2/15≈0.133`。`-0.1` 和 `0.05` 都落进靠中心的同一/相邻格,**几乎不可区分**——可它俩恰恰是出现最多的那类值,误差被放大。
- **NF4**:量化点是标准正态的 16 个分位点(约 `[-1, -0.70, -0.53, …, 0, …, 0.53, 0.70, 1]`,中间密两头疏)。`-0.1`、`0.05` 各自落到中心附近**不同的窄桶**,区分得开;`-0.9`、`0.8` 落到两侧宽桶,反正那里值少、误差影响小。

**块量化 + 双重量化的位宽账**:NF4 以**块大小 64** 为单位,每块存 1 个 FP32 缩放因子 scale。光这一项 scale 平摊下来是 `32/64 = 0.5 bit/参数`。**双重量化**把这些 scale 再用 8-bit 量化一遍,额外省下 **≈0.5 bit/参数** → 平均位宽从 ~4.5 降到 **~4.1 bit/参数**。

**为什么要分块(block-wise)而不是整张一个 scale?** 权重虽近似正态,但不同区域幅度有差异;整张一个 absmax,一个局部大值就把全张量 scale 撑大、其余块分辨率下降(和 [[094 LLM.int8 与离群值|离群值]] 同理)。分块(64 个一组各自 absmax)把这种「局部撑爆」隔离在一个小块内,**只牺牲那一块、不连累全局**——这是低比特量化抗离群的标准手段(对应 [[092 量化基础：对称非对称、per-tensor、per-channel、per-group|per-group]])。块越小越抗离群但 scale 元数据越多;NF4 取 64 是折中,再用双重量化把这笔元数据压回去。

**双重量化的两层账(算清)**:第一层,块大小 64,每块 1 个 FP32 scale → $32/64=0.5$ bit/参。第二层,把这些 scale 按外层块(256 个 scale 一组)用 8-bit 量化 + 1 个 FP32 二级 scale → scale 的存储从 32 bit 降到约 8 bit,省下约 $\frac{32-8}{64}\approx0.37\sim0.5$ bit/参。合计平均位宽 $\approx4.1$ bit/参,几乎逼近理论 4 bit,而精度几乎不掉。

![[quant-NF4分位量化.png]]

## 原理

**分位量化(quantile quantization)**。设权重经 absmax 归一化到 `[-1, 1]`、近似服从标准正态 $\mathcal N(0,1)$,CDF 为 $\Phi$。把 $[0,1]$ 概率区间等分成 $2^k=16$ 份,取分位点

$$
q_i = \Phi^{-1}\!\Big(\frac{i + 0.5}{2^k}\Big),\quad i=0,1,\dots,15
$$

再整体缩放到 `[-1, 1]`,即得 16 个量化值。NF4 还做了**对称微调**:让 0 被精确表示(便于表示稀疏/padding),正负两半各取分位点拼接。"每桶等概率"等价于在该分布下**最小化期望量化误差**,故称信息论最优。

**反量化与前向**。量化时对每块 $w$ 求 absmax $s=\max|w|$,存索引 $\text{idx}=\arg\min_i |w/s - q_i|$ 与缩放 $s$。前向需要时反量化:$\hat w = s\cdot q_{\text{idx}}$,再用 BF16 做矩阵乘。**基座 $\hat w$ 始终冻结、无梯度**。

**双重量化(Double Quantization)**。块大小 64 时,每 64 个权重配 1 个 FP32 的 $s$(占 $32/64=0.5$ bit/参数)。把所有块的 $s$ 再按更大的块(如 256)用 8-bit 量化:$s \approx s_2 \cdot \hat q$,省掉大部分 scale 开销,额外 $\approx0.5$ bit/参数。

**QLoRA 前向**。线性层输出

$$
y = \underbrace{\text{dequant}_{\text{NF4}}(\hat W)\,x}_{\text{冻结基座,4-bit 存}} \;+\; \underbrace{\frac{\alpha}{r}\,B A\,x}_{\text{16-bit LoRA,可训练}}
$$

只有 $A\in\mathbb R^{r\times d}$、$B\in\mathbb R^{d\times r}$($r$ 很小)有梯度。再配**分页优化器(Paged Optimizers)**,用统一内存在显存尖峰时把优化器状态临时换到 CPU,避免 OOM。详见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|QLoRA]]。

![[quant-QLoRA数据流.png]]

## 代码

```python
import numpy as np
from scipy.stats import norm

# ❌ 朴素:对正态权重用均匀 INT4 —— 中间高频区分辨率不足
def quant_int4_uniform(w):
    s = np.abs(w).max()
    wn = w / s                                  # 归一化到 [-1,1]
    levels = np.linspace(-1, 1, 16)             # 均匀 16 档(中间也只这么密)
    idx = np.abs(wn[:, None] - levels).argmin(1)
    return levels[idx] * s

# ✅ NF4:按正态分位取 16 个量化点 —— 中间密、两端疏
def make_nf4_levels():
    # 标准正态分位点,中点偏移避免取到 ±inf,再归一化到 [-1,1]
    qs = norm.ppf((np.arange(16) + 0.5) / 16)
    return qs / np.abs(qs).max()

NF4 = make_nf4_levels()
def quant_nf4(w, block=64):
    out = np.empty_like(w)
    for i in range(0, len(w), block):           # 块量化:每块一个 scale
        blk = w[i:i+block]
        s = np.abs(blk).max() + 1e-8
        idx = np.abs(blk[:, None] / s - NF4).argmin(1)
        out[i:i+block] = NF4[idx] * s           # 反量化回浮点
    return out

w = np.random.randn(4096).astype(np.float32) * 0.1   # 钟形权重
e_int4 = np.abs(w - quant_int4_uniform(w)).mean()
e_nf4  = np.abs(w - quant_nf4(w)).mean()
print(f"INT4 均匀 误差={e_int4:.5f}  NF4 分位 误差={e_nf4:.5f}")  # NF4 更小
```

实战直接用 bitsandbytes:`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`。

## 面试高频

- **NF4 和 INT4 有什么区别?为什么对权重更好?** INT4 均匀分桶;NF4 按**正态分位**分桶,使每桶含等量权重,中间密两侧疏。预训练权重近似零中心正态,故 NF4 把分辨率放到最需要的高频区,是该分布下信息论最优的 4-bit 编码。
- **双重量化省了什么?** 块量化里每块要存一个 FP32 scale(块大小 64 时占 0.5 bit/参数),把这些 scale 再 8-bit 量化一遍,额外省 ≈0.5 bit/参数,平均位宽约 4.1。
- **QLoRA 为什么能在单卡微调 65B?** 基座 NF4 存(显存约降到 1/4)且冻结无梯度/无优化器状态;只训练极小的 16-bit LoRA 旁路;再用分页优化器扛显存尖峰。详见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|QLoRA]]。
- **NF4 量化的是谁?LoRA 是 16-bit 还是 4-bit?** 量化冻结的**基座权重**;LoRA 旁路是 **16-bit 可训练**的,前向时与反量化后的基座相加。
- **QLoRA 推理慢的原因?** 前向每次要把 NF4 反量化成 BF16 再算,有额外开销;若追求部署吞吐通常会把 LoRA 合并并改用专门的量化推理路径([[096 AWQ 与 SmoothQuant|AWQ/SmoothQuant]] 等)。
- **NF4 假设权重正态,若不正态会怎样?** 误差变大;离群值(outlier)尤其麻烦,这正是 [[096 AWQ 与 SmoothQuant|AWQ、SmoothQuant]] 等针对激活/权重离群点的量化方法要解决的问题。块量化(块 64)能把局部离群隔离在一个小块内,缓解但不根治。
- **NF4 是浮点还是整数?和 INT4/GPTQ 的 int4 区别?** NF4 是 **4-bit 浮点**(16 个不等距的分位点作格点),INT4/GPTQ 的 int4 是**等距整数格点**。NF4 的格点中间密两头疏、贴合正态;int4 等距。NF4 主要面向 QLoRA 微调省显存,GPTQ int4 面向纯推理加速。
- **NF4 为什么要分块而非整张一个 scale?** 整张一个 absmax 会被局部大值撑爆、其余块分辨率下降;分块(64)把局部撑爆隔离在小块内。块越小越抗离群但 scale 越多,64 是折中,再用双重量化压回元数据。
- **QLoRA 训练慢在哪?能直接拿来做推理吗?** 前向每次要把 NF4 反量化成 BF16 再算,有反量化开销;部署追吞吐通常把 LoRA 合并、改用专门量化推理路径(GPTQ/AWQ 的 int4 kernel)。QLoRA 的定位是**省显存微调**,不是推理加速。
- **NF4 的 0 为什么要精确表示?** 通过对称微调让某个格点正好是 0,便于精确表示稀疏/padding/被剪枝的零权重(否则 0 被量化成小非零值,引入系统偏差)。

## 关键事实

- QLoRA 出自 Dettmers, Pagnoni, Holtzman, Zettlemoyer《QLoRA: Efficient Finetuning of Quantized LLMs》(2023,arXiv:2305.14314,NeurIPS 2023)。
- 三大组件:**NF4**(4-bit 正态浮点,信息论最优)、**双重量化**(量化 scale,省 ≈0.5 bit/参数)、**分页优化器**(防显存尖峰 OOM)。
- 默认**块大小 64** 做块量化;双重量化对 scale 用 8-bit、外层块大小 256。
- 效果:可在**单张 48GB GPU 上微调 65B** 模型,下游任务精度接近 16-bit 全参数微调。
- NF4 是 4-bit 浮点而非整数;与 [[096 AWQ 与 SmoothQuant|AWQ]]、[[098 FP8 训练推理与 AQLM 极低比特|AQLM]] 同属低比特权重压缩,但 NF4+QLoRA 面向**微调省显存**而非纯推理加速。
