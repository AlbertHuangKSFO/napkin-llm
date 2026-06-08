[[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]] 是在网络内部把每层的激活值重新拉到"均值约 0、方差约 1"的固定分布,再用可学习的缩放偏移还原表达力,从而让深层网络训得更快、更稳、对初始化更不敏感。

## 直觉

想象一条流水线,每个工位(层)都要在上一工位的半成品上加工。如果上游今天交来的零件偏大、明天偏小,下游工位就得不停重新校准卡尺,效率极低。**归一化就是在每个工位前装一台"标准化机":不管来料多大多小,先统一缩放到标准尺寸再加工**,下游永远面对稳定的来料。

四种归一化的差别只在一件事:**"沿哪个轴"求均值和方差**。

- **BatchNorm**:把"全班同一道题的分数"对齐(同一通道、跨整个 batch)。
- **LayerNorm**:把"同一个人所有科目的分数"对齐(同一样本、跨全部特征)。
- **RMSNorm**:LayerNorm 的省事版,只按"长度"缩放,不挪中心。
- **GroupNorm**:把科目分成几组,每组内对齐(同一样本、通道分组)。

![[nn-BN与LN归一化轴.png]]

## 例子

设一个 mini-batch 有 2 个样本,每个样本是 4 维特征向量:

$$x^{(1)}=[1,\,2,\,3,\,4],\qquad x^{(2)}=[2,\,4,\,6,\,8]$$

**LayerNorm(对每个样本,沿 4 个特征自己算)**:

样本 1:均值 $\mu=\frac{1+2+3+4}{4}=2.5$,方差 $\sigma^2=\frac{(1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2}{4}=1.25$,$\sigma\approx1.118$。

归一化后(取 $\epsilon=0$):

$$\hat x^{(1)}=\Big[\tfrac{1-2.5}{1.118},\,\tfrac{2-2.5}{1.118},\,\tfrac{3-2.5}{1.118},\,\tfrac{4-2.5}{1.118}\Big]=[-1.342,\,-0.447,\,0.447,\,1.342]$$

样本 2 的均值是 5、方差是 5,归一化后**和样本 1 完全一样**:$[-1.342,-0.447,0.447,1.342]$。这正是 LN 的特性:**每个样本独立处理,与 batch 里其他样本无关**。

**BatchNorm(对每个特征位,沿 2 个样本算)**:看第 1 维 $\{1,2\}$,$\mu=1.5,\ \sigma^2=0.25,\ \sigma=0.5$,归一化得 $\{-1,+1\}$;第 4 维 $\{4,8\}$,$\mu=6,\sigma=2$,归一化得 $\{-1,+1\}$。**换一批样本,μ、σ 就变了** —— 这就是 BN 推理时要靠移动平均、batch=1 时不稳的根源。

**RMSNorm(样本 1)**:不减均值,直接除均方根 $\text{RMS}=\sqrt{\frac{1+4+9+16}{4}}=\sqrt{7.5}\approx2.739$:

$$\hat x^{(1)}=[0.365,\,0.730,\,1.095,\,1.461]$$

注意它**没有把中心挪到 0**(因为没减均值),只把整体长度缩到 RMS=1。

**带 γ、β 的完整一步(LayerNorm,样本 1)**。归一化得 $\hat x^{(1)}=[-1.342,-0.447,0.447,1.342]$ 后,设这一层学到 $\gamma=[2,2,2,2]$、$\beta=[1,1,1,1]$,则输出 $y=\gamma\odot\hat x+\beta=[-1.684,0.106,1.894,3.684]$。可见 γ 控制每个特征的"幅度"、β 控制"中心",**网络若需要可把分布重新拉到任意位置**——这保证归一化不损失表达力(极端情形 $\gamma=\sigma,\beta=\mu$ 直接还原原始激活)。

**GroupNorm 手算(轴的差别看得最清)**。设一个样本有 4 个通道、每通道 1 个像素:$x=[1,2,3,4]$(通道维)。取 $G=2$ 组,通道 $\{1,2\}$ 一组、$\{3,4\}$ 一组。
- 第 1 组 $\{1,2\}$:$\mu=1.5,\sigma^2=0.25,\sigma=0.5$,归一化 $[-1,+1]$。
- 第 2 组 $\{3,4\}$:$\mu=3.5,\sigma^2=0.25,\sigma=0.5$,归一化 $[-1,+1]$。
- 结果 $[-1,1,-1,1]$。对比:$G=1$ 时对全 4 通道一起算($\mu=2.5$)= LayerNorm;$G=4$ 时每通道单独($1 个数无方差)= InstanceNorm。这就是"GN 是 LN 和 IN 之间的连续谱"的直观含义。

## 原理

**通用模板**。给一组要归一化的元素 $\{x_i\}_{i=1}^m$(具体是哪些元素由"轴"决定):

$$\mu=\frac{1}{m}\sum_{i=1}^m x_i,\qquad \sigma^2=\frac{1}{m}\sum_{i=1}^m (x_i-\mu)^2$$

$$\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y_i=\gamma\,\hat x_i+\beta$$

其中 $\epsilon$(典型 $10^{-5}$)防止除零;$\gamma,\beta$ 是**可学习**的缩放与偏移参数,让网络在需要时能把归一化"撤销"(若 $\gamma=\sqrt{\sigma^2+\epsilon},\ \beta=\mu$ 则还原),保证不损失表达力。

**四种的差别 = $m$ 取哪些元素**。设激活张量形状为 $(N,C,H,W)$(批量、通道、高、宽),或序列里的 $(N,T,D)$(批量、时间步、特征):

- **BN**:固定一个通道 $c$,$\{x_i\}$ 取所有 $(N,H,W)$ —— 跨样本、跨空间,**沿 batch 维**。每个通道一组 $(\gamma_c,\beta_c)$。
- **LN**:固定一个样本(及一个时间步),$\{x_i\}$ 取该位置的所有特征/通道 —— **沿特征维**,与 batch 无关。
- **GN**:固定一个样本,把 $C$ 个通道分成 $G$ 组,每组内沿 $(H,W)$ 和组内通道求 μ、σ。$G=1$ 退化为 LN(对全部通道),$G=C$ 退化为 InstanceNorm。
- **RMSNorm**:见下。

![[nn-四种归一化对比.png]]

**为什么有效:内部协变量偏移(Internal Covariate Shift)**。训练中,某层的输入分布会随前面所有层参数更新而不断漂移,后层得反复适应"移动的靶子",拖慢收敛、逼小学习率。归一化把每层输入钉到固定分布,让后层站在稳定地基上 —— 这是 Ioffe & Szegedy(2015)提出 BN 的原始动机。

不过 Santurkar 等(2018)用实验反驳:BN 的真正好处更可能是**让损失曲面更平滑**(梯度更可预测),而非消除协变量偏移。结论可争,但"BN 有效"无争议。

**BN 的训练/推理双模式(高频细节)**。
- **训练**:用当前 mini-batch 实时算 $\mu_B,\sigma_B^2$ 归一化;同时维护一个**移动平均(running mean / running var)**:$\hat\mu\leftarrow(1-m)\hat\mu+m\,\mu_B$,$m$ 是 momentum(常 0.1)。
- **推理**:不看 batch,直接用训练期累积的 $\hat\mu,\hat\sigma^2$(固定值),保证单样本推理结果确定、不依赖同 batch 其他样本。
- **常见 bug**:忘了 `model.eval()` → 推理仍用 batch 统计 → 结果随 batch 组成抖动;或微调时 BN 的 running stats 没更新好导致掉点。

**BN 还自带正则效应**:每个样本的归一化依赖同 batch 其他样本,引入了**与 batch 组成相关的噪声**(类似 dropout 的随机性),所以用了 BN 常可减小 dropout 强度。这也是 BN 和 dropout 同时用容易"打架"(variance shift)的原因之一。

![[nn-内部协变量偏移.png]]

**为什么序列模型 / LLM 用 LN 不用 BN**(面试核心):

1. BN 的 μ、σ 跨 batch 求,推理时 batch=1 或序列变长,统计量不稳,只能靠训练期累积的移动平均近似,容易翻车。
2. 变长序列里 padding 位会污染 batch 维统计;LN 每个 token 自己归一,长度无关。
3. 自回归生成时,预测下一词不能依赖 batch 里别的样本的统计量 —— LN 天然满足。
4. 大模型 batch 受显存限,常很小,BN 方差估计噪声大;LN/RMSNorm 不依赖 batch 大小。

**RMSNorm:去掉减均值这一步**(Zhang & Sennrich, 2019)。作者假设 LN 的"重新中心化(re-centering)"不重要,真正起作用的是"重新缩放(re-scaling)"。于是只保留按均方根缩放:

$$\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2},\qquad y_i=\frac{x_i}{\text{RMS}(x)+\epsilon}\cdot g_i$$

注意分母里**没有 $\mu$、也没有方差**(方差需要先算均值),省掉了"求均值 + 减均值"这一遍遍历,反向传播也更省。论文报告比 LN 提速 7%–64%、效果相当 —— 这正是 LLaMA、T5、Gemma 等现代大模型普遍改用 RMSNorm 的原因。它只有缩放参数 $g$,无偏移 $\beta$。

**Pre-LN vs Post-LN(Transformer 必考)**。归一化放在残差块的哪里,直接影响深层 Transformer 能否稳定训练:
- **Post-LN**(原版 Transformer):$x\to \text{LN}(x+\text{Sublayer}(x))$,归一化在残差**相加之后**。深层时梯度容易在底层变大,需要 warmup 才训得稳。
- **Pre-LN**(GPT-2 之后主流):$x\to x+\text{Sublayer}(\text{LN}(x))$,归一化在子层**输入处**,残差主干是干净的恒等通路,梯度更稳、可去掉或减轻 warmup,能堆更深。代价是表达力略有损失,常在最后再补一个 LN。
- 现代大模型(LLaMA 等)= **Pre-LN + RMSNorm** 的组合。

![[nn-PreLN与PostLN.png]]

**其它归一化亲戚**:
- **InstanceNorm**:每样本每通道单独归一(GN 的 $G=C$ 特例),风格迁移常用。
- **WeightNorm**:不归一激活,而是把权重向量分解为方向 × 幅度,归一化方向,batch 无关。
- **BatchRenorm / SyncBatchNorm**:前者修正小 batch 下 running stats 与 batch stats 不一致;后者多 GPU 间同步 BN 统计量(分布式检测/分割常用)。

**为什么归一化里要加 ε**:防止某组方差恰为 0(如常数输入)时除零;也限制极小方差被放大成巨大值。典型 $10^{-5}$,Transformer 里有时用 $10^{-6}$。

## 代码

```python
import numpy as np

# x: (N, D) 一个 mini-batch
x = np.array([[1., 2., 3., 4.],
              [2., 4., 6., 8.]])
gamma, beta, eps = 1.0, 0.0, 1e-5

# ❌ 错:在序列/LLM 里用 BatchNorm —— 沿 batch 维(axis=0)
#    推理 batch=1 时方差≈0,且依赖别的样本,自回归会泄漏
mu_b  = x.mean(axis=0, keepdims=True)        # 每个特征位跨样本
var_b = x.var(axis=0, keepdims=True)
bn = gamma * (x - mu_b) / np.sqrt(var_b + eps) + beta

# ✅ 对:LayerNorm —— 沿特征维(axis=1),每个样本独立
mu_l  = x.mean(axis=1, keepdims=True)        # 每个样本跨自己特征
var_l = x.var(axis=1, keepdims=True)
ln = gamma * (x - mu_l) / np.sqrt(var_l + eps) + beta
print("LN:\n", ln)        # 两行完全相同,与 batch 无关

# ✅ RMSNorm —— 不减均值,只按均方根缩放(LLaMA 用法)
rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)
g = 1.0
rmsn = x / rms * g
print("RMSNorm:\n", rmsn)
```

```python
# PyTorch:序列/Transformer 里正确的归一化层
import torch, torch.nn as nn

D = 512
ln  = nn.LayerNorm(D)                 # 沿最后一维(特征)归一,batch 无关
rms = nn.RMSNorm(D)                   # PyTorch 2.4+ 内置;省去减均值
gn  = nn.GroupNorm(num_groups=8, num_channels=64)  # 小 batch CNN 替代 BN

x = torch.randn(2, 10, D)             # (batch, seq_len, dim)
y = ln(x)                             # 每个 token 自己归一,与 batch / 序列长无关
```

```python
# BN 的训练/推理双模式,手写看清"移动平均"机制
import numpy as np

class BatchNorm1D:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.g, self.b = np.ones(d), np.zeros(d)         # γ, β(可学习)
        self.run_mu, self.run_var = np.zeros(d), np.ones(d)  # 推理用的移动平均
        self.m, self.eps = momentum, eps
    def __call__(self, x, train=True):
        if train:
            mu, var = x.mean(0), x.var(0)                # ✅ 训练:用 batch 统计
            self.run_mu = (1-self.m)*self.run_mu + self.m*mu      # 累积移动平均
            self.run_var = (1-self.m)*self.run_var + self.m*var
        else:
            mu, var = self.run_mu, self.run_var          # ✅ 推理:用累积统计(固定)
        return self.g * (x - mu) / np.sqrt(var + self.eps) + self.b

bn = BatchNorm1D(4)
xb = np.random.randn(8, 4)
_ = bn(xb, train=True)                # 训练几步更新 running stats
print("推理输出(用移动平均,不看 batch):\n", bn(xb[:1], train=False).round(3))
# ❌ 忘了 train=False → 单样本推理用 batch 方差(≈0)→ 输出爆炸/随机
```

## 面试高频

- **Q:BatchNorm 和 LayerNorm 区别?为什么 Transformer 用 LayerNorm?**
  A:BN 沿 batch 维归一(同一通道跨样本),LN 沿特征维归一(同一样本跨特征)。Transformer 用 LN 因为:序列变长 + batch 小 + 自回归不能跨样本,BN 的 batch 统计在这些场景不稳/泄漏,而 LN 每个 token 独立、与 batch 无关。
- **Q:RMSNorm 比 LayerNorm 省在哪?为什么能省?**
  A:省掉"减均值"。LN 需 re-centering(减 μ)+ re-scaling(除 σ);RMSNorm 假设 re-centering 可有可无,只做按均方根的 re-scaling,少一遍遍历、少存中间量,提速 7%–64%(Zhang & Sennrich 2019),效果相当。它也没有偏移参数 β。
- **Q:BN 训练和推理为什么不一样?**
  A:训练用当前 batch 的 μ、σ;推理 batch 可能为 1,改用训练期累积的**移动平均** μ、σ(固定值)。忘了切 eval 模式 / 移动平均没更新好是常见 bug。
- **Q:小 batch 下 BN 为什么差,该换什么?**
  A:小 batch 的 μ、σ 估计噪声大。换 **GroupNorm**(batch 无关,Wu & He 2018 在 batch=2 时比 BN 误差低 10.6%)或 LN。
- **Q:Pre-LN 和 Post-LN 区别?为什么现代大模型用 Pre-LN?**
  A:Post-LN 把 LN 放残差相加后($\text{LN}(x+\text{Sublayer})$),深层梯度不稳、依赖 warmup;Pre-LN 放子层输入前($x+\text{Sublayer}(\text{LN}(x))$),残差主干是干净恒等通路,梯度稳、可堆更深。LLaMA = Pre-LN + RMSNorm。
- **Q:归一化里那个 ε 是干嘛的?**
  A:防止方差为 0 时除零,并限制极小方差被放大。典型 $10^{-5}$~$10^{-6}$。
- **Q:BN 自带正则效应从哪来?**
  A:每个样本的归一化依赖同 batch 其他样本,引入与 batch 组成相关的噪声(类似 dropout),因此用 BN 常可减小 dropout。也因此 BN 与 dropout 同用易"打架"。
- **陷阱**:① 把 BN 放在 RNN/Transformer 里 —— 几乎总是错。② 以为 γ、β 是超参数 —— 它们是**可学习参数**,反向传播会更新。③ 以为 RMSNorm 输出均值为 0 —— 它没减均值,中心不动。④ BN 的 $\gamma$ 初始化为 0(残差块里的技巧)vs 1,影响初期梯度。⑤ 推理忘了 `model.eval()` → BN 用 batch 统计、dropout 仍开,结果随机。⑥ 微调小数据时 BN running stats 漂移导致掉点,可冻结 BN。

## 关键事实

- BatchNorm 由 Ioffe & Szegedy 提出(ICML 2015,arXiv:1502.03167),动机是减少内部协变量偏移。
- LayerNorm 由 Ba、Kiros、Hinton 提出(2016,arXiv:1607.06450),专为 RNN / 小 batch 设计。
- RMSNorm 由 Zhang & Sennrich 提出(NeurIPS 2019,arXiv:1910.07467):假设 re-centering 可省,只保留 re-scaling,提速 **7%–64%**,效果与 LN 相当。
- GroupNorm 由 Wu & He 提出(ECCV 2018,arXiv:1803.08494):batch 无关,ResNet-50 在 batch=2 时比 BN 误差低 **10.6%**;LN、InstanceNorm 是其特例。
- Santurkar 等(2018,"How Does Batch Normalization Help Optimization?",NeurIPS)用实验质疑"协变量偏移"解释,主张 BN 实为平滑损失曲面。
- Pre-LN vs Post-LN 的稳定性分析:Xiong et al., *On Layer Normalization in the Transformer Architecture*(ICML 2020)——证明 Pre-LN 梯度更稳、可去 warmup;Post-LN 需 warmup。
- InstanceNorm:Ulyanov et al.(2016,风格迁移);WeightNorm:Salimans & Kingma(NeurIPS 2016);BatchRenorm:Ioffe(NeurIPS 2017,修正小 batch)。
- 现代大模型 Pre-LN + RMSNorm 的实践:LLaMA(Touvron et al., 2023)、T5(Raffel et al., 2020)、Gemma 等。
- 关联:归一化常配 [[41 权重初始化(Xavier、He、正交)|权重初始化]] 与 [[42 正则化(L2、Dropout、早停、标签平滑)|正则化]] 一起稳训练;γ、β 的更新依赖 [[20 反向传播的数学推导|反向传播]]。后续 Transformer 系列全靠 LayerNorm / RMSNorm。
