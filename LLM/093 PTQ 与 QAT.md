[[093 PTQ 与 QAT|PTQ 与 QAT]]:两条把模型从浮点变低比特的路线。**PTQ**(Post-Training Quantization,训后量化)在**已训好的模型**上直接量化,只用少量校准数据统计分布、定 scale/zero-point,**不训练、分钟级完成**;**QAT**(Quantization-Aware Training,量化感知训练)在**训练 / 微调过程中插入「伪量化」节点模拟舍入误差**,让权重学会适应量化,**精度最高但要重训、要数据与算力**。它们都建立在 [[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]] 之上;LLM 因参数太大,主流是「更聪明的 PTQ」([[095 GPTQ|GPTQ]]、[[096 AWQ 与 SmoothQuant|AWQ]])。

## 直觉:事后将就 vs 边训边适应

量化必然引入误差(round 的零头)。区别在**模型有没有机会「知道」自己会被量化**:

- **PTQ**:模型训练时完全不知情,训完了我们才拿尺子去量它。好比一篇用钢笔写好的文章,事后用粗记号笔照描——快,但细节会糊。优点是**便宜**:不需要训练流程、不需要标注数据(只要几百条无标签样本做**校准**),几分钟搞定。缺点:位宽很低(int4)或模型对量化敏感时,**可能明显掉点**。
- **QAT**:训练时就**在前向里插入「先量化再反量化」**,让模型「感受到」量化误差,反向传播会把权重**推到对量化更鲁棒的位置**(比如避开舍入边界、压缩离群值)。好比知道最终要用记号笔,写字时就刻意写粗、留足间距——精度最好,尤其低比特;代价是要**完整训练或微调**,贵且慢、要数据。

一句话:**PTQ 求快求省,QAT 求精度上限**。LLM 时代,因为模型动辄数十上百亿参数,QAT 重训成本难以接受,于是工程上发展出**误差补偿型 PTQ**(GPTQ 逐层修正、AWQ 保护重要权重),用 PTQ 的代价逼近 QAT 的精度。

![[quant-ptq-vs-qat.png]]

## 例子:同一个 int4 量化,两条路差在哪

设一个权重 $w=0.47$,int4 的某段格点是 $\{0.40,\,0.50\}$(格距 $0.10$)。

- **PTQ**:训练已结束,$w=0.47$ 被**直接四舍五入**到 $0.50$,误差 $0.03$ 固定下来,没有任何补救——除非用 GPTQ 那样**把这 0.03 的误差转嫁补偿到同层后续权重上**。
- **QAT**:前向时 $w$ 被伪量化成 $0.50$ 参与计算,但**梯度照常更新真实的 $w$**(它仍是 FP 的 $0.47$ 在学)。若量化到 $0.50$ 让 loss 变大,梯度会把真实 $w$ 往 $0.40$ 那侧拉(或调整 scale),几个 step 后 $w$ 落到**量化后误差更小**的位置。等训练收敛,$w$ 已经「住进」一个对 int4 友好的点。

**关键卡点**:$\text{round}$ 函数是阶梯状的,**导数处处为 0**,梯度根本回传不了 → QAT 没法训。解决靠 **STE(Straight-Through Estimator,直通估计)**:前向照常 round,**反向时假装 round 是恒等函数($\frac{\partial q}{\partial x}\equiv1$)**,让上游梯度原样穿过量化节点。

**STE 走一遍数字**。设 $w=0.47$、$s=0.10$,前向伪量化:$\text{round}(0.47/0.10)=\text{round}(4.7)=5$,$\tilde w=5\times0.10=0.50$。上游传来损失梯度 $\frac{\partial L}{\partial\tilde w}=0.3$。**真梯度** $\frac{\partial\tilde w}{\partial w}$ 几乎处处为 0(round 是阶梯),若照实用,$\frac{\partial L}{\partial w}=0.3\times0=0$,$w$ 永远不更新。**STE 假装** $\frac{\partial\tilde w}{\partial w}=1$(只要 $w/s$ 没被 clip),于是 $\frac{\partial L}{\partial w}\approx0.3$ 原样穿过——优化器据此把真实的 $w=0.47$ 往降 loss 的方向挪。几步后,$w$ 可能落到 $0.42$(更靠近格点 $0.40$,量化误差从 $0.03$ 降到 $0.02$),或者把 scale 调到让 $0.47$ 正好落格点上。**没有 STE 这一步「假装 round 可导」,QAT 完全训不动**。

![[quant-ste-直通估计.png]]

## 原理:校准、伪量化、STE

**PTQ 的核心是「定范围」**。权重直接按 [[092 量化基础：对称非对称、per-tensor、per-channel、per-group|对称 per-channel]] 量化即可;难点在**激活**——它依赖输入,得用**校准集**(几十~几百条样本)跑前向,统计每层激活的 $[\min,\max]$ 或分布,再定 $s,z$。校准策略:

- **min-max**:直接取观测到的极值。简单但对**离群值**极敏感(一个大值撑爆尺子)。
- **百分位 / MSE / KL 校准**:裁掉极端尾部(如取 99.9% 分位),或选让量化 MSE / 与原分布 KL 散度最小的范围。更鲁棒,是实务常用。
- **KL 校准(TensorRT 的经典做法)**:把激活直方图按不同截断阈值量化,选**量化后分布与原分布 KL 散度最小**的那个阈值——即「丢掉的信息最少」。比 min-max 抗离群,适合激活有长尾的情形。校准集要**有代表性**(覆盖真实输入分布),否则定的范围在真实流量下失准。

**伪量化(fake quantization)** 是 QAT 的前向算子:

$$
\tilde x=\text{dequant}\big(\text{quant}(x)\big)=s\cdot\Big(\text{clip}\big(\text{round}(x/s)+z,\ q_{\min},q_{\max}\big)-z\Big)
$$

它**用浮点表示量化后的值**,插在权重和激活上,让前向「体验」到量化误差,但中间仍是可微的浮点张量,便于训练。

**STE 让梯度穿过 round**。$\text{round}$ 的导数几乎处处为 0,STE 近似:

$$
\frac{\partial \tilde x}{\partial x}\approx\mathbb{1}\big[q_{\min}\le x/s\le q_{\max}\big]
$$

即在**未被 clip 截断的区间内梯度为 1**(直通),被截断处梯度为 0。于是 $\dfrac{\partial L}{\partial x}\approx\dfrac{\partial L}{\partial \tilde x}$,反向能正常更新真实(全精度)权重。没有 STE,QAT 无法训练。

**PTQ vs QAT 取舍表**:

| 维度 | PTQ | QAT |
|---|---|---|
| 是否重训 | 否(仅校准) | 是(训练 / 微调) |
| 需要数据 | 少量无标签校准集 | 完整训练数据 |
| 成本 / 时间 | 极低,分钟级 | 高,等同一次训练 |
| int8 精度 | 通常够好 | 略胜 |
| int4 及更低 | 易掉点(需 GPTQ/AWQ 补救) | **精度最佳** |
| LLM 适用性 | **主流**(参数太大) | 罕用(太贵) |

**为什么 LLM 几乎不做 QAT**:重训一个 70B 模型成本接近重新预训练,得不偿失;所以 LLM 走「PTQ + 误差补偿」:[[094 LLM.int8 与离群值|LLM.int8]] 用混合精度绕开离群值,[[095 GPTQ|GPTQ]] 逐层做二阶误差补偿,[[096 AWQ 与 SmoothQuant|AWQ/SmoothQuant]] 保护 / 迁移重要尺度——都属于**精巧的 PTQ**。

**AdaRound:朴素 round 不是最优的(LLM 误差补偿思路的前身)**。一个反直觉但重要的事实:把每个权重「就近取整」并**不**最小化整层输出误差——有时把某个权重**向远的格点取整**,反而能让整层输出更准(因为权重间有相关性)。AdaRound(Nagel 2020)把「每个权重向上还是向下取整」当成一个可学习的二元变量,用少量校准数据优化,显著优于 RTN。GPTQ 可看作这条思路的二阶最优版:不是独立决定每个权重的取整方向,而是用 Hessian **把量化误差最优地补偿进后续权重**。记忆链:RTN(就近)→ AdaRound(学取整方向)→ GPTQ(二阶误差补偿),误差越来越小。

**weight-only vs weight+activation 量化**(LLM 量化的关键二分):
- **仅权重(W4A16 等)**:只量化权重到 int4,激活保 fp16。收益是**省显存 + 访存加速**(解码是 memory-bound,权重搬运减半即提速),实现简单、无激活离群难题。代表:GPTQ、AWQ。**LLM 部署主流**。
- **权重+激活(W8A8)**:权重和激活都量化,能吃 int8 Tensor Core 的**算力**(compute-bound 的 prefill / 大 batch 受益)。难点是激活离群,需 LLM.int8(混合精度)或 SmoothQuant(迁移难度)。代表:LLM.int8、SmoothQuant。

![[quant-W4A16vsW8A8.png]]

## 代码:PTQ 校准 vs QAT 伪量化 + STE(❌ vs ✅)

```python
import torch

# ---------- PTQ:训练后,用校准集定激活范围,直接量化 ----------
@torch.no_grad()
def ptq_calibrate(model, calib_loader):
    # ❌ 错:不校准、对激活用固定/拍脑袋的范围 —— 离群值撑爆尺子,严重掉点
    # ✅ 对:跑前向收集每层激活的 min/max(或百分位),据此定 scale
    act_min, act_max = {}, {}
    def hook(name):
        def fn(_m, _inp, out):
            act_min[name] = min(act_min.get(name, 1e9), out.min().item())
            act_max[name] = max(act_max.get(name, -1e9), out.max().item())
        return fn
    handles = [m.register_forward_hook(hook(n))
               for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    for x in calib_loader:               # 几十~几百条无标签样本即可
        model(x)
    for h in handles: h.remove()
    return act_min, act_max              # 用它们算每层 scale/zero-point

# ---------- QAT:伪量化 + STE,让 round 可训 ----------
class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, qmin, qmax):
        q = torch.clamp(torch.round(x / s), qmin, qmax)
        ctx.save_for_backward(x / s)         # 记录是否在范围内
        ctx.qmin, ctx.qmax = qmin, qmax
        return s * q                         # 反量化回浮点,前向「体验」误差
    @staticmethod
    def backward(ctx, g):
        (xs,) = ctx.saved_tensors
        mask = (xs >= ctx.qmin) & (xs <= ctx.qmax)   # 未被 clip 处直通
        return g * mask, None, None, None            # ✅ STE:梯度原样穿过 round

def fake_quant(x, s, bits=8):
    qmax = 2 ** (bits - 1) - 1
    return FakeQuantSTE.apply(x, s, -qmax, qmax)

# QAT 训练循环里:对权重(必要时激活)套 fake_quant,再正常算 loss/backward
# 真实全精度权重在更新;部署时按学到的 scale 真量化成 int
```

## 面试高频

- **PTQ 和 QAT 一句话区别?** PTQ 训练后才量化、只需校准、便宜快;QAT 训练中插伪量化、让模型适应量化、精度高但要重训。
- **STE 是什么?为什么需要它?** round 的梯度处处为 0,会切断反向传播;STE 在反向时把 round 当成恒等函数($\frac{\partial q}{\partial x}=1$,截断区为 0),让梯度直通,QAT 才训得动。
- **PTQ 为什么要校准集?校准什么?** 主要校准**激活**的数值范围(激活依赖输入,无法离线知道);用几十~几百条无标签样本跑前向,统计 min/max 或百分位定 scale/zero-point。权重一般不需校准(离线可见)。
- **为什么 LLM 用 PTQ 而不是 QAT?** LLM 参数太大,QAT 重训成本≈重新预训练,不划算;改用「误差补偿型 PTQ」(GPTQ、AWQ)在低比特下逼近 QAT 精度。
- **PTQ 在 int8 vs int4 表现?** int8 PTQ 通常几乎无损;int4 朴素 PTQ 易掉点,需 [[095 GPTQ|GPTQ]] 的逐层误差补偿或 [[096 AWQ 与 SmoothQuant|AWQ]] 的重要权重保护。
- **min-max 校准的坑?** 对离群值极敏感(一个大值撑宽尺子,其余值挤在少数格子);用百分位 / MSE / KL 校准更稳。
- **QAT 训练时权重是 int 还是 float?** 仍是 **float**(全精度权重在学);前向用伪量化「体验」误差,部署时才真折成 int。
- **AdaRound 解决什么?** 朴素就近取整不最小化整层输出误差;AdaRound 学「每个权重向上还是向下取整」,显著优于 RTN,是 GPTQ(二阶误差补偿)的思路前身。链条:RTN→AdaRound→GPTQ。
- **weight-only 和 W8A8 量化怎么选?** 解码 memory-bound、追省显存/低延迟 → 仅权重 int4(GPTQ/AWQ);prefill/大 batch compute-bound、追吞吐 → W8A8(SmoothQuant 吃 int8 算力)。前者无激活离群难题,是 LLM 部署主流。
- **PTQ 校准集要多大/什么要求?** 几十~几百条**无标签但有代表性**的样本(覆盖真实输入分布)即可;主要校准激活范围。校准集偏了会让定的 scale 在真实流量下失准,GPTQ/AWQ 这类对校准也有一定敏感性。

## 关键事实

- PTQ 与 QAT 的术语、伪量化与 per-channel 框架来自 Jacob et al.(Google,2018,arXiv:1712.05877);系统综述见 Nagel et al.《A White Paper on Neural Network Quantization》(2021,arXiv:2106.08295)、Gholami et al.(2021,arXiv:2103.13630)。
- STE(Straight-Through Estimator)由 Bengio et al.《Estimating or Propagating Gradients Through Stochastic Neurons》(2013,arXiv:1308.3432)提出,是训练带不可导离散算子(量化、二值网络)的标准技巧。
- 进阶 PTQ:AdaRound(Nagel et al.,2020,arXiv:2004.10568)用学习的方式决定每个权重向上还是向下取整,显著优于朴素 round,是 LLM 量化误差补偿思路的前身。
- LLM 实务:几乎都用 PTQ。int8 权重+激活靠 [[094 LLM.int8 与离群值|LLM.int8]] 处理离群值;int4 权重靠 [[095 GPTQ|GPTQ]]、[[096 AWQ 与 SmoothQuant|AWQ]];QAT 仅在少数小模型或对精度极致要求时使用。
- 关联:[[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]、[[094 LLM.int8 与离群值|LLM.int8]]、[[095 GPTQ|GPTQ]]、[[096 AWQ 与 SmoothQuant|AWQ 与 SmoothQuant]]、[[091 高效微调：LoRA、QLoRA、Adapter、Prefix|QLoRA(量化+微调)]]。
