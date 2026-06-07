[[062 学习率调度：warmup 加 cosine 与 WSD|学习率调度：warmup 加 cosine 与 WSD]] 讲 LLM 预训练里学习率怎么随步数变:先 **warmup**(线性升)防早期发散,再用 **cosine**(余弦衰减到底)或近年的 **WSD**(warmup-stable-decay:升、长平台、末尾短退火)把学习率慢慢压小。它是 [[40 学习率调度与 warmup、cosine|学习率调度]] 在大模型场景的具体配方。

## 直觉

学习率 $\eta$ 不是常数,而是一条随训练步数变化的曲线,分两段动机:

- **开头要小(warmup)**:训练初期参数随机、[[061 优化器与超参(AdamW)|AdamW]] 的二阶矩 $v$ 还没估准,这时若直接上大 lr,一步就可能把参数推飞、引发 [[066 训练不稳定：loss spike 与对策|loss spike]]。所以先从 0 线性爬到峰值 $\eta_{\max}$,给优化器一段「热身」。Pre-LN Transformer 尤其依赖 warmup,见 [[015 Transformer 训练稳定性|训练稳定性]]。
- **后面要小(decay)**:接近收敛时,大 lr 会在最优点附近反复横跳;把 lr 慢慢降下来,才能稳稳落进谷底、压低最终 loss。

两种主流降法:

- **cosine**:峰值后按余弦曲线平滑衰减到 $\eta_{\min}$(常取峰值的 10%)。缺点:**必须一开始就锁定总训练步数** $T$——余弦要知道终点在哪。中途想多训就尴尬了。
- **WSD(warmup-stable-decay)**:warmup 后把 lr **压平成一个长平台(stable)恒定不变**,要停训时才在末尾做**一段短退火(decay)**。好处:平台可任意延长、随时决定何时收尾、主干 checkpoint 能复用、退火段还方便做数据配比实验。MiniCPM、DeepSeek 等用这套。

一句话:**warmup 防早期炸,decay 防末期晃;cosine 要预知终点,WSD 把终点推迟到最后一刻**。

## 例子

设峰值 $\eta_{\max}=3\times10^{-4}$,总步 $T=100$,warmup $=10$ 步,$\eta_{\min}=3\times10^{-5}$。

**cosine**:第 $t$ 步学习率

- $t=5$(warmup 中):$\eta = 3\times10^{-4}\times\frac{5}{10}=1.5\times10^{-4}$(线性升)。
- $t=10$(到峰):$3\times10^{-4}$。
- $t=55$(衰减到一半进度,$\cos\frac{\pi}{2}=0$):$\eta = \eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})=1.65\times10^{-4}$。
- $t=100$(终点):$3\times10^{-5}$。

**WSD**:warmup 同上;$t=10\sim90$ 一直恒为 $3\times10^{-4}$(stable);$t=90\sim100$ 才从 $3\times10^{-4}$ 快速退火到 $3\times10^{-5}$。如果训到 90 步发现还想继续,直接延长 stable 平台即可——这是 cosine 做不到的。

![[train-warmup-cosine-WSD曲线.png]]

## 原理

**warmup(线性)**,设 warmup 步数 $T_w$:

$$\eta_t = \eta_{\max}\cdot \frac{t}{T_w},\qquad t \le T_w$$

**cosine 衰减**,$T_w < t \le T$:

$$\eta_t = \eta_{\min} + \frac{1}{2}\left(\eta_{\max}-\eta_{\min}\right)\left(1+\cos\frac{\pi\,(t-T_w)}{T-T_w}\right)$$

在 $t=T_w$ 时 $\cos 0=1$,$\eta=\eta_{\max}$;在 $t=T$ 时 $\cos\pi=-1$,$\eta=\eta_{\min}$;中间平滑下降。可见**必须知道 $T$**。

**WSD**,设 stable 到第 $T_s$ 步、其后退火长度 $T_d=T-T_s$:

$$\eta_t = \begin{cases}\eta_{\max}\cdot t/T_w & t\le T_w \quad(\text{warmup})\\[4pt]\eta_{\max} & T_w < t \le T_s \quad(\text{stable})\\[4pt]f\!\left(\frac{t-T_s}{T_d}\right)\cdot\eta_{\max} & T_s < t \le T \quad(\text{decay})\end{cases}$$

退火函数 $f$ 从 1 降到接近 0,可用线性、余弦或更激进的 $1-\sqrt{\cdot}$。WSD 论文(MiniCPM)指出:**只有末尾退火段 loss 才急剧下降**,而平台期长度可自由扩展,因此能用一条主干训练、在不同步数处分叉退火,做高效的 scaling law / 数据配比实验。直观上(River-Valley 损失面视角,arXiv 2410.05192):stable 沿「河谷」走,decay 才下沉到谷底。

**还有一支:inverse-sqrt(平方根倒数衰减)。** 原始 Transformer / T5 用 $\eta_t=\eta_{\max}\cdot\sqrt{T_w/t}$($t>T_w$),即峰值后按 $1/\sqrt t$ 缓降。它和 cosine 一样不需要知道终点(对续训友好),但衰减比 cosine 慢、末期 lr 不会压到很低,故最终 loss 常略逊 cosine,现代大模型多已转向 cosine 或 WSD。

**peak lr 怎么定(经验规律)。** ① **随宽度反比缩**:GPT-3 经验上隐藏维越大、峰值 lr 越小($\eta_{\max}$ 大致 $\propto 1/\sqrt{d}$ 或更陡);**muP / μTransfer**(Yang 等 2022)更进一步,把超参做成「宽度无关」,在小模型上调好 lr 后直接迁移到大模型,省去大模型上反复试 lr。② **随 batch 正比缩**:见原理第 4 点与 [[063 批大小、梯度累积与 critical batch size|线性缩放规则]]。两条规律方向相反(宽度↑→lr↓、batch↑→lr↑),实际是两者叠加。

**退火到零(D2Z)与最终 loss。** cosine 常只降到峰值 10%,而长训练里**一路退火到接近 0(decay-to-zero)** 往往能拿到更低的最终 loss——直觉是收尾阶段越小的 lr 越能精修进谷底。WSD 的 decay 段天然支持 D2Z,这也是它在长程训练里常优于「降到 10% 就停」的 cosine 的原因之一。

## 代码

```python
import math, torch

def lr_lambda_cosine(step, warmup, total, min_ratio=0.1):
    if step < warmup:                      # 线性 warmup
        return step / warmup
    prog = (step - warmup) / (total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return min_ratio + (1 - min_ratio) * cos   # 降到 min_ratio*peak

def lr_lambda_wsd(step, warmup, stable_end, total, min_ratio=0.0):
    if step < warmup:
        return step / warmup
    if step < stable_end:                  # 恒定平台
        return 1.0
    prog = (step - stable_end) / (total - stable_end)   # 末尾短退火
    return 1.0 - (1 - min_ratio) * prog    # 线性退火到 min_ratio

opt = torch.optim.AdamW(torch.nn.Linear(8, 8).parameters(), lr=3e-4)

# ❌ 错误:不要 warmup,一上来就峰值学习率
#   sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
#   大模型 + Pre-LN 极易在前几百步炸 loss(见 066)

# ✅ 正确:warmup + cosine(本例总步 10000,warmup 500)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lr_lambda=lambda s: lr_lambda_cosine(s, warmup=500, total=10000))

for step in range(10000):
    # ... forward / backward / opt.step() ...
    sched.step()
```

```python
import math
# —— 逐点核对 warmup+cosine 的数值(对应「例子」一节) ——
def cosine_lr(t, peak, warmup, total, min_ratio=0.1):
    if t < warmup: return peak * t / warmup                 # 线性 warmup
    prog = (t - warmup) / (total - warmup)
    return peak * (min_ratio + (1-min_ratio)*0.5*(1+math.cos(math.pi*prog)))

peak, warmup, total = 3e-4, 10, 100
for t in [5, 10, 55, 100]:
    print(f"t={t:>3}  lr={cosine_lr(t, peak, warmup, total):.3e}")
# t=5 → 1.5e-4(升一半);t=10 → 3e-4(峰);t=55 → 中点;t=100 → 3e-5(=10%峰)

# —— inverse-sqrt(T5 风格,无需预知终点) ——
def invsqrt_lr(t, peak, warmup):
    return peak * (min(t, warmup) / warmup) if t < warmup else peak * math.sqrt(warmup / t)
print([round(invsqrt_lr(t, 3e-4, 10), 6) for t in [5, 10, 40, 90]])
# 续训友好但末期降不深,最终 loss 常逊 cosine

# ❌ warmup 太短(几十步)→ 大模型前期 AdamW v 没估准就上峰值 lr,易 spike
# ✅ warmup 取总步 1%~2%(或几亿~几十亿 token);模型/batch 越大,warmup 越长
```

## 面试高频

- **为什么需要 warmup?** 训练初期参数随机、AdamW 二阶矩未估准,直接上大 lr 易使梯度/参数爆炸引发 [[066 训练不稳定：loss spike 与对策|loss spike]];线性升温给优化器统计量「热身」时间。Pre-LN 也依赖它。
- **cosine 的痛点是什么?** 必须预先确定总步数 $T$;中途想延长训练就得重设曲线,且续训不友好。
- **WSD 相对 cosine 好在哪?** 平台期可任意延长、随时收尾、主干可复用;退火段方便做数据配比/退火实验,且对续训友好——不必预设终点。
- **warmup 通常多长?** 经验上几百到几千步(或总步数的 1%~2%)、约几亿到几十亿 token;模型越大、batch 越大,往往需要更长 warmup。
- **decay 到多低?** cosine 常降到峰值的 10%(不到 0);WSD 退火可一直降到接近 0(decay-to-zero)。
- **lr 调度与 batch 的关系?** batch 增大常按线性缩放规则同步放大峰值 lr,见 [[063 批大小、梯度累积与 critical batch size|critical batch size]]。
- **峰值 lr 怎么随模型大小定?** 经验上随宽度反比缩(大模型 lr 更小);muP/μTransfer(Yang 2022)把超参做成宽度无关,小模型调好 lr 直接迁大模型,省去重调。
- **还有别的调度吗(inverse-sqrt)?** T5/原始 Transformer 用 $1/\sqrt t$ 衰减,和 cosine 一样不需预知终点,但降得慢、末期 lr 不低,最终 loss 常略逊 cosine。
- **退火到零(D2Z)有什么好?** 长训练里一路降到接近 0 常获更低最终 loss(收尾小 lr 精修进谷底);WSD 的 decay 段天然支持,优于「降到 10% 就停」的 cosine。

## 关键事实

- warmup + cosine 是 GPT-3(Brown et al., 2020)、LLaMA(Touvron et al., 2023)等的标准配方;cosine 常衰减到峰值 lr 的 10%。
- WSD(warmup-stable-decay)由 MiniCPM(Hu et al., 2024,arXiv:2404.06395)系统提出并用于高效 scaling 研究;DeepSeek 等也采用「常数 + 末段退火」式调度。
- River-Valley 损失面视角解释 WSD:stable 沿河谷前进、decay 沉入谷底(《Understanding WSD...》,2024,arXiv:2410.05192)。
- 退火段(cooldown)可线性 / 余弦 / decay-to-zero(D2Z);D2Z 在长训练中常获更低最终 loss。
