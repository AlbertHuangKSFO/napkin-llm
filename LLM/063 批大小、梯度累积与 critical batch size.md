[[063 批大小、梯度累积与 critical batch size|批大小、梯度累积与 critical batch size]] 讲预训练里 batch 多大合适:**critical batch size(临界批大小)**是「加大 batch 还能省训练步数」的拐点,过了它再加几乎白费算力;**梯度累积**让你在小显存上模拟大有效 batch;**线性缩放规则**告诉你 batch 变大时学习率该怎么跟。它建立在 [[37 梯度下降：BGD、SGD、Mini-batch|mini-batch 梯度下降]]之上。

## 直觉

batch 越大,每步梯度估计越准(噪声越小),理论上可以走更大、更稳的步。但不是越大越好:

- **小 batch 区**:梯度噪声主导。加大 batch 能显著降低噪声 ⇒ **每翻倍一次,到达目标 loss 所需步数大约减半**(线性缩放区)。这是数据并行省时间的来源。
- **大 batch 区(过了临界点)**:噪声已经很小,再加大 batch,梯度几乎没变准多少 ⇒ **步数几乎不再减少**。你花了双倍算力,却没换来双倍速度,**token 效率下降**。

这个拐点就是 **critical batch size(CBS)**。它不是固定值,由**梯度噪声尺度(gradient noise scale)**决定:不同样本梯度差异越大(噪声越大),CBS 越大,越能放心加大 batch;训练越往后、loss 越低,CBS 通常越大。

两个配套工具:

- **梯度累积(gradient accumulation)**:显存装不下大 batch 时,把一个大 batch 拆成几个 micro-batch,逐个前向+反向、**梯度累加不更新**,凑满再 `optimizer.step()` 一次。数学上等效于一个大 batch,但峰值显存只按 micro-batch 算。
- **线性缩放规则**:在 CBS 以内,把 batch 放大 $k$ 倍,学习率也大致放大 $k$ 倍,才能保持每步的有效进展。

一句话:**CBS 是并行省时的天花板;梯度累积用算力/时间换显存来逼近大 batch;线性缩放让 lr 跟上 batch**。

## 例子

**梯度累积**:显卡只放得下 micro-batch $=8$,但想要有效 batch $=32$。设 `accum_steps=4`:跑 4 个 micro-batch(各 8 条),每个反向后梯度**累加**进 `.grad`(不清零、不更新),第 4 个后才更新一次。

- 有效 batch $= 8 \times 4 = 32$,梯度统计与真正一次性跑 32 条**完全等效**。
- 峰值显存只按 8 条的激活算,小卡也能跑大有效 batch。
- 注意:`loss` 要除以 `accum_steps`,否则相当于把学习率放大了 4 倍。

**critical batch size**:某模型 CBS $\approx$ 200 万 token/batch。

- batch 从 50 万 → 100 万 token:步数约从 20 万 → 10 万(线性缩放区,翻倍省一半,划算)。
- batch 从 200 万 → 400 万 token:步数仅从 5 万 → 4.5 万(过了 CBS,多花一倍算力只省 10%,不划算)。

![[train-梯度累积等效大batch.png]]

![[train-critical-batch-size曲线.png]]

## 原理

**梯度噪声尺度**。设单样本梯度的协方差 $\Sigma$、真梯度 $G$,McCandlish 等定义(简化形式)

$$\mathcal B_{\text{noise}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G\rVert^2}$$

它就是 CBS 的量级:噪声大($\operatorname{tr}\Sigma$ 大)⇒ CBS 大。用 batch $B$ 训练时,达到目标 loss 的步数 $S$ 与处理样本数 $E=B\cdot S$ 满足近似关系

$$\frac{S}{S_{\min}} = 1 + \frac{\mathcal B_{\text{noise}}}{B},\qquad \frac{E}{E_{\min}} = 1 + \frac{B}{\mathcal B_{\text{noise}}}$$

- $B \ll \mathcal B_{\text{noise}}$:$S\approx S_{\min}(1+\mathcal B/B)$,**步数随 $B$ 增大反比下降**(线性缩放,省时间)。
- $B \gg \mathcal B_{\text{noise}}$:$S\to S_{\min}$ 触底,**再加 $B$ 不省步**,但样本量 $E$ 线性涨(浪费数据/算力)。

取 $B=\mathcal B_{\text{noise}}$,正好 $S=2S_{\min}$、$E=2E_{\min}$——这就是**临界批大小**:时间与算力的平衡点。

**线性缩放规则**(噪声主导区):batch $B\to kB$,梯度均值不变但噪声方差降 $k$ 倍,为保持每步对参数的有效更新幅度,学习率 $\eta\to k\eta$。大 batch 下要配合更长 [[062 学习率调度：warmup 加 cosine 与 WSD|warmup]] 防初期发散。注意:这是**近似规则**,接近或超过 CBS 时线性关系失效、需亚线性缩放或 batch ramp-up。

**把 CBS 公式代数字走一遍。** 设 $\mathcal B_{\text{noise}}=200$ 万 token,$S_{\min}=5$ 万步。用 $\frac{S}{S_{\min}}=1+\frac{\mathcal B}{B}$:

- $B=20$ 万:$S=5万\times(1+200/20)=5万\times11=55$ 万步,但样本量 $E=B\cdot S=20万\times55万$ 几乎触底 $E_{\min}$(省数据、费时间)。
- $B=200$ 万($=\mathcal B$):$S=5万\times2=10$ 万步,$E=2E_{\min}$——**临界点**,时间/算力各付 2 倍最小值。
- $B=2000$ 万:$S=5万\times(1+0.1)=5.5$ 万步(只比 $S_{\min}$ 多 10%,时间几乎到顶),但 $E=5.5万\times2000万$ 暴涨——**远超 CBS,纯烧数据不换速度**。

可见 CBS 正是「时间收益开始枯竭、数据浪费开始飙升」的交叉点。

**batch ramp-up(渐进加 batch)。** 既然 CBS 随训练后期变大(loss 越低、梯度噪声相对越大),最优策略不是全程定 batch,而是**前期小 batch、后期逐步加大**——前期 CBS 小,小 batch 已够;后期 CBS 大,加大 batch 才不浪费。GPT-3 即用 ramp-up:从约 3.2 万 token 的小 batch 起步,逐步升到约 **320 万 token/batch**,兼顾前期 token 效率与后期墙钟速度。

## 代码

```python
import torch
model = torch.nn.Linear(1024, 1024)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
accum_steps = 4                      # micro_bsz=8 → 有效 batch=32

# ❌ 错误:累积时每个 micro-batch 都更新 + 忘记除以 accum_steps
#   for mb in micro_batches:
#       loss = model(mb).sum()
#       loss.backward()
#       opt.step(); opt.zero_grad()  # 等于 batch 还是 8,根本没放大!
#   并且 loss 没 /accum,学习率被隐式放大 4 倍

# ✅ 正确:累加 4 次梯度,凑满才更新一次;loss 先归一化
opt.zero_grad()
for i, mb in enumerate(micro_batches):                # 4 个 micro-batch
    loss = model(mb).sum() / accum_steps              # 关键:除以 accum_steps
    loss.backward()                                   # 梯度累加进 .grad,不清零
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 见 066
        opt.step()                                    # 凑满才更新
        opt.zero_grad()
```

```python
# —— CBS 公式的数值直觉:步数/样本量随 batch 怎么变 ——
B_noise, S_min = 2_000_000, 50_000        # 临界批=200万token, 最小步数=5万
for B in [200_000, 2_000_000, 20_000_000]:
    S = S_min * (1 + B_noise / B)         # 到目标 loss 的步数
    E = B * S                             # 处理的总样本(token)
    print(f"B={B/1e6:>5.1f}M  步数={S/1e4:>5.1f}万  样本={E/1e9:>6.0f}B")
# 临界点 B=2M:步数=10万(=2·S_min),样本=2·E_min —— 时间/算力平衡
# B=20M(远超CBS):步数仅降到5.5万(快不了多少),样本暴涨(纯烧数据)

# —— 有效 batch = DP 并行度 × micro_bsz × accum_steps ——
def effective_batch(dp, micro_bsz, accum, seqlen):
    samples = dp * micro_bsz * accum
    return samples, samples * seqlen      # (样本数, token 数)
print(effective_batch(dp=64, micro_bsz=4, accum=2, seqlen=8192))
# (512 样本, 4.19M token);GPT-3 的 ~320 万 token/batch 即靠 DP×accum 共同达成
# ✅ 三者乘积决定有效 batch;显存不够时把 micro_bsz 调小、accum 调大补偿
```

## 面试高频

- **什么是 critical batch size?** 加大 batch 还能近似线性减少训练步数的上限;过了它,继续加 batch 几乎不省步数、只浪费算力(token 效率下降)。
- **CBS 由什么决定、会变吗?** 由梯度噪声尺度决定,噪声越大 CBS 越大;训练越往后、loss 越低,CBS 通常越大(所以有些训练用 batch ramp-up 逐步加大)。
- **梯度累积和真大 batch 等价吗?** 数学上等价(梯度统计相同),但吞吐略低(多次前/反向);它用时间换显存,让小卡跑大有效 batch。
- **梯度累积有哪些坑?** 累积期间不能 step/zero_grad;loss 必须除以 accum_steps;含 BatchNorm 的模型统计量不等价(LLM 用 LayerNorm/RMSNorm 无此问题)。
- **batch 翻倍学习率怎么调?** 线性缩放规则:大致同倍放大 lr,并配更长 warmup;但接近 CBS 时改为亚线性。
- **大 batch 训练为什么常见?** 数据并行下大 batch 直接缩短墙钟时间;只要不越过 CBS,就是「免费的加速」。见 [[070 ZeRO 与 FSDP|ZeRO/FSDP]]。
- **有效 batch 怎么算?** 有效 batch = DP 并行度 × micro_bsz × accum_steps;再乘序列长得 token 数。GPT-3 的 ~320 万 token/batch 即靠数据并行 × 梯度累积共同达成;显存不够就 micro_bsz 调小、accum 调大。
- **为什么要 batch ramp-up?** CBS 随训练后期变大(loss 越低噪声相对越大),前期小 batch 已够用且 token 效率高,后期加大 batch 才不浪费;GPT-3 从 ~3.2 万 token 起步升到 ~320 万。
- **超过 CBS 会怎样?** 步数几乎不再减少(墙钟到顶),但处理的样本量线性暴涨——纯烧数据/算力不换速度,token 效率骤降。

## 关键事实

- critical batch size 与梯度噪声尺度由 McCandlish, Kaplan, Amodei et al.《An Empirical Model of Large-Batch Training》(2018,arXiv:1812.06162)提出;GPT-3 即据此估 CBS 并用 batch ramp-up。
- 线性缩放规则(batch×k ⇒ lr×k)+ 渐进 warmup,见 Goyal et al.《Accurate, Large Minibatch SGD》(2017,arXiv:1706.02677);仅在噪声主导区成立。
- 近期工作重审 CBS 与模型/数据规模的关系:Zhang et al.《How Does Critical Batch Size Scale in Pre-training?》(2024,arXiv:2410.21676);CBS 主要随数据量增长。
- 实务上有效 batch 常达数百万 token(GPT-3 约 320 万 token/batch),靠数据并行 × 梯度累积共同达成。
