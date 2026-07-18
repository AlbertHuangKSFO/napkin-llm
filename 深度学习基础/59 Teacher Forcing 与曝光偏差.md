[[59 Teacher Forcing 与曝光偏差|Teacher Forcing]]是训练自回归解码器的标准做法——每步喂**真实**的上一个词而非模型自己的预测,训练快又稳;代价是训练(看真词)与推理(看自己预测)不一致,即**曝光偏差(exposure bias)**。

## 直觉

[[58 Seq2Seq 编码器与解码器|Seq2Seq]]解码器是自回归的:第 $t$ 步要用"上一步的词"当输入。这就有两种喂法:

- **Teacher Forcing(训练用)**:不管模型上一步预测对不对,都喂**数据集里的真实词** $y_{t-1}$。好比老师在每道题后立刻给标准答案,学生始终在正确轨道上学下一步。对**RNN 解码器**而言,这让每一步的输入确定,但隐藏状态仍递推,不能因此跨时间步并行；它的主要好处是稳定的监督信号与更易优化。
- **Free Running(推理时不得不用)**:没有标准答案,只能喂模型**自己上一步的预测** $\hat y_{t-1}$。

**矛盾(曝光偏差)**:训练时模型从没"见过自己犯错后的状态"——它只在"完美前缀"上练过。一旦推理时某步预测错了,就进入一个训练中从未出现过的状态分布,**错误像滚雪球一样沿生成链放大**,后面越错越离谱。训练分布 ≠ 推理分布,这就是曝光偏差。

## 例子

目标译文:`I love cats <EOS>`。

**训练(Teacher Forcing)**,即便模型第 2 步把 love 预测成了 like:
| 步 | 喂入(真实词) | 模型预测 | loss 对照 |
|---|---|---|---|
| 1 | `<BOS>` | I ✓ | vs I |
| 2 | I(真) | **like** ✗ | vs love |
| 3 | **love**(真,不是 like) | cats ✓ | vs cats |
| 4 | cats(真) | `<EOS>` ✓ | vs `<EOS>` |
第 3 步喂的是**真实的 love**,模型立刻被"拉回正轨",照样学对后面。

**推理(自由生成)**,同一个模型:
| 步 | 喂入(自己的预测) | 输出 |
|---|---|---|
| 1 | `<BOS>` | I |
| 2 | I | **like** ✗ |
| 3 | **like**(错的!) | …这状态训练时没见过 → 可能输出 dogs |
| 4 | dogs | … 越跑越偏 |
第 2 步一错,第 3 步喂进去的是 like,模型进入陌生状态,**错误累积**——这就是曝光偏差的现场。

**误差累积的量化直觉**。设每步独立正确率 $p=0.9$。Teacher Forcing 训练时每步都从真实前缀出发,所以"每步准确率"就是 0.9,各步互不拖累。但自由生成时**一步错就污染后面所有步**,生成长度 $T=10$ 的整句全对的概率 $\approx0.9^{10}\approx0.349$ —— 即便单步 90% 准,长句整体也常出错,且错误一旦发生就把模型推入训练中没见过的状态、后续更容易接着错。这就是"训练分布(完美前缀)≠ 推理分布(可能含错前缀)"的代价。

**Scheduled Sampling 退火手算**。设 $\epsilon$ 从 1.0 线性退火到 0(训练 10 个 epoch)。第 0 epoch $\epsilon=1.0$(全喂真词,等于纯 Teacher Forcing);第 5 epoch $\epsilon=0.5$(一半步喂真词、一半喂模型自己的预测);第 9 epoch $\epsilon\approx0.1$(主要喂预测,逼近推理分布)。这是 Bengio 等人(2015)提出的**启发式折中**:让模型从"温室"逐步过渡到"野外"；但它改变了原本的最大似然目标,并非无偏的一致估计。

![[rnn-teacher_forcing.png]]

![[Teacher Forcing-曝光偏差闭环.png]]

## 原理

**训练目标**:最大化真实序列的条件似然(等价 [[30 交叉熵与负对数似然|交叉熵]]):
$$\mathcal L=-\sum_{t=1}^{T}\log P_\theta(y_t\mid \underbrace{y_{<t}}_{\text{真实前缀}},x)$$
注意条件里是**真实前缀 $y_{<t}$**,这正是 Teacher Forcing。它定义了**训练数据如何移位喂入**；RNN 仍需按状态递推。Transformer 则没有循环状态，能把整条已右移的真实目标序列一次送入，并以因果掩码禁止位置 $t$ 看未来位置，故可按**位置并行**计算损失。并行来自 Transformer 的无循环计算图与掩码，不是 Teacher Forcing 单独带来的能力。

**推理目标**:无真实词,只能自回归
$$\hat y_t=\arg\max_{y}P_\theta(y\mid \underbrace{\hat y_{<t}}_{\text{自己的预测}},x)$$
条件里换成了**模型自己的预测前缀**。

**偏差来源**:训练里 $P_\theta$ 永远在"真实前缀"诱导的状态分布上被优化,从未在"模型自生成前缀"的分布上训练;推理时分布漂移 → 误差累积。这是一种**分布不匹配(train/test mismatch)**。

**缓解手段**:
- **Scheduled Sampling(计划采样,Bengio 2015)**:训练中以概率 $\epsilon$ 喂真实词、以 $1-\epsilon$ 喂模型自己的预测,$\epsilon$ 随训练**逐步退火**,让模型慢慢适应"看自己预测"的状态。简单有效,但理论上目标有偏。
- **序列级训练**:用 [[32 Agentic RL 与训练|强化学习]] / 最小风险训练直接优化 BLEU 等序列指标(如 MIXER、Actor-Critic),让模型在自生成轨迹上学。
- **Beam search** 推理:保留 top-$k$ 路径,降低单步错误的破坏力(治标)。
- **当前实践(截至 2026-07)**:主流 Transformer 语言模型预训练仍以右移真值 token 的 next-token 最大似然为主；它不是 Scheduled Sampling。是否采用序列级目标、偏好优化或 RL 取决于任务与后训练阶段，不能把这些方法都说成曝光偏差的通用解药。

**Transformer 的位置并行，和 RNN Teacher Forcing 不是一回事**。训练时真实前缀全已知，Transformer 可把右移后的序列 $[\text{BOS},y_1,\ldots,y_{T-1}]$ 一次输入；因果掩码令位置 $t$ 只可注意到 $<t$，每个位置预测 $y_t$，所以一轮前向可并行计算全部位置的 loss。RNN 即使用同样的真值前缀，$h_t=f(h_{t-1},y_{t-1})$ 仍有时间依赖。推理时两者都缺少未来真值，标准自回归解码仍逐词串行；这才是 LLM 常见的“训练吞吐高、解码受串行依赖限制”。

**曝光偏差到底有多严重(学界争议)**。它确实描述了“真值前缀训练、自生成前缀测试”的失配，但严重程度随任务、解码策略与模型能力而变；He 等(2019)发现其影响可小于常见直觉。Scheduled Sampling 的目标偏置与不稳定收益也限制了其通用性。因此更严谨的结论是：**纯最大似然/Teacher Forcing 仍是自回归 LM 预训练的标准基线；曝光偏差是要按任务实证诊断的问题，而非自动证明必须混入模型采样。**

## 代码

```python
import numpy as np
rng = np.random.default_rng(0)

def decode_train_tf(targets, step):
    """RNN Teacher Forcing:每步输入用真实上一词；状态仍按时间递推。"""
    inputs, loss = [], 0.0
    prev = "<BOS>"
    for t, y_true in enumerate(targets):
        pred, p_true = step(prev, t)         # 模型预测 + 真实词概率
        loss += -np.log(p_true + 1e-9)       # 交叉熵
        prev = y_true                        # ✅ 喂真实词,不喂 pred
    return loss

def decode_infer(step, max_len=4):
    """推理:只能喂自己的预测,错误会沿链累积(曝光偏差)"""
    out, prev = [], "<BOS>"
    for t in range(max_len):
        pred, _ = step(prev, t)
        out.append(pred); prev = pred        # ❌ 只能喂自己的预测
    return out

def decode_scheduled(targets, step, eps):
    """计划采样:以概率 eps 喂真实词,否则喂预测,缓解偏差"""
    prev, loss = "<BOS>", 0.0
    for t, y_true in enumerate(targets):
        pred, p_true = step(prev, t)
        loss += -np.log(p_true + 1e-9)
        prev = y_true if rng.random() < eps else pred   # 退火:eps 从 1 慢慢降
    return loss

# 关键对照:
# ❌ 训练全程 Teacher Forcing,推理却自由生成 → 训练/推理分布不一致(曝光偏差)
# ✅ Scheduled Sampling:训练时混入模型自己的预测,eps 随 epoch 退火逼近推理分布
print("eps 退火示意:", [round(1 - i / 10, 2) for i in range(6)])  # 1.0→0.5 渐降

# Transformer 的位置并行来自无循环计算图 + 因果掩码；不是 RNN Teacher Forcing 本身
import numpy as np
def causal_mask(T):
    m = np.triu(np.ones((T, T)), k=1)        # 上三角(未来位置)= 1
    return np.where(m == 1, -np.inf, 0.0)    # 未来位置加 -inf,softmax 后≈0
print("因果掩码(T=4):\n", causal_mask(4))
# ✅ Transformer:右移真值序列 + 因果掩码 = 所有位置损失一次并行算出
# ❌ 自生成前缀需先得到上一步 token,标准解码仍串行
# 推理时没有真实前缀,只能逐词串行 → LLM "训练快、推理慢" 的根源
```

要点:RNN Teacher Forcing 喂真词以提供稳定监督,但 RNN 状态仍串行；Transformer 的位置并行来自无循环架构与因果掩码。计划采样是有偏的启发式，是否使用应由任务验证决定。

## 面试高频

- **"什么是 Teacher Forcing?为什么用它?"** 训练自回归解码器时每步喂**真实**上一词而非模型预测;它提供稳定的真值前缀监督、通常更易优化。RNN 的隐藏状态依旧按时间递推。
- **"什么是曝光偏差?根因?"** 训练只在真实前缀上优化,推理却用自己的预测前缀;一旦出错就进入训练中未见的状态,误差沿链累积。本质是 train/test 分布不匹配。
- **"怎么缓解曝光偏差?"** 先按任务测量失配；可评估 Scheduled Sampling、序列级目标或解码策略，但 Scheduled Sampling 有偏，beam search 也不是从训练上消除失配。
- **"为什么 Transformer 训练能并行?"** 因为无循环状态，右移真值序列可整体输入，因果掩码限制每个位置只看过去，故所有位置 loss 可同时算；这与 RNN Teacher Forcing 的逐状态递推不同。
- **"Scheduled Sampling 的缺点?"** 它把自采样前缀混入训练，改变了原始 MLE 目标，可能导致有偏/不一致，效果应实证验证。
- **"曝光偏差在大模型时代还严重吗?"** 是真实的分布失配，但大小依任务、模型与解码而变；不能笼统说被高估或必然是瓶颈。主流 LM 预训练仍以真值 next-token MLE 为基线。
- **"为什么 LLM 训练快推理慢?"** 训练能在因果掩码下并行处理已知真值 token；生成时下一个 token 依赖刚生成的前缀，标准自回归解码有串行瓶颈。
- **"误差累积有多大?用概率估一下。"** 单步准确率 $p$、长度 $T$,自由生成整句全对约 $p^T$:$p=0.9,T=10$ 仅 $\approx0.35$。一步错还会污染后续状态、加剧后面出错。
- **"Teacher Forcing 和因果掩码什么关系?"** 前者是“下一步条件于真值前缀”的训练数据/目标约定；后者是 Transformer 防止未来泄漏的结构约束。二者可一起用于 Transformer，但位置并行还依赖无循环计算图。

## 关键事实

- **Teacher Forcing** 术语源自经典 RNN 训练文献(Williams & Zipser, *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*,Neural Computation, 1989),指用目标值替代网络输出作为下一步输入。
- **Scheduled Sampling**:Bengio, Vinyals, Jaitly & Shazeer, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*,NeurIPS 2015——提出训练中按退火概率混入模型自己的预测以缓解曝光偏差。
- **曝光偏差与序列级训练**:Ranzato et al., *Sequence Level Training with Recurrent Neural Networks*(MIXER, ICLR 2016)系统讨论曝光偏差并用 RL 直接优化序列指标。
- **曝光偏差危害被高估的反思**:He et al., *Quantifying Exposure Bias for Neural Language Generation*(2019)等工作指出其影响常被夸大,纯 MLE 训练已足够强。
- **因果掩码的自回归并行**:Vaswani et al.(2017)解码器用三角掩码;GPT(Radford et al., 2018/2019)即纯因果 LM + Teacher Forcing 预训练。
- **当前预训练实践(截至 2026-07)**:Touvron et al., *The Llama 3 Herd of Models*(2024)明确将预训练描述为 next-token prediction；这是右移真值 token 的最大似然训练。它支持“主流 LM 预训练仍以 Teacher Forcing 形式的 next-token 目标为基线”，不等于每个后训练阶段都不用序列级或偏好目标。
