[[59 Teacher Forcing 与曝光偏差|Teacher Forcing]]是训练自回归解码器的标准做法——每步喂**真实**的上一个词而非模型自己的预测,训练快又稳;代价是训练(看真词)与推理(看自己预测)不一致,即**曝光偏差(exposure bias)**。

## 直觉

[[58 Seq2Seq 编码器与解码器|Seq2Seq]]解码器是自回归的:第 $t$ 步要用"上一步的词"当输入。这就有两种喂法:

- **Teacher Forcing(训练用)**:不管模型上一步预测对不对,都喂**数据集里的真实词** $y_{t-1}$。好比老师在每道题后立刻给标准答案,学生始终在正确轨道上学下一步。**好处**:① 每步输入已知,可一次性并行算所有时间步;② 误差不沿生成链累积,梯度信号干净,收敛快。
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

**Scheduled Sampling 退火手算**。设 $\epsilon$ 从 1.0 线性退火到 0(训练 10 个 epoch)。第 0 epoch $\epsilon=1.0$(全喂真词,等于纯 Teacher Forcing);第 5 epoch $\epsilon=0.5$(一半步喂真词、一半喂模型自己的预测);第 9 epoch $\epsilon\approx0.1$(主要喂预测,逼近推理分布)。**让模型从"温室"逐步过渡到"野外"**,慢慢适应"看自己的错误前缀",缩小训练/推理鸿沟。

![[rnn-seq2seq瓶颈.png]]

## 原理

**训练目标**:最大化真实序列的条件似然(等价 [[30 交叉熵与负对数似然|交叉熵]]):
$$\mathcal L=-\sum_{t=1}^{T}\log P_\theta(y_t\mid \underbrace{y_{<t}}_{\text{真实前缀}},x)$$
注意条件里是**真实前缀 $y_{<t}$**,这正是 Teacher Forcing:每步的输入历史来自 ground truth,所以所有时间步可**并行**计算(Transformer 训练用因果掩码一次算全序列,正是它)。

**推理目标**:无真实词,只能自回归
$$\hat y_t=\arg\max_{y}P_\theta(y\mid \underbrace{\hat y_{<t}}_{\text{自己的预测}},x)$$
条件里换成了**模型自己的预测前缀**。

**偏差来源**:训练里 $P_\theta$ 永远在"真实前缀"诱导的状态分布上被优化,从未在"模型自生成前缀"的分布上训练;推理时分布漂移 → 误差累积。这是一种**分布不匹配(train/test mismatch)**。

**缓解手段**:
- **Scheduled Sampling(计划采样,Bengio 2015)**:训练中以概率 $\epsilon$ 喂真实词、以 $1-\epsilon$ 喂模型自己的预测,$\epsilon$ 随训练**逐步退火**,让模型慢慢适应"看自己预测"的状态。简单有效,但理论上目标有偏。
- **序列级训练**:用 [[32 Agentic RL 与训练|强化学习]] / 最小风险训练直接优化 BLEU 等序列指标(如 MIXER、Actor-Critic),让模型在自生成轨迹上学。
- **Beam search** 推理:保留 top-$k$ 路径,降低单步错误的破坏力(治标)。
- **更强的模型 + 大数据**:实践中 Transformer + 海量数据使曝光偏差影响被大幅稀释,纯 Teacher Forcing 已足够好——这也是为何现代 LLM 训练几乎清一色 Teacher Forcing。

**为什么 Teacher Forcing 能让 Transformer 并行训练(关键机制)**。自回归生成本是逐词串行的,但训练时**真实前缀全部已知**,于是可以把整条目标序列一次性喂进去,用**因果掩码(causal mask)**挡住每个位置对未来词的注意力,让"第 $t$ 个位置只看 $<t$"——所有位置的损失在一次前向里同时算出。这正是 Transformer 训练比 RNN 快几个数量级的根本原因:**Teacher Forcing + 因果掩码 = 训练全并行**。推理时没有真实前缀,只能逐词串行(无法并行),所以 LLM "训练快、推理慢"。

**曝光偏差到底有多严重(学界争议)**。早期普遍认为它是 Seq2Seq 性能瓶颈;但近年观点更冷静:① 大模型 + 海量数据下,模型在自己的高质量前缀上也很稳,影响有限;② Scheduled Sampling 引入了**目标偏置**(训练目标不再是干净的极大似然),收益不稳定,现代大模型基本弃用;③ 真正缓解长文本退化的,更多是更好的解码策略(采样而非贪心/beam)和 RLHF 等后训练。所以面试可以这样收口:**曝光偏差是真实存在的训练/推理失配,但在大模型时代它的危害被显著高估,纯 Teacher Forcing 仍是事实标准。**

![[attn-Bahdanau对齐.png]]

## 代码

```python
import numpy as np
rng = np.random.default_rng(0)

def decode_train_tf(targets, step):
    """Teacher Forcing:每步输入用真实上一词,可并行、误差不累积"""
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

# Transformer 训练并行的核心:因果掩码,让位置 t 只看 <t,一次算全序列
import numpy as np
def causal_mask(T):
    m = np.triu(np.ones((T, T)), k=1)        # 上三角(未来位置)= 1
    return np.where(m == 1, -np.inf, 0.0)    # 未来位置加 -inf,softmax 后≈0
print("因果掩码(T=4):\n", causal_mask(4))
# ✅ Teacher Forcing(真实前缀) + 因果掩码 = 所有位置损失一次并行算出 → 训练快
# ❌ 若喂模型自己的预测,则必须串行(下一步依赖上一步输出),无法并行
# 推理时没有真实前缀,只能逐词串行 → LLM "训练快、推理慢" 的根源
```

要点:Teacher Forcing 喂真词(并行、稳),推理喂预测(误差累积);计划采样通过退火 $\epsilon$ 把训练分布慢慢拉向推理分布。

## 面试高频

- **"什么是 Teacher Forcing?为什么用它?"** 训练自回归解码器时每步喂**真实**上一词而非模型预测;好处是可并行、收敛快、梯度信号干净。
- **"什么是曝光偏差?根因?"** 训练只在真实前缀上优化,推理却用自己的预测前缀;一旦出错就进入训练中未见的状态,误差沿链累积。本质是 train/test 分布不匹配。
- **"怎么缓解曝光偏差?"** Scheduled Sampling(训练混入预测、$\epsilon$ 退火)、序列级 RL/最小风险训练、beam search、加大模型+数据。
- **"为什么 Transformer 训练能并行?"** 用 Teacher Forcing + 因果掩码,每步条件于真实前缀,所有时间步同时算;若要喂自己的预测就无法并行(推理仍逐词)。
- **"Scheduled Sampling 的缺点?"** 训练目标变得有偏(条件分布混入模型自采样),可能引入不一致;现代大模型多直接用 Teacher Forcing 而非它。
- **"曝光偏差在大模型时代还严重吗?"** 危害被显著高估。大模型 + 海量数据让模型在自己前缀上也稳;Scheduled Sampling 有目标偏置且收益不稳;纯 Teacher Forcing 仍是事实标准,长文本退化更多靠解码策略和后训练缓解。
- **"为什么 LLM 训练快推理慢?"** 训练用 Teacher Forcing + 因果掩码,全序列一次并行算损失;推理无真实前缀,必须逐词串行(每步依赖上一步输出),无法并行。
- **"误差累积有多大?用概率估一下。"** 单步准确率 $p$、长度 $T$,自由生成整句全对约 $p^T$:$p=0.9,T=10$ 仅 $\approx0.35$。一步错还会污染后续状态、加剧后面出错。
- **"Teacher Forcing 和因果掩码什么关系?"** Teacher Forcing 提供"每步条件于真实前缀"的训练信号;因果掩码是它在 Transformer 里的并行实现手段(挡住未来),二者配合实现全序列并行训练。

## 关键事实

- **Teacher Forcing** 术语源自经典 RNN 训练文献(Williams & Zipser, *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*,Neural Computation, 1989),指用目标值替代网络输出作为下一步输入。
- **Scheduled Sampling**:Bengio, Vinyals, Jaitly & Shazeer, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*,NeurIPS 2015——提出训练中按退火概率混入模型自己的预测以缓解曝光偏差。
- **曝光偏差与序列级训练**:Ranzato et al., *Sequence Level Training with Recurrent Neural Networks*(MIXER, ICLR 2016)系统讨论曝光偏差并用 RL 直接优化序列指标。
- **曝光偏差危害被高估的反思**:He et al., *Quantifying Exposure Bias for Neural Language Generation*(2019)等工作指出其影响常被夸大,纯 MLE 训练已足够强。
- **因果掩码的自回归并行**:Vaswani et al.(2017)解码器用三角掩码;GPT(Radford et al., 2018/2019)即纯因果 LM + Teacher Forcing 预训练。
- 现代实践:Transformer 解码器训练用 Teacher Forcing + 因果掩码实现全序列并行(见 Seq2Seq 与后续 Transformer);LLM 预训练即逐词 Teacher Forcing。
