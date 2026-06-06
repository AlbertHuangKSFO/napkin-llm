[[57 GRU 与变体|门控循环单元(GRU)]]是 [[56 LSTM 门控机制|LSTM]] 的精简版:把三门压成两门(更新门 + 重置门)、去掉独立的 cell 状态(用 $h$ 兼当记忆),参数更少、训练更快,效果常与 LSTM 相当。

## 直觉

LSTM 强但"零件多":三道门 + 独立 cell,参数和计算都不小。GRU(Cho 2014)做了两处合并:

1. **遗忘门和输入门合二为一 → 更新门 $z_t$**。直觉:"保留多少旧记忆"和"写入多少新记忆"本是一枚硬币的两面,何必两个独立旋钮?GRU 用一个 $z_t$ 做**凸组合**:留 $(1-z_t)$ 的旧、写 $z_t$ 的新,自动此消彼长。
2. **去掉独立 cell,$h_t$ 同时当记忆和输出**;另设**重置门 $r_t$** 控制"计算新候选时,旧状态参与多少"——$r_t\approx0$ 就丢掉历史、只看当前输入(适合序列起点/话题切换)。

结果:GRU 比 LSTM **少一道门、少一组状态**,参数约少 1/4,在中小数据/短序列上常更快更稳,效果旗鼓相当。

**三者一图看清(朴素 RNN → GRU → LSTM)**:复杂度和能力递增。
- **朴素 RNN**:$h_t=\tanh(W[h_{t-1},x_t])$,1 组权重,无门,长依赖必崩。
- **GRU**:2 门(更新 $z$ + 重置 $r$),3 组权重,$h$ 兼记忆与输出,凸组合更新抗消失。
- **LSTM**:3 门(遗忘 $f$ + 输入 $i$ + 输出 $o$),4 组权重,独立 cell $c$ 与 $h$ 解耦,加性更新抗消失。
门越多越灵活也越重;GRU 是"够用且快"的折中,LSTM 是"最全但最重"的上限。

## 例子

承接 LSTM 例子:把值 0.8 长期保存在 $h$ 里。GRU 用更新门 $z_t$ 控制:
$$h_t=(1-z_t)\,h_{t-1}+z_t\,\tilde h_t$$
若网络学到 $z_t=0$(关闭写入):
$$h_t=(1-0)\cdot 0.8+0\cdot\tilde h_t=0.8$$
**$z=0$ 时 $h$ 原样直通**,跨多步不衰减——和 LSTM 遗忘门 $f=1$ 异曲同工。反向时 $\frac{\partial h_t}{\partial h_{t-1}}\supseteq(1-z_t)=1$,提供近恒等的梯度通道。

再看重置门:算 $\tilde h_t=\tanh(W[\,r_t\odot h_{t-1},\,x_t])$。话题切换时学到 $r_t\approx0$,候选 $\tilde h_t=\tanh(W[\,0,\,x_t])$ 只依赖当前词,**干净地"重置"上下文**。

**完整一步手算(更新门 + 重置门一起走)**。设 $h_{t-1}=0.6$,算得 $z_t=0.2$(主要保留旧的)、$r_t=0.9$(历史大部分参与候选)、候选 $\tilde h_t=\tanh(\dots)=0.5$。
- 凸组合更新:$h_t=(1-z_t)h_{t-1}+z_t\tilde h_t=0.8\times0.6+0.2\times0.5=0.48+0.10=0.58$。
- 解读:$z=0.2$ 小,所以 $h$ 主要保留旧值(0.6→0.58 只动一点);若 $z=0.9$ 则 $h_t=0.06+0.45=0.51$,新候选占主导。**一个旋钮 $z$ 同时控制"留旧"和"写新",此消彼长**,这是 GRU 比 LSTM(遗忘门 + 输入门两个独立旋钮)省的地方。

**梯度跨步手算(对应 LSTM 的 $f$)**。GRU 沿 $h$ 的梯度主项是 $\frac{\partial h_t}{\partial h_{t-1}}\supseteq(1-z_t)$。若连续三步 $z=[0.1,0,0.1]$,则保留系数 $(1-z)=[0.9,1.0,0.9]$,跨 3 步 $\approx0.9\times1.0\times0.9=0.81$ —— 和 LSTM 遗忘门 $f\approx1$ 一样,提供近恒等的梯度通道。

![[rnn-GRU门.svg]]

## 原理

时刻 $t$,输入 $x_t$ 和上一隐状态 $h_{t-1}$:
$$z_t=\sigma(W_z[h_{t-1},x_t])\quad\text{更新门:旧 vs 新 的开关}$$
$$r_t=\sigma(W_r[h_{t-1},x_t])\quad\text{重置门:历史参与候选的比例}$$
$$\tilde h_t=\tanh\big(W[\,r_t\odot h_{t-1},\,x_t\,]\big)\quad\text{候选状态(重置后的历史 + 当前输入)}$$
$$\boxed{\,h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t\,}\quad\text{凸组合更新}$$

**和 LSTM 的对应关系**:
| | LSTM | GRU |
|---|---|---|
| 门数 | 3(遗忘/输入/输出) | 2(更新/重置) |
| 状态 | $c$(记忆)+ $h$(输出),解耦 | 只有 $h$,身兼二职 |
| 写入控制 | $c_t=f\odot c_{t-1}+i\odot\tilde c$(两旋钮) | $h_t=(1-z)h_{t-1}+z\tilde h$(一旋钮凸组合) |
| 输出筛选 | 输出门 $o$ | 无单独输出门 |

**为什么也能抗梯度消失**:更新公式里 $h_{t-1}$ 有一条系数为 $(1-z_t)$ 的**加性直通**项,$z_t\to0$ 时 $\frac{\partial h_t}{\partial h_{t-1}}\to 1$,梯度近恒等流过——和 LSTM 的 cell 高速路、[[52 残差连接与深度可训练性|残差连接]]是同一类"加性旁路"机制。

**常见变体**:
- **Minimal GRU / MGU**:进一步只留一个门(把重置门去掉或与更新门合并),参数更省,小任务上够用;
- **双向 RNN(Bi-LSTM/Bi-GRU)**:正反两个方向各跑一遍再拼接,让每个位置同时看到左右上下文——序列标注/NER 标配;**但生成式自回归任务不能用**(会偷看未来,造成训练/推理不一致);
- **多层(堆叠)RNN**:把上一层每步的 $h$ 当下一层输入,加深表达力,常 2–4 层;
- **加 LayerNorm 的 GRU/LSTM**:在门内归一化激活,稳定深层/长序列训练;
- **卷积门控(ConvGRU/ConvLSTM)**:把门里的全连接换成卷积,处理时空数据(视频、气象);
- **LSTM vs GRU 怎么选**:Chung 2014 实证两者相当、无绝对赢家;数据多/序列长偏 LSTM,资源紧/要快偏 GRU——**实测为准**。

**整体定位(RNN 家族的句号)**:GRU/LSTM 是 2014–2017 年 NLP 的主力,之后被 Transformer 大面积取代(注意力可并行、长程一跳直达,见 [[60 注意力机制的起源(Bahdanau、Luong)|注意力起源]])。但在**算力受限、序列不太长、流式/在线推理(逐步到达、低延迟)**场景,门控 RNN 仍有价值;近年的状态空间模型(S4/Mamba)某种意义上是"现代化的、可并行的 RNN",让循环结构重新受到关注。

![[rnn-时间展开.svg]]

## 代码

```python
import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))

def gru_step(x, h, Wz, Wr, Wh):
    xh = np.concatenate([h, x])
    z = sigmoid(Wz @ xh)                       # 更新门
    r = sigmoid(Wr @ xh)                       # 重置门
    h_tilde = np.tanh(Wh @ np.concatenate([r * h, x]))   # 候选:重置后的历史
    return (1 - z) * h + z * h_tilde           # ✅ 凸组合:z=0 直通旧 h,z=1 全用新

H, D = 1, 1
Wz = np.full((H, H+D), -6.0)                   # z≈0 → 几乎只保留旧状态
Wr = np.zeros((H, H+D)); Wh = np.zeros((H, 2*H))
h = np.array([0.8])
for t in range(50):
    h = gru_step(np.zeros(D), h, Wz, Wr, Wh)
print("GRU 50 步后 h =", round(h[0], 4))       # ≈0.8,信息保住 ✅

# 参数量对比(隐藏维 H,输入维 D):忽略 bias
def n_params(H, D, kind):
    if kind == "lstm": return 4 * H * (H + D)   # 4 组权重
    if kind == "gru":  return 3 * H * (H + D)   # 3 组权重(z,r,候选)
for H_ in [128]:
    print(f"H={H_}: LSTM={n_params(H_,H_,'lstm')}  GRU={n_params(H_,H_,'gru')}")
    # GRU 约为 LSTM 的 3/4

# ❌ 误区:GRU 一定比 LSTM 差/好 —— 实证两者相当,看任务和数据量
# ✅ 默认 baseline 用 GRU(更快),长序列/大数据再试 LSTM

# PyTorch:GRU 与双向 GRU 一行用
import torch, torch.nn as nn
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
out, h_n = gru(torch.randn(4, 35, 10))
print("GRU 输出:", out.shape, " 末隐:", h_n.shape)   # [4,35,20] [2,4,20]

bigru = nn.GRU(10, 20, bidirectional=True, batch_first=True)
o, _ = bigru(torch.randn(4, 35, 10))
print("双向 GRU 输出:", o.shape)   # [4,35,40] 正反拼接
# ❌ 双向用于自回归生成 → 偷看未来,训练/推理不一致;仅用于非生成任务
```

代码验证 GRU 同样能用 $z\approx0$ 把 0.8 保 50 步不丢;参数量对比清楚显示 GRU 是 LSTM 的约 3/4。

## 面试高频

- **"GRU 和 LSTM 的核心区别?"** GRU 两门(更新+重置)、无独立 cell($h$ 兼记忆);LSTM 三门 + 独立 cell。GRU 参数更少(约 3/4)、更快;效果常相当。
- **"更新门 $z$ 的作用?"** 凸组合控制"保留旧 $h$ vs 写入新候选":$h_t=(1-z)h_{t-1}+z\tilde h_t$,一个旋钮替代 LSTM 的遗忘+输入两门。
- **"重置门 $r$ 的作用?"** 控制计算候选 $\tilde h$ 时旧状态的参与度;$r\approx0$ 丢弃历史、只看当前输入,适合话题切换/序列起点。
- **"GRU 为什么也抗梯度消失?"** 更新公式含 $(1-z)h_{t-1}$ 的加性直通,$z\to0$ 时近恒等梯度通道。
- **"实际怎么选 GRU / LSTM?"** 无定论(Chung 2014);资源紧/序列短用 GRU,数据大/序列长可试 LSTM,以验证集为准。
- **"双向 RNN 解决什么?"** 让每个位置同时获得左右上下文,适合非生成式任务(序列标注、NER);生成式自回归任务不能用(会看到未来)。
- **"GRU 比 LSTM 少了哪个机制?有何影响?"** 少了独立 cell 和输出门($h$ 兼记忆与输出)。影响:参数少约 1/4、更快;但 LSTM 的"长期记忆 $c$ 与对外 $h$ 解耦"在超长依赖/大数据上偶有优势。
- **"为什么用凸组合 $(1-z)h+z\tilde h$ 而不是两个独立门?"** 凸组合天然保证"留旧 + 写新 = 1",省一个门且自动此消彼长;LSTM 的 $f$ 和 $i$ 独立则更灵活但更重。
- **"门控 RNN 现在还用吗?"** 大模型/长序列已被 Transformer 取代;但流式/低延迟/算力受限场景仍用,且状态空间模型(Mamba)让"可并行的循环"重新流行。
- **"RNN 家族都靠什么抗梯度消失?"** 共同点是"加性/门控旁路":LSTM 的 cell 加性更新、GRU 的 $(1-z)$ 直通,都让梯度有近恒等通道,与残差连接同源。

## 关键事实

- **GRU 出处**:Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk & Bengio, *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*,EMNLP 2014(arXiv:1406.1078)——在 RNN Encoder-Decoder 框架里首次提出 GRU。这篇也是 [[58 Seq2Seq 编码器与解码器|seq2seq]] 与注意力的同源工作。
- **系统对比**:Chung, Gulcehre, Cho & Bengio, *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*(arXiv:1412.3555, 2014)——LSTM 与 GRU 均显著优于朴素 tanh-RNN,两者之间无一致赢家。
- **双向 RNN**:Schuster & Paliwal, *Bidirectional Recurrent Neural Networks*,IEEE Trans. Signal Processing(1997)。
- **变体搜索**:Greff et al.(2017)、Jozefowicz et al.(ICML 2015)大规模架构搜索,结论是标准 LSTM/GRU 已接近最优,门控的"加性记忆通道"是关键。
- **ConvLSTM**:Shi et al.(NeurIPS 2015,降水临近预报)把门内全连接换成卷积处理时空数据。
- **状态空间模型(现代化的可并行 RNN)**:S4(Gu et al., ICLR 2022)、Mamba(Gu & Dao, 2023)——让循环结构在长序列上既高效又可并行,部分场景挑战 Transformer。
- **被 Transformer 取代的节点**:Vaswani et al.(2017)之后,NLP 主力从 RNN 转向自注意力(可并行 + 长程一跳),见 [[60 注意力机制的起源(Bahdanau、Luong)]]。
