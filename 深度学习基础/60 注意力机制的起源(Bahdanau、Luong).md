[[60 注意力机制的起源(Bahdanau、Luong)|注意力机制]]打掉了 [[58 Seq2Seq 编码器与解码器|Seq2Seq]] 的定长瓶颈:解码每一步不再只用一个固定 $c$,而是对编码器**所有**隐状态按相关性**加权求和**(软对齐),得到该步专属上下文——这正是 Transformer 自注意力的前身。

## 直觉

人翻译长句不会先把整句背成一个"压缩包",而是**译到哪、看源句的哪**:写 love 时眼睛盯着"爱",写 cats 时盯着"猫"。注意力就是把这个"边译边看"显式建模。

[[58 Seq2Seq 编码器与解码器|Seq2Seq]]的瓶颈是:整句被压成**一个**定长向量 $c$,长句信息被压垮。注意力的破局只有一句话:

> **别再用单一的 $c$。解码第 $t$ 步,用当前解码状态去"查询"编码器每个位置 $h_j$,算出一组权重 $\alpha_{tj}$(相关就大、无关就小),再 $c_t=\sum_j\alpha_{tj}h_j$——每步都现算一个专属上下文。**

三步:**打分(score)→ 归一化(softmax 得权重)→ 加权求和(得上下文)**。权重 $\alpha_{tj}$ 排成矩阵就是**对齐热图**,能看出译词和源词的软对应——首次让神经翻译"可解释"。瓶颈被解除:信息不再挤独木桥,而是按需直取源端任意位置。

## 例子

翻译"我 爱 猫 → I love cats",编码器给出 $h_1$(我)、$h_2$(爱)、$h_3$(猫)。解码到要生成 **love** 这一步,用上一解码状态 $s_{t-1}$ 去和每个 $h_j$ 打分(加性打分),假设得到原始分数:
$$e_1=0.2,\quad e_2=2.1,\quad e_3=0.5$$
过 [[27 Softmax 与温度|softmax]] 归一化成权重:
$$\alpha=\text{softmax}([0.2,2.1,0.5])\approx[0.10,\ 0.78,\ 0.12]$$
(算法:$\frac{e^{0.2}}{e^{0.2}+e^{2.1}+e^{0.5}}=\frac{1.22}{1.22+8.17+1.65}=\frac{1.22}{11.04}\approx0.11$,余类推,数值略。)
**权重集中在"爱"($\alpha_2=0.78$)**——正是 love 该对齐的词。上下文向量:
$$c_t=0.10\,h_1+0.78\,h_2+0.12\,h_3\ \approx\ h_2$$
解码器拿这个**聚焦于"爱"的 $c_t$** 去预测,自然输出 love。把每个目标词的 $\alpha$ 行堆起来,就是对齐热图(下图右),对角线附近发亮 = 词序大致单调对齐。

**softmax 权重的完整手算(把分数变权重)**。原始分数 $e=[0.2,2.1,0.5]$。
- 取指数:$e^{0.2}=1.221$,$e^{2.1}=8.166$,$e^{0.5}=1.649$;和 $=11.036$。
- 归一化:$\alpha_1=\frac{1.221}{11.036}=0.111$,$\alpha_2=\frac{8.166}{11.036}=0.740$,$\alpha_3=\frac{1.649}{11.036}=0.149$。
- 权重和 $=1$(✓),且**集中在分数最高的"爱"上**(0.74)。注意 softmax 的"赢家通吃"特性:分数差 1.9(2.1 vs 0.2)就让权重差近 7 倍——分数尺度越大越尖锐(这也是 Transformer 要除 $\sqrt d$ 防止点积过大、softmax 过尖的原因)。

**上下文向量手算**。$h_1=[1,0,0]$(我)、$h_2=[0,1,0]$(爱)、$h_3=[0,0,1]$(猫),用上面的 $\alpha=[0.111,0.740,0.149]$:
$$c_t=0.111\,h_1+0.740\,h_2+0.149\,h_3=[0.111,\,0.740,\,0.149]$$
最大分量在第 2 维(对应"爱")—— $c_t$ **几乎就是 $h_2$**,说明这一步上下文聚焦在"爱"上,解码器据此输出 love。

![[attn-Bahdanau对齐.png]]

## 原理

设编码器隐状态 $\{h_1,\dots,h_{T_x}\}$,解码第 $t$ 步的解码状态 $s_{t-1}$。

**① 打分**(query=$s_{t-1}$,被查=$h_j$):
$$e_{tj}=\text{score}(s_{t-1},h_j)$$

**② 归一化成注意力权重**(softmax 保证 $\sum_j\alpha_{tj}=1$):
$$\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_{k}\exp(e_{tk})}$$

**③ 加权求和得上下文**(这一步取代了固定 $c$):
$$c_t=\sum_{j}\alpha_{tj}\,h_j$$
再把 $c_t$ 喂进解码器算 $s_t$、预测 $y_t$。

**两种经典打分函数**:

- **Bahdanau 2014(加性/拼接注意力)**:用一个小前馈网络打分,query 与 key 维度可不同。
$$e_{tj}=v^\top\tanh(W_s\,s_{t-1}+W_h\,h_j)$$
特点:用 $s_{t-1}$(上一步状态)、双向编码器、加性结构,可学的对齐网络。这是**注意力机制的开山之作**,首次解除定长瓶颈。

- **Luong 2015(乘性注意力)**:打分更简洁高效,给出三种 score,并用 $s_t$(当前步)。
$$e_{tj}=\begin{cases}s_t^\top h_j & \text{dot(点积)}\\ s_t^\top W\,h_j & \text{general(一般)}\\ v^\top\tanh(W[s_t;h_j]) & \text{concat(拼接)}\end{cases}$$
还区分 **global**(看源句全部位置)与 **local**(只看预测出的一个窗口,省算力)注意力。**点积打分**计算最省,直接启发了后来的缩放点积注意力。

**加性 vs 乘性的取舍(高频对比)**:
- **加性(Bahdanau)**:$e=v^\top\tanh(W_s s+W_h h)$。query 和 key **维度可不同**(各自有投影矩阵),有额外参数 $v,W_s,W_h`;表达灵活,但每对 $(s,h)$ 要过一个小网络,**不易矩阵化、较慢**,高维下数值更稳。
- **乘性/点积(Luong)**:$e=s^\top h$(或 $s^\top W h`)。**无额外参数(dot 版)、能整批矩阵乘、最快**;但要求 query/key 同维,且**高维时点积方差大**($\sim d$),softmax 易进饱和区、梯度小——这正是 Transformer 加 $\frac1{\sqrt d}$ 缩放的动机。
结论:小模型/维度不齐用加性,大规模/可并行用缩放点积(现代主流)。

**global vs local 注意力(Luong)**:
- **global**:每步对源句**全部位置**算权重,信息全但长源句算量大。
- **local**:先预测一个对齐中心 $p_t$,只在 $[p_t-D,p_t+D]$ 窗口内算注意力,省算力、适合长序列;代价是窗口外信息看不到。

**注意力的"可解释性"陷阱**:把 $\alpha$ 画成热图很诱人,但 **注意力权重 ≠ 因果重要性**(Jain & Wallace 2019)——同一预测可能对应多组不同的注意力分布,高权重位置未必是模型"真正依赖"的。面试若被追问,要点出这一争议,别把热图当成模型决策的金标准。

**通往 Transformer(关键事实里再点)**:把 $s_{t-1}$ 抽象成 **query** $Q$,把 $h_j$ 既当 **key** $K$ 又当 **value** $V$,注意力就是 $\text{softmax}(QK^\top)V$。Transformer 做了两步质变:① 把打分换成**缩放点积** $\frac{QK^\top}{\sqrt{d}}$;② 让序列**内部**每个位置互相 attend(**自注意力**),彻底去掉 RNN 的循环,从而**可并行**、长程一跳直达。注意力从"RNN 的附件"变成"整个架构的主干"。

**Q/K/V 抽象(为后面 Transformer 打底)**:在 Bahdanau 里 query=解码状态 $s_{t-1}$、key=value=编码隐状态 $h_j`;Transformer 把三者**各自用一个投影矩阵**从同一输入生成($Q=XW_Q,K=XW_K,V=XW_V$),并区分 key(用来打分匹配)和 value(用来加权聚合)。"用 query 去匹配 key、按匹配度聚合 value"这个三件套,从 Bahdanau 的跨编解码注意力,一路用到 Transformer 的自注意力、交叉注意力,是理解一切现代注意力的统一模板。

![[rnn-seq2seq瓶颈.png]]

## 代码

```python
import numpy as np
rng = np.random.default_rng(0)

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

# 编码器隐状态(3 个源位置,4 维)
H = np.array([[1., 0, 0, 0],     # h1 "我"
              [0., 1, 0, 0],     # h2 "爱"
              [0., 0, 1, 0]])    # h3 "猫"
s = np.array([0., 1, 0, 0])      # 解码状态 s_{t-1}:正巧偏向"爱"的方向

# ✅ Bahdanau 加性打分:e = v^T tanh(Ws·s + Wh·h)
d = 4
Ws = rng.normal(0, .5, (d, d)); Wh = rng.normal(0, .5, (d, d)); v = rng.normal(0, .5, d)
e_add = np.array([v @ np.tanh(Ws @ s + Wh @ h) for h in H])
a_add = softmax(e_add)

# ✅ Luong 点积打分:e = s^T h(最省,Transformer 雏形)
e_dot = H @ s                    # [s·h1, s·h2, s·h3] = [0,1,0]
a_dot = softmax(e_dot)
print("点积权重   =", a_dot.round(3))     # 偏向 h2"爱" ✅
c_dot = a_dot @ H                          # 上下文 = 加权求和
print("上下文 c_t =", c_dot.round(3))      # 接近 h2

# 对齐热图:每个目标词一行权重
targets_states = [np.array([1.,0,0,0]), np.array([0.,1,0,0]), np.array([0.,0,1,0])]
align = np.stack([softmax(H @ st) for st in targets_states])
print("对齐矩阵(行=目标词 I/love/cats,列=源词 我/爱/猫):\n", align.round(2))
# 近似单位阵 → 词序单调对齐,正是热图的对角亮线

# ❌ 旧 Seq2Seq:解码每步都用同一个固定 c = h_last,长句信息被压垮
# ✅ 注意力:c_t = Σ α_tj·h_j,每步现算、按需直取源端任意位置

# 缩放点积:高维下点积方差≈d,不缩放 softmax 会过尖、梯度小
d = 64
q = rng.normal(size=d); ks = rng.normal(size=(3, d))
raw = ks @ q                          # 原始点积,方差≈d=64,量级大
scaled = raw / np.sqrt(d)             # ✅ 除 √d,把方差拉回≈1
print("不缩放 softmax:", softmax(raw).round(3))      # 极端尖锐(接近 one-hot)
print("缩放后 softmax:", softmax(scaled).round(3))   # 平滑、梯度健康
# 这正是 Bahdanau→Transformer 的关键改动之一(另一个是自注意力)
```

代码同时给出 Bahdanau 加性与 Luong 点积两种打分,验证权重会聚焦到最相关的源位置,且对齐矩阵呈对角亮线(单调对齐)——这正是右图热图的来源。

## 面试高频

- **"注意力解决了 Seq2Seq 的什么问题?"** 定长上下文瓶颈。解码每步用 $c_t=\sum_j\alpha_{tj}h_j$ 对编码器所有隐状态加权求和,按需取信息,长句不再失忆。
- **"注意力三步是什么?"** ① 打分 $e_{tj}=\text{score}(s,h_j)$;② softmax 归一化得权重 $\alpha_{tj}$;③ 加权求和 $c_t=\sum\alpha_{tj}h_j$。
- **"Bahdanau 和 Luong 注意力的区别?"** Bahdanau=加性(小前馈 $v^\top\tanh(W_s s+W_h h)$),用 $s_{t-1}$、双向编码器;Luong=乘性(dot/general/concat),用 $s_t$,更简洁高效,并提出 global/local。
- **"加性 vs 点积注意力,优劣?"** 点积无额外参数、可矩阵化、最快,但 query/key 须同维且高维时点积方差大(故 Transformer 加 $1/\sqrt{d}$ 缩放);加性更灵活、维度可不同,但有额外参数、稍慢。
- **"注意力权重 $\alpha$ 的可解释性?"** 排成对齐热图可看源/目标词的软对应;但要警惕"注意力权重 ≠ 因果重要性"(Jain & Wallace 2019 的争议)。
- **"为什么缩放点积要除 $\sqrt d$?"** 两个 $d$ 维零均值单位方差向量的点积方差 $\approx d$,$d$ 大时点积量级大、softmax 进饱和区(接近 one-hot)、梯度变小;除 $\sqrt d$ 把方差拉回 $\approx1$,保持 softmax 平滑、梯度健康。
- **"global 和 local 注意力区别?"** global 对源句全部位置算权重(全但慢);local 先预测对齐中心、只在窗口内算(快但可能漏窗外信息)。
- **"注意力权重能当解释吗?"** 谨慎。$\alpha$ 热图直观,但注意力权重 ≠ 因果重要性(Jain & Wallace 2019),同一预测可有多组不同注意力分布,别当金标准。
- **"Q、K、V 分别是什么角色?"** query 去匹配、key 被匹配(打分)、value 被加权聚合(输出内容)。Bahdanau 里 query=解码状态、key=value=编码隐状态;Transformer 用三个投影矩阵从同一输入生成 Q/K/V。
- **"从 Bahdanau 注意力到 Transformer 自注意力变了什么?"** 抽象出 Q/K/V;打分换缩放点积 $QK^\top/\sqrt d$;序列内部自相关(self-attention)取代跨编解码;去掉循环 → 可并行、长程一跳直达。

## 关键事实

- **Bahdanau 注意力(开山)**:Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*,arXiv:1409.0473(2014;ICLR 2015)——首次提出注意力,用加性打分让解码器"软搜索"源句相关片段,解除定长瓶颈,英法翻译显著提升并给出可解释对齐。
- **Luong 注意力**:Luong, Pham & Manning, *Effective Approaches to Attention-based Neural Machine Translation*,EMNLP 2015,pp. 1412-1421(arXiv:1508.04025)——提出 dot/general/concat 三种打分与 global/local 注意力,乘性打分更高效。
- **早期相关思想**:Graves(2013)手写生成中的注意力、Mnih et al.(2014)视觉注意力 RNN,均为同期萌芽;注意力概念并非凭空出现。
- **缩放点积与 $1/\sqrt d$**:Vaswani et al.(2017)在 Luong 点积基础上加缩放,解决高维点积过大导致 softmax 饱和的问题。
- **注意力可解释性争议**:Jain & Wallace, *Attention is not Explanation*(NAACL 2019)与 Wiegreffe & Pinter, *Attention is not not Explanation*(EMNLP 2019)正反交锋。
- **更早的注意力萌芽**:Graves, *Generating Sequences With RNNs*(2013)手写生成里的高斯注意力;Mnih et al., *Recurrent Models of Visual Attention*(NeurIPS 2014)。
- **下一步就是把注意力做成整个架构**:Transformer(Vaswani et al., "Attention Is All You Need", 2017)用缩放点积自注意力彻底替代循环,实现并行与长程建模——这是从本篇起源走向现代大模型的关键一跃(详见 LLM 域 Transformer 笔记)。
