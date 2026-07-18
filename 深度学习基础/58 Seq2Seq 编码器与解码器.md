[[58 Seq2Seq 编码器与解码器|序列到序列(Seq2Seq)]]用两个 RNN——编码器把整条源序列压成**一个定长上下文向量** $c$,解码器再从 $c$ 自回归地生成目标序列;那个定长 $c$ 就是著名的"信息瓶颈",正是注意力要解决的痛点。

## 直觉

机器翻译"我爱猫 → I love cats":输入输出**长度不同、不一一对应**,普通逐位置 RNN 干不了。Seq2Seq 的办法是**两段式**:
- **编码器(Encoder)**:一个 RNN 从左到右读完整个源句,把它的"意思"浓缩进最后的隐状态 $h_{|x|}$,记作上下文向量 $c$。
- **解码器(Decoder)**:另一个 RNN 以 $c$ 为起点,**逐词生成**译文——每步用上一步生成的词当输入(自回归),直到吐出结束符 `<EOS>`。

天才之处是统一了"变长输入 → 变长输出"的所有任务(翻译、摘要、对话、问答)。

**为什么普通 RNN 干不了"变长到变长"(零基础)**。逐位置的 RNN 假设输入第 $t$ 个对应输出第 $t$ 个(一一对齐),适合"等长同步"任务(如词性标注)。但翻译里源句 3 词可能译成 4 词、语序还会颠倒("我爱猫"→"I love cats" 词序巧合一致,但"红苹果"→"red apple",而法语 "pomme rouge" 语序反转),根本不是逐位置对齐。Seq2Seq 用"先读完再生成"的两段式解耦了输入长度和输出长度,这才装得下任意的长度与语序变化。

**致命短板**:无论源句多长,都被硬塞进**同一个定长向量 $c$**。短句还行,句子一长(30+ 词),$c$ 这个"独木桥"装不下,信息严重丢失——这就是**Seq2Seq 瓶颈**,直接催生了 [[60 注意力机制的起源(Bahdanau、Luong)|注意力]]。

## 例子

翻译"我 爱 猫 → I love cats",编码器 RNN(隐状态用标量示意):
- 读"我":$h_1=f(h_0,\text{我})$
- 读"爱":$h_2=f(h_1,\text{爱})$
- 读"猫":$h_3=f(h_2,\text{猫})$ → **$c=h_3$**(整句压缩成这一个向量)

解码器从 $c$ 起步,自回归生成:
- 输入 `<BOS>` + $c$ → 输出 `I`
- 输入 `I` + 状态 → 输出 `love`
- 输入 `love` + 状态 → 输出 `cats`
- 输入 `cats` → 输出 `<EOS>`,停止

**瓶颈直观感受**:若源句是 50 个词的长句,编码器仍只把全部信息塞进 $c$,解码"第 1 个词"时其实最该看源句开头,但开头的信息早被后面 49 个词的更新冲淡了。**Sutskever 2014 的工程 trick**:把源句**逆序**输入(读"猫 爱 我"),让源句开头和译文开头在时间上更近、引入更多短程依赖,BLEU 明显提升——这恰恰反证了定长瓶颈的存在。

**信息论看"独木桥"为什么必然丢信息**。源句信息量随长度增长(50 词比 3 词信息多得多),但 $c$ 是**固定维度**(比如 512 维 float)。固定容量的管道塞进越来越多信息,必然有损——就像无论文件多大都压成同样大小的 zip,文件大了就压坏。BLEU 随源句长度下降的实测曲线(Bahdanau 2014 图)正是这条"瓶颈定律"的证据:**无注意力的 Seq2Seq 在长句上崩,注意力版几乎不随长度退化**。

**beam search 手算示意(为什么不贪心)**。词表 {A,B},解码两步,每步给出概率。
- 贪心:第 1 步 $P(A)=0.6>P(B)=0.4$ 选 A;第 2 步在 A 之后 $P(B|A)=0.3$,得序列 AB 总分 $0.6\times0.3=0.18$。
- beam(k=2):第 1 步留 A(0.6)、B(0.4);第 2 步展开,假设 $P(A|B)=0.9$,则 BA 总分 $0.4\times0.9=0.36>0.18$。**贪心错过了整体更优的 BA**,因为第 1 步的局部最优(A)不等于全局最优。beam 保留 top-$k$ 路径就是为防这种短视。

![[rnn-beam_search.png]]

![[rnn-seq2seq瓶颈.png]]

## 原理

**编码器**:RNN/LSTM/GRU 逐步读入源序列 $x_1,\dots,x_{T_x}$,
$$h_t=\text{RNN}(h_{t-1},x_t),\qquad c=h_{T_x}\ (\text{或对所有 }h_t\text{ 池化})$$
$c$ 是定长向量,**与源句长度无关**——这是瓶颈的根。

**解码器**:以 $c$ 初始化解码状态 $s_0$,自回归生成,每步条件于"上下文 + 已生成的词":
$$s_t=\text{RNN}(s_{t-1},[\,y_{t-1};c\,]),\qquad P(y_t\mid y_{<t},x)=\text{softmax}(W s_t+b)$$
整体最大化条件似然(对应 [[30 交叉熵与负对数似然|交叉熵]]损失):
$$\log P(y\mid x)=\sum_{t=1}^{T_y}\log P(y_t\mid y_{<t},c)$$

**训练**:用 [[59 Teacher Forcing 与曝光偏差|Teacher Forcing]]——解码器第 $t$ 步喂**真实**的 $y_{t-1}$(而非自己上一步的预测),便于稳定地计算所有目标位置的损失、通常收敛更快;但带来训练/推理不一致的"曝光偏差"。对这里的 **RNN 解码器**,状态递推 $s_t=\operatorname{RNN}(s_{t-1},\dots)$ 仍沿时间串行,Teacher Forcing 不会把这个状态计算变成位置并行;位置并行训练是带因果掩码的非循环 Transformer 解码器的性质。

**推理**:没有真实词,只能喂自己上一步的预测。逐词贪心容易短视,实践用 **beam search**(束搜索):每步保留概率最高的 $k$ 条候选路径,最后取整体得分最高的序列。

**beam search 的工程细节**:
- **束宽 $k$**:$k=1$ 退化为贪心;$k$ 越大搜索越充分但越慢,翻译常用 4–10。$k$ 过大反而可能掉点(偏好太"安全"的短句)。
- **长度归一化**:序列越长对数概率累加越负(每词都 $<1$),不归一会**偏好短句**;除以长度 $T^\alpha$($\alpha\approx0.6$)校正。
- **覆盖惩罚 / 重复惩罚**:防止漏译或重复生成。
- **终止**:某条路径吐出 `<EOS>` 即完成,放入候选池;所有束都结束或达最大长度后,从候选池按归一化分数选最优。

**贪心 / beam / 采样的取舍**:确定性任务(翻译、摘要)用 beam(求最可能);开放生成(对话、写作)用**采样**(top-k / top-p / 温度,见 [[27 Softmax 与温度|温度采样]]),beam 在开放生成里反而无聊、易重复。

**瓶颈的本质**:把任意长源句压成固定维 $c$,信息论上必然有损;序列越长损失越大。Bahdanau 2014 直接打掉这个独木桥——**解码每步不再只用 $c$,而是回看编码器全部 $h_t$ 做加权求和**(注意力),信息瓶颈被解除。这是通往 [[60 注意力机制的起源(Bahdanau、Luong)|注意力]]乃至 Transformer 的关键一跃。

![[rnn-seq2seq编解码结构.png]]

![[attn-Bahdanau对齐.png]]

## 代码

```python
import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))
rng = np.random.default_rng(0)

H, V = 8, 6                      # 隐藏维、词表大小
Wx = rng.normal(0, .3, (H, V)); Wh = rng.normal(0, .3, (H, H))
def rnn(h, x_onehot): return np.tanh(Wx @ x_onehot + Wh @ h)
def onehot(i): v = np.zeros(V); v[i] = 1; return v

# 编码器:读完整个源序列 -> 定长 c(瓶颈!)
src = [0, 1, 2]                   # 源 token id
h = np.zeros(H)
for tok in src: h = rnn(h, onehot(tok))
c = h.copy()                      # ✅ 整句被压成这一个向量
print("上下文向量 c 维度 =", c.shape, " 与源句长无关")

# 解码器:从 c 自回归生成(贪心),每步只能看 c + 上一步词
Wo = rng.normal(0, .3, (V, H))
s = c.copy(); prev = onehot(0)    # <BOS>
out = []
for _ in range(4):
    s = np.tanh(Wx @ prev + Wh @ s)          # 注意:真实模型解码器有独立权重,这里简化
    logits = Wo @ s; p = np.exp(logits) / np.exp(logits).sum()
    tok = int(p.argmax()); out.append(tok)
    prev = onehot(tok)
print("解码输出 token =", out)

# ❌ 瓶颈写法:不管源句多长,解码每步都只用同一个固定 c —— 长句信息被压垮
# ✅ 注意力(下一篇):解码第 t 步用 c_t = Σ α_tj · h_j,回看编码器所有 h_j
#    伪代码:scores = [score(s, hj) for hj in enc_states]; alpha = softmax(scores)
#            c_t = sum(a * hj for a, hj in zip(alpha, enc_states))
```

代码点明瓶颈:$c$ 的维度与源句长度**无关**,长句被压垮;注释里给出注意力的修复方向(逐步用 $c_t=\sum\alpha_{tj}h_j$ 取代固定 $c$)。

```python
# beam search 骨架:每步保留 top-k 路径,带长度归一化
import numpy as np

def beam_search(step_logprob, k=3, max_len=5, eos=99):
    """step_logprob(prefix) -> dict{token: logprob};返回最优序列"""
    beams = [([], 0.0)]                       # (序列, 累计对数概率)
    finished = []
    for _ in range(max_len):
        cand = []
        for seq, score in beams:
            if seq and seq[-1] == eos:
                finished.append((seq, score)); continue
            for tok, lp in step_logprob(seq).items():
                cand.append((seq + [tok], score + lp))     # 累加对数概率
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:k]                       # ✅ 只留 top-k
    finished += beams
    # 长度归一化,否则偏好短句(每词 logprob<0,长序列分数更负)
    finished.sort(key=lambda x: x[1] / (len(x[0]) ** 0.6), reverse=True)
    return finished[0][0]

# ❌ 贪心:每步只取 argmax,易错过整体更优的序列
# ✅ beam:保留 k 条路径 + 长度归一化,近似全局最优
```

## 面试高频

- **"Seq2Seq 是什么、解决什么?"** 编码器-解码器两个 RNN,把变长源序列映射到变长目标序列;统一了翻译、摘要、对话等任务。
- **"Seq2Seq 的瓶颈在哪?"** 编码器把整句压成**一个定长上下文向量** $c$,长序列信息丢失;注意力通过解码每步对所有编码器隐状态加权求和来解除瓶颈。
- **"Sutskever 2014 为什么把源句逆序输入?"** 拉近源句开头与译文开头的时间距离、引入更多短程依赖,优化更易,BLEU 提升——侧面印证定长瓶颈。
- **"训练和推理的解码有何不同?"** 训练常用 [[59 Teacher Forcing 与曝光偏差|Teacher Forcing]] 喂真实词,推理只能喂自己的预测、常配 beam search;不一致导致曝光偏差。对 RNN 解码器,即使训练时已知全部真实词,隐藏状态仍按时间递推;Transformer 的因果掩码才允许各位置在训练时并行计算。
- **"为什么推理用 beam search 而非贪心?"** 贪心每步选局部最优易陷次优整句;beam 保留 top-$k$ 路径,近似找全局最优序列。
- **"beam search 为什么要长度归一化?"** 对数概率累加越长越负,不归一会系统性偏好短句;除以 $T^\alpha$($\alpha\approx0.6$)校正。还常加覆盖/重复惩罚。
- **"beam 越大越好吗?"** 不是。$k$ 太大变慢且偏好"安全"的短/平淡序列,翻译质量可能掉点;开放生成里 beam 更易重复无聊,改用 top-k/top-p 采样。
- **"为什么 Seq2Seq 长句性能差,从信息论解释?"** 固定维 $c$ 容量有限,源句越长信息越多、压缩越有损;BLEU 随长度下降即证据。注意力按需直取源端任意位置,解除瓶颈。
- **"编码器的 $c$ 一定是最后隐状态吗?"** 不一定;可对所有 $h_t$ 做池化(均值/最大),或用双向编码器拼接两端。注意力则保留**全部** $h_t$ 供解码每步加权。

## 关键事实

- **Seq2Seq 出处**:Sutskever, Vinyals & Le, *Sequence to Sequence Learning with Neural Networks*,NeurIPS 2014(arXiv:1409.3215)——多层 LSTM 编码-解码,**逆序源句**显著提升 BLEU;明确指出定长向量是性能瓶颈。
- **同期 RNN Encoder-Decoder**:Cho et al.,EMNLP 2014(arXiv:1406.1078)——提出编码器-解码器框架并引入 [[57 GRU 与变体|GRU]],同样把整句编成定长向量。
- **瓶颈的破解**:Bahdanau, Cho & Bengio(arXiv:1409.0473, 2014)用注意力让解码器软搜索源句相关片段,解除定长瓶颈,见 [[60 注意力机制的起源(Bahdanau、Luong)]]。
- **Beam search** 是 Seq2Seq 推理的标准解码;BLEU(Papineni 2002)是机器翻译标准自动评测指标。
