[[017 Dropout 在 Transformer 中的位置|Dropout 在 Transformer 中的位置]] 讲的是这个经典正则手段(见 [[42 正则化(L2、Dropout、早停、标签平滑)|Dropout]])具体**加在 block 的哪几处**:① 嵌入 + 位置编码之后,② 注意力权重上,③ 每个子层输出、残差相加之前,④ FFN 内部。原版 Transformer 一律用 $p=0.1$。装错位置(尤其装进残差「干净通道」)会破坏稳定性。

## 直觉

Dropout 的核心是「训练时随机关掉一部分激活,逼网络别依赖任何单个单元、学冗余鲁棒的特征」(见 [[42 正则化(L2、Dropout、早停、标签平滑)|正则化]])。但在 Transformer 里**不是哪都能撒**——关键看「丢了之后会不会破坏那条让梯度顺畅流动的残差通道」(见 [[009 残差连接与梯度流|残差连接]])。

把 block 想成一条主干道(残差 `x`)上挂着两个加工车间(注意力、FFN)。**正确做法是只在「车间的产出」上撒 dropout,主干道 `x` 本身不动**:

- 写成 `x = x + dropout(子层(x))` —— 丢的是子层产出 $f(x)$,残差 `x` 原样保留。
- **绝不能**写成 `x = dropout(x + 子层(x))` 或 `x = dropout(x)` —— 那等于在主干道上随机断路,把残差「无损直达输出」的性质毁了,深层会训不稳。

四个落点正是「四个产出口」:嵌入刚生成时、注意力权重算好时、每个子层即将汇入残差前、FFN 内部激活后。一句话:**丢产出、不丢主干;残差加法外面包 dropout,加法本身不碰。**

## 例子

**注意力权重 dropout 怎么丢(小数字)**。设一行注意力权重(某 query 对 4 个 key)softmax 后为 $A=[0.1,\,0.5,\,0.3,\,0.1]$,$p=0.5$(为看清效果取大值,实际 0.1)。

训练时随机把第 2、4 个置 0,再对保留的按 $\frac{1}{1-p}=2$ 倍放大(inverted dropout 保持期望):

$$A' = [0.1\times 2,\ 0,\ 0.3\times 2,\ 0] = [0.2,\,0,\,0.6,\,0]$$

效果:这一步该 query **看不见**第 2、4 个 token,被迫从别的位置凑信息——避免过度依赖某一条注意力连接。注意这之后 $A'$ 行和不再是 1,但因为是乘在 $V$ 上、且训练随机平均,整体期望不变。

**残差处的对错**。设 $x=[1,1]$,子层产出 $f(x)=[0.4,0.4]$,$p=0.5$:

- ✅ 对:`x + dropout(f(x))`。若 dropout 关掉第 2 维 → $f$ 变 $[0.8,0]$ → $x'=[1.8,1]$。主干 `x=[1,1]` 完好,梯度仍能沿 `x` 无损回传。
- ❌ 错:`dropout(x + f(x))`。先加得 $[1.4,1.4]$,再 dropout 关第 2 维 → $[2.8,0]$。**主干信息第 2 维被整个抹掉**,残差的「恒等捷径」断了,深层梯度受损。

![[tf-dropout位置.svg]]

## 原理

**Dropout 形式**(inverted dropout,训练时缩放)。对激活向量 $a$,采样掩码 $m_i\sim\text{Bernoulli}(1-p)$:

$$\tilde a_i = \frac{m_i}{1-p}\,a_i\qquad(\text{训练});\qquad \tilde a_i = a_i\quad(\text{推理})$$

除以 $1-p$ 使 $\mathbb{E}[\tilde a_i]=a_i$,所以**推理时直接关掉 dropout、不需缩放**。这也是为什么忘了切 `model.eval()` 会让推理结果带随机噪声(常见 bug)。

**四个位置(原版 Vaswani 等 2017,$p=0.1$)**:

1. **嵌入 + 位置编码之后**:$x^{(0)}=\text{Dropout}(E_{\text{id}}+p_{\text{pos}})$。对输入表示做正则,防止过度记忆具体 token 组合。
2. **注意力权重上**:$A=\text{Dropout}(\text{softmax}(QK^\top/\sqrt{d_k}))$,再乘 $V$。随机屏蔽部分「注意力连接」,逼模型别只盯一两个位置(原论文称 attention dropout)。
3. **每个子层输出、残差相加之前**:$x\leftarrow x+\text{Dropout}(\text{Sublayer}(\text{LN}(x)))$。这是原论文的核心描述——"applied to the output of each sub-layer, before it is added to the sub-layer input"。**dropout 包在残差加法外侧、子层产出上**,保证残差通道 $x$ 不被丢弃。
4. **FFN 内部**:常见放在激活之后:$\text{Dropout}(\text{GELU}(uW_1))W_2$;有些实现把 FFN 的「子层输出 dropout」并入位置 3。

**为什么不能丢残差主干**。残差的价值在于提供恒等捷径,使梯度 $\frac{\partial \mathcal{L}}{\partial x^{(\ell-1)}}$ 含一个 $+1$ 项不衰减(见 [[009 残差连接与梯度流|残差连接]])。若对主干 $x$ 或 $x+f(x)$ 整体 dropout,会随机切断这条捷径,等价于深层网络梯度被随机置零,训练失稳。故 dropout 只作用于**子层产出 $f(\cdot)$**。

**取值与现代趋势**。原版统一 $p=0.1$。现代大模型(GPT-3、LLaMA 等)在**大规模预训练**时常把 dropout 设为 **0**——数据足够多、单遍即足,过拟合风险低,dropout 反而拖慢收敛;但在**小数据微调**时仍常开 0.1 防过拟合。注意力 dropout 在长上下文下也常关闭。

**为什么"海量数据时 dropout 反而有害"**。Dropout 的作用是防过拟合(数据少、模型大时记住噪声)。预训练时数据量远超模型容量、且常只过一遍(每个样本只见一次),根本没机会过拟合;此时 dropout 只是平白丢掉信息、加噪、拖慢收敛,得不偿失。微调时数据少、易过拟合,才重新启用。这是"正则强度该随数据量反向调"的一个具体体现。

**相关变体:Stochastic Depth / DropPath**。比单元级 dropout 更狠的是**整层级**随机:训练时按概率**整个跳过某个残差子层**(直接 $x\to x$,不走子层),等价于随机训练一个更浅的子网络,推理时全用。它对很深的 Transformer/ViT 正则更强,但同样**只丢子层产出、不丢残差主干**(跳过=只走那条 $+x$ 恒等路),与"不能丢主干"的原则一致。注意区分:dropout 丢的是激活元素,DropPath 丢的是整个子层。

**Dropout 与其他正则的分工**(见 [[42 正则化(L2、Dropout、早停、标签平滑)|正则化]]):
- **Dropout**:训练随机丢激活,逼学冗余特征,推理用全网络(集成视角)。
- **权重衰减(L2/AdamW)**:压小权重,防参数过大。LLM 预训练主要靠它而非 dropout。
- **标签平滑**:把 one-hot 目标软化(如 0.9/0.1),防 logits 过自信、改善校准,原版 Transformer 用了 $\epsilon_{ls}=0.1$。
现代 LLM 预训练的正则组合常是"weight decay + (可选)标签平滑 + dropout=0",微调才加回 dropout。

## 代码

```python
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)  # ② 注意力权重 dropout
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(),
            nn.Dropout(p),                       # ④ FFN 内 dropout(激活后)
            nn.Linear(4*d, d),
        )
        self.drop = nn.Dropout(p)                # ③ 子层输出 dropout

    def forward(self, x):                        # x: (B, L, d)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop(a)                     # ✅ 丢「子层产出」,残差 x 不动
        x = x + self.drop(self.ffn(self.ln2(x))) # ✅ 同上
        return x

class Embed(nn.Module):
    def __init__(self, V, d, p=0.1):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        self.drop = nn.Dropout(p)                # ① 嵌入 + 位置后 dropout
    def forward(self, ids):
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        return self.drop(x)
```

```python
# ❌ 错:对残差主干 / 整个相加结果做 dropout —— 随机切断残差捷径,深层训不稳
def forward_wrong(self, x):
    a, _ = self.attn(x, x, x)
    x = self.drop(x + a)        # ✗ 丢的是 (x + a) 整体,主干 x 被随机抹掉
    return x

# ✅ 对:只对子层产出 dropout,残差 x 原样保留
def forward_right(self, x):
    a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
    x = x + self.drop(a)        # ✓ dropout 包在加法外侧、作用于 a
    return x

# ⚠️ 别忘了推理切 eval,否则 dropout 仍随机生效
# model.eval()   # 关闭所有 dropout,输出可复现
```

## 面试高频

- **Q:Transformer 里 dropout 加在哪几个地方?** A:① 嵌入+位置编码后;② 注意力权重(softmax 之后、乘 V 之前);③ 每个子层输出、残差相加之前;④ FFN 内部(激活后)。原版统一 $p=0.1$。
- **Q:为什么 dropout 要放在残差相加的「外侧」而不是主干上?** A:残差提供恒等捷径让梯度不衰减;若对主干 $x$ 或 $x+f(x)$ 整体 dropout,会随机切断捷径、深层训不稳。所以写成 `x = x + dropout(f(x))`,只丢子层产出。
- **Q:注意力权重 dropout 丢的是什么?会破坏概率归一吗?** A:丢的是 softmax 后的注意力权重(随机屏蔽部分连接)。会让该行权重和暂时不为 1,但因 inverted dropout 缩放 + 训练随机平均,期望不变;且后续乘 V 时整体仍合理。
- **Q:推理时 dropout 怎么处理?** A:全关。inverted dropout 训练时已除以 $1-p$ 保持期望,推理直接用原激活、不缩放。忘了 `model.eval()` 会让输出带随机性,是常见 bug。
- **Q:为什么很多大模型预训练把 dropout 设 0?** A:海量数据 + 单遍训练,过拟合风险低,dropout 反而拖慢收敛、损失信息;小数据微调时才重新开 0.1。
- **Q:为什么海量数据预训练时 dropout 反而有害?** A:dropout 防过拟合;预训练数据远超模型容量、常只过一遍,没机会过拟合,dropout 只是丢信息、加噪、拖慢收敛。微调数据少才启用。
- **Q:Stochastic Depth / DropPath 和 dropout 区别?** A:dropout 丢激活元素;DropPath 按概率整层跳过残差子层(只走 $+x$ 恒等路),等价随机训更浅子网,对很深模型正则更强;同样不丢残差主干。
- **Q:dropout、权重衰减、标签平滑分别防什么?** A:dropout 学冗余特征(集成);权重衰减压小权重;标签平滑软化目标防过自信、改善校准(原版 Transformer 用 0.1)。LLM 预训练主要靠权重衰减,dropout 常设 0。
- **Q:inverted dropout 为什么推理不用缩放?** A:训练时已除以 $1-p$ 保持期望,推理直接用原激活;反过来的"普通 dropout"才在推理时乘 $1-p$,现代框架都用 inverted。
- **陷阱**:① 别对残差主干 dropout;② 训练/推理模式要切对(`train()`/`eval()`),忘切 eval 是推理带随机性的经典 bug;③ 注意力 dropout 与 FFN dropout 可独立设率;④ dropout 和 LayerNorm 顺序、与残差的相对位置都影响稳定性(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre/Post-LN]])。

## 关键事实

- 原始 Transformer(Vaswani 等,NeurIPS 2017,arXiv:1706.03762)对**每个子层输出(残差相加前)**和**嵌入+位置编码之和**施加 dropout,$P_{\text{drop}}=0.1$;另对注意力权重施加 dropout。
- Dropout 本身由 Srivastava 等提出(JMLR 2014):训练随机丢弃、推理用全网络,等价于集成指数多个子网络(见 [[42 正则化(L2、Dropout、早停、标签平滑)|正则化]])。
- inverted dropout(训练时按 $1/(1-p)$ 缩放)使推理无需任何缩放,是主流实现。
- 现代大模型预训练常将 dropout 设为 0(GPT-3、LLaMA 等),小数据微调时再启用;长上下文下注意力 dropout 也常关闭。
- 关联:正则化总论 [[42 正则化(L2、Dropout、早停、标签平滑)|Dropout]];残差通道为何不能丢 [[009 残差连接与梯度流|残差连接]];归一化位置交互 [[010 层归一化：Pre-LN 与 Post-LN|Pre/Post-LN]];整体结构见 [[013 Transformer 整体数据流(逐张量形状)|整体数据流]];与训练稳定性配合 [[015 Transformer 训练稳定性|训练稳定性]]。
