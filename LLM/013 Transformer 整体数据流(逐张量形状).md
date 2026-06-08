[[013 Transformer 整体数据流(逐张量形状)|Transformer 整体数据流]] 是把前面 001–012 学过的零件(嵌入、自注意力、多头、FFN、残差、LayerNorm)串成一条完整流水线:输入一批 token id `(B, L)`,逐层走过 N 个相同结构的 block,每一步张量都是 `(B, L, d)`,最后投影到词表得到 logits `(B, L, V)`。这是「看懂整个架构」的总集篇。

## 直觉

把 Transformer 想成一条**等宽的传送带**。一批句子先被编码成 `(B, L, d)` 的张量——`B` 个句子、每句 `L` 个 token、每个 token 一个 `d` 维向量。之后传送带上有 N 个**长得一模一样的工位(block)**,每个工位做两件事:

1. **自注意力**:让每个 token「环顾全句」,把别的位置的信息按相关度搬一点过来(见 [[002 自注意力 Self-Attention|自注意力]]、[[005 多头注意力 Multi-Head|多头注意力]])。
2. **前馈网络 FFN**:对每个位置**单独**做一次「先放大 4 倍再压回」的非线性加工(见 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]])。

关键在于:**每个工位的进料和出料形状完全相同,都是 `(B, L, d)`**。正因为形状不变,才能把同样的工位**叠 N 层**;也正因为形状不变,[[009 残差连接与梯度流|残差连接]]才能把「输入」直接加到「子层输出」上(两端必须同形)。最后传送带尽头,把每个位置的 `d` 维向量投影到 `V` 维(词表大小),就是对「下一个词是谁」的打分(logits)。

一句话:**Transformer = 嵌入一次 + 重复 N 次(注意力混合 + FFN 加工 + 残差 + LayerNorm)+ 投影一次**。

## 例子

取一组小数字走一遍前向。设 `B=2`(两句话)、`L=4`(每句 4 个 token)、`d=8`(模型维)、`h=2`(头数)、`dk = d/h = 4`、词表 `V=1000`、层数 `N=6`。

1. **输入**:token id 张量,形状 `(2, 4)`,里面是整数(如 `[[5, 88, 3, 0], [12, 7, 91, 4]]`)。
2. **嵌入查表**:嵌入矩阵 `E ∈ (1000, 8)`,用 id 查行,得 `(2, 4, 8)`;再**逐元素加**位置编码 `PosEnc(4, 8)`,得 `x⁰ : (2, 4, 8)`。
3. **进入第 1 层**:
   - LayerNorm(Pre-LN)→ 仍 `(2, 4, 8)`。
   - 投影出 `Q, K, V` 各 `(2, 4, 8)`,拆成 2 头 → `(2, 2, 4, 4)`。
   - 算 `QKᵀ/√dk` → softmax,得注意力权重 `A : (2, 2, 4, 4)`(每个头一张 `4×4` 方阵)。
   - `A·V` 合头 → `Wo` → `(2, 4, 8)`;**残差相加** `x = x + Attn` → `(2, 4, 8)`。
   - 再 LayerNorm → FFN:`(2,4,8) → (2,4,32) → GELU → (2,4,8)`;残差相加 → `(2, 4, 8)`。
4. **第 2…6 层**:完全重复第 3 步,形状始终 `(2, 4, 8)`。
5. **最终 LayerNorm** → `(2, 4, 8)`。
6. **输出投影**:乘 `Eᵀ ∈ (8, 1000)`(tied embedding,见笔记 016)→ logits `(2, 4, 1000)`;对最后一维 softmax 得每个位置「下一个词」的概率分布。

注意一个数字:整条链上**唯一会「膨胀」的中间张量**只有两处——注意力权重 `A` 的 `(2, 2, 4, 4)`(其中 `L×L = 4×4`)和 FFN 中间层的 `(2, 4, 32)`(其中 `4d = 32`)。`L×L` 这一块正是长序列的瓶颈(见 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]])。

![[tf-整体数据流.png]]

## 原理

**符号**:批大小 $B$、序列长 $L$、模型维 $d$、头数 $h$、每头维 $d_k=d/h$、词表 $V$、层数 $N$。用上标表示层,$x^{(0)}$ 是嵌入后输入,$x^{(\ell)}$ 是第 $\ell$ 层输出。

**1. 嵌入 + 位置(一次)**。token id 矩阵 $T\in\{0,\dots,V-1\}^{B\times L}$,嵌入矩阵 $E\in\mathbb{R}^{V\times d}$,查表即按行索引:

$$x^{(0)}_{b,i} = E_{T_{b,i}} + p_i,\qquad x^{(0)}\in\mathbb{R}^{B\times L\times d}$$

其中 $p_i\in\mathbb{R}^d$ 是第 $i$ 位的位置编码。常见实现还会乘 $\sqrt{d}$ 缩放嵌入。

**2. 每一层(重复 $N$ 次,Pre-LN 形式)**。第 $\ell$ 层把 $x^{(\ell-1)}$ 变成 $x^{(\ell)}$,两个子层各带一条残差:

$$z = x^{(\ell-1)} + \mathrm{MHA}\big(\mathrm{LN}(x^{(\ell-1)})\big)$$
$$x^{(\ell)} = z + \mathrm{FFN}\big(\mathrm{LN}(z)\big)$$

注意力子层内部(见 [[006 注意力的矩阵形式与张量形状|注意力的矩阵形式]]):令 $\tilde x=\mathrm{LN}(x^{(\ell-1)})$,

$$Q=\tilde x W_Q,\ K=\tilde x W_K,\ V=\tilde x W_V\quad(W_\cdot\in\mathbb{R}^{d\times d})$$

拆头后每头 $Q_h,K_h,V_h\in\mathbb{R}^{B\times L\times d_k}$,

$$A_h=\mathrm{softmax}\!\Big(\frac{Q_hK_h^\top}{\sqrt{d_k}}\Big)\in\mathbb{R}^{B\times L\times L},\qquad \mathrm{head}_h=A_hV_h$$

合并所有头再过输出投影 $W_O\in\mathbb{R}^{d\times d}$ 得 $\mathrm{MHA}(\tilde x)\in\mathbb{R}^{B\times L\times d}$。

FFN 子层(逐位置共享、放大 4 倍):

$$\mathrm{FFN}(u)=\mathrm{GELU}(uW_1+b_1)W_2+b_2,\quad W_1\in\mathbb{R}^{d\times 4d},\ W_2\in\mathbb{R}^{4d\times d}$$

**形状不变性**:子层的输入 $x^{(\ell-1)}$ 和输出 $\mathrm{MHA}/\mathrm{FFN}$ 都是 $B\times L\times d$,残差 $x+f(x)$ 要求同形——这正是 Transformer 能「堆 N 层」且形状从头到尾恒为 $(B,L,d)$ 的数学原因。

**整模型参数量公式(面试常让你现场估)**。略去 LayerNorm 和偏置(占比极小):
$$\text{每层} = \underbrace{4d^2}_{\text{注意力 QKVO}} + \underbrace{8d^2}_{\text{FFN}(d\to4d\to d)} = 12d^2$$
$$\text{总参} \approx N\cdot12d^2 + \underbrace{Vd}_{\text{嵌入(tied 则只一份)}}$$
拿 GPT-2 small 核对:$N=12,d=768$ → $12\times12\times768^2\approx8500$ 万(主体)+ 嵌入 $50257\times768\approx3900$ 万 ≈ 1.24 亿,正是 124M。**记住 $12Nd^2$ 这个主体公式**,能秒估任何标准 Transformer 的参数量。

**前向计算量与训练计算量**。前向 FLOPs $\approx 2\times\text{参数量}\times$ token 数(每个参数一次乘一次加);训练再 ×3(前向 1 + 反向 2)。所以训练总 FLOPs $\approx 6\times N_{\text{params}}\times N_{\text{tokens}}$——这就是著名的 **"6ND" 估算公式**(Kaplan/Chinchilla scaling law 的基础),面试估算训练成本必用。

![[tf-参数量与6ND.png]]

**3. 输出(一次)**。最终归一化后投影到词表:

$$\text{logits}=\mathrm{LN}(x^{(N)})\,W_{\text{out}}\in\mathbb{R}^{B\times L\times V},\qquad P=\mathrm{softmax}(\text{logits})$$

权重绑定(tied embedding)时 $W_{\text{out}}=E^\top$(见 [[016 输出层、tied embedding 与 logits|输出层与 tied embedding]])。

![[tf-block逐张量.png]]

## 代码

```python
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):                 # x: (B, L, d)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a                         # 残差 1,形状不变 (B, L, d)
        x = x + self.ffn(self.ln2(x))     # 残差 2,形状不变 (B, L, d)
        return x

class TinyLM(nn.Module):
    def __init__(self, V, d=8, h=2, N=6):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(N)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight   # tied embedding(笔记 016)

    def forward(self, ids):               # ids: (B, L) 整数
        B, L = ids.shape
        x = self.emb(ids) + self.pos[:, :L]  # (B, L, d)
        for blk in self.blocks:
            x = blk(x)                    # 每层进出都 (B, L, d)
        return self.head(self.lnf(x))     # logits (B, L, V)

m = TinyLM(V=1000)
ids = torch.randint(0, 1000, (2, 4))      # (B=2, L=4)
print(m(ids).shape)                       # torch.Size([2, 4, 1000])  ← (B, L, V)
```

```python
# ❌ 常见错:在 block 里改变了 d,残差相加形状不匹配会直接报错
#   x = x + self.ffn_wrong(x)   # ffn_wrong 输出 (B, L, 4d) → 与 x (B,L,d) 不能相加
# ✅ 对:子层最后必须压回 d 维,保证 (B, L, d) 进、(B, L, d) 出,残差才合法
```

## 面试高频

- **Q:从一个 token 到一个 logit,数据走了哪几步?** A:① 查嵌入表 + 加位置编码 `(B,L)→(B,L,d)`;② 进 N 个相同 block,每个 = LN→自注意力→残差→LN→FFN→残差,形状始终 `(B,L,d)`;③ 最终 LN;④ 投影到词表 `(B,L,d)→(B,L,V)`;⑤ softmax 得概率。
- **Q:为什么 Transformer 每层进出形状都一样?有什么好处?** A:残差相加要求两端同形,所以子层必须把张量压回 `(B,L,d)`;形状不变意味着能无限堆同构的层、能 tied 权重、参数与序列长解耦——这是它易扩展(scaling)的结构基础。
- **Q:整条前向里哪两处张量会「膨胀」?** A:注意力权重 `(B,h,L,L)`(随 L 平方增长,长序列瓶颈)和 FFN 中间层 `(B,L,4d)`(参数大头,约占一层 2/3 参数)。
- **Q:编码器和解码器(GPT 这类)的区别?** A:结构件相同;解码器的自注意力加**因果掩码**,让位置 $i$ 只能看 $\le i$,从而能自回归生成(见 [[002 自注意力 Self-Attention|自注意力]])。
- **Q:怎么估一个标准 Transformer 的参数量?** A:主体 $\approx 12Nd^2$(每层 $4d^2$ 注意力 + $8d^2$ FFN),加嵌入 $Vd$。例:GPT-2 small $N=12,d=768$ → $12\cdot12\cdot768^2+50257\cdot768\approx124$M。
- **Q:训练总计算量怎么估?** A:$\approx 6\cdot N_{\text{params}}\cdot N_{\text{tokens}}$(前向 2ND + 反向 4ND);这是 scaling law 的 "6ND" 公式,估训练成本必用。
- **Q:GPT 这类 Decoder-Only 和原版数据流差在哪?** A:① 自注意力加因果掩码;② 没有编码器和交叉注意力;③ 现代用 Pre-LN/RMSNorm、RoPE 位置编码、SwiGLU FFN、GQA;但"嵌入→N 个同构 block→投影"的主干完全一样。
- **陷阱**:别把 FFN 当成跨位置混合——它**逐位置**独立,跨位置的信息交换**只发生在注意力里**;LayerNorm 沿最后一维(`d`),与 `B`、`L` 无关(见 [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]]);别忘了位置编码——删掉它模型对语序无感(见 001、002 的置换等变性)。

## 关键事实

- 原始架构出自 Vaswani 等《Attention Is All You Need》(NeurIPS 2017,arXiv:1706.03762),base 模型 $d=512$、$h=8$、$N=6$、FFN 中间维 $2048=4d$。
- 残差 + LayerNorm 包住每个子层是 Transformer 可深可堆的关键;原版是 Post-LN,现代大模型多用 Pre-LN(更稳,见 [[015 Transformer 训练稳定性|训练稳定性]])。
- FFN 中间维取 $4d$ 是经验默认,占单层参数约 2/3(见 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]])。
- 关联:嵌入 [[04 Embedding 与向量数据库|Embedding]];注意力家族 [[002 自注意力 Self-Attention|自注意力]]/[[005 多头注意力 Multi-Head|多头]]/[[006 注意力的矩阵形式与张量形状|矩阵形式]];稳定性 [[010 层归一化：Pre-LN 与 Post-LN|Pre/Post-LN]]/[[015 Transformer 训练稳定性|训练稳定性]];效率 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]];输出 [[016 输出层、tied embedding 与 logits|输出层]];正则 [[017 Dropout 在 Transformer 中的位置|Dropout 位置]]。
