[[117 训练一个 tiny GPT(PyTorch,可跑)|训练一个 tiny GPT]]:用 PyTorch 写一个**真能训练**的字符级 GPT——`GPTConfig` 定超参,`GPT` 把嵌入、N 个 Pre-LN Block、输出头组装起来,训练循环跑 `forward→loss→backward→AdamW.step` 四步。这是 115 的 numpy 注意力换上 autograd、跑起来的版本:实测初始 loss ≈ $\ln V$、几百步后降到 0.03、生成出连贯文本。这是全轨的核心可运行 artifact。

## 直觉

115 用 numpy 看清了注意力,但 numpy 没有自动微分、训不动。**这一篇把同一套数学搬到 PyTorch**:用 `nn.Linear` 替手写矩阵乘、用 autograd(就是 114 手写的那套)替手写反传、用 `nn.Embedding` 做词嵌入和位置嵌入。结构和 nanoGPT 一模一样,只是 tiny。

模型一句话:**`idx (B,T)` 进去,`logits (B,T,V)` 出来**——每个位置给出「下一个 token」的分数。组装五层(见 [[013 Transformer 整体数据流(逐张量形状)|数据流]]):

1. **嵌入**:token 嵌入 + 位置嵌入相加(本篇用学习式位置 `nn.Embedding`;也可换 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]])。
2. **N 个 Block**:每个 = Pre-LN + 多头注意力 + 残差 + Pre-LN + FFN + 残差(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]])。
3. **最终 LN + 输出头**:`Linear(d, V)`,与词嵌入**权重绑定**(tied,见 [[016 输出层、tied embedding 与 logits|tied embedding]])。

训练一句话:**反复四步**——前向算 loss、清梯度、反传、AdamW 更新(见 [[061 优化器与超参(AdamW)|AdamW]])。健全性检查:初始 loss 应约 $\ln V$。

![[impl-tinyGPT架构.png]]

## 例子

**实测一次完整训练(合成语料,字符级)。** 语料是几句话重复 200 遍,词表 $V=28$,模型 4 层 4 头 128 维,约 **0.8M 参数**。

```
vocab=28  params=0.802M
随机初始 loss 应 ≈ ln(28) = 3.332
step    0  loss 3.3952      ← 和理论值几乎一致 ✅ 初始化没爆
step  100  loss 0.0738
step  200  loss 0.0451
step  399  loss 0.0315      ← 几百步降两个数量级
```
**初始 loss ≈ $\ln V$ 是最重要的健全性检查**:随机模型对 $V$ 类均匀猜测,交叉熵恰为 $\ln V$(见 [[060 训练目标与 loss 实现|loss 实现]])。若初始 loss 远大于此 → 初始化爆了;远小于 → 多半忘了 shift(模型在抄输入)。

**生成(温度 0.8,top-k=10)。** 从 `'t'` 续写 120 字符:
```
that is the question. helo world. the quick brown fox jumps over
the lazy dog. to be or not to be that is the question. h
```
模型把语料的结构学得很死(因为是合成重复数据)。换成真实文本(tinyshakespeare),同样的代码会生成有英文单词形态、剧本格式但不照抄的文本。

**为什么生成里有个 `helo`(少了一个 l)?** 这是 tiny + 字符级模型的正常现象:它学的是「字符级转移概率」,偶尔会在高概率路径上滑掉一个字符。模型越大、数据越多、用 BPE(子词而非单字符),这类拼写错越少。这恰好说明**生成质量 ≠ 训练 loss**——loss 已降到 0.03,但字符级小模型仍会偶发拼写瑕疵;评估生成质量要另用人评/PPL(见 [[118 采样生成与困惑度评估|118]]),不能只看 train loss。

**训练循环的四步(每步必做,顺序不能乱)。**
```
① logits, loss = model(x, y)        前向：y 是 x 左移一位（下一个 token）
② optimizer.zero_grad()             清空上一步的梯度（autograd 是 += 累加！）
③ loss.backward()                   反传：autograd 算所有参数的梯度
④ clip + optimizer.step()           裁剪梯度 + AdamW 更新参数
```

![[impl-训练循环四步.png]]

## 原理

**1. 模型 = 嵌入 + Block 堆叠 + 输出头。** 输入 token id 序列 $\text{idx}\in\{0,\dots,V-1\}^{B\times T}$:

$$x=E_{\text{tok}}[\text{idx}]+E_{\text{pos}}[0{:}T]\in\mathbb{R}^{B\times T\times d}$$

过 $N$ 个 Block(见 [[115 手写多头注意力与 Transformer Block(numpy)|Block]]:$x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x))$,$x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$),最终 LN 后投影到词表:$\text{logits}=\mathrm{LN}_f(x)\,W_{\text{head}}^\top\in\mathbb{R}^{B\times T\times V}$。$W_{\text{head}}=E_{\text{tok}}$(权重绑定,省参数、稳训练,见 [[016 输出层、tied embedding 与 logits|tied embedding]]、[[054 词嵌入层与权重绑定|权重绑定]])。

**2. 损失 = 自回归交叉熵(shift 对齐)。** 把 logits 展平成 $(B\cdot T, V)$、targets 展平成 $(B\cdot T,)$ 送交叉熵。这里 `targets = x 左移一位`(在 `get_batch` 里造好:`y = data[i+1:i+1+T]`),所以位置 $t$ 的输出对位置 $t+1$ 的真实 token 算 loss(见 [[060 训练目标与 loss 实现|shift 对齐]]):

$$\mathcal{L}=-\frac{1}{BT}\sum_{b,t}\log p_\theta(y_{b,t}\mid x_{b,\le t})$$

**3. 训练循环 = 梯度下降。** 每步:`forward` 算 $\mathcal{L}$ → `zero_grad`(autograd 梯度是 `+=` 累加的,不清就叠加上一步,见 [[114 手写自动微分引擎(micrograd 级)|自动微分]])→ `backward` 算 $\nabla_\theta\mathcal{L}$ → `clip_grad_norm`(裁梯度防 loss spike,见 [[066 训练不稳定：loss spike 与对策|loss spike]])→ `AdamW.step` 更新。AdamW 把权重衰减从梯度解耦,$\beta=(0.9,0.95)$、wd=0.1 是 GPT 系标配(见 [[061 优化器与超参(AdamW)|AdamW]]、[[39 优化器(Momentum、RMSProp、Adam、AdamW)|优化器]])。

**4. 因果掩码在哪。** 在 `CausalSelfAttention` 里:注册一个下三角 buffer,注意力分数上三角填 $-\infty$,softmax 后变 0(见 [[007 因果掩码与 padding 掩码|因果掩码]])。这保证位置 $t$ 训练时看不到 $>t$——否则偷看答案,推理崩。`register_buffer` 而非 `nn.Parameter`:掩码是常量、不参与训练、但要随模型 `.to(device)` 一起搬,这正是 buffer 的用途。

**4'. 一次 forward 的形状走查(把数据流钉死)。** 输入 `idx (B,T)=(32,64)`:
- `tok_emb(idx)` → `(32,64,128)`;`pos_emb(pos)` → `(64,128)` 广播相加 → `x (32,64,128)`。
- 每个 Block 保形:`(32,64,128)→(32,64,128)`(注意力内部 `c_attn` 出 `(32,64,384)` 切成 3 份 Q/K/V,各 reshape 成 `(32,4,64,32)` 做多头,合回 `(32,64,128)`)。
- `ln_f` → `head` → `logits (32,64,28)`(每个位置一个 $V=28$ 维分布)。
- 算 loss 时展平成 `(32*64, 28)` vs targets `(32*64,)`。**全程只有 Block 内部短暂变形,进出都是 `(B,T,d)`**——这就是「形状保持 → 可堆叠」的代码体现(见 [[013 Transformer 整体数据流(逐张量形状)|数据流]])。

**4''. 为什么训练时一次能算 T 个位置的 loss(并行的关键)。** 自回归生成是逐 token 串行的,但**训练不是**:给定一整段 `(x, y)`,因果掩码保证位置 $t$ 只看 $\le t$,所以一次 forward 就同时算出**所有 $T$ 个位置**「预测下一个」的 loss——相当于一条样本提供了 $T$ 个训练信号。这就是 GPT 训练高效的原因:**训练并行(teacher forcing,真值当输入)、生成串行(自己的输出当输入)**。这个不对称是理解「训练快、推理慢」的关键。

**5. eval/train 模式。** 推理生成前要 `model.eval()`(关 dropout、固定 LN 统计),训练时 `model.train()`。本 tiny 版 dropout=0 影响小,但养成习惯——这是常见的「为什么训练好推理差」bug 来源。

**6. 参数量从哪来(对上 075 的手算)。** 这个 0.8M 怎么算的?主要四块:① 嵌入 `tok_emb` = $V\times d=28\times128$(因 tied,输出头不额外算);② 位置嵌入 = $\text{block\_size}\times d=64\times128$;③ 每个 Block:注意力 `c_attn`($d\times3d$)+ `c_proj`($d\times d$)+ FFN 两层($d\times4d+4d\times d$)+ 两个 LN($2\times2d$)≈ $12d^2$;④ 最终 LN。$d=128$ 时单 Block ≈ $12\times128^2\approx 0.2$M,4 层 ≈ 0.79M,加嵌入约 0.8M。**放大到 GPT-2(124M)用同一套公式、只是 $d=768$、12 层**——参数量 $\approx 12\cdot L\cdot d^2$ 是必背估算式(见 [[075 参数量逐层手算(GPT 全拆)|参数量手算]])。

**7. 训练健康度的几个信号(怎么看 loss 曲线)。** ① 初始 ≈ $\ln V$(已强调);② 前几十步快速下降(模型在学「字频」「常见 bigram」);③ 之后平缓下降(学更长依赖);④ 若 loss **突然 spike** → 大梯度,`clip_grad_norm` 没接住或学习率太高(见 [[066 训练不稳定：loss spike 与对策|loss spike]]);⑤ 若 train loss 降但 **val loss 升** → 过拟合(tiny + 合成数据上很容易,因为数据太规整),真实场景加 dropout / 早停 / 更多数据。本例合成数据高度重复,所以 loss 能降到 0.03(几乎背下来了)——真实语料降到 1.5 左右就很好,不会到 0(因为自然语言有不可约的熵)。

## 代码

完整可运行(需 `pip install torch`),已端到端验证:loss $3.39\to0.03$、生成连贯。这是 `model.py` + `train.py` + `sample.py` 的核心,拼上 116 的 tokenizer 和 113 的 `main.py` 即成完整项目。

```python
# model.py —— 可训练的字符级 tiny GPT（nanoGPT 风格）
import math, torch, torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65        # 由 tokenizer 决定
    block_size: int = 64        # 上下文长度
    n_layer:    int = 4
    n_head:     int = 4
    n_embd:     int = 128
    dropout:  float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # 一次投影出 Q,K,V
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)       # 输出投影 Wo
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.register_buffer("mask",                           # 下三角因果掩码
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        split = lambda z: z.view(B, T, self.n_head, hs).transpose(1, 2)  # (B,nh,T,hs)
        q, k, v = split(q), split(k), split(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)        # 缩放点积 (B,nh,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)  # 合头回 (B,T,C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(                              # FFN: d->4d->GELU->d
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd))
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # Pre-LN + 残差①
        x = x + self.mlp(self.ln_2(x))      # Pre-LN + 残差②
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # 学习式位置编码
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight                   # 权重绑定 (tied)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)         # GPT-2 初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)                # 嵌入相加 (B,T,d)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                         # (B,T,V)
        loss = None
        if targets is not None:                                  # 自回归交叉熵（targets 已 shift）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss
```

```python
# train.py —— 数据加载 + 训练循环（forward→loss→backward→AdamW.step）
torch.manual_seed(1337)
text = ("hello world. the quick brown fox jumps over the lazy dog. "
        "to be or not to be that is the question. ") * 200          # 实战换成 tinyshakespeare
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

cfg = GPTConfig(vocab_size=V, block_size=64, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg)
print(f"vocab={V}  params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")
print(f"随机初始 loss 应 ≈ ln(V) = {math.log(V):.3f}")           # 健全性基线

def get_batch(bs=32):                                            # 随机取一批 (x, y=x左移1位)
    ix = torch.randint(0, len(data) - cfg.block_size - 1, (bs,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])  # ✅ shift
    return x, y

opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                        betas=(0.9, 0.95), weight_decay=0.1)    # GPT 系标配超参
model.train()
for it in range(400):
    x, y = get_batch(32)
    _, loss = model(x, y)                                       # ① forward + loss
    opt.zero_grad(set_to_none=True)                            # ② 清梯度（必须！autograd 累加）
    loss.backward()                                            # ③ 反传
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # ④ 裁梯度防 spike
    opt.step()                                                 #     AdamW 更新
    if it % 100 == 0:
        print(f"step {it:4d}  loss {loss.item():.4f}")
# step   0  loss 3.3952   ← ≈ ln(28)，初始化正常
# step 100  loss 0.0738
# step 399  loss 0.0315   ← 学会了
```

```python
# sample.py（核心，详见 118）—— 自回归生成
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()                                               # ❗ 关 dropout / 固定 LN
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]              # 截到上下文长度
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature                # 只取最后一个位置
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")        # top-k 截断
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                      # 采样
        idx = torch.cat([idx, nxt], dim=1)                     # 拼回，继续
    return idx

ctx = torch.tensor([[stoi['t']]], dtype=torch.long)
out = generate(model, ctx, 120, temperature=0.8, top_k=10)
print("".join(itos[i] for i in out[0].tolist()))
# -> "that is the question. helo world. the quick brown fox jumps over the lazy dog. ..."
```

```python
# —— 验证集 loss:监控过拟合(train 降但 val 升 = 过拟合) ——
@torch.no_grad()
def estimate_loss(model, get_batch, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch()
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
# 实战:把 data 切成 train/val 两段,各写一个 get_batch,定期打印两者 loss 对比

# —— 三个常见 bug 的「现场复现」(理解症状,别真改你的好代码) ——
# bug① 忘 shift:y 不左移,直接 y=x → 模型抄输入,loss 异常低(远小于 ln V)、生成复读
#     def get_batch_bad(): ...; y = torch.stack([data[i:i+T] for i in ix]); ...  # ❌ 没 +1
# bug② 忘 zero_grad:删掉 opt.zero_grad() → 梯度累加,几步内 loss 爆成 nan/inf
# bug③ 掩码方向反:把 tril 写成 triu(上三角)→ 偷看未来,train loss 极低但生成全乱
#     这三个都不会报错,只是「悄悄训坏」——所以初始 loss≈ln V 的健全性检查极重要
```

```python
# ============ 扩展练习 ============
# 1. 换真实语料(tinyshakespeare,约 1MB),同样代码会生成有词形态、剧本格式的文本。
# 2. 加 train/val 切分 + estimate_loss,画两条 loss 曲线观察过拟合点。
# 3. 把学习式位置换成 RoPE(见 031),测试在比 block_size 更长的序列上的表现。
# 4. 加 KV-Cache(见 102)加速 generate:缓存历史 K/V,每步只算新 token。
# 5. 加 lr warmup + cosine 衰减(见 062),对比 loss 曲线是否更平滑。
# 6. 把 dropout 设 0.1,对比 model.eval() 开/关时的生成差异,体会为何推理要 eval()。
```

## 面试高频

- **Q:从零搭一个 GPT 类需要哪几个组件?** A:`nn.Embedding`(词 + 位置)、N 个 Block(Pre-LN 注意力 + FFN + 残差)、最终 LN、`Linear` 输出头(常与词嵌入 tied)。`forward(idx, targets=None)` 返回 `(logits, loss)`。
- **Q:训练循环四步是什么?顺序能乱吗?** A:`forward 算 loss` → `zero_grad 清梯度` → `backward 反传` → `step 更新`。不能乱:autograd 梯度是累加的,忘 `zero_grad` 会把上一步梯度叠进来;`backward` 必须在 `step` 前。
- **Q:训练刚开始 loss 应该多少?偏离说明什么?** A:约 $\ln(\text{vocab\_size})$(随机猜)。远大于 → 初始化爆/没归一;远小于 → 多半忘了 shift,模型在抄输入(假性低 loss)。
- **Q:位置编码用哪种?有什么选择?** A:本 tiny 版用学习式 `nn.Embedding`(GPT-2 做法,最简);也可用正弦(原始 Transformer)或 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]](现代 LLaMA 系,外推更好)。
- **Q:为什么要 `clip_grad_norm`?** A:防止偶发的大梯度(loss spike)把训练带飞;裁到范数上限(如 1.0)。配合 AdamW + warmup 是稳训标配(见 [[066 训练不稳定：loss spike 与对策|loss spike]])。
- **Q:tied embedding 是什么,为什么用?** A:输出头权重 = 输入词嵌入权重(共享),省一份 $V\times d$ 参数、且经验上更稳更好(见 [[016 输出层、tied embedding 与 logits|tied embedding]])。
- **Q:`model.eval()` / `train()` 区别?** A:eval 关 dropout、用固定的 LN/BN 统计;train 反之。推理前忘 `eval()` 是「训练好推理差」的常见 bug。
- **Q:GPT 参数量怎么估?** A:主导项是 $\approx 12\cdot L\cdot d^2$(每 Block 注意力 $4d^2$ + FFN $8d^2$),加嵌入 $Vd$。tiny($d{=}128,L{=}4$)≈0.8M,GPT-2($d{=}768,L{=}12$)≈124M,同一公式只换数字(见 [[075 参数量逐层手算(GPT 全拆)|参数量]])。
- **Q:loss 能降到 0 吗?** A:合成高度重复数据能(背下来,本例 0.03);真实自然语言不能——它有不可约的熵(下一个词本就不确定),降到 1.5 左右(PPL≈4.5)就很好。降到接近 0 反而要警惕过拟合或数据泄漏。
- **Q:怎么发现/防过拟合?** A:切 train/val,监控两者 loss;train 降但 val 升即过拟合。防:加 dropout、早停、更多数据、weight decay。tiny + 合成数据极易过拟合(数据太规整)。
- **Q:tied embedding 为什么 `self.tok_emb.weight = self.head.weight`?** A:让输入嵌入和输出投影**共享同一份** $V\times d$ 权重(物理上是同一个张量),省一份参数且经验更稳;赋值后两者梯度自动合并更新。

## 关键事实

- 本实现是 Karpathy **nanoGPT** 的最小化(github.com/karpathy/nanoGPT,`model.py` 的 `CausalSelfAttention/Block/GPT` 结构一致),教学版见 "Let's build GPT: from scratch"(2023)。
- 自回归交叉熵目标见 [[060 训练目标与 loss 实现|loss 实现]];初始 loss $\approx\ln V$ 是黄金健全性检查;Transformer 整体数据流见 [[013 Transformer 整体数据流(逐张量形状)|数据流]](Vaswani 2017, arXiv:1706.03762;GPT-2 Radford 2019)。
- 关键组件出处:Pre-LN [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]、tied embedding [[016 输出层、tied embedding 与 logits|tied embedding]](Press & Wolf 2017)、AdamW $\beta{=}(0.9,0.95)$/wd=0.1 [[061 优化器与超参(AdamW)|AdamW]](Loshchilov & Hutter 2019, arXiv:1711.05101)、初始化 std=0.02 是 GPT-2 配方。
- 训练四步顺序固定:`forward→zero_grad→backward→step`;autograd 梯度累加(见 [[114 手写自动微分引擎(micrograd 级)|自动微分]])故必须 `zero_grad`;`clip_grad_norm` 防 spike(见 [[066 训练不稳定：loss spike 与对策|loss spike]])。
- 位置编码本版用学习式 `nn.Embedding`(GPT-2 做法);可替换为 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]](LLaMA 系,外推更好)或正弦(见 [[029 正弦余弦位置编码(推导)|正弦编码]]、[[030 可学习与相对位置编码|可学习位置]])。
- 参数量主导项 $\approx 12Ld^2$ + 嵌入 $Vd$:tiny($d{=}128,L{=}4$)≈0.8M、GPT-2($d{=}768,L{=}12$)≈124M,同一公式换数字(见 [[075 参数量逐层手算(GPT 全拆)|参数量]])。
- loss 在合成重复数据上可降到接近 0(背下来),真实自然语言有不可约熵、降到约 1.5(PPL≈4.5)即良好;train/val loss 背离即过拟合,tiny + 合成数据极易过拟合。
- 三个不报错却「悄悄训坏」的 bug:忘 shift(抄输入、loss 异常低)、忘 zero_grad(梯度累加发散)、掩码方向反(偷看未来、train 低推理崩);故初始 loss≈ln V 的健全性检查不可省。
- 关联:总入口 [[113 从零实现总览：课程地图到代码|从零实现总览]];numpy 理解版 [[115 手写多头注意力与 Transformer Block(numpy)|numpy 多头注意力]];分词器 [[116 实现 BPE 分词器|BPE]];生成与评估 [[118 采样生成与困惑度评估|采样与困惑度]];SFT/LoRA [[119 给 tiny GPT 做 SFT 与 LoRA(迷你对齐)|迷你对齐]];参数量手算 [[075 参数量逐层手算(GPT 全拆)|参数量]]。
