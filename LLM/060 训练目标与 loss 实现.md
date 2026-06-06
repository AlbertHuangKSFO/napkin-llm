[[060 训练目标与 loss 实现|loss 实现]] 把自回归训练目标落到代码:**next-token 交叉熵、labels 左移对齐(shift)、teacher forcing、以及 loss mask(用 -100 屏蔽提示/padding)**。这是 [[056 预训练目标：自回归、掩码与去噪|自回归目标]] 从公式到 PyTorch 的最后一公里,踩错(忘 shift、off-by-one、忘 mask)会让 loss 假性极低或居高不下。

## 直觉

自回归训练就一句话:**让位置 $t$ 的输出去预测位置 $t+1$ 的真实 token**。实现上有三个必须对齐的细节:

1. **shift(左移对齐)**:模型在每个位置吐出「下一个词」的分布,所以要拿 `logits[:-1]` 配 `labels[1:]`——预测「下一个」,而不是预测「自己」。**忘了 shift,模型只需把输入抄一遍,loss 假性趋零,什么也没学到。**
2. **teacher forcing**:训练时每一步喂的是**真实**的上一个 token(而非模型自己上一步的预测),所以一次前向就能并行算出所有位置的 loss。推理时才改成自回归(喂模型自己的输出),这个训练—推理差异叫**曝光偏差**,见 [[59 Teacher Forcing 与曝光偏差|teacher forcing]]。
3. **loss mask**:不是所有位置都该算 loss。指令微调时只对「助手回答」算(提示部分屏蔽);padding 的 `<pad>` 不算。用把 label 设成 `-100` 来屏蔽。

一句话:**shift 对齐预测目标,teacher forcing 让训练能并行,loss mask 决定哪些位置算账。**

## 例子

**shift 对齐(手画)。** 序列「猫 坐 在 垫 上」(L=5):
```
取 logits[:-1]（位置 0..3，各自预测「下一个」）：  猫→?  坐→?  在→?  垫→?
取 labels[1:] （真正的下一个 token）：            坐     在     垫     上
```
位置 0 的输出该预测「坐」,位置 1 该预测「在」……最后位置 4「上→?」没有下一个 token,丢弃。每个位置算一次 [[30 交叉熵与负对数似然|交叉熵]],再平均。

![[train-loss计算流.svg]]

**为什么忘 shift 会假性低 loss。** 若错用位置 $t$ 的 logits 配 token $t$(没左移),模型只要在每个位置「复制当前输入」就全对,loss 迅速趋近 0——但它学的是恒等映射,生成时一片混乱。这是新手最常见、最隐蔽的 bug:**loss 漂亮但模型废了**。

**loss mask 的两个场景。**
```
指令微调：[系统/用户提示] [助手回答]
          ↑ labels 全设 -100   ↑ 这里才算 loss   （只学怎么答，不学复述提示）

padding：  [真实 token] [pad pad pad]
          ↑ 算 loss     ↑ labels 设 -100        （否则逼模型"预测填充符"，学坏）
```
PyTorch 交叉熵默认 `ignore_index=-100`:把不想算的位置 label 设成 -100,该位既不算 loss 也不回传梯度。

![[train-loss-mask.svg]]

## 原理

**1. 交叉熵 = 负对数似然。** 自回归目标最大化 $\prod_t p(x_t\mid x_{<t})$,等价于最小化平均负对数似然,即 [[30 交叉熵与负对数似然|交叉熵]]:

$$\mathcal{L}=-\frac{1}{N}\sum_{t}\log p(x_{t+1}\mid x_{\le t}),\qquad p(\cdot)=\mathrm{softmax}(\text{logits})$$

其中 $p$ 由 [[27 Softmax 与温度|softmax]] 把 logits 归一化得到(训练时温度恒为 1)。$N$ 是**实际算 loss 的位置数**(被 mask 的不计入分母)。这个 loss 取指数就是 [[32 困惑度 Perplexity|困惑度]]。

**2. shift 的张量操作。** logits 形状 $(B,L,V)$,tokens 形状 $(B,L)$:

$$\text{shift\_logits}=\text{logits}[:, :L-1, :],\qquad \text{shift\_labels}=\text{tokens}[:, 1:]$$

展平成 $(B(L-1),V)$ 和 $(B(L-1),)$ 后送交叉熵。off-by-one(多移/反向)会让标签整体错位,loss 居高不下、学不动。

**3. teacher forcing 与并行。** 因为有 [[007 因果掩码与 padding 掩码|因果掩码]],位置 $t$ 只能看 $\le t$。训练时喂的是真实序列(teacher forcing),一次前向就拿到所有位置的 logits,所有位置的 loss **并行**算完——这是 GPT 训练高效的关键。推理改为逐 token 自回归(喂自己的输出),训练只见「真实历史」、推理见「自己生成的历史」,分布偏移即**曝光偏差**(见 [[59 Teacher Forcing 与曝光偏差|曝光偏差]])。

**4. loss mask 的数学。** 引入掩码 $m_t\in\{0,1\}$:

$$\mathcal{L}=-\frac{\sum_t m_t\log p(x_{t+1}\mid x_{\le t})}{\sum_t m_t}$$

`ignore_index=-100` 等价于把 $m_t=0$ 的位置 label 置 -100,分子分母都只统计 $m_t=1$ 的位置。**分母要用有效位置数**,否则 padding 多时 loss 被稀释、不可比。

**5. 序列打包(packing)。** 把多条短样本拼成一条满窗序列省 padding(见 [[058 数据配比与课程|数据]] 的算力意识)。必须用文档分隔符(`<|endoftext|>`)或重置注意力掩码,**防止跨样本注意 / 跨样本算 loss**——否则前一条样本会"泄漏"给后一条。

**6. next-token 交叉熵的逐步手算。** 词表 $V=4$,某位置 logits $z=[2.0,1.0,0.1,-1.0]$,真实下一个 token 是 id=0。先 softmax:$e^z=[7.389,2.718,1.105,0.368]$,和 $=11.58$,$p=[0.638,0.235,0.095,0.032]$。该位 loss $=-\log p_0=-\log 0.638=0.449$。若预测得差(真实是 id=3):loss $=-\log0.032=3.45$。把一条序列每个位置这样算一遍再平均,就是序列 loss;对它取 $\exp$ 就是 [[32 困惑度 Perplexity|困惑度]](此例若每位都 0.449,PPL $=e^{0.449}=1.57$)。**数值稳定**:实现绝不先 $\exp$ 再 $\log$(会溢出),而用 log-sum-exp:$-\log p_y=\text{logsumexp}(z)-z_y$,减去 $\max z$ 后再算。

**7. 标签平滑(label smoothing)。** 把硬 one-hot 目标 $[0,0,1,0]$ 软化成 $[\frac{\epsilon}{V},\dots,1-\epsilon+\frac{\epsilon}{V},\dots]$($\epsilon$ 如 0.1),loss 变成对软目标的交叉熵。作用:防止模型对正确类输出过度自信(logits 无界增大)、轻微正则、改善校准。原始 Transformer(Vaswani 2017)用 $\epsilon=0.1$;但**GPT 类大模型预训练通常不用**(会抬高困惑度、且大数据下过自信问题不突出),多见于翻译/分类。面试问到「为什么 GPT 不太用标签平滑」即答此。

**8. z-loss(稳定输出 logits)。** 大模型训练里,输出 logits 的规模可能漂移变大,导致 softmax 数值不稳、加剧 [[066 训练不稳定：loss spike 与对策|loss spike]]。**z-loss** 加一项惩罚 $\lambda_z\,(\log Z)^2$,其中 $Z=\sum_v e^{z_v}$ 是 softmax 的归一化常数(配分函数),把 $\log Z$ 拉向 0、抑制 logits 整体膨胀。PaLM、部分 MoE(router z-loss,见 [[048 路由稳定性：router z-loss|z-loss]])用它,典型 $\lambda_z\approx10^{-4}$。

**9. 梯度累积与 loss 归约方式的耦合(易错)。** 用 `reduction="mean"` 时,每个 micro-batch 的 loss 已对**该 micro-batch 的有效 token 数**取平均;做梯度累积(见 [[063 批大小、梯度累积与 critical batch size|梯度累积]])时若直接把各 micro 的 mean-loss 相加再 step,等于把学习率放大了 `accum_steps` 倍——必须 `loss /= accum_steps`。更隐蔽的坑:各 micro-batch 有效 token 数不等时,「先各自 mean 再平均」与「全局 token 加权平均」**不等价**,严格做法应按有效 token 数加权(token-averaged loss),否则短样本被高估。

## 代码

```python
import torch, torch.nn.functional as F

# —— 标准自回归 loss：shift + 交叉熵 ——
def causal_lm_loss(logits, input_ids, loss_mask=None):
    # logits:(B,L,V)  input_ids:(B,L)
    shift_logits = logits[:, :-1, :].contiguous()      # 位置 0..L-2 预测「下一个」
    shift_labels = input_ids[:, 1:].contiguous()       # ✅ 左移一位对齐
    if loss_mask is not None:                          # 1=算 loss, 0=屏蔽
        shift_labels = shift_labels.masked_fill(loss_mask[:, 1:] == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,                             # -100 位置不算 loss、不回传
    )

# 困惑度 = exp(loss)
def perplexity(loss):
    return torch.exp(loss)
```

```python
# —— 指令微调：只对「助手回答」算 loss ——
def build_sft_labels(prompt_ids, answer_ids):
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids[:]   # ✅ 提示部分屏蔽
    return torch.tensor([input_ids]), torch.tensor([labels])
# 模型内部会再 shift；HF Trainer 直接吃这种带 -100 的 labels
```

```python
# ❌ 错一：忘了 shift —— 用位置 t 的 logits 配 token t
#   loss = F.cross_entropy(logits.view(-1,V), input_ids.view(-1))   # 模型只需复制输入！loss 假性→0
# ❌ 错二：off-by-one —— 多移一位 / 移反方向 → 标签错位，loss 居高不下
# ❌ 错三：忘 mask padding —— pad 位算进 loss，逼模型预测 <pad>、loss 被稀释失真
# ❌ 错四：分母用总长度而非有效位置数 —— padding 多时 loss 不可比
# ✅ 对：shift 对齐 + ignore_index=-100 屏蔽提示/padding + 分母只数有效位
```

```python
import torch, torch.nn.functional as F
# —— 数值稳定的 next-token CE:logsumexp - z_y,绝不先 exp 再 log ——
z = torch.tensor([2.0, 1.0, 0.1, -1.0])         # 某位置 logits, V=4
y = 0                                            # 真实下一个 token id
loss = torch.logsumexp(z, dim=-1) - z[y]         # = -log softmax(z)[y]
print("手算 loss =", loss.item())                # 0.449...
print("对照 F.cross_entropy:", F.cross_entropy(z.unsqueeze(0), torch.tensor([y])).item())
print("困惑度 PPL =", torch.exp(loss).item())     # exp(0.449)=1.57

# —— 一个可运行的极简自回归训练循环(含 shift + 梯度累积 + 裁剪) ——
torch.manual_seed(0)
V, d, L, B = 100, 32, 16, 4
emb  = torch.nn.Embedding(V, d)
core = torch.nn.GRU(d, d, batch_first=True)      # 占位:任意因果序列模型
head = torch.nn.Linear(d, V, bias=False); head.weight = emb.weight   # 权重绑定
params = list(emb.parameters()) + list(core.parameters())
opt = torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

accum = 2
for step in range(6):
    opt.zero_grad()
    for _ in range(accum):                       # 梯度累积模拟更大 batch
        ids = torch.randint(0, V, (B, L))
        h, _ = core(emb(ids))
        logits = head(h)                         # (B,L,V)
        shift_logits = logits[:, :-1, :].reshape(-1, V)   # ✅ 左移对齐
        shift_labels = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels) / accum  # ✅ 除以 accum
        loss.backward()                          # 梯度累加
    torch.nn.utils.clip_grad_norm_(params, 1.0)  # ✅ 裁剪防 spike(见 066)
    opt.step()
    print(f"step {step}  loss={loss.item()*accum:.3f}")
# ❌ 把 loss 不除 accum / 累积时每步都 step → 等效放大 lr 或没放大 batch
```

## 面试高频

- **Q:自回归 loss 怎么实现?为什么要 shift?** A:取 `logits[:, :-1]` 配 `labels[:, 1:]`,让位置 $t$ 的输出预测 token $t+1$,再算交叉熵。因为模型每个位置吐的是「下一个词」分布,必须左移对齐;最后一个位置无下一个 token 丢弃。
- **Q:忘了 shift 会怎样?** A:相当于让位置 $t$ 预测 token $t$,模型只需复制输入即可全对,**loss 假性趋零但什么也没学到**——最隐蔽的常见 bug。
- **Q:什么是 teacher forcing?和曝光偏差什么关系?** A:训练时每步喂真实的上一个 token(而非模型预测),配合因果掩码可一次并行算所有位置 loss;推理改喂自己的输出,训练—推理分布不一致即曝光偏差。
- **Q:loss mask / ignore_index 干嘛的?** A:用 -100 屏蔽不该算 loss 的位置:指令微调只算助手回答(屏蔽提示)、padding 不算。PyTorch 交叉熵默认 `ignore_index=-100`,该位不算 loss 也不回传梯度;分母只数有效位。
- **Q:loss 和困惑度什么关系?** A:loss 是平均交叉熵(负对数似然),困惑度 = exp(loss);温度训练时恒为 1。
- **Q:序列打包要注意什么?** A:多条短样本拼一条满窗省 padding,但要用文档分隔符或重置注意力掩码,防止跨样本注意/跨样本算 loss。
- **Q:交叉熵实现为什么不能先 exp 再 log?** A:会数值溢出。用 log-sum-exp:$-\log p_y=\text{logsumexp}(z)-z_y$,且先减 $\max z$ 再算,稳定。手算:logits $[2,1,0.1,-1]$、真值 id0 → loss $=0.449$。
- **Q:GPT 预训练用标签平滑吗?** A:通常不用——会抬高困惑度,且大数据下过自信问题不突出;标签平滑($\epsilon=0.1$)多见于翻译/分类(原始 Transformer 用过)。
- **Q:z-loss 干嘛的?** A:惩罚 $\lambda_z(\log Z)^2$($Z$ 为 softmax 配分函数),把 logits 整体规模拉回、稳定数值、抑制 loss spike;PaLM、MoE router 用,典型 $\lambda_z\approx10^{-4}$。
- **Q:梯度累积时 loss 怎么处理?** A:`reduction="mean"` 下每 micro 已平均,累积须 `loss/=accum_steps` 否则隐式放大 lr;各 micro 有效 token 不等时严格应按 token 数加权(token-averaged),否则短样本被高估。
- **陷阱**:shift 方向/位数别错;分母用有效位置数;CE 用 logsumexp 防溢出;梯度累积别忘除 accum;不可信场景别让特殊 token 被注入(见 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]])。

## 关键事实

- 自回归交叉熵目标 $\mathcal{L}=-\frac1N\sum_t\log p(x_{t+1}\mid x_{\le t})$,见 [[30 交叉熵与负对数似然|交叉熵]];softmax 归一化见 [[27 Softmax 与温度|Softmax]](GPT 论文 Radford 2018/2019、Brown 2020 arXiv:2005.14165)。
- PyTorch `nn.CrossEntropyLoss` 默认 `ignore_index=-100`,用于屏蔽 padding / 提示位置;HF `transformers` 约定 labels 用 -100 标记不算 loss 的位。
- teacher forcing 训练并行、推理自回归,差异即曝光偏差(Bengio et al. 2015, Scheduled Sampling),见 [[59 Teacher Forcing 与曝光偏差|Teacher Forcing 与曝光偏差]]。
- 困惑度 = exp(平均交叉熵 loss),见 [[32 困惑度 Perplexity|困惑度]];按字节归一可得 bits-per-byte(见 [[109 语言模型评估：困惑度与 bits-per-byte|语言模型评估]])。
- 序列打包(packing)用文档分隔/重置掩码避免跨样本泄漏,是预训练吞吐优化常规手段。
- 关联:自回归目标对比 [[056 预训练目标：自回归、掩码与去噪|预训练目标]];因果掩码 [[007 因果掩码与 padding 掩码|因果掩码]];GPT 实现 [[036 GPT 系列：自回归与规模化|GPT]];数据/算力意识 [[058 数据配比与课程|数据配比]];优化器与超参 [[061 优化器与超参(AdamW)|AdamW]]。
