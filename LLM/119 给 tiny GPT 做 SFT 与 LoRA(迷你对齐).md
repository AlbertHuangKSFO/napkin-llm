[[119 给 tiny GPT 做 SFT 与 LoRA(迷你对齐)|给 tiny GPT 做 SFT 与 LoRA]]:把训好的 tiny GPT 从「会续写」推到「会听指令」——**SFT** 用「指令-回答」数据微调,关键是 **loss mask**(只对回答算 loss、屏蔽提示);**LoRA** 手写一个 `LoRALinear` 旁路,冻结主干、只训低秩 $B A$ 增量。这是 [[081 指令微调 SFT 与数据构造|SFT]] 与 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]] 落到 tiny GPT 的 `finetune.py`,也是全课程的收尾:从零造的模型,现在能对齐了。

## 直觉

预训练让模型「会接话」(续写),但它不知道你想要它回答而不是续写问题。**SFT(监督微调)** 用一堆「指令 → 理想回答」对继续训练,让它学会「看到指令就回答」(见 [[080 后训练总览：SFT 到 RM 到 RLHF|后训练总览]])。

SFT 和预训练唯一的差别是 **loss mask**:把「提示(系统+用户)」部分的 label 设成 `-100`,**只对「助手回答」算 loss**(见 [[060 训练目标与 loss 实现|loss mask]])。为什么?提示是输入条件,不该让模型去预测它——否则浪费容量学复述问题。

但全参数微调一个大模型很贵。**LoRA** 的洞察:微调时权重的改变量是**低秩**的,所以冻结原权重 $W$,只学一个低秩增量 $\Delta W=BA$($A$ 降维到秩 $r$,$B$ 升回)。可训练参数从 $d^2$ 降到 $2rd$,常占总量 0.1%~1%(见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]])。把 $B$ 初始化为 0,**起步时旁路输出为 0,等同原模型**——不破坏已学的能力,平滑过渡。

![[impl-SFT-loss-mask.png]]

## 例子

**SFT 数据怎么构造(loss mask)。** 一条样本「问:法国首都?答:巴黎。」:
```
input_ids = [提示 token...] + [回答 token...]
labels    = [-100, -100, ...] + [回答 token id...]    ← 提示位全 -100
```
实测:`input=[1,2,3,4, 5,6,7]`(前 4 是提示、后 3 是回答),`labels=[-100,-100,-100,-100, 5,6,7]`。算 loss 时(shift + `ignore_index=-100`),**只有 3 个回答位置参与**——和回答 token 数一致。模型内部仍照常 shift 对齐,唯一的不同就是这个 mask(见 [[081 指令微调 SFT 与数据构造|SFT 数据构造]])。

**手写 LoRA 的三个验证(实测)。**

1. **起步等同原模型**:$B=0$ 时 `lora(x) == base(x)`,误差 $<10^{-6}$。✅ 不破坏已学能力。
2. **只训 A、B,主干冻结**:可训练参数 `['A','B']`,冻结 `['base.weight','base.bias']`;在 16→16 的层上,可训练参数仅占 **32%**(真实大模型上是 0.1%~1%,因为 $d$ 越大占比越小)。
3. **训练后主干不变**:跑 50 步 AdamW,loss 下降,`base.weight` 一字未改——**学到的全在 $A,B$ 里**。这意味着一个主干能挂多个 LoRA(多任务),随时切换。

![[impl-LoRA旁路注入.png]]

**为什么 LoRA 省到离谱。** 一个 $768\times768$ 的权重 = 59 万参数;LoRA $r=4$ 只需 $2\times4\times768=6144$ 个,**省 99%**。推理时还能把 $BA\cdot\frac{\alpha}{r}$ 合并进 $W$,零额外延迟(见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]])。

**省的不只是参数,更是显存(这才是 LoRA 真正解决的痛点)。** 全参数微调一个 7B 模型,显存账(见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]):参数 14GB(fp16)+ 梯度 14GB + AdamW 优化器状态 28GB(每参数 2 个 fp32 状态)= 56GB,单卡放不下。LoRA 只对 $A,B$(占 <1%)算梯度和优化器状态,**梯度 + 优化器状态从 42GB 降到几百 MB**;冻结主干只需前向(还能量化成 4-bit,即 QLoRA),于是**单张消费级卡就能微调 7B/13B**。这就是 LoRA/QLoRA 让「人人能微调大模型」的根本原因——参数省是表象,**梯度和优化器状态只覆盖那 <1% 的参数**才是省显存的实质。

**为什么「微调增量是低秩」这个假设成立?** 直觉:预训练已经把通用能力学进 $W$,下游任务往往只是「在某个低维子空间上的小调整」(换个领域、换个风格、学个格式),不需要重写整个 $W$。Aghajanyan 等的工作发现微调的「本征维度(intrinsic dimension)」确实很低——这正是 LoRA 低秩假设的实证依据。$r$ 取 4~64 即够,任务越复杂取越大。

## 原理

**1. SFT = 带 mask 的自回归微调。** 目标和预训练同为交叉熵,但只在回答位置算(见 [[060 训练目标与 loss 实现|loss mask]]、[[081 指令微调 SFT 与数据构造|SFT]]):

$$\mathcal{L}_{\text{SFT}}=-\frac{\sum_t m_t\log p_\theta(y_{t+1}\mid x_{\le t})}{\sum_t m_t},\quad m_t=\begin{cases}1&\text{回答位}\\0&\text{提示位}\end{cases}$$

实现上把 $m_t=0$ 的位置 label 置 `-100`,`F.cross_entropy(..., ignore_index=-100)` 自动跳过(不算 loss、不回传)。对话还要套**对话模板**(角色标记、`<eos>`,见 [[053 词表、特殊 token 与对话模板|对话模板]])。

**2. LoRA 的低秩假设。** 全参数微调学 $\Delta W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$。LoRA 假设 $\Delta W$ 低秩,分解为(见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]]):

$$\Delta W=BA,\quad A\in\mathbb{R}^{r\times d_{\text{in}}},\ B\in\mathbb{R}^{d_{\text{out}}\times r},\ r\ll d$$

前向变成 $h=Wx+\frac{\alpha}{r}BAx$。$W$ 冻结,只训 $A,B$。$B$ 初始 0(旁路起步为 0,= 原模型),$A$ 高斯小初始化;$\frac{\alpha}{r}$ 是缩放,让 $r$ 变化时学习率不用重调。可训练参数 $2rd$ vs 全参 $d^2$,省 $\sim r/d$。

**3. 为什么旁路而非改 $W$。** ①省显存:不存 $W$ 的优化器状态/梯度(只存 $A,B$ 的,见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]);②可组合:一个主干挂多个 LoRA,按任务切换;③推理零延迟:可把 $\frac{\alpha}{r}BA$ 加进 $W$ 合并。QLoRA 进一步把 $W$ 量化成 4-bit 再挂 LoRA(见 [[097 NF4 与 QLoRA 4-bit|QLoRA]])。

**4. 注入哪些层。** 通常注入注意力的 $Q,K,V,O$ 投影(本 tiny 版包 `c_attn`、`c_proj`)。把 `nn.Linear` 替换成 `LoRALinear(原层)` 即可——`forward` 仍是 `Wx`,只多一条旁路。冻结主干:`requires_grad=False`,优化器只收 `requires_grad=True` 的参数。

**5. 它和反传/优化器的关系。** LoRA 不改训练循环——还是 [[117 训练一个 tiny GPT(PyTorch,可跑)|forward→loss→backward→step]] 四步,只是 autograd(见 [[114 手写自动微分引擎(micrograd 级)|自动微分]])只对 $A,B$ 算梯度、AdamW 只更新它们。这就是「高效微调」高效的全部来源:**梯度和优化器状态只覆盖那 0.1% 的参数**。

**6. PEFT 家族(LoRA 不是唯一)。** LoRA 属于「参数高效微调(PEFT)」一大类(见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|PEFT]]):
- **Adapter**:在每层插入小瓶颈 MLP(降维→非线性→升维),只训这些 adapter。比 LoRA 早,但推理有额外串行计算(不能合并进 $W$)。
- **Prefix/Prompt Tuning**:在每层 KV 前拼一段可学习的「虚拟前缀向量」,只训这些前缀,连权重旁路都不动。参数最少,但表达力弱于 LoRA。
- **QLoRA**:LoRA + 把冻结主干量化成 4-bit(NF4)+ paged optimizer,单卡微调 65B(见 [[097 NF4 与 QLoRA 4-bit|QLoRA]])。
- **DoRA / LoRA+ / rsLoRA**:LoRA 的改进变体(分解幅度方向、A/B 用不同学习率、改缩放),边际提质。

LoRA 之所以成主流:**推理可合并(零延迟)+ 可组合(一主干挂多 LoRA 按任务切)+ 效果接近全参微调**,三者兼得。

**7. SFT 的数据与陷阱(对齐的现实部分)。** SFT 效果**主要由数据质量决定**,不是技巧:① 「Less is More」(LIMA, 2023)——**1000 条高质量指令数据**就能让基座模型变得很会聊,质量远比数量重要;② 数据要**多样**(覆盖任务类型)、**格式统一**(套同一对话模板,见 [[053 词表、特殊 token 与对话模板|对话模板]]);③ 警惕 SFT 教模型「自信地编」——若数据里全是「知道答案」的样本,模型会学到「任何问题都要给个肯定答案」,加剧幻觉;④ SFT 是后训练第一步,之后常接偏好对齐(RM + RLHF/DPO,见 [[080 后训练总览：SFT 到 RM 到 RLHF|后训练总览]])。

## 代码

完整可运行,接 117 的 GPT。已验证:LoRA 起步等同原模型、主干冻结不变、SFT mask 只算回答位。这是 `finetune.py`。

```python
# finetune.py —— 手写 LoRA 注入 + SFT loss mask
import torch, torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """包住一个冻结的 nn.Linear，旁挂低秩 B·A 增量。"""
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                       # ❄ 冻结主干
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)   # 降维（小高斯）
        self.B = nn.Parameter(torch.zeros(out_f, r))         # 升维（初始 0 ⇒ 起步=原模型）
        self.scale = alpha / r                               # 缩放，解耦 r 与学习率
    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

def inject_lora(model, r=4, alpha=8):
    """把 GPT 各 Block 注意力里的 c_attn / c_proj 换成 LoRALinear。"""
    for blk in model.blocks:
        blk.attn.c_attn = LoRALinear(blk.attn.c_attn, r, alpha)
        blk.attn.c_proj = LoRALinear(blk.attn.c_proj, r, alpha)
    return model
```

```python
# —— 验证 LoRA 的三个性质 ——
base = nn.Linear(16, 16)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(2, 5, 16)
assert torch.allclose(lora(x), base(x), atol=1e-6)        # ① B=0 起步等同原模型 ✅
trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
print("可训练:", trainable)                                # ② ['A', 'B']，主干冻结 ✅
n_tr = sum(p.numel() for p in lora.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in lora.parameters())
print(f"可训练占比: {100*n_tr/n_all:.1f}%")                 # 32%（小层）；大模型上 <1%
```

```python
# —— SFT：构造带 loss mask 的样本，只对回答算 loss ——
def build_sft_example(prompt_ids, answer_ids):
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids[:]      # ✅ 提示位 -100，回答位真值
    return torch.tensor([input_ids]), torch.tensor([labels])

def sft_loss(model, input_ids, labels):
    logits, _ = model(input_ids)                           # (B,T,V)
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)               # shift 对齐
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)  # 跳过 -100

prompt, answer = [1, 2, 3, 4], [5, 6, 7]                   # "问..." / "答..."
x, labels = build_sft_example(prompt, answer)
print("labels:", labels.tolist())          # [[-100,-100,-100,-100, 5,6,7]]
n_active = (labels[:, 1:] != -100).sum().item()
print(f"参与 loss 的位置: {n_active}（= 回答 token 数 {len(answer)}）")   # 3 ✅
```

```python
# —— 完整迷你 SFT 训练循环（LoRA + mask），把 117 的 model 拿来对齐 ——
# from model import GPT, GPTConfig          # 117
# model = GPT(cfg); model.load_state_dict(torch.load("pretrained.pt"))   # 载入预训练
# model = inject_lora(model, r=4, alpha=8)  # 注入 LoRA，冻结主干

def finetune(model, dataset, steps=200, lr=1e-3):
    # 只把 requires_grad=True 的参数（A,B）交给优化器
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"微调参数量: {sum(p.numel() for p in params)}（仅 LoRA A/B）")
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for it in range(steps):
        prompt, answer = dataset[it % len(dataset)]        # (指令ids, 回答ids)
        x, labels = build_sft_example(prompt, answer)
        loss = sft_loss(model, x, labels)                  # 只对回答算
        opt.zero_grad(); loss.backward(); opt.step()       # 还是四步，只是只更新 A/B
        if it % 50 == 0:
            print(f"step {it:4d}  sft_loss {loss.item():.4f}")
    return model
# 微调后：base 权重一字未改，所有改变都在 LoRA 的 A/B 里 → 可保存/切换/合并
```

```python
# —— 把 LoRA 注入 117 训好的 GPT,验证可训练占比 + 主干冻结 ——
# 接 117 的 model（已预训练好的 tiny GPT）:
# model = inject_lora(model, r=4, alpha=8)            # 注入到每个 Block 的注意力
# 再冻结除 A/B 外的所有参数:
def freeze_except_lora(model):
    for n, p in model.named_parameters():
        leaf = n.split('.')[-1]
        p.requires_grad = (leaf in ('A', 'B'))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"可训练 {n_tr} / 总 {n_all} = {100*n_tr/n_all:.2f}%")   # tiny GPT 上约 1.5%
    return model
# 实测:0.8M 的 tiny GPT 注入 r=4 LoRA 后,可训练参数仅 1.50%(12288 个);
# 注入瞬间因 B=0,forward 输出与注入前完全一致(loss 不变);训练后主干 tok_emb 一字未改。
# 真实大模型上这个比例更低(<1%),因为 d 越大、主干越大,LoRA 占比越小。

# —— 推理时合并 LoRA 进主干:零额外延迟 ——
@torch.no_grad()
def merge_lora(lora_linear):
    """把 scale·B·A 加进 base.weight,推理时就是一个普通 Linear(无旁路开销)。"""
    delta = lora_linear.scale * (lora_linear.B @ lora_linear.A)   # (out, in)
    lora_linear.base.weight += delta
    return lora_linear.base                                       # 合并后可丢掉 A/B
```

```python
# ============ 扩展练习 ============
# 1. 造 5~10 条「指令→回答」样本(套对话模板),用上面的 finetune 跑迷你 SFT,
#    生成对比 SFT 前后:模型是否从「续写问题」变成「回答问题」。
# 2. 改 r(2/8/32)和 alpha,观察可训练参数量与收敛速度的变化。
# 3. 训两个不同任务的 LoRA(各存 A/B),验证「一个主干挂多 LoRA、按需切换」。
# 4. 实现 merge_lora 后保存模型,确认合并版与挂 LoRA 版输出一致(零延迟无损)。
# 5. 把主干用 4-bit 量化(模拟 QLoRA,见 097)再挂 LoRA,体会单卡微调大模型。
# 6. 故意去掉 loss mask(对提示也算 loss),观察模型是否开始「复述问题」。
```

## 面试高频

- **Q:SFT 和预训练有什么不同?** A:目标都是自回归交叉熵,唯一差别是 **loss mask**——SFT 只对「助手回答」算 loss(提示位 label 设 -100 屏蔽),还要套对话模板。本质是「在指令数据上继续训练 + 屏蔽提示」。
- **Q:为什么要屏蔽提示的 loss?** A:提示是输入条件,不是要模型生成的目标;对它算 loss 会让模型学复述问题、浪费容量。只对回答算,模型才学「如何回答」。
- **Q:LoRA 原理?为什么省?** A:假设微调的权重增量 $\Delta W$ 低秩,分解为 $BA$($r\ll d$),冻结 $W$ 只训 $A,B$。可训练参数从 $d^2$ 降到 $2rd$,常占 0.1%~1%;梯度和优化器状态也只覆盖这部分,故省显存。
- **Q:LoRA 的 $B$ 为什么初始化为 0?** A:让旁路起步输出为 0,微调开始时模型完全等同原预训练模型,不破坏已学能力,平滑过渡。$A$ 用小高斯。
- **Q:$\alpha/r$ 缩放干嘛?** A:解耦秩 $r$ 与有效学习率——改 $r$ 时不用重调 lr。常见 $\alpha=2r$ 或固定 $\alpha$。
- **Q:LoRA 注入哪些层?推理有额外开销吗?** A:通常注意力的 $Q,K,V,O$ 投影(也可加 FFN)。推理时可把 $\frac{\alpha}{r}BA$ 合并进 $W$,零额外延迟。
- **Q:QLoRA 是什么?** A:LoRA + 把冻结主干量化成 4-bit(NF4),进一步省显存,让单卡微调大模型成为可能(见 [[097 NF4 与 QLoRA 4-bit|QLoRA]])。
- **Q:LoRA 真正省的是参数还是显存?** A:本质是显存。全参微调一个 7B 要参数 14G + 梯度 14G + AdamW 状态 28G ≈ 56G;LoRA 的梯度和优化器状态只覆盖 <1% 的参数,从 42G 降到几百 MB,单卡即可微调。参数省是表象。
- **Q:为什么低秩假设成立?** A:预训练已学到通用能力,下游任务多是低维子空间的小调整(换领域/风格/格式),不需重写整个 $W$;微调的本征维度实证很低,故 $\Delta W=BA$($r$ 取 4~64)即够。
- **Q:LoRA、Adapter、Prefix Tuning 区别?** A:LoRA 旁挂低秩 $BA$(可合并、零延迟);Adapter 插小瓶颈 MLP(有串行开销、不可合并);Prefix/Prompt Tuning 训可学习虚拟前缀(参数最少、表达力弱)。LoRA 因可合并+可组合+效果好成主流。
- **Q:SFT 效果靠什么?** A:主要靠数据质量而非技巧——LIMA 用 1000 条高质量数据就能对齐;数据要多样、格式统一;警惕全是「知道答案」的样本会加剧幻觉(模型学到「凡问必答」)。

## 关键事实

- SFT(指令微调)是后训练第一步,见 [[081 指令微调 SFT 与数据构造|SFT]]、[[080 后训练总览：SFT 到 RM 到 RLHF|后训练总览]];loss mask 用 `ignore_index=-100` 屏蔽提示位,见 [[060 训练目标与 loss 实现|loss mask]];对话模板见 [[053 词表、特殊 token 与对话模板|对话模板]]。
- LoRA 出自 Hu et al. 2021(arXiv:2106.09685):$\Delta W=BA$,$B$ 初始 0、$A$ 小高斯、缩放 $\alpha/r$,可训练参数 $2rd\ll d^2$;系统见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|高效微调 PEFT]]。
- 三条已验证性质:$B=0$ 起步等同原模型;只训 $A,B$、主干 `requires_grad=False` 全程不变;一个主干可挂多个 LoRA、推理可合并进 $W$ 零延迟。
- QLoRA(4-bit 主干 + LoRA)见 [[097 NF4 与 QLoRA 4-bit|QLoRA]](Dettmers et al. 2023, arXiv:2305.14314);省显存原理(梯度/优化器状态只覆盖 LoRA 参数)见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]。
- 训练循环不变,仍是 [[117 训练一个 tiny GPT(PyTorch,可跑)|forward→loss→backward→step]] 四步,只是 autograd 只对 LoRA 参数算梯度、AdamW 只更新它们(见 [[114 手写自动微分引擎(micrograd 级)|自动微分]]、[[061 优化器与超参(AdamW)|AdamW]])。
- LoRA 真正省的是显存:梯度与优化器状态只覆盖 <1% 的参数,把全参微调 7B 的约 42GB 梯度+优化器状态降到几百 MB,单卡可微调(见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]])。
- 低秩假设的实证依据是微调本征维度低(Aghajanyan et al. 2020);$r$ 取 4~64,任务越复杂越大;PEFT 家族还含 Adapter、Prefix/Prompt Tuning、DoRA/LoRA+ 等变体(见 [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|PEFT]])。
- 推理可把 $\frac{\alpha}{r}BA$ 合并进 $W$(`merge_lora`,delta 形状 $(out,in)$ 与权重一致),合并后与挂 LoRA 输出完全一致、零额外延迟;一主干可挂多 LoRA 按任务切换。
- SFT 效果主要由数据质量决定:LIMA(Zhou et al. 2023, arXiv:2305.11206)用 1000 条高质量数据即可对齐;数据需多样、格式统一,警惕「凡问必答」样本加剧幻觉。
- 关联:总入口 [[113 从零实现总览：课程地图到代码|从零实现总览]];模型与训练 [[117 训练一个 tiny GPT(PyTorch,可跑)|tiny GPT]];偏好对齐下一步 [[082 偏好数据与标注|偏好数据]]、[[083 奖励模型 RM|RM]]、[[084 策略梯度与 PPO 基础|PPO]];这是全课程从零实现轨的收尾。
