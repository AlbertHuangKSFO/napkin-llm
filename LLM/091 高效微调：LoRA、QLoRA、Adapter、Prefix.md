[[091 高效微调：LoRA、QLoRA、Adapter、Prefix|高效微调(PEFT)]]:在**冻结预训练大模型主干**的前提下,只训练极少量新增参数(常占总量 0.1%~3%)就能适配下游任务的一类方法;代表是 **LoRA**(学低秩权重增量 $\Delta W=BA$)、**QLoRA**(主干压成 4-bit 再挂 LoRA)、**Adapter**(层内插小模块)、**Prefix/Prompt tuning**(在输入端加可训练前缀)。它让 [[081 指令微调 SFT 与数据构造|SFT]] 等 [[080 后训练总览：SFT 到 RM 到 RLHF|后训练]] 在单卡上可行。

## 直觉:别动那座大山,只在旁边搭个小台阶

全参数微调要把模型**每一个权重**都当成可训练变量。对一个 7B 模型,这意味着不仅要存 7B 参数,还要为它们存梯度、还要为 [[061 优化器与超参(AdamW)|AdamW]] 存两份动量(一阶、二阶),光优化器状态就是参数的好几倍——[[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存]]直接爆。

PEFT 的核心洞察:**适配一个任务,真正需要改的「信息量」很小**。预训练已经把知识装进主干了,微调只是把模型「拨」到某个方向。既然改动量小,就没必要让全部 $d\times d$ 个权重都自由变化——**把改动约束在一个低维子空间里**(LoRA),或者干脆只在**输入端 / 模块间**加一小撮可训参数(Prefix / Adapter)。主干一律**冻结**:不算它的梯度、不存它的优化器状态,显存和算力立刻省下来。

四种思路按「改哪里」分两类:

- **改权重 / 模块**:LoRA(给权重加低秩旁路)、Adapter(在层内串入小瓶颈模块)。
- **改输入 / KV**:Prefix tuning(在每层 KV 前拼可训前缀)、Prompt tuning(只在最底层输入端加可训「软提示」)。

![[post-高效微调家族.png]]

## 例子:LoRA 把可训参数砍到 0.4%(小数字)

设某个注意力投影权重 $W\in\mathbb{R}^{d\times d}$,$d=4096$。

- **全参数微调**:可训参数 $=d^2=4096^2\approx1678$ 万(单这一个矩阵)。
- **LoRA**,取秩 $r=8$:只新增 $A\in\mathbb{R}^{r\times d}$ 和 $B\in\mathbb{R}^{d\times r}$,可训参数 $=2rd=2\times8\times4096=65536$,即 **6.5 万**。
- 比例 $=\dfrac{2rd}{d^2}=\dfrac{2r}{d}=\dfrac{16}{4096}\approx0.39\%$。

也就是说,LoRA 用不到 0.4% 的可训参数撬动这个矩阵的适配。整模型挂上 LoRA 后,优化器只需为这 0.4% 存动量,显存从「几倍参数量」骤降到「主干只读 + 一点点适配器」。

**QLoRA 再进一步**:把冻结的主干从 16-bit 压成 4-bit([[097 NF4 与 QLoRA 4-bit|NF4]] 量化),主干存储再省 4 倍。Dettmers 等人因此在**单张 48GB 显卡**上微调了 **65B** 模型——全参数 16-bit 训练它需要约 780GB。

![[post-lora-低秩旁路.png]]

## 原理:低秩分解 + 旁路相加

**1)LoRA**。微调的目标是给冻结权重 $W_0$ 找一个增量 $\Delta W$,使 $W=W_0+\Delta W$。LoRA 假设**这个增量是低秩的**,把它分解成两个瘦矩阵之积:

$$
\Delta W = B A,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d},\ r\ll d
$$

前向计算变成:

$$
h = W_0 x + \Delta W\,x = W_0 x + \frac{\alpha}{r}\,B A\,x
$$

这里把 $W_0$ 矩阵向量乘和 $BA$ 旁路**相加**,「旁路」这个低秩通路才有梯度。低秩的合法性来自 [[10 SVD 奇异值分解|低秩]]:任何矩阵都能用前 $r$ 个奇异方向近似,微调增量的「有效秩」实测很低,$r=4\sim64$ 足矣。

**初始化**:$A$ 用高斯随机、$B=0$,于是训练起点 $\Delta W=BA=0$,模型**完全等于原始预训练模型**,微调从「不破坏」开始。

**缩放 $\alpha/r$**:$\alpha$ 是一个超参,$\dfrac{\alpha}{r}$ 让你调 $r$ 时不必重调学习率(常令 $\alpha=2r$ 或 $\alpha=r$)。

**推理零延迟**:训练完把 $W=W_0+\frac{\alpha}{r}BA$ **算出来合并**成一个矩阵,部署时和原模型一模一样,**没有额外计算**——这是 LoRA 相对 Adapter 的关键工程优势。

**为什么 $B=0$ 初始化而不是两个都随机?** 若 $A,B$ 都随机,起点 $\Delta W=BA\neq0$,模型一开始就被一个随机扰动破坏(等于在已训好的权重上加噪声),训练得先「修复」这个破坏再学任务,既慢又可能损伤预训练知识。令 $B=0$ ⇒ $\Delta W=0$ ⇒ 起点严格等于原模型,**从不破坏开始**;同时 $A$ 随机保证一旦 $B$ 开始更新,梯度能流向 $A$ 的不同方向(若 $A$ 也为 0,则 $B,A$ 的梯度恒为 0,永远学不动)。所以「一个零、一个随机」是经过设计的。

**多适配器热插拔与服务**:因为 LoRA 旁路独立于主干,可以为不同任务各训一份小 $BA$(每份几 MB),部署时**同一主干 + 按请求切换适配器**——一台机器服务几十个定制模型而只载一份大主干。要零延迟就 merge 进主干(但 merge 后不能再切);要多租户同时服务就**不 merge**、用 batched-LoRA(如 S-LoRA、vLLM 的 multi-LoRA)在 kernel 里按请求选不同 $BA$,牺牲一点延迟换多适配器并发(系统实现见 [[LLM Infra/115 多 LoRA 服务：S-LoRA、Punica 与热插拔|多 LoRA 服务]])。

**LoRA 的进阶变体**(面试加分):① **rsLoRA**——把缩放从 $\alpha/r$ 改成 $\alpha/\sqrt r$,让大 $r$ 时不塌缩,高秩更有效;② **DoRA**(Weight-Decomposed LoRA)——把权重分解成「幅度 + 方向」,只用 LoRA 学方向、单独学幅度,更接近全参微调质量;③ **LoRA+**——给 $A,B$ 设不同学习率($B$ 大于 $A$)加速收敛;④ **VeRA**——多层共享冻结随机 $A,B$、只训极小缩放向量,参数再省一个量级。这些都不改 LoRA 的低秩旁路内核,只调缩放/分解/共享。

**2)QLoRA**(数据流见下图)。主干以 4-bit 存储、冻结;前向时**逐块反量化**回 BF16 做矩阵乘;梯度只回流到 BF16 的 LoRA 旁路。三个关键技术:① **NF4**(4-bit NormalFloat,信息论意义上对正态分布权重最优的量化格点);② **双重量化**(把量化用的缩放常数再量化一次,进一步省内存);③ **分页优化器**(显存峰值时把优化器状态换到 CPU,防 OOM)。

![[post-qlora-数据流.png]]

**3)Adapter**。在每个 Transformer 子层后插入一个**瓶颈模块**:$\text{Adapter}(h)=h+W_{\text{up}}\,\sigma(W_{\text{down}}\,h)$,先降维到小维度 $m\ll d$、激活、再升回 $d$,并带残差。只训 $W_{\text{down}},W_{\text{up}}$。代价:它**串在前向路径里**,推理多一跳、无法像 LoRA 那样合并,稍增延迟。

**4)Prefix / Prompt tuning**。不碰任何权重,只在**输入端拼一段可训练的连续向量**(「软提示」,不是真实 token)。Prefix tuning 在**每一层**的 K、V 前都拼可训前缀(表达力强);Prompt tuning 只在**最底层**加(更省、但需大模型才好用)。本质是用「可学习的上下文」引导冻结模型,适配能力一般略弱于 LoRA。

![[post-peft-四家结构对比.png]]

## 代码:用 PEFT 给模型挂 LoRA(❌ vs ✅)

```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# ❌ 错:直接全参数微调 —— 7B 全部可训,优化器状态爆显存,且易灾难性遗忘
# for p in base.parameters():
#     p.requires_grad = True
# opt = torch.optim.AdamW(base.parameters(), lr=2e-4)   # 单卡基本 OOM

# ✅ 对:冻结主干,只在注意力投影上挂低秩 LoRA
config = LoraConfig(
    r=8,                                   # 秩,常 4~64
    lora_alpha=16,                         # 缩放 alpha/r = 16/8 = 2
    target_modules=["q_proj", "v_proj"],   # 给哪些权重加旁路(常选 q/v 或全部线性层)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)
model.print_trainable_parameters()
# 典型输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# 只有 LoRA 参数有梯度;主干 requires_grad=False,优化器只为 0.06% 参数分配状态
opt = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=2e-4
)
# 训练完可合并回主干,部署零额外延迟:
# merged = model.merge_and_unload()
```

QLoRA 只需在加载时多一步 4-bit 量化(`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`)再 `get_peft_model`,其余流程相同。

## 选型卡:PEFT 方法怎么挑

| 场景 | 选什么 | 为什么 |
|---|---|---|
| 默认首选(显存够、要质量) | **LoRA**(r=8~64,target 全线性层) | 可合并、推理零延迟、质量接近全参 |
| 单卡微调大模型(显存极紧) | **QLoRA**(4-bit NF4 主干 + LoRA) | 主干再省 4×,48GB 卡可调 65B,质量近 16-bit |
| 一台机器服务几十个定制模型 | LoRA **不合并** + multi-LoRA(S-LoRA/vLLM) | 同一主干常驻、按请求切 BA,几 MB/任务热插拔 |
| 多任务共享主干、极致省参 | Prefix/Prompt tuning | 完全不碰权重,但表达力弱、对超参敏感 |
| 想更逼近全参质量 | DoRA / rsLoRA(大 r) | DoRA 拆幅度+方向;rsLoRA 改 α/√r 救高秩 |

## 面试高频

- **LoRA 为什么有效?$\Delta W$ 凭什么能低秩?** 预训练已注入知识,下游适配的「内在维度」很低——微调增量在 [[10 SVD 奇异值分解|奇异值]]意义上有效秩很小,故用秩 $r\ll d$ 的 $BA$ 就能逼近,实测 $r=8$ 常已够用。
- **LoRA 加在哪些层?$r$ 怎么选?** 经典做法加在注意力的 $W_q,W_v$;现代实践常加到**所有线性层**(含 FFN)效果更好。$r$ 越大容量越高也越贵,$4\sim64$ 之间按任务调,$\alpha$ 常取 $2r$。
- **LoRA 推理会变慢吗?** 不会。训练后把 $W_0+\frac{\alpha}{r}BA$ 合并成单个权重,和原模型完全一致;Adapter 因串在前向路径里**会**多一点延迟。
- **QLoRA 和 LoRA 差在哪?** QLoRA = 4-bit [[097 NF4 与 QLoRA 4-bit|NF4]] 量化的冻结主干 + LoRA;靠 NF4、双重量化、分页优化器把 65B 微调塞进单卡,质量接近 16-bit 全参微调。注意主干 4-bit 只用于**存储/前向**,梯度仍在 BF16 的 LoRA 上。
- **LoRA vs Adapter vs Prefix 怎么选?** LoRA:可合并、零延迟、效果接近全参,**默认首选**;Adapter:模块化清晰但增延迟;Prefix/Prompt:完全不碰权重、最省,但表达力弱、对超参敏感,适合多任务共享主干。
- **PEFT 为什么省显存?** 冻结主干 ⇒ 不存其梯度、不存其优化器动量(AdamW 是参数的 2 倍),省的大头在**优化器状态和梯度**,不是参数本身。具体账:全参微调 7B 用 AdamW,参数(2 字节)+ 梯度(2)+ 一阶动量(4)+ 二阶动量(4)≈ 12 字节/参 ×7B ≈ 84GB(还没算激活);LoRA 只为 0.06% 参数存这些,主干仅只读,优化器状态几乎为零。
- **LoRA 会灾难性遗忘吗?** 比全参微调**轻**:主干不动,适配只在低秩子空间,通用能力保留得更好,且可热插拔切换/卸载适配器。
- **$\alpha/r$ 缩放是干嘛的?改 $r$ 要重调学习率吗?** $\alpha/r$ 让 $\Delta W$ 的有效幅度与 $r$ 解耦——你调 $r$(容量)时不必重调学习率。常令 $\alpha=2r$ 或 $\alpha=r$。进阶 rsLoRA 改用 $\alpha/\sqrt r$,使大 $r$ 不塌缩。
- **多个 LoRA 适配器怎么同时上线服务?** 不 merge,用 multi-LoRA(S-LoRA / vLLM):同一主干常驻,按请求在 kernel 里选不同 $BA$,实现一主干服务几十个定制模型;要极致单模型延迟则 merge 进主干(但失去切换能力)。
- **DoRA / rsLoRA / LoRA+ 各改了什么?** DoRA 拆「幅度+方向」只用 LoRA 学方向、更逼近全参;rsLoRA 改缩放为 $\alpha/\sqrt r$ 救高秩;LoRA+ 给 $B$ 比 $A$ 更大学习率加速。都保留低秩旁路内核。
- **QLoRA 训练时主干 4-bit,梯度也是 4-bit 吗?** 不。主干 4-bit 仅用于**存储**,前向时逐块反量化成 BF16 算;梯度只在 BF16 的 LoRA 旁路上回流,主干无梯度。所以精度损失主要来自存储量化,被 NF4 的最优编码压到很小。

## 关键事实

- LoRA:Hu et al.《LoRA: Low-Rank Adaptation of Large Language Models》(Microsoft,2021,arXiv:2106.09685,ICLR 2022)。GPT-3 175B 上可训参数比全参微调少 1 万倍、显存约 1/3,质量持平。
- QLoRA:Dettmers et al.《QLoRA: Efficient Finetuning of Quantized LLMs》(2023,arXiv:2305.14314,NeurIPS 2023)。提出 [[097 NF4 与 QLoRA 4-bit|NF4]]、双重量化、分页优化器;单张 48GB GPU 微调 65B,接近 16-bit 全参质量。
- Adapter:Houlsby et al.《Parameter-Efficient Transfer Learning for NLP》(2019,arXiv:1902.00751,ICML 2019),开 PEFT 先河。
- Prefix tuning:Li & Liang(2021,arXiv:2101.00190);Prompt tuning:Lester et al.《The Power of Scale for Parameter-Efficient Prompt Tuning》(2021,arXiv:2104.08691),指出模型越大、软提示越接近全参微调。
- 工程默认:SFT 阶段以 LoRA/QLoRA 为主流;`r=8~64`、`alpha=2r`、target 常含全部线性层;Hugging Face `peft` 库是事实标准。
- 进阶变体:DoRA(Liu et al. 2024,arXiv:2402.09353,权重幅度/方向分解)、rsLoRA(Kalajdzievski 2023,arXiv:2312.03732,$\alpha/\sqrt r$ 缩放)、LoRA+(Hayou et al. 2024,arXiv:2402.12354,$A/B$ 异学习率)、VeRA(共享冻结随机矩阵 + 极小缩放向量);多适配器服务:S-LoRA(arXiv:2311.03285)、vLLM multi-LoRA。
- 显存直觉:全参 AdamW ≈12 字节/参(参数+梯度+两份动量);LoRA 把这笔大头砍到只覆盖 <0.1% 参数,主干只读,是单卡微调可行的根因。
- 关联:[[081 指令微调 SFT 与数据构造|SFT]]、[[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存]]、[[061 优化器与超参(AdamW)|AdamW]]、[[10 SVD 奇异值分解|低秩分解]]、[[097 NF4 与 QLoRA 4-bit|NF4]]、[[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]。
