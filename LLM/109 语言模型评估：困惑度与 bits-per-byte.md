[[109 语言模型评估：困惑度与 bits-per-byte]]:**评估语言模型最底层的两个内在指标——困惑度($\text{PPL}=e^{\text{CE}}$)读起来直观但绑死 tokenizer,bits-per-byte 按原始字节归一,做到跨分词、跨模型可比**。

## 直觉

语言模型的本职是「预测下一个 token 的概率分布」。最朴素的评估问题是:**它对一段真实文本有多「意外」?** 把模型在每个位置赋予真词的 $-\log$ 概率(也就是 [[30 交叉熵与负对数似然|交叉熵]])在全文上平均,就量化了这种意外。

- 取指数 $e^{(\cdot)}$,得到 [[32 困惑度 Perplexity|困惑度]],尺度是「有效候选词数」,人能读:`PPL=30` ≈ 模型每步像在 30 个等可能词里抽签。
- 但困惑度有个致命脏点:**它是「逐 token」算的,而不同模型的 token 切法不同**。分得越细,序列越长,每个 token 越好猜,逐 token 的 PPL 就越低——**这跟模型真本事无关,纯是分词假象**。所以两个用不同 tokenizer 的模型,PPL 数字根本不能直接比。

bits-per-byte(BPB)就是来补这个洞的:它把「每 token 多少 bit」换算成「**每个原始字节多少 bit**」。分母统一成「字节」这个所有模型共享的物理单位,分词怎么切都被抵消掉,于是 BPB 可以跨 tokenizer、跨模型在同一份原始文本上公平对比。本质上 BPB 衡量的是「这个模型把这段文本无损压缩到多小」——这正是语言建模的信息论根。

![[eval-困惑度与bpb.png]]

## 例子

**例 1:困惑度怎么读。** 词表 $V=1000$,模型对每个词均匀乱猜给 $1/1000$ → 每步交叉熵 $\ln 1000=6.908$ nat → $\text{PPL}=e^{6.908}=1000=V$。乱猜的困惑度恰等于词表大小;完美预测则 $\text{PPL}=1$(下界)。所以 `PPL=20` 说明模型把「有效候选」从 1000 压到了约 20 个。

**例 2:PPL 会骗人。** 同一句 `"unbelievable!!"`(15 字节),两个语言能力相当的模型:

- 模型 A 粗分词成 3 个 token,每 token 交叉熵约 $4.5$ nat → 逐 token $\text{PPL}\approx e^{4.5}=90$。
- 模型 B 细分词成 7 个 token,每 token 交叉熵约 $1.93$ nat → 逐 token $\text{PPL}\approx e^{1.93}=6.9$。

逐 token PPL 差了 13 倍($90$ vs $6.9$),但两个模型对这句话的拟合其实一样好。**这就是为什么不能拿不同分词的 PPL 排名次。**

**例 3:BPB 救场。** 把上面两个换成 BPB(用 $\text{BPB}=\text{CE}\cdot\frac{N_{\text{tok}}}{N_{\text{byte}}}/\ln 2$):

$$\text{A}:\ 4.5\times\tfrac{3}{15}/\ln 2=1.30\qquad \text{B}:\ 1.93\times\tfrac{7}{15}/\ln 2=1.30$$

两者都 $\approx 1.30$ bit/byte,完全一致——因为它们压缩同一份原始字节的真实代价相同。

**例 4:困惑度的尺度感(几个锚点要背)。**
- 完美模型:$\text{PPL}=1$(每步都把正确 token 概率推到 1)。
- 随机均匀猜 $V$ 类:$\text{PPL}=V$(上界,英语字符级约 27,GPT-2 词表约 5 万)。
- 一个还行的英语字符级模型:$\text{PPL}\approx 3\text{–}5$(BPB $\approx 1.1\text{–}1.4$);GPT-2 在 WikiText 上词级 PPL 约 18–35(模型越大越低)。
- 公开数据里**人类对英语文本的 BPB 估计约 1.0–1.2 bit/char**(Shannon 1951 的经典实验);前沿大模型在通用文本上 BPB 已逼近甚至低于这个估计——「压缩即智能」的论调由此而来。

**例 5:温度会改 PPL 吗?(易错)** 报告 PPL 时**用模型原始概率**(温度=1),不能用采样温度。温度是**解码**旋钮(影响生成多样性,见 [[118 采样生成与困惑度评估|采样]]),PPL 是**评估**指标(衡量对真实文本的拟合)——两者不能混。若误用温度<1 算 PPL,会把分布人为变尖、虚低 PPL。

![[eval-bpb跨分词对比.png]]

## 原理

**起点都是每词平均交叉熵。** 对长度 $N$(以 token 计)的文本,模型给出条件概率 $p(w_t\mid w_{<t})$,以自然对数为底:

$$\text{CE}=-\frac{1}{N}\sum_{t=1}^{N}\ln p(w_t\mid w_{<t})\quad(\text{单位 nat/token})$$

**困惑度**是它的指数,等价于「整段文本概率的几何平均的倒数」:

$$\boxed{\ \text{PPL}=\exp(\text{CE})=\Big(\prod_{t=1}^{N}p(w_t\mid w_{<t})\Big)^{-1/N}\ }$$

赋予真实文本的概率越高 → CE 越低 → PPL 越低。详细性质见 [[32 困惑度 Perplexity|困惑度]]。

**bits-per-byte** 做两步换算。第一步把单位从 nat 换成 bit(除以 $\ln 2$);第二步把分母从「token 数」换成「字节数」(乘以 $N_{\text{tok}}/N_{\text{byte}}$):

$$\boxed{\ \text{BPB}=\frac{1}{\ln 2}\cdot\frac{N_{\text{tok}}}{N_{\text{byte}}}\cdot\Big(-\frac{1}{N_{\text{tok}}}\sum_{t}\ln p(w_t\mid w_{<t})\Big)=\frac{-\sum_t \log_2 p(w_t\mid w_{<t})}{N_{\text{byte}}}\ }$$

右边那个形式更直白:**总信息量(bit)÷ 原始字节数**。这就是「编码每个字节平均要几 bit」,即无损压缩率。

**为什么 BPB 抵消了分词。** 细分词让 token 数 $N_{\text{tok}}$ 变大,但每个 token 越好猜、$\ln p$ 的绝对值越小,二者乘积 $\sum_t(-\log_2 p)$ ——即「描述整段文本的总比特数」——基本不变。而 $N_{\text{byte}}$ 是原始文本固有的、与分词无关的常量。所以 $\text{BPB}=\frac{\text{总比特}}{\text{字节}}$ 对分词不敏感。

**一致性要求(易错点)。** 底数无所谓但要统一:$\text{PPL}=2^{\text{以 bit 计的 CE}}=e^{\text{以 nat 计的 CE}}$,同一段文本两者数值相等。报告 PPL 时务必固定 **tokenizer + 测试集 + 上下文长度 + 滑窗步长**,否则不可比;跨分词比较一律改用 BPB(或 bits-per-character)。这一指标与 [[079 Scaling Law 与 Chinchilla 最优|Scaling Law]] 也密切相关:扩规模时常用 BPB / loss 作为纵轴,因为它跨配置可比。

**滑窗(stride)为什么重要(实现细节)。** 长文本超过模型上下文 $L$,要分段算 PPL。最朴素的「不重叠分段」有个 bug:每段开头的 token **几乎没有上文条件**(它前面的 token 在上一段、被切断了),这些「冷启动」位置 PPL 虚高,拉高整体。**滑动窗口**解法:窗口每次只前移 stride(<$L$)步,只对窗口**最后 stride 个**位置算 loss(它们有完整 $L$ 的上文),重叠部分只当上下文不计 loss。stride 越小越准但越慢($L/\text{stride}$ 倍计算)。HuggingFace 的 PPL 文档专门强调这一点——这是「为什么我算的 PPL 比论文高」的常见原因。

![[eval-滑窗困惑度.png]]

**内在 vs 外在评估(定位)。** PPL/BPB 是**内在(intrinsic)**指标:不依赖任何下游任务,直接量化「对文本的概率拟合」,便宜、可在任何语料上算、是预训练阶段监控收敛的主力(loss 曲线就是它)。但它**不直接等于「有用」**:一个 PPL 很低的模型可能照样不会遵循指令、不会推理。所以现代评估必须叠加**外在(extrinsic)**指标——下游基准(见 [[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|下游基准]])和人评(见 [[112 人评、LLM-as-judge 与 Arena|人评]])。一句话:**PPL 测「会不会说话」,基准测「会不会做事」,人评测「人喜不喜欢」。**

![[eval-bpb跨分词对比.png]]

## 代码

```python
import numpy as np

# ❌ 错:直接拿两个不同分词模型的逐 token PPL 比大小
def compare_bad(ce_per_tok_A, ce_per_tok_B):
    ppl_A = np.exp(ce_per_tok_A)   # 90  (粗分词)
    ppl_B = np.exp(ce_per_tok_B)   # 6.9 (细分词)
    return "B 更好" if ppl_B < ppl_A else "A 更好"   # 假象!分词不同根本不可比

# ✅ 对:统一换算成 bits-per-byte,跨分词可比
def bits_per_byte(ce_per_tok_nat, n_tokens, n_bytes):
    # ce_per_tok_nat: 每 token 平均交叉熵(nat);n_tokens/n_bytes: 该段的 token 数与原始字节数
    total_bits = ce_per_tok_nat * n_tokens / np.log(2)   # nat -> bit,得到总比特
    return total_bits / n_bytes                           # 除以原始字节数

# 同一句 "unbelievable!!"(15 字节),两种分词
print(bits_per_byte(4.50, 3, 15))   # ≈ 1.30  (模型 A 粗分词)
print(bits_per_byte(1.93, 7, 15))   # ≈ 1.30  (模型 B 细分词)  → 一致,可比

# 锚点自检:随机均匀猜 V 类,PPL 应 = V
import math
V = 1000
ce_uniform = math.log(V)            # 每步交叉熵 = ln(V)
print("均匀猜 PPL =", round(math.exp(ce_uniform)))   # 1000 = V ✅(随机基线 = 词表大小)
```

```python
import torch, torch.nn.functional as F

# ✅ 长文本用滑窗算 PPL:只对「有完整上文」的最后 stride 个位置计 loss
def ppl_sliding(model, ids, max_len=1024, stride=512):
    nlls, n_tok = [], 0
    prev_end = 0
    for begin in range(0, len(ids), stride):
        end = min(begin + max_len, len(ids))
        trg_len = end - prev_end                 # 本窗真正计 loss 的位置数
        x = ids[begin:end].unsqueeze(0)
        y = x.clone()
        y[:, :-trg_len] = -100                    # 前面只当上下文,不计 loss(关键!)
        logits = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              y.view(-1), ignore_index=-100, reduction='sum')
        nlls.append(nll); n_tok += trg_len
        prev_end = end
        if end == len(ids): break
    return torch.exp(torch.stack(nlls).sum() / n_tok)   # stride 越小越准、越慢
# ❌ 不重叠分段:每段开头 token 无上文、PPL 虚高,整体被拉高
```

```python
import torch, torch.nn.functional as F

def ppl_and_bpb(logits, targets, n_bytes):
    # logits: [N_tok, V]; targets: [N_tok] 真实下一个 token; n_bytes: 这段文本的原始字节数
    ce = F.cross_entropy(logits, targets, reduction='mean')      # 平均 nat 交叉熵
    ppl = torch.exp(ce)                                          # 困惑度(绑分词,仅同模型内比较)
    n_tok = targets.numel()
    bpb = (ce / torch.log(torch.tensor(2.0))) * n_tok / n_bytes  # bits-per-byte(跨分词可比)
    return ppl.item(), bpb.item()
```

## 面试高频

- **「困惑度和交叉熵什么关系?」** $\text{PPL}=e^{\text{平均交叉熵}}$,纯单调变换;PPL 在「词数」尺度上人类可读(有效候选数),CE 在 log 尺度上便于求导。
- **「两个模型的 PPL 能直接比吗?」** 只有 **相同 tokenizer + 相同测试集 + 相同上下文长度** 才能比。换分词后逐 token PPL 不可比——细分词会人为压低 PPL。跨分词请用 bits-per-byte。
- **「bits-per-byte 凭什么跨 tokenizer 可比?」** 它把分母从 token 数换成原始字节数(分词无关的物理量),分子是描述整段文本的总比特(分词怎么切总量基本不变),所以分词被抵消;物理含义是「无损压缩每字节要几 bit」。
- **「PPL 低就一定生成好吗?」** 不一定。PPL/BPB 只衡量对测试文本的概率拟合,与指令遵从、事实性、人类偏好弱相关。所以现代评估还要叠加 [[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|下游基准]] 与人评(见 112)。
- **「BPB 和压缩有什么联系?」** 语言建模 = 无损压缩,$\text{BPB}$ 就是该模型对这段文本的压缩率下界(算术编码可逼近)。这也是「压缩即智能」论调的量化抓手。
- **「长文本算 PPL 为什么要滑窗?」** 不重叠分段时每段开头 token 没上文、PPL 虚高拉高整体;滑窗每次只对有完整上下文的最后 stride 个位置算 loss,重叠部分只当上下文。stride 越小越准越慢。这是「我算的 PPL 比论文高」的常见原因。
- **「算 PPL 用采样温度吗?」** 不用,用模型原始概率(温度=1)。温度是解码旋钮、PPL 是评估指标,混用(温度<1)会人为变尖、虚低 PPL。
- **「内在指标和外在指标怎么分工?」** PPL/BPB(内在)测对文本的拟合,便宜、监控预训练收敛;下游基准(外在)测做事能力;人评测人类偏好。PPL 低 ≠ 有用,三者互补。

## 关键事实

- 困惑度 $\text{PPL}=\exp(\text{每词交叉熵})=\big(\prod_t p(w_t\mid w_{<t})\big)^{-1/N}$,是语言模型内在评估的经典指标(Jelinek et al. 1977 引入;Jurafsky & Martin《Speech and Language Processing》第 3 版 LM 章)。
- PPL 强依赖分词,跨 tokenizer 不可直接比较;HuggingFace 困惑度文档明确建议固定分词、用滑动窗口(stride)计算以减小边界效应(2024)。
- bits-per-byte / bits-per-character 即「以 2 为底、按字节/字符归一的交叉熵」,被引入正是为了让不同分词下的语言建模分数可比;The Pile 论文(Gao et al. 2020,arXiv 2101.00027)即用 BPB 报告各子集结果,序列长 2048、stride 1024。
- 换算关系:$\text{BPB}=\text{CE}_{\text{nat/token}}\cdot\dfrac{N_{\text{tok}}}{N_{\text{byte}}}\big/\ln 2$;等价于「总比特数 ÷ 原始字节数」,物理含义为无损压缩率。
- 现代 LLM 技术报告中 PPL/BPB 已非唯一指标,需配合下游基准与人类评估(GPT、LLaMA 等技术报告通行做法);相关采样与解码影响见「118 采样生成与困惑度评估」。
- 困惑度尺度锚点:完美=1、随机均匀猜 V 类=V(上界);Shannon(1951)估计英语约 1.0–1.2 bit/char,前沿模型 BPB 已逼近/低于此,支撑「压缩即智能」论。
- 长文本算 PPL 须用滑动窗口(stride < 上下文长度),仅对有完整上文的最后 stride 个位置计 loss,否则段首冷启动位置虚高拉高整体(HuggingFace Perplexity 文档,2024)。
- 报告 PPL 用模型原始概率(温度=1),不用解码采样温度;PPL 是内在评估指标,与下游基准、人评互补构成完整评估三件套。
