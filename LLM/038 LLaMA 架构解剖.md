[[038 LLaMA 架构解剖]]:Meta 2023 年的开源 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|decoder-only]] 大模型——把原始 Transformer 块换上四个零件(Pre-RMSNorm、RoPE、SwiGLU、GQA),其余骨架不变;这套组合成了 2023 年后几乎所有开源 LLM 的“出厂默认配方”。

## 直觉:骨架不变,换四个零件
LLaMA **不是**全新架构,而是 [[036 GPT 系列：自回归与规模化|GPT 式]] decoder-only 的“精装修”。骨架还是那句老话——**注意力 + 前馈 + 残差,堆 N 层**;但每个零件都换成了当时更优的版本:

1. **Pre-RMSNorm**:归一化放在子层**前面**(Pre-Norm,训练更稳),并用更简单的 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|RMSNorm]] 代替 LayerNorm(去掉减均值,只缩放,更快)。
2. **RoPE**:位置不再用正弦绝对编码加在输入,而用[[031 RoPE 旋转位置编码(推导与实现)|旋转位置编码]]直接旋转 Q/K,相对位置天然、外推更好。
3. **SwiGLU**:[[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 的激活从 ReLU/GELU 换成[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|SwiGLU]]门控,质量更高(代价是多一个矩阵)。
4. **GQA**(LLaMA-2 70B 起):注意力从 MHA 换成 [[019 GQA 分组查询注意力|分组查询注意力]],推理省 [[102 KV-Cache|KV-Cache]]。

外加一个小习惯:**几乎去掉所有 bias 项**(线性层无偏置),省参数、训练更稳。

记忆口诀:**“前置 RMS、旋转位置、门控前馈、分组 KV”**——四件套,看到任何现代开源 LLM 基本都是这套。

## 例子:LLaMA-2 7B 的真实配置(小数字)
- 层数 $N=32$,隐藏维 $d=4096$,注意力头 $h=32$(每头 $d_{head}=128$);
- FFN 隐藏维 ≈ **11008**(≈ $\tfrac{8}{3}\times4096$ 后向上取整对齐,而非 $4\times4096=16384$——因为 SwiGLU 多一个矩阵,需缩小隐藏维保持参数量相当);
- 上下文长度 4096,词表 32000,LM Head 与输入 embedding **权重绑定**([[016 输出层、tied embedding 与 logits|tied embedding]])。

到了 **LLaMA-2 70B**:$h=64$ 个 Q 头,GQA 的 K/V 头数 $g=8$ → [[102 KV-Cache|KV-Cache]] 只有 MHA 的 $8/64=1/8$。LLaMA-1(2023.02)还是纯 MHA,GQA 是 LLaMA-2(2023.07)才加上的——这是个高频细节坑。

**手算 LLaMA-2 7B 的参数量(看钱花哪了)**。一层 = 注意力 + SwiGLU FFN:

- 注意力(7B 是 MHA,$d=4096$):$W_q,W_k,W_v,W_o$ 各 $4096\times4096$,共 $4\times4096^2\approx 67$M。
- SwiGLU FFN:三个矩阵 gate/up($4096\times11008$)+ down($11008\times4096$),共 $3\times4096\times11008\approx 135$M。
- **一层 ≈ 202M**,FFN 占了约 2/3。× 32 层 ≈ **6.5B**。
- 再加嵌入($32000\times4096\approx 131$M,因 tied embedding 只算一次)+ 最后 RMSNorm ≈ **6.7B**,正好 7B。

记住这个账:**FFN 是参数大头**(SwiGLU 三矩阵更甚),所以后面 MoE 拿 FFN 开刀(见 [[042 MoE 动机：稀疏激活与容量解耦|MoE]]),GQA 则专砍注意力的 KV-Cache。

**LLaMA 1/2/3 三代速查**(高频对比):

| | LLaMA-1(2023.02) | LLaMA-2(2023.07) | LLaMA-3(2024) |
|---|---|---|---|
| 注意力 | 纯 MHA | 34B/70B 起 GQA | 全系 GQA |
| 上下文 | 2048 | 4096 | 8192(后扩到 128k) |
| 词表 | 32000(SP-BPE) | 32000 | **128256**(tiktoken-BPE) |
| 尺寸 | 7/13/33/65B | 7/13/34/70B | 8/70/405B |

LLaMA-3 最显眼的变化是**词表从 3.2 万暴涨到 12.8 万**——压缩率更高、序列更短、多语言/代码更好(代价是嵌入表更大,见 [[050 分词总览与子词动机|分词总览]])。

![[hist-LLaMA块.png]]

## 原理:逐零件拆解一层 decoder 块
设输入 hidden 为 $x$,一层 LLaMA 块的前向(Pre-Norm 残差):
$$h = x + \text{Attn}\big(\text{RMSNorm}(x)\big),\qquad y = h + \text{FFN}_{\text{SwiGLU}}\big(\text{RMSNorm}(h)\big)$$

**① RMSNorm**(对比 [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]] 去掉了减均值):
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}\odot g$$
只有缩放参数 $g$、没有平移 $\beta$,也不减均值——少一遍均值/方差统计,更快,经验上质量不降。Pre-Norm 把它放在子层**输入端**,保证残差主干是干净的恒等路径,[[009 残差连接与梯度流|梯度]]畅通,深层也好训。

**② RoPE**:对 Q、K 的每一对维度按位置 $m$ 旋转角度 $m\theta_i$。两个位置 $m,n$ 的注意力打分只依赖**相对位置** $m-n$:
$$\big(R_m q\big)^\top \big(R_n k\big) = q^\top R_{n-m}\, k$$
细节见 [[031 RoPE 旋转位置编码(推导与实现)|RoPE 推导]]。

**③ SwiGLU FFN**:把单条上投影拆成“门 + 值”两条,门走 Swish 激活后逐元素乘值:
$$\text{FFN}(x) = W_{\text{down}}\Big(\underbrace{\text{Swish}(W_{\text{gate}}\,x)}_{\text{门}}\ \odot\ \underbrace{(W_{\text{up}}\,x)}_{\text{值}}\Big),\quad \text{Swish}(z)=z\,\sigma(z)$$
三个矩阵(而非经典 FFN 的两个),所以隐藏维取 $\tfrac{8}{3}d$ 来抵消多出来的参数。门控让网络能“动态决定每个通道放行多少”,经验上比 ReLU/GELU 强。

**④ GQA**:把 $h$ 个 Q 头分成 $g$ 组共享 K/V,[[102 KV-Cache|KV-Cache]] $\propto g$ 而非 $h$。详见 [[019 GQA 分组查询注意力|GQA]]。

**为什么去 bias 还能稳?** 经典 Transformer 的线性层都带偏置 $b$。LLaMA 把注意力/FFN 的所有 $b$ 去掉,只留权重 $W$。理由:① 在大规模 + Pre-Norm 下,偏置对效果几乎无贡献;② 少一组参数和一次加法,省显存/带宽;③ 经验上去 bias 训练更稳(偏置有时会漂移)。RMSNorm 也只留缩放 $g$、连平移 $\beta$ 都不要,是同一思路——**能省的都省,只留真正有用的**。

**SwiGLU 隐藏维到底取多少?** 经典 FFN 隐藏维 $4d$、两个矩阵,参数 $2\cdot d\cdot 4d=8d^2$。SwiGLU 有三个矩阵,若也用 $4d$ 则参数 $3\cdot d\cdot 4d=12d^2$,多了一半。为保持 $8d^2$ 不变,解 $3\cdot d\cdot h=8d^2 \Rightarrow h=\tfrac{8}{3}d$。LLaMA-2 7B:$\tfrac{8}{3}\times4096\approx10923$,再向上对齐到 256 的倍数 → **11008**。这就是「为什么是 11008 而不是 16384」的完整推导。

![[hist-SwiGLU.png]]

## 代码:一层 LLaMA 块(可运行骨架)
```python
import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):                          # ✅ 只缩放,不减均值
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.g

class SwiGLU(nn.Module):                            # ✅ 三矩阵门控 FFN
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)   # 注意:无 bias
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))  # SiLU = Swish

class LlamaBlock(nn.Module):
    def __init__(self, d, h, g, hidden):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQAWithRoPE(d, h, g)           # GQA + RoPE(见 019/031)
        self.ffn = SwiGLU(d, hidden)
    def forward(self, x, rope):
        x = x + self.attn(self.n1(x), rope)        # Pre-Norm 残差:norm 在子层前
        x = x + self.ffn(self.n2(x))
        return x
```

```python
# ❌ 常见错误:照搬原始 Transformer 的 Post-Norm,且 FFN 隐藏维仍写死 4*d
class BlockWrong(nn.Module):
    def forward(self, x, rope):
        x = self.n1(x + self.attn(x, rope))        # Post-Norm:深层易不稳(LLaMA 用 Pre-Norm)
        x = self.n2(x + self.ffn4d(x))             # 4*d:SwiGLU 多一矩阵,会让参数量超标
        return x
# ✅ LLaMA:Pre-Norm(norm 在子层输入端),SwiGLU 隐藏维取 ~8/3*d 以对齐参数量
```

```python
# 参数量核对:一层 LLaMA-2 7B 大概多少?
def llama_layer_params(d=4096, ffn_hidden=11008):
    attn = 4 * d * d                       # Wq,Wk,Wv,Wo(7B 是 MHA)
    ffn  = 3 * d * ffn_hidden              # SwiGLU 三矩阵 gate/up/down
    return attn, ffn
a, f = llama_layer_params()
print(f"注意力 {a/1e6:.0f}M, FFN {f/1e6:.0f}M, 一层 {(a+f)/1e6:.0f}M")  # ≈67M / 135M / 202M
print(f"32 层 ≈ {(a+f)*32/1e9:.1f}B + 嵌入 ≈ 6.7B")   # FFN 占大头 → 后续 MoE 改的就是它
```

## 配方零件各自的「之前/之后」

把四件套和它替换掉的旧零件对照,记忆更牢(面试爱问「相对原始 Transformer 改了啥」):

| 零件 | 原始 Transformer(2017) | LLaMA | 换它为了 |
|---|---|---|---|
| 归一化位置 | Post-LN(子层后) | Pre-Norm(子层前) | 深层训练稳 |
| 归一化类型 | LayerNorm(减均值+缩放+平移) | RMSNorm(只缩放) | 更快、更省 |
| 位置编码 | 正弦绝对(加在输入) | RoPE(旋转 Q/K) | 相对位置 + 外推 |
| FFN 激活 | ReLU | SwiGLU(门控) | 效果质量 |
| 注意力 | MHA | GQA(2 代 70B 起) | 省 KV-Cache |
| 偏置 | 有 bias | 去 bias | 省、稳 |

## 面试高频
- **LLaMA 相对原始 Transformer 改了哪几处?** 四处:Post-LN→**Pre-RMSNorm**、正弦绝对位置→**RoPE**、ReLU/GELU FFN→**SwiGLU**、MHA→**GQA**(2 代 70B 起);外加去 bias、tied embedding。骨架(attn+FFN+残差)不变。
- **RMSNorm 比 LayerNorm 省在哪?** 不减均值、无平移参数,少一遍统计 → 更快;经验上质量不降,故现代模型普遍采用。见 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化对比]]。
- **为什么 SwiGLU 的 FFN 隐藏维不是 4d?** 它有 3 个矩阵(gate/up/down)而经典 FFN 只有 2 个;为保持参数量相当,隐藏维缩到约 $\tfrac{8}{3}d$。考点常问这个“为什么不是 4d”。
- **LLaMA-1 有 GQA 吗?** 没有。LLaMA-1 是纯 MHA;GQA 从 **LLaMA-2 的 34B/70B** 开始用($g=8$)。这是高频混淆点。
- **为什么去掉 bias?** 省参数、对训练稳定性几乎无损,大规模下还能省一点显存/带宽;现代配方普遍这么做。
- **这套配方为什么会成为默认?** 每个零件分别解决一个痛点:Pre-RMSNorm→训练稳 + 快;RoPE→相对位置 + 外推;SwiGLU→质量;GQA→省 KV-Cache。组合起来在“稳定、效果、推理成本”上都占优,于是被 [[039 Mistral、Qwen、DeepSeek 架构选择|Mistral/Qwen/DeepSeek]] 等几乎全盘继承,见 [[040 现代 decoder-only 配方汇总|配方汇总]]。
- **LLaMA-2 7B 的 FFN 隐藏维 11008 怎么来的?** $\tfrac{8}{3}\times4096\approx10923$,向上对齐到 256 倍数 → 11008。目的是让 SwiGLU(三矩阵)的参数量与 $4d$ 两矩阵 FFN 相当。
- **LLaMA-3 和 2 最大的架构差异?** 词表从 32000 涨到 128256(换成 tiktoken 风格 BPE),全系用 GQA,上下文 8192 起;骨架仍是四件套。
- **LLaMA 参数主要在 FFN 还是注意力?** FFN(SwiGLU 三矩阵)约占一层 2/3,注意力约 1/3 → 这正是 MoE 改 FFN、GQA 改注意力 KV 的分工逻辑。
- **为什么 RMSNorm 没有平移参数 β?** Pre-Norm 下平移项对效果几乎无贡献,去掉省参更稳;只保留逐通道缩放 $g$ 已足够。

## 关键事实
- 出处:Touvron et al.,*LLaMA: Open and Efficient Foundation Language Models*,2023,arXiv:2302.13971(7B—65B)。LLaMA-2:Touvron et al.,2023,arXiv:2307.09288(加入 GQA、上下文 4096)。
- 四件套:**Pre-RMSNorm**(Zhang & Sennrich 2019)+ **RoPE**(Su et al. 2021,arXiv:2104.09864)+ **SwiGLU**(Shazeer 2020,arXiv:2002.05202)+ **GQA**(Ainslie et al. 2023,arXiv:2305.13245,2 代 70B 用)。
- LLaMA-2 7B 关键超参:$N=32$,$d=4096$,$h=32$,FFN hidden ≈ 11008,词表 32000,上下文 4096,无 bias,tied embedding。
- 性能:LLaMA-13B 在多数基准上超过 GPT-3(175B);65B 与 Chinchilla-70B、PaLM-540B 可比——印证 [[079 Scaling Law 与 Chinchilla 最优|数据充分的小模型胜过欠训的大模型]]。
- 三代演进:LLaMA-1(2048 上下文、纯 MHA、词表 32000)→ LLaMA-2(4096、34B/70B GQA)→ LLaMA-3(8192→128k、全系 GQA、词表 **128256** 换 tiktoken-BPE、8/70/405B)。
- 参数分布:一层 ≈ 注意力 $4d^2$ + FFN $3d\cdot\tfrac83 d\approx 8d^2$,FFN 占约 2/3;7B 模型一层 ≈ 202M × 32 层 + 嵌入 ≈ 6.7B。
- 去 bias + tied embedding 是现代配方的两个小惯例,省参且不损质量。
- 与邻接概念:逐组件链回 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]、[[031 RoPE 旋转位置编码(推导与实现)|RoPE]]、[[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]]/[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|SwiGLU]]、[[019 GQA 分组查询注意力|GQA]];后续衍生见 [[039 Mistral、Qwen、DeepSeek 架构选择|各家架构选择]]。
