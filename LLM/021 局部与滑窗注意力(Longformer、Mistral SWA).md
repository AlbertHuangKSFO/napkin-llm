[[021 局部与滑窗注意力(Longformer、Mistral SWA)]]:让每个 token **只注意邻近 W 个**(带状掩码),把全注意力的 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 砍成 $O(n\cdot W)$;长程依赖靠"层叠扩感受野 + 少量全局 token"补回。

## 直觉:语言的依赖大多是局部的
全注意力让每个 token 看所有 token,代价 $O(n^2)$,长文档跑不动。但观察:**大部分语义依赖发生在近邻**(一句话内、邻近几句)。于是限制每个 token 只看左右各 $W/2$ 个邻居——注意力矩阵从"满"变成对角线附近的一条**带**(band)。

代价显然:单层看不到远处。两个补丁:
1. **层叠扩感受野**:像 CNN 堆卷积,第 $k$ 层的有效感受野约 $W\cdot k$——堆够层数,顶层就能间接"看到"很远(信息逐层中转)。
2. **全局 token**(Longformer):指定少数关键位(如 `[CLS]`、问句 token)为"全局"——它看所有人、所有人也看它,给远程信息一条直达通道。

类比:开放式办公室里每人只跟邻座聊(滑窗),但有几位"协调员"(全局 token)跟所有人对接;信息要么逐层传话,要么走协调员。

## 例子:Mistral 7B 的 SWA
Mistral 7B:窗口 $W=4096$,层数 32。

- **每层**:token $i$ 只注意 $[i-4096,\ i]$ 这 4096 个 → 单层算力 $\propto n\cdot 4096$ 而非 $n^2$。
- **理论感受野**:$W\times \text{层数} = 4096\times 32 \approx 131072 \approx 131\text{K}$ token——顶层能间接覆盖很长上下文。
- **KV-Cache 上限固定**:配"滚动缓存"(rolling buffer),只保留最近 $W$ 个 token 的 K/V,缓存大小不随序列长 $n$ 增长 → 长序列内存恒定。

小数字对照:$n=32768$ 时,全注意力每层算 $32768^2\approx 10.7$ 亿对;SWA 只算 $32768\times4096\approx 1.34$ 亿对,**约 1/8**;若 $n$ 更长,差距按 $n/W$ 继续拉大。

**感受野逐层手算(为什么"堆层能看远")。** 设窗口左侧半径 $W=4096$(因果版只向左看)。某 token 的"有效感受野"= 经若干层信息中转后,它最终能间接依赖到的最远历史:
- 第 1 层:直接看左侧 4096 → 感受野 4096。
- 第 2 层:它看的那些邻居,在第 1 层各自又看了它们左侧 4096 → 最远到 4096+4096 = 8192。
- 第 $k$ 层:约 $W\cdot k$。
- 第 32 层:$4096\times32\approx131072\approx131\text{K}$。
所以 Mistral 单层窗口只有 4K,**靠 32 层叠加,顶层 token 能间接依赖约 131K 个历史 token**——这正是"窗口小但仍能建模长文"的机制,和 CNN 堆卷积扩感受野同理。注意是"间接依赖"(信息逐层接力),不是"顶层直接 attend 到 131K"。

**手算单层 FLOPs 节省比。** 一层全注意力的 $QK^\top$ 算 $n^2$ 对;SWA 算 $n\cdot W$ 对。比值 $=W/n$。$n=128\text{K}, W=4\text{K}$ 时省到 **1/32**;$n=1\text{M}$ 时省到 **1/256**——$n$ 越长,SWA 越划算,这就是它做长文本便宜的根因。

## 原理:带状掩码
设位置 $i$ 的可见集合(因果 + 窗口):
$$\mathcal{N}(i)=\{\,j:\ i-W < j \le i\,\}$$
注意力只在带内计算:
$$\text{Attn}(i)=\sum_{j\in\mathcal{N}(i)} \frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_{j'\in\mathcal{N}(i)}\exp(q_i^\top k_{j'}/\sqrt{d})}\;v_j$$
非零项每行约 $W$ 个,总复杂度:
$$O(n\cdot W)\quad(\text{当 }W\ll n\text{ 时近似线性})$$

**感受野随层增长**:第 1 层位置 $i$ 直接看 $[i-W,i]$;第 2 层 $i$ 看的邻居又各自在第 1 层看了它们的邻居 → 第 $k$ 层有效感受野约 $W\cdot k$(类似空洞/堆叠卷积)。

**Longformer 的两种窗口 + 全局**:
- 滑窗(sliding window):上面的带状。
- 空洞滑窗(dilated):窗口内跳着取(留空隙),用同样算力覆盖更宽范围。
- 全局注意力(global):少数 $g_0$ 个 token 行/列填满,复杂度增加 $O(n\cdot g_0)$,因 $g_0\ll n$ 仍是 $O(n)$。

**空洞(dilated)的作用细算。** 普通滑窗每层感受野 $+W$;空洞率 $r$ 的滑窗在窗口内每隔 $r$ 个取一个 key,**同样 $W$ 个非零项却横跨 $W\cdot r$ 的范围** → 单层感受野从 $W$ 涨到 $W\cdot r$,$k$ 层达 $W\cdot r\cdot k$。代价:窗口内有"空隙",细粒度局部信息变疏。Longformer 的做法是**低层用稠密滑窗(管局部),高层用空洞滑窗(管长程)**——类似 CNN 里 dilated convolution 的多尺度堆叠,兼顾近邻分辨率与远程覆盖。

**滚动缓存里的"注意力陷阱"(StreamingLLM 的 attention sink)。** 把 KV-Cache 截到最近 $W$ 个看似无损,但实测**一丢掉最早几个 token(尤其第 0 个),困惑度会突然爆炸**——因为模型训练时学会把大量注意力"倾倒"到序列开头的几个 token 上(称 attention sink,一个 softmax 必须把概率分到某处的副产物)。修法([[107 长上下文推理：YaRN、位置插值与 StreamingLLM|StreamingLLM]]):缓存**始终保留开头 4 个 token + 最近的滑动窗口**,即"sink + window",就能稳定地无限长流式生成。这是滑窗/滚动缓存最易踩的坑。

这是降低 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]]里**算力**那一维(与只省缓存的 [[019 GQA 分组查询注意力|GQA]] 互补,二者可叠加)。

![[attn-滑窗带状.png]]

## 代码:带状掩码 + 滚动缓存
```python
import torch, torch.nn.functional as F

# ❌ 全注意力:n×n 掩码,长序列爆显存
def full_causal(q, k, v):                              # (B,h,n,d)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

# ✅ 滑窗:只在 [i-W, i] 内计算(这里用掩码示意;实战用块稀疏 kernel 才真省算)
def sliding_window(q, k, v, W):
    n = q.size(-2)
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    mask = (j <= i) & (j > i - W)                      # 因果 + 窗口 → 带状
    bias = torch.where(mask, 0.0, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

# ✅ 滚动缓存(rolling buffer KV-Cache):缓存只保留最近 W 个,内存不随 n 涨
class RollingCache:
    def __init__(self, W, B, h, d):
        self.W = W
        self.k = torch.zeros(B, h, W, d)
        self.v = torch.zeros(B, h, W, d)
        self.pos = 0
    def update(self, k_t, v_t):                        # 每步写入一个 token 的 K/V
        idx = self.pos % self.W                        # 环形覆盖最旧的
        self.k[:, :, idx] = k_t
        self.v[:, :, idx] = v_t
        self.pos += 1
        return self.k, self.v                          # 永远只 W 个,显存恒定

# 注意:naïve 掩码版仍生成 n×n 的 bias,不省显存;真正的线性加速靠
# FlexAttention / 块稀疏 kernel 跳过窗口外的块(见 [[025 FlashAttention(IO 感知精确注意力)]])
```

## 面试高频
- **SWA 怎么看到全局?** 不靠单层,靠**层叠**:第 $k$ 层有效感受野 ≈ $W\cdot k$(像堆卷积),配合少量全局 token 兜底长程依赖。
- **SWA 的真实卖点是什么?** 配滚动缓存,**KV-Cache 上限固定 = $W$**,长序列内存不随 $n$ 涨;算力降为 $O(nW)$。这是 Mistral 长上下文便宜的原因。
- **Longformer 的全局 token 怎么用?** 任务相关:分类放 `[CLS]`,QA 放问句 token。全局位看所有、被所有看,代价 $O(n\cdot g_0)$ 仍线性。
- **naïve 掩码真能省显存吗?** 不能!生成 $n\times n$ 的 bias 一样 $O(n^2)$ 显存。真正线性要**块稀疏 kernel** 直接跳过窗口外的块(FlexAttention / 自定义 kernel),否则只省了"逻辑上"的算力。
- **SWA 与 [[022 稀疏注意力(BigBird、块稀疏)|BigBird]] 区别?** SWA 只有窗口(+ 可选全局);BigBird = 窗口 + 全局 + **随机**,随机边缩短任意两点的图距离、连通性更强。
- **和 [[019 GQA 分组查询注意力|GQA]] 冲突吗?** 不冲突,正交叠加:GQA 省 KV 头数(带宽),SWA 省序列维算力/缓存长度,实际模型常同时用(Mistral = GQA + SWA)。
- **截断 KV 到最近 W 个为什么会崩、怎么救?** 模型把大量注意力倒进开头的 attention sink,直接丢掉开头会破坏 softmax 归一 → 困惑度爆炸;救法是 StreamingLLM 的"保留开头 4 个 sink + 滑动窗口",即可稳定无限流式。
- **空洞滑窗解决什么?** 用同样算力扩大单层感受野($W\to W\cdot r$),代价是窗口内有空隙;常低层稠密、高层空洞做多尺度。
- **SWA 模型还需要位置编码吗?** 需要。窗口只限制"能看谁",不提供"在第几位/离多远";Mistral 仍用 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]。窗口 + RoPE 各管一摊。
- **滑窗会损失什么任务能力?** 需要"精确跨长程检索/复制"的任务(大海捞针、远距离指代)可能掉点,因为长程信息要逐层接力、可能稀释;纯局部依赖任务几乎无损。

## 关键事实
- Longformer:Beltagy、Peters、Cohan,*Longformer: The Long-Document Transformer*,2020,arXiv:2004.05150。机制 = 滑窗 + 空洞滑窗 + 任务驱动的全局注意力,复杂度随序列长**线性**。
- Mistral 7B:Mistral AI,2023,arXiv:2310.06825。纯 SWA,$W=4096$;32 层 → 理论感受野 ≈ $4096\times32\approx131\text{K}$;用滚动缓存把 KV-Cache 上限固定为 $W$。
- 复杂度:$O(n\cdot W)$($W\ll n$ 时近似线性);全局 token 追加 $O(n\cdot g_0)$。
- 感受野:第 $k$ 层 ≈ $W\cdot k$(类堆叠卷积)。
- 与邻接概念:同属降 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的稀疏家族,更复杂的图样见 [[022 稀疏注意力(BigBird、块稀疏)|BigBird]];精确但 IO 友好的另一路是 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]];彻底换成线性的是 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]]与 [[027 状态空间模型与 Mamba|SSM/Mamba]];滚动缓存属于 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]。
