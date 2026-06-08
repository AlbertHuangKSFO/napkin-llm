[[107 长上下文推理：YaRN、位置插值与 StreamingLLM]]:在推理期把模型用到超过训练长度的上下文,有两条互补路线——**真正扩大有效窗口**(位置插值 PI、NTK-aware、[[032 RoPE 外推：NTK-aware、位置插值、YaRN|YaRN]],需少量长文微调),与**只保证流式不崩**([[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|StreamingLLM]] 的 attention sink + 滑窗,免微调但丢中段)。

## ① 直觉:为什么超长就崩,两类需求两条路

模型在 4K/8K 长度上训练,推理时塞进 100K,会出两类问题:

1. **位置编码没见过 → 注意力崩坏。** [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 把位置编成「旋转角度」:位置越远旋转越多。超过训练见过的最大角度,模型遇到「没见过的旋转」,注意力分数失真、困惑度爆炸。这是**外推(extrapolation)失败**。
2. **KV-Cache 无限增长 + 流式场景。** 即便位置能处理,逐 token 缓存的 [[102 KV-Cache|KV-Cache]] 随长度线性涨,显存撑不住;聊天机器人这种「永远在线、不断追加」的流式场景尤其要命。

两类需求,对应两条路:

- **需要「真读懂」长文(跨长距离检索、长文档问答)** → 必须**真正扩大有效上下文窗口**:改 RoPE 让远位置落回「见过的角度」+ 少量长文续训。代表:位置插值 PI、NTK-aware、YaRN。
- **只需要「流式不崩、能一直续」(长对话、日志流),不要求记住很久以前** → 用 **StreamingLLM**:KV 只留头部 sink + 近窗,免微调、显存定长,但**丢掉中段**(记不住窗外内容)。

关键区分:**StreamingLLM 不扩大有效窗口**——它让模型「不崩地无限往下生成」,但窗外的内容真的被丢了。两者解决的是不同问题,别混。

## ② 例子:把 2K 模型用到 8K

设模型训练长度 2048,想用到 8192。

**直接外推(什么都不做)**:位置 2049~8192 的旋转角度从未在训练中出现,注意力失真,生成很快胡言乱语。困惑度从个位数飙到几十甚至上百。

**位置插值(PI)**:把所有位置**乘以 2048/8192 = 0.25**,即把 8192 个位置「压缩」回 [0, 2048) 区间——每个位置的旋转角度都落回训练见过的范围。代价:相邻 token 的角度差变小(分辨率被压扁),需在少量长文上微调几百步恢复。简单有效,LLaMA 早期长文版本就靠它。

**NTK-aware / YaRN**:不一刀切压缩,而是**按频率分维处理**。RoPE 不同维度对应不同频率:高频维管「局部、近距离」关系,低频维管「全局、长距离」。

- 高频维**几乎不插值**(保住近程分辨率,否则相邻词都分不清);
- 低频维**充分插值**(延展长程覆盖)。

YaRN 在此基础上再加**注意力温度缩放**(补偿长序列下 logits 分布的变化),仅需 **~0.1% 的预训练数据微调**就达到 SOTA 的窗口扩展效果。DeepSeek、Qwen 等长文模型广泛采用。详细推导见 [[032 RoPE 外推：NTK-aware、位置插值、YaRN|RoPE 外推]]。

![[infer-位置插值与YaRN.png]]

**StreamingLLM(免微调)**:不碰位置、不扩窗口,只改 KV 保留策略——留前 4 个 token(attention sink)+ 最近 2048 个 token 的滑窗,中段全丢。模型能稳定地一路生成下去(实测到 400 万 token 不崩),KV 显存恒定。但它**记不住 2048 窗口外的内容**,所以不能用来做「读完整本书再回答」。

**把 PI 的「压缩」算一个具体数。** RoPE 第 0 对维度(最高频)在位置 8000 的旋转角度本是 $8000\times\text{base}^{0}=8000$ rad。训练只见过到位置 2047 的角度。PI 把位置乘 $2048/8192=0.25$:位置 8000 变成「等效位置 2000」,角度 $2000$ rad——落回训练见过的 $[0,2048)$。代价是相邻两位置(如 7999 与 8000)的角度差从 $1$ rad 缩到 $0.25$ rad,**分辨率被压扁 4 倍**——模型更难区分「相邻」和「隔几个」的 token,所以高频维(管近程)受损最重,需要微调补偿。这正是 NTK/YaRN 改用「高频少压、低频多压」的动机。

**「大海捞针(Needle-in-a-Haystack)」——长上下文的标准压测。** 把一句无关的话(「针」)藏进很长的文档(「海」)的某个深度,问模型这句话内容。横轴是文档长度、纵轴是针的位置,画成热力图。理想是全绿(任何长度、任何位置都能捞出)。它暴露两个常见病:① **超有效窗口后断崖**(外推没做好);② **「中间遗忘」(lost in the middle)**——很多模型对开头/结尾记得牢、对正中间的针召回差。扩窗方法(YaRN/PI)要靠这个测「真扩了没」,而 StreamingLLM 在这个测试上**中段必然失败**(窗外信息已丢)——这恰好印证它「不扩窗、只防崩」。

![[infer-大海捞针热图.png]]

## ③ 原理:RoPE 插值的数学,与 attention sink 的成因

**RoPE 插值为何可行。** RoPE 第 $i$ 对维度的旋转角度:

$$
\theta_i(\text{pos}) = \text{pos}\cdot \text{base}^{-2i/d},\quad i=0,\dots,d/2-1
$$

外推崩的本质:$\text{pos}$ 超过训练最大值,$\theta_i$ 进入未见区间。

- **位置插值(PI)**:把 $\text{pos}$ 换成 $\text{pos}\cdot \frac{L_{\text{train}}}{L_{\text{target}}}$,所有角度等比缩回训练范围。简单但高频维也被压(分辨率损失)。
- **NTK-aware**:不缩 $\text{pos}$ 而是**改 base**:$\text{base}\to \text{base}\cdot s^{d/(d-2)}$($s$ 为扩展倍数)。效果是高频维几乎不变、低频维大幅插值——「该保的保、该插的插」。
- **YaRN(NTK-by-parts + 温度)**:把维度按波长分三段(完全外推 / 完全插值 / 线性过渡),再对注意力 logits 乘温度因子 $\frac{1}{t}=0.1\ln s + 1$,补偿长度变化对 softmax 的影响。最省数据、效果最好。

**为什么改 base 就能「高频少压、低频多压」(NTK 的巧思)。** 旋转角 $\theta_i=\text{pos}\cdot\text{base}^{-2i/d}$。高频维($i$ 小,如 $i=0$):指数 $-2i/d\approx0$,$\theta_i\approx\text{pos}$,**几乎不受 base 影响**——改 base 对它无感,分辨率保住。低频维($i$ 大,接近 $d/2$):指数接近 $-1$,$\theta_i\approx\text{pos}/\text{base}$,**对 base 高度敏感**——base 放大,$\theta$ 缩小,等效插值。所以「调一个 base」就自动实现了「高频维基本不动、低频维充分插值」,无需逐维设参数。这就是 NTK-aware 比 PI(一刀切缩 pos)聪明的地方。

**YaRN 的温度因子在补什么。** 插值后,注意力分数的整体尺度会变(等效位置变密、相邻角度差变小,点积分布收窄),softmax 的「锐度」随之改变。乘一个温度因子 $\frac{1}{t}=0.1\ln s+1$($s$ 是扩展倍数)把 logits 整体放大一点,恢复 softmax 应有的锐度——这是个**经验校正项**,但实测显著提升长上下文质量,也是 YaRN 比纯 NTK 更好的原因之一。

**attention sink 为什么存在(StreamingLLM 的核心洞察)。** 注意力每行经过 softmax,**权重之和恒为 1**。当当前 query 其实「没什么特别该看的」token 时,这多出来的注意力总得有个去处——模型学会把它**倾倒到最初几个 token**上。为什么是开头?因为开头 token 对**所有**后续位置都可见(因果掩码下,位置 0 被每个 query 看到),是天然稳定的「公共停车场」。这些 token 几乎不携带语义,纯当注意力的**泄洪口**。

证据:注意力热图里开头几列总是异常亮;实验把它们删掉(只留滑窗),困惑度立刻爆炸;保留它们,流式生成稳如初。所以 StreamingLLM = **4 个 sink(泄洪口)+ 滑窗(真正有用的近期上下文)**,KV 显存 = 常数。

![[infer-StreamingLLM注意力sink.png]]

**训练侧也要 sink(一个延伸事实)。** 与其推理期被动留 sink,不如**训练时主动加一个专门的「sink token」**(如一个可学习占位 token 放在序列最前),让模型显式拥有「泄洪口」,对 sink 的依赖更干净、长上下文外推更稳——这把「StreamingLLM 的事后补丁」前移成了「架构设计」。

**还有一条「不改位置编码」的扩窗路:稀疏/局部注意力。** Longformer/BigBird 类用「滑动窗口局部注意力 + 少量全局 token」把 $O(n^2)$ 降到 $O(n)$,从源头让长序列可算;但现代 LLM 多仍走 dense + RoPE 外推(YaRN),因为稀疏注意力在「任意跨距精确检索」上不如 dense。这是长上下文的另一象限,与位置外推互补(见 [[014 注意力复杂度 O(n²) 与瓶颈|注意力复杂度]])。

**两条路的取舍**:扩窗口路线(PI/YaRN)真增有效上下文,能跨长距离推理,但要微调、要存全量 KV(配合 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV 压缩]]、[[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 降开销);StreamingLLM 路线零微调、定长 KV,但只解决「不崩」、不解决「记住」。长文训练侧的配合见 [[067 长上下文训练与续训|长上下文训练]]。

## ④ 代码:PI / NTK 缩放 RoPE,与 StreamingLLM 的 KV 保留

```python
import torch

def rope_freqs(dim, base=10000.0):
    i = torch.arange(0, dim, 2).float()
    return base ** (-i / dim)                      # 各维基础频率 θ_i

# ❌ 直接外推:位置超训练长度,角度进入未见区间 → 注意力崩
# ✅ 位置插值(PI):把位置等比缩回训练范围
def rope_pi(pos, dim, L_train, L_target):
    scale = L_train / L_target
    return pos * scale * rope_freqs(dim)           # 所有维同等压缩(高频也被压)

# ✅ NTK-aware:改 base 而非缩位置,高频维几乎不变、低频维充分插值
def rope_ntk(pos, dim, scale_s, base=10000.0):
    base_ntk = base * (scale_s ** (dim / (dim - 2)))
    return pos * (base_ntk ** (-torch.arange(0, dim, 2).float() / dim))

# StreamingLLM:KV 只保留 sink + 滑窗,显存恒定
def streaming_kv_indices(T, n_sink=4, window=2048):
    sink = list(range(min(n_sink, T)))             # 前 4 个 attention sink
    recent = list(range(max(n_sink, T - window), T))  # 最近 window 个
    return sink + recent                            # 中段全丢 → 记不住窗外内容
# 注意:这只保证流式不崩,不等于扩大有效上下文窗口
```

## 面试高频

- **Q:为什么模型一超训练长度就崩?怎么救?** RoPE 把位置编成旋转角度,超长后角度进入训练未见区间、注意力失真。救法:位置插值(缩位置)、NTK-aware(改 base)、YaRN(分段 + 温度),配少量长文微调真正扩窗。
- **Q:PI、NTK-aware、YaRN 的区别?** PI 一刀切缩所有位置(高频分辨率受损);NTK-aware 改 base 使高频少缩、低频多缩;YaRN = NTK-by-parts 分段插值 + 注意力温度缩放,最省数据(~0.1%)效果最好。
- **Q:StreamingLLM 是扩上下文吗?** 不是。它只保证「流式不崩、KV 定长」,丢掉滑窗外的中段,记不住窗外内容;真扩窗要用 YaRN/PI + 微调。
- **Q:attention sink 是什么、为什么不能删?** softmax 强制注意力和为 1,模型把多余注意力倾倒到开头几个 token(对所有位置可见、当泄洪口)。删掉后注意力被迫重分配、分布失稳、困惑度爆炸。
- **Q:StreamingLLM 留几个 sink、配什么?** 通常 4 个 sink + 一个滑动窗口;KV 显存 = 4 + 窗口大小,恒定。
- **Q:长文推理还要配合什么降显存/算力?** 扩窗后 KV 暴涨,需 KV 量化/GQA/分页(103、026)、FlashAttention(025)降注意力开销。
- **Q:怎么验证「真的扩窗了」?** 大海捞针(Needle-in-a-Haystack):把一句话藏进超长文档的不同深度,看模型能否召回。常暴露「中间遗忘(lost in the middle)」——开头结尾记得牢、正中间召回差。StreamingLLM 在此测中段必然失败(窗外已丢),印证它不扩窗。
- **Q:PI 为什么损分辨率?** 它把所有位置等比缩小,相邻位置的角度差也被压(如压 4 倍后相邻角度差从 1 rad 变 0.25 rad),高频维(管近程区分)受损最重,故需微调补偿;NTK/YaRN 改成高频少压、低频多压来缓解。
- **Q:除了 RoPE 外推,长上下文还有别的路吗?** 有:稀疏/局部注意力(Longformer/BigBird,滑窗局部 + 全局 token,把 $O(n^2)$ 降到 $O(n)$)。但现代 dense LLM 多走 RoPE 外推,因稀疏在任意跨距精确检索上偏弱。

## 关键事实

- 直接外推超训练长度会因 RoPE 角度进入未见区间导致困惑度爆炸;位置插值(PI)将位置等比缩回训练范围 + 少量微调即可扩窗(Chen et al., 2023, arXiv:2306.15595)。
- NTK-aware 插值改 RoPE base 使高频维少插值、低频维多插值,兼顾近程分辨率与长程覆盖(bloc97 / Reddit 社区 2023,后被学界吸收)。
- YaRN(Yet another RoPE extensioN)= NTK-by-parts 分段插值 + 注意力温度缩放,仅用约 0.1% 预训练数据微调即达 SOTA 窗口扩展(Peng et al., 2023, arXiv:2309.00071,见 [[032 RoPE 外推：NTK-aware、位置插值、YaRN]])。
- StreamingLLM:保留前几个 attention sink + 滑动窗口,免微调稳定流式生成至 400 万 token;attention sink 源于 softmax 把多余注意力倾倒到初始(对所有位置可见的)token(Xiao et al., 2023, arXiv:2309.17453)。
- StreamingLLM 不扩大有效上下文(丢弃窗外中段),与 PI/YaRN 的「真扩窗」是互补而非替代;扩窗后需配合 KV 压缩与 FlashAttention 控制开销(见 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享]]、[[025 FlashAttention(IO 感知精确注意力)]]、[[067 长上下文训练与续训]])。
- 大海捞针(Needle-in-a-Haystack)是长上下文标准压测:藏一句话进超长文档的不同深度测召回,暴露「中间遗忘(lost in the middle)」(Liu et al., 2023, arXiv:2307.03172,《Lost in the Middle》)。
- 训练期主动加 sink token(可学习占位 token)比推理期被动留 sink 更稳,把 StreamingLLM 的事后补丁前移为架构设计;稀疏/局部注意力(Longformer/BigBird)是另一条不改位置编码的长上下文路线,与 RoPE 外推互补。
