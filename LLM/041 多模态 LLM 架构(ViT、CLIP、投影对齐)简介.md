[[041 多模态 LLM 架构(ViT、CLIP、投影对齐)简介]]:让 [[038 LLaMA 架构解剖|decoder-only LLM]] “看懂图”的主流配方——用 ViT 把图切块编码、用 CLIP 学到带语义的视觉特征,再经一个投影器把视觉向量“翻译”成 LLM 能读的 token 拼进上下文(LLaVA 式)。

## 直觉:把图片“翻译成 token”塞进上下文
LLM 只会处理 token 序列。要让它看图,核心思路朴素得惊人:**把图片也变成一串 token,和文字 token 拼在一起喂给同一个 LLM**。整条流水线三步:

1. **ViT(切块编码)**:把图片切成一格格 patch,每个 patch 像一个“词”,过一个 Transformer encoder,得到 N 个 patch 特征向量。
2. **CLIP(对齐语义)**:ViT 通常用 CLIP 预训练的版本——CLIP 在 4 亿图文对上做对比学习,使图特征落进**带语义**的空间(知道“狗”长什么样),比纯分类 encoder 更适合接 LLM。
3. **投影对齐(LLaVA 式)**:视觉特征维度 ≠ LLM 的 token 维度,中间加一个**投影器**(一个小 MLP 或 Q-Former),把视觉向量“翻译”成 LLM embedding 空间里的“图 token”,与文本 token 拼成一条序列,LLM 自回归生成回答。

类比:LLM 是个只懂中文的人,图片是法语。ViT+CLIP 先把图片整理成有意义的句子,投影器是翻译官,把它转成中文塞进对话——LLM 根本不知道这几个词原本来自图片,照常往下接。

## 例子:一次“看图问答”的数据流(小数字)
输入一张 $336\times336$ 的图,CLIP-ViT 用 $14\times14$ 的 patch → $24\times24=576$ 个 patch,每个编码成 $1024$ 维特征。

- 视觉 encoder 输出:$576$ 个 $1024$ 维向量。
- 投影器(MLP):$1024 \to 4096$(LLM 隐藏维),得到 $576$ 个 **图 token**(每个 $4096$ 维,与文本 token 同维)。
- 文本问题 “这张图里有什么?” → 比如 $7$ 个文本 token($4096$ 维)。
- 拼接:$[\,576\ \text{图 token}\ \Vert\ 7\ \text{文本 token}\,]$ = $583$ 个 token 的序列,送进 LLaMA,自回归输出 “一只狗在草地上奔跑。”

注意:**576 个图 token 占了绝大部分上下文** → 这正是多模态 LLM 上下文紧张、以及后续工作(如减少图 token 数)的动机。

**分辨率怎么上去:切图块(tiling)**。一张 336×336 只够看清整体,看不清小字/细节。LLaVA-1.5 的 AnyRes / Qwen-VL 等的做法:把高分辨率大图**切成多个 336 子块**分别过 ViT,再拼一个缩略全局图。比如 672×672 切成 4 块 + 1 缩略 = 5 张图 → $5\times576=2880$ 个图 token。代价直接是上下文暴涨——这就是「视觉 token 压缩」(Q-Former、token merging、pixel shuffle 降采样)成为热门方向的原因。

**两条对齐路线的 token 账(MLP vs Q-Former)**:

| 投影器 | 输出图 token 数 | 上下文占用 | 代表 |
|---|---|---|---|
| MLP(LLaVA) | = patch 数(576) | 多 | LLaVA |
| Q-Former(BLIP-2) | = 可学习 query 数(如 32) | 少 18× | BLIP-2 |
| pixel shuffle / pooling | patch 数 ÷ 4 等 | 中 | InternVL 等 |

Q-Former 用一组固定数量的可学习 query 去「问」576 个 patch,把信息压成 32 个 token,**省上下文但可能丢细节**;MLP 简单粗暴一一对应,**保信息但占上下文**。这是工程取舍。

![[hist-ViT-CLIP.svg]]

## 原理:三个组件各自在做什么
**① ViT(Vision Transformer)**:把图 $x\in\mathbb{R}^{H\times W\times3}$ 切成 $P\times P$ 的 patch,展平后线性投影成 patch embedding,加位置编码,前面拼一个 `[CLS]` token,过标准 Transformer encoder:
$$z_0 = [\,x_{\text{cls}};\ E\,x_p^{(1)};\dots;E\,x_p^{(N)}\,] + E_{\text{pos}},\quad N=\frac{HW}{P^2}$$
关键洞见:**图像不需要卷积也能用 Transformer**,把 patch 当 token 即可,大数据下超过 CNN。

**② CLIP(对比学习对齐)**:两个编码器——图编码器(ViT)给图向量 $v$,文本编码器给文本向量 $t$。一个 batch 内 $n$ 对图文,目标(InfoNCE)让**配对的 $(v_i,t_i)$ 相似度高、不配对的低**:
$$\mathcal{L}=-\frac1n\sum_i\log\frac{\exp(\langle v_i,t_i\rangle/\tau)}{\sum_j \exp(\langle v_i,t_j\rangle/\tau)}\ +\ (\text{对称项})$$
训练后图、文落入**同一语义空间**,可零样本分类。其图编码器语义丰富,成为多模态 LLM 最常用的视觉 backbone。这与 [[04 Embedding 与向量数据库|文本 embedding]] 的对比学习思想一脉相承(也是 [[15 多模态 RAG|多模态 RAG]] 的基础)。

**③ 投影对齐(LLaVA 式)**:视觉特征 $H_v\in\mathbb{R}^{N\times d_v}$,经投影器 $W$ 映到 LLM 维度 $d$:
$$H_{\text{img-tok}} = W\,H_v\in\mathbb{R}^{N\times d}$$
然后 $[\,H_{\text{img-tok}}\Vert H_{\text{text}}\,]$ 进 LLM。投影器可以是简单 MLP(LLaVA),也可以是带可学习 query 的 **Q-Former**(BLIP-2,把 N 个 patch 压成少量 query token,省上下文)。

**两阶段训练**(LLaVA):
- 阶段 1(对齐):**冻结视觉 encoder 和 LLM,只训投影器**,用图文对让图 token 落进 LLM embedding 空间;
- 阶段 2(指令微调):**解冻投影器 + LLM**,用“图-问-答”三元组教模型看图对话(视觉 encoder 常仍冻结)。

![[hist-投影对齐.svg]]

## 代码:LLaVA 式前向骨架(可运行结构)
```python
import torch, torch.nn as nn

class LlavaStyle(nn.Module):
    def __init__(self, vision_encoder, llm, d_v=1024, d=4096):
        super().__init__()
        self.vision = vision_encoder          # CLIP-ViT,通常冻结
        self.llm = llm                        # decoder-only LLM
        self.proj = nn.Sequential(            # ✅ 投影器:视觉维 → LLM 维(训练重点)
            nn.Linear(d_v, d), nn.GELU(), nn.Linear(d, d)
        )
    def forward(self, image, text_ids):
        with torch.no_grad():
            patch_feats = self.vision(image)          # (B, N, d_v)  ← 冻结,不回传
        img_tokens = self.proj(patch_feats)           # (B, N, d)    ← 当作图 token
        text_emb = self.llm.embed(text_ids)           # (B, T, d)
        seq = torch.cat([img_tokens, text_emb], dim=1)# (B, N+T, d)  ← 图在前,文在后
        return self.llm.forward_from_embeds(seq)      # 自回归生成回答
```

```python
# ❌ 错误:直接把视觉 encoder 的输出(d_v 维)拼到文本 token(d 维)上
seq = torch.cat([patch_feats, text_emb], dim=1)   # 维度不一致 → 报错;且未对齐语义空间

# ✅ 必须先过投影器对齐维度 + 语义;阶段1 通常只训这个 proj,encoder/LLM 冻结
```

```python
# 图 token 怎么吃上下文:一张图 vs 切块高分辨率
patch = 576                       # 336x336, ViT-L/14 → 24x24=576 patch
print("单图(MLP投影):", patch, "个图 token")          # 576
print("Q-Former:", 32, "个图 token(压缩,省上下文)")   # 32
print("672x672 切4块+缩略:", 5*576, "个图 token")       # 2880 —— 上下文爆炸
print("→ 文本只剩", 4096 - 2880, "/4096 token 给问答")  # 1216,所以要压缩视觉 token
```

## 主流多模态 LLM 谱系(一句话各一个)

零基础常被一堆名字砸晕,记「桥接方式」即可分类:

- **CLIP**(2021):对比学习对齐图文,**不是 LLM**,但是后面所有 MM-LLM 的视觉地基。
- **BLIP-2**(2023):冻结视觉 + 冻结 LLM,中间用 **Q-Former** 桥接(可学习 query 压缩 + 两阶段训练)。
- **LLaVA**(2023):最简方案——**MLP 投影** + 视觉指令微调,效果/成本俱佳,成为开源事实标准模板。
- **Flamingo**(2022):不拼进输入序列,而在 LLM 各层插入 **gated cross-attention**,让文本去 attend 视觉特征(另一条路线)。
- **Qwen-VL / InternVL / LLaVA-1.5/Next**:在 LLaVA 模板上加高分辨率切块、更大数据、token 压缩。
- **原生多模态**(如 Gemini、GPT-4o 倾向):从头把多模态混在一起预训练,而非「事后拼接」。

两条桥接范式:**拼接进输入序列(LLaVA/BLIP-2)** vs **cross-attention 注入(Flamingo)**。前者简单主流,后者省上下文但改 LLM 结构。

## 面试高频
- **怎么让纯文本 LLM “看懂”图?** 三步:ViT 把图切 patch 编码 → (CLIP 预训练让特征带语义)→ 投影器把视觉向量映成 LLM 维度的“图 token”,与文本 token 拼接进同一序列,LLM 当普通 token 处理。
- **ViT 的核心思想?** 把图像切成 patch,每个 patch 当一个 token,用标准 Transformer encoder 处理——“图像不需要卷积”。大数据下超过 CNN。
- **CLIP 在多模态 LLM 里的作用?** 提供**语义丰富的视觉 encoder**:CLIP 用对比学习把图文对齐到同一空间,其图编码器比纯分类 backbone 更懂语义,接 LLM 效果更好。
- **投影器为什么关键?为什么常用 MLP 或 Q-Former?** 它负责把视觉特征**对齐到 LLM 的 embedding 空间**(维度 + 语义)。MLP 简单(LLaVA);Q-Former 用少量可学习 query 把 N 个 patch 压成几十个 token,**节省上下文**(BLIP-2)。
- **LLaVA 的两阶段训练是什么?** 阶段1 冻结 encoder/LLM 只训投影器(对齐);阶段2 解冻投影器 + LLM 做指令微调(学看图对话)。
- **多模态 LLM 的上下文为什么吃紧?** 一张图就可能展开成几百个图 token(如 576),挤占上下文 → 催生“减少视觉 token”类优化。
- **和 [[15 多模态 RAG|多模态 RAG]] 什么关系?** 多模态 RAG 常用 CLIP 把图文嵌到同一空间做**检索**;这里 CLIP 用来把图喂进 LLM 做**生成**——同一对齐思想的两种用法。
- **高分辨率怎么处理?** 把大图切成多个子块各过 ViT(tiling/AnyRes)+ 一张缩略全局图,token 数随块数线性涨 → 催生 token 压缩(Q-Former、pixel shuffle、pooling)。
- **拼接进序列 vs cross-attention 注入,两种范式区别?** LLaVA/BLIP-2 把图 token 拼进输入序列(简单、主流);Flamingo 在 LLM 各层插 gated cross-attention 让文本去 attend 视觉(省上下文但改结构)。
- **为什么视觉 encoder 通常冻结?** 它已被 CLIP 预训练得很好,冻结可省算力、防止小数据微调把它训坏;只训投影器(阶段1)+ LLM(阶段2)。
- **CLIP 的对比损失温度 τ 有什么用?** 缩放相似度 logits、控制对比的「锐度」;CLIP 把 τ 设成可学习参数,训练自动调到合适尖锐度。

## 关键事实
- ViT:Dosovitskiy et al.,*An Image is Worth 16x16 Words*,2020,arXiv:2010.11929(ICLR 2021)。patch + Transformer encoder,大数据下超 CNN。
- CLIP:Radford et al.,*Learning Transferable Visual Models From Natural Language Supervision*,2021,arXiv:2103.00020。**4 亿图文对**、对比学习(InfoNCE)、零样本迁移。
- LLaVA:Liu et al.,*Visual Instruction Tuning*,2023,arXiv:2304.08485。视觉 encoder(CLIP-ViT)+ MLP 投影 + LLM(Vicuna),两阶段训练。
- BLIP-2:Li et al.,2023,arXiv:2301.12597,提出 **Q-Former** 桥接冻结视觉 encoder 与冻结 LLM,大幅省上下文。
- 典型规格:CLIP-ViT-L/14 @336,patch 14、约 576 视觉 token,特征维 1024;投影到 LLM 维(如 4096)。
- Flamingo:Alayrac et al., 2022,arXiv:2204.14198,用 gated cross-attention 注入视觉(非拼接范式);Perceiver Resampler 压缩视觉 token。
- LLaVA-1.5(2024,arXiv:2310.03744)把投影器从单层换成两层 MLP、加 AnyRes 高分辨率;InternVL/Qwen-VL 进一步做 token 压缩与大数据。
- token 压缩方向:Q-Former(query 压缩)、pixel shuffle/pooling(空间降采样)、token merging——核心都是减少 576 这个大头。
- 与邻接概念:LLM 主体是 [[038 LLaMA 架构解剖|LLaMA 式 decoder-only]];对齐思想同 [[04 Embedding 与向量数据库|Embedding]] 对比学习;检索侧用法见 [[15 多模态 RAG|多模态 RAG]];本笔记为“简介”,深入的视觉 token 压缩/分辨率策略另文展开。
