[[075 Medusa 与多头草稿|Medusa]](Cai et al. 2024,arXiv:2401.10774,COLM 2024)用一种比 [[073 投机解码系统：draft-verify 全流程|独立草稿模型]] 更简单的方式做投机解码:**不要任何独立草稿模型**,而是在目标模型最后一层的同一个 hidden state 上,**并接 K 个额外解码头(Medusa head),各自直接预测未来第 1、2、3… 个位置的 token**。多个头一次性吐出多个位置的候选,再用**树状注意力(tree attention)**把各头的 top-k 候选组成候选树,目标模型一次前向并行验证整棵树,取被接受最长的一条路径。它解决了 [[073 投机解码系统：draft-verify 全流程|投机解码]] 落地的痛点:获取并维护一个分词/词表对齐的独立草稿模型很麻烦。Medusa 只需在冻结主干上**训练几个轻量头**。它和 [[074 EAGLE-3：工业标准投机解码|EAGLE]] 同属"轻量草稿头"路线,是 [[LLM/104 投机解码与 Medusa、Lookahead|Medusa]] 的系统视角;但因各头并行预测、彼此不条件,接受率通常**低于** EAGLE。

## 直觉

独立草稿模型要"养一个完整的实习生"——找一个分词、词表都和主编一致的小模型,还要单独跑它,麻烦。

Medusa 的思路是:不养实习生,而是给主编**多长几只手**。主编写下一个字时,大脑(最后一层 hidden state)其实已经隐约知道再往后几个字大概是什么。Medusa 就在这同一个大脑状态上**接几只手,每只手抢着写一个未来位置**:第一只手写 t+2,第二只手写 t+3……

代价是:这几只手**互相不商量**(各头独立、不条件于前一只手写了什么),所以经常前后接不上。补救办法是**每只手都多写几个候选**,组成一棵树,让主编一次扫一整棵树、挑出最连贯的一条接受。这就是树状注意力。

## 例子

设 4 个头(预测 t+1..t+4),各取 top-k 候选:head0 取 1、head1 取 3、head2 取 2、head3 取 2。则候选树叶子数 $=1\times3\times2\times2=12$ 条路径,目标模型**一次前向**(用树掩码)验完这 12 条。

每个头的边际接受率随距离衰减(越远越难猜):比如 $\alpha_1=0.7, \alpha_2=0.55, \alpha_3=0.45$。Medusa-1(冻结主干只训头)在 Vicuna-7B 上报告约 **2.2×** 加速;Medusa-2(主干与头联合微调,接受率更高)约 **2.3–2.8×**。

对比同期 EAGLE 的 $\tau\approx 4$,Medusa 的有效接受长度偏低,因为**各头不条件于前一草稿 token**——这是它简单(无序列依赖)的代价,也是 EAGLE 用"特征级自回归"超越它的地方。

## 原理

设主干在位置 $t$ 输出 hidden $h_t$。第 $i$ 个 Medusa 头是一个轻量残差 MLP + 一个线性投影到词表:

$$
p^{(i)}(x_{t+1+i}\mid h_t) \;=\; \mathrm{softmax}\!\big(W^{(i)}_2\,(\,\mathrm{SiLU}(W^{(i)}_1 h_t)+h_t\,)\big),\quad i=0,\dots,K{-}1
$$

注意所有头都只条件于**同一个** $h_t$,彼此并行、无序列依赖——这是它和 [[074 EAGLE-3：工业标准投机解码|EAGLE]] 自回归草稿头的根本区别(EAGLE 第 $i$ 步条件于前 $i{-}1$ 步)。

**树状注意力**:把各头 top-k 候选展开成一棵前缀树,构造一个特制注意力掩码,让每个候选 token 只能看到它在树中的祖先。这样目标模型**一次前向**就对所有路径并行打分。验证用 [[073 投机解码系统：draft-verify 全流程|draft-verify]] 接受规则(或论文的 typical acceptance 变体),取被接受的最长路径 — 仍保持(典型接受下近似)目标分布。Medusa 论文给出"typical acceptance":接受满足 $p(x)>\epsilon$ 的 token,以温度>0 时换取更高吞吐(此变体非严格无损,贪心时无损)。

这点最易误记为"总是无损"。下图对照两种判据:EAGLE 走严格 $\min(1,p/q)$ + 残差重采,输出分布严格等于目标 $p$(无损);Medusa 的 typical acceptance 只按阈值 $\epsilon$ 收 token、不补残差,温度>0 时会截断/偏移分布,**非严格无损**,仅贪心(argmax)时退化为无损:

![[spec-Medusa近似vs EAGLE严格.png]]

## 图

![[spec-medusa多头.png]]

![[spec-075树注意力验证.png]]

## 代码

```python
# ✅ vLLM:Medusa 作为草稿头加载（需配套训练好的 medusa heads 权重）
from vllm import LLM
llm = LLM(
    model="lmsys/vicuna-7b-v1.3",
    speculative_config={
        "method": "medusa",
        "model": "FasterDecoding/medusa-vicuna-7b-v1.3",  # 配套 medusa heads
        "num_speculative_tokens": 4,                       # = 头的数量 K
    },
)
```

```python
# 概念示意：多头并行预测 + 树验证（伪代码）
h = backbone.last_hidden(prefix)            # 一次主干前向
drafts = [head_i(h).topk(k_i) for head_i in medusa_heads]   # K 个头并行，各取 top-k
tree = build_candidate_tree(drafts)         # 笛卡尔积成树
logits = target(tree, attn_mask=tree_mask)  # ✅ 一次前向，树掩码并行验证所有路径
accepted = longest_accepted_path(tree, logits)
```

```python
# ❌ 反例：把头数 K 开得很大且每头 top-k 也大
"num_speculative_tokens": 12       # 候选树爆炸 → 验证 batch 过大、占满算力
# 大 batch 下树太宽反而拖慢；远距离头接受率低，收益边际递减
```

`❌` 树开太宽 = 验证的有效 batch 暴涨,远端头接受率又低,得不偿失;`✅` K 取 4–5、各头 top-k 递减(近的多取、远的少取),让候选树"上宽下窄"。

## 面试高频

- **Q:Medusa 和投机解码的核心区别?** 不用独立草稿模型,在冻结主干同一 hidden 上并接 K 个解码头各预测一个未来位置,用树状注意力一次验证多候选。
- **Q:Medusa 为什么接受率不如 EAGLE?** 各头只条件于同一个 $h_t$、**彼此独立无序列依赖**;EAGLE 在特征层自回归,第 $i$ 步看到前 $i{-}1$ 步,候选更连贯。
- **Q:树状注意力是干嘛的?** 各头多取候选,笛卡尔积成前缀树,用特制掩码让每个候选只看祖先,目标模型一次前向并行验证所有路径,取最长接受路径。
- **Q:Medusa-1 vs Medusa-2?** Medusa-1 冻结主干只训头(部署简单);Medusa-2 主干与头联合微调,接受率更高但需更重训练。
- **Q:typical acceptance 是无损吗?** 贪心解码下无损;温度>0 时的 typical acceptance 是用轻微近似换吞吐的变体,非严格无损(对比 EAGLE 始终走严格 $\min(1,p/q)$)。

## 关键事实

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**,Cai, Li, Geng, Peng, Lee, Chen, Dao,arXiv:2401.10774(2024),COLM 2024;代码 FasterDecoding/Medusa。
- 机制:冻结主干 + K 个并行解码头(预测 t+1..t+K)+ 树状注意力一次验证多候选;**无独立草稿模型**。
- 收益:Medusa-1 约 **2.2×**、Medusa-2 约 **2.3–2.8×**(Vicuna 系列);接受长度低于 EAGLE,因各头**无序列依赖**。
- 历史定位:开创"轻量草稿头"路线,后被 EAGLE-1/2/3 的特征级自回归草稿超越,现工业多默认 EAGLE-3(见 [[074 EAGLE-3：工业标准投机解码|EAGLE-3]])。
