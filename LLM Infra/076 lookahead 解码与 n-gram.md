[[076 lookahead 解码与 n-gram|lookahead 解码]](Fu et al. 2024,arXiv:2402.02057,ICML 2024,LMSYS)是一种**不需要任何草稿模型、不需额外参数、不需训练**的投机解码:它把朴素自回归 decode 看成解一个非线性方程组,用 **Jacobi 迭代**并行更新多个未来位置,产出一堆"可能错位"的 n-gram,缓存进一个 **n-gram 池**;每一步前向同时干两件事——推进 Jacobi 迭代造新 n-gram,以及从池里取出匹配的 n-gram 候选**并行验证**。和 [[074 EAGLE-3：工业标准投机解码|EAGLE]]、[[075 Medusa 与多头草稿|Medusa]] 需要训练草稿头不同,lookahead **零训练、即插即用**,代价是每步 FLOPs 显著增大(吃 [[004 算力 vs 带宽：Roofline 与算术强度|闲置算力]] 换步数,对 [[014 Decode 阶段：访存受限|访存受限]] 的小 batch 划算)。它是 [[LLM/104 投机解码与 Medusa、Lookahead|Lookahead]] 的系统视角,验证规则保证**输出与原模型一致(无损)**。

## 直觉

朴素 decode 是"解一道必须从左到右一步步代入的连环方程":第 t 个字依赖第 t−1 个字……串行,慢。

Jacobi 迭代的洞察:**先随便给所有未来位置填一批猜测值**,然后一次性把整个方程组代入更新——错位的地方会逐步被"夹"到正确值,像同时拧好几颗螺丝、来回各拧一点直到都到位。在这个"拧螺丝"过程中,虽然多数位置还没收敛,但**沿途冒出的局部连续片段(n-gram)往往是对的**。

lookahead 把这些半成品 n-gram 攒进一个池子。下一步只要发现"我刚写完的最后一个字,正好是池里某条 n-gram 的开头",就**整条拽出来当草稿**,让模型一次验证一长串。所以它既不养实习生(草稿模型),也不长新手(草稿头),纯靠**把 decode 的串行依赖改写成可并行的迭代** + **复用历史 n-gram** 来抢 token。

## 例子

lookahead 有两个旋钮:窗口宽 $W$(每步并行猜多少个未来位置)、n-gram 级数 $N$(往前看多少步的 Jacobi 轨迹)。每步的额外计算量 ≈ 验证 $W$ 个位置 + 池中候选,FLOPs 约放大到朴素的 $O(W)$ 倍。

设接受到的平均 n-gram 长度为 3(每步平均吐 3 个 token)。则步数压缩 ~3×;但每步耗时因 FLOPs 增大而上升(在算力闲置的小 batch 下,每步只多花一点点),净加速约 **1.5–2×**。LMSYS 报告在贪心解码、单 batch 下约 **1.5–2.3×** 加速(模型/任务相关,代码类因 n-gram 重复多收益更高)。

注意:$W$、$N$ 越大,造的 n-gram 越多、接受越长,但每步 FLOPs 越大——**batch 一大就立刻不划算**(算力本就吃满,见 [[079 投机解码与连续批、前缀缓存的兼容|与连续批兼容]])。这是它与 EAGLE 的关键差异:lookahead 靠"多花算力"换 token,batch 敏感。

## 原理

把长度 $m$ 的自回归生成写成不动点方程:$x_i = \arg\max p_\theta(x_i \mid x_{<i})$ 对所有 $i$ 同时成立。**Jacobi 迭代**给一个初值 $\mathbf{x}^{(0)}$,反复并行更新:

$$
x_i^{(k+1)} \;=\; \arg\max_{x} \; p_\theta\!\big(x \mid x_{<i}^{(k)}\big),\qquad i=1,\dots,m \ \text{(全部并行)}
$$

每次迭代是**一次前向**(对所有位置并行),不动点(全部收敛)即等于逐 token 贪心结果。lookahead 用一个 $W\times N$ 的 2D 窗口跟踪 Jacobi 轨迹,从未收敛点抽取**互不相交的 n-gram** 存入池。

**验证分支**与 lookahead 分支在**同一次前向**完成:用注意力掩码,让从池中取出的候选 n-gram 像 [[073 投机解码系统：draft-verify 全流程|draft-verify]] 一样被并行验证,接受满足目标分布的最长前缀。因为验证沿用标准接受规则,**输出与原模型逐 token 一致(lossless)**——Jacobi 与 n-gram 池只是"造草稿"的手段,不碰正确性。代价是单步 FLOPs 上升,所以它在**算力闲置(小 batch、decode 访存受限)** 时收益最大。

## 图

![[spec-lookahead解码.png]]

![[spec-076四种草稿路线.png]]

## 代码

```python
# ✅ lookahead 风格里最实用的工业变体:n-gram / prompt-lookup（vLLM 内置，零训练）
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "prompt_lookup_max": 4,   # 候选 n-gram 最大长度
        "prompt_lookup_min": 2,   # 最小长度
        "num_speculative_tokens": 4,
    },
)
# 从已生成 + prompt 的 n-gram 里找匹配当草稿，对「输入有大量重复」的任务（改写、RAG、代码）特别有效
```

```bash
# ✅ SGLang 的 n-gram 投机（无独立草稿模型，从历史 token 建 n-gram 缓存）
python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM
```

```python
# ❌ 反例：在大 batch、高并发服务上盲开 lookahead/ngram 且把窗口开大
"prompt_lookup_max": 10, "num_speculative_tokens": 10   # 每步 FLOPs 暴涨
# 大 batch 时算力本已吃满，多算的草稿验证直接拖慢整体 → 负优化
```

`❌` 高并发大 batch + 大窗口 = 纯亏算力;`✅` lookahead/n-gram 适合**低并发、低延迟、输入重复度高**(代码补全、文档改写、RAG 引用)的场景,窗口取小;高接受率需求请上 EAGLE-3。

## 面试高频

- **Q:lookahead 解码不需要什么?** 不需要草稿模型、不需要额外参数、不需要训练;纯靠 Jacobi 迭代 + n-gram 池在线造草稿。
- **Q:Jacobi 迭代怎么并行?** 把自回归写成不动点方程,给所有未来位置初值后并行更新,不动点即等于贪心结果;沿途未收敛轨迹冒出的 n-gram 拿来当草稿。
- **Q:它和 EAGLE/Medusa 的根本权衡?** lookahead 零训练但**靠多花 FLOPs 换步数**,batch 敏感(大 batch 失效);EAGLE/Medusa 需训练草稿头但草稿便宜、对 batch 更稳。
- **Q:为什么对代码/RAG 特别有效?** n-gram 池靠重复片段命中,代码与引用类输入重复度高,n-gram 命中率高 → 接受长。
- **Q:n-gram / prompt-lookup 是什么关系?** 是 lookahead 思想的最简工业落地:直接从 prompt + 已生成文本的 n-gram 里找草稿,vLLM/SGLang 内置 `ngram` 方法。

## 关键事实

- **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**,Fu, Bairi, Khabsa, Stoica, Zhang(LMSYS),arXiv:2402.02057(2024),ICML 2024;博客 lmsys.org 2023-11。
- 机制:Jacobi 迭代(自回归 = 解非线性不动点方程)+ 2D 窗口造 disjoint n-gram + n-gram 池 + 同步并行验证;**无草稿模型、零训练、无损**。
- 旋钮:窗口宽 $W$、n-gram 级数 $N$;每步 FLOPs ~$O(W)$ 倍 → **batch 敏感**,大 batch 收益消失。
- 收益:贪心单 batch 约 **1.5–2.3×**(LMSYS);工业最常用其简化版 **n-gram / prompt-lookup**(vLLM `method="ngram"`、SGLang `NGRAM`),对重复输入(代码、RAG)收益最高。
