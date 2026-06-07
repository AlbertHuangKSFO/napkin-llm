[[074 EAGLE-3：工业标准投机解码|EAGLE-3]](Li et al. 2025,arXiv:2503.01840)是 2025 年事实上的工业标准草稿器,把 [[073 投机解码系统：draft-verify 全流程|投机解码]] 的草稿环节做到极致。它不要独立小模型,而是接一个**单层 Transformer 草稿头**,关键不同于 [[075 Medusa 与多头草稿|Medusa]]:草稿头**在目标模型的特征层(hidden state)上自回归**,而非在 token 层。EAGLE-3 相对前代有两处关键升级:(1)**融合低/中/高三层特征**(tri-layer fusion),而非只用顶层特征;(2)**training-time test**——训练时把草稿头自己的预测喂回输入,模拟推理时的误差累积,对齐训练/推理分布。EAGLE-3 还放弃了 EAGLE-1 的"先预测特征再出 token",改为**直接预测 token**,使精度随训练数据持续 scale。它在 [[079 投机解码与连续批、前缀缓存的兼容|vLLM、SGLang]] 中是默认推荐的投机算法。验证仍走 draft-verify 的无损接受规则,所以 **EAGLE-3 不改变输出分布**。

## 直觉

回到主编(目标模型)与实习生(草稿器)的类比。独立小草稿模型的实习生**只能看到主编写出的字**(token),信息少,常猜错,主编得频繁打回。

EAGLE 的实习生不一样:他被允许**偷看主编脑子里的思考过程**——也就是目标模型每一层的 hidden state(特征)。特征比 token 信息量大得多(token 是特征经过 softmax 压成的离散结果,丢了大量信息)。EAGLE-3 更进一步:不只偷看主编"最后一刻的想法"(顶层特征),而是同时看**他思考的早、中、晚三个阶段**(低/中/高三层),综合判断下一个词。所以同样是猜 k 个词,EAGLE 的实习生猜得更准,主编打回得更少——接受率因此远高于独立小草稿。

## 例子

衡量草稿器质量的指标是**平均接受长度** $\tau$(average acceptance length):每轮 draft-verify 平均被接受的 token 数。

- 独立 1B 小草稿验 70B 大模型:$\tau \approx 2{\sim}3$,净加速 ~2×。
- **EAGLE-3 草稿头**:$\tau$ 在 LLaMA 类模型上常达 **4–7**,论文报告整体 **3–6.5× 加速**,显著高于 EAGLE-1/2 与 Medusa。

把 $\tau$ 套进收益模型:草稿头本身极轻(单层,远小于目标模型的一层),草稿开销近似可忽略,故净加速 $\approx \tau / (1 + c_\text{draft})$,$c_\text{draft}$ 很小,使 EAGLE-3 接近"$\tau$ 倍"理想加速。这就是它成为工业标准的原因:草稿便宜 + 接受率高,两头都占。

## 原理

设目标模型在位置 $t$ 输出三层特征 $h^{lo}_t, h^{mid}_t, h^{hi}_t$。EAGLE-3 草稿头先融合:

$$
f_t \;=\; W\,[\,h^{lo}_t \,;\, h^{mid}_t \,;\, h^{hi}_t\,] \;+\; \text{Embed}(x_t)
$$

再用单层 Transformer 在 $f$ 序列上**自回归**地生成草稿 token 与下一步特征,逐步扩成一棵候选树(继承 EAGLE-2 的动态树)。

**training-time test** 解决的是分布漂移:朴素训练时草稿头每步的输入都是"教师强制"的真值特征;但推理时第 2 步起输入的是草稿头**自己上一步的(可能有误的)预测**,分布不匹配 → 误差累积 → 接受率掉。做法是训练时也让草稿头**多步自回归、把自己的输出喂回**,使训练目标直接覆盖推理时的多步分布:

$$
\mathcal{L} \;=\; \sum_{j=1}^{m}\; \mathrm{CE}\big(\,p_\text{target}(x_{t+j}),\; \hat{p}^{(j)}_\text{draft}\,\big),\quad \hat{p}^{(j)} \text{ 用前 } j{-}1 \text{ 步自身预测}
$$

验证沿用 [[073 投机解码系统：draft-verify 全流程|draft-verify]] 的接受规则 $\min(1, p/q)$,所以无论草稿头多准,**输出分布严格等于目标模型**——EAGLE-3 是加速器,不是新模型。

## 图

![[spec-eagle特征级草稿.png]]

![[spec-074三层融合与TTT.png]]

## 代码

SGLang 用 EAGLE-3 的配置(2025 推荐):

```bash
# ✅ SGLang:EAGLE-3 为推荐投机算法；三个旋钮留空则自动调优
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  # --speculative-num-steps / --speculative-eagle-topk / --speculative-num-draft-tokens
  #   三者要么全留空(auto-tune),要么一起显式设
```

```python
# ✅ vLLM:EAGLE3 草稿头随目标模型加载
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
```

```bash
# ❌ 反例：草稿头与目标模型不配套(版本/词表不对)
--speculative-draft-model-path some-eagle-for-a-different-base-model
#   特征维度/词表对不上 → 接受率崩到接近 0，负优化
```

`❌` EAGLE 草稿头是**针对特定目标模型训练**的,换底座必须换配套草稿头;`✅` 用官方放出的配套权重(如 `yuhuili/EAGLE3-*`),并优先让框架自动调 num_steps/topk/draft_tokens。

## 面试高频

- **Q:EAGLE 和独立小草稿模型有什么本质区别?** EAGLE 草稿头吃**目标模型的内部特征(hidden state)**而非只看 token,信息量大,接受率显著更高;且草稿头极轻(单层),开销近乎可忽略。
- **Q:EAGLE-3 相比 EAGLE-1/2 改了什么?** 三点:三层特征融合(低/中/高 vs 仅顶层)、training-time test 对齐训练/推理分布、直接预测 token(放弃"先预测特征"),使精度可随数据 scale。
- **Q:training-time test 解决什么问题?** 推理时草稿头第 2 步起吃自己的(含误差)预测,与教师强制训练分布不匹配 → 误差累积;训练时也多步自回归喂回自身预测来对齐。
- **Q:EAGLE-3 会改变输出质量吗?** 不会。验证仍走 $\min(1,p/q)$ 无损接受规则,分布严格等于目标模型;EAGLE 只是更高接受率的草稿器。
- **Q:为什么说它是工业标准?** vLLM/SGLang/TensorRT-LLM 都默认推荐 EAGLE-3,$\tau$ 高(4–7)、草稿便宜、即插即用,综合加速 3–6.5×。

## 关键事实

- **EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test**,Li et al.,arXiv:2503.01840(2025),NeurIPS 2025;前作 EAGLE-1(2024,特征级自回归)、EAGLE-2(2024,动态草稿树)。
- 三大改动:**tri-layer 特征融合**、**training-time test**(训练时多步自回归喂回自身预测)、**直接预测 token**。
- 指标:平均接受长度 $\tau$ 常达 **4–7**,论文报告 **3–6.5×** 加速,优于 EAGLE-2 / Medusa / lookahead。
- 工程现状(2025):vLLM `method="eagle3"`、SGLang `--speculative-algorithm EAGLE3` 均为推荐路径;草稿头须与目标模型配套训练。
