[[18 RAG 评估|RAG 评估]]是给一个 [[01 什么是 RAG|RAG]] 系统打分的方法学,难点在于它**两段串联**(检索→生成)且**常无标准答案**:错了到底是检索捞错了片段、还是 LLM 拿到对片段却乱编?评估的第一要务就是**把这两段拆开各自归因**,而不是只看一个端到端的"答对没"。它是 [[20 RAG 开源生态全景|生态]]里最容易被跳过、却最该先建的一层。

## 本质:两段串联,必须分段归因
一次 RAG 回答的质量是两个独立子系统的乘积:**检索质量** × **生成质量**。只看最终答案对错,无法区分两类完全不同的病:
- **检索病**:相关片段根本没被捞回来(召回低),或捞回一堆噪声(精度低)。这时再强的 LLM 也巧妇难为无米之炊。
- **生成病**:片段明明捞对了,LLM 却没用好——要么忽略证据靠 [[01 什么是 RAG|参数记忆]]蒙、要么在证据外**编造**(忠实度低)、要么答非所问。

把这两轴画成二维平面,任何 RAG 系统都落在某个象限,对应不同的修法:

![[RAG 评估-二维.png]]

诊断逻辑:**先看检索指标**,检索不达标就别看生成(脏输入进、脏输出出);检索达标后生成仍差,才去查忠实度/相关性。这套"分段归因"正是 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 里 critic / 评分器在线上做的同一件事,只是评估是离线批量、Self-RAG 是在线单条。

## 机制

### 检索指标(retrieval metrics):有 ground truth 时
当你有"哪些文档/片段是该被检索的"标注(gold passages),用经典 IR 指标:
- **Hit Rate / Recall@k**:top-k 里是否命中至少一个(或全部)相关片段。RAG 最该盯的是 **recall**——漏掉证据是不可逆的,后面再强也救不回。
- **MRR(Mean Reciprocal Rank)**:第一个正确片段排名的倒数均值,在意"对的有没有排前面"。
- **nDCG(normalized Discounted Cumulative Gain)**:对排序位置加权打折,既看相关性分级又看位置,是排序质量的金标准。和 [[10 重排序 Reranking|重排序 Reranking]] 强相关——重排的收益主要体现在 nDCG/MRR 上。
- **Context Precision / Context Recall**:Ragas 把它们做成无需逐文档标注的版本(见下)。

### 生成指标(generation metrics)
- **Faithfulness(忠实度/groundedness)**:答案的每个论断是否都被检索上下文支撑——这是 RAG **最核心**的安全指标,直接对应 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 里"不许在证据外编"。
- **Answer Relevancy(答案相关性)**:答案是否切题(不跑题、不啰嗦冗余),与 ground truth 无关,只看答案 vs 问题。
- **Answer Correctness / Accuracy**:答案与标准答案的事实一致度,**需要 ground truth**,接近传统 QA 的 EM/F1 但用语义判定。

### Ragas:四指标如何无标注地算
[[20 RAG 开源生态全景|Ragas]](`explodinggradients/ragas`)的卖点是**reference-free**——多数指标不需要人写标准答案,靠 LLM 拆解+判定:
- **Faithfulness**:LLM 把答案拆成一组**原子论断(claims)**,逐条问"检索上下文支持/矛盾/沉默?",得分 = 被支持论断的比例。

**Faithfulness 手算**。答案「RAG 由 Lewis 等人 2020 年提出,**首次发表在 ICML**」,检索上下文只说「RAG 由 Lewis et al. 2020 NeurIPS 论文提出」。LLM 拆出 2 条原子 claim:① 「Lewis 等人 2020 年提出 RAG」——上下文**支持**(✓);② 「首次发表在 ICML」——上下文说的是 NeurIPS,**矛盾**(✗)。于是
> $$\text{Faithfulness} = \frac{\text{被支持的 claim 数}}{\text{总 claim 数}} = \frac{1}{2} = 0.5$$
读法:0.5 意味着**一半内容在证据外编造**(这里把 NeurIPS 错写成 ICML)。把答案拆到原子粒度,正是为了让「编了一句」也能被定位扣分,而不是整段一个模糊的「对/错」。
- **Answer Relevancy(新版文档称 Response Relevancy)**:让 LLM 由答案**反推可能的问题**,与原问题算嵌入相似度——答案越切题,反推出的问题越像原问题。
- **Context Precision**:检索回来的片段里,相关片段是否排在前面(信噪+排序)。
- **Context Recall**:把标准答案拆成句子,看每句能否从检索上下文推出——衡量证据**够不够全**(这一项需要 reference)。

> 核验提示:Ragas 官方近版(0.2+)对指标做了**重命名与扩充**——"answer relevancy"在文档里多写作 **Response Relevancy**,"context relevance"拆成了 `LLMContextPrecisionWithoutReference` 等带/不带 reference 的变体,并新增 Noise Sensitivity、Context Entities Recall、Response Groundedness、多模态忠实度等。落地时**以你装的版本文档为准**,别照搬旧四指标名。

### LLM-as-judge:机制与坑
上面几乎所有"语义级"判定底层都是 **LLM-as-judge**:用一个强 LLM 当裁判给答案打分/打标签。它便宜、可扩展、能处理开放式答案,但坑很密:
- **位置偏置 / 冗长偏置**:裁判偏爱排前面的、更长的答案。
- **自我偏好(self-preference)**:裁判倾向给"和自己风格像"的答案高分,用同一个模型既生成又当裁判会系统性虚高。
- **不稳定 / 不可复现**:同一输入多次打分会飘,temperature 要压到 0 并多次取众数。
- **分数无校准**:LLM 给的 1–5 分不是线性的,跨数据集不可比;最好转成二元判定 + 人工抽样核对。
- **必须有人工锚点**:任何 LLM-judge 流水线都要留一小撮人工标注做校准(ARES 的 PPI 就是为此),否则你在用一个没刻度的尺子。这与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 里对 agent 轨迹做 LLM 评判面临的是同一组陷阱。

### 评估框架生态
- **Ragas**(`explodinggradients/ragas`)——RAG 评估事实标准,reference-free 指标 + 合成测试集生成,集成 LlamaIndex/LangChain。
- **TruLens**(`truera/trulens`)——提出 **RAG Triad**(context relevance / groundedness / answer relevance)三角,反馈函数 + 追踪;TruEra 2024 年 5 月被 Snowflake 收购。
- **DeepEval**(`confident-ai/deepeval`)——pytest 原生、最适合塞进 CI 的评估框架,指标库最广(含 G-Eval 自定义),延伸到 agent/chatbot/多模态。
- **ARES**(`stanford-futuredata/ARES`,Saad-Falcon et al. 2023,arXiv:2311.09476)——用合成数据微调**轻量 LM 裁判** + 少量人工标注做 **PPI(预测驱动推断)**纠偏,跨域仍稳。
- **Phoenix**(`Arize-ai/phoenix`)——基于 OpenTelemetry 的开源追踪+评估,偏线上可观测性,与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的 tracing 同源。

## 可跑最小代码
```python
# 用 Ragas 评估一条 RAG 样本:检索+生成四指标一把抓
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # 生成侧:答案是否被上下文支撑(忠实度)
    answer_relevancy,      # 生成侧:答案是否切题(新版叫 ResponseRelevancy)
    context_precision,     # 检索侧:相关片段是否排在前面
    context_recall,        # 检索侧:证据是否够全(需 ground_truth)
)
from datasets import Dataset

# 评估数据:每条含问题、答案、检索到的上下文、标准答案
data = Dataset.from_dict({
    "question":     ["RAG 是谁提出的?"],
    "answer":       ["RAG 由 Lewis 等人在 2020 年提出。"],
    "contexts":     [["RAG 由 Lewis et al. 2020 NeurIPS 论文提出。", "向量库存稠密向量。"]],
    "ground_truth": ["Lewis 等人 2020 年提出 RAG。"],
})

result = evaluate(
    data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)   # {'faithfulness': 1.0, 'answer_relevancy': 0.94,
                #  'context_precision': 1.0, 'context_recall': 1.0}
# 读法:先看 context_* 两项(检索),都达标后再信 faithfulness/relevancy(生成)
```

## 对比表
| 指标 | 归属轴 | 要不要 ground truth | 在意什么 |
|---|---|---|---|
| Hit Rate / Recall@k | 检索 | 要 gold passages | 该捞的捞回来没 |
| MRR / nDCG | 检索 | 要 gold + 排序 | 对的有没有排前面 |
| Context Precision | 检索 | 否(Ragas) | 相关片段是否靠前、信噪 |
| Context Recall | 检索 | 要 reference | 证据够不够全 |
| Faithfulness | 生成 | 否 | 有没有在证据外编造 |
| Answer Relevancy | 生成 | 否 | 答案切不切题 |
| Answer Correctness | 生成 | 要 | 与标准答案的事实一致度 |

## 何时用 / 坑
- **先建小金标集**:哪怕 50 条人工标注的 (问题, gold 片段, 标准答案),也比纯 reference-free 自评可信得多——它是你所有 LLM-judge 的校准锚。
- **别只看端到端分**:端到端答对率掩盖归因。检索 recall 不达标时,任何生成层优化都是浮沙建塔。
- **裁判别用同一个模型**:生成和评判用同一模型会 self-preference 虚高;裁判尽量换更强的独立模型,并人工抽检。
- **分数会随评估模型版本漂**:Ragas/裁判模型升级后历史分数不可直接比,固定评估模型版本并记录。
- **指标名随版本变**:Ragas 0.2+ 改了命名(见上),拿旧博客的四指标名直接 import 会报错。
- **评估也要进 CI**:把 Ragas/DeepEval 跑进回归测试,改了分块/embedding/prompt 后能看见分数升降,否则优化全凭感觉——这与 [[13 Modular RAG|Modular RAG]] 的"可插拔即可度量"是配套的。

## 关键事实
- Ragas 论文:Es et al. 2023,**RAGAS: Automated Evaluation of Retrieval Augmented Generation**,arXiv:2309.15217,首倡 reference-free 的 faithfulness / answer relevance / context relevance 三维。
- ARES 用 **PPI**(prediction-powered inference)以少量人工标注纠正 LLM 裁判的系统偏差,是"LLM-judge 必须有人工锚点"的工程化范本。
- TruLens 的 **RAG Triad** = context relevance + groundedness + answer relevance,是同一"分段归因"思想的另一套命名。
- RAG 评估与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 共享 LLM-as-judge 的全部坑(位置/冗长/自我偏好/不可复现),区别只在评估对象是"单条问答"还是"多步轨迹"。

## 工业界实践

评估在工业界的核心命题是:**把「凭感觉调参」变成「有标尺的回归测试」**。下面是真实落地的关键决策。

**1)金标集(golden set)怎么建——一切的地基**
- 哪怕只有 **50~200 条**人工标注的 `(query, gold_chunks, reference_answer)`,也比纯 reference-free 自评可信得多——它是所有 LLM-judge 的**校准锚**。
- 来源:① 从生产日志里**采样真实 query**(覆盖头部高频 + 长尾难例);② 用 **Ragas / DeepEval 的合成测试集生成器**(从你的文档反向造问答对)冷启动,再人工筛。
- **分层标注**:简单事实题、多跳题、「库里没有答案」的拒答题(测幻觉)、对抗题(诱导编造)。拒答题极重要——很多系统在「该说不知道」时会硬编。

**2)分段归因的工程流水线**
```
检索层先过 → recall@k / nDCG 不达标就停,先修检索(别动生成)
        ↓ 达标
生成层再查 → faithfulness(有没有编)+ answer relevancy(切不切题)+ correctness(对不对)
```
**先看检索、再看生成**是铁律:脏输入进、脏输出出,检索不达标时任何生成层优化都是浮沙建塔。

**3)主流框架的工业定位**
- **Ragas**(`explodinggradients/ragas`)——事实标准,reference-free 四指标 + 合成集;注意 **0.2+ 改了命名**(answer relevancy → Response Relevancy,context precision 拆出带/不带 reference 的变体,新增 Noise Sensitivity、Context Entities Recall、多模态忠实度),**以装机版本文档为准**。
- **DeepEval**(`confident-ai/deepeval`)——**pytest 原生**,最适合塞进 CI/CD;**G-Eval**(LLM-as-judge + CoT 自定义任意标准)和 **DAG metric**(决策树式确定性评分,要可复现时用)是它区别于 Ragas 的杀手锏;背后是 Confident AI 云平台。
- **TruLens**(`truera/trulens`)——**RAG Triad**(context relevance / groundedness / answer relevance)+ 反馈函数;TruEra 2024-05 被 Snowflake 收购,现深度绑 Snowflake 生态。
- **ARES**(`stanford-futuredata/ARES`)——合成数据微调**轻量 LM 裁判** + 少量人工标注做 **PPI** 纠偏,跨域稳;是「LLM-judge 必须有人工锚点」的工程范本。
- **RAGChecker**(`amazon-science/RAGChecker`,NeurIPS 2024,arXiv:2408.08067)——**claim-level entailment**:把答案和 reference 都拆成原子 claim,逐条做蕴含判定,给出 **overall + retriever + generator** 三组细粒度诊断指标(claim precision/recall、context utilization、noise sensitivity、hallucination 等),与人工判断相关性显著高于旧指标。要**精细定位「检索的锅还是生成的锅」**时上它。
- **Phoenix**(`Arize-ai/phoenix`)/ **Langfuse** / **LangSmith**——偏**线上**追踪 + 评估,基于 OpenTelemetry,把离线指标搬到生产做持续监控(在线抽样跑 faithfulness,掉了就告警)。

**4)引用归因的评估**(与 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 配套)
- **ALCE**(`princeton-nlp/ALCE`,EMNLP 2023,arXiv:2305.14627)——首个自动评 LLM 带引用生成的基准,三维:**fluency / correctness / citation quality**(citation precision = 引的来源是否真支持该句,citation recall = 该支持的句子是否都引了)。生产里「引用是否名副其实」就用这套思路:逐句做 NLI 蕴含判定。

**5)离线 + 在线双轨**
- **离线**:每次改分块/embedding/rerank/prompt,在金标集上跑全套指标,进 CI(分数回退则 block 合并)。
- **在线**:生产流量抽样(如 1%)跑 faithfulness/relevancy,监控漂移;真实点击/点赞/人工反馈回流补充金标集。

**6)最佳实践 / 踩坑**
- **裁判别用同生成模型**——self-preference 会系统性虚高;裁判换更强的独立模型并人工抽检。
- **temperature=0 + 多次取众数**压 LLM-judge 的不稳定;**二元判定优于 1–5 打分**(LLM 的分数无校准、跨集不可比)。
- **固定评估模型版本**:Ragas/裁判模型升级后历史分数不可直接比,记录版本。
- **位置/冗长偏置**:裁判偏爱靠前、更长的答案,成对比较时**交换顺序各跑一次**消偏。
- **指标名随版本漂**:照旧博客 import 旧四指标名会报错。

## 面试高频

**Q1:RAG 答错了,你怎么定位是检索的锅还是生成的锅?**
标准答:**分段归因**。先看检索指标(recall@k / nDCG / context recall),检索没捞回证据 = 检索病,LLM 巧妇难为无米之炊;检索达标但答案仍错,再看生成指标(faithfulness 看有没有编、answer relevancy 看切不切题)。把检索/生成画成二维平面,落在哪个象限决定修哪段。
- 追问「检索达标怎么定义?」→ gold passage 的 recall@k 达阈值,或 Ragas 的 context recall/precision 达标。
- 陷阱:只看端到端答对率——它掩盖归因,无法指导优化。

**Q2:没有标准答案(reference),怎么评 RAG?**
标准答:用 **reference-free** 指标。**Faithfulness**(把答案拆成原子 claim,逐条问检索上下文是否支持,得分=被支持比例)和 **Answer/Response Relevancy**(让 LLM 由答案反推问题,与原问题算嵌入相似度)都不需要 reference。只有 **context recall / answer correctness** 需要 reference。底层都是 **LLM-as-judge**。

**Q3:LLM-as-judge 有哪些坑?怎么缓解?**
标准答:位置偏置、冗长偏置、**self-preference**(同模型既生成又评会虚高)、不可复现、分数无校准。缓解:裁判换独立强模型、temperature=0 多次取众数、成对比较交换顺序、二元判定代替打分、**留人工锚点校准**(ARES 的 PPI)。
- 追问「为什么二元比 1–5 分好?」→ LLM 的 1–5 分非线性、跨数据集不可比,二元判定更稳更可比。

**Q4:Faithfulness 和 Answer Correctness 有什么区别?**
标准答:**Faithfulness** 看答案是否被**检索上下文**支撑(不需 reference,防的是「在证据外编造」);**Answer Correctness** 看答案与**标准答案**的事实一致度(需 reference,防的是「答错」)。一个对齐上下文,一个对齐真值——RAG 可以 faithful 但 incorrect(上下文本身就错),所以两者都要看。

**Q5:Ragas 的四个指标分别评什么?哪些要 reference?**
标准答:Context Precision(相关片段是否靠前,不需 ref)、Context Recall(证据够不够全,**需 ref**)、Faithfulness(有没有编,不需 ref)、Answer Relevancy(切不切题,不需 ref)。前两个检索轴、后两个生成轴。
- 追问注意 **0.2+ 命名变了**(Response Relevancy 等),别照搬旧名。

**Q6:评估怎么进 CI/CD?**
标准答:用 **DeepEval**(pytest 原生)或 Ragas 把金标集跑成回归测试,设阈值断言,改了分块/embedding/prompt 后分数回退就 block 合并。配合在线抽样监控漂移。这与 [[13 Modular RAG|Modular RAG]] 的「可插拔即可度量」配套。

## 知识拓展

- **指标和困惑度的关系**:传统语言模型评估(困惑度 / bits-per-byte,见 [[LLM/109 语言模型评估：困惑度与 bits-per-byte|语言模型评估：困惑度与 bits-per-byte]])评的是「模型对下一个 token 的预测分布」,**不评事实正确性**;RAG 评估恰恰要评「答案是否被证据支撑、是否对」,所以困惑度低 ≠ RAG 好,这是两套正交的尺子。RAG 评估更接近下游基准(MMLU/QA 的 EM/F1)而非内在指标。
- **Answer Relevancy 的相似度底层**:Ragas 的 answer relevancy 用「答案反推问题 → 与原问题算嵌入相似度」,相似度就是 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]] 里的余弦——评估和检索共用同一套向量相似度数学。
- **前沿**:**RAGChecker**(2024)的 claim-level entailment 把忠实度从「整答案一个分」细化到「每条 claim 一个判定」,诊断粒度更细;**RAGAS 多模态忠实度**扩展到图文 RAG([[15 多模态 RAG|多模态 RAG]]);**FActScore / SAFE**(原子事实分解评分)是另一支「拆 claim 再逐条验证」的路线,常被借来评长文忠实度。
- **边界与反模式**:① 只信 reference-free 自评、不留人工金标集(尺子没刻度);② 裁判用同一模型(self-preference 虚高);③ 只看端到端分不分段归因;④ 评估不进 CI(优化全凭感觉);⑤ 拿对话/创意类任务套 faithfulness(无证据可对齐,指标失效)。
- **与在线评判同源**:[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 里的 critic/评分器在**线上单条**做的判定,和离线评估在**批量**做的是同一件事——评估是 Self-RAG 的离线版,Self-RAG 是评估的在线内嵌版。RAG 评估的全部 LLM-judge 坑,[[38 Agent 评估与可观测性|Agent 评估与可观测性]] 评多步轨迹时一个不少。

## 来源
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217. — reference-free RAG 评估,faithfulness/answer relevance/context relevance。
- Saad-Falcon et al. 同 ARES;另见 RAGChecker(Ru et al., 2024, NeurIPS,arXiv:2408.08067,`amazon-science/RAGChecker`)claim-level entailment 细粒度诊断;ALCE(Gao et al., EMNLP 2023,arXiv:2305.14627,`princeton-nlp/ALCE`)citation precision/recall 引用归因评估。
- Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M. (2023). **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**. arXiv:2311.09476. — 合成数据微调轻量裁判 + PPI 纠偏,`stanford-futuredata/ARES`。
- Ragas 官方文档(docs.ragas.io,2026)——当前指标集已重命名/扩充(Response Relevancy、Noise Sensitivity、带/不带 reference 的 context precision 变体等),以装机版本为准。
- TruLens 文档(trulens.org)——RAG Triad 三角;TruEra 2024-05 被 Snowflake 收购,`truera/trulens`。
