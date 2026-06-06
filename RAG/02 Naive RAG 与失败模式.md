[[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]里,"Naive RAG"指**全部用默认配置**拼出来的最朴素 [[RAG]]:固定大小分块 + 单一稠密向量检索 + 直接把 top-k 塞进 Prompt + 直接生成。它能跑通 demo,却在真实数据上系统性翻车。本笔记拆开它的失败模式,并把每类病**指向对应的解法笔记**。

## 本质:三段论 Naive → Advanced → Modular
Gao et al. 2023 综述(见[[#来源]])把 RAG 演进归为三段:
- **Naive RAG**:索引—检索—生成一条直线,每环都用最朴素做法。问题最多。
- **Advanced RAG**:在检索前后打补丁——预检索做查询优化(见 [[07 查询变换 Query Transformation|查询变换 Query Transformation]]),后检索做 [[10 重排序 Reranking|重排序 Reranking]] 与压缩。仍是线性管道。
- **Modular RAG**:把各环节解耦成可重排、可循环、可路由的模块(检索、记忆、融合、路由……),支持迭代与自适应,见 [[13 Modular RAG|Modular RAG]]。

记住一句:**Naive 的每一类病,Advanced/Modular 都有对症模块**。所以"失败模式"这张表本质是一张"该读哪篇解法笔记"的导航图。

## 机制:失败发生在哪一侧
失败分两侧:**检索侧**(没捞到 / 捞错 / 切坏)和**生成侧**(捞到了也用不好)。

![[Naive RAG 与失败模式.svg]]

### 检索侧失败(Retrieval-side)
- **召回不全 / top-k 选偏**:单一稠密向量召不全,关键文档排在 k 之外。→ 用 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 补稀疏召回、[[10 重排序 Reranking|重排序 Reranking]] 把对的顶上来。
- **chunk 切坏切碎**:固定字符分块把一个完整语义切两半,或片段太碎丢失上下文。→ [[03 分块策略 Chunking|分块策略 Chunking]]、[[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]](检索小块、喂大块)。
- **query-doc 语义鸿沟**:用户口语问法与文档书面措辞向量不对齐。→ [[07 查询变换 Query Transformation|查询变换 Query Transformation]](改写、HyDE、子问题分解)。
- **库里压根没有**:数据缺失或权限/治理问题导致正确内容不在库内。→ 这是数据问题,见 [[17 检索数据治理|检索数据治理]] 与 [[16 检索安全与访问控制|检索安全与访问控制]]。

### 生成侧失败(Generation-side)
- **仍幻觉 / 不忠实证据**:LLM 脱离检索片段自由发挥,答案与证据冲突。→ [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 做引用约束;[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 让模型自评证据是否够、要不要重检。
- **抓不住重点 lost-in-the-middle**:上下文太长时,夹在中间的关键片段被模型忽视;或被截断逻辑直接丢出窗口。→ [[10 重排序 Reranking|重排序 Reranking]] 把关键证据排到首尾、上下文压缩。
- **冗余 / 重复 / 半截答案**:多片段内容重叠,或只给出部分真相。→ 后检索去重压缩 + [[13 Modular RAG|Modular RAG]] 的迭代检索补全。

## 七个失败点(Seven Failure Points)
Barnett et al. 2024 从三个真实领域系统(研究、教育、生物医药)总结出**工程化 RAG 的七个失败点**,是落地排障的权威清单:

| 编号 | 失败点 | 含义 | 主要侧 | 指向解法 |
|---|---|---|---|---|
| FP1 | Missing Content 内容缺失 | 答案根本不在库里 | 检索 | [[17 检索数据治理|检索数据治理]] |
| FP2 | Missed Top Ranked 漏掉应排前的文档 | 相关文档没进 top-k | 检索 | [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]]、[[10 重排序 Reranking|重排序 Reranking]] |
| FP3 | Not in Context 不在上下文 | 文档检到了,但拼装/截断时被丢出窗口 | 衔接 | [[10 重排序 Reranking|重排序 Reranking]]、上下文压缩 |
| FP4 | Not Extracted 未被抽取 | 答案在上下文里,LLM 却没抽出(噪声/矛盾干扰) | 生成 | [[10 重排序 Reranking|重排序 Reranking]]、[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] |
| FP5 | Wrong Format 格式错误 | 要求表格/列表,模型无视格式指令 | 生成 | Prompt/输出约束 |
| FP6 | Incorrect Specificity 粒度不当 | 答案太泛或太细,不贴合需求 | 生成 | [[07 查询变换 Query Transformation|查询变换 Query Transformation]] |
| FP7 | Incomplete 答案不完整 | 给出半截真相,漏掉关键片段 | 生成 | [[13 Modular RAG|Modular RAG]] 迭代检索、[[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] |

注意:这七点**横跨检索与生成两侧**,且不少病(如 FP3/FP4)发生在"检到了但没用好"的衔接处——单纯换更强的嵌入模型救不了,得靠 Advanced/Modular 的成套手段。

## 可跑最小代码
```python
# 演示 Naive RAG 的脆弱点:固定分块 + 单稠密检索 + 直接塞
def naive_rag(query, corpus, embed, llm, chunk_size=200, k=3):
    # 失败点①:固定字符分块,可能从句子中间切断(chunk 切坏)
    chunks = [c[i:i+chunk_size]
              for c in corpus for i in range(0, len(c), chunk_size)]
    # 失败点②:仅单一稠密向量召回,召不全 → FP2 漏掉应排前文档
    qv = embed.encode([query])[0]
    cv = embed.encode(chunks)
    top = (cv @ qv).argsort()[::-1][:k]
    ctx = "\n".join(chunks[i] for i in top)
    # 失败点③:直接塞、直接生成,无重排/无证据校验 → FP3/FP4 不在上下文/抽不出
    return llm(f"根据资料回答:\n{ctx}\n\n问题:{query}")

# 进阶方向:分块策略 / 混合检索 / 重排序 / Self-RAG 逐一修补上面三处
```

## 何时用 / 坑
- **Naive 只配做原型**:用来快速验证"这批数据值不值得做 RAG",别上生产。
- **先量后治**:不要盲目堆模块。先用 [[18 RAG 评估|RAG 评估]] 定位是检索侧还是生成侧出血,再对症下药——检索召回低就补混合检索/重排,生成不忠实就上引用归因/Self-RAG。
- **失败常在衔接处**:FP3/FP4 提醒——"检到了"不等于"用上了",分块、排序、上下文长度共同决定证据能否真正进入生成。
- **升级路径**:线性补丁见 Advanced;要循环/自适应/路由见 [[13 Modular RAG|Modular RAG]];把检索决策交给模型自主见 [[36 Agentic RAG|Agentic RAG]]。可观测性串联到 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

## 工业界实践

生产里没人会停在 Naive,但**几乎所有人都从 Naive 起步**——它是基线和探针:先用最朴素配置跑通,再用评估定位出血点,逐点升级。这套「先量后治」的方法论比盲目堆模块靠谱得多。

**七个失败点 → 生产对策映射**(落地排障速查):

| 失败点 | 生产里怎么治 | 用什么 |
|---|---|---|
| FP1 内容缺失 | 数据治理:补数据源、修权限、加 ingestion 监控 | Unstructured.io 解析覆盖率告警、[[17 检索数据治理|治理]] |
| FP2 漏掉应排前文档 | 混合检索(dense+BM25)+ 重排 | Qdrant/Weaviate 原生 hybrid、Cohere Rerank |
| FP3 不在上下文 | 控制 k、上下文压缩、重排顶到首尾 | LongLLMLingua 压缩、`top_n` 收紧 |
| FP4 未被抽取 | 重排去噪 + 引用约束 prompt | bge-reranker、强制 `[n]` 引用 |
| FP5 格式错误 | 结构化输出约束 | JSON mode / function calling / Pydantic |
| FP6 粒度不当 | 查询改写 + 多粒度索引 | HyDE、small-to-big |
| FP7 答案不完整 | 迭代/多跳检索 | [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳]]、[[13 Modular RAG|Modular]] |

**真实排障流程**(团队实际这么做):
1. 用 **Ragas** 在测试集上跑 context recall / precision / faithfulness 三个指标(见 [[18 RAG 评估|18]]、[[#来源]])。
2. **context recall 低** → 检索侧出血(FP1/FP2):查数据是否在库、是否 dense 召不全。
3. **context precision 低** → 召回里噪声多,排序乱(FP3/FP4):上重排。
4. **faithfulness 低** → 生成侧出血(FP4):LLM 没忠实证据,上引用约束 / [[12 Self-RAG、CRAG 与 Adaptive RAG|CRAG]]。
5. 用 **Langfuse / LangSmith** 抓线上 trace,人工看 bad case 落在哪个 FP,再回到第 1 步。

```python
# 生产排障骨架:用 Ragas 把失败定位到「检索侧 or 生成侧」
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness
# dataset 含 question / contexts(检到的片段)/ answer / ground_truth
report = evaluate(dataset, metrics=[context_recall, context_precision, faithfulness])
# context_recall 低 → FP1/FP2(检索没捞全)→ 补混合检索/治理
# context_precision 低 → FP3(噪声多/排序乱)→ 上重排
# faithfulness 低 → FP4(检到没用好)→ 上引用约束 / Self-RAG
```

**踩坑速记**:
- **别一步到位上全套模块**:重排、多跳、Self-RAG 都加延迟和成本。没评估指撑就加 = 烧钱不增效。
- **FP3/FP4 是隐形杀手**:「检到了」≠「用上了」。k 设大反而把对的片段挤到 lost-in-the-middle,或被 token 截断丢出窗口。
- **Naive 的固定分块是 FP 重灾区**:从句子中间切断直接制造 FP2/FP4。换 [[03 分块策略 Chunking|递归字符分块]] 是最低成本的第一步升级。

## 面试高频

**Q1:Naive RAG 会有哪些失败模式?分别怎么解?**
标准答:分检索侧(召回不全→混合检索、chunk 切坏→[[03 分块策略 Chunking|分块策略]]、query-doc 语义鸿沟→[[07 查询变换 Query Transformation|查询变换]])和生成侧(仍幻觉→[[11 生成层：引用归因与忠实度|引用归因]]、lost-in-the-middle→[[10 重排序 Reranking|重排]]、答案不全→[[13 Modular RAG|迭代检索]])。能背出 Barnett 2024 的七个失败点 FP1–FP7 是高分项。

**Q2:RAG 的 Naive / Advanced / Modular 三范式区别?**
标准答:Naive = 索引→检索→生成一条直线,每环最朴素;Advanced = 检索前(查询优化)后(重排、压缩)打补丁,仍线性;Modular = 解耦成可重排/可循环/可路由的模块,支持迭代与自适应。出处 Gao et al. 2023 综述。

**Q3(陷阱):「检到了相关文档但答错了」是哪类失败?换更强的 embedding 能解决吗?**
标准答:这是 **FP3(不在上下文)或 FP4(未被抽取)**,发生在「检索→生成」的衔接处,**换 embedding 救不了**。要靠重排把证据顶到首尾、上下文压缩去噪、引用约束 prompt。这题专抓「一遇到 RAG 问题就想换模型」的人。

**Q4:为什么说 RAG 的难点常常不在检索精度?**
标准答:Barnett 2024 明确指出工程化难点常在「检到但没用好」(FP3/FP4)、数据缺失(FP1)、格式/粒度(FP5/FP6)等环节,而非纯检索召回。系统性能由分块、排序、上下文长度、prompt 共同决定。

**Q5:给你一个翻车的 RAG,排障第一步做什么?**
标准答:**先评估,不先改代码**。用 Ragas 看 context recall(检索全不全)/ precision(噪声多不多)/ faithfulness(生成忠不忠实),定位是检索侧还是生成侧,再对症。陷阱答法是「直接换更大的模型 / 加重排」——没定位就动手是反模式。

## 知识拓展

- **七个失败点的来源**:Barnett et al. 2024(arXiv:2401.05856)从研究、教育、生物医药三个真实系统总结 FP1–FP7,是落地排障的权威清单(论文标题:*Seven Failure Points When Engineering a RAG System*)。
- **失败模式与解法的全景对照**:本篇是一张「该读哪篇」的导航图——每类病都有对应进阶笔记。从这里可一路走到 [[13 Modular RAG|Modular RAG]](循环/路由)和 [[36 Agentic RAG|Agentic RAG]](自主决策),可观测性接 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- **反模式清单**:① 不评估就堆模块;② 用换 embedding 模型应对一切问题(救不了 FP3–FP7);③ 把 k 设很大以为「召回越多越好」(反而 lost-in-the-middle + 截断);④ Naive 固定分块上生产(FP 重灾区,见 [[03 分块策略 Chunking|03]])。
- **前沿:让系统自己处理失败**。[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG(Asai 2023)/ CRAG(2024)]] 让模型自评证据质量、不够就触发纠错检索或重写,把「人工排障」的一部分自动化;[[09 多跳检索：IRCoT、Self-Ask、FLARE|FLARE]] 在生成中按需动态检索,直接缓解 FP7 答案不完整。
- **检索为什么会漏(数学层面)**:dense 召回不全的根因是语义向量在精确实体/罕见专名上区分度低,几何上即向量过近难分,背景见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];这也是 [[08 混合检索 Hybrid Search|混合检索]] 要叠 BM25 的原因。

## 关键事实
- "Naive / Advanced / Modular" 三范式由 Gao et al. 2023 综述提出。
- "七个失败点"出自 Barnett et al. 2024,标准编号 FP1–FP7;论文标题用的是 "Seven Failure Points **When** Engineering a Retrieval Augmented Generation System"。
- 七点横跨检索与生成,且明确指出 RAG 工程化的难点常在"检到但没用好"的衔接环节,而非单纯检索精度。

## 来源
- Gao, Y., Xiong, Y., Gao, X., et al. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997. — Naive / Advanced / Modular 三范式。
- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). **Seven Failure Points When Engineering a Retrieval Augmented Generation System**. arXiv:2401.05856. — 三领域案例总结 FP1–FP7。
- Es, S., et al. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217. — context recall/precision、faithfulness 等指标,用于把失败定位到检索侧 or 生成侧。
