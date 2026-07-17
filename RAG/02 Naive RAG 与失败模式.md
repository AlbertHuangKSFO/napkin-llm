[[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]把 "Naive RAG"当作一个**显式的简单基线**：固定大小分块、单一检索器、把 top-k 直接拼进 Prompt、一次生成。它可用于确认数据与任务是否值得继续投入；在自己的语料、模型、k 值和访问控制条件下是否可用，不能仅凭名称判断。本笔记把常见症状转成可检验的诊断假设，并链接候选解法。

## 本质:三段论 Naive → Advanced → Modular
Gao et al. 2023 综述(见[[#来源]])把 RAG 演进归为三段:
- **Naive RAG**:索引—检索—生成一条直线，每环采用较少的增强模块，适合做可比较的基线。
- **Advanced RAG**:在检索前后打补丁——预检索做查询优化(见 [[07 查询变换 Query Transformation|查询变换 Query Transformation]]),后检索做 [[10 重排序 Reranking|重排序 Reranking]] 与压缩。仍是线性管道。
- **Modular RAG**:把各环节解耦成可重排、可循环、可路由的模块(检索、记忆、融合、路由……),支持迭代与自适应,见 [[13 Modular RAG|Modular RAG]]。

记住一句：**分类帮助列出候选干预，不替代诊断。** 同一个错误可能来自数据、检索、上下文拼装或生成；升级模块前先以 trace 和标注样本确认假设。

## 机制:失败发生在哪一侧
失败分两侧:**检索侧**(没捞到 / 捞错 / 切坏)和**生成侧**(捞到了也用不好)。

### 直觉/生活类比:体温不是病因
RAG 指标像急诊分诊时的一项体温读数：发烧提示“值得检查”，却不能直接诊断为哪种感染。要知道是数据缺失、ACL 拒绝、召回偏差还是生成未用证据，仍要看病历、化验单——对应到 RAG 就是 trace、gold evidence 和 claim 标注。

### 小数字手算:分数先提示，再验证
设 3 个问题各有一条 gold evidence，某配置在 top-2 中分别找回了“是、否、是”三条。则

$$
\mathrm{Recall@2}=\frac{2}{3}\approx0.667
$$

这个分数提示“至少有一个问题的证据没进入 top-2”，却不能说明原因是 FP1、FP2、ACL、分块还是 query 表述；必须打开这 3 条 trace 和 gold 文档状态核验。

![[Naive RAG 与失败模式.png]]

### 检索侧诊断假设(Retrieval-side)
- **召回不全 / top-k 选偏**:若带 gold evidence 的样本显示相关文档长期在 k 外，单一稠密检索可能是原因之一。对照 [[08 混合检索 Hybrid Search|混合检索]]、[[10 重排序 Reranking|重排序]] 后比较 recall@k、延迟与成本，而不是预设它们必然更好。
- **chunk 切坏切碎**:若 trace 显示答案所需句子被切开或缺失上下文，分块边界/大小是候选原因。用标注样本离线比较固定、递归和 [[06 检索粒度：父文档与句子窗口检索|父文档/句子窗口]]策略。
- **query-doc 语义鸿沟**:若查询改写或词法检索对同一 gold 集显著改善召回，才支持该假设。候选方法见 [[07 查询变换 Query Transformation|查询变换]]。
- **库里压根没有**:先按文档版本、ACL 和 ingestion 状态检查 gold evidence 是否可访问；缺失、权限拒绝与检索失败是不同问题，见 [[17 检索数据治理|检索数据治理]] 与 [[16 检索安全与访问控制|检索安全与访问控制]]。

### 生成侧诊断假设(Generation-side)
- **仍幻觉 / 不忠实证据**:若逐 claim 标注显示答案无法由已拼入的上下文支持，生成侧提示、约束或模型行为是候选原因。再 A/B 测 [[11 生成层：引用归因与忠实度|引用归因]]、[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG/CRAG]]。
- **抓不住重点 lost-in-the-middle**:若正确片段在 trace 的上下文中却未被使用，位置、长度或冲突证据可能相关。改变排序和压缩前后需做成对测试，不把单个案例当因果结论。
- **冗余 / 重复 / 半截答案**:先标注缺失 claim 与重复证据，确认问题在证据覆盖还是生成整合；再比较去重、压缩或 [[13 Modular RAG|Modular RAG]]迭代检索。

## 七个失败点(Seven Failure Points)
Barnett et al. (arXiv v1：2024-01-11)从研究、教育、生物医药三个案例写成经验报告，归纳出 FP1–FP7。它是**案例驱动的诊断分类**，不是权威穷尽清单，也不规定“一个症状只能有一个根因”。把编号用于组织排障时，应回到具体 trace、文档版本和人工标注验证。

| 编号 | 失败点 | 含义 | 主要侧 | 候选干预笔记 |
|---|---|---|---|---|
| FP1 | Missing Content 内容缺失 | 答案根本不在库里 | 检索 | [[17 检索数据治理\|检索数据治理]] |
| FP2 | Missed Top Ranked 漏掉应排前的文档 | 相关文档没进 top-k | 检索 | [[08 混合检索 Hybrid Search\|混合检索 Hybrid Search]]、[[10 重排序 Reranking\|重排序 Reranking]] |
| FP3 | Not in Context 不在上下文 | 文档检到了,但拼装/截断时被丢出窗口 | 衔接 | [[10 重排序 Reranking\|重排序 Reranking]]、上下文压缩 |
| FP4 | Not Extracted 未被抽取 | 答案在上下文里,LLM 却没抽出(噪声/矛盾干扰) | 生成 | [[10 重排序 Reranking\|重排序 Reranking]]、[[11 生成层：引用归因与忠实度\|生成层：引用归因与忠实度]] |
| FP5 | Wrong Format 格式错误 | 要求表格/列表,模型无视格式指令 | 生成 | Prompt/输出约束 |
| FP6 | Incorrect Specificity 粒度不当 | 答案太泛或太细,不贴合需求 | 生成 | [[07 查询变换 Query Transformation\|查询变换 Query Transformation]] |
| FP7 | Incomplete 答案不完整 | 给出半截真相,漏掉关键片段 | 生成 | [[13 Modular RAG\|Modular RAG]] 迭代检索、[[09 多跳检索：IRCoT、Self-Ask、FLARE\|多跳检索：IRCoT、Self-Ask、FLARE]] |

注意：这七点横跨检索、拼装与生成。FP3/FP4 可提示“检到了但未用好”的假设；它们**不能单独证明** embedding、分块或生成器中的哪一个是根因。用相同 query 的完整 trace 和标注证据做对照，再选择干预。

## 可跑最小代码
```python
# 可直接运行：把简单基线的 chunk、检索与拼装 trace 打出来。
# trace 只产生诊断材料；不能据此直接给某个 FP 判根因。
from collections import Counter
import re

corpus = ["ACME 在 Q2 2023 的营收同比增长 3%。", "TS-999 是支付网关错误码。"]

def tokens(text):
    # 英数词保留完整 token；中文按字切分，避免整段中文只能零分并列。
    return Counter(re.findall(r"[A-Za-z0-9-]+|[\u4e00-\u9fff]", text.lower()))

def overlap_score(query_tokens, chunk):
    return sum((query_tokens & tokens(chunk)).values())

def naive_trace(query, chunk_size=14, k=2):
    chunks = [doc[i:i + chunk_size] for doc in corpus for i in range(0, len(doc), chunk_size)]
    q = tokens(query)
    ranked = sorted(((chunk, overlap_score(q, chunk)) for chunk in chunks),
                    key=lambda pair: pair[1], reverse=True)
    context = [chunk for chunk, _ in ranked[:k]]
    scores = dict(ranked)
    # 第二段“营收同比增长”含 营/收/增/长，与 query 的真实重叠为 4，不是零分并列。
    assert scores[chunks[1]] == 4 and chunks[1] in context
    return {"query": query, "chunks": chunks, "retrieved": context,
            "scores": scores, "prompt": "\n".join(context),
            "chunk_size": chunk_size, "k": k}

trace = naive_trace("ACME Q2 2023 营收增长")
for field, value in trace.items():
    print(f"{field}: {value}")

# 下一步：为 gold evidence、文档/索引版本、ACL 结果和人工 claim 标注补字段，
# 再比较递归分块、混合召回或重排等候选配置。
```

## 何时用 / 坑
- **Naive 可作受控基线**:用它比较“这批数据是否值得做 RAG”；生产配置需在目标语料、流量、权限要求和风险容忍度上完成评测后决定。
- **先量、再提假设、后验证**:指标用于缩小排查范围，不直接给根因定案。把 [[18 RAG 评估|评估]] 分数、完整 trace 和标注样本放在一起，再 A/B 测候选改动。
- **失败常在衔接处**:FP3/FP4 提醒“检到了”不等于“用上了”；分块、排序、上下文长度与提示都可能相关，必须验证。
- **升级路径**:线性补丁见 Advanced;要循环/自适应/路由见 [[13 Modular RAG|Modular RAG]];把检索决策交给模型自主见 [[36 Agentic RAG|Agentic RAG]]。可观测性串联到 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

## 工业界实践

简单配置可以是可复现的基线和探针：先固定语料、索引、模型、提示和 k 值，再用评估和 trace 比较候选改动。它并不预设所有团队都应从同一套 Naive 配置起步。

**七个失败点 → 候选假设与验证**(落地排障速查；不是一一因果映射):

| 观察到的 FP | 候选假设 | 先验证什么 | 可能的下一次实验 |
|---|---|---|---|
| FP1 内容缺失 | 文档未 ingestion、版本不对或 ACL 拒绝 | gold 文档是否存在、可访问且已进入当前索引 | 修数据源/ACL 后重跑同一 gold 集 |
| FP2 漏掉应排前文档 | 召回器、查询表述或 k 值不足 | 该文档在各阶段的 rank 与 recall@k | 对照词法/混合/重排候选配置 |
| FP3 不在上下文 | 拼装、预算或过滤丢了证据 | retrieved 列表、截断日志与最终 prompt | 调整预算/排序/压缩并成对比较 |
| FP4 未被抽取 | 证据冲突、提示或生成行为 | claim—evidence 标注及引用位置 | 对照引用约束、去噪或生成配置 |
| FP5 格式错误 | 约束不足或解析器不匹配 | 输出 schema、原始响应与失败样本 | 对照结构化输出和重试策略 |
| FP6 粒度不当 | 任务意图、问题改写或答案策略不匹配 | 标注的目标粒度和 query trace | 对照查询改写/多粒度证据 |
| FP7 答案不完整 | 证据覆盖不足或生成遗漏 claim | 缺失 claim、召回与上下文覆盖 | 对照补充检索/多跳策略 |

**排障流程**:
1. 固定评测集与版本：记录语料/索引、embedding、生成模型、提示、k、Ragas 版本和评测模型。
2. 计算 context recall / precision / faithfulness 等指标，把异常视为**诊断假设**，不是 FP 或根因的自动标签。
3. 抽取异常样本的 trace：查询改写、候选与排名、最终上下文、token 截断、文档版本/ACL、回答和引用。
4. 用 gold evidence、人工相关性判断或 claim—evidence 标注验证假设；例如 recall 低可能是库缺失、ACL 或召回问题，faithfulness 低也可能是上下文拼装与提示共同作用。
5. 一次只改一个候选变量，离线成对比较并观察线上抽样；有收益再保留，并回归第 1 步。

```python
# ❌ 朴素：从单个指标直接下根因结论；这个函数故意展示不该采用的做法。
def direct_blame(metrics):
    return "FP2：换 embedding" if metrics["context_recall"] < 0.8 else "生成器有问题"

# ✅ 高效：指标只触发假设；同时检查 trace 与 gold evidence，再决定下一步实验。
def diagnostic_hypotheses(metrics, trace, gold_evidence):
    ideas = []
    if metrics["context_recall"] < 0.8:
        missing = gold_evidence - set(trace["retrieved_ids"])
        if missing:
            ideas.append(f"检查 {sorted(missing)} 是否已 ingestion、ACL 可访问、且进入 top-k")
    if metrics["context_precision"] < 0.8:
        ideas.append("检查 trace 中候选相关性、排序、去重与最终上下文是否一致")
    if metrics["faithfulness"] < 0.8:
        ideas.append("对 trace 的回答逐 claim 标注，检查是否被最终上下文支持")
    return ideas or ["指标未触发该阈值；仍需抽样审计错误与漂移"]

report = {"context_recall": 0.62, "context_precision": 0.71, "faithfulness": 0.66}
trace = {"retrieved_ids": ["doc-error-code"], "prompt": "...", "answer": "..."}
gold_evidence = {"doc-acme-q2"}
print("❌", direct_blame(report))
for idea in diagnostic_hypotheses(report, trace, gold_evidence):
    print("待验证：", idea)
```

**踩坑速记**:
- **别一步到位上全套模块**:重排、多跳、Self-RAG 通常会增加调用、延迟或成本；没有评测证据就叠加，收益无法确认。
- **FP3/FP4 是常被忽略的假设**:「检到了」≠「用上了」。k 增大可能让关键片段处于不利位置或触发截断；用最终 prompt 与答案标注验证。
- **分块是候选变量，不是预设根因**:固定长度可能切断语义，也可能恰好满足任务；[[03 分块策略 Chunking|递归字符分块]]是可比较的候选基线。用离线 gold 集和 trace 验证后再采用。

## 面试高频

**Q1:Naive RAG 会有哪些失败模式?分别怎么解?**
标准答:按检索、拼装和生成列出**候选假设**：召回不足可比较混合检索，分块边界可比较 [[03 分块策略 Chunking|分块策略]]，query-doc 表述差异可测试 [[07 查询变换 Query Transformation|查询变换]]；生成侧可评估 [[11 生成层：引用归因与忠实度|引用归因]]、[[10 重排序 Reranking|重排]]和迭代检索。Barnett FP1–FP7 是三领域案例的诊断分类；先看 trace 和标注再选解法才是高分回答。

**Q2:RAG 的 Naive / Advanced / Modular 三范式区别?**
标准答:Naive = 索引→检索→生成一条直线,每环最朴素;Advanced = 检索前(查询优化)后(重排、压缩)打补丁,仍线性;Modular = 解耦成可重排/可循环/可路由的模块,支持迭代与自适应。出处 Gao et al. 2023 综述。

**Q3(陷阱):「检到了相关文档但答错了」是哪类失败?换更强的 embedding 能解决吗?**
标准答:FP3/FP4 是有用的**候选标签**，但先核对最终 prompt 和 claim—evidence 标注：相关文档可能被截断、没被采用、也可能本就不支持答案。更强 embedding 只有在它改善目标 gold 集召回时才是合理实验；还应成对测试排序、压缩与引用约束。

**Q4:为什么说 RAG 的难点常常不在检索精度?**
标准答:Barnett 的三个案例显示，问题可以出现在内容覆盖、检索、上下文和生成多个环节。它不证明任何系统的主因都相同；用 trace 和标注拆开证据覆盖、排序、上下文长度、提示与生成行为，才能知道检索精度是否真是瓶颈。

**Q5:给你一个翻车的 RAG,排障第一步做什么?**
标准答:**先固定版本并收集可复现证据**。用 Ragas 等指标生成诊断假设，再查看 trace、gold evidence/人工标注和 ACL/索引状态；指标不是因果判定。确认假设后，一次只改变一个变量并做成对评测。

## 知识拓展

- **七个失败点的来源**:Barnett et al. 2024(arXiv:2401.05856)从研究、教育、生物医药三个案例总结 FP1–FP7；它是经验报告中的案例驱动分类，不是穷尽或权威诊断标准(论文标题：*Seven Failure Points When Engineering a Retrieval Augmented Generation System*)。
- **失败模式与解法的全景对照**:本篇是一张“该读哪篇”的导航图。每一项解法都是待评测的候选干预；可继续到 [[13 Modular RAG|Modular RAG]](循环/路由)和 [[36 Agentic RAG|Agentic RAG]](自主决策)，可观测性接 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- **反模式清单**:① 不评估就堆模块；② 让任何单一指标或 FP 标签替代 trace/标注；③ 把 k 设很大却不测截断、延迟和答案质量；④ 未在目标语料验证就把固定或递归分块当默认答案。
- **前沿:让系统自己处理失败**。[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG(Asai 2023)/ CRAG(2024)]]可让模型自评证据质量并触发候选的纠错检索或重写；[[09 多跳检索：IRCoT、Self-Ask、FLARE|FLARE]]在生成中按需动态检索，是否改善 FP7 仍须在目标任务验证。
- **检索为什么会漏(数学层面)**:精确实体、罕见专名或错误码可能是纯稠密检索的薄弱情形之一，但取决于模型、语料和 query。对这类切片单独报告 recall@k，并与 [[08 混合检索 Hybrid Search|混合检索]]做离线对照，背景见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]]。

## 关键事实
- Gao et al. 的综述 arXiv v1 发布于 **2023-12-18**，以 Naive / Advanced / Modular 组织 RAG 方法；这是一种综述分类。
- Barnett et al. 的经验报告 arXiv v1 发布于 **2024-01-11**，从三个案例归纳 FP1–FP7；应把它作为案例驱动分类，而非穷尽或权威诊断标准。
- Ragas arXiv v1 发布于 **2023-09-26**，提出 reference-free 的多维评估框架；指标结果是诊断信号，需和 trace、gold evidence 或人工标注结合验证。

## 来源
- Gao, Y., Xiong, Y., Gao, X., et al. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997. <https://arxiv.org/abs/2312.10997> — Naive / Advanced / Modular 分类与发布日期。
- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). **Seven Failure Points When Engineering a Retrieval Augmented Generation System**. arXiv:2401.05856. <https://arxiv.org/abs/2401.05856> — 三领域案例与 FP1–FP7。
- Es, S., et al. (2023). **Ragas: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217. <https://arxiv.org/abs/2309.15217> — reference-free 多维评估框架，不提供单指标到根因的因果映射。
