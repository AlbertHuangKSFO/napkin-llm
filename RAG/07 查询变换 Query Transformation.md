[[07 查询变换 Query Transformation|查询变换 Query Transformation]] 的本质是:在检索**之前**(pre-retrieval),把用户原始 query 改写成一个或多个**更利于检索器命中**的形式——消歧、补全、抽象、拆解、或干脆换成"假设答案"——再拿改写后的查询去召回。一句话:**问题没问对,后面再强的检索器也救不回来**,所以先把问题改好。

它是 [[01 什么是 RAG|什么是 RAG]] 管线中**低侵入、值得优先评测的候选**:不动索引、不换模型,只在 query 进检索器前加一步改写。它是否比别的优化更省钱、更有效,要和原 query 基线一起做离线集与线上 A/B 评测,不能预设「最低成本、最高收益」。它也是 [[36 Agentic RAG|Agentic RAG]] 里"query 重写"那一步的**非 agent 化拆解版**——这里每种技法是写死的流程,Agentic RAG 是让模型自主决定用哪种、用几次。

## 本质:用户怎么问 ≠ 语料怎么写

检索器(尤其稠密向量)在 query 和 doc 的**表述对齐**时才好使。但用户的原始 query 往往:

- **含糊 / 带指代**:"那个事件后来怎么样了"——"那个"是什么?直接 embed 这句话,向量落在语义空洞处,召回一团糟。
- **太具体 / 钻进细节**:"2023 年 Q3 苹果在印度的 iPhone 出货量同比"——细节多到没有哪篇文档逐字命中,但"苹果印度市场表现"这个上位问题有大量相关文档。
- **词法不匹配**:用户说"心脏病发作",语料写"myocardial infarction / 心肌梗死"。稠密 embedding 有时能跨过去,有时跨不过(见 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 对罕见词/专名的讨论)。
- **复合多跳**:"特斯拉 CEO 创办的另一家航天公司总部在哪"——一次检索拿不全,得拆(见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]])。

Query Transformation 的统一思路:**别拿原话硬查**,先用 LLM 把它变成检索器爱吃的形式。

![[查询变换 Query Transformation.png]]

## 机制:六种技法逐一拆

### 1. HyDE(Hypothetical Document Embeddings)
反直觉但最巧:**不 embed 问题,embed 一篇假设答案**。先让 LLM 对 query"零样本生成一篇假设性文档/答案"(可能含虚构细节、甚至有错),再 embed 这篇假设文档去检索。

为什么有效:query 和 doc 处在不同的"语言空间"——问题短、口语;文档长、陈述句。直接算 query↔doc 相似度是跨空间比对。而假设文档**和真实文档同属答案空间**,doc↔doc 比对更准。生成的虚构细节会被 encoder 的稠密瓶颈"过滤"掉,留下相关性模式。出处:Gao, Ma, Lin, Callan《Precise Zero-Shot Dense Retrieval without Relevance Labels》(arXiv:2212.10496, 2022;ACL 2023),实验显示在零样本设定下显著超过无监督稠密检索器 Contriever,跨多语言多任务有效。坑:对**需要事实精确**的小众查询,假设文档若编得离谱反而把检索带偏;且多一次 LLM 生成,增延迟。

![[rag-HyDE跨空间.png]]

### 2. Multi-Query(多查询并集召回)
让 LLM 把一个 query 改写成 **N 个语义等价但措辞不同**的查询(换同义词、换角度、换粒度),每个都去检索,**取召回结果的并集**去重。原理:单一措辞会系统性漏掉用某种说法索引的文档;多措辞覆盖面更广,降低"一种问法的盲区"。代价:N 倍检索调用 + 一次改写。

### 3. RAG-Fusion(多 query + RRF 融合)
Multi-Query 的升级:不止取并集,而是对每个子 query 的排名列表做 **RRF(Reciprocal Rank Fusion,见 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]])** 融合重排——在多个子查询里都排得靠前的文档,综合得分最高。出处:Adrian Raudaschl 在 Towards Data Science 提出《Forget RAG, the Future is RAG-Fusion》(2023-10,博客 + GitHub `Raudaschl/rag-fusion`);后有 Rackauckas《RAG-Fusion: a New Take on Retrieval-Augmented Generation》(arXiv:2402.03367, 2024)做系统评估。注意它**不是同行评审顶会论文起源**,是工程社区方法 + 后续 arXiv 报告,引用时别误标成会议论文。

**RRF 手算(2 query)**。这里取 $k=60$ 只是便于复现的**评测起点**,不是协议默认值;应随语料、候选深度和质量/延迟目标调参。两条改写各召回一个排名列表,RRF 给每篇文档累加 $\frac{1}{k+\text{rank}}$(rank 从 1 起)。设 doc A 在 q1 排第 1、q2 排第 3,doc B 在 q1 排第 2、q2 排第 1:
$$\text{score}(A)=\tfrac{1}{60+1}+\tfrac{1}{60+3}=0.01639+0.01587=0.03227$$
$$\text{score}(B)=\tfrac{1}{60+2}+\tfrac{1}{60+1}=0.01613+0.01639=0.03252$$
B 略高于 A,最终排 B 在前。直觉:B 虽没拿到任何一个第 1 名,但**两条 query 里都靠前**(2、1),综合稳过「只在一条里夺冠、另一条掉到第 3」的 A——这正是 RRF「跨列表一致靠前者胜出」的精髓,且只吃名次不吃原始分。

### 4. Step-Back Prompting(退一步问)
对**钻进细节**的具体问题,先让 LLM"退一步"问一个更**抽象的上位问题**,用上位问题去检索拿到高层原理/背景,再连同原始问题一起喂生成。例:原问"理想气体在压强翻倍温度翻倍时体积如何变",退一步问"这道题涉及哪条物理定律"→检索到理想气体定律→再代入算。出处:Zheng, Mishra, Chen, Cheng, Chi, Le, Zhou《Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models》(arXiv:2310.06117, 2023),在 PaLM-2L 上 TimeQA +27%、MuSiQue +7%、MMLU 物理/化学 +7%/+11%。

### 5. Query Decomposition(查询分解)
把一个复合问题**拆成若干可独立检索的子问题**,各自检索、各自(或联合)回答,最后综合。"特斯拉 CEO 创办的航天公司总部在哪"→ ①特斯拉 CEO 是谁 ②他创办的航天公司是哪家 ③该公司总部在哪。和 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 的区别:**分解是一次性把子问题列全(可并行)**;多跳是**前一跳结果决定后一跳查什么(串行依赖)**。两者常混用,但依赖结构不同。

### 6. Query-aware Routing(查询感知路由)
**不改 query 内容,而是为这一次请求产出可审计的检索计划**。它不是「事实必走向量、结构化必走 SQL」的硬编码口诀:同一问题可能同时需要产品文档、表格数据与代码证据。router 先打分,再在权限与可用源的边界内选路;它和前五种变换正交,例如可以先选 hybrid,再在该路内做 Multi-Query。

router 的输入至少包括:

- **复杂度**:实体数、约束数、是否需要多跳;高复杂度可触发 Decompose,而非盲目多查。
- **时效性与权威性要求**:问题是否要求「最新」、法规/公告等可核验的一手来源,以及允许的来源白名单。
- **结构化程度**:是否含聚合、分组、范围过滤、主键/指标等可由受控查询计划表达的信号。
- **ACL 与数据源可用性**:先由确定性权限层过滤用户可见、当前健康的源;LLM 不能凭文本扩大权限。
- **首轮置信度**:top-$k$ 分数间隔、跨 query 重叠、证据覆盖度与引用可用性。阈值要用带标注集校准,不是照搬一个常数。

| 条件(均已通过 ACL/可用性过滤) | 首选输出 | fallback / abstain |
|---|---|---|
| 纯改写、算术或答案已在用户提供的上下文中 | `no-retrieval` | 若发现外部事实依赖,重新路由而不是猜测 |
| 语义问答、内部文档为主,且首轮证据充分 | `vector` | 置信度不足时尝试 `hybrid`,记录原因 |
| 既要语义召回又有专名/精确字串 | `hybrid` | 候选仍不足则请求澄清或 `abstain` |
| 可被参数化、只读的聚合/过滤 | `SQL` | 无表权限、查询不安全或结果缺失时 `abstain` 或转到获准的文档源 |
| 新近公共信息且组织策略允许联网 | `web` | 对同一获准时效源可有界重试;超时或来源冲突默认 `abstain`。仅当请求明确允许历史快照时,才可降到**日期上界明确**的获准源,并在答案中醒目声明「截至 YYYY-MM-DD,可能已过期」 |
| 已获授权的代码仓库与符号/字串查询 | `code` | 索引不可用或 ACL 不允许时 `abstain`,不改走网页泄露代码 |

每次都返回 `{route, allowed_sources, transform, confidence, fallback, abstain_reason}`。`fallback` 是受策略约束的下一次**受限检索**,不是无限重试;对时效型 `web` 请求,默认 fallback 只能是有界重试或 `abstain`,不能静默换成可能陈旧的内部知识。只有用户/产品策略明确接受历史答案,且输出暴露日期上界与陈旧风险,才允许该历史降级。候选均不足或不能安全访问时明确 `abstain` 并说明缺什么证据。

⚠️ **网页检索的信任边界**:网页正文、页面内的"指令"与链接文本都是**不可信数据**,不是工具授权、权限变更或系统指令。网页只能提供待引用的事实候选;工具白名单、下载/执行权限与 ACL 必须由系统在网页文本之外强制。外部页面可携带间接提示注入,因此应分隔标记并按不可信输入处理(OWASP LLM01, 2025)。

## 可运行代码:❌ 零基名次 vs ✅ 一基名次

下面的断言复现上面的手算。`enumerate()` 默认从 0 开始;若不显式指定 `start=1`,首名会错误得到 $1/60$ 而不是 $1/61$。

```python
from math import isclose

rankings = [["A", "B"], ["B", "C", "A"]]  # A: 第 1、第 3; B: 第 2、第 1

def rrf_zero_based(rankings, k=60):              # ❌ Python 默认 rank=0
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return scores

def rrf_one_based(rankings, k=60):                # ✅ RRF 公式的 rank 从 1 起
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return scores

expected_a = 1 / 61 + 1 / 63
expected_b = 1 / 62 + 1 / 61
assert not isclose(rrf_zero_based(rankings)["A"], expected_a, rel_tol=0, abs_tol=1e-12)
assert isclose(rrf_one_based(rankings)["A"], expected_a, rel_tol=0, abs_tol=1e-12)
assert isclose(rrf_one_based(rankings)["B"], expected_b, rel_tol=0, abs_tol=1e-12)
assert rrf_one_based(rankings)["B"] > rrf_one_based(rankings)["A"]
print("RRF one-based regression check passed")
```

## 代码骨架(接入你的检索器)

```python
# 四种主流变换的最小骨架(伪检索器 retrieve 已给定)
def hyde(query, llm, embed, vstore):
    # HyDE:生成假设文档 → embed 它(而非 query)→ 检索
    hypo_doc = llm(f"写一段话回答这个问题(可以编细节):{query}")
    return vstore.search_by_vector(embed(hypo_doc), k=5)

def multi_query(query, llm, retrieve, n=4):
    # Multi-Query:N 个改写 → 召回并集去重
    rewrites = llm(f"把下面问题改写成 {n} 个措辞不同但意思一样的查询,每行一个:\n{query}").splitlines()
    hits = {}
    for q in [query] + [r.strip() for r in rewrites if r.strip()]:
        for doc in retrieve(q, k=5):
            hits[doc.id] = doc          # 用 id 去重
    return list(hits.values())

def rag_fusion(query, llm, retrieve, n=4, k_rrf=60):
    # RAG-Fusion:多 query 各自排名 → RRF 融合
    queries = [query] + llm(f"改写成 {n} 个查询,每行一个:\n{query}").splitlines()
    scores = {}
    for q in queries:
        for rank, doc in enumerate(retrieve(q.strip(), k=10), start=1):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k_rrf + rank)  # RRF: rank 从 1 起
    return sorted(scores, key=scores.get, reverse=True)[:5]

def step_back(query, llm, retrieve):
    # Step-Back:退一步问上位问题,两路证据都喂生成
    abstract_q = llm(f"为回答下面问题,先问一个更抽象的上位问题:\n{query}")
    return retrieve(abstract_q, k=3) + retrieve(query, k=3)
```

要点:① HyDE 的关键是 `search_by_vector(embed(hypo_doc))` 而非 embed(query);② Multi-Query 用 doc.id 去重避免并集重复;③ RRF 的 `1/(k+rank)` 只吃名次不吃原始分,且 `rank` 必须从 1 起;④ 每种变换都至少多一次 LLM 调用,是否值得用要以召回与端到端答案质量评测为准。

## 对比表

| 技法 | 变换对象 | 解决什么 | 主要代价 | 出处 |
|---|---|---|---|---|
| HyDE | query→假设文档 | query/doc 跨空间不对齐 | 1 次生成 + 可能编歪 | arXiv:2212.10496 |
| Multi-Query | 1→N 个改写 | 单一措辞的盲区 | N 倍检索 | 工程实践 |
| RAG-Fusion | N 改写 + RRF | 同上 + 排名融合 | N 倍检索 + 融合 | TDS 2023 / arXiv:2402.03367 |
| Step-Back | query→上位问题 | 太具体、缺背景原理 | 1 次生成 | arXiv:2310.06117 |
| Decompose | 1→多子问题 | 复合问题一次查不全 | 多次检索 | — |
| Routing | 不改内容,选源 | 问题异质、源不同 | router 一次分类 + 策略校验 | 见 [[05 Routing\|Routing]] |

## 何时用 / 坑

**该用**:用户 query 含糊/带指代(Multi-Query、改写消歧)、向量召回质量差(HyDE)、问题太具体缺背景(Step-Back)、复合多跳(Decompose)、多源异质(Routing)。这些是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 里"检索召回差"的标准解药。

**坑**:
- **延迟与成本翻倍**:每种变换都加 LLM 调用,Multi-Query/RAG-Fusion 还把检索次数乘 N。对延迟敏感场景要算账。
- **改写跑偏**:LLM 把原意改没了,越查越歪。改写 prompt 要强调"保留核心意图",并**保留原 query 兜底**(代码里都把原 query 也纳入)。
- **HyDE 反噬**:对需要事实精确、语料稀疏的查询,假设文档编得离谱会把检索带向错误聚类。HyDE 更适合"答案空间稠密"的通用域。
- **过度工程**:简单清晰的 query 别强加变换,白烧 token 还可能引入噪音。先测 baseline,确认 query 质量是瓶颈再上。
- **和下游协同**:变换可能带来更多候选与噪音;是否接 [[10 重排序 Reranking|重排序 Reranking]]、取多少候选,也应以端到端质量、延迟和成本的评测决定。

## 工业界实践

Query Transformation 在生产里是**低侵入但必须验证的实验变量**:它只在 query 进检索器前加一步改写,不动索引、不换模型;但每种技法都加 LLM 调用、Multi-Query/RAG-Fusion 还把检索次数乘 N。先测原 query 基线,再在固定数据集、固定预算下逐项 A/B,同时看召回、答案忠实度、p95 延迟与每请求成本。

### 框架与现成实现(组件名随版本变化)

下表是实现思路的索引,**不是当前版本兼容性承诺**;接入前应记录所用框架版本、官方文档访问日期与自己的回归评测结果。

| 技法 | 框架组件 |
|---|---|
| HyDE | LlamaIndex `HyDEQueryTransform`;LangChain `HypotheticalDocumentEmbedder` |
| Multi-Query | LangChain `MultiQueryRetriever`(开箱即用,自动改写 + 并集去重) |
| RAG-Fusion | LangChain 用 LCEL 串「多改写 + RRF」;社区仓库 `Raudaschl/rag-fusion` |
| Step-Back | LangChain 官方 cookbook 有 step-back prompting 模板 |
| Decompose | LlamaIndex `SubQuestionQueryEngine`(拆子问题 + 分源回答 + 综合) |
| Routing | LlamaIndex `RouterQueryEngine`;LangChain 用 LLM/embedding 路由 |

### 典型生产架构(把变换嵌进检索栈)

```
用户 query
  → [Query-aware Routing] ACL/可用性过滤 + 复杂度/时效/结构化/首轮置信度 → route plan(no-retrieval/vector/hybrid/SQL/web/code, fallback/abstain)
  → [Transform] 在选定的源上做变换:
        召回质量差 → HyDE(embed 假设文档)
        含糊/单一措辞盲区 → Multi-Query / RAG-Fusion
        太具体缺背景 → Step-Back
        复合问题 → Decompose
  → [混合检索 08] dense + sparse 召回(变换后的 query)
  → [重排序 10] 若评测显示候选噪音需要,用 cross-encoder 收口
  → 生成
```

**可复用组合**:Multi-Query/RAG-Fusion 用的 RRF 和 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 的 RRF 是同一个算法(`Σ 1/(k+rank)`,rank 从 1 起)。可以共用一套融合逻辑,但 $k=60$ 只应作为实验起点,由自己的离线集和线上指标决定是否保留或调整。

### 延迟、成本与缓存

- **HyDE / Step-Back**:+1 次 LLM 生成,延迟增一个生成往返;可先评测较低成本的改写模型,不要假定它一定足够或一定优于主模型。
- **Multi-Query / RAG-Fusion**:+1 次改写 LLM + N 路检索。N 路检索可**并行**(异步并发发起),把墙钟延迟压到接近单路;但向量库 QPS 成本是 N 倍,要评估。
- **缓存**:对高频/热门 query,可尝试缓存改写结果和召回结果(query 文本做 key),再实测命中率、陈旧风险与节省的成本;不要只因「热门」就默认收益显著。
- **保留原 query 兜底**:所有变换都应把**原始 query 也纳入检索池**(代码里 `[query] + rewrites`),防止改写跑偏时彻底丢掉正确召回。

### 评估与可观测

- 用 [[18 RAG 评估|RAG 评估]] 的 Ragas **context recall** 量化每种变换的召回增益,**逐一开关 A/B**:别一次全开,分不清是哪项在起作用,也可能互相打架(如 HyDE 编歪 + Multi-Query 放大噪声)。
- 可观测:记录「每种变换的触发率 / 改写后的 query 文本 / 召回 overlap(改写 vs 原 query 的召回交集)」。改写召回与原 query 召回**几乎完全重叠**说明变换没带来增量、纯烧 token;**完全不重叠**则要警惕改写跑偏。用 LangSmith / Langfuse / Phoenix 追踪改写前后的 query 与召回链路。

### 踩坑与最佳实践

- **过度工程是头号坑**:简单清晰的 query 强加变换,白烧 token 还引噪音。先测 baseline,确认「query 质量是瓶颈」再上。
- **改写 prompt 必须锁意图**:强调「保留核心意图、别引入文档外信息」,否则 LLM 自由发挥把原意改没。
- **HyDE 看语料密度**:答案空间稠密的通用域受益大;事实精确、语料稀疏的小众域,假设文档编离谱反而把检索带偏,慎用或换 Multi-Query。
- **变换后是否 rerank 是待评测假设**:变换可能扩大召回也可能带进杂质;比较「不重排 / 不同候选深度 / cross-encoder 重排」的端到端指标后再定。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1:HyDE 为什么 embed 一篇「假设答案」而不是 embed 问题?编错了不会带偏检索吗?**
标准答:因为 query 和 doc 处在不同语言空间(问题短、口语;文档长、陈述句),直接算 query↔doc 相似度是跨空间比对;而假设文档**和真实文档同属答案空间**,doc↔doc 比对更准。生成的虚构细节会被 encoder 的稠密瓶颈过滤掉,留下相关性模式。编错的风险确实存在——所以 HyDE 更适合「答案空间稠密」的通用域;事实精确、语料稀疏的查询会被带偏。出处 arXiv:2212.10496。
- 追问「和 query expansion 区别?」→ query expansion 加同义词/相关词仍在 query 空间;HyDE 直接跳到答案空间。

**Q2:Multi-Query 和 RAG-Fusion 区别?**
标准答:Multi-Query 是「N 个等价改写各自检索,取召回并集去重」;RAG-Fusion 在此之上**不止取并集,而是对每个子 query 的排名列表做 RRF 融合重排**——在多个子查询里都排靠前的文档综合得分最高。RAG-Fusion = Multi-Query + RRF。
- 陷阱:RAG-Fusion **不是会议论文起源**,是 Raudaschl 2023 工程博客方法 + 后续 arXiv:2402.03367 报告,引用时别误标成顶会论文。

**Q3:Step-Back Prompting 解决什么 query?为什么退一步反而更好?**
标准答:解决「钻进细节、太具体」的 query——具体到没有文档逐字命中,但其上位问题有大量相关文档。退一步问一个更抽象的上位问题,用它检索拿到高层原理/背景,再连同原始问题一起喂生成。出处 arXiv:2310.06117,PaLM-2L 上 TimeQA +27%。

**Q4(关键区分):Query Decomposition 和多跳检索(IRCoT/Self-Ask)有什么不同?**
标准答:**分解是一次性把子问题列全、可并行**;多跳是**前一跳结果决定后一跳查什么、串行依赖**。「特斯拉 CEO 创办的航天公司总部在哪」如果能一次拆成 3 个独立子问题就是 decompose;如果必须先查出 CEO 是谁才能查下一步就是多跳。两者常混用但依赖结构不同,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]]。

**Q5:这些变换和 Agentic RAG 是什么关系?**
标准答:Query Transformation 是 [[36 Agentic RAG|Agentic RAG]] 里「query 重写」步骤的**写死版**——每种技法是固定流程;Agentic RAG 让模型**自主决定用哪种、用几次、是否还要再查**。前者是确定性管线,后者是 agent 自主决策。

## 知识拓展

- **HyDE 的检索学渊源**:它是经典 **pseudo-relevance feedback / query expansion** 思想的 LLM 化——传统 PRF 用初次检索的 top 文档扩展 query,HyDE 干脆让 LLM 直接「幻想」出一篇伪相关文档。两者都在缩小 query-doc 表述鸿沟,只是 HyDE 用生成模型一步到位。
- **RRF 是贯穿检索栈的通用融合算子**:RAG-Fusion 用它融合「多个改写 query 的召回」,[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 用它融合「稀疏+稠密两路」。它只吃名次不吃原始分,适合做异质排名列表的候选融合(出处 Cormack et al. SIGIR 2009);$k=60$ 可作起点,但不是无需调参的默认真理。理解 RRF 一次,两处都通。
- **变换 vs 重排的分工**:Query Transformation 是 **pre-retrieval**(检索前改 query),[[10 重排序 Reranking|重排序 Reranking]] 是 **post-retrieval**(检索后精排候选)。一个改输入、一个筛输出,生产里常前后搭配成完整检索栈,别把两者混为一谈。
- **前沿与边界**:近年趋势是把这些写死的变换交给 agent 自主编排(Agentic RAG / [[12 Self-RAG、CRAG 与 Adaptive RAG|Adaptive RAG]]),由模型按 query 难度动态决定「要不要变换、用哪种、查几轮」,避免对简单 query 过度工程。反模式:对所有 query 无脑全开变换(延迟成本翻倍 + 引噪声);正确做法是先测 baseline、确认 query 质量是瓶颈、单项 A/B 验证后再上。

## 关键事实

- Query Transformation = **pre-retrieval 优化**,在检索前改写 query,不动索引/模型;是否有性价比必须相对原 query 基线评测。
- 六种技法:HyDE(embed 假设文档)、Multi-Query(多改写并集)、RAG-Fusion(多改写 + RRF)、Step-Back(退一步问上位)、Decompose(拆子问题)、Routing(选源,链 [[05 Routing|Routing]])。
- HyDE 出处 arXiv:2212.10496(Gao et al. 2022/ACL 2023);Step-Back 出处 arXiv:2310.06117(Zheng et al. 2023);RAG-Fusion 是 Raudaschl 2023 工程方法 + arXiv:2402.03367 后续报告(非会议起源,别误标)。
- Decompose vs 多跳:分解是**一次列全子问题(可并行)**,多跳是**前跳决定后跳(串行依赖)**,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]]。
- 是 [[36 Agentic RAG|Agentic RAG]] 里"query 重写"步骤的写死版;Agentic RAG 让模型自主选技法、自主决定查几次。
- Query-aware Routing 先看复杂度、时效/权威性、结构化程度、ACL/源可用性、首轮置信度,再输出 no-retrieval/vector/hybrid/SQL/web/code 与受策略约束的 fallback/abstain。网页内容是不可信数据,不能授权工具或扩权(OWASP LLM01, 2025)。
- 常可与 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]](RRF 同款融合)、[[10 重排序 Reranking|重排序 Reranking]](候选收口)组合;是否启用、$k$ 与 rerank 深度均须评测。

## 来源
- Cormack, Clarke, Büttcher.《Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods》. SIGIR, 2009.
- Gao, Ma, Lin, Callan.《Precise Zero-Shot Dense Retrieval without Relevance Labels》(HyDE). arXiv:2212.10496, 2022;ACL 2023.
- Zheng, Mishra, Chen, Cheng, Chi, Le, Zhou.《Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models》. arXiv:2310.06117, 2023.
- Raudaschl, A. H.《Forget RAG, the Future is RAG-Fusion》. Towards Data Science, 2023-10;GitHub `Raudaschl/rag-fusion`. 后续:Rackauckas.《RAG-Fusion: a New Take on Retrieval-Augmented Generation》. arXiv:2402.03367, 2024.
- OWASP.《LLM01:2025 Prompt Injection》, 2025. https://genai.owasp.org/llmrisk/llm01-prompt-injection/
