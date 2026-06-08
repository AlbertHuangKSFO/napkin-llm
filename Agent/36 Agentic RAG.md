[[36 Agentic RAG|Agentic RAG]] 的本质是:把「检索」从一条写死的管线步骤,变成 agent 手里的一个**工具**——由模型自主决定**是否检索、检索什么、检索几轮**,并能对检索结果**反思后再查**,直到信息足够才生成答案。

它是 [[15 Function Calling 工具调用|Function Calling 工具调用]] 与 [[RAG]] 的合流:传统 RAG 是固定流程,Agentic RAG 是把 retrieve 包成工具塞进 [[03 Agent 核心循环|Agent 核心循环]],于是检索变成了 agent 可以反复、自适应使用的能力。也可以看作 agent 对外部知识的「读记忆」操作([[19 Agent 记忆系统|Agent 记忆系统]] 的检索面)。

## 本质:从「写死的管线」到「自主的工具」

先看传统 [[RAG]]([[RAG]] 是悬空链,指 Retrieval-Augmented Generation):它是一条**固定的线性管线**——

> query → 检索一次(取 top-k)→ 把结果拼进上下文 → 生成答案

这条线**单向、固定、一次性**。它的死穴全在这「固定」二字:

- **检索不好也只能将就**。第一次检索召回的文档不相关、不充分,管线照样把它拼进去生成——garbage in, garbage out。模型没有「再查一次」的机会。
- **不会重写问题**。用户问得含糊(「那个事件后来怎么样了」),直接拿原话去向量检索,召回一团糟。传统 RAG 不会先把 query 改写清楚。
- **搞不定多跳**。「A 公司 CEO 的母校在哪个城市」需要先查 CEO 是谁、再查母校、再查城市——三跳。一次检索拿不全,传统 RAG 抓瞎。
- **不会判断「要不要查」**。哪怕模型自己就知道答案(简单常识),管线也强制检索一次,浪费且可能引入噪音。
- **单一知识源**。只会查它绑定的那个向量库,不会按问题路由到 SQL、API、网页等不同源。

Agentic RAG 的洞见:**让模型自己掌控检索**。把 retrieve 做成一个工具,模型在 [[09 ReAct|ReAct]] 式循环里自主调用——该查就查、不该查就直接答、查得不好就改写 query 再查、需要多跳就连查几轮、不同问题路由到不同源。检索从「流程的一环」升级成「agent 的一项能力」。

![[Agentic RAG.png]]

## 机制:检索成为工具后多出来的几种能力

把 retrieve 当工具后,agent 获得传统 RAG 没有的几种自适应行为:

### 1. 自主决定是否检索
模型先判断「这题我需要外部知识吗?」。简单常识、已在上下文里的信息,直接生成,**不浪费一次检索**(省钱省延迟,也避免无关文档污染上下文)。只有判断「我不知道 / 需要最新事实」时才调 retrieve。

### 2. Query 重写(query rewriting)
不直接拿用户原话去检索,而是先**改写成更适合检索的查询**:消解指代(「它」→具体实体)、补全上下文、拆成多个子查询、或转成关键词。改写后召回质量大幅提升。这一步本身就是一次模型调用,是 agentic 比传统 RAG「多花的钱」之一,但通常很值。

### 3. 多跳检索(multi-hop)
一个问题拆成连续几跳,**前一跳的检索结果决定后一跳查什么**。查到「CEO 是张三」→ 再查「张三的母校」→ 再查「该校所在城市」。这正是 [[09 ReAct|ReAct]] 的 Thought→Action→Observation 在检索场景的体现:每跳的 Observation(检索结果)喂给下一跳的推理。

### 4. 检索后自评 + 再查(self-reflection / corrective)
拿到检索结果后,模型**评估它够不够好**(相关吗?充分吗?能回答问题吗?)。不够好就**回到 query 重写、换个查法或换个源再查一轮**,而不是将就着生成。这是 Agentic RAG 区别于传统 RAG 最核心的一环——**带反馈回路**(对应 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的思想)。学界的 **Self-RAG**、**Corrective RAG (CRAG)** 就是把这个自评机制做成专门方法:Self-RAG 让模型生成特殊的「反思 token」来决定是否检索、检索结果是否相关、生成是否有据;CRAG 用一个轻量评估器给检索结果打分,差就触发「网页搜索补救」。

### 5. 路由到不同知识源
按问题类型把检索**路由**(见 [[05 Routing|Routing]])到合适的源:事实问题查向量库、结构化数据查 SQL、实时信息查网页搜索、代码库用 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 的 grep。一个 agent 可以挂多个检索工具,自己选用哪个。

### 完整流程

![[Agentic RAG-流程.png]]

串起来就是一个自适应循环:**query 重写/路由 → 判断是否需检索 → retrieve → 评估结果 → 不够好就回去重写再查 → 够好才生成**。红色回边(再查)是它的灵魂;和所有 agent 循环一样,要设**最大轮数**防止反复查不收敛。

## 可跑的最小实现

把 retrieve 包成工具,套进一个带自评的 agent 循环:

```python
def retrieve(query, source="default"):
    """检索工具:可指定知识源。返回 top-k 文档片段。"""
    return vector_db[source].search(query, k=5)   # 生产可挂多源:vector/sql/web

def agentic_rag(question, llm, max_hops=4):
    context = ""
    for hop in range(max_hops):
        # 1) 模型自主决策:够了就答,不够就给出「下一步检索的 query」
        decision = llm(
            "你在做多跳检索问答。基于已有证据判断:"
            "若信息已足够,输出 ANSWER:<最终答案>;"
            "否则输出 SEARCH:<改写后的检索query>|<知识源>。\n"
            f"问题:{question}\n已检索到的证据:\n{context or '(暂无)'}"
        )
        if decision.startswith("ANSWER:"):
            return decision[7:]                    # 信息够,生成答案,退出
        _, body = decision.split("SEARCH:", 1)
        query, source = (body.split("|") + ["default"])[:2]
        docs = retrieve(query.strip(), source.strip())

        # 2) 自评:这次检索够不够好?不好则下一轮模型会换查法(隐式 corrective)
        good = llm(f"这些结果能推进回答「{question}」吗?只答 yes/no:\n{docs}")
        if good.strip().lower().startswith("yes"):
            context += f"\n[hop{hop}] {query.strip()} → {docs}"
        else:
            context += f"\n[hop{hop}] {query.strip()} 召回不佳,需换查法"
    return llm(f"基于以下证据尽力回答:\n{context}\n问题:{question}")
```

要点:① retrieve 是**工具**,模型自己决定 query 和源,而非写死;② 每轮模型判断「够了没」,够了就 ANSWER 退出(自主决定检索几轮);③ 显式 `good` 自评 + 下一轮换查法,就是 corrective 行为;④ `max_hops` 是必备护栏。把它换成结构化 [[15 Function Calling 工具调用|Function Calling 工具调用]] 即为生产形态。

## 对比:传统 RAG vs Agentic RAG

| 维度 | 传统 RAG | Agentic RAG |
|---|---|---|
| 流程 | 固定线性管线 | 自适应多轮环 |
| 是否检索 | 永远检索一次 | 模型自主决定(可不查) |
| query | 用原话 | 先重写/分解 |
| 检索轮数 | 恰好 1 次 | 0 到多次,自适应 |
| 多跳 | 做不到 | 原生支持 |
| 检索不好 | 将就生成 | 自评后再查/换源 |
| 知识源 | 单一绑定 | 多源路由 |
| 成本/延迟 | 低、可预测 | 高、不定(多轮模型调用) |
| 适合 | 简单事实问答 | 复杂、多跳、需高准确率 |

核心差别一句话:**传统 RAG 是「检索固定为流程一步」,Agentic RAG 是「检索升级为 agent 的工具」**——后者多了「判断、重写、多跳、自评、路由」,代价是更贵更慢。

**多跳成本手算(LLM 调用账)**。传统 RAG 回答一题:检索 1 次 + 生成 1 次 = **1 次 LLM 调用**(检索本身走向量库,不算 LLM)。同一题走 3 跳 Agentic RAG,每跳要「① 模型决策下一步查什么 + ② 检索后自评够不够」,3 跳就是 $3\times2=6$ 次 LLM 调用,再加最终 1 次生成 = **7 次**;若每跳还做一次 query 重写,则 $3\times3+1=10$ 次。即同一道多跳题,LLM 调用数从 1 涨到 **6~10 次**,$\approx 6\!\sim\!10\times$。再算 token:多跳把每轮检索结果(设各 800 token)累积进上下文,第 3 跳的 prompt 已含约 $800\times2=1600$ token 的历史证据,输入 token 也随跳数线性堆高——这就是「多跳又贵又慢、延迟不定」的具体来源,也是为什么简单单跳题别上 agent。

## 何时用 / 坑

**该上 Agentic RAG**:多跳问题、需要高准确率(检索差就再查)、问题类型多样需路由多源、用户问得含糊需重写、或检索质量参差不齐需要纠错回路。

**不该上**:简单单跳事实问答(传统 RAG 又快又便宜,套 agent 是杀鸡用牛刀)、对延迟极敏感的场景(多轮检索拖慢响应)、或检索源单一且质量稳定。

**坑**:
- **不收敛 / 死循环**:自评一直说「不够好」,反复重查同样的东西。必须设 `max_hops`,并允许「实在查不到就基于现有证据尽力答」。
- **成本/延迟失控**:每跳都是「检索 + 一到两次模型调用」,多跳累积起来又贵又慢。控制最大跳数,简单问题走快路径(直接答)。
- **query 重写跑偏**:改写把原意改没了,越查越歪。重写要保留核心意图,可保留原 query 兜底。
- **自评不可靠**:让模型评自己的检索质量,它可能过度自信(觉得够了其实不够)或过度保守(无限再查)。自评 prompt 要给明确标准。
- **多源路由错配**:把该查 SQL 的问题路由到向量库,召回全是噪音。路由本身([[05 Routing|Routing]])要可靠。
- **上下文累积膨胀**:多跳把每轮检索结果都堆进上下文,token 暴涨且 lost-in-the-middle。需配合 [[20 上下文工程|上下文工程]]:旧 hop 的结果压缩或只留摘要。

## 关键事实

- Agentic RAG = **把检索做成 agent 的工具**,模型自主决定是否检索、检索什么、检索几轮,并能自评后再查。
- 对比传统 [[RAG]]:后者是**固定线性管线**(一次检索→拼接→生成),前者是**自适应多轮环**。
- 多出来的五种能力:**自主决定是否检索、query 重写、多跳检索、检索后自评+再查、多源路由**。
- 自评+纠错回路是灵魂,对应 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];学界做法有 **Self-RAG**(反思 token)、**Corrective RAG / CRAG**(评估器 + 网页补救)。
- 实现本质:一个把 `retrieve` 当工具的 [[09 ReAct|ReAct]] 循环,用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 落地,必设最大跳数护栏。
- 代价:比传统 RAG **更贵更慢、延迟不定**(多轮模型调用);简单单跳问答别上,复杂/多跳/要高准确率才值。
- 与 [[19 Agent 记忆系统|Agent 记忆系统]] 同源:检索本就是 agent 的「读记忆」;与 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 互补,检索源可以是向量库也可以是 grep/SQL/网页。

## 主流开源实现 / Python 库

- **`run-llama/llama_index`**(pip `llama-index`):数据接入/索引/查询引擎最强,内建 agentic retrieval、多源 query engine——**做检索层(retrieval)的首选**。
- **`langchain-ai/langgraph`**(pip `langgraph`):把「query 重写→检索→自评→再查」的自适应环显式建成图,官方有 Self-RAG / CRAG / Adaptive-RAG 教程——**做编排(orchestration)首选**。2026 生产常见组合即 LlamaIndex 取检索 + LangGraph 管编排。
- **Self-RAG / CRAG**:学界方法,多以 LangGraph 教程或作者参考实现落地(反思 token / 评估器 + 网页补救),非单一权威库。
- **`deepset-ai/haystack`**(pip `haystack-ai`):管线式、强类型,金融/医疗/法律等「答错代价高」的生产场景见长。
- 当下首选:检索 `llama_index`、编排 `langgraph`;要稳健生产管线用 `haystack`。

## 工业界实践

**真实落地的不是「Self-RAG 论文复现」,而是一条务实的 Adaptive 路由管线。** 生产里很少有人把 Self-RAG 的反思 token 原样上线(要么微调专用模型、要么 prompt 模拟,稳定性差);更常见的是把 **CRAG 的「评估器打分→不行就网搜补救」** 和 **Adaptive-RAG 的「按问题复杂度分流」** 揉成一套。2026 业界共识把 **Adaptive-RAG 视为生产路由的「emerging best practice」**:用一个轻量分类器(甚至一次小模型调用)把 query 分成「不检索 / 单跳 / 多跳」三档,简单问题走快路径直接答,复杂问题才开多轮——这正是本篇「自主决定是否检索」的工程化身。

**主流技术栈分工(2026):**
- **检索层**:`llama-index`(多源 query engine、节点后处理、rerank 内建)是取数首选;向量库用 pgvector / Qdrant / Weaviate;rerank 几乎是标配(Cohere Rerank、BGE-reranker),它对「检索不好」的缓解比加 hop 更便宜。
- **编排层**:`langgraph` 把「重写→检索→评分→再查」显式建成有状态图,官方博客有 **Self-Reflective RAG / CRAG / Adaptive-RAG** 整套教程(`blog.langchain.com/agentic-rag-with-langgraph`),失败节点可重试、可断点续跑。典型生产组合就是 **LlamaIndex 取检索 + LangGraph 管编排**。
- **评估层**:`ragas` 打 faithfulness / context-precision / context-recall,卡进 CI;trace 用 `langfuse` / `arize-phoenix` 看每个 retriever span 的召回质量(见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]])。

**典型架构(自适应 RAG 服务):**
```
                    ┌─ 复杂度分类器(Adaptive)
query ──► 改写/消歧 ─┤  simple → 直接答(0 跳)
                    │  moderate → 单跳检索 + rerank → 生成
                    └─ complex → LangGraph 多跳环:
                         retrieve → CRAG 评分器
                           ├ 高分 → 生成
                           ├ 中分 → 改写 query 再查
                           └ 低分 → 网页搜索补救(web fallback)
                         (max_hops 护栏 + 旧 hop 压缩)
```

**规模化与成本/延迟。** 真实痛点是「自适应」带来的**长尾延迟**:simple 问题 200ms、complex 问题可能 8~15s。生产做法:① 复杂度分类前置,80% 流量走快路径,只为 20% 难题付多跳的钱;② 每跳检索 + 评分都缓存(同 query 重写命中复用);③ 多跳的中间 LLM 调用用便宜小模型(评分、判断「够不够」不需要旗舰模型),只有最终生成上大模型——这是 [[35 Agent 成本与延迟优化|成本优化]] 的模型分级在 RAG 的应用;④ rerank 优先于加 hop:一次 rerank 比多一跳便宜得多,常常召回质量上去了就不用再查。

**可观测运维。** 每个 retriever span 记 `query / 命中 doc_id / 相似度分 / rerank 后排序 / 是否触发 web fallback`。线上盯三条曲线:平均 hop 数(飙升=分类器失准或评分器太苛刻)、web fallback 率(飙升=知识库覆盖不足,该补数据)、context-precision(掉=召回在变脏)。

**踩坑与最佳实践:**
- **「再查」别只换 query 不换源**:CRAG 的精髓是低分时切到 **web search 补救**,光在同一个向量库里换措辞往往还是查不到。
- **rerank 是性价比最高的一步**:很多团队上了多跳才发现,加个 reranker 把 top-50 重排到 top-5,单跳就够了,白折腾了多轮。
- **复杂度分类器要离线评**:它分错档(把多跳问题判成 simple)直接导致答错,要单独有回归集盯它的准确率。
- **多跳上下文必须压缩**:旧 hop 只留摘要进下一轮,否则 token 暴涨 + lost-in-the-middle(见 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩与卸载]])。

## 面试高频

**Q1:Agentic RAG 和传统 RAG 到底差在哪?一句话本质。**
传统 RAG 是「检索固定为流程里写死的一步」,Agentic RAG 是「检索升级为 agent 手里的工具」。后者多了五种自适应能力:**自主决定是否检索、query 重写、多跳、检索后自评+再查、多源路由**。代价是更贵更慢、延迟不定。
- *追问:为什么「自评+再查」是灵魂?* 因为它是唯一带**反馈回路**的环节(对应 [[13 Reflection 与 Reflexion|Reflection]]):传统 RAG 检索不好只能将就生成(garbage in garbage out),Agentic RAG 能评估「召回够不够」,不够就换查法/换源再来。
- *陷阱:面试官问「那是不是所有 RAG 都该升级成 Agentic?」* 不是。简单单跳事实问答用传统 RAG 又快又便宜,套 agent 是杀鸡用牛刀;对延迟极敏感、检索源单一稳定的场景也不该上。

**Q2:Self-RAG 和 CRAG 有什么区别?**
都是给 RAG 加「自评纠错」,但机制不同:**Self-RAG**(ICLR 2024 Oral)让模型生成特殊**反思 token**,自己决定「要不要检索 / 检索结果相不相关 / 生成有没有据」,需要微调专用模型;**CRAG(Corrective RAG)**用一个**轻量评估器**给检索结果打分,低分就触发**网页搜索补救**,可挂在现成模型外面,更易落地。
- *追问:生产里更常用哪个?* CRAG 的思路 + Adaptive-RAG 的复杂度路由,因为不依赖微调、易接现有栈。Self-RAG 原样上线稳定性差。

**Q3:Agentic RAG 怎么防止死循环 / 成本失控?**
必设 `max_hops`;允许「实在查不到就基于现有证据尽力答」(不能无限再查);简单问题前置分类走快路径;中间评分/判断用便宜小模型,只最终生成上大模型;每跳检索和重写结果都缓存。
- *陷阱:「自评不可靠怎么办?」* 让模型评自己的检索,会过度自信或过度保守。对策:rubric 给明确标准 + few-shot;评分器最好是独立小模型或专门 reranker 分数,而非让生成模型自己拍脑袋。

**Q4(系统设计):给你一个企业知识库问答,要支持「公司 A 的 CEO 母校在哪个城市」这种多跳问题,怎么设计?**
标准答:① query 改写消歧;② 复杂度分类判为 complex;③ LangGraph 多跳环——hop1 查 CEO 是谁、hop2 查母校、hop3 查城市,每跳的 Observation 喂下一跳;④ 每跳后 CRAG 评分,低分换源/网搜;⑤ 旧 hop 压缩进上下文;⑥ max_hops 护栏 + 最终兜底生成。这本质就是 [[09 ReAct|ReAct]] 在检索场景的展开。

## 知识拓展

**进阶方法谱系(带年份):**
- **Self-RAG**(2023,ICLR 2024 Oral):反思 token 控制检索与生成的自我评判。
- **CRAG / Corrective RAG**(2024):评估器 + 网页补救;Meta KDD Cup 2024 的 CRAG benchmark 专测动态/多样问答。
- **Adaptive-RAG**(NAACL 2024):训练小分类器按问题复杂度路由「不检索 / 单跳 / 多跳」,2026 被视为生产路由的事实标准。
- **GraphRAG**(微软,2024):先把语料抽成知识图谱 + 社区摘要,再在图上做检索,擅长「需要全局综合」的问题(如「整个文档讲了什么主题」),传统向量 RAG 在这类全局问题上很弱。是 Agentic RAG 的正交增强——可作为其一个检索源。
- **HyDE**(2022):先让模型「假想一个答案」再用假想答案去检索,缓解 query 与 doc 的语义鸿沟,是 query 重写的一种。

**边界与反模式:**
- **反模式:为了「显得高级」上多跳**。多数企业问答 80% 是单跳,rerank + 好的 chunk 策略就够;盲目上 agentic 只换来延迟和成本。
- **边界:Agentic RAG ≠ 长上下文模型的替代**。当文档能整篇塞进上下文窗口(百万 token 模型)时,「全塞进去」有时比检索更准——但贵、慢、且 lost-in-the-middle 仍在;两者是权衡不是取代。
- **边界:检索的尽头是 [[24 Agentic Search：grep vs 向量检索|Agentic Search]]**。代码库、结构化场景里,grep / SQL 这类**精确检索**常胜过向量相似度;Agentic RAG 的「多源路由」就该把它们当平级工具挂上。

**相关链接:** 与 [[19 Agent 记忆系统|Agent 记忆系统]] 同源(检索即「读记忆」);灵魂回路对应 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];落地靠 [[15 Function Calling 工具调用|Function Calling 工具调用]] + [[09 ReAct|ReAct]] 循环;路由见 [[05 Routing|Routing]];评估见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]](尤其 `ragas` 的 faithfulness/context-recall);成本控制见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
