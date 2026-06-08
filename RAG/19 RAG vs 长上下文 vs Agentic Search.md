[[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]是给"怎么把外部知识喂给模型"做的**选型决策**:同一个目标——让模型用到训练时没见过的知识——有三条互斥又可叠的路线。**RAG**(检索 top-k 拼进上下文)、**长上下文**(把整库塞进 1M+ 窗口)、**Agentic Search**(让 [[01 什么是 AI Agent|agent]] 自主多轮 grep/检索/下钻)。本质不是"谁更好",而是**在成本、延迟、可更新、准确率、规模、可溯源六个轴上各占什么位置**。

## 本质:三条路线在六轴上的取舍
没有银弹,只有权衡。把三者钉在同一组轴上,选型一目了然:

![[RAG-长上下文-AgenticSearch 决策.png]]

- **[[01 什么是 RAG|RAG]]**:离线建索引,在线一次检索取 top-k 拼上下文。**便宜、快、可更新、可溯源、规模无上限**,但**召回是天花板**——漏检的证据不可逆,后面再强也救不回。
- **长上下文(long context)**:不检索,直接把相关文档(甚至整库)塞进模型的超长窗口,让注意力自己找。**省去检索工程、能跨文档关联**,但**贵(按 token 计费)、慢(prefill 随长度暴涨)、不易更新(每次重塞)、规模卡在窗口上限(1~2M token)、且有 lost-in-the-middle**。
- **[[24 Agentic Search：grep vs 向量检索|Agentic Search]]**:agent 在 [[09 ReAct|ReAct]] 循环里自主决定搜什么、搜几轮、用 grep 还是向量检索、读哪个文件再下钻。**准、能多跳、能自我纠错、能读实时源**,但**最慢最贵**(每轮一次 LLM 调用)。

## 机制

### 长上下文的两个硬伤:lost-in-the-middle 与成本
"把所有东西塞进窗口"听着省事,但 **Liu et al. 2023(Lost in the Middle)**发现一条 **U 形曲线**:模型对上下文**开头和结尾**的信息利用最好,**正中间**的准确率显著塌陷。于是"塞得越多越好"是错觉——关键证据若落在长上下文中段,等于没给。

更要命的是**单事实 vs 多事实的鸿沟**:Gemini 1.5 Pro 在 needle-in-haystack(单针)上 1M token 仍 >99.7% 召回,**但真实多事实检索召回平均只有约 60%**;且 1M token 请求的延迟是 RAG 管线的 **30~60 倍**、单查成本约 **1250 倍**。所以长上下文的高召回基准**严重高估**了它在生产多跳查询上的表现。

**1250× 怎么算出来的**(同一单价下的账)。设输入 token 单价 $p$(元/token)。RAG 一次只把 top-k 检索到的约 **2K token** 拼进上下文,长上下文则把整库约 **1M token** 全塞进去:
> $$\frac{C_{\text{长上下文}}}{C_{\text{RAG}}} = \frac{1{,}000{,}000 \times p}{2{,}000 \times p} = \frac{10^6}{2\times10^3} = 500\times$$
单看 prompt token 已是 **500 倍**;再叠上长序列的 attention 计算/显存随长度超线性增长、KV-Cache 膨胀等开销,综合到约 **1250×**(行业实测)。关键直觉:**单价一样,贵在「每查重复付一遍整库的 prefill」**——RAG 只为真正用到的那 2K 付费,长上下文为没用上的 99.8% 也付了费。这也是为什么"长上下文要取代向量库"的论调被反复证伪——它是**特定场景**(小库、高频跨文档关联)的工具,不是通用替代。

### RAG 的瓶颈:召回,以及它的全套补丁
RAG 的成本/延迟/可溯源都赢,唯一软肋是**召回**——一次性检索若没捞回证据就彻底丢失。但这恰是整个 [[RAG]] 体系在猛攻的方向,等于召回有一整套补丁:[[07 查询变换 Query Transformation|查询变换 Query Transformation]] 改写问句、[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 融合稀疏+稠密兜底关键词、[[10 重排序 Reranking|重排序 Reranking]] 精排、[[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索]] 处理需要串联的问题、[[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引]] 补全块内语义。把这些叠满,RAG 的召回能逼近 agentic search,而成本/延迟仍远低于它。

### Agentic Search:用多轮换准确率
当查询需要**探索**(不知道答案在哪、要先搜再根据结果决定下一步)、**多跳**(A→B→C 的推理链)、或源本身**结构化可下钻**(代码库、文件树),单次检索的 top-k 范式就力不从心。Agentic Search 让模型**把检索当工具反复调用**:搜一轮→看结果→决定再搜什么或读哪个文件→下钻。它天然多跳、能自我纠错、每步可溯源,代价是串行多轮 LLM 调用带来的延迟和 token 成本。[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 还指出:在代码这类**精确匹配**场景,agent 直接 grep 往往比向量检索更准更省——agentic search 不必绑定向量库。

### 三者不互斥:Agentic RAG 是交集
现实系统几乎都是**混合体**,把三者按需叠起来:
1. 用 RAG 召回候选片段(便宜、规模大);
2. 把多个候选塞进长上下文一起读(利用窗口做跨片段关联,但控制总长避开 lost-in-the-middle);
3. 让 agent 多轮迭代——召回不够就改写 query 再检索、需要细节就下钻读原文。
这个"agent 自主决定要不要检索、检索几次、怎么改写"的形态就是 [[36 Agentic RAG|Agentic RAG]];它本质是 **RAG + Agentic Search** 的融合,长上下文则作为"单步能装多少"的容量旋钮。选型的真正问题因此不是"三选一",而是**以哪条为主、另两条补到什么程度**。

## 决策树(何时选谁)
1. **知识库规模?**
   - 全部相关内容 ≤ 1~2M token 且**查询高频、要跨文档关联** → **长上下文**(省检索工程,但盯成本)。
   - 大到塞不下(亿级文档) → 进 2。
2. **查询形态?**
   - **单跳为主、要可溯源、成本敏感** → **RAG**(叠混合检索+重排把召回顶上去)。
   - **多跳/需探索/源结构化(代码、文件树),准确率优先于成本延迟** → **Agentic Search**。
3. **既要规模又要多跳又要可控?** → **[[36 Agentic RAG|Agentic RAG]]**(RAG 召回 + agent 迭代,长上下文当单步容量)。

横轴小抄:**成本/延迟** RAG ≪ 长上下文 ≈ Agentic Search;**可更新** RAG/Agentic > 长上下文;**多跳准确率** Agentic > RAG > 长上下文;**规模上限** RAG/Agentic ≫ 长上下文;**可溯源** RAG/Agentic > 长上下文。

## 可跑最小代码
```python
# 一个"按场景路由三条路线"的选型骨架(伪逻辑,展示决策而非实现)
def route_knowledge_injection(query, corpus_tokens, needs_citation, hops):
    # corpus_tokens: 相关知识总量;hops: 估计需要的推理跳数
    if corpus_tokens <= 1_500_000 and hops <= 1 and not needs_citation:
        return "long_context"     # 小库、单跳、不强求溯源:整段塞窗口
    if hops >= 2 or query_is_exploratory(query):
        return "agentic_search"   # 多跳/需探索:让 agent 多轮检索+下钻
    return "rag"                  # 默认:大库、要溯源、成本敏感 → 检索 top-k

def answer(query, corpus):
    route = route_knowledge_injection(query, corpus.tokens,
                                      needs_citation=True, hops=estimate_hops(query))
    if route == "rag":
        ctx = hybrid_retrieve(query, corpus)        # 混合检索+重排,见对应笔记
        return llm(build_prompt(query, ctx))
    if route == "long_context":
        return llm_long(build_prompt(query, corpus.top_docs()))  # 注意 lost-in-the-middle
    return agentic_loop(query, corpus)              # ReAct 循环多轮搜索
```

## 对比表
| 轴 | RAG | 长上下文 | Agentic Search |
|---|---|---|---|
| 成本 | 低 | 高(token 计费) | 高(多轮 LLM) |
| 延迟 | 低(一次检索) | 高(prefill 慢) | 高(串行多步) |
| 可更新 | 高(改库即更新) | 中(需重塞) | 高(读实时源) |
| 准确率 | 中(卡召回) | 单事实高/多跳塌 | 高(自我纠错) |
| 规模上限 | 高(亿级) | 低(≤1~2M token) | 高(逐层下钻) |
| 可溯源 | 强(片段引用) | 弱(难定位) | 强(每步可查) |
| 维护 | 索引/embedding 工程 | 几乎免维护 | agent 编排复杂 |

## 何时用 / 坑
- **别被 needle-in-haystack 基准忽悠**:单针 99% 召回 ≠ 生产多事实 60% 召回,长上下文在真实多跳上远不如基准好看。
- **长上下文的"省事"有隐藏账单**:省了检索工程,但每查贵上千倍、慢几十倍,高频场景成本爆炸。
- **RAG 召回不达标别急着上长上下文**:先把 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]]+[[10 重排序 Reranking|重排序 Reranking]]+[[07 查询变换 Query Transformation|查询变换 Query Transformation]] 叠满,多数"RAG 不行"其实是"Naive RAG 不行"(见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]])。
- **Agentic Search 要给预算上限**:不限轮数会无限搜下去,成本/延迟失控,务必配 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]] 的 step 上限与缓存。
- **可溯源是合规硬需求时**:长上下文很难精确定位"这句话出自哪页",RAG/agentic 的片段级引用(见 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]])几乎是唯一选择。
- **多数生产系统是混合**:用 [[20 上下文工程|上下文工程]] 思路把三者当旋钮调,而非教条三选一;[[21 上下文压缩与卸载|上下文压缩与卸载]] 帮你在长上下文里只留精华、规避 lost-in-the-middle。

## 关键事实
- **Lost in the Middle**(Liu et al. 2023, TACL):长上下文利用呈 U 形,中段信息准确率显著下降——"塞满窗口"是错觉。
- **Gemini 1.5 Pro**(arXiv:2403.05530):needle-in-haystack 1M token >99.7% 召回,但**多事实真实检索召回约 60%**,1M 请求延迟约 RAG 的 30~60×、成本约 1250×。
- RAG 的唯一硬伤是**召回**,且有一整套补丁(混合检索/重排/多跳/进阶索引)可把它逼近 agentic 水平,而成本仍最低。
- 三者非互斥:[[36 Agentic RAG|Agentic RAG]] = RAG 召回 + agent 多轮迭代,长上下文作单步容量旋钮;选型实质是"以谁为主、补到多少"。

## 工业界实践

选型在工业界从来不是「三选一」,而是「**以谁为主、另两条补到什么程度**」,且被**成本和延迟**狠狠约束。

**1)成本/延迟的真实账单(为什么 RAG 仍是主力)**
- 长上下文的「省事」有隐藏账单:把整库塞窗口意味着**每查重复付一遍 prefill**。1M token 请求相比 RAG 管线,延迟约 **30~60×**、单查成本约 **1250×**(行业实测)。高频场景直接成本爆炸。
- **Prompt caching** 改变了部分算式:如果上下文是**稳定前缀**(同一份长文档反复问),Anthropic/OpenAI/DeepSeek 的 prompt cache 能让重复 prefill 降到 ~10% 成本——这让「固定知识库 + 长上下文」在**单文档高频问答**场景变得可行。但库一变 cache 失效,**可更新性差**的硬伤还在。
- 结论:**RAG 仍是规模化生产的默认主力**(便宜、可更新、可溯源、规模无上限);长上下文是「小库 + 高频跨文档关联 + 能吃 cache」的特定工具,不是通用替代。

**2)长上下文的两个被基准掩盖的硬伤**
- **Lost-in-the-middle**(Liu et al. 2023):U 形利用曲线,关键证据落中段约等于没给。生产对策:把最相关片段放**开头或结尾**(用 rerank 决定摆位),别均匀铺。
- **单事实 vs 多事实鸿沟**:needle-in-haystack 单针 99%+ 召回是**营销数字**;真实**多事实**检索召回平均仅约 **60%**(Gemini 1.5 Pro 实测)。别拿单针基准给长上下文背书。

**3)RAG 召回不达标?先叠补丁,别急着换路线**
多数「RAG 不行」其实是「**Naive RAG 不行**」(见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]])。召回有一整套补丁,叠满能逼近 agentic 水平而成本仍最低:
[[07 查询变换 Query Transformation|查询变换]] 改写问句 → [[08 混合检索 Hybrid Search|混合检索]] 稀疏+稠密兜底关键词 → [[10 重排序 Reranking|重排序]] 精排 → [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索]] 串联多步 → [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引]] 补块内语义。

**4)Agentic Search 的生产护栏**
- 让 agent 在 [[09 ReAct|ReAct]] 循环里自主多轮检索/下钻,准但**最慢最贵**(每轮一次 LLM 调用)。
- 必须配**预算上限**:step 上限、token 预算、超时;不限轮数会无限搜下去成本失控(见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]])。
- 代码/文件树这类**精确匹配**场景,agent 直接 **grep 往往比向量检索更准更省**(见 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]])——agentic search 不必绑向量库。**Claude Code、Cursor、Cline** 等代码 agent 大量用 grep/ripgrep 而非向量库,就是这个道理。

**5)生产混合架构(几乎所有真实系统)**
```
RAG 召回候选(便宜、规模大)
   ↓ rerank 决定摆位(最相关放首尾,避 lost-in-the-middle)
塞进受控长度的上下文(用窗口做跨片段关联,但不塞满)
   ↓ 召回不够 / 需细节
agent 多轮迭代:改写 query 再检索 / 下钻读原文
```
这就是 [[36 Agentic RAG|Agentic RAG]]:**RAG + Agentic Search** 的融合,长上下文作「单步能装多少」的容量旋钮。选型实质是调这三个旋钮的配比。

**6)踩坑**
- 被 needle-in-haystack 单针基准忽悠(单针 99% ≠ 多事实 60%)。
- 把长上下文当「免维护银弹」,忽略每查的 token 账单。
- RAG 召回没叠满补丁就断言「RAG 不行」转投长上下文。
- Agentic Search 不设预算上限,成本/延迟失控。
- 合规要可溯源时选长上下文——它**难精确定位「这句出自哪页」**,这是硬伤。

## 面试高频

**Q1:RAG、长上下文、Agentic Search 怎么选?**
标准答:没有银弹,在**成本/延迟/可更新/准确率/规模/可溯源**六轴上权衡。RAG 便宜快可更新可溯源规模无上限、软肋是召回;长上下文省检索工程能跨文档关联、但贵慢难更新有 lost-in-the-middle、规模卡窗口;Agentic Search 准能多跳能自我纠错、但最慢最贵。生产几乎都是混合(Agentic RAG),选型实质是「以谁为主、补到多少」。

**Q2:长上下文(1M+ 窗口)会取代 RAG / 向量库吗?**
标准答:不会取代,是特定场景工具。理由三条:① **lost-in-the-middle** 中段塌陷;② 单针基准 99% 召回掩盖真实多事实仅约 60%;③ 每查贵约 1250×、慢约 30~60×,高频场景成本爆炸,且**不易更新、难精确溯源**。长上下文适合「小库 + 高频跨文档关联」。
- 陷阱:只背「窗口够大就不用检索」——忽略了成本、可更新性、可溯源三个生产硬约束。

**Q3:什么是 lost-in-the-middle?对架构有什么影响?**
标准答:Liu et al. 2023 发现长上下文利用呈 **U 形**,模型对开头结尾信息利用最好、**正中间准确率显著塌陷**。影响:别把关键证据均匀铺满窗口,要用 rerank 把最相关片段摆到**开头或结尾**;也是「塞满窗口」是错觉的根因。

**Q4:什么时候必须上 Agentic Search 而非单次 RAG?**
标准答:查询需要**探索**(不知道答案在哪、要边搜边定下一步)、**多跳**(A→B→C 推理链)、或源**结构化可下钻**(代码库、文件树)时。单次 top-k 范式力不从心,让模型把检索当工具反复调用,天然多跳 + 能自我纠错 + 每步可溯源,代价是串行多轮 LLM 调用。代码场景常用 grep 不用向量库。

**Q5:RAG 的唯一硬伤是什么?能补到什么程度?**
标准答:**召回**——一次性检索没捞回证据就彻底丢失,不可逆。但有一整套补丁(查询变换 + 混合检索 + 重排 + 多跳 + 进阶索引),叠满能把召回逼近 agentic 水平,而成本/延迟仍远低于长上下文和 agentic search。多数「RAG 不行」是「Naive RAG 不行」。

## 知识拓展

- **Prompt caching 重塑了部分边界**:长上下文最大的成本来自重复 prefill,prompt cache(稳定前缀降到 ~10% 成本)让「固定大文档高频问答」从「太贵」变「可行」。但库一变 cache 失效,RAG 的可更新优势依旧。这也是 [[20 上下文工程|上下文工程]] / [[21 上下文压缩与卸载|上下文压缩与卸载]] 关心的旋钮:在长上下文里只留精华、规避 lost-in-the-middle。
- **与长上下文推理技术的连接**:长上下文不是免费的——它依赖 [[LLM/107 长上下文推理：YaRN、位置插值与 StreamingLLM|长上下文推理]] 的位置外推(YaRN/PI)和 [[LLM/102 KV-Cache|KV-Cache]] 管理;窗口越长,KV-Cache 显存和 prefill 算力越爆炸,这是「贵和慢」的物理根源。
- **可溯源是合规硬需求时**:长上下文很难精确定位「这句话出自哪页」,RAG/agentic 的片段级引用(见 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]])几乎是唯一选择——这条往往直接决定金融/医疗/法律场景的选型。
- **前沿**:**RAG vs 长上下文之争**反复被证伪又重提,主流共识(2024–2026)是**互补而非替代**——长上下文当「RAG 的单步容量放大器」,让每次能塞更多召回片段、做更强跨片段关联。**Self-Route / 自适应路由**(按 query 难度动态选 RAG 还是长上下文)是降本的活跃方向,和 [[12 Self-RAG、CRAG 与 Adaptive RAG|Adaptive RAG]] 同源。
- **反模式**:① 教条三选一(现实是混合);② 拿单针基准给长上下文背书;③ 召回没叠满补丁就弃 RAG;④ Agentic Search 不设预算;⑤ 用长上下文做强溯源场景。本篇的选型结论要和 [[18 RAG 评估|RAG 评估]](先量化召回再决定换不换路线)、[[36 Agentic RAG|Agentic RAG]](混合形态的落地)一起读。

## 来源
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). **Lost in the Middle: How Language Models Use Long Contexts**. TACL 2024 / arXiv:2307.03172. — 长上下文 U 形利用、中段塌陷。
- Gemini Team, Google (2024). **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context**. arXiv:2403.05530. — 1M~2M 窗口 needle-in-haystack >99% 召回(单针)。
- 行业实测(2026,multiple)——单针 99% vs 多事实约 60% 召回鸿沟、1M 请求 30~60× 延迟与约 1250× 成本,佐证"长上下文非通用替代"。
