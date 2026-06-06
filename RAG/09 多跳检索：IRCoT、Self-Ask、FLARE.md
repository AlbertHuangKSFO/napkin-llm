[[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 的本质是:把检索从"一次性"变成"**迭代多步**",让**检索和推理交织**进行——检索拿到证据→推理一步→根据推理结果决定下一步查什么→再检索……直到信息足够才生成。核心洞见一句话:**下一跳查什么,取决于已经推出了什么**,而这又取决于上一跳查到了什么;一次 retrieve-and-read 根本撑不起这种依赖。

它修的是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 里最硬的一类:**多跳/组合问题**。"特斯拉 CEO 创办的航天公司总部在哪"——得先知道 CEO 是谁、再知道他的航天公司是哪家、再查总部。单跳 RAG 拿原话一次检索,这三个事实没有任何一篇文档同时包含,必然抓瞎。这几个方法是 [[36 Agentic RAG|Agentic RAG]] 的**前身/非 agent 化版本**:它们把"检索↔推理循环"写成固定流程或固定触发规则,而 Agentic RAG 把同一个循环交给模型在 [[09 ReAct|ReAct]] 式 [[03 Agent 核心循环|Agent 核心循环]] 里自主调度。

## 本质:单跳 vs 多跳交织

单跳 [[01 什么是 RAG|什么是 RAG]]:`query →（原话）检索一次 → 拼接 → 生成`。它假设"答案就在一次检索召回的 top-k 里"。对单跳事实问答成立,对组合问题崩。

多跳的关键结构:**检索与推理交替成环**,每一跳的检索结果喂给下一跳的推理,推理产物又当作下一跳的检索 query。这正是 [[09 ReAct|ReAct]] 的 Thought→Action(retrieve)→Observation 在检索场景的落地。

![[多跳检索-交织.svg]]

跟 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 里的 **Decompose** 区分:分解是**一次性把子问题列全、可并行**;多跳是**串行依赖——前跳结果决定后跳查什么**,不到上一跳出结果你都不知道下一跳要问啥(像"CEO 是谁"没查出来,就没法构造"他的航天公司"这一跳)。

## 机制:三种方法逐一拆

### Self-Ask(显式提后续子问题)
让模型在回答前**显式地自问自答 follow-up 子问题**,每个子问题可以插一次外部检索来回答。输出形如:
> 需要后续问题吗?是。
> 后续问题:特斯拉的 CEO 是谁? 中间答案:Elon Musk（← 这里可调检索）
> 后续问题:Musk 创办的航天公司是? 中间答案:SpaceX
> 后续问题:SpaceX 总部在哪? 中间答案:Hawthorne, 加州
> 所以最终答案是:Hawthorne, 加州

出处:Press, Zhang, Min, Schmidt, Smith, Lewis《Measuring and Narrowing the Compositionality Gap in Language Models》(arXiv:2210.03350, 2022;EMNLP Findings 2023)。论文先提出"组合性鸿沟"(compositionality gap):模型能答对所有子问题、却答不对整体问题的比例;随模型变大,单跳能力涨得比多跳快,鸿沟不缩小。Self-Ask 用**结构化的显式 follow-up**改进 CoT,且这种结构**天然能插搜索引擎**回答子问题,准确率再升。它最贴近 [[09 ReAct|ReAct]],被视作 ReAct 在 QA 上的近亲。

### IRCoT(检索与 CoT 交错)
把检索**交错进 CoT 的每一句**:不是先想完再查,也不是先查完再想,而是**生成一句 CoT → 用这句话当 query 检索 → 把召回文档拼进上下文 → 生成下一句 CoT → 再检索……** 双向增益:CoT 指导检索(retrieval guided by CoT),检索结果又改进下一句 CoT(reasoning improved by retrieval)。出处:Trivedi, Balasubramanian, Khot, Sabharwal《Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions》(arXiv:2212.10509, 2022;ACL 2023)。在 HotpotQA、2WikiMultihopQA、MuSiQue、IIRC 上,检索 +21 分、下游 QA +15 分,并减少幻觉、让 CoT 更事实准确;小模型(Flan-T5-large)免训练也能用。和 Self-Ask 的差别:Self-Ask 以**离散子问题**为单位插检索,IRCoT 以 **CoT 的每一句**为单位插检索,粒度更细。

### FLARE(前瞻生成,主动检索)
针对**长文本生成**:大多数 RAG 只在开头检索一次,但生成长答案时,信息需求是**沿途变化**的。FLARE(Forward-Looking Active REtrieval)在生成时**先试生成下一句**;若这句里**有 token 的预测概率低于阈值**(模型对这部分没把握、可能要编),就把这些低置信 span 当作**检索信号**组成 query,去检索,再**用召回内容重新生成那一句**。即:**模型自己不确定时才主动触发检索**。出处:Jiang, Xu, Gao, Sun, Liu, Yang, Callan, Neubig《Active Retrieval Augmented Generation》(arXiv:2305.06983, 2023;EMNLP 2023)。它把"何时检索"从"固定一次"升级成"由生成置信度动态触发",是这三者里最接近 [[36 Agentic RAG|Agentic RAG]]"自主决定是否检索"的。

## 可跑最小代码

```python
# 多跳交织的最小骨架:迭代「推理一步 → 检索 → 拼证据」直到收敛(IRCoT/Self-Ask 风味)
def multi_hop(question, llm, retrieve, max_hops=4):
    evidence, cot = [], ""
    for hop in range(max_hops):
        # 1) 推理一步:基于已有证据,生成下一句 CoT(或下一个 follow-up 子问题)
        step = llm(
            "你在做多跳问答。基于已有证据,写出推理的下一句;"
            "若已能作答,以 'ANSWER:' 开头给出最终答案。\n"
            f"问题:{question}\n已有证据:\n{chr(10).join(evidence) or '(无)'}\n已有推理:{cot}"
        )
        if step.startswith("ANSWER:"):
            return step[len("ANSWER:"):].strip()      # 信息够,收敛退出
        cot += " " + step
        # 2) 用这句 CoT 当 query 去检索(下一跳查什么 = 刚推出的内容)
        evidence += [d for d in retrieve(step, k=3)]    # 串行依赖:依赖上一步产物
    # 触顶仍未收敛:基于现有证据尽力答(必备兜底)
    return llm(f"基于以下证据尽力回答:{question}\n{chr(10).join(evidence)}")

# FLARE 风味:只在「下一句里有低置信 token」时才触发检索
def flare_step(draft_sentence, token_logprobs, retrieve, llm, thresh=-2.0):
    low_conf = [tok for tok, lp in token_logprobs if lp < thresh]  # 模型没把握的 span
    if not low_conf:
        return draft_sentence                          # 有把握 → 不检索,直接用
    docs = retrieve(" ".join(low_conf), k=3)           # 低置信 span 当检索信号
    return llm(f"参考资料重写这句,使其有据:{draft_sentence}\n资料:{docs}")
```

要点:① 检索 query 来自**上一步的推理产物**(`retrieve(step)`),这就是"串行依赖";② `max_hops` 是**必备护栏**——交织循环不收敛会无限查,跟 [[36 Agentic RAG|Agentic RAG]] 一样要设上限 + 兜底答;③ FLARE 的差别在**触发条件**:用 token 置信度决定是否检索,而非每跳都查。

## 对比表

| 方法 | 检索触发单位 | 何时检索 | 一句话特征 | 出处 |
|---|---|---|---|---|
| Self-Ask | 显式 follow-up 子问题 | 每个子问题前 | 自问自答,可插搜索 | arXiv:2210.03350 |
| IRCoT | CoT 的每一句 | 每生成一句 CoT | 检索与 CoT 逐句交错 | arXiv:2212.10509 |
| FLARE | 低置信 token span | 模型没把握时 | 前瞻生成,置信度触发 | arXiv:2305.06983 |
| Decompose([[07 查询变换 Query Transformation|查询变换 Query Transformation]]) | 一次列全子问题 | 检索前一次拆 | 并行、无串行依赖 | — |
| [[36 Agentic RAG|Agentic RAG]] | 模型自主 | 模型自主决定 | 上面这些的 agent 化 | — |

## 何时用 / 坑

**该用多跳**:组合/多跳问题(答案散在多篇文档,需逐步链)、需长文本生成且信息需求沿途变化(FLARE)、单跳召回率明显不够的知识密集 QA。这是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 中"多跳搞不定"的正解,也是迈向 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 和 [[36 Agentic RAG|Agentic RAG]] 的中间台阶。

**坑**:
- **不收敛 / 死循环**:交织循环反复查同样的东西。必设 `max_hops` + "查不到就基于现有证据尽力答"。
- **错误累积**:某一跳推理或检索错了,后续跳都建立在错前提上,越走越偏。中间步骤最好能自评(这就引向 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 和 CRAG 的纠错)。
- **延迟 × 跳数**:每跳 = 一次推理 + 一次检索,多跳累积慢且贵。简单单跳问题别套多跳。
- **上下文膨胀**:每跳的证据都堆进上下文,token 暴涨 + lost-in-the-middle。需配 [[20 上下文工程|上下文工程]],旧跳证据压缩或只留摘要。
- **FLARE 阈值难调**:置信阈值太松→几乎不检索(退化成纯生成);太紧→几乎每句都查(退化成 IRCoT 还更贵)。

## 关键事实

- 多跳检索 = **迭代/多步检索,检索与推理交织**;核心:下一跳查什么取决于已推出什么。修的是单跳 RAG 搞不定的组合/多跳问题。
- **Self-Ask**(arXiv:2210.03350, Press et al. 2022):显式自问 follow-up 子问题,可插搜索;提出"组合性鸿沟"。最贴近 [[09 ReAct|ReAct]]。
- **IRCoT**(arXiv:2212.10509, Trivedi et al. 2022):检索与 CoT **逐句交错**,双向增益;HotpotQA 等 +15~21 分。
- **FLARE**(arXiv:2305.06983, Jiang et al. 2023):前瞻生成,**低置信 token 触发**检索;最接近"自主决定是否检索"。
- 三者都是 [[36 Agentic RAG|Agentic RAG]] 的**前身/非 agent 化版本**:把检索↔推理循环写成固定流程/固定触发,而非模型自主调度。
- vs Decompose([[07 查询变换 Query Transformation|查询变换 Query Transformation]]):多跳是**串行依赖**,Decompose 是**并行列全**。
- 必备护栏:`max_hops` + 兜底答;否则不收敛。延迟/成本随跳数线性涨。

## 工业界实践

工业界很少把 IRCoT/Self-Ask/FLARE 当成"论文实现"原样照搬,而是把它们的**核心 pattern**(检索↔推理交织、置信度触发)接进编排框架里跑。

**主流落地框架与组件**
- **LangGraph**:把多跳写成**有环状态图**——节点 = `retrieve` / `generate_step` / `grade` / `decide_next`,边带条件,`max_hops` 用图的递归上限 (`recursion_limit`) 兜底。社区 Adaptive-RAG / Self-RAG / CRAG 官方教程都用它,多跳是同一套骨架。这是 2025 年生产里跑迭代检索最常见的载体。
- **LlamaIndex**:`SubQuestionQueryEngine`(偏 Decompose 并行)、`MultiStepQueryEngine`(偏串行多跳)、`QueryPipeline` + `FnComponent` 自定义循环;`StepDecomposeQueryTransform` 显式把"已有答案"喂回去生成下一跳 query,正是串行依赖的工程化。
- **DSPy**:`dspy.ReAct` / 多跳 `dspy.ChainOfThought` + retriever 工具,配 `MIPROv2` / `BootstrapFewShot` **编译**——不手写多跳 prompt,而是用少量标注让优化器自动搜出"下一跳查什么"的提示词,框架开销也最低(约 3.5ms)。多跳的 prompt 工程在这里变成可优化的程序。
- **检索后端**:每一跳的检索仍走标准 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] + [[10 重排序 Reranking|重排序 Reranking]];向量库用 Qdrant / Weaviate / pgvector / Milvus,检索质量直接决定多跳能不能"接得上"。

**典型架构(生产多跳问答)**
```
query → [复杂度路由] ──简单──→ 单跳 RAG
                     └─复杂──→ LangGraph 环:
                        ┌────────────────────────────┐
                        │ generate_step(基于已有证据) │
                        │   ↓ 提取下一跳 query        │
                        │ retrieve + rerank          │
                        │   ↓ grade(够答了吗?)       │
                        └──否→回到 generate_step──────┘
                                  ↓是 / 触顶 max_hops
                               grounded 生成 + 引用
```

**规模化关键**
- **延迟 = 跳数 × (推理 + 检索 + 重排)**,是单跳的数倍。工程上:① 用 Adaptive 路由让 80% 简单流量走单跳,只有真多跳才进环;② 跳间证据**压缩成摘要**再带入下一跳,压住 token 与 lost-in-the-middle;③ 每跳推理可换小模型(下一跳 query 生成不需要旗舰模型),只在最终生成用大模型。
- **索引选型**对多跳尤其敏感:漏召一跳全盘崩,召回阶段宁可用 HNSW(高召回、可调 `efSearch`)而非激进量化的 IVF-PQ;`ef` 设大些保召回,把精度交给重排。
- **缓存**:多跳里中间子问题高度重复("X 的 CEO 是谁"),对子问题 query→证据做语义缓存(GPTCache 类)能省掉大量重复检索。

**评估与可观测**
- **Ragas** 的多跳专项指标:除 faithfulness / context precision 外,有针对 multi-hop 的 `MultiHopAbstractQA` / `MultiHopSpecificQA` 合成数据生成,专测"证据要跨多片段拼"的题。
- **TruLens / LangSmith / Phoenix**:用 OpenTelemetry **trace 每一跳**——看哪一跳召回崩了、哪一跳推理跑偏、总共跳了几次、是否触顶。多跳调试的核心就是"逐跳看 trace",没有 tracing 等于盲调。
- 数据集:**HotpotQA、2WikiMultihopQA、MuSiQue、IIRC**(IRCoT 论文用的四个),线上回归测试常驻这几套。

**踩坑与最佳实践**
- 多跳**不要默认全开**:简单题套多跳是纯浪费(延迟 ×N、还可能因冗余检索引噪)。先上 Adaptive 路由([[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]])。
- 每跳**必须 grade**(够不够答),否则要么提前停(漏信息)、要么跑满 max_hops(浪费)。grade 是收敛信号。
- 跳间一定要**去重证据**:同一篇文档反复被召,既占 token 又给模型"反复确认"的错觉。
- FLARE 的置信阈值**离线在验证集上调**,别拍脑袋;并监控"平均触发检索次数",过高过低都说明阈值偏了。

## 面试高频

**Q1:多跳检索和查询分解(Decompose)有什么区别?**
标准答:Decompose 是**一次性把子问题全列出来、可并行检索**;多跳是**串行依赖——下一跳查什么取决于上一跳的结果**,不到上一跳出答案你都构造不出下一跳 query("他的航天公司"必须先知道"他是谁")。
- 追问"什么时候用哪个?":子问题之间**无依赖**(如"对比 A 和 B 的营收")用 Decompose 并行,快;**有依赖链**(如"X 的 CEO 创办的公司的总部")必须多跳串行。
- 陷阱:面试官给一个看似多跳实则可并行的题,考你能否识别"依赖 vs 独立"。

**Q2:IRCoT、Self-Ask、FLARE 三者的核心差异?**
标准答:差在**检索的触发单位和时机**。Self-Ask 以**显式 follow-up 子问题**为单位,每个子问题前检索;IRCoT 以 **CoT 的每一句**为单位,逐句交错检索;FLARE 以**低置信 token span** 为触发,只在模型"没把握"时才检索。粒度:Self-Ask(子问题)> IRCoT(句)> FLARE(token 置信)。
- 追问"FLARE 凭什么决定检索?":生成下一句时若有 token 的预测概率低于阈值,就把低置信 span 当检索 query,检索后重写该句。它把"何时检索"从固定改成置信度动态触发。
- 追问"哪个最接近 Agentic RAG?":FLARE,因为它最接近"模型自主决定是否检索"。

**Q3:多跳检索为什么容易"错误累积"?怎么缓解?**
标准答:每一跳的推理/检索建立在上一跳产物上,某跳错了,后续全建在错前提上,越走越偏。缓解:① 每跳 **grade/self-check**(对应 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 和 CRAG 纠错);② 保留多候选而非贪心走单条链(beam);③ 允许回溯/重查。
- 陷阱:只答"加 max_hops" 不够——max_hops 防的是不收敛,防不了错误累积,二者是不同的坑。

**Q4:多跳系统怎么保证不死循环?**
标准答:① 硬上限 `max_hops`;② 收敛判据(grade 判"信息够了"提前停);③ 触顶兜底("查不到就基于现有证据尽力答");④ 跳间去重防止反复查同一个东西。四件缺一不可。

**Q5(场景题):用户问"诺贝尔物理学奖得主中,谁的母校也培养过现任某科技公司 CEO?"——你怎么设计?**
要点:识别为多跳依赖链 → LangGraph/IRCoT 式环 → 每跳 retrieve+rerank → 跳间证据摘要压缩 → grade 收敛 → 触顶兜底 → 最终 grounded 生成带引用。能说出"路由判复杂度、逐跳 trace、证据去重"是加分项。

## 知识拓展

**进阶方法与前沿(带年份)**
- **IR-CoT / Self-Ask / FLARE(2022–2023)** 是第一代"交织检索"。再往后是把循环交给模型自主调度的方向:
- **Self-RAG(Asai 2023,ICLR 2024)**:把"是否检索 + 是否相关 + 是否有据"内化成反思 token,见 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]。
- **Adaptive-RAG(Jeong 2024,NAACL)**:前置复杂度分类器,把"该不该多跳"路由化——正是多跳"别默认全开"的正解。
- **Search-o1 / RAG with reasoning models(2025)**:把检索接进 o1/R1 式**长推理链**,推理模型在 think 过程中自主决定何时调 search 工具,是 FLARE"置信度触发"思想在推理大模型上的延伸。
- **强化学习训检索-推理(2025)**:**Search-R1 / R1-Searcher / DeepRetrieval** 等用 RL(GRPO/PPO)直接训模型"在推理中何时检索、查什么",不再手写多跳流程,把多跳能力**学进策略**——这是当前多跳研究最活跃的方向。
- **测试时计算(test-time compute)视角**:多跳本质是"用更多检索-推理步换准确率",和 CoT 加长、beam 加宽同属 test-time scaling;[[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]] 里"agentic search"就是多跳的极端形态。

**边界与反模式**
- **反模式一:对单跳题套多跳**——延迟 ×N、还可能因冗余检索引入噪声反而降准。多跳是把双刃剑,默认全开是典型误用。
- **反模式二:不 grade 的固定跳数**——要么提前停漏信息,要么跑满浪费;固定跳数几乎总是次优。
- **边界**:当单篇文档已含全部所需事实(真单跳),多跳零收益纯亏成本;当依赖链极长(>5 跳)时错误累积通常压垮收益,此时该考虑 [[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]]——多跳事实关系用图遍历比反复 LLM 推理更稳更省。

**相关链接**:它是 [[36 Agentic RAG|Agentic RAG]] 的非 agent 化前身,机制上是 [[09 ReAct|ReAct]] 在检索场景的落地(Thought→retrieve→Observation);承接 [[07 查询变换 Query Transformation|查询变换 Query Transformation]](Decompose 是其并行近亲),向后接 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]];纠错思想来自 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]],上下文管理靠 [[20 上下文工程|上下文工程]]。

## 来源
- Press, Zhang, Min, Schmidt, Smith, Lewis.《Measuring and Narrowing the Compositionality Gap in Language Models》(Self-Ask). arXiv:2210.03350, 2022;EMNLP Findings 2023.
- Trivedi, Balasubramanian, Khot, Sabharwal.《Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions》(IRCoT). arXiv:2212.10509, 2022;ACL 2023.
- Jiang, Xu, Gao, Sun, Liu, Yang, Callan, Neubig.《Active Retrieval Augmented Generation》(FLARE). arXiv:2305.06983, 2023;EMNLP 2023.
