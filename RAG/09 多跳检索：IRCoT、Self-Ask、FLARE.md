[[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 的本质是:把检索从"一次性"变成"**迭代多步**",让**检索和推理交织**进行——检索拿到证据→推理一步→根据推理结果决定下一步查什么→再检索……直到信息足够才生成。核心洞见一句话:**下一跳查什么,取决于已经推出了什么**,而这又取决于上一跳查到了什么;一次 retrieve-and-read 根本撑不起这种依赖。

它修的是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 里最硬的一类:**多跳/组合问题**。"特斯拉 CEO 创办的航天公司总部在哪"——得先知道 CEO 是谁、再知道他的航天公司是哪家、再查总部。单跳 RAG 拿原话一次检索,这三个事实没有任何一篇文档同时包含,必然抓瞎。这几个方法是 [[36 Agentic RAG|Agentic RAG]] 的**前身/非 agent 化版本**:它们把"检索↔推理循环"写成固定流程或固定触发规则,而 Agentic RAG 把同一个循环交给模型在 [[09 ReAct|ReAct]] 式 [[03 Agent 核心循环|Agent 核心循环]] 里自主调度。

## 本质:单跳 vs 多跳交织

单跳 [[01 什么是 RAG|什么是 RAG]]:`query →（原话）检索一次 → 拼接 → 生成`。它假设"答案就在一次检索召回的 top-k 里"。对单跳事实问答成立,对组合问题崩。

多跳的关键结构:**检索与推理交替成环**,每一跳的检索结果喂给下一跳的推理,推理产物又当作下一跳的检索 query。这正是 [[09 ReAct|ReAct]] 的 Thought→Action(retrieve)→Observation 在检索场景的落地。

![[多跳检索-交织.png]]

跟 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 里的 **Decompose** 区分:分解是**一次性把子问题列全、可并行**;多跳是**串行依赖——前跳结果决定后跳查什么**,不到上一跳出结果你都不知道下一跳要问啥(像"CEO 是谁"没查出来,就没法构造"他的航天公司"这一跳)。

像侦探查案:你**先查出公司的 CEO 是谁**,拿到名字才知道下一步该去翻谁的银行流水;名字没出来,后面那条线索压根无从查起——线索是一环扣一环顺着挖,不是一开始就把所有要问的人列在白板上同时审。

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

❌ 朴素写法只会把所有证据不断塞回 prompt：既看不出哪一跳慢/错，也无法区分 `LLM`、检索和重排的成本。

```python
def bad_multi_hop(question, llm, retrieve):
    evidence = []
    for _ in range(4):                    # 固定跑满，不检查是否已足够
        step = llm(f"问题:{question}\n证据:{evidence}")
        evidence.extend(retrieve(step, k=3))  # 重复证据会继续膨胀
    return llm(f"请作答:{question}\n证据:{evidence}")
```

✅ 下例把每跳拆成「推理 → 检索 → 重排」，记录每段延迟和输入/输出 token；传入真实的 `llm`、`retrieve`、`rerank`、`count_tokens` 回调即可运行。`count_tokens` 应使用与线上模型一致的 tokenizer，或改为读取供应商实际返回的 usage。

```python
from dataclasses import asdict, dataclass
from time import perf_counter


@dataclass
class HopTrace:
    hop: int
    llm_ms: float
    retrieve_ms: float
    rerank_ms: float
    input_tokens: int
    output_tokens: int
    query: str


def timed(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, round((perf_counter() - start) * 1000, 1)


def multi_hop(question, llm, retrieve, rerank, count_tokens, max_hops=4):
    evidence, traces = [], []
    for hop in range(1, max_hops + 1):
        prompt = (
            "基于证据写下一个可检索的子问题；若证据已充分则以 ANSWER: 开头。\n"
            f"问题:{question}\n证据:\n{chr(10).join(evidence) or '(无)'}"
        )
        step, llm_ms = timed(llm, prompt)
        if step.startswith("ANSWER:"):
            traces.append(HopTrace(hop, llm_ms, 0, 0,
                                   count_tokens(prompt), count_tokens(step), ""))
            return step.removeprefix("ANSWER:").strip(), [asdict(t) for t in traces]

        docs, retrieve_ms = timed(retrieve, step, 8)
        ranked, rerank_ms = timed(rerank, question, docs, 3)
        evidence.extend(ranked)
        traces.append(HopTrace(hop, llm_ms, retrieve_ms, rerank_ms,
                               count_tokens(prompt), count_tokens(step), step))

    final_prompt = f"仅基于下列证据回答。\n问题:{question}\n证据:\n{chr(10).join(evidence)}"
    answer, _ = timed(llm, final_prompt)  # 终答不是检索 hop；生产中另记该 span
    return answer, [asdict(t) for t in traces]
```

`traces` 可逐跳写入 trace：出现错误答案时先看该跳的 query/证据；出现慢请求时分别看 `llm_ms`、`retrieve_ms`、`rerank_ms`；容量与计费分析则汇总 `input_tokens`、`output_tokens`。FLARE 风味只需把 `step` 的产生改为「试生成下一句 → 低置信 token span 触发检索 → 带证据重写该句」；它的触发条件仍是置信度，而不是 IRCoT 的逐步固定检索。

## 对比表

| 方法 | 检索触发单位 | 何时检索 | 一句话特征 | 出处 |
|---|---|---|---|---|
| Self-Ask | 显式 follow-up 子问题 | 每个子问题前 | 自问自答,可插搜索 | arXiv:2210.03350 |
| IRCoT | CoT 的每一句 | 每生成一句 CoT | 检索与 CoT 逐句交错 | arXiv:2212.10509 |
| FLARE | 低置信 token span | 模型没把握时 | 前瞻生成,置信度触发 | arXiv:2305.06983 |
| Decompose([[07 查询变换 Query Transformation\|查询变换 Query Transformation]]) | 一次列全子问题 | 检索前一次拆 | 并行、无串行依赖 | — |
| [[36 Agentic RAG\|Agentic RAG]] | 模型自主 | 模型自主决定 | 上面这些的 agent 化 | — |

## 何时用 / 坑

**该用多跳**:组合/多跳问题(答案散在多篇文档,需逐步链)、需长文本生成且信息需求沿途变化(FLARE)、单跳召回率明显不够的知识密集 QA。这是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 中"多跳搞不定"的正解,也是迈向 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 和 [[36 Agentic RAG|Agentic RAG]] 的中间台阶。

**坑**:
- **不收敛 / 死循环**:交织循环反复查同样的东西。必设 `max_hops` + "查不到就基于现有证据尽力答"。
- **错误累积**:某一跳推理或检索错了,后续跳都建立在错前提上,越走越偏。中间步骤最好能自评(这就引向 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 和 CRAG 的纠错)。
- **延迟与 token 累积**:每跳 = 一次 LLM + 一次检索 + 一次重排，简单单跳问题别套多跳。设第 $h$ 跳的三段延迟分别为 $t_h^{\mathrm{llm}}$、$t_h^{\mathrm{ret}}$、$t_h^{\mathrm{rank}}$，串行墙钟时间为
  $$T=\sum_{h=1}^{H}\left(t_h^{\mathrm{llm}}+t_h^{\mathrm{ret}}+t_h^{\mathrm{rank}}\right).$$
  **下列仅为示意，不是通用性能承诺**：三跳分别记录为 $(420,80,45)\text{ms}$、$(510,85,50)\text{ms}$、$(650,90,55)\text{ms}$，则 $T=(545+645+795)\text{ms}=1.985\text{s}$。同一 trace 同时记录每跳 `input_tokens`、`output_tokens`，才知道慢在模型、后端还是重排。

  若每跳都保留历史证据，令初始 prompt 为 $I_0$ token，第 $j$ 跳新增证据加输出为 $e_j+o_j$，第 $h$ 跳输入近似
  $$I_h=I_0+\sum_{j=1}^{h-1}(e_j+o_j),\qquad I_{\mathrm{total}}=\sum_{h=1}^{H}I_h.$$
  因此相同长度证据持续累积时，$I_{\mathrm{total}}$ 对跳数可能呈**超线性（近似二次）**增长，而不是简单的 $H$ 倍；摘要、去重、截断会改变这个曲线。不能只报总时延，也不能假定 token 成本必然线性。
- **上下文膨胀**:每跳的证据都堆进上下文,token 暴涨 + lost-in-the-middle。需配 [[20 上下文工程|上下文工程]],旧跳证据压缩或只留摘要。
- **FLARE 阈值难调**:置信阈值太松→几乎不检索(退化成纯生成);太紧→几乎每句都查(退化成 IRCoT 还更贵)。

## 关键事实

- 多跳检索 = **迭代/多步检索,检索与推理交织**;核心:下一跳查什么取决于已推出什么。修的是单跳 RAG 搞不定的组合/多跳问题。
- **Self-Ask**(arXiv:2210.03350, Press et al. 2022):显式自问 follow-up 子问题,可插搜索;提出"组合性鸿沟"。最贴近 [[09 ReAct|ReAct]]。
- **IRCoT**(arXiv:2212.10509, Trivedi et al. 2022):检索与 CoT **逐句交错**,双向增益;HotpotQA 等 +15~21 分。
- **FLARE**(arXiv:2305.06983, Jiang et al. 2023):前瞻生成,**低置信 token 触发**检索;最接近"自主决定是否检索"。
- 三者都是 [[36 Agentic RAG|Agentic RAG]] 的**前身/非 agent 化版本**:把检索↔推理循环写成固定流程/固定触发,而非模型自主调度。
- vs Decompose([[07 查询变换 Query Transformation|查询变换 Query Transformation]]):多跳是**串行依赖**,Decompose 是**并行列全**。
- 必备护栏:`max_hops` + 兜底答;否则不收敛。延迟按串行 hop 累积，未压缩的历史上下文会使输入 token 成本可能超线性增长。

## 工程落地与评估

将多跳实现为显式状态循环即可：`generate_step → retrieve → rerank → grade`，仅当 `grade` 判定证据不足且未触及 `max_hops` 时回到 `generate_step`。对简单问题可先走单跳路径；是否路由到多跳应由自己的离线集和 trace 数据验证，而非套用固定流量比例或框架结论。每一跳至少记录本节代码中的 query、文档 ID、`llm_ms`、`retrieve_ms`、`rerank_ms`、`input_tokens`、`output_tokens`，以定位漏召、错误前提、超时或上下文膨胀。

每跳仍可组合 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 与 [[10 重排序 Reranking|重排序 Reranking]]；工程护栏是：

- `grade` 作为提前停止信号，`max_hops` 作为硬上限，触顶后只基于已留存证据回答或明确无法确认。
- 以文档 ID / chunk ID 去重，并对历史证据做摘要或预算截断，避免同一材料重复占用上下文。
- 离线调 FLARE 阈值，并观察平均检索次数、终答正确性和 token 成本的联动；阈值不能凭直觉固定。

### Ragas：合成测试集，不是通用多跳指标

**版本化说明：以下名称对应 Ragas v0.2.12 的 testset API。** 官方的 `MultiHopAbstractQuerySynthesizer` 与 `MultiHopSpecificQuerySynthesizer`（常被口语化误写成 `MultiHopAbstractQA` / `MultiHopSpecificQA`）是从知识图谱/文本块簇**生成多跳测试问题与参考答案的 synthesizer**，不是对一次线上回答直接返回分数的通用 metric。它们用来构造「需要跨片段拼证据」的回归集；回答质量仍应由你选择并版本锁定的评估指标、金标答案和逐跳 trace 一起判断。升级 Ragas 前，需对照所安装版本的 [v0.2.12 synthesizers 文档](https://docs.ragas.io/en/v0.2.12/references/synthesizers/)；API 和默认分布均可能变化。

IRCoT 论文的评测涵盖 **HotpotQA、2WikiMultihopQA、MuSiQue、IIRC**；它们可用于方法对照，但上线回归集还应纳入自己语料的真实链式问题和失败样本。

## 面试高频

> 面试地图：[[RAG 面试题库]]

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
要点:识别为多跳依赖链 → 显式状态循环（生成下一跳 query → retrieve + rerank → grade）→ 跳间证据摘要压缩 → grade 收敛 → 触顶兜底 → 最终 grounded 生成带引用。能说出"路由判复杂度、逐跳 trace、证据去重"是加分项。

## 知识拓展

**进阶方法与前沿(带年份)**
- **IR-CoT / Self-Ask / FLARE(2022–2023)** 是第一代"交织检索"。再往后是把循环交给模型自主调度的方向:
- **Self-RAG(Asai 2023,ICLR 2024)**:把"是否检索 + 是否相关 + 是否有据"内化成反思 token,见 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]。
- **Adaptive-RAG(Jeong 2024,NAACL)**:前置复杂度分类器,把"该不该多跳"路由化——正是多跳"别默认全开"的正解。
- **Search-R1**（Jin et al., arXiv v1 2025-03-12，[原论文](https://arxiv.org/abs/2503.09516)）：用强化学习训练模型在逐步推理中发起多轮搜索。它是「让策略学会何时/查什么」的 2025 研究方向，实验设置、搜索后端、奖励和数据分布都需复现核验，**不应作为生产多跳的默认方案**。
- **DeepRetrieval**（Jiang et al., arXiv v1 2025-02-28，[原论文](https://arxiv.org/abs/2503.00223)）：以检索指标为奖励训练 LLM 生成查询，覆盖真实搜索引擎、检索器与 SQL 搜索等实验；它更直接研究「如何生成能召回的 query」，也可启发多跳的 query policy。它同样是研究方向，**不应绕过自身离线评测而默认投入生产**。
- **测试时计算(test-time compute)视角**:多跳本质是"用更多检索-推理步换准确率",和 CoT 加长、beam 加宽同属 test-time scaling;[[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]] 里"agentic search"就是多跳的极端形态。

**边界与反模式**
- **反模式一:对单跳题套多跳**——延迟 ×N、还可能因冗余检索引入噪声反而降准。多跳是把双刃剑,默认全开是典型误用。
- **反模式二:不 grade 的固定跳数**——要么提前停漏信息,要么跑满浪费;固定跳数几乎总是次优。
- **边界**:当单篇文档已含全部所需事实(真单跳),多跳通常只增加成本；依赖链很长时，先用逐跳正确率、召回率、时延和 token 成本验证是否仍有净收益，再决定是否需要为结构化关系另建索引或图遍历能力，不能用固定 hop 阈值替代评估。

**相关链接**:它是 [[36 Agentic RAG|Agentic RAG]] 的非 agent 化前身,机制上是 [[09 ReAct|ReAct]] 在检索场景的落地(Thought→retrieve→Observation);承接 [[07 查询变换 Query Transformation|查询变换 Query Transformation]](Decompose 是其并行近亲),向后接 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]];纠错思想来自 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]],上下文管理靠 [[20 上下文工程|上下文工程]]。

## 来源
- Press, Zhang, Min, Schmidt, Smith, Lewis.《Measuring and Narrowing the Compositionality Gap in Language Models》(Self-Ask). arXiv:2210.03350, 2022;EMNLP Findings 2023.
- Trivedi, Balasubramanian, Khot, Sabharwal.《Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions》(IRCoT). arXiv:2212.10509, 2022;ACL 2023.
- Jiang, Xu, Gao, Sun, Liu, Yang, Callan, Neubig.《Active Retrieval Augmented Generation》(FLARE). arXiv:2305.06983, 2023;EMNLP 2023.
- Jin et al.《Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning》. arXiv:2503.09516, v1 2025-03-12. https://arxiv.org/abs/2503.09516
- Jiang et al.《DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning》. arXiv:2503.00223, v1 2025-02-28. https://arxiv.org/abs/2503.00223
- Ragas v0.2.12 testset synthesizers 文档（2024-12-14）: https://docs.ragas.io/en/v0.2.12/references/synthesizers/
