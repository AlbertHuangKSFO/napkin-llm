[[29 Deep Research Agent|Deep Research Agent]]:给一个开放、模糊、需要多步的问题,它自主**规划 → 扇出多源搜索 → 抓取 → 交叉核验 → 综合成一份带引用的报告**。一句话——它把"人花一下午查资料写综述"这件事自动化,核心不是单次搜索,而是"先搜后读再综合"的多步研究闭环。

## 本质:为什么"搜一次"远远不够

你问 ChatGPT 一个简单事实,它搜一下答你——这是普通 RAG。但你问的是"对比近三年三种主流向量数据库的成本、性能与生态,给出选型建议"——这种问题没有单一来源能回答:它需要**拆解**(分成成本、性能、生态三个子问题)、**多源搜索**(每个子问题查多个网站/论文)、**抓取与阅读**(打开页面读正文,不只看摘要)、**交叉核验**(三个来源说的数字打架时取哪个)、**综合**(把碎片拼成有结构、有引用、能追溯的报告)。

Deep Research Agent 就是把这套"研究方法论"做成 agent 循环。它跟普通 [[36 Agentic RAG|Agentic RAG]] 的区别是**深度和编排**:普通 Agentic RAG 通常是"检索-生成"一两轮;Deep Research 是规划驱动的多轮、多路、可迭代加深的研究,常常跑几分钟、查几十上百个网页。它天然是 [[22 多智能体系统|多智能体系统]] 的好场景(一个 orchestrator + 多个研究子 agent),也天然要用 [[06 Parallelization|Parallelization]](N 个子问题并行查)。

## 机制:plan → 扇出 → 核验 → 综合(分步)

![[Deep Research Agent.png]]

标准管线分四步,且第三步可回灌迭代:

1. **规划(Plan)**:把开放问题拆成若干**子问题 / 研究角度 / outline**。好的规划是成败关键——拆得全且不重叠,后面才查得到位。有的实现(STORM)更进一步,先生成"多个不同视角的提问者"去逼问,模拟专家从不同角度调研。这一步是 [[10 Plan-and-Execute|Plan-and-Execute]] 思想的应用。
2. **扇出多源搜索 + 抓取(Search & Fetch)**:每个子问题**并行**发起搜索(联动 [[06 Parallelization|Parallelization]]),拿到候选 URL 后**真的把页面打开抓正文**(fetch),而不是只读搜索摘要——这是"deep"的关键,摘要往往不够、会误导。对应"先搜后读再综合"里的"搜"和"读"。在某些代码型实现里,这一步用的搜索甚至复用了 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 的思路。
3. **交叉核验(Verify)**:把多个来源的信息对照——去重、剔除低质量/过时来源、发现矛盾时判断谁更可信。这一步是对抗幻觉的护城河:只有被多源支撑的结论才进报告。发现覆盖有缺口,就**回灌补一轮检索**(可迭代加深,图中虚线回路)。
4. **综合成文(Synthesize)**:把核验过的碎片组织成**结构化报告**,关键一点是**每条结论都挂引用 [1][2]**,做到可追溯。对应"先搜后读再综合"里的"综合"。综合阶段要做 [[21 上下文压缩与卸载|上下文压缩与卸载]]——抓来的原文动辄几十万 token,不能全塞进窗口,要先摘要/筛选再喂给写作模型。

整个流程就是 [[03 Agent 核心循环|Agent 核心循环]] 套上"研究"这个特定 playbook:规划用 [[10 Plan-and-Execute|Plan-and-Execute]],并行用 [[06 Parallelization|Parallelization]],核验是内建的反思,综合靠上下文工程。

## 来源:产品与时间线

- **OpenAI Deep Research**(2025-02):ChatGPT 内的深度研究模式,基于 o 系列推理模型,自主多步搜网、读页、综合,产出长篇带引用报告;把"Deep Research"这个产品品类带火。
- **Google Gemini Deep Research**(2024-12 起):Gemini 内的深度研究,会先给你一份**研究计划**让你确认,再扇出执行,最后出报告。
- **Perplexity**(及其 Deep Research 模式):以"答案即引用"起家,深度研究模式做多步检索综合,主打来源透明。
- 共同点:都把"搜一次直接答"升级成"规划-多轮检索-核验-带引用综合",且都强调**引用可追溯**作为区别于普通聊天的卖点。

## 可跑最小代码:gpt-researcher

```python
# pip install gpt-researcher
import asyncio
from gpt_researcher import GPTResearcher

async def main():
    researcher = GPTResearcher(
        query="对比 2024-2026 主流开源向量数据库的成本/性能/生态,给选型建议",
        report_type="research_report",   # 也有 detailed_report / outline 等
    )
    await researcher.conduct_research()  # 规划→并行搜索→抓取→核验
    report = await researcher.write_report()  # 综合成带引用的报告
    print(report)

asyncio.run(main())
```

### Deep Research 内核伪码

```python
subqs = planner.plan(question)                 # ① 拆成 N 个子问题
results = parallel_map(research_one, subqs)     # ② 并行扇出(Parallelization)

def research_one(q):
    urls = web_search(q)                        # 搜
    docs = [fetch_and_extract(u) for u in urls] # 读:真打开页面抓正文
    return summarize_with_sources(docs)         # 带来源的小结

facts = cross_verify(results)                   # ③ 多源对照,去重剔矛盾
if has_gaps(facts):                             # 有缺口
    subqs += planner.followups(facts)           #   补一轮(迭代加深)
    ...                                         #   再跑一遍 research_one
report = synthesize(question, facts, cite=True) # ④ 综合 + 挂引用 [1][2]
return report
```

## 对比表:几种开源 Deep Research 的取向

| 项目 | 出品 / 仓库 | 取向 | 输出形态 |
|---|---|---|---|
| gpt-researcher | assafelovic / gpt-researcher | 最成熟,UX+可定制完善,多 agent 协作 | 5-6 页报告(MD/PDF/Docx) |
| STORM | stanford-oval / storm | 学术,多视角提问 + 大纲驱动 | 维基百科式长文 |
| open_deep_research | langchain-ai / open_deep_research | 生产级,supervisor + 隔离子 agent | 可配置多源报告 |
| open-deep-research (smolagents) | huggingface / smolagents | 代码型 CodeAgent,能即兴写代码取数 | 灵活,工具驱动 |
| local-deep-researcher | langchain-ai / local-deep-researcher | 全本地(Ollama),隐私优先 | 本地报告 |

## 何时用 / 坑

**何时用**:问题**开放、多步、需要综合多个来源**且要**可追溯引用**时(市场调研、技术选型、文献综述、尽调)。如果只是查个单一事实或单源就能答,用普通搜索或 [[36 Agentic RAG|Agentic RAG]] 就够,Deep Research 是杀鸡用牛刀(慢且贵)。

**坑**:

- **幻觉引用**:最致命的坑——模型给结论挂了一个引用,但引用其实**不支持**该结论,甚至编造 URL。必须做引用回链校验(结论→原文片段能对上),核验阶段不能省。
- **搜索深度 vs 成本的权衡**:查得越深(更多子问题、更多页、更多迭代)越准,但 token 和时间线性甚至超线性增长。要设**预算上限**(最多 N 个来源、最多 M 轮迭代),见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- **来源质量参差**:搜索引擎前几条不一定权威(SEO 垃圾、内容农场、过时页)。核验要会**给来源评级**,优先一手/权威源。
- **上下文爆炸**:抓来的全文塞不进窗口,必须 [[21 上下文压缩与卸载|上下文压缩与卸载]]——先摘要/抽取再综合,否则要么截断丢信息,要么爆 token。
- **规划偏了全盘偏**:子问题拆错或漏了关键角度,后面查得再勤也补不回来。规划质量是上限。
- **时效与一致性**:多路并行抓取的页面时间戳不一,可能拿到互相矛盾的旧新数据,核验时要考虑时效。

## 关键事实

- Deep Research 的"深"= **真抓取页面正文 + 多轮迭代加深**,不是单次搜索;这是它区别于普通 [[36 Agentic RAG|Agentic RAG]] 的本质。
- 流程对应口诀"先搜后读再综合":搜=扇出检索,读=fetch 抓正文,综合=带引用成文。
- **并行扇出**(见 [[06 Parallelization|Parallelization]])是性能关键:N 个子问题串行查会慢到不可用。
- **引用可追溯**是产品级 Deep Research 的硬指标,也是评估重点(引用是否真支撑结论),评估方法见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- 天然适合 [[22 多智能体系统|多智能体系统]]:orchestrator 负责规划与综合,多个研究子 agent 各查一个子问题(context 隔离,互不污染)。
- gpt-researcher 受 STORM 论文启发;两者取向不同——前者偏开发者向、深度研究报告,后者偏维基式知识策展。

## 主流开源实现 / Python 库

> 均已 web 核验为 2026 年活跃项目。

- **gpt-researcher**(`assafelovic/gpt-researcher`,pip `gpt-researcher`):最成熟的开源深度研究 agent,多 agent 协作从规划到成文,平均产出 5-6 页带引用报告(MD/PDF/Docx),UX 与可定制性最完善。首选。
- **STORM**(`stanford-oval/storm`,pip `knowledge-storm`):斯坦福 OVAL 出品,先用多视角提问者逼问、生成大纲,再写维基百科式长文;Co-STORM 支持人机协作策展。学术取向。
- **open_deep_research**(`langchain-ai/open_deep_research`):LangChain 基于 LangGraph 的生产级开源深研框架,supervisor agent 把研究简报拆给若干 context 隔离的子 agent,MIT 许可,可跨多模型/搜索工具/MCP。
- **smolagents open-deep-research**(`huggingface/smolagents`,examples/open_deep_research,pip `smolagents`):HuggingFace 的代码型实现,用 CodeAgent 即兴写代码取数,工具驱动、灵活。
- **local-deep-researcher**(`langchain-ai/local-deep-researcher`):全本地的网页研究助手,配合 Ollama 等本地 LLM,隐私优先。

链接:[[22 多智能体系统|多智能体系统]]、[[36 Agentic RAG|Agentic RAG]]、[[06 Parallelization|Parallelization]]、[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]、[[10 Plan-and-Execute|Plan-and-Execute]]、[[03 Agent 核心循环|Agent 核心循环]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]、[[38 Agent 评估与可观测性|Agent 评估与可观测性]]。跨域可链 [[RAG]]。

## 工业界实践

把"Deep Research"从 demo 做成生产服务,工业界踩出来的几条主线:

**① 平台化:从"聊天里的一个模式"到"API + 托管模型"。**
- **OpenAI Deep Research API**(2025-06 起):不再只在 ChatGPT 里跑,而是开放了**专用模型** `o3-deep-research`(强,贵)和 `o4-mini-deep-research`(快,便宜)两档,带快照版本号(如 `o3-deep-research-2025-06-26`)。调用走 **Responses API**,且**强制至少挂一个数据源**:`web_search`(联网)、远程 **MCP server**(接私有数据,见 [[17 MCP 模型上下文协议|MCP 模型上下文协议]])、或 `file_search` + 向量库。这把"deep research = 一个会用工具的推理模型"这件事产品化了——模型自己决定搜几轮、抓哪些页、何时停。
- **Azure AI Foundry Deep Research**(微软):把 OpenAI 的 `o3-deep-research` 包成 Foundry Agent Service 里的托管工具,企业可在自己租户内合规调用,接 Bing/私有索引。
- **Google Gemini Deep Research**、**Perplexity**:产品侧两大对照。Gemini 的招牌动作是**先出"研究计划"让用户确认再执行**(把规划暴露成可干预的交互);Perplexity 主打"答案即引用"的来源透明。

**② 典型生产架构 = orchestrator + 隔离 worker + 写作器。** 主流是 [[07 Orchestrator-Workers|Orchestrator-Workers]] 形态(LangChain `open_deep_research` 的 supervisor 模式即此):一个主 agent 把研究简报拆成若干子任务,扇给**各自 context 隔离**的子 agent(每个只看自己那一摊,避免上下文互相污染),子 agent 各自搜+读+小结,主 agent 再做交叉核验与综合。隔离是关键:几十个网页的原文若全塞一个窗口,既爆 token 又互相干扰,见 [[21 上下文压缩与卸载|上下文压缩与卸载]]、[[20 上下文工程|上下文工程]]。

![[DeepResearch-生产架构.png]]

**③ 规模化、成本与延迟。** Deep Research 是 Agent 里**最烧钱、最慢**的形态之一:一次任务常跑几分钟、几十到上百次工具调用、几十万 token。工程上压成本/延迟的手段:
- **预算护栏**:硬上限——最多 N 个来源、最多 M 轮迭代、最长 T 分钟;超了就停并用已有材料成文。
- **分层模型**:规划/综合用强模型,子问题的"搜+读小结"用便宜小模型(典型 mini 档),省 80% 成本。
- **并行扇出**:N 个子问题必须并行(见 [[06 Parallelization|Parallelization]]),串行查会慢到不可用;但要给搜索/抓取并发设上限,别打爆下游 API 配额。
- **抓取去重与缓存**:同一 URL 在多个子问题里重复出现,缓存正文;搜索结果做语义去重。

**④ 可观测与运维。** Deep Research 的 trace 极长(多 agent × 多轮 × 多工具),没有 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 几乎没法 debug。生产标配:用 LangSmith / Langfuse / Phoenix 记录完整树状 trace(哪个子 agent、查了哪些 URL、引用挂在哪);对**引用忠实度**做离线评测(每条结论能否回链到原文片段);监控每任务的 token/时延/工具调用数分布,抓长尾爆炸的 case。

**⑤ 踩坑与最佳实践。**
- **引用回链校验**做成硬性后处理:综合完一遍,逐条结论去原文里 grep 支撑片段,挂不上的标"未核实"或删——这是对抗幻觉引用最有效的一招。
- **来源评级**:给域名/来源打权威度先验(一手文档、论文、官方 > 内容农场/SEO 垃圾),核验冲突时按级取。
- **规划可干预**(学 Gemini):长任务先把 outline 给人看一眼再执行,省得方向跑偏后白烧十分钟。
- **时效一致性**:并行抓的页面时间戳不一,数字打架时优先新的,并在报告里标注口径与日期。

```python
# 生产化 Deep Research 的几个关键护栏(伪配置)
config = {
    "planner_model": "o3",              # 规划/综合用强模型
    "worker_model":  "o4-mini",         # 子问题搜+读用便宜模型
    "max_sources":   40,                # 来源数硬上限(控成本)
    "max_iterations": 3,                # 迭代加深轮数上限
    "concurrency":   8,                 # 并行扇出上限(别打爆下游配额)
    "require_citation_check": True,     # 综合后逐条结论回链校验
    "source_ranking": "authority_prior" # 来源权威度评级
}
```

## 面试高频

**Q1:Deep Research 和普通 RAG / Agentic RAG 到底差在哪?**
标准答:普通 RAG 是"检索一次→生成";Agentic RAG 多了一两轮"检索-反思-再检索";Deep Research 是**规划驱动的多轮、多路、可迭代加深**的研究闭环,核心两点——**真抓取页面正文**(不只读搜索摘要)+ **交叉核验后带引用成文**。一句话口诀:"先搜后读再综合"。追问"为什么一定要 fetch 正文?":摘要常不全/误导,数字、口径、前提都在正文里,只读摘要会得出错结论。

**Q2:Deep Research 的护城河 / 最致命的坑是什么?**
答:**幻觉引用**——模型给结论挂了引用,但引用并不支撑该结论,甚至编造 URL。这是它区别于"好玩 demo"和"能用产品"的分水岭。解法:核验阶段做**引用回链校验**(结论→原文片段能对上才保留),并对来源评级。追问"怎么自动评测引用质量?":离线用 LLM-as-judge 逐条判"该引用是否支撑该句",算 citation faithfulness / precision,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**Q3:让你设计一个 Deep Research 系统,怎么控成本和延迟?**
答:分层模型(强模型规划综合 + 便宜模型搜读)、并行扇出但设并发上限、预算护栏(来源数/迭代轮数/时长上限)、URL 缓存与去重、可迭代加深但"有缺口才补轮"。追问"并行扇出为什么是性能关键?":N 个子问题串行查,时延线性叠加到几分钟以上不可用;并行后约等于最慢一路。陷阱:有人答"全量抓取更准"——会爆 token 且超线性涨成本,正确做法是预算约束下的有界搜索。

**Q4:多 agent 还是单 agent 做 Deep Research?**
答:天然适合 [[22 多智能体系统|多智能体系统]]——orchestrator 规划+综合,多个子 agent 各查一个子问题且 **context 隔离**(互不污染、可并行)。但要权衡:多 agent 协调开销大、token 翻倍,简单研究单 agent 多轮也够。陷阱:为多 agent 而多 agent,子任务拆得过细反而内耗。

**Q5(陷阱):Deep Research 慢主要慢在哪?能不能靠加大模型解决?**
答:慢在**串行的工具调用链**(搜→抓→读→再规划),不是单次推理。加大模型只会更慢更贵。正解是**并行化扇出** + 有界迭代 + 分层模型,这是架构问题不是模型问题。

## 知识拓展

- **STORM / Co-STORM(Stanford OVAL,2024)**:`stanford-oval/storm`。先生成"多个不同视角的提问者"互相对话逼问、再生成大纲、最后写维基式长文;Co-STORM 引入人机协作策展。gpt-researcher 受其启发但更偏开发者向、深度报告。论文 *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models*(NAACL 2024)。
- **测试时计算(test-time compute)视角**:Deep Research 本质是把推理算力花在"多搜多读多核验"上,与推理模型(o 系列、R1)"多想几步"是同一哲学的不同载体——都是用更多 token 换更高质量,见 [[32 Agentic RL 与训练|Agentic RL 与训练]] 里 RLVR 训出来的长推理。
- **协议视角**:私有数据接入正从一次性集成转向 **MCP**(见 [[17 MCP 模型上下文协议|MCP 模型上下文协议]]);Deep Research API 已支持挂远程 MCP server 当数据源。跨组织调度研究子 agent 则可走 [[30 A2A 协议|A2A 协议]]。
- **评估前沿**:研究型 agent 的 benchmark 在快速演进——除答案正确性,更看**引用忠实度、来源覆盖度、报告结构化程度**。学界有 BrowseComp(OpenAI,2025,极难的网页检索题集)、GAIA(通用助手任务)等衡量"会不会真做研究"。
- **边界 / 反模式**:① 单一事实查询用 Deep Research = 杀鸡用牛刀(慢且贵),退回普通搜索或 [[36 Agentic RAG|Agentic RAG]];② 规划拆错则全盘错,规划质量是上限(garbage plan in, garbage report out);③ 无核验直接综合 = 把幻觉规模化;④ 不设预算 = 成本/时延长尾爆炸。
- **相关兄弟**:[[10 Plan-and-Execute|Plan-and-Execute]](规划骨架)、[[06 Parallelization|Parallelization]](扇出)、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](迭代加深的反思内核)、[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]](子 agent 隔离)、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]](预算护栏)。
