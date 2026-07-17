[[13 Modular RAG|Modular RAG]] 是把 RAG 系统**拆成可插拔的乐高积木**的设计范式:不再把它当成「检索→生成」一条写死的直线,而是拆成一组**独立模块(modules)+ 可替换算子(operators)**,再由编排层(orchestration)按需把它们拼成不同的流程拓扑。它给工程的价值不是承诺更高分,而是把查询变换、混合检索、重排序、自评纠错和 [[19 Agent 记忆系统|记忆]] 放进可替换、可观测、可评测的边界里。

Gao et al. 2024 的《Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks》(arXiv:2407.21059)用**modules 与 specialized operators**描述这种可重构性，并讨论 **linear、conditional、branching、looping** 四类 RAG flow。本页据此给出一套便于实现和评测的工程读法；后面的六层并非该论文规定的一组固定模块清单。

![[Modular RAG-乐高.png]]

### 直觉：乐高盒与交通调度

把算子想成有统一插口的乐高：`rewrite`、`search`、`rerank`、`generate` 都能替换；把 flow 想成交通调度：直行是 linear，岔路择一是 conditional，多车齐发后汇合是 branching，绕回上一站补材料是 loop。**模块回答“做什么”，flow 回答“按什么控制关系做”**；Memory 则是所有站点都必须按权限领取、归还和清理的物料柜。

## 机制

### 从 Naive 到 Advanced 到 Modular 的三段演进

理解 Modular RAG，先看它在「演进谱系」里的位置(这套 Naive / Advanced / Modular 三分法源自同一作者团队更早的 RAG 综述):

- **Naive RAG**:最朴素的一条直线——`索引 → 检索一次(top-k)→ 拼进 prompt → 生成`。死穴一堆(召回差也将就、不重写 query、搞不定多跳……),详见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。对应 flow 就是纯 **linear(线性)**。
- **Advanced RAG**:在直线两端加料补救——**检索前(pre-retrieval)**做查询变换/路由([[07 查询变换 Query Transformation|查询变换 Query Transformation]]),**检索后(post-retrieval)**做重排序、压缩、过滤([[10 重排序 Reranking|重排序 Reranking]])。流程仍是一条加长的直线,只是首尾更讲究。
- **Modular RAG**:**打破直线**。把整个系统抽象成**模块化的组件 + 可任意编排的 flow**,允许条件分支、并行分支、循环回路。Advanced RAG 的 pre/post 处理,在这里只是 Modular 框架的一个特例(linear flow)。

一句话:**Naive 是直线、Advanced 是加料的直线、Modular 是可重构的图**。

### 本页采用的六层工程分层

为让模块边界能落到代码、指标与故障排查，**本页**把常见 RAG 管线整理成下面六层工程分层。它是实现用的检查清单，不应误读成 Gao 论文钦定的“六个模块”；同一产品也可按数据源、合规边界或团队职责另行拆分。

| 模块 | 干什么 | 可换的算子举例 | 关联笔记 |
|---|---|---|---|
| **Indexing 索引** | 把语料切块、编码、建库 | 分块策略 / 层级索引 / 稀疏+稠密 | [[03 分块策略 Chunking\|分块策略 Chunking]]、[[04 Embedding 与向量数据库\|Embedding 与向量数据库]] |
| **Pre-retrieval 检索前** | 加工 query | 改写 / 分解 / 扩展 / 路由 | [[07 查询变换 Query Transformation\|查询变换 Query Transformation]]、[[05 Routing\|Routing]] |
| **Retrieval 检索** | 召回候选 | 向量 / 混合 / 多源融合 | [[08 混合检索 Hybrid Search\|混合检索 Hybrid Search]] |
| **Post-retrieval 检索后** | 加工召回结果 | 重排序 / 压缩 / 过滤 / 精炼 | [[10 重排序 Reranking\|重排序 Reranking]] |
| **Generation 生成** | 据证据生成答案 | 受约束生成 / 引用归因 / 忠实校验 | [[11 生成层：引用归因与忠实度\|生成层：引用归因与忠实度]] |
| **Orchestration 编排** | 调度上面五层、决定走哪条 flow | 路由 / 调度 / 条件 / 循环控制 | —— |

### Memory 是一等考虑，而非隐式全局变量

**Memory 不塞进第七个模糊大桶**，而是横切六层的一等数据契约：它既可给 Pre-retrieval 提供会话上下文，也可像另一检索源一样进入 Retrieval/Post-retrieval 的候选集合，还必须由 Orchestration 明确决定何时读、何时写、何时失效。每次读写至少记录 `namespace/user`、来源、时间/TTL、权限、版本与删除路径；不要把全部聊天记录或用户资料无筛选地拼进 prompt。

工程上可把它分成两种作用域：**短期记忆**保存单个会话/任务的状态，支持中断恢复；**长期记忆**保存跨会话且经授权的偏好、事实或摘要。两者都应先经检索、过滤和权限检查，再与外部文档同样接受重排、引用和评测。这样 Memory 才是可替换的模块接口，而不是绕过检索质量与数据治理的后门，相关设计见 [[19 Agent 记忆系统|Agent 记忆系统]] 与 [[16 检索安全与访问控制|检索安全与访问控制]]。

模块化的好处是**可替换、可复用、可单独测量**:想升级召回可替换 Retrieval 算子；想加忠实度校验可在 Generation 后插入检查；但改 Indexing 的分块策略，仍可能需要连带调 Retrieval 与 Post-retrieval。解耦是设计目标，不是“换一块就必然不用复测”的保证。

### 四种 RAG Flow 模式

Gao et al. 用下列四类 flow 描述常见的编排拓扑；它们关注的是节点之间的控制关系，不规定节点一定叫什么或必须使用哪一个框架:

1. **Linear(线性)**:模块顺序串联,一条过。`pre → retrieve → post → generate`。**Naive RAG 和 Advanced RAG 都是 linear**,只是模块多少之差。
2. **Conditional(条件)**:中间有个判别节点,**按条件选不同后续路径**。典型就是 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 里的 **Adaptive-RAG**——复杂度分类器把 query 路由到「不检索 / 单步 / 多步」三条路;[[05 Routing|Routing]] 也属此类。
3. **Branching(分支)**:**多条路并行跑、再汇聚**。比如把 query 分解成多个子查询、各自检索、结果聚合;或同一 query 走多个检索源并行召回再融合(对应 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 的并行召回 + RRF 融合思路)。
4. **Loop(循环)**:**带反馈回边,可反复迭代**——检索 → 生成/自评 → 不满意就回去重写 query 再检索,直到收敛。这是四种里最强的,**Loop pattern 就是 [[36 Agentic RAG|Agentic RAG]] 的雏形**:CRAG 的「召回差→纠错再检索」、Self-RAG 的「ISSUP 不通过→重生成」、多跳检索的 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 迭代,全是 loop。再往前一步——把 retrieve 做成 agent 能自主反复调用的工具——就跨进 Agentic RAG。

可以这样串记:**Naive=linear,Adaptive/路由=conditional,多查询/多源=branching,自评纠错/多跳=loop(→ Agentic RAG)**。

**RAG 范式 / flow 选型卡**(按查询形态查):

| 查询形态 | 选哪种 flow / 范式 | 为什么 |
|---|---|---|
| 简单事实问答、查询模式单一 | **Linear**(Naive/Advanced) | 一条直线最省心,过度模块化只增延迟和复杂度;先跑通基线 |
| 简单/复杂混流、要按难度分路 | **Conditional**([[05 Routing\|Routing]]/Adaptive-RAG) | 复杂度分类器把简单 query 分流走快路径,只对难题开多轮,省成本 |
| 一个问题含多个可拆子问 / 多源 | **Branching**(子查询并行 + RRF 融合) | 无依赖分支并行铺开一次汇总,比串行快;注意并行执行别串行 |
| 召回质量参差、需自评纠错 / 多跳 | **Loop**(CRAG/Self-RAG/IRCoT) | 带回边反复改写再查直到收敛,是 [[36 Agentic RAG\|Agentic RAG]] 雏形;必设 `max_iter` |
| 全局/汇总型问题(整库主题) | **GraphRAG** | 向量召回只取局部 top-k,答不了全语料汇总,见 [[14 GraphRAG 知识图谱检索\|14]] |
| 流程要 agent 运行时自主决定 | **Agentic RAG** | 控制权从「预编排图」交给 LLM,更灵活但更难调、成本飘,见 [[36 Agentic RAG\|36]] |

**branching vs loop 别混(一个具体例子)**。同一个 query「比较 A 和 B 公司的营收与员工数」:**branching** 把它拆成 2 个子查询(「A 的营收员工」「B 的营收员工」)**同时并行**各检索一次,4 路结果**一次性汇聚**——分支之间无依赖、无回头,DAG 没有回边。**loop** 则是同一个 query **改写 3 次串行迭代**:第 1 轮查得不够 → 自评「证据不足」→ 改写 query → 第 2 轮再查 → 仍不够 → 再改 → 第 3 轮……带**回边**反复跑同一条路,直到收敛或撞 `max_iter`。一句话:**branching 是"并行铺开、一次汇总"(无回边),loop 是"串行试错、反馈重来"(有回边)**。

### 小数字手算：并行分支省在哪里

假设一个查询拆成 4 个相互独立的子查询，每路远程检索耗时 120 ms，汇聚重排耗时 20 ms。若串行执行，时间是 $4\times120+20=500\text{ ms}$；若四路确实并发，理想时间是 $\max(120,120,120,120)+20=140\text{ ms}$，少等 $360\text{ ms}$，约为 $500/140\approx3.57\times$ 加速。这个数字只说明**等待可重叠**：共享限流、CPU 密集重排或下游拥塞都会吃掉收益。

### 公式：由依赖关系决定延迟

令第 $i$ 个独立检索分支耗时为 $t_i$，汇聚耗时为 $t_m$。串行实现必须累加：

$$
T_{\mathrm{serial}}=\sum_{i=1}^{n}t_i+t_m
$$

无依赖、资源允许并发时，等待最长分支即可：

$$
T_{\mathrm{branch}}\approx\max_{1\le i\le n}t_i+t_m
$$

因此 `asyncio.gather` 是实现 branching 的一种手段，不是定义本身；只要分支相互独立且在汇聚前不需要彼此结果，就可用异步任务队列、线程池或工作流引擎实现同一拓扑。

## 可跑最小代码

下面是**不依赖第三方包、可直接运行**的教学 mock。真实系统只需把 `rewrite`、`split_query`、`search`、`answer` 换成带超时、重试与权限校验的适配器；`memory` 仍应先被选择和过滤，而不是直接把全量历史塞入 prompt。

完整示例同时展示 modules 的统一 state 接口、Memory 的显式读入，以及四种 flow：

```python
import asyncio
from typing import TypedDict


class State(TypedDict, total=False):
    raw: str
    query: str
    docs: list[str]
    memory: list[str]
    branch_docs: list[str]
    loop_attempts: int
    trace: list[str]
    answer: str


MEMORY = {"demo": ["用户偏好：中文、要引用来源"]}


async def rewrite(query: str) -> str:
    await asyncio.sleep(0.001)  # 教学 mock：真实场景可换成 LLM 调用
    return query.strip()


async def split_query(query: str) -> list[str]:
    await asyncio.sleep(0.001)
    return [f"{query}：子问题 A", f"{query}：子问题 B"]


async def search(query: str) -> list[str]:
    await asyncio.sleep(0.01)  # 模拟 I/O；分支应并发等待
    # loop 的首轮故意只返回一条外部证据；改写后的第二轮才补足两条
    if "第 1 次改写" in query:
        return [f"检索证据<{query}>#1", f"检索证据<{query}>#2"]
    return [f"检索证据<{query}>#1"]


def rerank(query: str, docs: list[str], k: int = 3) -> list[str]:
    return list(dict.fromkeys(docs))[:k]  # 去重只是 mock，不代表真实排序器


def answer(state: State) -> State:
    state["answer"] = f"Q: {state['raw']}\n证据: {state['docs']}"
    return state


# ---- modules：统一吃/吐 State；Memory 也以显式模块读入 ----
async def m_memory(state: State, namespace: str = "demo") -> State:
    state["memory"] = MEMORY.get(namespace, []).copy()
    return state


async def m_pre(state: State) -> State:
    state["query"] = await rewrite(state["raw"])
    return state


async def m_retrieve(state: State) -> State:
    state["docs"] = await search(state["query"])
    return state


def m_post(state: State) -> State:
    # memory 是候选上下文，和外部证据一起经过同一选择边界
    state["docs"] = rerank(state["query"], state["memory"] + state["docs"])
    return state


async def prepare(state: State) -> State:
    return m_post(await m_retrieve(await m_pre(state)))


# ---- Flow 1：linear ----
async def flow_linear(state: State) -> State:
    return answer(await prepare(await m_memory(state)))


# ---- Flow 2：conditional ----
async def flow_conditional(state: State) -> State:
    if len(state["raw"]) < 12:  # mock 路由器；真实场景须离线评测阈值
        state["answer"] = f"直接答: {state['raw']}"
        return state
    return await flow_loop(state)


# ---- Flow 3：branching；gather 是真正并发，而非串行 for ----
async def flow_branching(state: State) -> State:
    state = await m_memory(state)
    sub_queries = await split_query(state["raw"])
    # ❌ 不要写成：for sub in sub_queries: docs.extend(await search(sub))
    # ✅ gather 同时提交所有独立 I/O；返回结果顺序与 sub_queries 一致
    branch_results = await asyncio.gather(*(search(sub) for sub in sub_queries))
    state["branch_docs"] = [doc for docs in branch_results for doc in docs]
    state["query"] = state["raw"]
    # 教学示例保留全部合并结果，避免 top-k 截断掩盖 B 分支已经并发完成。
    state["docs"] = rerank(state["query"], state["memory"] + state["branch_docs"], k=5)
    return answer(state)


# ---- Flow 4：loop；有回边，必须有限制 ----
async def flow_loop(state: State, max_iter: int = 3) -> State:
    state = await m_memory(state)
    state["trace"] = []
    for attempt in range(max_iter):
        state = await prepare(state)
        state["loop_attempts"] = attempt + 1
        external_docs = [doc for doc in state["docs"] if doc not in state["memory"]]
        evidence_enough = len(external_docs) >= 2  # Memory 不计作本轮检索证据
        if evidence_enough:
            state["trace"].append(f"retrieve:{attempt}:sufficient")
            break
        state["trace"].append(f"retrieve:{attempt}:insufficient")
        state["raw"] = f"{state['raw']}（第 {attempt + 1} 次改写）"
        state["trace"].append(f"rewrite:{attempt + 1}")
    return answer(state)


async def main() -> None:
    branching = await flow_branching({"raw": "比较 A 与 B 的营收"})
    assert any("子问题 A" in doc for doc in branching["branch_docs"])
    assert any("子问题 B" in doc for doc in branching["branch_docs"])
    assert any("子问题 A" in doc for doc in branching["docs"])
    assert any("子问题 B" in doc for doc in branching["docs"])

    looping = await flow_loop({"raw": "需要迭代的复杂问题"}, max_iter=2)
    assert looping["loop_attempts"] == 2
    assert looping["trace"] == ["retrieve:0:insufficient", "rewrite:1", "retrieve:1:sufficient"]
    assert looping["answer"]
    print(branching["answer"])
    print(looping["trace"])


if __name__ == "__main__":
    asyncio.run(main())
```

要点:① `m_memory`、`m_pre`、`m_retrieve`、`m_post` 都围绕同一个 `State` 交换数据，算子可替换但契约不变；② `flow_branching` 的 `asyncio.gather` 会同时提交独立检索任务，不能写成逐个 `await` 的 `for` 循环；③ `flow_loop` 的回边有 `max_iter` 护栏；④ 把检索做成由模型在运行时决定何时、如何调用，才会接近 [[36 Agentic RAG|Agentic RAG]] 的控制方式，不能把所有 loop 一概等同于 agent。

## 何时用 / 坑

**该上 Modular RAG 思维**:系统要长期演进、需要灵活替换组件(今天换个 reranker、明天加个网搜源)、或要支持多种查询模式(简单/复杂混流)。把 RAG 当乐高拆,比写死一条管线更可维护、可实验(A/B 不同算子)。

**不必上**:一次性 demo、查询模式单一且稳定的场景——直接 linear 一条龙更省心,过度模块化是负担。

**坑**:
- **过度工程化**:不是所有 RAG 都需要 conditional/branching/loop。简单事实问答 linear 足矣,硬上模块化和 flow 编排只增加复杂度和延迟。先 Naive 跑通,有明确瓶颈再模块化。
- **flow 越复杂越难调试**:branching/loop 的失败点散在多个分支/多轮里,定位 bug 比直线难得多。需要 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 级别的 trace/日志。
- **loop 不收敛**:和所有带回边的系统一样,自评一直说「不够」会反复空转。`max_iter` 是必备护栏(同 [[36 Agentic RAG|Agentic RAG]] 的最大跳数)。
- **模块边界看似清晰、实则耦合**:换 Indexing 的分块策略,往往得连带调 Retrieval 和 Post-retrieval(块变小→召回数要变→重排策略要改)。模块解耦是理想,实践中相邻模块常牵一发动全身。
- **「模块化」≠「自动变好」**:Modular RAG 是组织框架,不是性能银弹。每个模块里塞的算子(用哪个 embedding、哪个 reranker)才决定效果——框架只负责让你好换好试。

## 关键事实

- Modular RAG = 把 RAG 拆成**可插拔模块(modules)+ 可替换算子(operators)+ 可编排 flow**,从「写死的直线」升级为「可重构的图」(LEGO-like)。
- 演进谱系:**Naive(纯 linear,见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]])→ Advanced(加 pre/post 的 linear)→ Modular(模块化 + 任意 flow)**;Advanced 是 Modular 的 linear 特例。
- **本页的六层工程分层**是 Indexing / Pre-retrieval / Retrieval / Post-retrieval / Generation / Orchestration；它不是 Gao 论文规定的固定清单，Orchestration 负责调度其余五层、决定走哪条 flow。
- [[19 Agent 记忆系统|Memory]] 是横切的一等数据契约：短期状态与长期记忆应分作用域，读写必须可授权、可追溯、可过期和可删除，并与外部证据一样接受选择和评测。
- 四种 flow:**Linear(=Naive/Advanced)、Conditional(=[[05 Routing|Routing]]/Adaptive 路由)、Branching(多子查询/多源并行汇聚)、Loop(自评纠错/多跳迭代)**。
- Loop 是带回边的预编排流程；当模型在运行时决定是否、何时以及如何调用检索/工具时，才更接近 [[36 Agentic RAG|Agentic RAG]] 的控制方式。
- 价值是**工程组织 + 可实验性**(好换好试),不是性能银弹;实际效果取决于各模块里塞的具体算子。
- 无依赖 branching 应真并发：$T_{\mathrm{branch}}\approx\max_i t_i+t_m$，不能把逐个 `await` 的 `for` 循环误称为并行；loop 必有次数、时间或成本护栏。
- LangGraph 只是可选实现之一；本文使用的版本化参考是 Python `langgraph==1.2.9`（2026-07-10），见来源。

## 工程落地

Modular RAG 不是某个产品或框架的同义词。先为每个节点定义输入/输出 schema、超时、重试、权限和评测指标，再把条件、并行和回边写成可审计的拓扑；实现可以是普通 `asyncio` 任务、队列消费者、状态机或图工作流引擎。

### 一个 Corrective / Adaptive 的流程草图

下图只是把 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 的评估与重写步骤表达成图，不暗示它是唯一或默认架构：

```
        ┌─────────────┐
query → │ route(分类) │──simple──→ generate(直接答)
        └─────┬───────┘
         single/multi
              ↓
      ┌───────────────┐    docs    ┌──────────────┐
      │   retrieve    │──────────→ │ grade_docs   │ 给每篇打相关分
      └───────┬───────┘            └──────┬───────┘
              ↑ rewrite                   │ 相关不足
              └──── transform_query ←──────┤ (conditional edge)
                                          │ 相关够
                                          ↓
                                     ┌──────────┐
                                     │ generate │→ hallucination/answer 检查
                                     └────┬─────┘ 不通过→回 retrieve/rewrite
                                          ↓ 通过
                                        answer
```

- `grade_documents` 节点可承载 CRAG 的相关性评估；`transform_query` 回边表达 loop 的重写再查；`generate` 后的检查可承载 Self-RAG 式的证据/答案评估。它展示了 linear 主干上叠加 conditional 与 loop；如有独立子问题，还可在检索前增加 branching 后再汇聚。

### Memory、并发与可观测性

- **Memory**:将读取、筛选、写回、删除记录为独立事件。把跨会话长期记忆与本次工作流状态分开存放，并在 trace 中留下 namespace、权限判定和最终被使用的记忆 ID，避免“回答看似有根据，实际来自不可追溯历史”。
- **并发与护栏**:只有无依赖分支才并发；在分支数、队列长度、超时与下游 QPS 上设预算。loop 则必须有 `max_iter`、总 token/时间预算和退出原因，否则“证据不足”的评估可能空转。
- **可观测与评估**:逐节点记录输入摘要、输出 ID、耗时、token、路由决定、Memory 读写及循环次数；替换 reranker、embedding 或路由阈值时，分别看检索召回@k、路由准确率和端到端答案质量，详见 [[18 RAG 评估|RAG 评估]] 与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- **版本化拓扑**:把节点 schema、边、阈值和 Memory 策略作为配置/代码版本随评测集一起变更，才能 diff、回滚和解释线上差异。

### 可选实现：LangGraph（非范式要求）

若团队需要持久化状态图，可选用 LangGraph；本文查阅的 **Python `langgraph==1.2.9`（2026-07-10 发布）**可用 `StateGraph` 表达节点和条件边，并以 checkpointer 保存单个 thread 的状态快照。其官方文档区分 checkpointer 的短期 thread memory 与跨 thread 的 store；这与本页“Memory 要显式分作用域”的约束相符，但 API 与版本会演进，生产实现应锁定依赖并以对应版本文档为准。也可完全不使用它而实现相同拓扑。

## 面试高频

**Q1:Naive / Advanced / Modular RAG 怎么区分?**
A:Naive=纯 linear(索引→一次检索→拼 prompt→生成);Advanced=在直线两端加料(pre-retrieval 查询变换/路由 + post-retrieval 重排/压缩),**仍是直线**;Modular=打破直线,抽象成「模块 + 可任意编排的 flow(可条件、可并行、可循环)」,Advanced 是它的 linear 特例。追问「Modular 凭什么更强」→ 答:能表达 conditional/branching/loop 这些**直线表达不了的拓扑**,从而收编自纠错、多跳、自适应路由等方法。

**Q2:Modular RAG 的四种 flow 各对应什么真实技术?**
A:Linear=Naive/Advanced;Conditional=[[05 Routing|Routing]] / Adaptive-RAG(复杂度分类器选路);Branching=多子查询分解并行 + 结果融合(RRF)/ 多源并行召回;Loop=带回边的自评纠错(CRAG / Self-RAG)、多跳迭代(IRCoT / FLARE)。陷阱:把 branching 和 loop 混为一谈——branching 是**并行多路再汇聚**(无回边),loop 是**串行带反馈回边**(同一路反复跑)。

**Q3:Loop flow 和 Agentic RAG 什么关系?**
A:二者都可能有 loop，但**是否有回边不是分界线**。把「检索→自评→不满意则重写再检索」写成预定义图，是 corrective/adaptive 的 Modular RAG；若由模型在运行时决定是否检索、调用几次、调用哪个工具，控制权才转向 [[36 Agentic RAG|Agentic RAG]]。区别在「谁决定流程」，不是「有没有循环」。

**Q4:既然 Modular 这么好,为什么不无脑上?**
A:① 过度工程化——简单事实问答 linear 足矣,硬上 flow 编排只增延迟和复杂度;② 调试难——失败点散在多分支多轮,没 trace 调不动;③ loop 不收敛风险;④「模块化≠自动变好」,框架只让你好换好试,效果取决于塞进去的具体算子。**标准答:先 Naive 跑通基线、有明确瓶颈再模块化。**

**Q5(追问/陷阱):模块解耦是真的吗?**
A:理想是换一个模块别处不动,**实践常牵一发动全身**——改 Indexing 的分块粒度,往往要连带调 Retrieval 的 top-k 和 Post-retrieval 的重排策略(块变小→召回数要变→压缩比要改)。所以「模块边界清晰」是设计目标而非现实保证,相邻模块要一起评估。

## 知识拓展

- **与 Agentic RAG 的边界**:Modular RAG 的控制流通常由工程师预先编排；Agentic RAG 则可由 LLM 在运行时决定下一步。两者可组合，但应以延迟、成本、可审计性和任务成功率的评测结果决定控制权放在哪里，见 [[36 Agentic RAG|Agentic RAG]] 与 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]。
- **反模式**:① 「flow 套娃」——条件里再嵌分支再嵌循环,深到没人看得懂,出问题定位以小时计;② 「无护栏 loop」——不设 `max_iter`/预算,生产会被一个边角 query 烧穿成本;③ 「模块化但不可观测」——拆得再漂亮,没有逐节点 trace 等于黑盒。
- **实验边界**:把算子、flow 拓扑、路由阈值和 Memory 策略都当作待评测的候选干预；一次只改一个变量，并记录质量、P95 延迟、成本与权限命中率，避免把“结构更复杂”误判成“效果更好”。
- **延伸阅读**:Modular RAG 的四种 flow 把库内各篇串成一张网——Indexing→[[03 分块策略 Chunking|分块策略 Chunking]]/[[04 Embedding 与向量数据库|Embedding 与向量数据库]],Pre→[[07 查询变换 Query Transformation|查询变换 Query Transformation]]/[[05 Routing|Routing]],Retrieval→[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]],Post→[[10 重排序 Reranking|重排序 Reranking]],Generation→[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]],Orchestration→[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]/[[36 Agentic RAG|Agentic RAG]]。落地框架全景见 [[20 RAG 开源生态全景|RAG 开源生态全景]]。

## 来源

- Gao, Xiong, Wang, Wang. *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*. 2024. arXiv:2407.21059. <https://arxiv.org/abs/2407.21059>
- LangGraph 官方:Python `langgraph==1.2.9` release (2026-07-10). <https://github.com/langchain-ai/langgraph/releases/tag/1.2.9>；Memory / persistence 文档（2026-07-17 查阅）: <https://langchain-ai.github.io/langgraph/how-tos/persistence-functional/>。
- (同作者团队)Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey*. 2023. arXiv:2312.10997(Naive / Advanced / Modular 三分法的出处)。
