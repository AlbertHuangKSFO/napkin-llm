[[06 Parallelization|Parallelization]] 是**让多个 LLM 调用并行跑、再把结果合起来**,有两个变体:**Sectioning**(拆成独立子任务并行)和 **Voting**(同一任务多跑几次投票)。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:并行的两种动机
单次串行调用有两个天然短板:一是把多件互不相关的事塞进一个上下文,既慢又互相干扰;二是单次采样有随机性,可能恰好出错。Parallelization 用「并行」分别对治这两点:

- **Sectioning(分块)**:任务能切成**互不依赖**的子任务 → 各开一路 LLM 并行做,再聚合。动机是**聚焦 + 提速**:每路上下文更干净,且并发省墙钟时间。
- **Voting(投票)**:同一个任务**跑多次**(变温度/变提示),对结果**投票取共识**。动机是**提可靠性**:用多次采样压低方差,防假阴/假阳。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:并行的路数与切分都是开发者**预先设计**的,不是运行时由模型决定。与全自主的 [[09 ReAct|ReAct]] 对照。

## 变体一:Sectioning(扇出—扇入)
![[Parallelization-Sectioning.svg]]

1. **切分(设计期)**:把任务拆成若干**彼此无依赖**的子任务。判据:子任务 A 的输入不需要子任务 B 的输出。
2. **扇出并行**:每个子任务一路独立 LLM 调用,**同时**发起。
3. **扇入聚合**:用代码或一次 LLM 调用把各路结果**合并 / 拼装**成最终产物。

典型例子:① **关注点分离**——一路 LLM 正常回答用户,另一路专门审「这条回答有无越权/违规」,两路并行,互不污染上下文;② 把一份长文档切成多段,各段并行做摘要/打分再汇总;③ 自动化评测里,多条评测准则各跑一路。

## 变体二:Voting(多路采样投票)
![[Parallelization-Voting.svg]]

1. **多次采样**:**同一个**任务跑 N 次——可调高温度、或用略不同的提示措辞,制造多样性。
2. **投票/共识**:对 N 个结果做**多数表决**,或设阈值(如「≥2 路说有漏洞才判有」),按容错需求调松紧。
3. **出结论**:取共识为最终结果。

典型例子:① **代码审漏洞**——多个 prompt 各审一遍,任一路报警就深查,或多数表决降误报;② 内容审核里「几票同意才放行」,可调成保守(多票才通过)或激进。

两个变体可叠加:每个 section 内部再 voting。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中明确把 Parallelization 分为 **Sectioning** 与 **Voting** 两类,并指出它适合「子任务可并行以提速」或「需要多视角/多次尝试以提高置信」的场景。

## 可跑最小代码(伪代码)
```python
import concurrent.futures as cf

# --- Sectioning:独立子任务并行,再聚合 ---
def sectioning(doc):
    tasks = {
        "summary":  f"用三句话总结:\n{doc}",
        "safety":   f"这段内容有无违规?只答 是/否 + 理由:\n{doc}",
        "keywords": f"抽 5 个关键词:\n{doc}",
    }
    with cf.ThreadPoolExecutor() as ex:
        results = {k: ex.submit(llm, p) for k, p in tasks.items()}
        out = {k: f.result() for k, f in results.items()}
    return out                      # 聚合:这里直接拼成 dict

# --- Voting:同一任务多跑,投票 ---
def voting(code, n=3, threshold=2):
    prompt = f"这段代码有安全漏洞吗?只答 yes/no:\n{code}"
    with cf.ThreadPoolExecutor() as ex:
        votes = [f.result().strip().lower()
                 for f in [ex.submit(llm, prompt, temperature=0.8)
                           for _ in range(n)]]
    return "有漏洞" if votes.count("yes") >= threshold else "无"
```

## 对比表
| 维度 | Sectioning | Voting | [[07 Orchestrator-Workers|Orchestrator-Workers]] |
|---|---|---|---|
| 并行的是 | **不同**子任务 | **同一**任务多次 | 动态生成的子任务 |
| 目的 | 聚焦 + 提速 | 提可靠性 / 降方差 | 处理不可预知的拆分 |
| 子任务来源 | 设计期预定 | 设计期预定 | **运行时**由编排者定 |
| 聚合方式 | 合并/拼装 | 投票/共识 | 编排者综合 |

## 何时用 / 坑
- **何时用 Sectioning**:任务能切成**真正独立**的子任务,且并行能省时间或让每路上下文更聚焦。
- **何时用 Voting**:任务有**明确对错/高风险**,单次采样不够稳,需多次尝试提置信。
- **坑一**:Sectioning 要求子任务**无依赖**;一旦有依赖(A 要等 B),并行就错了,应改回 [[04 Prompt Chaining|Prompt Chaining]] 串行。
- **坑二**:Voting 的成本是 N 倍调用;只在可靠性收益值得这笔算力时用,且要定好投票规则(多数?阈值?加权?)。
- **坑三**:聚合环节常被低估——拼装/投票逻辑写错,前面并行做得再好也白搭。
- **坑四**:它是**静态**并行;若子任务数量/内容要运行时才知道,那是 [[07 Orchestrator-Workers|Orchestrator-Workers]],不是 Parallelization。

## 关键事实
- 两个变体动机不同:Sectioning 主**性能/聚焦**,Voting 主**可靠性**——别混为一谈。
- 与 [[05 Routing|Routing]] 的一句话区分:Routing 分类后**只走一条**支路;Parallelization **多条全跑**再聚合。
- 与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的本质差别:这里的并行任务是**预先写死**的;那里的子任务是**编排者运行时动态拆**的。
- 常与 [[05 Routing|Routing]]、[[04 Prompt Chaining|Prompt Chaining]] 嵌套组合,真实系统很少只用单一模式。

## 工业界实践
两个变体在生产里走的是两条完全不同的工程路线,别用一套话术套两者。

**Sectioning(分块并行)——这是延迟优化的主力。**
- **落地形态:** 任何「关注点分离」都该并行。最经典的是**主回答 + 安全审查并行**:一路正常回答用户,另一路专门判「有无越权/违规/PII 泄露」,两路同时跑、互不污染上下文,审查路否决就拦截。客服、内容平台普遍这么做。
- **框架支持:** **LangGraph** 用图的 fan-out/fan-in 节点原生表达并行分支;**LlamaIndex Workflows**、**Haystack** 的 pipeline、**OpenAI Agents SDK** 都支持并发子任务。底层是 `asyncio.gather` / 线程池——并行的是 **I/O 等待**(等模型返回),所以用协程比多线程更省资源。
- **长文档处理:** RAG 里把长文切多段并行摘要/打分再汇总(map-reduce 式),是把「上下文塞不下」转成「分而治之」的标准手法,见 [[21 上下文压缩与卸载|上下文压缩与卸载]]。

**Voting(投票)——这是可靠性 / 准确率优化的主力,学术上就是 self-consistency。**
- **核心方法:** Self-Consistency(Wang et al. 2023)——同一推理任务采样 N 条路径,对**最终答案多数表决**,在 GSM8K/MATH 等有唯一解的任务上稳定涨点。生产里用于数学/代码/抽取这类「有明确对错」的场景。
- **代码审漏洞 / 内容审核:** 多路各审一遍,按容错需求调阈值——保守场景「任一路报警就深查」(高召回),激进场景「多数同意才放行」(降误报)。
- **成本是 N 倍**,所以工业界普遍用**自适应采样**:先采少量,若已高度一致就早停,只有分歧大才继续采(对应 DeepConf、Adaptive-Consistency 等思路),把 N 倍成本压回来。

**规模化与成本/延迟:** Sectioning 用并发**换墙钟时间**(总时间 ≈ 最慢那一路,而非各路之和),但**总 token 不降反略升**,要的是延迟不是省钱;Voting 则是**纯花算力换准确率**,token 是 N 倍。两者都要做**并发上限 + 限流**,防止瞬时打爆下游模型配额。

**可观测与运维:** 并行链路要记录**每路的耗时、成败、聚合/投票的最终裁决**;尾延迟(P99)由最慢一路决定,要对慢路设超时 + 兜底(超时就用已返回的部分聚合)。聚合/投票逻辑是常被低估的 bug 温床,要单测。

```python
# Sectioning:异步并发(并行的是 I/O 等待),带超时兜底
import asyncio
async def sectioning(doc):
    tasks = {
        "summary":  llm_async(f"三句话总结:\n{doc}"),
        "safety":   llm_async(f"有无违规?是/否+理由:\n{doc}"),
        "keywords": llm_async(f"抽 5 个关键词:\n{doc}"),
    }
    done = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {k: (v if not isinstance(v, Exception) else None)   # 某路挂了不拖垮整体
            for k, v in zip(tasks, done)}
```

## 面试高频
**Q1:Parallelization 的两个变体分别解决什么问题?别答串了。**
标准答:**Sectioning** 解决**性能与聚焦**——把无依赖子任务并行,省墙钟时间、让每路上下文更干净;**Voting** 解决**可靠性**——同一任务多次采样投票,用多样性压低单次采样的方差。一个主延迟,一个主准确率,动机完全不同。
- 陷阱:面试官故意问「并行是不是为了省钱」→ 都不是为了省钱。Sectioning 省的是**时间**(token 还略升),Voting 反而**花 N 倍算力**。

**Q2:Voting 和学术里的 self-consistency 什么关系?**
标准答:Voting 就是 self-consistency(Wang et al. 2023)的工程化——多采样 + 多数表决。适用前提是任务**有明确对错**(答案可比较/可投票);对开放生成(摘要、代码)字符串相等失效,要用 Universal Self-Consistency(让 LLM 自己判哪个输出最好)或排序投票等变体。

**Q3:什么时候 Sectioning 是错的?**
标准答:子任务**有依赖**时(A 的输入要等 B 的输出)。这时并行会拿到过期/缺失的输入,必须改回 [[04 Prompt Chaining|Prompt Chaining]] 串行。判据:子任务 A 的输入是否需要子任务 B 的输出——需要就不能并行。

**Q4:Parallelization 和 Orchestrator-Workers 都是扇出扇入,怎么区分?**
标准答:**唯一本质差别**是子任务来源——Parallelization 的子任务**设计期写死**,数量内容固定;Orchestrator-Workers 的子任务**编排者 LLM 运行时动态生成**。问自己一句:「要拆成几个、各干什么,事先知道吗?」知道用 Parallelization,不知道用 Orchestrator-Workers。

**陷阱题:Voting 跑了 5 路结果 3 比 2,怎么定?** → 看任务的**容错方向**:高风险/宁可错杀(如安全审核)用低阈值(少数报警就拦),要降误报用多数甚至加权投票。没有固定答案,要先问「错哪个方向代价大」。

## 知识拓展
- **前沿(2024-2025):** Voting 这条线很活跃——**排序投票自一致性**(Ranked Voting,arXiv 2505.10772,2025,用 Borda/即时决选等)让投票更稳;**DeepConf / Deep Think with Confidence**(2025)用置信度自适应决定采多少路,把 N 倍成本压下来;**Self-Consistency Preference Optimization**(2025)甚至把「偏好一致答案」做成无监督对齐目标。多智能体辩论(multi-agent debate)是 Voting 的升级版——让模型互相质疑再达共识,见 [[22 多智能体系统|多智能体系统]]。
- **Best-of-N 与 Voting 的区别:** Voting 靠**多数表决**(无需打分器);Best-of-N 靠一个**奖励模型/verifier 选最高分**那条。后者需要一个好的 reward model,质量上限更高但更重,见 [[32 Agentic RL 与训练|Agentic RL 与训练]]。
- **边界与反模式:** ① 子任务有依赖却强行并行 → 拿到过期输入,经典错误;② 把动态子任务硬塞进静态并行 → 应上 [[07 Orchestrator-Workers|Orchestrator-Workers]];③ 盲目堆高 N → 前几路收益最大,后面边际递减还线性烧钱;④ 聚合/投票逻辑没单测 → 前面并行做得再好,聚合写错全白搭。
- **与兄弟模式:** 与 [[05 Routing|Routing]] 一句话区分——Routing 分类后**只走一条**,Parallelization **多条全跑**再聚合;两者常嵌套(先 Routing 选支路,支路内部再 Sectioning)。延迟/成本系统化优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
