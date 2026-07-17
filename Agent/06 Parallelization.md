[[06 Parallelization|Parallelization]] 是**让多个 LLM 调用并行跑、再把结果合起来**,有两个变体:**Sectioning**(拆成独立子任务并行)和 **Voting**(同一任务多跑几次投票)。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:并行的两种动机
单次串行调用有两个天然短板:一是把多件互不相关的事塞进一个上下文,既慢又互相干扰;二是单次采样有随机性,可能恰好出错。Parallelization 用「并行」分别对治这两点:

- **Sectioning(分块)**:任务能切成**互不依赖**的子任务 → 各开一路 LLM 并行做,再聚合。动机是**聚焦 + 提速**:每路上下文更干净,且并发省墙钟时间。
- **Voting(投票)**:同一个任务**跑多次**(变温度/变提示),对结果**投票取共识**。动机是**提可靠性**:用多次采样压低方差,防假阴/假阳。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:并行的路数与切分都是开发者**预先设计**的,不是运行时由模型决定。与全自主的 [[09 ReAct|ReAct]] 对照。

**生活类比:** 像三位同学同时检查一份作业：一位看错别字、一位看计算、一位看引用，这叫 Sectioning；若三位都独立做同一道选择题再多数表决，才叫 Voting。前提都是他们不必等彼此的答案。

## 变体一:Sectioning(扇出—扇入)
![[Parallelization-Sectioning.png]]

1. **切分(设计期)**:把任务拆成若干**彼此无依赖**的子任务。判据:子任务 A 的输入不需要子任务 B 的输出。
2. **扇出并行**:每个子任务一路独立 LLM 调用,**同时**发起。
3. **扇入聚合**:用代码或一次 LLM 调用把各路结果**合并 / 拼装**成最终产物。

典型例子:① **关注点分离**——一路 LLM 正常回答用户,另一路专门审「这条回答有无越权/违规」,两路并行,互不污染上下文;② 把一份长文档切成多段,各段并行做摘要/打分再汇总;③ 自动化评测里,多条评测准则各跑一路。

## 变体二:Voting(多路采样投票)
![[Parallelization-Voting.png]]

1. **多次采样**:**同一个**任务跑 N 次——可调高温度、或用略不同的提示措辞,制造多样性。
2. **投票/共识**:对 N 个结果做**多数表决**,或设阈值(如「≥2 路说有漏洞才判有」),按容错需求调松紧。
3. **出结论**:取共识为最终结果。

典型例子:① **代码审漏洞**——多个 prompt 各审一遍,任一路报警就深查,或多数表决降误报;② 内容审核里「几票同意才放行」,可调成保守(多票才通过)或激进。

两个变体可叠加:每个 section 内部再 voting。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中明确把 Parallelization 分为 **Sectioning** 与 **Voting** 两类,并指出它适合「子任务可并行以提速」或「需要多视角/多次尝试以提高置信」的场景。

## 可跑最小代码
```python
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, sleep

def inspect(part):
    sleep(0.03)  # 模拟一次独立的模型/API 调用
    return part.upper()

# ❌ 朴素写法：彼此独立的工作仍串行等待。
def serial_sectioning(parts):
    return [inspect(part) for part in parts]

# ✅ 改进写法：扇出独立工作，再按原顺序扇入聚合。
def fanout_sectioning(parts):
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        return list(pool.map(inspect, parts))

parts = ["摘要", "安全", "关键词"]
start = perf_counter()
serial = serial_sectioning(parts)
serial_seconds = perf_counter() - start
start = perf_counter()
fanout = fanout_sectioning(parts)
fanout_seconds = perf_counter() - start
assert serial == fanout and fanout_seconds < serial_seconds
print(f"串行={serial_seconds:.3f}s, 并行={fanout_seconds:.3f}s, 结果={fanout}")
```

## 对比表
| 维度    | Sectioning | Voting     | [[07 Orchestrator-Workers\|Orchestrator-Workers]] |
| ----- | ---------- | ---------- | ------------------------------------------------- |
| 并行的是  | **不同**子任务  | **同一**任务多次 | 动态生成的子任务                                          |
| 目的    | 聚焦 + 提速    | 提可靠性 / 降方差 | 处理不可预知的拆分                                         |
| 子任务来源 | 设计期预定      | 设计期预定      | **运行时**由编排者定                                      |
| 聚合方式  | 合并/拼装      | 投票/共识      | 编排者综合                                             |

## 何时用 / 坑
- **何时用 Sectioning**:任务能切成**真正独立**的子任务,且并行能省时间或让每路上下文更聚焦。
- **何时用 Voting**:任务有**明确对错/高风险**,单次采样不够稳,需多次尝试提置信。
- **坑一**:Sectioning 要求子任务**无依赖**;一旦有依赖(A 要等 B),并行就错了,应改回 [[04 Prompt Chaining|Prompt Chaining]] 串行。
- **坑二**:Voting 的成本是 N 倍调用;只在可靠性收益值得这笔算力时用,且要定好投票规则(多数?阈值?加权?)。
- **坑三**:聚合环节常被低估——拼装/投票逻辑写错,前面并行做得再好也白搭。
- **坑四**:它是**静态**并行;若子任务数量/内容要运行时才知道,那是 [[07 Orchestrator-Workers|Orchestrator-Workers]],不是 Parallelization。

## 关键事实
- 两个变体动机不同：Sectioning 主**性能/聚焦**，Voting 主**可靠性**，不可混为一谈。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 与 [[05 Routing|Routing]] 的一句话区分：Routing 分类后选定支路；Parallelization 多条预设调用全跑后再聚合。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的关键差别：这里的并行任务是**预先写死**的；后者由编排者按输入动态拆分。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 可与 [[05 Routing|Routing]]、[[04 Prompt Chaining|Prompt Chaining]] 嵌套组合；组合后仍应针对任务测量质量、成本和尾延迟。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)

## 小数字手算与公式推导

**Sectioning 省的是墙钟时间，不自动省调用成本。** 设三条独立支路耗时为 $t_1,t_2,t_3$，聚合耗时为 $t_m$。串行与并行的理想墙钟时间分别是

$$
T_{serial}=t_1+t_2+t_3+t_m,\qquad
T_{parallel}=\max(t_1,t_2,t_3)+t_m.
$$

若三路为 $2,3,4$ 秒，聚合为 $0.5$ 秒：

$$
T_{serial}=2+3+4+0.5=9.5\text{s},
\quad T_{parallel}=\max(2,3,4)+0.5=4.5\text{s}.
$$

理想加速比为 $9.5/4.5\approx2.11$。这忽略队列、限流和慢尾；请求数仍是 3 路加 1 次聚合，token 成本是否变化取决于 prompt 与聚合策略。

**Voting 只在误差不高度相关时才会放大可靠性。** 设每次判断错误率为 $e$，三票多数错意味着恰有两票或三票错：

$$
P_{wrong}=\binom{3}{2}e^2(1-e)+\binom{3}{3}e^3.
$$

若 $e=0.2$ 且三次采样独立：

$$
P_{wrong}=3\times0.2^2\times0.8+0.2^3=0.104.
$$

错误率从 $0.2$ 降到 $0.104$。同模型、同 prompt 的错误常相关，因此这个数字是独立性下界式的教学例子；是否有收益必须实测，不可保证。

## 工业界实践
两个变体在生产里走的是两条完全不同的工程路线,别用一套话术套两者。

**Sectioning(分块并行)——适合独立工作带来的延迟改善。**
- **落地形态:** 任何「关注点分离」都该并行。最经典的是**主回答 + 安全审查并行**:一路正常回答用户,另一路专门判「有无越权/违规/PII 泄露」,两路同时跑、互不污染上下文,审查路否决就拦截。客服、内容平台普遍这么做。
- **落地方式:** 用任务队列、协程或线程池把独立调用同时发起；具体并发模型取决于 SDK、限流规则与运行环境。无论框架如何，须给每一路设置超时、取消与部分结果策略。
- **长文档处理:** RAG 里把长文切多段并行摘要/打分再汇总(map-reduce 式),是把「上下文塞不下」转成「分而治之」的标准手法,见 [[21 上下文压缩与卸载|上下文压缩与卸载]]。

**Voting(投票)——用额外采样换取条件性的可靠性提升。**
- **核心方法:** Self-Consistency 提出对同一问题采样多条推理路径，再聚合最一致的答案；它适合可规范化、可比较的最终答案。开放式写作不能直接按字符串多数投票，需改用任务特定的聚合或外部验证器。[Wang et al., 2022/ACL 2023, arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **代码审漏洞 / 内容审核:** 多路各审一遍,按容错需求调阈值——保守场景「任一路报警就深查」(高召回),激进场景「多数同意才放行」(降误报)。
- **成本与停止:** $N$ 次完整采样通常近似 $N$ 倍生成成本；可用预算上限或在结果已满足预设置信规则时早停，但早停本身也要在离线数据上验证偏差。

**规模化与成本/延迟:** Sectioning 用并发压低墙钟时间（接近最慢一路加聚合），但总 token 不会因「同时发」而天然下降；Voting 则用额外采样换取可能的可靠性收益。两者都需并发上限与限流，避免瞬时耗尽下游配额。

**可观测与运维:** 并行链路要记录**每路的耗时、成败、聚合/投票的最终裁决**;尾延迟(P99)由最慢一路决定,要对慢路设超时 + 兜底(超时就用已返回的部分聚合)。聚合/投票逻辑是常被低估的 bug 温床,要单测。

```python
# Sectioning:异步并发；生产中还应在 llm_async 内或外层加入超时/取消策略
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
> 面试地图：[[Agent 面试题库]]
**Q1:Parallelization 的两个变体分别解决什么问题?别答串了。**
标准答:**Sectioning** 解决**性能与聚焦**——把无依赖子任务并行,省墙钟时间、让每路上下文更干净;**Voting** 解决**可靠性**——同一任务多次采样投票,用多样性压低单次采样的方差。一个主延迟,一个主准确率,动机完全不同。
- 陷阱:面试官故意问「并行是不是为了省钱」→ 都不是为了省钱。Sectioning 省的是**时间**(token 还略升),Voting 反而**花 N 倍算力**。

**Q2:Voting 和学术里的 self-consistency 什么关系?**
标准答:Voting 是 self-consistency 的常见工程化形态：多采样再聚合。它特别适合最终答案可规范化、可比较的任务；对摘要、代码等开放产物，字符串相等会失效，需采用任务特定的排序、测试或 verifier，且要验证聚合器本身。

**Q3:什么时候 Sectioning 是错的?**
标准答:子任务**有依赖**时(A 的输入要等 B 的输出)。这时并行会拿到过期/缺失的输入,必须改回 [[04 Prompt Chaining|Prompt Chaining]] 串行。判据:子任务 A 的输入是否需要子任务 B 的输出——需要就不能并行。

**Q4:Parallelization 和 Orchestrator-Workers 都是扇出扇入,怎么区分?**
标准答:**唯一本质差别**是子任务来源——Parallelization 的子任务**设计期写死**,数量内容固定;Orchestrator-Workers 的子任务**编排者 LLM 运行时动态生成**。问自己一句:「要拆成几个、各干什么,事先知道吗?」知道用 Parallelization,不知道用 Orchestrator-Workers。

**陷阱题:Voting 跑了 5 路结果 3 比 2,怎么定?** → 看任务的**容错方向**:高风险/宁可错杀(如安全审核)用低阈值(少数报警就拦),要降误报用多数甚至加权投票。没有固定答案,要先问「错哪个方向代价大」。

## 知识拓展
- **证据边界:** Anthropic 将 Parallelization 分成 Sectioning 与 Voting，并将前者用于独立子任务、后者用于多尝试提高置信；这是一种模式选择建议，不是对每个任务的效果承诺。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- **Best-of-N 与 Voting 的区别:** Voting 靠**多数表决**（无需额外打分器）；Best-of-N 靠奖励模型或 verifier 选择最高分候选。后者把成败转移到 scorer 质量上，可能更适合可验证任务，但不保证一定优于投票，见 [[32 Agentic RL 与训练|Agentic RL 与训练]]。
- **边界与反模式:** ① 子任务有依赖却强行并行 → 拿到过期输入；② 把动态子任务硬塞进静态并行 → 应考虑 [[07 Orchestrator-Workers|Orchestrator-Workers]]；③ 未验证独立性便盲目堆高 $N$；④ 聚合/投票逻辑没有针对性测试。
- **与兄弟模式:** 与 [[05 Routing|Routing]] 一句话区分——Routing 分类后**只走一条**,Parallelization **多条全跑**再聚合;两者常嵌套(先 Routing 选支路,支路内部再 Sectioning)。延迟/成本系统化优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
