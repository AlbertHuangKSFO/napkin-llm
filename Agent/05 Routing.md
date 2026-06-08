[[05 Routing|Routing]] 是**先给输入分类,再把它分流到一条专门的后续处理**——不同类别走不同的 prompt、不同模型、甚至不同工具集。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:分类与处理解耦
一个万能 prompt 想同时处理所有类型的输入,往往会顾此失彼:对简单请求过度复杂,对复杂请求又力不从心,还容易把不同场景的指令互相干扰。

Routing 的核心动作是**先分类、后分流**:一个轻量的 router 先判定输入属于哪一类,再把它交给该类**专属优化过**的处理支路。这样每条支路只服务一种场景,可以各自做到最优,互不拖累。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:支路是开发者**预先定义**好的,router 只是在运行时选一条,不会凭空造出新支路。与全自主的 [[09 ReAct|ReAct]] 对照。

## 机制:一进多出的分叉
![[Routing.png]]

1. **分类(router)**:用一个**小模型 / 小分类器 / 甚至关键词规则**判定输入类别。router 要便宜、快——它本身不解决问题,只决定「交给谁」。常见实现:让小 LLM 输出一个 label,或微调一个分类器。
2. **分流**:据 label 路由到对应支路。每条支路有**自己的 prompt / 模型 / 工具**:
   - 高频、流程固定的类 → 便宜小模型 + 精简 prompt;
   - 需查数据、调 API 的类 → 配工具的支路;
   - 罕见、高价值、疑难的类 → 大模型,或升级人工。
3. **处理与返回**:命中的支路独立完成任务并返回。各支路彼此隔离,改一条不影响其他。

关键洞察:**把分类这件事单独拎出来**,正是因为「判定类别」和「处理该类」是两种不同的能力,合在一个 prompt 里会互相稀释。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中典型例子:客服系统先把进来的工单分成「退款 / 技术故障 / 一般咨询」等类别,各类用专门的下游 prompt 与流程;以及「把简单问题路由给小模型(省钱省延迟)、难问题路由给大模型」的成本优化用法。

## 可跑最小代码(伪代码)
```python
def route(query):
    # 1) router:小模型只做分类,输出一个 label
    label = small_llm(
        f"把用户请求分到 [refund, tech, general] 之一,只输出标签:\n{query}"
    ).strip()

    # 2) 分流:每条支路有各自的 prompt / 模型 / 工具
    if label == "refund":
        return small_llm(REFUND_PROMPT + query)          # 流程固定,小模型够用
    elif label == "tech":
        return llm_with_tools(TECH_PROMPT + query,        # 需查日志、调 API
                              tools=[search_logs, call_api])
    else:  # general / 兜底
        return big_llm(GENERAL_PROMPT + query)            # 疑难走大模型
```
要点:router 的输出是个**离散标签**;分流逻辑是**确定性代码**,选错支路就全错——所以 router 的分类质量是命门。

## 对比表
| 维度 | Routing | [[04 Prompt Chaining|Prompt Chaining]] | [[06 Parallelization|Parallelization]] |
|---|---|---|---|
| 拓扑 | 一进多出(选 1 条) | 串行单链 | 一进多出(全部跑) |
| 运行时动作 | 分类后**只走一条** | 顺序走完所有步 | **同时**跑多条 |
| 解决什么 | 输入异质、各类需不同处理 | 任务可拆固定子步 | 子任务独立 / 多次采样 |
| 成败命门 | router 分类准不准 | gate 与拆分质量 | 聚合 / 投票策略 |

## 何时用 / 坑
- **何时用**:输入有**清晰、可区分的类别**,且各类**确实需要不同处理**(不同 prompt/模型/工具)。典型:客服分流、按难度做成本优化、多语言/多领域分发。
- **坑一**:类别边界要清晰。若类别模糊、大量输入卡在分界,router 频繁误判,收益被抵消。
- **坑二**:router 选错则**整条全错**——它是单点。高风险场景给 router 配兜底类、置信度阈值,低置信时升级人工或走通用支路。
- **坑三**:别把 router 做得太重。它该是**小模型/小分类器**;若 router 本身就要复杂推理,说明分类没设计好。
- **坑四**:支路一多就难维护。类别数控制在「少而互斥」,不要无限细分。

## 关键事实
- router **可以用很便宜的模型甚至非 LLM 分类器**,这是它常被用作**成本优化**手段的原因:难易分流,把贵模型只留给难题。
- Routing 是**静态分流**——支路集合预先固定;若要运行时**动态生成**子任务,那是 [[07 Orchestrator-Workers|Orchestrator-Workers]],别混淆。
- 常与其他模式嵌套:某条支路内部可以是一条 [[04 Prompt Chaining|Prompt Chaining]],或一组 [[06 Parallelization|Parallelization]]。
- 与 Parallelization 的一句话区分:Routing 分类后**只走一条**;Parallelization **多条全跑**再聚合。

## 工业界实践
Routing 在生产里几乎无处不在,但很少叫这个名字——它常被包装成「模型路由 / model router」「网关 / gateway」「分发」。落地分三层成熟度。

**主流框架与服务(具体名 + 定位):**
- **RouteLLM**(LMSYS,2024-07 开源 / ICLR 2025):用**人类偏好数据**训练 router,把简单 query 路给小模型、难 query 路给大模型。官方数据:MT-Bench 上省 85%+ 成本、MMLU 省 45%、GSM8K 省 35%,同时保住 ~95% GPT-4 质量;整体成本降 2 倍以上不显著掉质量。是「成本优化型 Routing」的标杆实现。
- **vLLM Semantic Router / LLM Semantic Router**(2025,Red Hat 等推动):与 vLLM + Envoy `ext_proc` 集成的**信号驱动**路由,把成本、隐私、延迟、安全等异构信号编成路由策略。典型用法:用 BERT 类小分类器判 query 是否需要「开推理 / reasoning」,不需要的直接走快路径(对应论文《When to Reason》arXiv 2510.08731)。
- **语义路由(semantic routing)**:用 embedding 把 query 匹配到预设「话题向量」选支路,代表库 **semantic-router**(Aurelio AI)。比让 LLM 输 label 更快更便宜,适合意图清晰的场景。
- **托管网关**:**OpenRouter**、**LiteLLM Router**、**Portkey**、**Martian** 等在 API 层做跨厂商/跨模型路由 + 降级(failover)+ 负载均衡,是「多模型混用」工程的事实标准入口。
- **平台内置**:不少 IDE/客服产品有「auto」档,内部就是一个难度 router:轻问题给小模型,重问题给旗舰模型。

**典型架构:** 入口网关 → router(小分类器 / embedding / 小 LLM)→ 命中支路(各自的 prompt + 模型 + 工具)。router 与业务逻辑解耦,常单独部署成一个低延迟微服务;路由表(label→支路)放配置中心,改路由不动代码。

**规模化与成本/延迟:** router 必须**比最便宜的支路还便宜**,否则得不偿失——所以工业界优先用**非 LLM 分类器**(逻辑回归 / 小 BERT / embedding 最近邻),P50 延迟压到个位数毫秒;只有类别语义很微妙时才退而用小 LLM。成本账本上,Routing 的核心价值是把**贵模型的调用占比**从 100% 压到「只有难题的那一小撮」,这是大规模降本最直接的杠杆之一。

![[Routing-成本分流.png]]

**可观测与运维:** 必须埋点记录**每条 query 的 label、命中支路、置信度、最终是否被人工纠偏**。线上要监控**路由分布漂移**(某类突然暴涨往往是上游变化或被攻击的信号)和**误分类率**;评估见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。router 是单点,要做灰度发布 + 影子流量(shadow:新 router 只记录不生效,比对老 router)。

**踩坑与最佳实践:**
- router 给**兜底类 + 置信度阈值**:低置信走通用支路或转人工,别硬猜。
- 路由决策要**可解释、可回放**:出事时能复盘「为什么路到这条」。
- 类别集合**少而互斥**;新增类走数据驱动(从误分类样本里聚类发现新意图),别拍脑袋。
- router 与支路**分开评估**:router 看分类指标(准确率/混淆矩阵),支路看任务指标。

```python
# 生产化 router:优先小分类器,带置信度兜底
def production_route(query):
    label, conf = classifier.predict(query)        # 小 BERT / 逻辑回归,毫秒级
    log_routing(query, label, conf)                 # 必埋点:可观测 + 回放
    if conf < THRESHOLD:                            # 低置信不硬猜
        return general_branch(query)                # 兜底走通用支路
    return BRANCHES[label](query)                   # 命中专属支路
```

## 面试高频
**Q1:Routing 和 Orchestrator-Workers 都「一进多出」,本质区别是什么?**
标准答:Routing 的支路集合是**设计期预先固定**的,router 运行时只是**选一条**;Orchestrator-Workers 的子任务是**编排者 LLM 运行时动态生成**的,数量和内容事先不知道。一句话:Routing 是「选择已有支路」,Orchestrator-Workers 是「创造新子任务」。
- 追问「那 Routing 算 workflow 还是 agent?」→ workflow 一侧,因为没有模型自主决定控制流,只是按预设规则分流。

**Q2:为什么 router 要单独拎出来,不直接写一个万能 prompt?**
标准答:「判定类别」和「处理该类」是两种不同能力,塞进一个 prompt 会互相稀释——对简单请求过度复杂、对复杂请求力不从心、不同场景指令互相干扰。拆开后每条支路只服务一种场景可各自最优,且 router 能用更便宜的模型。
- 陷阱:别答「为了模块化」这种泛词,要点在**能力解耦 + 成本分层**。

**Q3:router 用什么模型?**
标准答:**越轻越好**。优先非 LLM 分类器(embedding 最近邻 / 小 BERT / 逻辑回归),次选小 LLM 输 label。绝不该用旗舰大模型当 router——那样路由本身的成本就吃掉了路由的收益。
- 追问「router 需要复杂推理才能分对类怎么办?」→ 说明**分类设计本身有问题**,类别边界不清,该重新设计类别,而不是给 router 加力。

**Q4:Routing 的单点风险怎么兜?**
标准答:router 选错则整条全错,它是单点。三道防线:① 置信度阈值,低置信走通用支路/转人工;② 兜底类(default branch)接住所有未命中;③ 影子流量 + 灰度发布换 router。

**陷阱题:「把简单问题给小模型、难问题给大模型」是哪个模式?** → 就是 Routing(成本优化用法),不是什么新东西;RouteLLM 是其代表实现。

## 知识拓展
- **进阶:从「硬路由」到「软路由 / cascade」。** 硬路由一次定终身;**级联(model cascade)**则是「先用小模型试,不行再升级到大模型」,本质是带 fallback 的串行 Routing,在「大多数 query 小模型就够」时更省。FrugalGPT(2023)是经典思路,RouteLLM 把它学习化了。
- **前沿(2024-2025):** RouteLLM 用**偏好数据**学路由策略(arXiv 2406.18665,ICLR 2025);vLLM Semantic Router 把路由做成**多信号决策**(成本/隐私/延迟/安全联合),《When to Reason》(arXiv 2510.08731,2025)专门研究「何时该开推理模式」的路由——这是推理模型时代的新路由维度:不只路由到哪个模型,还路由「要不要思考链」。
- **边界与反模式:** ① 类别模糊、大量 query 卡在分界 → router 频繁误判,收益被抵消,此时别用 Routing,考虑统一支路 + 更强模型;② router 做得太重(本身要复杂推理)→ 反模式,说明分类没设计好;③ 支路无限细分 → 维护灾难,保持「少而互斥」。
- **与兄弟模式的关系:** Routing 与 [[06 Parallelization|Parallelization]] 拓扑都是一进多出,但前者**选一条**后者**全跑**;某条支路内部常嵌一条 [[04 Prompt Chaining|Prompt Chaining]] 或一组 Parallelization。真正动态分流要上 [[07 Orchestrator-Workers|Orchestrator-Workers]]。Routing 也是 [[36 Agentic RAG|Agentic RAG]] 的常见入口(先判 query 该不该检索、走哪个知识库)。成本/延迟权衡见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
