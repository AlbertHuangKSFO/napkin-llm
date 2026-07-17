 [[05 Routing|Routing]] 是**先给输入分类,再把它分流到一条专门的后续处理**——不同类别走不同的 prompt、不同模型、甚至不同工具集。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:分类与处理解耦
一个万能 prompt 想同时处理所有类型的输入,往往会顾此失彼:对简单请求过度复杂,对复杂请求又力不从心,还容易把不同场景的指令互相干扰。

Routing 的核心动作是**先分类、后分流**:一个轻量的 router 先判定输入属于哪一类,再把它交给该类**专属优化过**的处理支路。这样每条支路只服务一种场景,可以各自做到最优,互不拖累。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:支路是开发者**预先定义**好的,router 只是在运行时选一条,不会凭空造出新支路。与全自主的 [[09 ReAct|ReAct]] 对照。

**生活类比:** 像医院分诊台先判断病人该去内科、骨科还是急诊，再送到对应科室。分诊台不替医生看病；它选错科会增加绕路，所以要有“症状不明 → 急诊/人工复核”的兜底。

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

## 可跑最小代码
```python
# 这是可直接运行的路由器替身；生产里可替换为规则、分类器或 LLM。
def universal_model(query):
    return f"通用支路（成本 10）：{query}"

def classify(query):
    if "退款" in query:
        return "refund", 0.95
    if "报错" in query:
        return "tech", 0.90
    return "general", 0.40

def refund_branch(query):
    return f"退款支路（成本 2）：{query}"

def tech_branch(query):
    return f"技术支路（成本 6）：{query}"

# ❌ 朴素写法：所有请求都交给一个昂贵的通用处理器。
def universal_handle(query):
    return universal_model(query)

# ✅ 改进写法：先分类，再进入固定支路；低置信和未知标签走兜底。
def routed_handle(query, threshold=0.80):
    label, confidence = classify(query)
    branches = {"refund": refund_branch, "tech": tech_branch}
    if confidence < threshold:
        return universal_model(query)
    return branches.get(label, universal_model)(query)

assert "通用支路" in universal_handle("想退款")
assert "退款支路" in routed_handle("想退款")
assert "通用支路" in routed_handle("今天怎么样？")
print(routed_handle("登录报错"))
```
要点:router 的输出是个**离散标签**；分流逻辑可以是确定性代码。错路由会提高下游失配风险，但不必然「整条全错」——通用兜底、二次校验或人工升级仍可恢复，因此要把恢复率一并纳入评估。

## 对比表
| 维度    | Routing      | [[04 Prompt Chaining\|Prompt Chaining]] | [[06 Parallelization\|Parallelization]] |
| ----- | ------------ | ----------------------------------------- | --------------------------------------- |
| 拓扑    | 一进多出(选 1 条)  | 串行单链                                      | 一进多出(全部跑)                               |
| 运行时动作 | 分类后**只走一条**  | 顺序走完所有步                                   | **同时**跑多条                               |
| 解决什么  | 输入异质、各类需不同处理 | 任务可拆固定子步                                  | 子任务独立 / 多次采样                            |
| 成败命门  | router 分类准不准 | gate 与拆分质量                                | 聚合 / 投票策略                               |

## 何时用 / 坑
- **何时用**:输入有**清晰、可区分的类别**,且各类**确实需要不同处理**(不同 prompt/模型/工具)。典型:客服分流、按难度做成本优化、多语言/多领域分发。
- **坑一**:类别边界要清晰。若类别模糊、大量输入卡在分界,router 频繁误判,收益被抵消。
- **坑二**:router 是高影响单点。高风险场景给它配兜底类、置信度阈值与升级路径；低置信时走通用支路或人工，不要硬猜。
- **坑三**:router 的成本与延迟必须用端到端账本验证。轻量分类器常合适，但若语义确实复杂，使用更强 router 也可能合理；关键是它带来的质量收益是否超过代价。
- **坑四**:支路一多就难维护。类别数控制在「少而互斥」,不要无限细分。

## 关键事实
- router 可以用较低成本模型或非 LLM 分类器；分类可准确时，难易分流才可能带来成本/速度收益。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- Routing 是**静态分流**——支路集合预先固定；若要运行时**动态生成**子任务，应看 [[07 Orchestrator-Workers|Orchestrator-Workers]]。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 路由支路可嵌一条 [[04 Prompt Chaining|Prompt Chaining]]，或嵌一组 [[06 Parallelization|Parallelization]]；组合后的质量仍须按端到端任务评测。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 与 Parallelization 的一句话区分：Routing 分类后选定支路；Parallelization 同时运行多个预设调用再聚合。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)

## 工业界实践
Routing 常部署在入口层：`输入 → 分类/置信度 → 固定支路 → 兜底或升级`。分类器可以是规则、传统模型、embedding 检索或 LLM；选择不是信仰题，而是由准确率、延迟、模型价格、错误代价和可观测性共同决定。

**成本手算：先算期望值再谈省钱。** 设始终调用强模型的单次成本为 $C_b$，router 成本为 $C_r$，轻支路成本为 $C_s$，被判为轻任务的比例为 $p$。若其余请求走强支路，则

$$
E[C_{route}] = C_r + pC_s + (1-p)C_b
$$

相对全走强模型的节省为

$$
\Delta C = C_b-E[C_{route}] = p(C_b-C_s)-C_r.
$$

例如 $C_b=10$、$C_s=2$、$C_r=0.4$、$p=0.7$：

$$
E[C_{route}]=0.4+0.7\times2+0.3\times10=4.8,
\qquad \Delta C=10-4.8=5.2.
$$

这说明该**假设下**成本下降；它没有计入错路由造成的返工、人工升级或质量损失，故上线前还要在标注集与线上影子流量中验证端到端效用。

**研究例证（非通用承诺）:** RouteLLM 用偏好数据在强、弱模型间学习路由；论文报告某些基准配置可把成本降到一半以下而不牺牲其测量的响应质量，但结果依赖模型对、数据与阈值，不能外推成固定节省比例。[Ong et al., 2024/2025, arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

![[Routing-成本分流.png]]

**可观测与运维:** 记录每条 query 的 label、命中支路、置信度、最终结果、升级/纠偏情况。监控路由分布漂移、每类的混淆矩阵和端到端成功率；高风险类别尤其看召回率而非只看总体准确率。新 router 可先走影子流量：只记录、不生效，再与当前策略比对。评估方法见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践:**
- router 给**兜底类 + 置信度阈值**:低置信走通用支路或转人工,别硬猜。
- 路由决策要**可解释、可回放**:出事时能复盘「为什么路到这条」。
- 类别集合**少而互斥**;新增类走数据驱动(从误分类样本里聚类发现新意图),别拍脑袋。
- router 与支路**分开评估**:router 看分类指标(准确率/混淆矩阵),支路看任务指标。

```python
# 生产化 router:优先小分类器,带置信度兜底
def production_route(query):
    label, conf = classifier.predict(query)        # 分类器实现由实际延迟/质量目标决定
    log_routing(query, label, conf)                 # 必埋点:可观测 + 回放
    if conf < THRESHOLD:                            # 低置信不硬猜
        return general_branch(query)                # 兜底走通用支路
    return BRANCHES.get(label, general_branch)(query)  # 未知标签仍走兜底
```

## 面试高频
**Q1:Routing 和 Orchestrator-Workers 都「一进多出」,本质区别是什么?**
标准答:Routing 的支路集合是**设计期预先固定**的,router 运行时只是**选一条**;Orchestrator-Workers 的子任务是**编排者 LLM 运行时动态生成**的,数量和内容事先不知道。一句话:Routing 是「选择已有支路」,Orchestrator-Workers 是「创造新子任务」。
- 追问「那 Routing 算 workflow 还是 agent?」→ workflow 一侧,因为没有模型自主决定控制流,只是按预设规则分流。

**Q2:为什么 router 要单独拎出来,不直接写一个万能 prompt?**
标准答:「判定类别」和「处理该类」是两种不同能力,塞进一个 prompt 会互相稀释——对简单请求过度复杂、对复杂请求力不从心、不同场景指令互相干扰。拆开后每条支路只服务一种场景可各自最优,且 router 能用更便宜的模型。
- 陷阱:别答「为了模块化」这种泛词,要点在**能力解耦 + 成本分层**。

**Q3:router 用什么模型?**
标准答:先比较端到端效用。类别清晰时，规则、embedding 或小分类器往往足够；语义复杂时可用 LLM router。用强模型并非绝对错误，只是要证明其增量质量、风险降低或成本收益能覆盖 router 本身的代价。
- 追问「router 需要复杂推理才能分对类怎么办?」→ 先检查类别定义、训练数据和兜底策略；若仍需复杂判断，就把更强 router 的成本与误路由代价一起做 A/B 测，而不是凭直觉断言它不该存在。

**Q4:Routing 的单点风险怎么兜?**
标准答:router 是高影响单点，但错路由可被恢复。三道防线:① 置信度阈值，低置信走通用支路/转人工；② 兜底类(default branch)接住未命中；③ 影子流量 + 灰度发布，并量测每类端到端结果。

**陷阱题:「把简单问题给小模型、难问题给大模型」是哪个模式?** → 就是 Routing(成本优化用法),不是什么新东西;RouteLLM 是其代表实现。

## 知识拓展
- **进阶:从「硬路由」到「软路由 / cascade」。** 硬路由一次定支路；**级联(model cascade)**则是「先用较低成本支路，再按明确的失败/置信规则升级」。它是否更省，仍由前文的 $\Delta C$、升级率和质量指标决定。
- **证据边界:** Anthropic 将 Routing 定义为「把输入分类并导向专用后续任务」，要求类别可准确地由 LLM 或传统分类器判断；它没有给出适用于所有业务的模型、延迟或节省比例。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- **边界与反模式:** ① 类别模糊、大量 query 卡在分界 → router 频繁误判，需比较统一支路、重定义类别和升级策略；② 没算 $\Delta C$ 与错路由代价就声称省钱；③ 支路无限细分 → 维护灾难，保持「少而可评估」。
- **与兄弟模式的关系:** Routing 与 [[06 Parallelization|Parallelization]] 拓扑都是一进多出,但前者**选一条**后者**全跑**;某条支路内部常嵌一条 [[04 Prompt Chaining|Prompt Chaining]] 或一组 Parallelization。真正动态分流要上 [[07 Orchestrator-Workers|Orchestrator-Workers]]。Routing 也是 [[36 Agentic RAG|Agentic RAG]] 的常见入口(先判 query 该不该检索、走哪个知识库)。成本/延迟权衡见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
