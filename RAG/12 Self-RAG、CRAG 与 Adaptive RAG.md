[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 是 2023–2024 年三篇代表性论文,共同回答同一个问题:**怎么让 RAG 不再傻乎乎地「永远检索一次、检索到啥都硬塞进去生成」,而是带上自我评判、自我纠错、自我路由的能力**。它们是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 到 [[36 Agentic RAG|Agentic RAG]] 的中间形态——把「自评 + 决策」这件事,分别放在生成时、检索后、检索前三个不同位置。

三者都在攻击 Naive RAG 的同一组死穴:**该不该检索不判断(简单常识也强行查)、检索到的料相不相关不评估(噪音照塞)、生成有没有据不核验(放任编造)**。区别在解法的「决策点」位置和「靠训练还是靠外挂」。

![[Self-RAG-CRAG-Adaptive 对比.png]]

## 机制

### Self-RAG:把自评「内化」成模型自己生成的反思 token

**Self-RAG**(Asai et al. 2023,ICLR 2024)的核心思想:**训练一个 LLM,让它在生成普通文本之余,还会吐出特殊的「反思 token(reflection tokens)」来自我控制检索和评判生成**。不是外挂一个评估器,而是把判断能力**烧进模型权重**里。四类反思 token:

| 反思 token | 含义 | 决定什么 |
|---|---|---|
| **Retrieve** | 这一步要不要检索(yes / no / continue) | 是否触发检索 |
| **ISREL**(IsRel) | 检索到的片段与问题**相关**吗(relevant / irrelevant) | 这段料用不用 |
| **ISSUP**(IsSup) | 生成的内容被这段证据**支撑**吗(fully / partially / no support) | 有没有据、是否编造 |
| **ISUSE**(IsUse) | 这个答案对问题**有用**吗(打分 1–5) | 答案整体质量 |

运行时,模型按需检索:遇到需要外部知识的地方生成 `Retrieve=yes`,触发检索后**并行**地对多个候选片段各生成一段答案,每段都带上自己的 ISREL/ISSUP/ISUSE 自评,最后用这些反思 token 的分数做一个 **segment-level beam search**,挑出「相关 + 有据 + 有用」综合最优的那段。

这套机制一举拿下三件事:**自适应检索**(不需要就不查,`Retrieve=no`)、**相关性过滤**(ISREL 把无关片段筛掉)、**忠实度自评**(ISSUP 强制答案有据,直接服务 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]])。代价是**需要专门训练**——要构造带反思 token 标注的数据、微调出会生成这些 token 的模型,不能即插即用到任意现成 LLM。

### CRAG:在检索后外挂一个「检索评估器 + 受控网页补救」

**CRAG / Corrective RAG**(Yan et al. 2024)走相反路线:**不动生成器,在检索和生成之间插一个轻量的检索评估器(retrieval evaluator)**,给召回结果打一个置信度,据此走三条纠正路径。它是「外挂、即插即用、能套到各种现成 RAG 上」的设计。

评估器(一个轻量微调的 T5 级模型)对「问题 + 召回文档」打分,映射到三档置信度:

- **Correct(高置信)**:召回靠谱,但仍可能夹带噪音。走 **decompose-then-recompose(先拆解再重组)** 的知识精炼:把文档切成更细的 knowledge strips(知识条),逐条判相关性,**滤掉无关条、保留并重组相关条**,再喂给生成器。
- **Incorrect(低置信)**:本地语料没召回到对的料(静态有限语料的天然缺陷)。**丢弃本地结果,转而做大规模网页搜索(web search)** 作为知识源补充——用开放网络兜住封闭语料的盲区。
- **Ambiguous(不确定)**:评估器拿不准。**两手都要**——本地精炼结果 + 网页搜索结果一起用,降低单边押错的风险。

CRAG 的精髓是「**检索可能出错,那就在检索后加一道纠错闸**」,且纠错手段包括跳出原语料去网搜。它与 Self-RAG 互补:Self-RAG 把判断内化进生成器(要训练),CRAG 把判断外挂成独立评估器(不动生成器、可套任意 RAG)。

**网页分支不是“搜到就喂”**:远端页面先经过来源 allowlist,再以**不可信数据**隔离解析——页面中的“忽略此前指令”“调用工具”等文字绝不能成为模型指令,防 [[AI 安全/05 Prompt Injection 提示注入|间接提示注入]] 与 [[AI 安全/11 向量与嵌入弱点与 RAG 投毒|检索投毒]]。同时按用户、租户和文档标签校验 ACL;对外部请求限频、限预算和超时,避免 [[AI 安全/12 Unbounded Consumption 成本型 DoS|成本型 DoS]]。每条抽取出的原子事实必须带来源 URL、抓取时间和可定位片段,通过 `citation entailment`(该来源确实支持该断言)后,才可进入生成上下文;高风险结果还应由 [[AI 安全/24 沙箱、最小权限与人审闸门|最小权限与人审闸门]] 复核。

### Adaptive-RAG:在检索前用「复杂度分类器」路由

**Adaptive-RAG**(Jeong et al. 2024,NAACL 2024)的着眼点不同——前两者管「检索到的料好不好」,它管「**这题到底配得上多重的检索策略**」。核心是一个**问题复杂度分类器**,在检索前就把 query 路由到三档策略之一:

- **A 简单(无需检索)**:常识、模型自己就会的题,**直接让 LLM 闭卷回答**,一次检索都不做——省掉无谓的检索开销和噪音。
- **B 中等(单步检索)**:需要外部事实但单跳可解,走**标准的「检索一次→生成」**。
- **C 复杂(多步检索)**:多跳问题,走**迭代式多步检索**(IRCoT 风格,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]]),边推理边检索直到信息足够。

分类器怎么训?论文用了个巧办法:**没有人工标注的复杂度标签**,于是用「各档策略在训练集上的实际成败」来自动打标——简单策略能答对的就标简单、必须多步才答对的就标复杂——再训一个小分类器学这个映射。

Adaptive-RAG 的价值是**效率与准确率的平衡**:简单题不浪费算力(不像 Self-RAG/CRAG 每题都要跑评估流程),复杂题不偷懒(给足多步检索)。它本质是 [[05 Routing|Routing]] 思想在 RAG 检索深度上的应用。

### 三者并置:决策点的位置之差

| | Self-RAG | CRAG | Adaptive-RAG |
|---|---|---|---|
| **决策点** | 生成时(逐 token) | 检索后 | 检索前 |
| **靠什么** | 训练内化的反思 token | 外挂检索评估器 | 前置复杂度分类器 |
| **管的事** | 是否检索 + 相关 + 有据 + 有用 | 召回质量 + 纠错(网搜) | 检索深度(0/1/多步) |
| **要不要改模型** | 要(微调出 reflection token) | 不要(生成器不动) | 不要(只加分类器) |
| **标志动作** | ISSUP 强制有据 | Incorrect→网页搜索 | 简单题不检索 |
| **即插即用** | 否 | 是 | 是(加个分类器) |

一句话记忆:**Self-RAG 在生成时自评(内化)、CRAG 在检索后纠错(外挂 + 网搜)、Adaptive-RAG 在检索前路由(分难度)**。

类比成**查资料写报告**,三者把"质检"卡在三个不同时机:Adaptive-RAG 像**进图书馆前**先掂量"这题要不要查、查多深"(检索前路由);CRAG 像**抱回一摞书后**先翻一遍"靠不靠谱,不行就改去网上搜"(检索后纠错);Self-RAG 像**边写边自查**"这句话书上真有依据吗、有没有用"(生成时自评)。同一件"别瞎写"的事,卡在借书前、借书后、下笔时——这就是三者的本质分野。

## 小数字手算

下面是**工程 policy**,不是三篇论文各自的原始打分式。设一个问题被 Adaptive 分类器判为 A(低复杂度),但它问的是“今天的某项监管要求,请给出处”。令时效、高风险、需出处三个布尔量分别为 $t=1,r=1,c=1$,证据闸门为 $g=\max(t,r,c)$。则 $g=\max(1,1,1)=1$。

闭卷资格为 $a_{\mathrm{closed}}=\mathbb{1}[\mathrm{level}=A](1-g)=1\times(1-1)=0$。所以即使分类器说“A”,也**不能闭卷直答**;必须先检索受控来源。若网页分支找到三条候选事实,其中两条通过 allowlist、ACL、注入隔离和引用蕴含核验,则进入上下文的证据卡数量是 $2$,不是 $3$。这个小例子说明:分类器在优化资源,policy 在约束可不可以省证据。

## 公式推导

把“路由默认值”和“安全硬约束”分两层写,可避免把分类器分数当作授权:

$$
g(q)=\mathbb{1}[\mathrm{time\_sensitive}]
\lor \mathbb{1}[\mathrm{high\_risk}]
\lor \mathbb{1}[\mathrm{requires\_citations}]
$$

$$
a_{\mathrm{closed}}(q)=\mathbb{1}[r(q)=A]\,(1-g(q))
$$

其中 $r(q)\in\{A,B,C\}$ 是 Adaptive-RAG 的复杂度分类。推导很直接:当 $g(q)=1$ 时,$a_{\mathrm{closed}}(q)=0$,低复杂度不能跳过证据;当 $g(q)=0$ 且 $r(q)=A$ 时,才允许闭卷路径。对 CRAG 的网页证据 $e$ 再加准入谓词

$$
\mathrm{admit}(e)=\mathrm{allowlisted}(e)\land\mathrm{ACL}(e)\land\neg\mathrm{injection}(e)\land\mathrm{entails}(e,\mathrm{claim})
$$

只有 $\mathrm{admit}(e)=1$ 的证据卡可进入上下文;这把“能不能查、能用什么、能不能相信”分别交给路由、访问控制和引用核验。

## 可跑最小代码

下面用同一套伪检索把三者的「决策骨架」并列出来,凸显决策点位置之差:

```python
# ---------- Self-RAG:生成时用反思 token 自控(此处用 LLM 模拟 token 判别) ----------
def self_rag(q, llm, retrieve):
    need = llm(f"回答「{q}」需要外部知识吗?只答 yes/no")          # Retrieve token
    if need.strip().lower().startswith("no"):
        return llm(f"直接回答:{q}")                                # Retrieve=no,闭卷
    best, best_score = None, -1
    for psg in retrieve(q, k=4):                                    # 并行评判候选片段
        if "no" in llm(f"片段与问题相关吗(yes/no)?\n{psg}\nQ:{q}").lower():
            continue                                                # ISREL 过滤无关
        ans = llm(f"仅据此片段回答 {q}:\n{psg}")
        sup = llm(f"答案被片段支撑吗(fully/partial/no)?\n片段:{psg}\n答:{ans}")  # ISSUP
        use = int(llm(f"答案对问题有用吗,打1-5分,只回数字:\nQ:{q}\nA:{ans}") or 1)  # ISUSE
        score = use - (2 if "no" in sup.lower() else 0)             # 综合反思分
        if score > best_score:
            best, best_score = ans, score
    return best                                                     # 选最优 segment

# ❌ 错误:把“分类为简单”当成闭卷回答的充分条件,并把网页原文直接拼进 prompt。
def unsafe_adaptive_rag(q, classifier, llm):
    if classifier(q) == "A":
        return llm(f"直接回答:{q}")

# ✅ 策略覆盖:时效、高风险、或用户明确要出处时,必须取得可核验证据。
def requires_grounded_evidence(q):
    return q.is_time_sensitive or q.is_high_risk or q.requires_citations

def approved_web_evidence(q, user, web_search, fetch, allowlist, acl,
                          has_indirect_injection, extract_atomic_claims,
                          citation_entailment, audit):
    approved = []
    for hit in web_search(q, domains=allowlist):                    # ① 只搜批准来源
        if hit.domain not in allowlist or not acl.can_read(user, hit):
            audit("web_rejected", hit.url, reason="allowlist_or_acl")
            continue
        raw_page = fetch(hit.url)                                   # ② 远端文本始终是不可信数据
        if has_indirect_injection(raw_page):
            audit("web_rejected", hit.url, reason="indirect_injection")
            continue
        claims = extract_atomic_claims(raw_page)                    # 只抽取事实与可定位片段,不执行页面指令
        for claim in claims:
            if citation_entailment(claim, hit.url, raw_page):       # ③ 来源真正支持断言
                approved.append({"claim": claim, "citation": hit.url})
    return "\n".join(f"{item['claim']}\n来源:{item['citation']}"  # ④ 仅批准证据可进入上下文
                     for item in approved)

# ---------- CRAG:检索后评估器三分支 + 受控网页补救 ----------
def crag(q, user, llm, retrieve, web_search, **web_guard):
    docs = retrieve(q, k=5)
    conf = llm(f"召回文档对问题的整体质量?只答 correct/incorrect/ambiguous\n{docs}\nQ:{q}")
    conf = conf.strip().lower()
    if conf.startswith("correct"):
        knowledge = refine(docs, q, llm)                            # decompose-then-recompose
    elif conf.startswith("incorrect"):
        knowledge = approved_web_evidence(q, user, web_search, **web_guard)
    else:                                                           # ambiguous:两者都要
        knowledge = refine(docs, q, llm) + approved_web_evidence(q, user, web_search, **web_guard)
    return llm(f"据以下知识回答 {q}:\n{knowledge}")

def refine(docs, q, llm):
    """拆成知识条,逐条留相关的,再重组(去噪)。"""
    strips = [s for d in docs for s in d.split("。") if s.strip()]
    return "。".join(s for s in strips
                     if "yes" in llm(f"「{s}」与「{q}」相关吗?yes/no").lower())

# ---------- Adaptive-RAG:检索前复杂度路由,但服从证据策略 ----------
def adaptive_rag(q, classifier, llm, retrieve, multi_hop_retrieve):
    level = classifier(q)                                           # 训练好的复杂度分类器
    if level == "A" and not requires_grounded_evidence(q):         # 仅低风险、非时效、无需出处才可闭卷
        return llm(f"直接回答:{q}")
    if level == "B":                                                # 中等:单步检索
        return llm(f"据此回答 {q}:\n{retrieve(q, k=5)}")
    if level == "C":
        return multi_hop_retrieve(q, llm)                           # 复杂:多步(IRCoT)
    return llm(f"基于可引用证据回答:{q}\n{retrieve(q, k=5)}")     # A 被策略覆盖后仍取证
```

要点:① Self-RAG 的判断**散在生成过程中**(对每个候选片段逐项 ISREL/ISSUP/ISUSE 评再选优),真实论文里这些是模型直接生成的 token 而非额外 LLM 调用;② CRAG 的判断**集中在检索后一个评估器**,`incorrect` 时可跳出本地语料网搜,但网页证据必须经过准入闸;③ Adaptive-RAG 的判断**前置到检索前**,但分类器只能给出默认路由——时效、高风险或需出处的问题不能因 `A` 而跳过证据。三段 `if` 的位置,就是三篇论文的本质差异。

## 何时用 / 坑

**怎么选(含 policy override)**:

| 场景 | 默认选择 | 不可绕过的策略覆盖 |
|---|---|---|
| 愿意训练、需要把相关性与证据支撑纳入解码 | Self-RAG 的反思 token | ISSUP 只是模型自评;仍要用外部评测或引用核验检查忠实度。 |
| 生成器不能动、担心本地召回质量 | CRAG 的评估器与知识精炼 | 网搜只能走来源 allowlist、间接注入隔离、ACL 与 `citation entailment` 后的证据卡。 |
| 分类器判为 A(低复杂度) | Adaptive-RAG 可闭卷回答 | **时效、医疗/法律/金融等高风险、或用户明确要求出处的问题,不得仅因低复杂度跳过证据**;至少检索受控来源并给出可核验引用。 |
| 同时需控成本、纠正召回与核对生成 | 组合三个决策点 | 先执行硬性 policy,再让路由器决定检索深度;不能把成本优化置于证据义务之前。 |

**坑**:
- **Self-RAG 的训练成本**:要构造反思 token 标注数据并微调,落地门槛高;社区多以 LangGraph 教程版「用 prompt 模拟反思 token」替代,效果打折但能跑。
- **CRAG 的网搜双刃**:网页搜索能补盲区,但也引入**开放网络的噪音、时效与安全风险**;不能把原始网页直接放进 prompt,要执行 allowlist、隔离、ACL 与引用蕴含核验。评估器若误判 correct/incorrect,纠错方向就反了。
- **Adaptive 的分类器是单点**:复杂度判错——把多跳题判成简单题——会直接答错;而把时效/高风险/需出处的问题判成简单,还可能违反证据义务。因此分类器只能优化默认路径,不能覆盖 policy。
- **共同坑:多了一层自评 = 多了延迟和成本**。每道自评/评估/分类都是额外调用,Naive RAG 一次搞定的事现在要多跑几趟。简单稳定场景未必划算。
- **自评不可靠**:让模型/小评估器判「相关吗、有据吗、多复杂」,都可能过度自信或过度保守,和 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的通病一样——自评 prompt/标准要校准。

## 关键事实

- 三者共同攻击 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 的死穴(不判是否检索 / 不评相关性 / 不核忠实度),区别在**决策点位置**:**Self-RAG 生成时、CRAG 检索后、Adaptive-RAG 检索前**。
- **Self-RAG**(Asai 2023,ICLR 2024):训练模型生成 **Retrieve / ISREL / ISSUP / ISUSE** 反思 token,自决是否检索、判相关、判有据、判有用;**需微调**,内化判断;ISSUP 直接服务 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]]。
- **CRAG**(Yan 2024):轻量**检索评估器**给 correct/incorrect/ambiguous,**incorrect 触发网页搜索**,correct 走 **decompose-then-recompose** 知识精炼;**不动生成器、可套现成 RAG**。
- **Adaptive-RAG**(Jeong 2024,NAACL 2024):**复杂度分类器**前置路由到「无需检索 / 单步 / 多步(IRCoT 式,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]])」;在工程中应让时效、高风险与需出处 policy 覆盖“无需检索”的默认分支。
- 三者是 Naive RAG → [[36 Agentic RAG|Agentic RAG]] 的过渡:把「自评 + 纠错 + 路由」做成专门方法;再往前一步(检索成为 agent 可反复调用的工具)就是 Agentic RAG。它们也是 [[13 Modular RAG|Modular RAG]] 里 conditional / loop flow 的具体实例。
- 思想上承接 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](自评 + 再做)与 [[05 Routing|Routing]](按难度/类型分流);代价是每道自评都加延迟和成本。

## 工业界实践

原论文描述的是三种控制点;工程实现可以保留这些控制点,用编排层、评估器和引用核验来实现,也可以为 Self-RAG 专门训练会生成反思 token 的模型。

**可映射的实现组件**
- **LangGraph**:可把 `retrieve / grade_documents / generate / web_search / grade_generation` 表示为状态图的节点与条件边,分别落地“相关吗→用不用”“有据吗→重不重生成”“复杂吗→单步还是多步”。
- **LlamaIndex**:`RouterQueryEngine`(对应 Adaptive 的路由)、各种 `node_postprocessor` + 评估器(对应 CRAG 的检索后过滤)、`RetryQueryEngine`/`RetrySourceQueryEngine`(对应 self-check 重试)。
- **DSPy**:把"自评/路由"写成可优化模块,用 `MIPROv2` 编译出评判 prompt,而非手调;适合把 CRAG 的评估器、Adaptive 的分类器做成可训练组件。
- **真原版 Self-RAG**:`selfrag.github.io` 放出了基于 Llama2 微调的带反思 token 模型权重,可直接推理,但要换模型,不能套任意现成 LLM——这是它落地少的根因。

**一种组合架构(三者组合)**
```
query → [policy:时效/高风险/需出处?] ──是──→ 必须取证
       └─否→ [Adaptive 复杂度分类] ──简单──→ 闭卷直答(不检索)
                              ├─单步──→ 检索一次
                              └─复杂──→ 多步(IRCoT 式)
   检索结果 → [CRAG 评估器] correct→知识精炼 / incorrect→受控网搜 / ambiguous→两者
   网页证据 → allowlist → 注入隔离 → ACL → citation entailment → 上下文
   生成 → [Self-RAG 式 ISSUP 自核] 有据→放行 / 无据→重生成或标不确定
```
这套组合里,**Adaptive 管检索深度、CRAG 管召回纠错、Self-RAG 式 ISSUP 管生成有据**;最先执行的是证据与访问控制 policy。

**规模化与成本**
- 核心权衡:**每多一道自评/评估/分类 = 更多模型调用 = 更高延迟与成本**。Adaptive 路由可在 policy 允许时避免不必要的检索,但不能降低时效、高风险或需出处问题的证据标准。
- CRAG 的网搜分支引入**外部延迟、配额成本与开放网络风险**;要限频、设超时、过滤来源,并在证据进入上下文前完成安全准入。
- 评估器/分类器尽量用**小模型**(T5 级、小 LLM),别用旗舰模型做"判相关吗"这种轻活。

**评估与可观测**
- **Ragas**:`context_precision`(CRAG 精炼后噪音降没降)、`faithfulness`(Self-RAG ISSUP 是否真起作用)、答案正确率(Adaptive 路由有没有把复杂题判错)。
- **TruLens / LangSmith / Phoenix**:trace 每道自评的判定——哪些 query 被判"简单不检索"、哪些召回被判 incorrect 触发了网搜、哪些生成因 ISSUP 失败重生成。**自评系统的调试核心就是看这些分支命中率**。
- 关键监控:Adaptive **分类器混淆矩阵**(复杂判简单 = 直接答错的高危错误)、CRAG **网搜触发率**(过高说明本地语料覆盖差)、Self-RAG **重生成率**。

**踩坑与最佳实践**
- **prompt 模拟版效果打折**:社区 LangGraph 版用 prompt 让普通 LLM 当"反思 token / 评估器",比原版微调弱,但能即插即用——接受这个取舍。
- **评估器误判方向反转**:CRAG 评估器把 incorrect 判成 correct,就不会网搜兜底;把 correct 判成 incorrect,白白丢掉好召回去网搜。评估器质量是 CRAG 上限。
- **分类器是单点故障**:Adaptive 把多跳题判成简单题 → 直接答错。分类器需训练数据(论文用"各策略实际成败"自动打标),线上要持续回流纠错样本。
- **别全场景上自评**:简单稳定、容错高的场景,多套一层自评未必划算,Naive RAG 反而更省。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1:Self-RAG、CRAG、Adaptive-RAG 的本质区别是什么?**
标准答:差在**决策点的位置**和**靠训练还是外挂**。**Self-RAG 在生成时**自评(反思 token,需微调内化);**CRAG 在检索后**纠错(外挂评估器,不动生成器,可套现成 RAG);**Adaptive-RAG 在检索前**路由(复杂度分类器,决定检索深度)。一句话:**生成时自评 / 检索后纠错 / 检索前路由**。
- 追问"哪个即插即用?":CRAG 和 Adaptive(都不改生成器);Self-RAG 要微调出反思 token,不能套任意 LLM。
- 陷阱:把三者当"同类竞品三选一"——其实是三个不同决策点,生产常组合用。

**Q2:Self-RAG 的四类反思 token 分别管什么?**
标准答:**Retrieve**(要不要检索 yes/no/continue)、**ISREL**(检索片段相不相关)、**ISSUP**(生成内容被证据支撑吗 fully/partial/no)、**ISUSE**(答案对问题有用吗 1–5 分)。运行时按需检索 → 并行对候选片段各生成带自评的答案 → 用反思分做 segment-level beam search 选"相关+有据+有用"最优段。
- 追问"ISSUP 服务什么?":直接服务 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 的忠实度——把"答案有没有据"内化成模型自己吐的 token。

**Q3:CRAG 的检索评估器有哪三档?各自怎么处理?**
标准答:**Correct**(高置信)→ decompose-then-recompose 知识精炼(切知识条、滤无关、重组);**Incorrect**(低置信,本地没召到对的)→ **丢弃本地结果转网页搜索**(CRAG 最标志的一招,用开放网络兜封闭语料盲区);**Ambiguous**(不确定)→ 本地精炼 + 网搜**两者都用**降风险。
- 追问"为什么要网搜?":本地语料是**静态有限**的,天然有盲区;网搜补开放知识。
- 陷阱:漏掉 ambiguous 档,或答反 correct/incorrect 的处理。

**Q4:Adaptive-RAG 的复杂度分类器没有人工标签,怎么训?**
标准答:用**各档策略在训练集上的实际成败**自动打标——简单策略(闭卷/单步)就能答对的标"简单",必须多步检索才答对的标"复杂"——再训小分类器学这个映射。三档:A 无需检索(闭卷)/ B 单步检索 / C 多步检索(IRCoT 式,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]])。
- 追问"价值在哪?":效率与准确率平衡——简单题不浪费算力(不像 Self-RAG/CRAG 每题都跑评估),复杂题不偷懒。

**Q5:这三者和 Agentic RAG 是什么关系?**
标准答:它们是 **Naive RAG → Agentic RAG 的中间形态**——把"自评+纠错+路由"做成**固定流程/固定规则**;再往前一步,检索成为 agent 可反复自主调用的工具、由模型在循环里自行调度,就是 [[36 Agentic RAG|Agentic RAG]]。它们也是 [[13 Modular RAG|Modular RAG]] 里 conditional/loop flow 的具体实例。

**Q6(场景题):大量问题是简单常识,少数是多跳,还担心本地语料覆盖不全,怎么设计?**
标准答:先跑**policy override**——时效、高风险或需出处的问题强制进入受控取证;其余问题交给 **Adaptive 路由**(简单闭卷/单步,复杂多步) + **CRAG 式评估器**(incorrect 触发经过 allowlist、注入隔离、ACL 与 citation entailment 的网搜) + 生成端 **ISSUP 自核**。先守证据义务,再路由省成本,最后纠错并核忠实度。

## 知识拓展

**延展视角**
- **Self-RAG(Asai 2023,ICLR 2024)→ 后续**:后续研究会把“何时检索、查什么、何时停止”作为可训练的决策;阅读具体方法时应同时核对原论文 URL、发布日期与实验任务条件,避免把不同数据集的结果直接横比。
- **CRAG(Yan 2024)→ 演进**:纠错思想扩展到**多源融合**(本地 + 网搜 + 工具)、**Agentic CRAG**(把"评估-纠错"交给 agent 多轮),以及与 [[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]] 结合用图补召回盲区。
- **Adaptive-RAG(Jeong 2024,NAACL)→ 同脉**:路由器也可决定使用长上下文、单步 RAG 或多步检索,与 [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]] 的“该用 RAG 还是长上下文”选择相连;无论路由目标是什么,证据与权限 policy 都先于复杂度分类。
- **统一视角**:三者都是"**让 RAG 具备自我评判/纠错/路由**"的不同切面,2024 年起被并入 **Agentic RAG / self-reflective RAG** 大伞下;再上层是 [[03 Agent 核心循环|Agent 核心循环]] 把检索当工具自主调度。

**底层联系**:三者的"自评"通病和 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 一样——模型自评可能过度自信/保守,prompt 与标准要校准;路由思想是 [[05 Routing|Routing]] 的应用;Self-RAG 的 ISREL(判片段相关)本质是用相似度/判别近似"这段料对不对题",可回看 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]] 理解"相关性"如何度量。

**边界与反模式**
- **反模式一:简单稳定场景强上三件套**——每道自评加延迟加成本,Naive RAG 反而更划算。
- **反模式二:用旗舰大模型做"判相关吗/判复杂度"**——轻活用重模型,成本失控;评估器/分类器该用小模型。
- **反模式三:迷信自评**——评估器/分类器误判会让纠错方向反转(CRAG)或直接答错(Adaptive 把复杂判简单);自评不是免费的真值,需校准 + 回流纠错样本。
- **边界**:CRAG 网搜在**离线/内网/强合规**场景不可用(没法访问开放网络),此时退化为只做本地知识精炼;Self-RAG 原版要求能微调/换模型,纯 API 场景只能用 prompt 模拟版。

**相关链接**:它是 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] → [[36 Agentic RAG|Agentic RAG]] 的中间台阶,是 [[13 Modular RAG|Modular RAG]] 的 conditional/loop flow 实例;多步分支复用 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]];ISSUP 直接服务 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]];评估靠 [[18 RAG 评估|RAG 评估]]。

## 来源

- Asai, Wu, Wang, Sil, Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. 首次提交 2023-10-17,ICLR 2024. 任务条件:开放域 QA、推理、事实核验与长文生成。arXiv:2310.11511. <https://arxiv.org/abs/2310.11511>(项目页 selfrag.github.io)
- Yan, Gu, Zhu, Ling. *Corrective Retrieval Augmented Generation*(CRAG). 首次提交 2024-01-29. 任务条件:短文本与长文本生成数据集;轻量检索评估器、知识精炼与网页扩展。arXiv:2401.15884. <https://arxiv.org/abs/2401.15884>(code: HuskyInSalt/CRAG)
- Jeong, Baek, Cho, Hwang, Park. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. 首次提交 2024-03-21,NAACL 2024. 任务条件:不同复杂度的开放域 QA,在无检索、单步与迭代检索策略间路由。arXiv:2403.14403. <https://arxiv.org/abs/2403.14403>(code: starsuzi/Adaptive-RAG)
