[[06 Jailbreak 越狱|越狱]](Jailbreak)是指**绕过模型的安全对齐**,诱使它产出本该被拒绝的内容(危险指南、仇恨言论、违规信息等)。它针对的不是某个输入解析 bug,而是**安全训练(RLHF/对齐)留下的缝隙**——模型「会做」但「被训练成不该做」的事,被想办法重新撬开。

和 [[05 Prompt Injection 提示注入|提示注入]] 必须分清:**越狱绕对齐**(目标是让模型说出违规**内容**),**注入劫持指令**(目标是让模型听攻击者的**命令**、改变**行为**)。一次攻击可以同时用两者,但它们攻击的是模型的不同侧面。越狱直接挑战 [[03 内容安全与对齐边界|对齐边界]] 划定的红线。

## 为什么安全训练会失效:两大根因

![[sec-越狱手法谱系.svg]]

**Wei, Haghtalab, Steinhardt 2023《Jailbroken: How Does LLM Safety Training Fail?》**(arXiv 2307.02483)给出了至今最有解释力的框架,把越狱成功归到两类失效模式:

- **目标竞争(competing objectives)**:模型被同时训练成「有用/服从指令」和「安全/无害」。当攻击者刻意制造两者冲突的情境(角色扮演、强制开头),模型的「服从」本能压过「安全」本能。
- **泛化错配(mismatched generalization)**:模型的**能力**泛化到了某个域(它看得懂 Base64、罕见语言、超长上下文),但**安全训练没泛化到**那个域。把违规请求挪进这些「安全盲区」,模型就照答不误。

这个框架的价值在于:它把五花八门的越狱手法归到两个机制,从而能**指导设计新越狱**,也指导防御该补哪。

## 手法谱系

按上面两个根因展开,常见手法:

- **角色扮演 / DAN(Do Anything Now)**:让模型扮演一个「无限制 AI」或虚构人格,用人设把违规输出包装成「角色在说」。属**目标竞争**。
- **前缀注入 / 强制开头**:要求模型用「当然,这是……」之类肯定句开头,先让它「答应」,后续难以收回。属**目标竞争**。
- **编码 / 混淆绕过**:把违规请求用 Base64、罕见语言、拆词、同形字编码,绕过基于明文的安全识别。属**泛化错配**。
- **Many-shot jailbreak(多样本越狱)**:在超长上下文里塞入**数百个**「AI 顺从地回答有害问题」的问答示范,靠 in-context learning 把模型带偏。出自 **Anthropic 2024《Many-shot Jailbreaking》**(Anil et al.,NeurIPS 2024):随示范数量增加,越狱成功率按**幂律**上升;长上下文窗口越大越脆弱;对 Claude、GPT、Llama、Mistral 等普遍有效。
- **Crescendo(多轮渐进)**:不直说目标,从一个无害的泛问题开始,**多轮温和提问层层逼近**,每轮借模型自己上一轮的回复继续升级,直到滑到违规内容。出自 **Microsoft 2024《Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack》**(Russinovich et al.,arXiv 2404.01833,USENIX Security 25)。
- **GCG 通用对抗后缀**:不靠人工措辞,而用**梯度搜索**自动找出一段看似乱码的后缀,贴在请求后面即可触发违规;且**通用**(对多种请求有效)又**可跨模型迁移**(在开源模型上搜出的后缀能打 ChatGPT/Claude/Bard)。出自 **Zou et al. 2023《Universal and Transferable Adversarial Attacks on Aligned Language Models》**(arXiv 2307.15043),用 GCG(Greedy Coordinate Gradient)逐 token 优化。

按维度切:**手工**(DAN/Crescendo/编码,可读)vs **自动化**(GCG,乱码但可迁移);**单轮**(DAN/GCG/编码/前缀)vs **多轮**(Crescendo)。

## 对比表

| 手法 | 根因 | 手工/自动 | 轮次 | 出处 |
|---|---|---|---|---|
| 角色扮演 / DAN | 目标竞争 | 手工 | 单轮 | 社区 |
| 前缀注入 / 强制开头 | 目标竞争 | 手工 | 单轮 | Wei 2023 归类 |
| 编码 / 混淆 | 泛化错配 | 手工 | 单轮 | Wei 2023 归类 |
| Many-shot | 长上下文 ICL | 手工 | 单轮(超长) | Anthropic 2024 |
| Crescendo | 渐进诱导 | 手工/可自动 | 多轮 | Microsoft 2024 |
| GCG 对抗后缀 | 对抗优化 | 自动 | 单轮 | Zou 2023 |

## 防御

- **对齐 / 安全训练强化**:RLHF、宪法 AI、对抗训练把更多越狱样本喂进训练——是基线,但**追不上新手法**,且对 GCG 这类自动化对抗后缀鲁棒性有限。
- **输入输出 Guardrails**:独立的分类器在请求进入和回复送出时拦截违规内容,不依赖模型自身的对齐。见 [[21 Guardrails 与输入输出防护|Guardrails]]。
- **针对性补盲**:针对 many-shot 限制/检测超长重复示范;针对 Crescendo 做多轮对话的整体安全判定(而非逐轮);针对编码绕过先解码再过安全审查——即按「泛化错配」的盲区逐个补。
- **持续红队 + 基准测评**:把已知越狱手法做成回归测试,新模型上线前跑 [[23 安全评估基准|越狱基准]](如 AdvBench、HarmBench 等),量化越狱成功率,而非靠主观判断。这与 [[22 AI 红队与对抗测试|红队]] 闭环。

要点:越狱和注入一样**无法彻底封死**——只要模型「有能力」做某事,就存在重新撬开的攻击面;防御目标是抬高成本、覆盖盲区、可观测可回归。

## 工业界实践

越狱防御在工业界已成「**专门的护栏层 + 持续红队回归**」两条腿。没有人靠模型自身对齐单点防御。

**主流护栏产品(按层定位):**

- **Llama Guard 4(Meta,2025 LlamaCon)**:独立的**安全分类器模型**,对「输入请求」和「输出回复」分别打标,判定是否落入预设危害类目(暴力、自残、CSAM、隐私、IP 等);4 代起**统一多模态**(文本 + 图像)。定位:模型旁路的「安检门」,与被保护模型解耦。
- **Prompt Guard 2(Meta,86M / 22M 两档)**:专攻**越狱与提示注入检测**的轻量分类器,放在最前端做快速过滤;小模型保证低延迟。
- **LlamaFirewall(Meta 开源,arXiv 2505.03574)**:**编排框架**,把多个 guard 模型串成纵深防线——PromptGuard 拦注入/越狱、CodeShield 拦不安全代码、AlignmentCheck 查 agent 目标偏移。注意它本身被研究者绕过过(Trendyol Tech 案例:用编码/拆分绕过 PromptGuard 分类),印证「护栏≠铁壁」。
- **NeMo Guardrails(NVIDIA,开源)**:中间件,**input rails** 做越狱/注入检测,**dialog rails** 用 Colang 脚本约束对话流;可跑在任意 LLM 前。
- **AWS Bedrock Guardrails(托管)**:与 Bedrock 深度绑定的托管护栏,denied topics、内容过滤、PII、上下文 grounding 一站式;只服务 Bedrock 模型。
- **Lakera Guard**:实时 API,宣称 98%+ 检出、亚 50ms 延迟、100+ 语言;**2025 年 5 月被 Cisco 收购**并入 AI Defense(连同其 Gandalf 越狱众测数据集)。

**纵深防御架构(典型五层):** ① 输入分类器(Prompt Guard)→ ② 系统提示加固(指令优先级、分隔符)→ ③ 模型自身对齐 → ④ 输出分类器(Llama Guard)→ ⑤ 行为/调用层最小权限 + 监控。任何单层都会被绕,叠起来抬高成本。

**误报 / 延迟权衡(工程核心矛盾):** 护栏阈值调高 → 漏过越狱(漏报);调低 → 误伤正常请求(误报,伤体验)。分类器每请求加一次推理 → 延迟与成本上升。常见折中:**轻量分类器前置快筛 + 重型判定仅对可疑样本**;对 Crescendo 这类多轮攻击,需**整段会话**送判而非逐轮,延迟更高。

**检测响应:** 把越狱命中作为信号上报到 [[25 监控、可观测与事件响应|监控/SIEM]],对高频越狱来源**限流 / 封禁**;新出现的成功越狱抓取为样本,回灌红队回归集。

**最佳实践规则片段:**

```python
# 越狱纵深防御:前置快筛 → 模型 → 后置判定,多轮整体送判
def guarded_chat(session, user_msg):
    # ① 输入侧:轻量分类器拦注入/越狱(Prompt Guard 风格)
    if prompt_guard.is_jailbreak(user_msg, threshold=0.9):
        log_security("input_jailbreak", session.id); return REFUSAL

    # ② 多轮整体判定,防 Crescendo 逐轮滑坡
    full_ctx = session.history + [user_msg]
    if multi_turn_risk(full_ctx) > 0.8:        # 看整段轨迹,不只看本轮
        log_security("crescendo_suspect", session.id); return REFUSAL

    reply = llm.chat(session.system, full_ctx)

    # ③ 输出侧:独立安全分类器(Llama Guard 风格),与对齐解耦
    if not llama_guard.is_safe(reply):
        log_security("output_blocked", session.id); return REFUSAL
    return reply
```

## 面试高频

**攻击原理 + 缓解 + 为何不彻底:** 越狱本质是在「服从」与「安全」两个被同时训练的目标间制造冲突(目标竞争),或把请求挪到安全训练没覆盖的能力域(泛化错配)。缓解靠独立护栏 + 对齐强化 + 红队回归。**不彻底**的根因:只要模型**有能力**产出某内容,攻击面就存在;护栏是概率分类器,必有漏报;新手法(GCG 自动后缀、新编码)持续涌现,防御永远滞后。

- **Q:越狱和提示注入有什么区别?**
  A:越狱**绕对齐、改内容**(让模型说违规的东西);注入**劫持指令、改行为**(让模型听攻击者命令)。一次攻击常两者并用,但攻击的是模型的不同侧面。
  *追问:间接注入算不算越狱?* 不算同类——间接注入是把恶意指令藏进检索/外部内容里让模型执行,核心是「指令来源不可信」,可能**用来**触发越狱,但机制是注入。

- **Q:为什么 Base64 编码能越狱?**
  A:**泛化错配**——模型的解码**能力**泛化到了 Base64,但安全训练主要在明文上做,**没泛化到**编码域,于是违规请求换上编码外衣就绕过了明文安检。缓解:先解码再过安全审查。
  *陷阱:* 只加「禁止 Base64」黑名单没用,攻击者换 ROT13、十六进制、罕见语言、emoji 拆字即可——要补的是「先归一化再审查」的机制,不是堵单个编码。

- **Q:为什么 Many-shot 越狱随上下文变大更危险?**
  A:它靠 in-context learning——上下文里塞**数百个**「AI 顺从回答有害问题」的示范,成功率随示范数**幂律**上升;窗口越大,能塞的示范越多,越脆弱(Anthropic 2024,NeurIPS)。
  *追问:怎么防?* 限制/检测超长重复示范、对长上下文额外加权安全判定;但不能简单截断,会伤正常长文档场景。

- **Q:GCG 后缀为什么能跨模型迁移?**
  A:GCG 用梯度在开源模型上搜出通用对抗后缀,这些后缀利用了对齐模型间**共享的表征弱点**,因此在 GPT/Claude/Bard 上也常生效(Zou 2023)。这说明对齐脆弱性有共性,黑盒不等于安全。
  *陷阱:* 面试别说「GCG 是乱码所以容易过滤」——它确实可被困惑度检测抓到,但攻击者可加可读性约束生成「自然」后缀绕过该检测。

## 知识拓展

- **OWASP LLM Top 10(2025)**:越狱主要落在 **LLM01 Prompt Injection** 条目下(2025 版把越狱归为绕过系统指令/对齐的一种),与 [[03 内容安全与对齐边界|对齐边界]]、[[21 Guardrails 与输入输出防护|Guardrails]] 强相关。
- **MITRE ATLAS**:越狱在战术上属 **Defense Evasion / LLM Jailbreak** 技术;ATLAS 与 Center for Threat-Informed Defense 的 Secure AI 项目用于快速共享演进中的越狱手法。把越狱事件映射到 ATLAS 技术 ID,便于跨团队沟通与红队覆盖。
- **NIST AI RMF + GenAI Profile(AI 600-1,2024)**:200+ 条风险管理建议,把越狱归入「可信赖性—安全/鲁棒性」维度,提供治理侧的可执行动作清单。
- **红队工具(自动化越狱)**:**garak**(NVIDIA,LLM 漏洞扫描器,v0.15 起含 system-prompt-extraction、多轮 GOAT、Agent-breaker probe)做面扫;**PyRIT**(Microsoft,多轮自适应攻击引擎,2025 加 AI Red Teaming Agent)做深度多轮;**Promptfoo** 做回归。三者组合是完整 kill chain。详见 [[22 AI 红队与对抗测试|AI 红队]]。
- **对抗演化**:护栏出 → 攻击者用编码/拆分绕分类器(LlamaFirewall 被绕案例)→ 护栏补归一化 → 攻击转多轮 Crescendo 绕逐轮判定 → 防御转整段会话判定。攻防螺旋上升,没有终态。
- 兄弟链:[[05 Prompt Injection 提示注入|提示注入]](常与越狱组合)、[[07 敏感信息泄露与 System Prompt 泄露|信息泄露]](越狱常用于套系统提示)、[[09 数据与模型投毒|数据与模型投毒]](Sleeper Agents 是「天生越狱」的后门)、[[23 安全评估基准|越狱基准]]。

## 关键事实

- 越狱 = **绕过安全对齐**,产出本应被拒的内容;攻击的是安全训练的缝隙,非输入解析 bug。
- 与 [[05 Prompt Injection 提示注入|提示注入]] 的区别:越狱**绕对齐改内容**,注入**劫持指令改行为**;常组合使用。
- 两大失效根因(**Wei, Haghtalab, Steinhardt 2023《Jailbroken: How Does LLM Safety Training Fail?》**,arXiv 2307.02483):**目标竞争**(服从 vs 安全打架)、**泛化错配**(能力覆盖、安全没覆盖的域)。
- **GCG 通用对抗后缀**:**Zou et al. 2023《Universal and Transferable Adversarial Attacks on Aligned Language Models》**,arXiv 2307.15043;梯度搜索、通用、可跨模型迁移。
- **Many-shot jailbreaking**:**Anthropic 2024**(Anil et al.,NeurIPS 2024);长上下文塞数百违规示范,成功率随示范数**幂律**上升。
- **Crescendo 多轮渐进**:**Microsoft 2024**,arXiv 2404.01833(Russinovich et al.,USENIX Security 25);温和多轮、借模型自身回复层层升级。
- 防御:对齐强化 + 独立 [[21 Guardrails 与输入输出防护|Guardrails]] + 针对性补盲 + 持续红队与 [[23 安全评估基准|越狱基准]] 回归;无法彻底封死。
- 越狱挑战的是 [[03 内容安全与对齐边界|对齐边界]] 划定的红线。
