[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]] 的本质:把一段可复用的专长**打包成可移植模块**——一个文件夹、一个 `SKILL.md`,描述「这个技能做什么、怎么做」,让任何支持的 agent 拷过去就能用。它的核心机制是**渐进式披露(progressive disclosure)**:技能的元数据常驻上下文,详细指令和引用文件只在真正需要时才加载,从而让上下文保持精简。

## 本质:把专长从「提示」变成「可装的模块」
在 Skills 出现之前,想让 agent 会做某件专门的事(比如「按公司规范填 PDF 发票」),你通常把一大段指令塞进系统提示,或硬编码进应用逻辑。问题是:这段指令**占着上下文**(哪怕这次任务根本用不到它)、**不可移植**(换个项目就得重抄)、**难复用**(团队里每个人各搞一套)。

Agent Skills 的解法是把专长**封装成一个自包含、可移植的目录**:里面有一个 `SKILL.md` 说明书,可能还带几个引用文档、脚本、模板。这个目录拷到哪、放进哪个支持 Skills 的 harness(Claude.ai、Claude Code、Claude Agent SDK),agent 就立刻「会」这门手艺。**专长第一次变得像软件包一样可分发、可版本化、可共享**。

它和 [[16 工具设计与工具层|工具设计与工具层]] 里的「工具」有本质区别——这点后面专门讲;一句话先记住:**工具是「能调的动作」,Skill 是「怎么做的知识」**。

![[Agent Skills(SKILL.md)-目录结构.svg]]

## 来源:Anthropic 的两个时间点
- **Anthropic 2025 年 10 月推出 Skills**:作为让 agent 装载可复用专长的机制,登陆 Claude 各产品面(Claude.ai、Claude Code、Agent SDK)。
- **2026 年 3 月 Anthropic 把 Agent Skills 规范作为开放标准发布**:不再只是 Anthropic 自家的格式,而是开放给整个行业的规范——任何 harness/厂商都能遵循同一套 `SKILL.md` 约定。**这走的是当年把 [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 变成行业标准的同一套路**:先自己做出能打的格式,再开放成标准,靠生态把它变成事实标准。

理解这条「先产品、后开放标准」的路线很重要:它意味着 2026 年起,Skills 大概率会像 MCP 一样,成为跨厂商通用的「能力封装格式」,而不是绑死在某一家。

## 机制:渐进式披露的三层加载
Skills 最精巧的设计是**渐进式披露(progressive disclosure)**——按「便宜的先加载、昂贵的晚加载」分三层,核心目标是**让上下文保持精简**。

![[Agent Skills(SKILL.md)-渐进式披露.svg]]

**第一层:name / description —— 总在上下文(常驻)。** 每个已安装 skill 的 `name` 和 `description` **始终在上下文里**。它们极短(几十个 token),作用只有一个:让模型**知道「有这个技能、什么时候该用它」**。这一层是「目录卡片」,不含任何执行细节。哪怕你装了 50 个 skill,常驻成本也只是 50 张小卡片。

**第二层:SKILL.md 正文 —— 按需加载。** 当模型根据 description 判断「这个技能跟当前任务相关」,它才去**读入 `SKILL.md` 的完整正文**——里面是详细的 step-by-step 指令:怎么做这件事、注意什么、分哪几步。不相关的 skill,正文永远不进上下文。

**第三层:引用文件 / 脚本 —— 再按需加载。** `SKILL.md` 正文里可能指向更多资源:一个 `reference.md`(详细 API、边角知识)、一个 `scripts/fill_form.py`(可执行脚本)、一个 `templates/invoice.html`(模板)。这些**只在真正执行到那一步时才被读取或运行**。

这三层的精髓在于:**越昂贵(越占 token)的内容,加载得越晚、越按需**。模型对每个 skill 的「认知成本」从「几十 token 的卡片」起步,只有真正用到时才升级。这直接服务于 [[20 上下文工程|上下文工程]] 的核心目标——**在有限的上下文窗口里,只让当下相关的信息在场**。Skills 是 progressive disclosure 这一上下文工程理念的标准化落地。

## SKILL.md 的结构与真实示例
一个 Skill 就是一个目录,入口是 `SKILL.md`。它的结构是:**YAML frontmatter(name + description)+ Markdown 正文(指令)+ 可选的引用脚本/资源**。

真实的 `SKILL.md` 文件长这样:

```markdown
---
name: pdf-form-filler
description: 当用户需要把结构化数据填进 PDF 表单(如发票、合同、申请表)时使用本技能。处理字段定位、文本写入、复选框勾选。
---

# PDF 表单填写

## 何时用
用户给你一份 PDF 表单和一组要填的数据,需要生成填好的 PDF。

## 步骤
1. 用 `scripts/inspect_fields.py <pdf>` 列出表单里所有字段名。
2. 把用户数据按字段名对齐;不确定的映射先问用户确认。
3. 用 `scripts/fill_form.py <pdf> <data.json>` 生成填好的 PDF。
4. 复杂边角情况(加密 PDF、嵌套字段)见 `reference.md`。

## 注意
- 日期统一用 ISO 格式 YYYY-MM-DD。
- 金额字段保留两位小数,带千分位见 reference.md 第 3 节。
```

配套的目录:

```
pdf-form-filler/
├── SKILL.md            # 上面这个文件:入口
├── reference.md        # 详细边角知识,正文指向它时才读(第三层)
└── scripts/
    ├── inspect_fields.py
    └── fill_form.py     # 可执行脚本,用到才跑(第三层)
```

注意呼应三层加载:`description` 那一行**总在上下文**(第一层,让模型知道何时该用);`# PDF 表单填写` 以下的正文在**判定相关时**才读(第二层);`reference.md` 和 `scripts/*.py` 在**执行到那一步**才被读/跑(第三层)。

## Skills 与 MCP 的区别(别混淆)
这是最容易混的一对,务必分清:

- **[[17 MCP 模型上下文协议|MCP 模型上下文协议]] = agent 能调的工具/数据源(能力的「接口」)**。MCP 让 agent **连接到外部系统**——一个 MCP server 暴露若干工具(查数据库、调 API、读文件系统),agent 通过标准协议调用它们。MCP 解决的是「agent 能**做**什么动作、能**连**什么系统」。
- **Skills = 怎么做某件事的知识(能力的「说明书」)**。Skill 不提供新工具,它提供的是**用已有工具/能力把一件专门的事做好的步骤性知识**。Skill 解决的是「面对这类任务,该**怎么**一步步做」。

打个比方:MCP 给你一间装满工具的车间(钻、锯、焊),Skill 是一本「怎么用这些工具做一把椅子」的图纸。两者**正交、互补**:一个 Skill 的正文里完全可以指示模型去调某个 MCP 工具。它们都遵循「Anthropic 先做产品、再开放成标准」的路线,但解决的是不同维度的问题。

| 维度 | [[17 MCP 模型上下文协议|MCP 模型上下文协议]] | Agent Skills(SKILL.md) |
|---|---|---|
| 提供什么 | 工具 / 数据源(能调的动作) | 怎么做的知识(步骤说明书) |
| 形态 | 运行的 server + 协议 | 一个目录 + SKILL.md 文件 |
| 解决 | agent 能连/能做什么 | 面对某类任务该怎么做 |
| 加载 | 工具 schema 进上下文 | 渐进式披露三层 |
| 类比 | 一间工具车间 | 一本操作图纸 |
| 关系 | 二者正交;Skill 正文可调用 MCP 工具 | —— |

## 何时用 / 坑
- **何时用 Skills**:你有一类**反复出现、有固定做法、但细节较多**的任务(填表、按规范写报告、跑某条数据流水线),且希望这套做法**可复用、可共享、不长期占上下文**。
- **坑一:description 写不好**。第一层全靠 description 让模型判断「何时该用」。description 含糊,模型要么该用时没触发,要么乱触发。要写清楚**「在什么情况下用本技能」**,这是 Skill 能否被正确召回的命门。
- **坑二:把所有内容都堆进 SKILL.md 正文**。那就废了渐进式披露——正文一旦被读入就全进上下文。把详细 API、长边角知识、模板拆到 `reference.md`/`scripts/`(第三层),正文只留主干步骤。
- **坑三:混淆 Skill 和工具**。需要「新的外部能力」(连数据库、调 API)用 [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 或 [[16 工具设计与工具层|工具设计与工具层]];需要「把已有能力组织成做某事的步骤」才用 Skill。
- **坑四:Skill 不可移植**。如果 Skill 正文里写死了某个特定环境的绝对路径或私有依赖,拷到别处就废了。保持自包含,引用都用相对路径。
- **坑五:版本与信任**。Skill 里能带可执行脚本(第三层),装别人的 Skill 等于跑别人的代码——要走 [[23 Agent Harness 概览|Agent Harness 概览]] 的 Safety 层(沙箱、审查),别盲目执行。

## 2026 关键事实
- **Anthropic 2025 年 10 月推出 Skills**——把可复用专长打包成可移植模块的机制。
- 一个 **SKILL.md** 描述技能做什么,其 **name / description 总在上下文**,详细指令与引用文件**按需加载**——这种**渐进式披露(progressive disclosure)**让上下文保持精简。
- **2026 年 3 月 Anthropic 把 Agent Skills 规范作为开放标准发布**,走的是当年把 [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 变成行业标准的同一套路。
- SKILL.md 结构:**frontmatter(name/description)+ 正文指令 + 可选引用脚本/资源**;加载分三层(**元数据常驻 → 正文按需 → 引用文件再按需**)。
- 与 MCP 的关键区别:**Skills = 怎么做的知识,MCP = 能调的工具**——正交互补,Skill 正文可调用 MCP 工具。
- 关联:Skills 是 [[20 上下文工程|上下文工程]] 中渐进式披露理念的标准化产物;由现代 [[23 Agent Harness 概览|Agent Harness 概览]]/SDK 提供加载与执行;与 [[16 工具设计与工具层|工具设计与工具层]] 互补(知识 vs 动作)。

## 工业界实践

Skills 在 2026 年已从「Anthropic 自家格式」变成跨厂商的能力分发底座,生产落地集中在四个层面。

**主流落地面与生态。**
- **Claude Code / Claude Agent SDK / Claude.ai**:三个一等公民 harness。`SKILL.md` 放在仓库 `.claude/skills/<name>/` 或用户级 `~/.claude/skills/`,启动时只把每个 skill 的 `name`/`description` 注入系统提示(第一层),正文与脚本按需加载。团队把规范类知识(代码风格、发布流程、PII 脱敏步骤、内部 API 调用范式)封进 skill,提交进 git 跟代码一起版本化、code review。
- **Skills 作为开放标准(2026-03)**:类比 MCP 的 registry 化,出现了 skill 市场/索引(类似 npm 之于包),企业内部也建私有 skill registry,按团队/权限分发。一个 skill 等于一个可 `git clone`、可打 tag、可写 CHANGELOG 的「能力包」。
- **与 MCP 协同**:典型生产架构是 **MCP 提供动作(连数据库/调内部服务)+ Skill 提供做法(用这些动作完成某类合规任务的步骤)**。Skill 正文里直接指示模型调用某 MCP 工具,二者在同一 harness 内组合。

**典型企业架构(分层装配)。**
```
harness (Claude Code / Agent SDK)
├── 常驻上下文:所有已装 skill 的 name+description(几十×N token)
├── MCP servers:jira、postgres、internal-deploy-api(动作层)
└── .claude/skills/
    ├── pii-redaction/        # 合规:按公司规范脱敏(知识)
    ├── release-runbook/      # 发布流程:调 deploy MCP 的固定步骤
    └── invoice-filling/      # 业务:填表,带 scripts/ + reference.md
```
**规模化、成本与延迟。** Skills 的核心卖点本身就是省 token:N 个 skill 常驻成本 ≈ N 张小卡片(description),与「把全部指令塞系统提示」相比,首 token 成本随技能数近似常数而非线性膨胀。生产上配合 [[20 上下文工程|上下文工程]] 的 prompt caching——`name/description` 这段常驻前缀天然适合做缓存命中。延迟代价主要在第二/三层的「按需读取」:首次触发某 skill 时多一次文件读取/脚本执行往返;高频 skill 会被 harness 缓存正文。

**可观测与运维。** 生产要埋点:每个 skill 的**触发率、触发后是否真用上、误触发率**。误触发(description 太宽,不该用时被召回)和漏触发(description 太窄,该用时没召回)是两大运维信号,直接反馈去改 description。把 skill 的执行(尤其第三层脚本)纳入 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的轨迹日志。

**踩坑与最佳实践。**
- **description 是召回命门**:写成「**在什么情况下用 + 解决什么**」的判别式句子,而非功能罗列。坏:「PDF 工具」;好:「当用户需要把结构化数据填进 PDF 表单(发票/合同)时使用」。
- **正文只留主干**:详细 API、长边角、模板一律下沉到 `reference.md`/`scripts/`(第三层),否则正文一旦读入就全占上下文,废掉渐进式披露。
- **自包含 + 相对路径**:绝不写死绝对路径或私有依赖,否则不可移植。
- **第三层脚本=别人的代码**:装外部 skill 等于跑外部代码,必须走 [[24 沙箱、最小权限与人审闸门|沙箱与权限护栏]](沙箱、只读默认、敏感操作人审),防供应链投毒。

## 面试高频

**Q1:Agent Skills 和 MCP 的本质区别?为什么需要两个东西?**
标准答:**MCP = 能调的动作(能力的接口),Skill = 怎么做的知识(能力的说明书)**。MCP 让 agent 连外部系统、暴露工具 schema;Skill 不提供新工具,而是用已有工具把一类专门任务做好的步骤性知识。二者正交互补,Skill 正文里可以调 MCP 工具。
- 追问「能不能只用其中一个?」:不能。只有 MCP 没 Skill,模型每次都要现场摸索做法、且做法不可复用;只有 Skill 没 MCP,模型没有连外部系统的实际动作能力。
- 陷阱:把「需要新外部能力(连库/调 API)」的需求用 Skill 解决——那是 MCP/工具层的活。

**Q2:什么是渐进式披露(progressive disclosure)?三层分别是什么、为什么这么分?**
标准答:① `name/description` 常驻上下文(几十 token,让模型知道「有这技能、何时用」);② `SKILL.md` 正文,判定相关时才读(step-by-step 指令);③ 引用文件/脚本,执行到那步才读/跑。**分层原则:越占 token 的内容加载得越晚、越按需**,让上下文保持精简。
- 追问「装 50 个 skill 的常驻成本?」:≈ 50 张 description 小卡片,近似常数,不是 50 份完整指令。
- 追问「这和上下文工程什么关系?」:Skills 是 progressive disclosure 这一上下文工程理念的标准化落地。

**Q3:SKILL.md 的结构是什么?**
标准答:**YAML frontmatter(`name` + `description`)+ Markdown 正文(指令)+ 可选引用脚本/资源**。frontmatter 对应第一层常驻,正文对应第二层按需,引用文件对应第三层再按需。

**Q4(陷阱):把所有内容写进 SKILL.md 正文有什么问题?**
正文一旦被读入就全进上下文,直接废掉渐进式披露的省 token 价值。正确做法:正文只留主干步骤,详细内容下沉到 `reference.md`/`scripts/`。

**Q5:为什么 Anthropic 走「先产品、后开放标准」这条路?**
标准答:这是把 MCP 变成行业标准的同一套路——先自己做出能打的格式验证价值,再开放成规范,靠生态把它变成事实标准。2025-10 推出 Skills,2026-03 作为开放标准发布。

## 知识拓展

**与「微调/RAG/工具」的能力注入谱系。** 给模型注入新能力有四条路,Skills 占了独特生态位:**微调**(改权重,贵、难更新)、**RAG**(注入事实知识,见 [[36 Agentic RAG|Agentic RAG]])、**工具/MCP**(注入动作)、**Skills**(注入「做法/流程知识」且可移植可版本化)。Skills 填的是「程序性知识(procedural knowledge)」这一格,不动权重、不需检索基础设施、即插即用。

**与 [[18 工具检索与动态加载|工具检索与动态加载]] 的呼应。** 工具检索解决「工具太多、schema 塞不下」——动态按相关性加载工具 schema;Skills 的第一层 description 召回本质是同一思想在「知识包」维度的应用:都靠「便宜的索引常驻 + 昂贵的细节按需」对抗上下文膨胀。

**边界与反模式。**
- **反模式:Skill 当数据库用**。Skill 装的是「怎么做」的稳定流程,不是频繁变动的事实数据;后者该走 RAG/记忆([[19 Agent 记忆系统|Agent 记忆系统]])。
- **反模式:一个超大全能 Skill**。description 必然含糊、召回不准。应拆成多个职责单一的 skill,各自 description 清晰。
- **边界:skill 不改模型推理能力**。它只组织已有能力;模型本身不会做的事,写进 skill 也做不出来。

**前沿与延伸(带年份)。**
- **2025-10** Anthropic 推出 Agent Skills;**2026-03** 作为开放标准发布,生态向跨厂商 skill registry 演进。
- 学界相关脉络:**Toolformer(Meta, 2023)**、**Voyager(2023,Minecraft 中自动积累 skill 库并复用)**——Voyager 的「skill library + 自动复用」是 Agent Skills 工程化的思想先声;**ToolLLM / Gorilla(2023)** 则更偏工具调用一侧。
- 与 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进]] 的连接:让 agent **自己沉淀出新 skill**(把一次成功的复杂流程固化成 `SKILL.md` 供后续复用)是自我改进的一条具体路径,呼应 Voyager 的「越玩技能库越大」。

延伸链接:[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]](skill 可被多个 sub-agent 共享)、[[37 Agent 框架对比|Agent 框架对比]](各框架对 skill 的支持差异)、[[39 Agent 开源生态全景|Agent 开源生态全景]]。
