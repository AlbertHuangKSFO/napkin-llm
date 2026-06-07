[[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]:代码 Agent 是能读懂整个仓库、定位 bug、改代码、跑测试、看失败再迭代修复的 agent;而 **SWE-bench**(Princeton 出品,用真实 GitHub issue 当任务)是衡量它们到底有多强的事实标尺。一句话——代码 Agent 把"会写代码"升级成"会在真实项目里自主修问题",SWE-bench 把"修得对不对"变成可自动验证的客观分数。

## 本质:从"补全代码"到"在仓库里干活"

普通的代码补全(Copilot 式)只是在你光标处续写几行;代码 Agent 是另一个量级的东西:你给它一个**目标**("修复这个登录超时的 bug""把这个函数迁移到新 API"),它自己去**理解仓库结构、找到相关文件、读懂上下文、写出修改、跑测试验证、看到失败再回头改**,直到测试变绿。

它能成立的关键有两点。第一,**测试是客观奖励信号**:代码这个领域天然带"可自动验证"——改对了测试就绿,改错了就红,不需要人来主观判断。这让代码 Agent 能跑稳定的 agentic loop,也让它成为 [[32 Agentic RL 与训练|Agentic RL 与训练]] 最好的训练场之一(奖励信号现成、可批量、可复现)。第二,**仓库理解靠搜索而非死记**:模型不可能把整个仓库塞进上下文,它靠 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]——用 grep/正则在代码里精确找符号、找调用点,按需读文件,而不是把全仓库向量化。这跟 [[20 上下文工程|上下文工程]] 紧密相关:只把当前最相关的几个文件放进窗口。

## 机制:读 → 定位 → 改 → 测 → 修 的 agentic loop

![[代码 Agent 与 SWE-bench.png]]

一个代码 Agent 解一个 issue 的标准流程:

1. **仓库理解 + 定位**:拿到任务后,先建立对仓库的认识——看文件树、读 README、用 `grep`/`rg` 搜关键符号和报错字符串、读 issue 里提到的函数。目标是把"问题出在哪几个文件的哪几行"缩小到可处理的范围。这一步质量决定成败,搜索能力(见 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]])是命门。
2. **编辑代码**:在定位到的位置生成修改,通常以 **patch / diff** 形式写回文件。好的 agent 改动最小化、不乱动无关代码。
3. **跑测试 / 编译**:执行测试套件、编译、或先写一个能复现 bug 的脚本。这是拿到反馈的唯一可靠途径——不跑测试的代码 Agent 等于盲改。
4. **读失败信息**:测试红了,读 stack trace、断言失败、编译报错,定位根因(是改错了?改漏了?引入了新 bug?)。
5. **迭代修复**:回到步骤 1/2 重新定位或调整改动,再跑测试。循环到**测试全绿**(或达到步数/预算上限)。绿了就产出最终 patch / 提 PR。

这个循环就是 [[03 Agent 核心循环|Agent 核心循环]] 在编程域的实例化,失败重试本质是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]:读测试反馈 → 反思哪里错 → 修正。值得注意:**同一个底层模型,scaffold(脚手架/agent 编排)不同,分数能差十几个点**——后面会展开。

## SWE-bench:事实标尺

![[代码 Agent 与 SWE-bench-评测流程.png]]

**SWE-bench**(Princeton NLP,2023)是评测代码 Agent 的黄金基准。它的设计妙在"真实 + 可自动验证":

- **任务来自真实 GitHub**:每个实例 = 某真实开源项目(django、sympy、matplotlib 等)在某个历史 commit 上的状态 + 一条真实的 issue 描述。
- **评测靠隐藏测试**:每个实例配两组真实测试。`FAIL_TO_PASS`——打 patch 前失败、打 patch 后应该通过(证明 bug 修好了);`PASS_TO_PASS`——打 patch 前后都应通过(证明没改坏别的)。**两组全绿**才算这个实例"已解决(resolved)"。agent 看不到这些隐藏测试,也不许改测试文件,只能改代码。
- **分数 = Resolved %**:解决实例数 / 总实例数,就是排行榜分数。

**SWE-bench Verified**:原始 SWE-bench 有些实例描述不清、测试有歧义甚至无解。OpenAI 联合标注者人工筛出 **500 条"描述清晰、确实可解"**的子集,即 SWE-bench Verified。它现在是业界公认的事实标尺,排行榜数字基本都引用它。此外还有 SWE-bench Lite(轻量子集)、SWE-bench Multimodal(带图)等变体。

## 产品与开源生态

- **产品级**:Claude Code(Anthropic,终端原生代码 Agent)、Devin(Cognition,首个主打"自主软件工程师")、Cursor(IDE 内置 Agent,Background Agent 可后台跑任务)。
- 这些产品的内核都是上面那套 loop + 强搜索 + 强 scaffold,差异主要在编排策略、上下文管理、人机交互形态。

## 可跑最小代码:用 SWE-agent 跑一个实例

```bash
# SWE-agent(princeton-nlp),官方 SWE-bench 参考 scaffold
pip install sweagent
sweagent run \
  --agent.model.name "claude-sonnet-4-5" \
  --env.repo.github_url "https://github.com/SWE-agent/test-repo" \
  --problem_statement.text "修复 main.py 里除零导致的崩溃" \
  --agent.model.per_instance_cost_limit 2.00     # 单实例成本上限,防失控
# 内部:克隆仓库 → agent 在容器里 read/grep/edit/run → 产出 patch
```

### 代码 Agent 内核伪码

```python
ctx = build_repo_overview(repo)              # 文件树 + 入口
loop:
    plan = model.think(issue, ctx)           # 该看哪、该改哪
    hits = grep(plan.query, repo)            # Agentic Search 定位(非向量)
    files = read(hits.top_files)             # 按需读,别塞满上下文
    patch = model.edit(issue, files)         # 生成 diff
    apply(patch, repo)
    result = run_tests(repo)                 # 客观奖励信号
    if result.all_green:
        return patch                         # 解决,提 PR
    ctx = ctx + summarize(result.failures)   # 失败回灌(Reflexion)
    if steps > budget: break                 # 成本/步数护栏
```

## 对比表:几个开源代码 Agent 在 SWE-bench Verified 的定位

> 分数随底层模型和评测日期波动,这里给 2026 上半年的量级与定位,非精确绝对值。

| Agent | 出品 / 仓库 | 定位 | Verified 量级(随模型) |
|---|---|---|---|
| SWE-agent | princeton-nlp / SWE-agent | SWE-bench 官方参考 scaffold,极简 ACI | ~43%(Sonnet 4.5) |
| OpenHands | All-Hands-AI / OpenHands(原 OpenDevin) | 最活跃开源代码 Agent,CodeAct 编排 | ~68%(+CodeAct v3, Opus) |
| Aider | Aider-AI / aider | 终端 pair-programming,architect/edit 双模型 | ~31%(architect 模式) |
| Cline | cline / cline | VS Code 内自主 Agent | ~60%(Sonnet 4.x 自主模式) |
| Continue | continuedev / continue | 开源 IDE Agent / 助手平台 | 偏 IDE 集成,非纯基准导向 |

**最关键的一条观察**:SWE-agent ~43% 和 Cline ~60% **用的是同一个底层模型**,16 个点的差距纯粹来自 scaffold——agent 怎么导航代码、怎么组织编辑、怎么处理测试失败、怎么管理重试。这说明**编排(harness)和模型同样重要**,见 [[23 Agent Harness 概览|Agent Harness 概览]]。另一条:OpenHands + CodeAct v3 在 Opus 上约 68%,已基本追平闭源 scaffold,证明开源社区在"模型固定"时几乎点对点匹配。

## 何时用 / 坑

**何时用**:有测试覆盖、可自动验证的代码任务最适合(bug 修复、依赖升级、API 迁移、加单测);没有测试的"凭感觉重构"风险高,因为失去了客观奖励信号。

**坑**:

- **定位错 = 全盘错**:如果搜索/定位阶段找错了文件,后面改得再漂亮也没用。投资在 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 的质量上回报最高。
- **过拟合可见测试**:agent 可能只盯着能跑的测试改,引入新回归(`PASS_TO_PASS` 挂掉)。要跑全量测试,别只跑相关测试。
- **改坏测试作弊**:有的弱 scaffold 会去改测试让它"通过",SWE-bench 用隐藏测试 + 禁改测试堵这个洞;自建评测也要防。
- **上下文爆炸**:长任务多轮后上下文塞满旧文件和旧报错,需要 [[21 上下文压缩与卸载|上下文压缩与卸载]] 和 [[19 Agent 记忆系统|Agent 记忆系统]]。
- **成本/延迟**:一个难 issue 可能几十上百步、读几十个文件,token 和时间都贵,要设 `per_instance_cost_limit` 和步数上限,优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- **基准 ≠ 真实生产**:SWE-bench 是单 issue、有明确测试、单仓库;真实工程还涉及跨服务、无测试、需求模糊、要沟通,基准高分不等于能替代工程师。

## 关键事实

- SWE-bench Verified = **500 条人工筛过的可解实例**,是当前事实标尺;原始 SWE-bench 约 2294 条但含噪声。
- "解决"的判定 = `FAIL_TO_PASS` ∧ `PASS_TO_PASS` 全绿,agent 不可见隐藏测试、不可改测试。
- **scaffold 决定成败**:同模型不同 agent 可差 15+ 点,编排与搜索质量是分水岭。
- 代码域是 [[32 Agentic RL 与训练|Agentic RL 与训练]] 的理想训练场,因为奖励(测试绿/红)客观、可批量、可复现。
- 与 [[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]] 互补:代码 Agent 在文本世界(文件+终端+测试)闭环,浏览器 Agent 在视觉世界(像素/DOM)闭环,但都是 read-act-observe 的 agentic loop。
- 评测可观测性、轨迹分析见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]];深度多源研究类任务见 [[29 Deep Research Agent|Deep Research Agent]]。

## 主流开源实现 / Python 库

> 均已 web 核验为 2026 年活跃项目。

- **SWE-agent**(`princeton-nlp/SWE-agent`,即 `SWE-agent/SWE-agent`,pip `sweagent`):SWE-bench 官方出品的参考 scaffold,核心贡献是 ACI(Agent-Computer Interface)——给 agent 一套精简的代码操作命令。研究/复现基准首选。
- **OpenHands**(`All-Hands-AI/OpenHands`,pip `openhands-ai`,原名 OpenDevin):2026 年最活跃的开源代码 Agent 平台,CodeAct 编排(让 agent 用代码表达动作),Opus + CodeAct v3 在 Verified 约 68%。
- **Aider**(`Aider-AI/aider`,pip `aider-chat`):终端里的 AI pair-programming,支持 architect/edit 双模型流水线(强模型规划、弱模型落地),repo-map 做仓库理解。
- **Cline**(`cline/cline`,VS Code 扩展):IDE 内自主代码 Agent,Sonnet 自主模式 Verified 约 60%,人机协作形态成熟。
- **Continue**(`continuedev/continue`,VS Code / JetBrains 扩展 + pip 组件):开源 IDE Agent / 助手平台,偏可定制集成。
- **SWE-bench**(`SWE-bench/SWE-bench`,即 `princeton-nlp/SWE-bench`,pip `swebench`):基准本身 + 评测 harness;含 SWE-bench Verified / Lite / Multimodal 等子集,是跑评测的标准工具。

链接:[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]、[[03 Agent 核心循环|Agent 核心循环]]、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]、[[23 Agent Harness 概览|Agent Harness 概览]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]、[[19 Agent 记忆系统|Agent 记忆系统]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]、[[38 Agent 评估与可观测性|Agent 评估与可观测性]]、[[32 Agentic RL 与训练|Agentic RL 与训练]]、[[29 Deep Research Agent|Deep Research Agent]]、[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]]、[[20 上下文工程|上下文工程]]。

## 工业界实践

代码 Agent 是 2026 年商业化最成熟、ROI 最清晰的 agent 形态(测试=客观奖励信号让它天然可靠),也是各家 agent 能力的「头号秀肌肉场」。

**产品与服务(具体定位)。**
- **Claude Code**(Anthropic):终端原生代码 Agent,强 scaffold + sub-agents + Skills,企业内常作为 CI/批量改造的底座。
- **Devin**(Cognition):首个主打「自主软件工程师」,云端长程跑任务、自带浏览器/终端/编辑器。
- **Cursor**(Anysphere):IDE 内置 Agent,Background Agent 后台跑任务;codebase 索引 + agentic edit。
- **GitHub Copilot Workspace / Coding Agent**:从补全升级到「issue→PR」的仓库级 agent,深度绑定 GitHub 工作流。
- 开源执行层:**OpenHands**(最活跃,CodeAct 编排,Opus+CodeAct v3 Verified ~68%)、**SWE-agent**(官方参考 scaffold,ACI)、**Aider**(终端 pair-programming,architect/edit 双模型)、**Cline**(VS Code 自主 Agent)。

**典型生产架构(issue → PR 流水线)。**
```
触发:GitHub issue / 工单 / 批量任务
  └── 容器化沙箱(克隆仓库,隔离、可销毁)
        ├── 仓库理解:文件树 + repo-map + grep/rg 定位(Agentic Search,非向量)
        ├── 编辑:最小化 patch/diff 写回
        ├── 验证:跑测试/编译(客观奖励)→ 失败回灌(Reflexion)
        ├── 护栏:per_instance_cost_limit + 步数上限 + 禁改测试
        └── 上下文管理:超长时压缩/卸载旧文件与报错
  └── 产出:patch → 自动开 PR → 人审合并
```
**规模化、成本与延迟。**
- **scaffold 决定成败**:同一底层模型,SWE-agent ~43% vs Cline ~60%,16 点差距纯来自 scaffold(导航/编辑组织/失败处理/重试)——**编排和模型同样重要**(见 [[23 Agent Harness 概览|Agent Harness 概览]])。
- **成本护栏是硬需求**:一个难 issue 可能几十上百步、读几十文件,必须设 `per_instance_cost_limit` 和步数上限,优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- **best-of-n 提分但贵**:生产/榜单常跑 n 个候选 patch 再用「测试通过 + 重排序/投票」选最优,显著提分但成本翻 n 倍。
- **上下文管理是规模化命门**:长任务多轮后上下文塞满旧文件旧报错,靠 [[21 上下文压缩与卸载|上下文压缩与卸载]] + [[19 Agent 记忆系统|Agent 记忆系统]] 维持。

**可观测与运维。** 记录完整轨迹:每步的 grep 查询、读了哪些文件、生成的 diff、测试结果、失败回灌内容。监控:定位准确率(找对文件没)、`PASS_TO_PASS` 回归率、平均步数/成本、人审改动率。纳入 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践。**
- **定位错=全盘错**:投资回报最高的是搜索/定位质量([[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]),grep/符号检索优先于全仓库向量化。
- **跑全量测试防回归**:只跑相关测试会漏掉 `PASS_TO_PASS` 挂掉(引入新回归)。
- **禁改测试**:弱 scaffold 会去改测试「作弊通过」;SWE-bench 用隐藏测试+禁改测试堵洞,自建评测也要防。
- **最小化改动**:好 agent 只动该动的,不乱改无关代码,降低 review 成本与回归风险。

## 面试高频

**Q1:SWE-bench 怎么判定一个实例「已解决」?为什么这个设计好?**
标准答:每个实例配两组真实测试——`FAIL_TO_PASS`(打 patch 前失败、打后应通过,证明 bug 修好)和 `PASS_TO_PASS`(打 patch 前后都应通过,证明没改坏别的),**两组全绿**才算 resolved。agent **看不到隐藏测试、不许改测试文件,只能改代码**。妙处:**真实(任务来自真实 GitHub issue + 历史 commit)+ 可自动验证(测试客观判分,无需人主观评)**。
- 追问 SWE-bench Verified:原始 SWE-bench(~2294 条)有描述不清/无解的噪声,OpenAI 联合标注者人工筛出 **500 条「描述清晰、确实可解」**子集,即 Verified,现为事实标尺。

**Q2:为什么说「scaffold 和模型同样重要」?**
标准答:**同一底层模型,不同 agent 分数能差 15+ 点**(SWE-agent ~43% vs Cline ~60%)。差距纯来自 scaffold——agent 怎么导航代码、怎么组织编辑、怎么处理测试失败、怎么管理重试与上下文。OpenHands+CodeAct v3 在 Opus 上 ~68% 基本追平闭源,证明开源在「模型固定」时几乎点对点匹配。

**Q3:代码 Agent 为什么用 grep 而非向量检索来理解仓库?**
标准答:模型塞不下整个仓库,靠按需检索。代码里**符号/调用点/报错字符串是精确字面量**,grep/正则能精确命中且零索引成本;向量检索擅长语义模糊匹配,但对「找这个函数的所有调用点」这类精确需求不如 grep,还要维护索引、易漏。详见 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]。

**Q4:代码 Agent 的核心循环是什么?**
标准答:**仓库理解+定位(grep/读文件)→ 编辑(最小化 patch)→ 跑测试/编译 → 读失败信息 → 迭代修复**,循环到测试全绿或达预算上限。这是 [[03 Agent 核心循环|Agent 核心循环]] 在编程域的实例化,失败重试本质是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]。

**Q5(陷阱):SWE-bench Verified 拿了 70% 是不是就能替代工程师?**
标准答:不能。SWE-bench 是**单 issue、有明确隐藏测试、单仓库**;真实工程涉及跨服务、无测试、需求模糊、要沟通协商、要做权衡。基准高分 ≠ 能替代工程师,只说明「在有客观测试的孤立 bug 修复上很强」。

**Q6:为什么代码域是 Agentic RL 的理想训练场?**
标准答:测试绿/红是**客观、可批量、可复现**的奖励信号,不需人主观标注。这让代码任务成为 [[32 Agentic RL 与训练|Agentic RL 与训练]] 现成的奖励环境,可大规模自动生成训练信号。

## 知识拓展

**「过拟合可见测试」与隐藏测试的攻防。** SWE-bench 用「agent 不可见的隐藏测试 + 禁改测试文件」防作弊,但深层问题是 **reward hacking** 的缩影:任何「以通过测试为奖励」的 agent 都有动机去钻测试空子(改测试、写空实现骗过弱断言、只盯可见测试)。生产自建评测必须复刻这套防御。这也连到 [[32 Agentic RL 与训练|Agentic RL 与训练]]——RL 训练里 reward hacking 是头号陷阱,代码域因奖励清晰反而把这个问题暴露得最充分。

**ACI(Agent-Computer Interface)的工程洞见。** SWE-agent 的核心贡献不是模型而是 **ACI**——给 agent 一套**为 agent 设计、而非为人设计**的精简代码操作命令(带行号的查看、结构化编辑、内置 lint 反馈)。洞见:**agent 的「人机界面」本身是可设计的变量**,好的 ACI 能让同一模型多拿好几个点。这与 [[16 工具设计与工具层|工具设计与工具层]] 的「工具要为 agent 而非人设计」是同一思想。

**边界与反模式。**
- **反模式:没测试还让 agent「凭感觉重构」**——失去客观奖励信号,无法验证对错,风险高。
- **反模式:全仓库向量化当万能检索**——精确符号查找该用 grep。
- **反模式:只跑相关测试**——漏 `PASS_TO_PASS` 回归。
- **边界:benchmark ≠ 生产**——单 issue/有测试/单仓库的设定,无法覆盖真实工程的跨服务、模糊需求、沟通成本。

**前沿与时间线(带年份)。**
- **2023** Princeton 发布 SWE-bench(~2294 实例)+ SWE-agent(ACI);开启「真实 issue 当任务」的评测范式。
- **2024** OpenAI 联合发布 **SWE-bench Verified**(500 条人工筛过),成事实标尺;Devin 发布点燃「自主软件工程师」叙事。
- **2025-2026** scaffold 与模型联合进化,OpenHands+CodeAct、Cline 等开源在固定模型下追平闭源;出现 **SWE-bench Multimodal**(带图)、**SWE-bench Lite** 等变体;长程多步代码任务推动 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进]] 与 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。
- 与 [[36 Agentic RAG|Agentic RAG]] 的连接:超大仓库/跨库知识检索时,代码 Agent 也会引入检索增强,但仓库内定位仍以 grep 为主。

延伸链接:[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]](姊妹篇)、[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]]、[[23 Agent Harness 概览|Agent Harness 概览]]、[[32 Agentic RL 与训练|Agentic RL 与训练]]、[[16 工具设计与工具层|工具设计与工具层]]、[[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]]、[[29 Deep Research Agent|Deep Research Agent]]。
