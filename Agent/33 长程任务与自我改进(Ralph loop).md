[[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]] 的本质是把一个 agent 套进**反复运行同一个目标 prompt 的 while 循环**:每一轮都是一次全新的、上下文清零的短会话,读固定 prompt + 磁盘上的进度状态 → plan 一小步 → act → reflect → 落盘,然后**进程退出、再从顶上重启**;靠"无限次重启 + 文件系统当记忆 + 测试当反压"把一个超出单次上下文窗口的长任务一点点磨完——而不是奢望一个长会话一次想对。

这是 Geoff Huntley 在 2025 年底命名并引爆的 **"Ralph"(Ralph Wiggum)技法**:名字取自《辛普森一家》里那个一头撞向门框还乐呵呵喊"我在帮忙!"的角色,精髓就一句话——**"卡住了?再跑一次。"** 它是 [[03 Agent 核心循环|Agent 核心循环]] 在**长程**维度上的极简变体,也是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 自我修复思想落到工程上的最糙最有效的形态。

## 本质:不是让一个会话跑很久,而是让短会话被重启很多次

长程任务(写完一个完整项目、跑通一整套 SWE-bench 任务、做一份多阶段研究报告)的根本矛盾是:**任务的工作量远超模型单次上下文窗口能装下的状态量**。常规思路是"让一个 agent 会话一直跑下去",但这条路会撞上四堵墙(见下图):上下文爆炸、目标漂移、状态丢失、错误累积。

Ralph 的反直觉之处在于:**它不解决"如何让一个长会话稳住",而是干脆放弃长会话**。每一轮都是一个**全新的进程、全新的上下文**,读同一份固定的目标 prompt,加上磁盘上当前的代码与 `TODO.md`,只推进**一小步**,把成果 `git commit` 落盘,然后**退出**;外层一个 bash `while` 死循环立刻把它再启动一遍。

于是关键转移发生了:**跨轮的"记忆"不在对话历史里,而在文件系统上**——代码本身、`TODO.md`、git 历史、测试结果。每一轮 agent 都"失忆"地醒来,但它能从磁盘读到"上一个我做到哪了",于是接着往下推。这正是它能突破上下文窗口的原因:**任意长的任务,其状态都被卸载到了磁盘,单轮上下文永远只装"当前这一小步需要的东西"**(深度联动 [[21 上下文压缩与卸载|上下文压缩与卸载]] 与 [[19 Agent 记忆系统|Agent 记忆系统]] 的外部记忆思想)。

![[长程任务与自我改进(Ralph loop).svg]]

一句话对比:[[10 Plan-and-Execute|Plan-and-Execute]] 是**一次性**把全程计划想清楚再逐步执行;Ralph 恰恰相反,它**不预排全程**,每轮只在当下挑"最该做的一件事"做掉——计划是涌现出来的,不是预先排好的。这让它在**高度不确定、边界模糊、规格会变**的任务上反而更鲁棒。

## 机制:3 阶段 / 2 prompt / 1 循环

Huntley 把成熟的 Ralph 实践归纳成一个漏斗:**1 个循环、2 个 prompt、3 个阶段**。

**1 个循环**:最外层就是一个 shell `while` 死循环,反复调起 coding agent(Claude Code / aider / OpenHands / Codex 等),每次都是干净进程。

**2 个 prompt**(关键工程纪律,别让一个 prompt 既规划又施工):
- **PLANNING prompt**:只做**差距分析**——拿"规格 / 目标"对比"当前代码",输出一份**带优先级的 TODO 清单**。它**不写实现、不提交**,纯产出"接下来该干什么"。
- **BUILDING prompt**:假定 TODO 已存在,从中**挑一项**、实现它、**跑测试(这是反压 backpressure)**、测试过了才 `git commit`。

**3 个阶段**对应单轮内部的 plan → act → reflect:
1. **plan**:读固定 prompt + 规格 + `TODO.md`,选出**这一轮最该推进的一件事**(刻意只选一件,避免一轮咬太多导致上下文爆 + 半途而废)。
2. **act**:实际改代码 / 加文件 / 跑命令,把这件事做出来。
3. **reflect + 落盘**:**跑测试**作为客观反压——过了才 `git commit` 并更新 `TODO.md`(勾掉做完的、记下新发现的);**没过则不提交**,把失败信息留在磁盘/下一轮的输入里,下一轮醒来会先看到"上轮没过,先修它"。然后进程退出,回到循环顶部。

这套机制为什么 work,核心是**两个外部信号替代了"长会话的连续记忆"**:
- **文件系统 = 持久状态**:跨轮一切靠磁盘,单轮失忆无所谓。
- **测试 = 反压**:防止 agent"自我感觉良好地"把错误代码提交进去、让后续轮在烂地基上盖楼。没有反压的 Ralph 会快速发散成一堆能跑通编译却逻辑全错的代码。

**自我改进 / 自我修复**就内生于此:每一轮都对"上一轮留下的现实(代码 + 失败测试)"做一次反思与修正,等价于把 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的"试错→反馈→改进"做成了**跨进程、跨轮次的外循环**——反馈不靠 verbal memory 塞回上下文,而是直接写进代码与测试结果,下一轮自然读到。

## 长程的四个杀手与对策

![[长程任务与自我改进(Ralph loop)-checkpoint恢复.svg]]

| 杀手 | 现象 | Ralph 的对策 |
|---|---|---|
| **上下文爆炸** | 历史越滚越长、超窗、token 飙升、注意力被稀释 | 每轮**清零**,新轮只载摘要 + `TODO.md` + 相关代码片段([[21 上下文压缩与卸载|上下文压缩与卸载]]) |
| **目标漂移** | 忘了最初要做啥、在局部反复打转、越改越偏 | 每轮**重读同一份固定规格 + 验收标准**当锚,目标永不漂走 |
| **状态丢失** | 崩溃 / 超时 / OOM,进度全没、从头再来 | git commit 即**检查点**,崩了从最近 commit 续(durable execution 思想,见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]) |
| **错误累积** | 早期小错被后续轮建在其上、复合放大 | **测试反压**:不过不提交;下一轮优先修上一轮的错(自修复) |

**检查点恢复**是 Ralph 与生产级 agent 的交汇点:因为每一小步都 commit,任何崩溃最多损失"当前这一轮"的工作,回退到最近 commit 即可续跑,而非从零重来。这与 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]] 里 durable execution 的"崩溃后从最近持久化点重放恢复"是同一个原理——Ralph 用 git 当了一个穷人版的 checkpointer。

## 来源 / 出处

- **Geoff Huntley, _"Ralph Wiggum as a software engineer"_(ghuntley.com/ralph,2025 年底)**:命名与方法论原帖;配套 repo **`ghuntley/how-to-ralph-wiggum`**。核心主张:用纪律化的循环把软件成本压到"不及一个快餐工时薪"。
- **HumanLayer, _"A Brief History of Ralph"_**:梳理了该技法的演化与社区实践,提出"3 阶段 / 2 prompt / 1 循环"的漏斗式总结。
- 流传甚广的轶事:有工程师用 Ralph 技法以约 **$297 的 API 成本交付了一个原标价 $50,000 的合同**(MVP 已交付 + 测试 + 评审)——数字是宣传性质,但点出了该技法的成本叙事。
- 思想上它综合了 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](试错自改进)、[[03 Agent 核心循环|Agent 核心循环]](感知-决策-行动外循环)、以及"文件系统当上下文"这一更早就在 coding agent 圈流行的工程惯例。

需要分清:Ralph **不是一个库**,而是一种**用 5 行 bash 就能复刻**的方法论;它能跑在任何"接受一句 prompt、能改文件、能跑测试"的 coding agent 之上。

## 可跑的最小实现

最朴素的 Ralph 就是一段 shell:

```bash
# ralph.sh —— 最小 Ralph 循环
while true; do
  # 每轮全新进程,读固定 prompt;agent 会读 TODO.md + 代码,推进一步并自行 commit
  claude -p "$(cat PROMPT.md)" --dangerously-skip-permissions
  # 收敛判据:TODO 清空 且 测试全绿 → 跳出
  if [ ! -s TODO.md ] && pytest -q; then
    echo "done"; break
  fi
  sleep 2   # 喘口气,避免打满限流
done
```

把它写成更结构化的 Python 伪码,能看清"读 TODO → 做一步 → 提交 → 重启"的骨架,以及反压与收敛守卫:

```python
import subprocess, pathlib

GOAL = pathlib.Path("PROMPT.md").read_text()     # 固定不变的目标 prompt
MAX_ROUNDS = 200                                  # 防失控的硬上限

def tests_pass() -> bool:
    return subprocess.run(["pytest", "-q"]).returncode == 0

def todo_empty() -> bool:
    p = pathlib.Path("TODO.md")
    return (not p.exists()) or (p.read_text().strip() == "")

def run_one_round(agent):
    """单轮:全新上下文。agent 自己读 TODO.md + 代码,做一步,git commit。"""
    state = {
        "goal":  GOAL,                            # 目标锚:每轮重读,防漂移
        "todo":  pathlib.Path("TODO.md").read_text() if pathlib.Path("TODO.md").exists() else "",
        "last_test": "" if tests_pass() else subprocess.run(
            ["pytest", "-q"], capture_output=True, text=True).stdout,  # 反压信号回灌
    }
    agent(state)            # agent 内部:plan 一件事 → act → 测试过才 commit + 更新 TODO

def ralph(agent):
    for rnd in range(MAX_ROUNDS):
        run_one_round(agent)                      # ← 进程级"重启",上下文清零
        if todo_empty() and tests_pass():         # 收敛:活干完且测试全绿
            return f"完成,共 {rnd+1} 轮"
    return "达到轮数上限,人工介入"                 # 失控兜底
```

四个要点:① **`GOAL` 每轮重读**,这是防目标漂移的锚;② **`last_test` 把失败回灌**给下一轮,实现跨轮自修复(等价于 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的反馈回路,只是介质是磁盘);③ **commit 即检查点**,崩溃可续;④ **`MAX_ROUNDS` + 收敛判据**是必须的护栏,否则 Ralph 会在无法满足的目标上**无限烧钱**——这是它最危险的失败模式。

## 对比:Ralph loop vs 一次性 Plan-and-Execute

| 维度 | [[10 Plan-and-Execute|Plan-and-Execute]] | **Ralph loop** |
|---|---|---|
| 规划时机 | **前期一次**排好全程线性计划 | **不预排**,每轮当下挑一件最该做的事 |
| 会话模型 | 一个会话内逐步执行(必要时 replan) | **每轮全新进程、上下文清零** |
| 跨步记忆 | 对话历史 / 计划队列(在上下文里) | **文件系统**(代码 + TODO + git),不在上下文 |
| 上下文增长 | 随步数增长,长程会撞窗 | **每轮恒定**,天然不撞窗 |
| 崩溃恢复 | 通常丢状态(除非接 checkpointer) | git commit 即检查点,**天然可续** |
| 自修复 | 靠 replan 调整剩余计划 | 靠**测试反压 + 下轮重读现实** |
| 适合 | 步骤可预排、确定性高的流程 | **不确定、规格会变、超长**的开放任务 |
| 主要风险 | 初始计划在高不确定任务上不准 | **不收敛时无限烧钱**,需硬护栏 |

可以这样记:Plan-and-Execute 把智力**集中在前期一次规划**,Ralph 把智力**摊到无数次小重启**;前者赌"计划想得准",后者赌"重复 + 反压终会收敛"。两者并不互斥——成熟实践常在 Ralph 的单轮内部用一个小 [[10 Plan-and-Execute|Plan-and-Execute]] 或 [[09 ReAct|ReAct]] 推进那"一件事"。

## 何时用 / 坑

**该用**:目标可被"测试 / 验收标准"客观判定的**超长开放任务**——从规格生成整个项目、批量修一组 issue、把一份大重构推到测试全绿。只要"完成"有客观信号(测试、lint、类型检查),Ralph 的反压就有抓手。

**不该用**:**没有客观反压**的任务(纯创意写作、无法自动验证对错的研究结论)——没有测试当反压,Ralph 会乐呵呵地把错误成果一轮轮 commit,发散且不自知。也不适合**对每一步都要人审**的高风险操作(改生产数据库、发交易),那类任务该走 Human-in-the-Loop 而非无人值守循环。

**常见坑**:
- **不收敛 / 无限烧钱**:最危险。目标若实际不可达(测试本身有 bug、规格自相矛盾),Ralph 会永远跑下去。**必须**有 `MAX_ROUNDS`、预算上限、以及"连续 N 轮无进展就停"的探测。
- **没有反压 = 自我欺骗**:不跑测试或测试覆盖太弱,agent 会把"看起来对"的烂代码提交进去,后续轮在烂地基上盖楼。**测试质量直接决定 Ralph 上限**。
- **一轮咬太多**:让单轮做太多事会把上下文撑爆、半途失忆。纪律是**每轮只推进一件最该做的事**。
- **规划与施工混在一个 prompt**:会让 agent 一边想一边改、思路打架。分成 PLANNING / BUILDING 两个 prompt 是经验性的关键纪律。
- **TODO.md / git 状态被写坏**:跨轮记忆全靠它们,一旦 agent 把 TODO 清空错了或提交了坏状态,后续全乱。给 commit 加 hook 校验、TODO 加格式约束。
- **限流与并发**:死循环高频调起容易打满 API 限流;加 `sleep`、退避重试。

## 关键事实

- **Ralph = while 循环 + 固定 prompt + 文件系统当记忆 + 测试当反压**;不是库,是一种 5 行 bash 可复刻的方法论(Geoff Huntley,2025 年底命名,取自辛普森角色 Ralph Wiggum)。
- 突破上下文窗口的原理:**每轮上下文清零**,跨轮状态全部**卸载到磁盘**(代码 / TODO / git),单轮只装"当前一小步"——与 [[21 上下文压缩与卸载|上下文压缩与卸载]]、[[19 Agent 记忆系统|Agent 记忆系统]] 同源。
- 工程纪律:**1 循环 / 2 prompt(PLANNING 只规划、BUILDING 只施工)/ 3 阶段(plan→act→reflect)**。
- **测试是反压**:过了才 commit,不过则下轮先修;这同时实现了自我修复([[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的跨进程版)与防错误累积。
- **git commit 即检查点**,崩溃从最近 commit 续——是 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]] 中 durable execution"崩溃重放恢复"的穷人版。
- 与 [[10 Plan-and-Execute|Plan-and-Execute]] 正相反:**不预排全程**,计划逐轮涌现;赌的是"重复 + 反压收敛"而非"一次想对"。
- 头号风险:**不收敛时无限烧钱**——`MAX_ROUNDS` + 预算上限 + 无进展探测是必备护栏。

## 主流开源实现 / Python 库

- **`ghuntley/how-to-ralph-wiggum`**(GitHub):Geoff Huntley 官方的 Ralph 技法说明与脚本集,方法论原始出处。
- **`Aider-AI/aider`**(GitHub,pip `aider-chat`):终端 AI 结对编程,**自动 git commit**、architect/editor 双模型;`--yes-always` / 脚本模式让它极易被外层 while 循环包成 Ralph,是最常见的 Ralph 载体之一。2026 年仍高频发版、活跃维护。
- **`OpenHands/OpenHands`**(GitHub,前身 OpenDevin):开源自治开发 agent,能浏览/写码/执行,常被用于**长跑无人值守**的端到端任务,天然适配 Ralph 式外循环。
- **`gptme/gptme`**(GitHub,pip `gptme`):终端里的持久自治 agent,配 `gptme-agent-template` 提供 **git-backed 记忆 + 自治 run-loop**,定位就是"长期自我修改、跑很多轮的 agent"。
- **`langchain-ai/langgraph`**(GitHub,pip `langgraph`):用其 **checkpointer + 持久状态**可把 Ralph 式循环做成可中断/可恢复的有状态图,适合需要生产级 durable 持久循环时(详见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]])。
- 社区里还有大量 **`ralph-wiggum` 类极简脚本**(各种 `ralph.sh` / `ralph-loop` gist 与小仓库),本质都是上文那段 `while true; do agent; done` 的变体——Ralph 的精神就是简单到不需要框架。

## 工业界实践

**谁在用、用在哪。** Ralph 不是一个被某家公司"采购部署"的产品,而是 2025–2026 在**独立开发者、AI 咨询工作室、初创小团队**里口口相传的**无人值守批量交付**打法。典型场景:把一份 spec(PRD / 一组 GitHub issue / 一份 OpenAPI 定义)丢进循环,让 Claude Code / aider / OpenHands 自己跑一夜,早上来看测试是否全绿、diff 是否可审。它和"公司级 agent 平台"是两个世界——后者讲合规、审计、SLA,Ralph 讲**极致便宜地把一个有客观验收标准的脏活磨完**。

**典型生产架构(三层)。** 成熟团队不会真用裸 `while true`,而是把它包成更可控的形态:

1. **驱动层(harness)**:不是 bash 死循环,而是一个带**预算账本 + 无进展探测 + 结构化日志**的 Python/TS 进程。它负责调起 coding agent、记账(累计 token / 美元)、判收敛、写运行报告。
2. **执行层(coding agent)**:Claude Code(`-p` headless + `--dangerously-skip-permissions`)、aider(`--yes-always` 脚本模式)、OpenHands(headless 任务模式)三选一,跑在**一次性容器/沙箱**里(Docker / Firecracker / E2B / Daytona),保证每轮"干净进程"且**爆炸半径受限**——agent 跑飞了顶多毁掉一个容器,毁不了主机。
3. **状态层(git + 工件)**:代码、`TODO.md`、测试报告全落 git;每轮一个 commit,既是检查点也是审计轨迹。CI 在每个 commit 上跑一遍,给外层 harness 一个客观的"测试是否真绿"信号(防 agent 自报喜)。

**规模化与成本。** Ralph 的经济性来自两点:① **单轮上下文恒定**(每轮清零),所以 token 不随任务长度爆炸,成本可线性预估;② **无人值守**,人力从"逐步写码"降到"早上审一次 diff"。但代价是**总 token 可能很高**——它靠"多跑很多轮"换"每轮简单",轮数一多累计开销不小;再叠加**重复读 `TODO.md` + 代码片段**作前缀,**prompt caching 几乎必开**(联动 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]] 的前缀复用,把稳定的目标 prompt / 工具表缓存住,命中省 ~90% 输入)。延迟通常不敏感(本就跑一夜),所以优化重心是**省钱 + 防失控**,不是降延迟。

**可观测与运维。** 无人值守的最大风险是"没人看着它烧钱/跑飞",所以运维三件套是刚需:
- **预算熔断**:累计美元/token 到上限即停,硬性兜底。
- **无进展探测**:连续 N 轮 `git diff` 为空、或测试通过数不增、或同一文件反复改回去 → 判定卡死,停并告警。
- **结构化运行日志 + 每轮快照**:每轮的 plan/act/测试结果落盘,出问题能回溯是哪轮开始跑偏。

**踩坑与最佳实践(工业界沉淀)。**
- **测试覆盖是 Ralph 的天花板**:反压只和测试一样强。生产实践会**先要求 agent 写测试、人审测试质量,再开循环**——否则 agent 会把"编译过但逻辑错"的代码一轮轮提交。
- **分支隔离**:Ralph 只在**专用分支/worktree**上跑,绝不直接动 `main`;人审后再合。
- **PLANNING/BUILDING 双 prompt 物理分离**:两个独立文件、独立调用,别让一个 prompt 既想又干。
- **给 agent 限定爆炸半径**:沙箱 + 只读挂载敏感目录 + 网络白名单,防止 `--dangerously-skip-permissions` 下 agent 删库或外联。
- **收敛判据要"双信号"**:`TODO.md` 空 **且** CI 真绿,缺一不可——只信 TODO 会被 agent 自己清空骗过去。

```python
# 工业级 Ralph harness 骨架:预算 + 无进展探测 + 沙箱 + 双信号收敛
def ralph_prod(agent, budget_usd=50.0, no_progress_limit=5):
    spent, stale = 0.0, 0
    last_green = -1
    for rnd in range(MAX_ROUNDS):
        before = git_head()
        cost = run_in_sandbox(agent)         # 一次性容器内跑一轮,返回本轮花费
        spent += cost
        if spent > budget_usd:               # ① 预算熔断
            return alert("超预算,停")
        if git_head() == before:             # ② 无进展:本轮没产生 commit
            stale += 1
        else:
            stale = 0
        if stale >= no_progress_limit:       # 连续 N 轮无进展 → 卡死告警
            return alert("疑似卡死,人工介入")
        if todo_empty() and ci_all_green():  # ③ 双信号收敛(TODO 空 且 CI 真绿)
            return f"完成,{rnd+1} 轮,${spent:.2f}"
    return alert("达轮数上限")
```

## 面试高频

**Q1:Ralph loop 是什么?为什么它能突破上下文窗口?**
标准答:Ralph 是"反复重启同一个目标 prompt 的 while 循环"——每轮全新进程、上下文清零,从磁盘读 `TODO.md` + 代码续上一轮,只推进一小步,测试过才 commit,然后退出重启。它突破上下文窗口的原理是**把跨轮状态全卸载到文件系统**(代码/TODO/git),单轮上下文永远只装"当前这一小步",故任意长的任务单轮 token 都恒定,天然不撞窗。
- **追问:那它的"记忆"在哪?** 不在对话历史里,在磁盘上(代码、`TODO.md`、git 历史、上轮失败的测试输出)。这与 [[19 Agent 记忆系统|Agent 记忆系统]] 的外部记忆、[[21 上下文压缩与卸载|上下文压缩与卸载]] 的卸载思想同源。

**Q2:Ralph 里"反压(backpressure)"指什么?去掉会怎样?**
标准答:反压就是**测试**——只有测试过了才 `git commit`,没过就不提交、把失败信息留给下一轮先修。去掉反压(不跑测试/测试太弱),agent 会"自我感觉良好地"把编译过但逻辑错的代码一轮轮提交,后续轮在烂地基上盖楼,**发散且不自知**。所以**测试质量直接决定 Ralph 的上限**。
- **陷阱**:有人答"反压是限流/sleep"——错。sleep 只是防打满 API,反压特指**测试这个客观验收信号**。

**Q3:Ralph 和 Plan-and-Execute 有什么本质区别?**
标准答:Plan-and-Execute **前期一次**排好全程线性计划再逐步执行,记忆在上下文里,长程会撞窗;Ralph **不预排全程**,每轮当下挑一件最该做的事,记忆在文件系统、上下文每轮恒定。前者赌"计划想得准",后者赌"重复+反压终会收敛"。两者不互斥:成熟实践常在 Ralph 单轮内部用一个小 [[10 Plan-and-Execute|Plan-and-Execute]] 或 [[09 ReAct|ReAct]] 推进那"一件事"。

**Q4:Ralph 最危险的失败模式是什么?怎么防?**
标准答:**不收敛 → 无限烧钱**。若目标实际不可达(测试本身有 bug、规格自相矛盾),Ralph 会永远跑下去。必备护栏:`MAX_ROUNDS` 硬上限 + 预算/token 上限 + **无进展探测**(连续 N 轮没新 commit / 测试通过数不增就停)。
- **追问:为什么不能只看 `TODO.md` 是否清空来判完成?** 因为 agent 可能错误地清空 TODO 自报完成。收敛判据必须**双信号**:TODO 空 **且** 测试(CI)真绿。

**Q5:什么任务适合 Ralph,什么不适合?**
标准答:适合**有客观反压(测试/lint/类型检查)的超长开放任务**——从 spec 生成整个项目、批量修一组 issue、把大重构推到测试全绿。不适合**没有客观对错信号**的任务(纯创意写作、无法自动验证的研究结论)——没反压会发散;也不适合**每步都要人审的高风险操作**(改生产库、发交易),那类该走 Human-in-the-Loop。

**Q6(进阶):Ralph 的"自我改进/自我修复"体现在哪?**
标准答:每一轮都对"上一轮留下的现实(代码 + 失败测试)"做反思与修正,等价于把 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的"试错→反馈→改进"做成**跨进程、跨轮次的外循环**——反馈不靠 verbal memory 塞回上下文,而是直接写进代码与测试结果,下一轮失忆醒来自然读到。

## 知识拓展

**与 durable execution 的谱系关系。** Ralph 用 `git commit` 当检查点,是 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]] 中 durable execution"崩溃后从最近持久化点重放恢复"的**穷人版**:都靠"持久化进度 + 可恢复"扛长程,区别在 Ralph 的持久化粒度粗(整个 commit)、恢复靠重读磁盘而非事件重放,且无 exactly-once 副作用保证。需要生产级可靠性(精确恢复、幂等、人审中断)时,就该从 Ralph 升级到 LangGraph checkpointer / Temporal 这类正规 durable runtime。

**"文件系统当上下文"是更大的范式。** Ralph 把它推到极致,但这一思想更早就在 coding agent 圈流行:让 agent 把工作状态写进文件(`TODO.md`、scratchpad、笔记),而非全塞进上下文窗口。它本质是 [[21 上下文压缩与卸载|上下文压缩与卸载]] 的卸载策略 + [[19 Agent 记忆系统|Agent 记忆系统]] 的外部记忆,在"无限长任务"这个极端场景下的体现。Anthropic 2025 年也在工程博客里讨论过"用文件系统/磁盘当 agent 的外部记忆"以突破上下文限制的做法。

**相关研究与前沿(带年份)。**
- **Reflexion(Shinn et al., 2023)**:语言化自我反思——把失败的口头反馈存进记忆指导下一次尝试。Ralph 是它的"工程糙化版":反馈介质从 verbal memory 换成磁盘上的代码与测试结果,反思边界从单会话扩到跨进程。
- **SWE-bench(2023)/ SWE-bench Verified(2024)**:真实 GitHub issue 修复基准,正是 Ralph 这类无人值守 coding agent 的主战场与验收标尺(联动 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]])。
- **STaR / 自我改进训练(Zelikman et al., 2022)与 2024–2025 的 agentic self-improvement 研究**:从"模型层自我改进"角度,与 Ralph 的"系统层自我改进"互为镜像——前者改权重,后者改磁盘上的工件。

**边界与反模式。**
- **把 Ralph 当银弹**:它只在"有客观反压 + 任务可分解为小步"时 work;强上下文耦合、需全局一次性想清楚的任务(如复杂算法的整体架构设计)反而是它的弱项。
- **无沙箱裸跑 `--dangerously-skip-permissions`**:在主机上无隔离地跑死循环 agent,等于把删库/外联/改生产的权限交给一个会失忆的循环——必须沙箱化。
- **用 Ralph 替代代码评审**:Ralph 产出的是"测试绿的 diff",不等于"对的代码"。测试覆盖外的逻辑、安全、架构问题仍需人审,Ralph 是**降低人审频率**而非取消人审。
- **反模式:让单轮咬太多**。违背"每轮只推进一件最该做的事"的纪律,会把上下文撑爆、半途失忆,Ralph 的鲁棒性来自"小步 + 多轮",不是"大步 + 少轮"。

延伸阅读联动:[[03 Agent 核心循环|Agent 核心循环]](Ralph 是其长程极简变体)、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](自我修复思想源头)、[[34 Agent 部署与持久化执行|Agent 部署与持久化执行]](工业级正解)、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]](Ralph 的成本治理)。
