[[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的本质是:agent 是**非确定、多步、带轨迹、有副作用**的系统,所以「最终答案对不对」远远不够——你必须能**追整条 trace**,把一次运行拆成一步步的 span,才谈得上知道它好不好、坏在哪、为什么坏。评估是「打分」,可观测性是「看见」,两者是同一件事的两面。

## 本质:不能只看最终答案
传统模型评测是「给输入、看输出、对标准答案」。但 agent 不是一次函数调用,它是一条**轨迹(trajectory)**:规划 → 调工具 → 看结果 → 再决策 → ……。这带来四个传统评测处理不了的难点:

- **随机性**:同一输入,两次跑出不同轨迹(采样温度、工具返回时序)。单次结果说明不了什么,得看分布。
- **长程**:几十步里错一步,后面全崩;但最终答案可能**碰巧**还对——掩盖了过程的脆弱。
- **工具副作用**:agent 真的会写文件、发请求、改数据库。评估不只看「答得对吗」,还要看「它干了什么、有没有干坏事」。
- **错误归因难**:答案错了,是规划错?选错工具?工具参数错?工具本身挂了?还是综合那步塌了?不追轨迹根本说不清。

所以核心命题是:**把一次 run 变成可追溯的结构化数据(trace/span),再在这数据上同时做「监控」和「评估」。** 这呼应 [[03 Agent 核心循环|Agent 核心循环]]——你要观测的,正是循环的每一圈。

## 可观测性:把一次 run 拆成 trace 和 span
![[Agent 评估与可观测性.svg]]

三个核心概念(直接借自分布式追踪 / OpenTelemetry):

- **Trace(轨迹)**:一次完整运行,有唯一 `trace_id` 贯穿始终。对应「用户问一句 → agent 折腾一通 → 给出答案」的整段。
- **Span(跨段)**:trace 里的**一段操作**,有父子嵌套和计时。典型分三类:
  - **LLM span**:一次模型调用——记 prompt、输出、token 数、延迟、成本、模型名。
  - **tool span**:一次工具调用——记工具名、入参、出参、成功/报错(对应 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的每次调用)。
  - **retriever span**:一次检索——记 query、命中文档、召回质量(对应 [[36 Agentic RAG|Agentic RAG]] / [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]])。
- **嵌套**:一个 LLM span 决定调工具,下面挂一个 tool span;tool 内部又检索,再挂 retriever span。整棵树就是这次 run 的「解剖图」。

有了这棵树,**错误归因**才成立:答案错 → 翻 trace → 定位到某个 tool span 报错并重试 3 次 → 根因找到。没有 trace,你只能瞎猜。

## 评估维度:从「最终对不对」下钻到「每一步对不对」
![[Agent 评估与可观测性-评估维度分层.svg]]

分四层,**只测第一层是最常见也最危险的偷懒**:

- **① 结果层**:任务完成率、答案正确性、用户满意度。最易测,但会掩盖过程问题——答案碰巧对了不代表 agent 健康。
- **② 轨迹层**:**轨迹质量**——它走得绕不绕、有没有重复 / 死循环、计划合不合理、是否贴近「黄金轨迹」。
- **③ 步骤层 + 工具层**:
  - **step 成功率**:每一步决策是否正确(该规划时规划了吗、该停时停了吗)。
  - **tool 调用成功率**:这是 agent 评估里**最高信噪比的单一指标**——拆成两问:**选对工具了吗**(tool selection)、**参数填对了吗**(argument correctness)。很多 agent 失败本质是这一层崩了。
- **④ 资源层**(贯穿每一步):**成本($)、延迟(秒)、token、步数**。质量再高,贵到 / 慢到不可用也是失败。生产里这层常和质量层同等重要。

一句话:**结果层告诉你「坏了没」,轨迹/步骤/工具层告诉你「哪坏了」,资源层告诉你「值不值」。**

## 评估方法:三件套
### 1. LLM-as-judge(用模型当裁判)
让另一个(通常更强的)LLM 按 rubric 给输出 / 轨迹打分。优点:能评「有用性、连贯性、是否答非所问」这类**无标准答案**的软指标,可规模化。坑:

- 裁判有偏好(位置偏好、长度偏好、自我偏好——爱给自己家模型高分),要做去偏(打乱顺序、成对比较)。
- rubric 要具体、可执行,别让裁判「凭感觉打 1-10」;给清晰标准 + few-shot 例子。
- 裁判本身要被校准:拿人工标注的小集合验证「裁判和人一致吗」。

### 2. 黄金轨迹对比(trajectory matching)
准备「专家认可的正确轨迹」,比对 agent 实际轨迹:用了对的工具吗、顺序合理吗、有没有冗余步骤。分**严格匹配**(每步都得一样,太脆)和**软匹配**(关键工具调用对了即可,更实用)。适合工具调用类、流程相对固定的 agent。

### 3. 回归集(regression suite / eval set)
攒一批「输入 + 期望」的固定测试集,每次改 prompt / 换模型 / 调工具就全跑一遍,看指标有没有回退。这是把 agent 开发从「玄学调 prompt」变成「工程」的关键——**没有回归集,你的每次改动都是在赌博**。和软件单测一个道理,只是断言更模糊(常配 LLM-as-judge)。这与 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 同源:那是运行时的生成-评估闭环,回归集是开发期的离线版。

## 可跑最小代码
### A. 给 agent 打点(手动 trace/span,看清结构)
```python
import time, uuid, contextlib

class Tracer:
    def __init__(self): self.spans = []
    @contextlib.contextmanager
    def span(self, name, kind, **attrs):
        s = {"id": uuid.uuid4().hex[:8], "name": name, "kind": kind,
             "attrs": attrs, "start": time.time()}
        try:
            yield s
            s["status"] = "ok"
        except Exception as e:
            s["status"] = "error"; s["error"] = str(e); raise
        finally:
            s["latency"] = round(time.time() - s["start"], 3)
            self.spans.append(s)

tr = Tracer()
def agent_run(q):
    with tr.span("agent", "trace", query=q):
        with tr.span("plan", "llm", model="gpt-4o") as s:      # LLM span
            plan = llm(f"为问题制定计划:{q}"); s["attrs"]["tokens"] = 320
        with tr.span("search", "tool", tool="web_search") as s:# tool span
            s["attrs"]["args"] = {"q": q}
            hits = web_search(q)                               # 真有副作用
        with tr.span("answer", "llm", model="gpt-4o"):
            return llm(f"据 {hits} 回答 {q}")

agent_run("2025 年最流行的 agent 框架?")
# tr.spans 现在是一棵可分析的结构化轨迹:每段都有 kind/status/latency/tokens
# 生产里把这套换成 OpenTelemetry / LangSmith / Langfuse 的自动埋点即可
```
真实项目里你不手写 Tracer——上 OpenTelemetry 的 **GenAI 语义约定**(社区定的标准 span 属性名,如 `gen_ai.usage.input_tokens`、`gen_ai.request.model`),让任何后端都能读你的 trace。

### B. 一个简单的离线 eval(LLM-as-judge + tool 成功率,伪码)
```python
JUDGE_RUBRIC = """你是严格评审。对 agent 输出按以下维度各打 1-5 分,输出 JSON:
- correctness: 是否事实正确且答到点上
- trajectory: 过程是否高效无绕路
返回 {"correctness": x, "trajectory": y, "reason": "..."}"""

def evaluate(eval_set):
    rows = []
    for case in eval_set:                       # case = {"input":..., "gold_tools":[...]}
        spans = []; out = agent_run(case["input"])          # 跑一次,顺便收集 spans
        # 1) 工具调用成功率(选对工具 + 没报错)
        tool_spans = [s for s in spans if s["kind"] == "tool"]
        used = [s["attrs"]["tool"] for s in tool_spans]
        tool_ok = (set(case["gold_tools"]) <= set(used)) and \
                  all(s["status"] == "ok" for s in tool_spans)
        # 2) LLM-as-judge 评结果与轨迹
        verdict = json.loads(llm(JUDGE_RUBRIC + f"\n输入:{case['input']}\n输出:{out}"))
        # 3) 资源层
        cost = sum(s["attrs"].get("tokens", 0) for s in spans) * 1e-6
        rows.append({**verdict, "tool_ok": tool_ok, "cost": cost,
                     "latency": sum(s.get("latency", 0) for s in spans)})
    return aggregate(rows)   # 出一张:完成率/tool成功率/平均分/p95延迟/成本 的回归报告
```
这段把四层维度全打通了:judge 管结果+轨迹层、`tool_ok` 管工具层、`cost/latency` 管资源层。每次改动跑一遍,就是回归集。

## 可观测性工具对比
| 工具 | 定位 | 特点 | 何时选 |
|---|---|---|---|
| **LangSmith** | LangChain 官方平台 | 与 [[37 Agent 框架对比|LangGraph]] 深度集成,trace/数据集/eval/标注一体,托管 | 用 LangChain 全家桶;要最省心的端到端 |
| **Langfuse** | 开源 LLM 可观测平台 | 可自托管,框架无关,trace + eval + prompt 管理 + 成本看板 | 要开源/自托管、数据不出内网 |
| **OpenTelemetry(GenAI 语义约定)** | 厂商中立的追踪标准 | 不是平台是**协议**,定义标准 span 属性;接任何后端 | 想避免厂商锁定;已有 OTel 基建 |
| **Phoenix / Arize** | 开源 + 商业,偏评估/监控 | 内建 LLM 评估器、漂移监控,基于 OTel | 重离线评估与生产监控 |

选型要点:**别把可观测性绑死在某个框架**。OTel GenAI 语义约定的价值就是「埋点一次,后端随便换」。这件事在 [[37 Agent 框架对比|Agent 框架对比]] 选型时就要算进去——LangGraph+LangSmith 的丝滑 vs 框架无关的灵活,是一组真实权衡。

## 在线监控 vs 离线评估(别混)
- **离线评估(开发期)**:跑回归集,门禁式——指标回退就别合并。慢工出细活,可用昂贵的 LLM-as-judge。
- **在线监控(生产期)**:对真实流量采样打 trace,看实时完成率 / 错误率 / p95 延迟 / 成本曲线,异常**告警**。要轻量、低开销。
- **人审回路(human-in-the-loop)**:把模型不确定 / 低分 / 高风险的轨迹捞出来给人看,人工标注**反哺回归集和 judge 校准**——这是评估体系自我进化的飞轮。

## 何时用 / 坑
- **何时上**:只要 agent 要进生产,可观测性是**第一天就该有的基建**,不是出事后补的。哪怕原型期,手动打点也比裸跑强。
- **坑一:只测最终答案**。最常见的致命偷懒——答案对了不代表过程健康,见上面四层图。
- **坑二:没有回归集就改 prompt**。每次「我觉得这版更好」都是赌博;沉淀回归集才有可比性。
- **坑三:盲信 LLM-as-judge**。裁判有偏好、会漂移,必须拿人工小集校准,否则你在用一把没校准的尺子量。
- **坑四:忽略成本/延迟**。质量调上去了,token 翻 5 倍、延迟到分钟级,生产不可用——资源层和质量层同等重要。
- **坑五:可观测性事后补**。埋点要趁早;线上烧了一周才想追轨迹,数据早没了。
- **坑五连环**:[[22 多智能体系统|多智能体系统]] 的可观测性更难——多个 agent 的 trace 要能关联(谁派给谁、哪条子轨迹属于哪个 [[26 Sub-agents 与 Agent Teams|sub-agent]]),否则归因彻底失效。

## 关键事实
- **agent 评估的第一性原理**:它是轨迹不是答案,所以必须追 trace、分层评估、看分布而非单次。
- **tool 调用成功率**(选对工具 + 参数对)是单一最高信噪比指标;很多 agent 故障的根因都在这一层。
- **trace = 看见,eval = 打分**,二者共用同一份结构化 span 数据;先有可观测,才谈得上可评估。
- **OpenTelemetry GenAI 语义约定**是避免厂商锁定的关键:埋点遵循标准,后端(LangSmith / Langfuse / Phoenix)随便换。
- 评估体系的闭环:离线回归集(开发)→ 在线监控告警(生产)→ 人审捞坏例(反哺)→ 校准 judge & 扩充回归集,周而复始——和 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 是同一种「生成-评估-改进」精神,只是搬到了系统工程层面。
- 一个朴素结论:**做不出好的评估,就做不出好的 agent**。能稳定衡量,才能稳定改进;不能衡量的,只能靠运气。这也是 [[23 Agent Harness 概览|Agent Harness 概览]] 类成熟系统把评估/追踪当核心模块的原因。

## 主流开源实现 / Python 库

**可观测(trace)**
- **`langfuse/langfuse`**(pip `langfuse`,MIT,可自托管):框架无关的开源平台,trace + eval + prompt 管理 + 成本看板——**要开源/数据不出内网的首选**。
- **`Arize-ai/phoenix`**(pip `arize-phoenix`):OpenTelemetry-native(OpenInference 约定),内建评估器与漂移监控,重离线评估时强。
- **`traceloop/openllmetry`**(pip `traceloop-sdk`):基于 OTel 的自动埋点,几分钟接好、转发到任意后端——只想低成本埋点接现有可观测栈时选它。
- **LangSmith**(托管,非开源):LangChain 全家桶最丝滑的端到端方案。

**评估(eval)**
- **`confident-ai/deepeval`**(pip `deepeval`):pytest 式 LLM 评估框架,内建大量指标,适合做回归集断言。
- **`explodinggradients/ragas`**(pip `ragas`):专攻 RAG 评估(faithfulness/召回相关性等),与 [[36 Agentic RAG|Agentic RAG]] 配套;常被 Phoenix 等原生集成。
- **`AgentOps-AI/agentops`**(pip `agentops`):Python SDK 优先的 agent 调试/可观测,支持 CrewAI、AutoGen、OpenAI Agents 等多框架。
- 当下首选:开源全栈可观测 `langfuse`;RAG 评估 `ragas`;通用 LLM 断言 `deepeval`;多框架 agent 调试 `agentops`;埋点中立用 `openllmetry`/OTel。

## 工业界实践

**平台版图(2026,按定位分三类):**
- **托管端到端**:**LangSmith**(LangChain 官方,与 LangGraph 深度集成,trace/dataset/eval/标注/prompt 一体,最省心但**框架锁定**、自托管仅 Enterprise);**Braintrust**(**eval-first** 的代表,主打**结构化实验 + prompt 版本对比 + CI 卡门禁**,内建 proxy 做快速迭代,免费档很大方:1M spans/月、10K evals)。
- **开源/可自托管**:**Langfuse**(MIT,自托管标杆,trace + eval + prompt 管理 + 成本看板;**2026 年 1 月被 ClickHouse 收购**,能力延续,数据不出内网首选);**Phoenix / Arize**(OpenTelemetry-native,内建 LLM 评估器与漂移监控,ML 级严谨)。
- **协议层**:**OpenTelemetry GenAI 语义约定**——不是平台是**标准**,定义 `gen_ai.usage.input_tokens`、`gen_ai.request.model` 等标准 span 属性;埋点一次、后端随便换,是避免厂商锁定的关键。

**选型一句话:** LangChain 全家桶且能付费托管 → **LangSmith**;要系统化预上线实验 + CI 门禁 → **Braintrust**;自托管/数据不出门 → **Langfuse**;ML 级严谨监控 → **Phoenix**;RAG 专项打分 → **`ragas`**;评估写成单测卡 CI → **`deepeval`**(LLM 版 pytest)。

**典型评估闭环(生产工程化):**
```
开发期(离线,门禁式)
  回归集(input+期望)──► 跑 agent 收集 trace ──► 四层打分
     │                                          ├ 结果层:LLM-as-judge(correctness)
     │                                          ├ 轨迹层:judge / 黄金轨迹软匹配
     │                                          ├ 工具层:tool selection + arg correctness
     │                                          └ 资源层:cost / latency / tokens / 步数
     └─► 指标回退就 block merge(Braintrust/deepeval 卡 CI)
生产期(在线,轻量)
  真实流量采样打 trace ──► 实时完成率/错误率/p95/成本曲线 ──► 异常告警
     └─► human-in-the-loop:捞低分/高风险轨迹给人标 ──► 反哺回归集 & 校准 judge(飞轮)
```

**一个被反复验证的硬数据**(支撑本篇「只测最终答案最危险」):**只按最终输出质量评的 agent,会比轨迹级评估多通过 20~40% 的用例**——即过程其实塌了(选错工具、参数错、状态漂移),但答案碰巧对了,被掩盖。所以生产必须下钻到 **step / tool 层**,**tool 调用成功率(选对工具 + 参数对)是单一最高信噪比指标**。

**规模化与成本。** LLM-as-judge 本身要花钱花延迟:① 离线慢工出细活可以用强模型当裁判,**在线监控**只能用便宜小模型或规则做轻量打分;② judge 调用要采样(不是每条线上流量都评);③ trace 数据量大,Langfuse/ClickHouse 类列存 + 采样保留策略控成本。**多智能体的 trace 关联**是规模化难点——多个 agent 的 span 要能拼回一棵树(谁派给谁、哪条子轨迹属于哪个 [[26 Sub-agents 与 Agent Teams|sub-agent]]),否则归因彻底失效。

**踩坑与最佳实践:**
- **judge 必须校准**:裁判有位置偏好、长度偏好、**自我偏好**(爱给自家模型高分);上线前拿人工标注的小集合验「judge 和人一致吗」,定期复校。
- **rubric 要可执行**:别让裁判「凭感觉打 1-10」,给明确标准 + few-shot 例子 + 要求输出 JSON 带 reason。
- **埋点趁早、遵循 OTel**:线上烧一周才想追轨迹,数据早没了;埋点遵循 GenAI 语义约定,换后端不用重埋。
- **回归集是命根子**:没回归集就改 prompt = 赌博;每次改动跑一遍才有可比性。

## 面试高频

**Q1:为什么 agent 评估不能只看最终答案?**
因为 agent 是**轨迹(trajectory)不是一次函数调用**:它有随机性(同输入不同轨迹)、长程(错一步后面全崩但答案可能碰巧对)、工具副作用(真写文件/改库)、错误归因难。**只按最终输出评会比轨迹级评估多通过 20~40% 用例**——掩盖了过程的塌方。必须追 trace、分层评估、看分布而非单次。
- *追问:那要评哪几层?* 四层:**结果层**(坏了没)、**轨迹层 + 步骤/工具层**(哪坏了)、**资源层**(值不值)。
- *陷阱:「轨迹评估就是每步都要和黄金轨迹一模一样?」* 不。严格匹配太脆,生产用**软匹配**(关键工具调用对了即可)。

**Q2:trace 和 eval 是什么关系?**
**trace = 看见,eval = 打分**,二者共用同一份结构化 span 数据。先有可观测(把一次 run 拆成 trace + 嵌套 span:LLM span / tool span / retriever span),才谈得上可评估。没有 trace,错误归因只能瞎猜。
- *追问:span 三类各记什么?* LLM span 记 prompt/输出/token/延迟/成本/模型名;tool span 记工具名/入参/出参/状态;retriever span 记 query/命中文档/召回质量。

**Q3:LLM-as-judge 有哪些坑,怎么治?**
坑:① 裁判偏好(位置/长度/自我偏好)→ 打乱顺序、成对比较去偏;② rubric 太虚 → 给明确标准 + few-shot;③ judge 会漂移 → 拿人工小集合校准、定期复校。**盲信没校准的 judge = 用一把没校准的尺子量。**
- *陷阱:「judge 给 9 分就说明 agent 好?」* judge 分只在和人工标注校准过、且 rubric 可执行时才可信;且要配 tool 层硬指标交叉验证。

**Q4:离线评估和在线监控有什么区别?**
**离线(开发期)**:跑回归集,门禁式,指标回退就别合并,慢工出细活可用昂贵 judge。**在线(生产期)**:真实流量采样打 trace,看实时完成率/错误率/p95/成本曲线,异常告警,要轻量低开销。两者 + **human-in-the-loop** 捞坏例反哺,构成评估飞轮(与 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 同精神,搬到系统工程层)。

**Q5(开放):你怎么衡量一个生产 agent 是否健康?**
四层指标 + 一句总纲:**结果层**完成率/正确率/满意度、**轨迹层**绕路/死循环率、**工具层** tool 成功率(最高信噪比)、**资源层** p95 延迟/单次成本/token/步数;线上盯曲线 + 告警 + 人审捞坏例反哺回归集。总纲:**做不出好的评估,就做不出好的 agent**——能稳定衡量才能稳定改进。

## 知识拓展

**前沿与进阶:**
- **Agent-as-a-Judge**(2024):用一个**完整 agent**(能调工具、看中间状态)而非单次 LLM 调用来评另一个 agent 的轨迹,比纯 LLM-as-judge 更能抓到 step 级问题——评估本身也在「agent 化」。
- **τ-bench / τ²-bench**(Sierra,2024–2025):专测「agent 在真实工具环境 + 模拟用户多轮交互」下的可靠性,强调**多次运行的一致性**(pass^k:跑 k 次都对的概率),直击 agent 随机性这一难点。
- **过程奖励 / 过程监督**(Process Reward Model,2023–):给轨迹**每一步**打分而非只看终点,既是评估也是 RL 训练信号——与 [[32 Agentic RL 与训练|Agentic RL 与训练]] 接壤(评估的步骤层 ↔ 训练的过程奖励是同一个东西的两面)。
- **OpenTelemetry GenAI 语义约定**仍在演进中(span 属性逐步标准化),是「埋点一次后端随便换」长期可行的基石。

**边界与反模式:**
- **反模式:judge 当唯一真理**。judge 是软指标的规模化手段,不是 ground truth;高风险场景(医疗/金融/安全)仍需人审 + 硬指标兜底。
- **反模式:可观测性事后补**。它是**第一天的基建**,不是出事后补;哪怕原型期手动打点也比裸跑强。
- **边界:指标好 ≠ 用户满意**。回归集和 judge 都可能 overfit;线上真实满意度(显式反馈、留存)是最终裁判。
- **边界:成本/延迟是一等指标**。质量调上去但 token 翻 5 倍、延迟到分钟级,生产即不可用——资源层和质量层同等重要(展开见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]])。

**相关链接:** RAG 专项评估见 [[36 Agentic RAG|Agentic RAG]](`ragas` 的 faithfulness/context-recall);评估精神同源 [[08 Evaluator-Optimizer|Evaluator-Optimizer]](运行时闭环 ↔ 开发期离线版);多体可观测难点见 [[22 多智能体系统|多智能体系统]] 与 [[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]];选型与框架绑定权衡见 [[37 Agent 框架对比|Agent 框架对比]] 与 [[39 Agent 开源生态全景|Agent 开源生态全景]] ④ 层;过程监督接 [[32 Agentic RL 与训练|Agentic RL 与训练]]。
