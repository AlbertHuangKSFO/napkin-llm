[[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]] 的核心主张反直觉到值得贴在墙上:**别再手搓 prompt 了——把 prompt 和流程当成一段可以「编译、可以被优化」的程序来写**。你只声明「输入是什么、输出是什么」(Signature),组装出推理流程(Module),给一个评判好坏的 metric 和一小撮样例,然后让 **Optimizer 自动**去选 few-shot 例子、重写指令文本,**编译**出一份优化过的 prompt。手写 prompt 那种「改一个词、跑一遍、肉眼看、再改」的炼丹循环,被换成了「写程序 + 自动调参」。

这把 prompt engineering 从**手艺**变成了**工程**:就像没人愿意手写汇编、而是写高级语言交给编译器,DSPy 想让你写「声明式的 LM 程序」交给优化器去生成 prompt。它和 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 是同一种精神在不同层面的体现——后者是 agent 运行时让一个评估者驱动一个生成者迭代;DSPy 是**离线、在开发期**用 metric 驱动优化器迭代 prompt,产物是一份固定下来的优化程序。

![[Agent 提示词优化(DSPy).svg]]

## 本质:prompt 是「待优化的参数」,不是手写的字符串

传统 prompt engineering 的隐含假设是「prompt 是人写的文本」。DSPy 把这个假设掀翻:**prompt 里的指令措辞、few-shot 示例的选取,本质上都是「参数」,既然是参数就应该被自动优化,而不是人手调。**

这带来三层解耦,正是 DSPy 全部价值的来源:
1. **逻辑与措辞解耦**:你描述「这一步要把问题变成答案」(逻辑),具体用什么措辞、给几个例子(措辞)交给优化器。
2. **程序与模型解耦**:同一段 DSPy 程序,换个底座模型(GPT → Claude → 本地 Qwen)只需**重新 compile 一遍**,优化器会针对新模型重新挑例子、重写指令——而手写 prompt 换模型往往要从头调。
3. **流程与 prompt 解耦**:一个多步 pipeline(检索→推理→生成)里每一步的 prompt 都能被**联合优化**,而不是各调各的。

一句话:**手写 prompt 是把「人对模型脾气的猜测」硬编码进字符串;DSPy 是把这件事交给数据和 metric 去搜索**。当任务有客观可评(数学对错、代码能不能跑、答案是否匹配)时,这种自动搜索通常比人手调更稳、更省时间,且换模型不返工。

## 机制:Signature → Module → Optimizer → 编译产物(分步)

DSPy 的四个核心抽象,层层叠加:

**① Signature(签名)——声明输入输出契约**
最小形式是一个字符串:`"question -> answer"`,意思是「这一步吃一个 question,吐一个 answer」。注意:**你没有写任何 prompt 文本**,只声明了接口。也可以写成带类型和描述的 class 形式,给每个字段加说明(`question: str = dspy.InputField(desc="用户问题")`)。Signature 是「这一步在干嘛」的语义契约,**prompt 的实际文本由 DSPy 在运行/编译时按签名 + Module 策略自动生成**。

**② Module(模块)——给签名套上推理策略**
Module 把一个 Signature 包装成可调用的组件,内置不同的推理策略:
- `dspy.Predict(sig)`:最朴素,直接让模型按签名产出;
- `dspy.ChainOfThought(sig)`:在产出答案前自动插入「让模型先逐步推理」的结构(即把思维链 CoT 这种提示技巧**程序化**,你不用手写「Let's think step by step」);
- `dspy.ReAct(sig, tools=[...])`:把签名变成一个能调工具的 [[09 ReAct|ReAct]] 循环(思考→调工具→观察→再思考)。
Module 可以**像神经网络层一样组合**:在一个 `dspy.Module` 子类里串起检索、ChainOfThought、生成,构成一个完整 pipeline。这正是「programming, not prompting」——你写的是控制流和模块,不是 prompt 字符串。

**③ Metric + 训练集——定义「好」并提供少量样例**
你写一个 `metric(gold, pred) -> bool/float`(答案对不对、F1、是否通过测试……),再给**一小撮**带标注的例子(常见几十到几百条,DSPy 的卖点之一就是**省标注**)。metric 是优化器的指南针——这步直接呼应 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]:**没有可靠的 metric,优化无从谈起**,优化器优化的就是这个 metric。

**④ Optimizer(优化器,旧称 Teleprompter)——compile 出优化后的程序**
这是 DSPy 的灵魂。调用 `optimizer.compile(program, trainset=...)`,优化器会**自动**做两件事:
- **挑 few-shot 示例**:从训练集里(或自己 bootstrap 生成的成功轨迹里)选出最能提升 metric 的示例塞进 prompt;
- **重写指令文本**:对每个 predictor,用 LLM 生成多个候选指令措辞,搜索出 metric 最高的那版。

主要优化器:
- **`BootstrapFewShot`**:让程序自己在训练集上跑,把**成功的轨迹**当 few-shot 示例回填(自举),最常用的入门优化器;
- **`MIPROv2`**(Multi-prompt Instruction PRoposal Optimizer v2):**联合优化指令 + few-shot 示例**,用贝叶斯优化在「指令措辞 × 示例组合」的空间里搜最优,是 DSPy 的主力优化器;
- **`GEPA`**(DSPy 3 新增):基于反思的优化器,**当你能给出文本反馈**(报错、schema 违规、测试 diff)时收敛更快,把错误信息变成改 prompt 的信号;
- **`SIMBA`** 等:还有面向不同场景的优化器。

**编译产物 = 一份「填好的 prompt 模板 + 精选 demo」的程序**,可以保存、加载、直接部署。后续换模型,重新 compile 即可。

## 来源 / 出处

- **DSPy 出自 Stanford NLP**(Omar Khattab 等),前身叫 **DSP**(Demonstrate-Search-Predict),后演进为 DSPy;奠基论文 *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*(arXiv 2310.03714,2023)。
- 仓库 **`stanfordnlp/dspy`**,2026 年仍是该方向最活跃的开源项目,已发布 **DSPy 3**(引入 GEPA、SIMBA 等新优化器,强化 agent 与 LM 程序的联合优化)。
- 同赛道还有 **TextGrad**(把 LLM 的文本反馈当「梯度」做反向传播)、**AdalFlow**(LLM-AutoDiff,自动微分任意 LLM 工作流)、**APE / OPRO**(把 prompt 优化当黑盒搜索)。2026 年这类工具整体仍偏「研究级 + 早期生产」,DSPy 是其中工程化最成熟、社区最大的。

## 可跑的最小实现

```python
# pip install dspy
import dspy

# 0) 配置底座模型
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 1) Signature:只声明输入→输出,不写 prompt
class QA(dspy.Signature):
    """根据问题给出简短答案。"""
    question: str = dspy.InputField()
    answer: str   = dspy.OutputField(desc="尽量简短")

# 2) Module:给签名套 ChainOfThought(自动加逐步推理)
program = dspy.ChainOfThought(QA)

# 直接跑(未优化版,DSPy 已按签名自动生成了 prompt)
print(program(question="法国的首都是?").answer)   # -> 巴黎

# 3) Metric + 训练集
trainset = [
    dspy.Example(question="2+2=?", answer="4").with_inputs("question"),
    dspy.Example(question="水的化学式?", answer="H2O").with_inputs("question"),
    # ... 几十条即可
]
def metric(gold, pred, trace=None):
    return gold.answer.strip().lower() == pred.answer.strip().lower()

# 4) Optimizer:编译出优化后的程序(自动选 demo + 重写指令)
opt = dspy.MIPROv2(metric=metric, auto="light")
optimized = opt.compile(program, trainset=trainset)

optimized.save("qa_compiled.json")   # 产物可保存/部署
print(optimized(question="意大利的首都是?").answer)
```

对比**手写 prompt** 的同一件事:你会写一段 `f"你是问答助手,请逐步推理后回答。示例:Q...A...\n问题:{q}"`,然后凭感觉改措辞、换示例、换模型时全部返工。DSPy 把「写哪段措辞、选哪几个示例」整个交给 `compile()`,你只对 metric 负责。

## 对比:手写 Prompt vs DSPy

| 维度 | 手写 Prompt | DSPy |
|---|---|---|
| prompt 文本 | 人手写、人手调 | **优化器自动生成/重写** |
| few-shot 示例 | 人手挑 | **自动从数据/成功轨迹里选** |
| 换模型 | 往往从头重调 | **重新 compile 即可** |
| 多步 pipeline | 每步各调,易顾此失彼 | **可联合优化整条链** |
| 依赖 | 直觉 + 试错 | **metric + 少量标注数据** |
| 可复现 | 难(改动散在字符串里) | 高(逻辑在代码,措辞是编译产物) |
| 适用前提 | 任何任务 | **任务要有可量化的 metric** |

## 何时用 / 何时别用(及坑)

**该用 DSPy**:
- 任务**有客观可评的 metric**(分类对错、抽取 F1、答案匹配、代码能否通过、数学对错)——优化器有指南针才好使;
- 你有(或能 bootstrap 出)**少量标注样例**;
- **会频繁换模型**或要把同一逻辑迁到多个模型(DSPy 重 compile 即可,省返工);
- 多步 LM pipeline / agent,想**端到端**优化而非逐段手调。

**别硬上**:
- 任务**无法量化好坏**(纯开放式创作、主观审美)——没 metric,优化器瞎搜;
- 一次性、简单的小 prompt——直接写更快,套 DSPy 是过度工程;
- 你连**评估基线都没有**——先把 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的评估搭起来,再谈优化。

**坑**:
- **metric 是天花板也是地板**。优化器只会无脑最大化你给的 metric——metric 写歪了(比如只测准确率不测格式),它就会优化出一个「准但不可用」的 prompt。metric 工程往往比 prompt 工程还关键。
- **compile 有成本**。MIPROv2 等要跑很多次模型调用来搜索,token 和时间开销不小;`auto="light/medium/heavy"` 控制搜索强度,要权衡。
- **编译产物会过拟合小训练集**。样例太少、太同质,优化出的 prompt 可能只在训练分布上好看;要留**验证集**单独评(再次回到 [[38 Agent 评估与可观测性|Agent 评估与可观测性]])。
- **学习曲线**:Signature / Module / Optimizer 这套抽象有门槛,简单任务上「为了用 DSPy 而用 DSPy」不划算。
- **DSPy 优化的是「不改权重」的 prompt 层**。它和 [[32 Agentic RL 与训练|Agentic RL 与训练]] 是两条路:DSPy 调 prompt/示例(便宜、快、不动模型),RL 直接训练模型权重(贵、慢、上限更高)。两者可叠加——先 DSPy 把 prompt 调好,再决定要不要上 RL。

## 关键事实速记

- DSPy 一句话:**把 prompt/流程当可编译可优化的程序;你写 Signature+Module+metric,优化器 compile 出 prompt**。
- 四件套:**Signature(声明 I/O)→ Module(Predict/ChainOfThought/ReAct)→ Metric+训练集 → Optimizer.compile()**。
- 主力优化器:`BootstrapFewShot`(自举示例)、`MIPROv2`(联合优化指令+示例,贝叶斯搜索)、`GEPA`(反思式,吃文本反馈)。
- 出处:**Stanford NLP,前身 DSP**,论文 arXiv 2310.03714,2026 已到 **DSPy 3**。
- 适用前提:**任务有可量化 metric + 少量样例**;换模型只需重 compile。
- 与 RL 的分工:**DSPy 不改权重(prompt 层),[[32 Agentic RL 与训练|Agentic RL 与训练]] 改权重(模型层)**。

## 主流开源实现 / Python 库

- **`stanfordnlp/dspy`**(pip:`dspy`):本主题的本体,「programming—not prompting」框架,提供 Signature/Module/Optimizer 全套,2026 已 DSPy 3,社区最大、最成熟。
- **`zou-group/textgrad`**(pip:`textgrad`):把 LLM 给出的**文本反馈当「梯度」**做反向传播来优化 prompt/变量,与 DSPy 互补的另一条技术路线。
- **`SylphAI-Inc/AdalFlow`**(pip:`adalflow`,原 LightRAG 团队的 LLM-AutoDiff):**自动微分任意 LLM 工作流**,面向构建+优化 agent 式多步流程。
- **`keirp/automatic_prompt_engineer`**(APE):把 prompt 优化当**黑盒搜索**的早期经典实现(LLM 生成候选指令再打分筛选),概念奠基性强,工程化弱于 DSPy。

## 工业界实践

**① GEPA 是 2025–2026 这条线上的"明星",值得单独细说。** `GEPA`(Genetic-Pareto,论文 arXiv 2507.19457,Agrawal 等 2025,**ICLR 2026 Oral**)是 DSPy 3 引入的反思式优化器,它把"省样本"和"用文本反馈"两件事推到了新高度,工程意义极大:
- **机制**:让 LLM **反思整条执行轨迹**(输入、输出、失败、报错/测试 diff 等文本反馈),针对某个 module 提出新的指令文本——把"错误信息"直接变成"改 prompt 的信号"。
- **Pareto 前沿**:不只进化"全局最优"那一版(易陷局部最优/停滞),而是**维护一组 Pareto 前沿候选**(每个候选至少在某个评测样本上最好),从前沿采样杂交,泛化更稳。
- **惊人的样本效率**:**10 条训练样例、20–100 次评测**就能出效果;对比 **GRPO 这类 RL 常要 10000+ rollout**,MIPROv2 推荐 200+ 样例 + 40+ trial。论文标题直接喊出 *Reflective Prompt Evolution **Can Outperform Reinforcement Learning***——在 Qwen3-8B 上**比 GRPO 高出最多 20%、比 MIPROv2 高 13%**。这给出一个反直觉但重要的工程判断:**很多场景下,先把 prompt 优化(尤其 GEPA)做透,可能就够了,不必上昂贵的 RL**(对照 [[32 Agentic RL 与训练|Agentic RL 与训练]])。
- GEPA 也独立成库 `gepa-ai/gepa`,可优化 prompt、代码等任意文本组件,不限于 DSPy。

**② 生产中怎么把 DSPy 用对。**
- **典型场景**:分类/抽取/RAG 问答/工具路由/SQL 生成这类**有客观 metric**的任务,DSPy 把"调 prompt"变成"写 metric + compile",可复现、可回归、换模型不返工。
- **换模型不返工是核心卖点**:底座从 GPT 换 Claude 换本地 Qwen,只需 `recompile`,优化器重新挑示例、重写指令。生产里模型迭代频繁,这省下大量重调成本。
- **优化器选型**:入门 `BootstrapFewShot`(自举成功轨迹当 demo);主力 `MIPROv2`(联合优化指令+示例,贝叶斯搜索,要 200+ 样例);**样本少 / 有文本反馈(报错、schema 违规、测试 diff)选 `GEPA`**;还有 `SIMBA` 等。
- **`auto` 档位控成本**:`auto="light/medium/heavy"` 调搜索强度——compile 本身要跑大量模型调用,token 不便宜,先 light 试。

**③ 可观测与运维。**
- compile 是**离线开发期**行为,产物是"填好的 prompt 模板 + 精选 demo"的固定程序,可 `save/load`、可版本化、可进 CI。把 compiled 程序当**代码资产**纳入 git,而非散在字符串里的炼丹笔记。
- **必须留验证集**:小训练集易过拟合,优化出的 prompt 可能只在训练分布好看;独立 dev/test 集评 metric,接 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 做回归。
- **DSPy + 可观测平台**:配 MLflow / Langfuse 记录每次 compile 的 metric 曲线、最终 prompt、token 开销,便于对比迭代。

```python
# 生产里更常见的形态:GEPA 吃文本反馈 + 留验证集
import dspy
program = dspy.ChainOfThought("question -> answer")

def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ok = gold.answer.strip().lower() == pred.answer.strip().lower()
    # GEPA 的关键:返回分数 + 文本反馈(失败原因),让优化器据此反思改 prompt
    fb = "正确" if ok else f"答错:期望「{gold.answer}」实际「{pred.answer}」"
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback=fb)

gepa = dspy.GEPA(metric=metric_with_feedback, auto="light",
                 reflection_lm=dspy.LM("openai/gpt-4o"))  # 反思用强模型
optimized = gepa.compile(program, trainset=trainset, valset=valset)  # 必留 valset
optimized.save("compiled_gepa.json")   # 产物当代码资产入库
```

## 面试高频

**Q1:DSPy 一句话讲清是什么?为什么叫 "programming, not prompting"?**
标准答:**把 prompt 和流程当成可编译、可优化的程序**——你声明输入输出(Signature)、组装推理流程(Module)、给 metric + 少量样例,优化器自动选 few-shot、重写指令,**compile** 出优化过的 prompt。你写的是逻辑(代码),措辞是编译产物,而非手搓字符串。追问"和手写 prompt 的本质差别":手写是把"人对模型脾气的猜测"硬编码进字符串;DSPy 是把这件事交给**数据 + metric 去搜索**。

**Q2:DSPy 四件套是什么?**
答:**Signature**(声明 I/O 契约,如 `"question -> answer"`,不写 prompt)→ **Module**(套推理策略:`Predict`/`ChainOfThought`/`ReAct`)→ **Metric + 训练集**(定义"好" + 少量样例)→ **Optimizer.compile()**(自动挑 demo + 重写指令)。追问主力优化器:`BootstrapFewShot`、`MIPROv2`、`GEPA`(各自适用见 Q4)。

**Q3:DSPy 和 Agentic RL 是什么关系?(高频对照)**
答:**同目标两条路**——DSPy **不改权重**,只在 prompt 层选示例/改指令(便宜、快、可逆、受底座能力封顶);[[32 Agentic RL 与训练|Agentic RL]] **改权重**(贵、慢、能突破底座原有能力)。口诀:**DSPy 调嘴(prompt),RL 改脑(权重)**。实践次序:先穷尽 prompt/上下文工程(尤其 GEPA),确认天花板真在模型本身,再上 RL。追问"GEPA 论文为什么说能 outperform RL":样本效率高 100×(10 样例 vs 10000+ rollout),很多任务 prompt 层就够顶。

**Q4:三个主力优化器怎么选?**
答:① **样本多(200+)、要联合优化指令+示例** → `MIPROv2`(贝叶斯搜索);② **样本少(~10)、有文本反馈(报错/测试 diff/schema 违规)** → `GEPA`(反思式,把错误变信号,样本效率最高);③ **快速入门、自举示例** → `BootstrapFewShot`。

**Q5(陷阱):DSPy 适用任何任务吗?最大的坑是什么?**
答:**不**。前提是**任务有可量化 metric**——纯开放创作/主观审美没法量化,优化器没指南针只能瞎搜。最大的坑:**metric 是天花板也是地板**——优化器只会无脑最大化你给的 metric,metric 写歪(只测准确率不测格式)就优化出"准但不可用"的 prompt。metric 工程往往比 prompt 工程更关键。次坑:小训练集过拟合(必留验证集)、compile 有 token 成本。

## 知识拓展

- **同赛道技术谱系(都在"自动优化 LLM 程序")**:
  - **DSPy**(Stanford NLP,Omar Khattab 等;前身 DSP=Demonstrate-Search-Predict;论文 arXiv 2310.03714,2023):工程化最成熟、社区最大。
  - **TextGrad**(`zou-group/textgrad`):把 LLM 的**文本反馈当"梯度"**做反向传播,与 GEPA 的"反思即信号"哲学相通,但形式上更像自动微分。
  - **AdalFlow / LLM-AutoDiff**(`SylphAI-Inc/AdalFlow`):自动微分任意 LLM 工作流。
  - **APE / OPRO**:把 prompt 优化当黑盒搜索的早期经典(LLM 生成候选指令再打分),奠基性强、工程化弱。
- **GEPA 的更大意义**:它把"反思 + 进化 + Pareto 多样性"组合起来,证明在样本稀缺时,**结构化文本反思可以逼近甚至超过 RL**——这条线(reflective text evolution)2026 正热,且独立成库可优化 prompt/代码/任意文本组件。论文 ICLR 2026 Oral。
- **与 Evaluator-Optimizer 的关系**:DSPy 和 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 是同一精神在不同层面——后者是 agent **运行时**评估者驱动生成者迭代;DSPy 是**开发期离线**用 metric 驱动优化器迭代 prompt,产物固定。
- **边界 / 反模式**:① 无可量化 metric 硬上(优化器瞎搜);② 一次性小 prompt 套 DSPy(直接写更快,过度工程);③ 没评估基线就谈优化(先搭 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]);④ 不留验证集导致过拟合小训练集;⑤ metric 写歪优化出"准但不可用"的 prompt。
- **相关兄弟**:[[32 Agentic RL 与训练|Agentic RL 与训练]](改权重那条路,本篇的对照面)、[[08 Evaluator-Optimizer|Evaluator-Optimizer]](运行时迭代)、[[38 Agent 评估与可观测性|Agent 评估与可观测性]](metric 的来源)、[[09 ReAct|ReAct]](`dspy.ReAct` 把它程序化)、[[20 上下文工程|上下文工程]](prompt 优化是其中一环)。
