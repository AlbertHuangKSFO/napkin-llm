[[32 Agentic RL 与训练|Agentic RL 与训练]] 是把 agent 能力提升从「调 prompt」推进到「**直接训练模型权重**」的那条路:不再只在推理时用 [[09 ReAct|ReAct]]、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 这些 prompt 技巧绕,而是用强化学习(RL)**真正改变模型的策略**,让它在多步推理、调工具、长程任务上从经验里学会「怎么做更容易成功」。一句话——**prompt 优化是教模型「这次怎么答」,agentic RL 是训模型「以后遇到这类任务都更会做」**。

它和 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]] 是同一目标的两条路、互为对照:DSPy **不动权重**,只在 prompt 层选示例、改指令(便宜、快、可逆);agentic RL **改权重**(贵、慢、需算力,但能力上限更高、能学到 prompt 学不到的东西)。两者常叠加:先用 DSPy/prompt 把基线拉好,再决定要不要上 RL 把天花板顶上去。本篇讲后者。

![[Agentic RL 与训练.png]]

## 本质:从「对齐人类偏好」到「在任务里赢」

经典 LLM 训练的 RL 阶段([[LLM/085 RLHF 全流程与 KL 约束、奖励黑客|RLHF]])解决的是**对齐**:让模型的输出更符合人类偏好(有用、无害、诚实)。**Agentic RL 把目标换了**:不只对齐口味,而是训练模型在**真实任务环境**里——写代码、解数学、跑多步检索、操作浏览器、修 GitHub issue——**多步决策并最终成功**。

关键差异在于「一次交互」的含义变了:
- 传统 RLHF 里,一条「轨迹」≈ 模型对一个 prompt 的**一次回答**;
- agentic RL 里,一条**轨迹是一整段 agent 与环境的多步交互**:思考→调工具→看观察→再思考→……→交付。奖励往往要等**整段轨迹跑完**(代码能不能通过测试、issue 修没修好)才给得出。

这就把 RL 的两个老大难问题放大了:**长程信用分配**(成功了,功劳算哪一步?)和**奖励从哪来**(谁来判这段多步轨迹的好坏?)。Agentic RL 的全部技术演进,基本都在回答这两个问题。

## 机制:rollout → reward → policy update 的训练环(分步)

把上图拆成一个闭环,反复迭代:

**① Rollout(采样):让当前策略在环境里跑**
拿**当前的 policy(待训 LLM)**,对一批任务各采样若干条完整轨迹。在 agentic 设定下,这一步就是让模型在**环境**里跑一个 [[09 ReAct|ReAct]] 式的多步循环:思考→调工具→观察→再思考。环境可以是代码沙箱、数学校验器、检索引擎、浏览器([[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]])、或一个真实的 GitHub 仓库([[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]])。一条轨迹 = 一段「状态-动作-观察」序列 + 最终结果。

**② Reward(打分):给轨迹算奖励**
这是 agentic RL 的胜负手,有几条来路(详见下方演进图):
- **[[LLM/088 GRPO 与可验证奖励|可验证奖励(RLVR)]]**:用**确定性工具**判对错——代码跑测试通过 = 1,数学答案校验对 = 1,否则 0。**客观、不可钻空子、不用训奖励模型也不用人标**,是 2025 起推理模型大爆发的核心。
- **奖励模型(RLHF/RLAIF)**:训一个模型来给分(人类偏好 → RLHF;AI 偏好/宪法 → [[LLM/090 RLAIF、宪法 AI 与过程奖励 PRM|RLAIF]])。开放式任务(写作、对话)没法验证,只能靠它。
- **过程奖励 vs 结果奖励**:**结果奖励(ORM)** 只看终点对不对(信号稀疏但便宜客观);**过程奖励(PRM)** 给每一步打分(信号稠密、缓解长程信用分配,但要训 PRM、成本高且易被钻)。

**③ Policy Update(更新):把高分轨迹的概率推上去**
用 RL 算法根据奖励更新模型权重 θ,核心思想都是「**让得高分的轨迹更可能、得低分的更不可能**」:
- **[[LLM/084 策略梯度与 PPO 基础|PPO]]**(Proximal Policy Optimization):经典 actor-critic,需要额外训一个 **critic(价值网络)** 估计每步的期望回报,四个模型(policy + reference + critic + reward model)一起转,重。
- **[[LLM/088 GRPO 与可验证奖励|GRPO]]**(Group Relative Policy Optimization,DeepSeek 提出):**砍掉 critic**。对同一个 prompt 采样**一组**(典型 16 条)回答,用**组内相对优势**(每条减去这组的均值再归一化)当 advantage,省掉价值网络。再叠加 **RLVR**(连奖励模型也省掉),四模型塌成两个(policy + reference),训练大幅简化——这是 DeepSeek-R1 路线的关键。

**组内相对优势手算。** 设一组采 $G=8$ 条轨迹,RLVR 给出 reward(过测试=1,否则 0):$r=[1,1,0,1,0,0,1,0]$($4$ 条成功)。先算组内统计:$\mu=\frac{4}{8}=0.5$,$\sigma=\sqrt{\frac{1}{8}\sum(r_i-\mu)^2}=\sqrt{0.25}=0.5$。再按 $\hat{A}_i=\frac{r_i-\mu}{\sigma+\epsilon}$ 归一化:成功轨迹 $\hat{A}=\frac{1-0.5}{0.5}=+1$,失败轨迹 $\hat{A}=\frac{0-0.5}{0.5}=-1$。于是这组 advantage 是 $[+1,+1,-1,+1,-1,-1,+1,-1]$——**比组内平均好的(成功)拿正优势、概率被推高,差的(失败)拿负优势、被压低**。整组**不需要任何价值网络**估期望回报,优势完全来自「跟同组兄弟比」,这就是 GRPO 砍掉 critic 的核心招。注意:若一组**全对或全错**($\sigma=0$),所有 $\hat{A}\approx0$、梯度为零白算——这正是 DAPO「动态采样丢掉全对/全错组」要解决的问题。
- 通常还加 **KL 约束**:惩罚新策略偏离参考模型(ref)太远,防止训崩、防奖励钻空子(reward hacking)。
- **拒绝采样(rejection sampling / best-of-N)**:更轻量的做法——采样多条,只把**通过验证的**留下来做监督微调(SFT),不一定走完整 RL,常用作冷启动或与 RL 混合。

闭环重复:新策略再去 rollout、再打分、再更新,直到任务 metric 收敛。

![[Agentic RL 与训练-奖励范式演进.png]]

## 来源 / 出处

- **RLHF**(基于人类反馈的 RL):奠基于 OpenAI 的 InstructGPT(2022),是 ChatGPT 对齐的基础范式;算法常用 **PPO**。
- **RLAIF / Constitutional AI**(Anthropic):用 AI 反馈 + 一套「宪法」原则替代大量人工标注,把奖励信号规模化。
- **RLVR + GRPO**:由 **DeepSeek** 的工作推到台前。**DeepSeek-R1 / R1-Zero**(2025-01)证明:仅用 GRPO + 可验证奖励(数学/代码),**不做 SFT 也能让推理能力暴涨**(R1-Zero 在 AIME 2024 从 15.6% 升到 77.9%),把「RLVR 训推理」变成行业主线。
- **Agentic / 工具与长程方向**:**SWE-RL**(Meta,用真实软件演化数据 + 规则奖励训代码 agent)、OpenAI o 系列对**RL + 工具使用**的强调(SWE-bench、Codeforces 等),以及一批专做**多轮、工具调用、长程**的 agentic RL 框架(见末尾)。
- 2026 现状:RLVR/GRPO 已是事实标配;研究前沿在**长程信用分配、过程奖励、多轮工具 RL、把 RL 接到真实软件/终端/GUI 环境**。

## 可跑的最小实现(GRPO 伪码 + 框架调用)

**GRPO 核心思想(伪码,去掉工程细节)**:

```python
# 对每个 prompt:采样一组,用组内相对优势更新,无需 critic
for prompt in batch:
    # ① rollout:同一 prompt 采 G 条完整轨迹(agentic 下是多步工具调用轨迹)
    group = [policy.rollout(prompt, env) for _ in range(G)]      # G≈16

    # ② reward:可验证奖励(代码跑测试 / 数学校验),无需奖励模型
    rewards = [verify(traj) for traj in group]                   # 每条 0/1 或标量

    # ③ 组内相对优势:减均值除标准差(不用价值网络)
    mean, std = avg(rewards), stdev(rewards)
    advantages = [(r - mean) / (std + 1e-6) for r in rewards]

    # ④ policy update:高优势轨迹↑概率,带 KL 约束贴住参考模型
    loss = grpo_objective(group, advantages,
                          ref_model=reference, kl_coef=0.04)
    loss.backward(); optimizer.step()
```

**用框架真实跑(以 `verl` / `TRL` 为例,概念调用)**:

```python
# TRL 的 GRPO(huggingface),最贴近上面伪码的现成实现
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, **kwargs):
    # 可验证奖励:跑测试 / 校验数学答案,返回每条的分数
    return [1.0 if passes_tests(c) else 0.0 for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B",
    reward_funcs=reward_fn,            # 可验证奖励,无需训奖励模型
    train_dataset=dataset,
    args=GRPOConfig(num_generations=16, beta=0.04),  # 组大小 + KL 系数
)
trainer.train()
```

agentic 场景下,`reward_fn` 评的是**一整段多步轨迹**(agent 调了几次工具最后有没有解决问题),`rollout` 要在一个**带工具/环境**的 loop 里跑——这正是 `verl`、`OpenRLHF`、`ART`、`SkyRL` 这类框架在 TRL 之上额外解决的工程难点。

## 对比:Prompt 优化(DSPy) vs Agentic RL

| 维度 | [[31 Agent 提示词优化(DSPy)\|Agent 提示词优化(DSPy)]] | Agentic RL |
|---|---|---|
| 改什么 | **prompt / few-shot 示例** | **模型权重 θ** |
| 改不改模型 | 不改(可逆) | 改(不可逆,得存 checkpoint) |
| 成本 | 低(几十~几百次模型调用) | 高(大量 GPU、长训练) |
| 数据需求 | 少量带标注样例 | 大量任务 + 可算的奖励 + rollout 算力 |
| 能力上限 | 受底座模型本身能力封顶 | **能突破底座原有能力**(学新策略) |
| 迭代速度 | 分钟~小时 | 天~周 |
| 何时选 | 先用它拿基线、快速试 | 基线见顶、且任务可批量验证奖励时 |
| 典型产物 | 优化后的 prompt 程序 | 一个更会做 agent 任务的新模型 |

记忆:**DSPy 调嘴(prompt),RL 改脑(权重)**。绝大多数项目应**先穷尽 prompt/上下文工程**,确认天花板确实在模型本身,再上 RL——RL 贵且工程复杂,不是首选。

## 何时上 Agentic RL / 何时别(及坑)

**该上**:
- 任务**可批量自动验证奖励**(代码测试、数学、SQL 执行结果、检索命中)——RLVR 的主场,这是性价比最高的场景;
- prompt / 上下文工程 / 框架优化都**见顶**了,瓶颈确实在模型策略本身;
- 你有**算力和数据工程能力**(rollout 环境、奖励函数、分布式训练栈)。

**别上**:
- 任务**无法自动判好坏**(纯开放创作)——只能靠奖励模型,又贵又容易被钻;先想别的;
- 还没把 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 和 prompt 调好——**没有可靠奖励/评估,RL 必崩**;
- 小团队、没 GPU 集群——优先 DSPy / 上下文工程 / 蒸馏,别硬啃 RL。

**坑**:
- **奖励钻空子(reward hacking)是头号敌人**。模型会找到「让奖励变高但没真正解决问题」的捷径(比如猜到测试用例、输出讨巧格式)。奖励函数的鲁棒性比算法更重要。
- **长程信用分配难**。一段几十步的 agent 轨迹只有最后一个 0/1 信号,中间哪步对哪步错很难归因;过程奖励(PRM)能缓解但要额外训 PRM 且自身易被钻。
- **rollout 是瓶颈**。agentic RL 每步都要真在环境里跑(调工具、跑沙箱),rollout 又慢又贵,工程上常靠 vLLM/Ray 异步、环境并行来扛——这也是 `verl`/`OpenRLHF` 等框架的主要卖点。
- **训练不稳**。KL 系数、组大小、学习率没调好极易训崩或退化;要密集监控并能回滚 checkpoint。
- **能力会偏科 / 灾难性遗忘**。猛训某类可验证任务(数学/代码)可能损伤通用能力,需混合数据、配 KL 约束、留通用评测把关。
- 与 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的区别要分清:Reflexion 是**推理时**用语言反思改下一次尝试(不改权重);agentic RL 是**训练时**用奖励改权重。两者可类比(都靠「失败反馈→改进」),但一个在 prompt/记忆层、一个在参数层。

## 关键事实速记

- 一句话:**agentic RL = 用 RL 直接训模型在多步/工具/长程任务里的策略**,产物是更会做 agent 的新模型(对比 DSPy 只调 prompt)。
- 训练环:**rollout(环境里多步跑)→ reward(打分)→ policy update(更新权重)**,闭环迭代。
- 奖励演进:**RLHF(人类偏好)→ RLAIF(AI 偏好/宪法)→ RLVR(确定性可验证奖励)**;正交维度 **过程奖励(PRM)vs 结果奖励(ORM)**。
- 算法:**PPO**(带 critic,重)→ **GRPO**(组内相对优势,砍 critic)+ **RLVR**(砍奖励模型)→ DeepSeek-R1 路线;轻量替代:**拒绝采样 / best-of-N + SFT**。
- 头号坑:**reward hacking、长程信用分配、rollout 成本、训练不稳**。
- 标志性来源:**DeepSeek-R1(RLVR+GRPO)、SWE-RL(代码 agent)**。

## 主流开源实现 / Python 库

- **`volcengine/verl`**(亦在 `verl-project/verl`,pip 名 `verl`):字节跳动开源的高性能 RL 后训练框架(HybridFlow),scalability 强、支持 GRPO/PPO 等,2026 最主流的 agentic RL 训练栈之一,生态最大。
- **`huggingface/trl`**(pip:`trl`):HuggingFace 的 Transformer RL 库,提供 `GRPOTrainer`/`PPOTrainer` 等,API 最易上手,入门首选。
- **`OpenRLHF/OpenRLHF`**(pip:`openrlhf`):基于 **Ray + vLLM** 的高性能 RLHF/agentic RL 框架,2026 已加多轮 VLM RL、异步 RL,生产级。
- **`OpenPipe/ART`**(Agent Reinforcement Trainer,pip:`openpipe-art`):专为**多步 agent**用 GRPO 做「在岗训练」的轻量框架,能直接接 LangGraph 等 agent,贴 agentic 场景。
- **`NovaSky-AI/SkyRL`**:模块化全栈 RL 库,含 `skyrl-train`(训练)+ `skyrl-gym`(数学/代码/检索/SQL 等工具环境)+ `skyagent`(长程真实 agent),环境-训练一体化。
- **`RAGEN-AI/RAGEN`**:基于 verl 扩展,专注**多轮对话 + 多样 RL 环境**的 agentic RL,补 verl 在多轮交互上的能力。

## 工业界实践

**① 主流训练栈与算法版图(2026)。**
- **算法**:`PPO`(带 critic,重,四模型)→ `GRPO`(组内相对优势砍 critic,DeepSeek-R1 路线)→ **`DAPO`**(字节,GRPO 的工业改良:解耦 clip 上/下界 + 动态采样 + token-level 损失 + 软超长惩罚,**在 Qwen2.5-32B 上 AIME 2024 达 50 分,超过 GRPO 基线**,2026 已是 verl/OpenRLHF 标配)。还有 `REINFORCE++`、`RLOO`、`GSPO`(序列级)等变体。
- **训练框架**:`verl`(字节,HybridFlow,生态最大、最主流)、`OpenRLHF`(Ray+vLLM,已支持 DAPO/REINFORCE++/VLM/异步 RL)、`TRL`(HF,最易上手)。专做 agentic 多步的:`ART`、`SkyRL`、`RAGEN`、`VerlTool`(verl 之上做工具用 RL)。

**② rollout 是头号工程瓶颈,工业界的解法收敛到"异步 + 解耦"。** agentic RL 每步都要**真在环境里跑**(调工具、跑沙箱、查检索),rollout 又慢又贵,且多工具轨迹**异步展开、各工具返回快慢不一**,同步等会让 GPU 大量空转。主流工程手段:
- **异步 rollout**:verl 用 asyncio 协程异步发 rollout 请求,工具调用返回前不阻塞 GPU(`agent_loop`);OpenRLHF 走 async RL。
- **Rollout-as-a-Service / 解耦三段流水线**:NVIDIA **ProRL Agent**(2026)把多轮 agent 训练拆成 **INIT → RUN(多轮轨迹收集)→ EVAL(打分)** 三段,各段独立 worker 池、阶段重叠流水,把"采样"从"训练"里解耦出去当服务,显著提吞吐。这是 2026 多轮 agentic RL 规模化的代表架构。
- **推理引擎复用**:rollout 用 vLLM/SGLang 高吞吐生成,训练用 FSDP/Megatron,二者通过权重同步衔接。

**③ 奖励工程比算法更决定成败。**
- **RLVR 是性价比之王**:用确定性工具判对错(代码跑测试=1、数学校验=1),客观、不可钻、免训奖励模型免人标——这是 2025 起推理模型大爆发的核心(DeepSeek-R1-Zero 仅 GRPO+RLVR,AIME 2024 从 15.6%→77.9%,**不做 SFT**)。
- **奖励钻空子(reward hacking)是头号敌人**:模型会找"奖励变高但没真解决问题"的捷径(猜测试用例、输出讨巧格式、利用校验器漏洞)。工程上要把奖励函数做得鲁棒(沙箱隔离、隐藏测试、多重校验),奖励函数的健壮性比算法选型更重要。
- **过程 vs 结果奖励**:`ORM`(只看终点,稀疏但客观便宜)vs `PRM`(每步打分,稠密缓解长程信用分配,但要训 PRM、自身易被钻)。2026 前沿在用"可验证的中间检查点"做半稠密奖励,折中两者。

**④ 可观测与运维(RL 训练的特殊性)。**
- 必须**密集监控**:reward 曲线、KL 散度、response 长度、熵、各任务子 metric——KL 飙升/熵塌缩/长度爆炸都是训崩前兆。
- **必须能回滚 checkpoint**:RL 改权重不可逆,训退化要能秒退到上一个好 checkpoint。
- **留通用评测把关灾难性遗忘**:猛训数学/代码可能损伤通用能力,混合数据 + KL 约束 + 定期跑 MMLU/通用 benchmark 兜底。
- 训练前先用 [[31 Agent 提示词优化(DSPy)|DSPy]] / 上下文工程拉满基线,**确认天花板真在模型本身**再上 RL——RL 贵且工程复杂,绝不是首选。

```python
# DAPO 相比 GRPO 的几个工业改良(概念配置)
config = {
    "algo": "dapo",
    "clip_ratio_low": 0.2, "clip_ratio_high": 0.28,  # 解耦 clip 上下界(放宽探索)
    "dynamic_sampling": True,    # 动态采样:丢掉全对/全错的组(梯度为 0,白算)
    "loss_agg": "token_mean",    # token-level 损失(长轨迹各 token 公平加权)
    "overlong_penalty": "soft",  # 软超长惩罚(超长轨迹温和扣分,防长度爆炸)
    "kl_coef": 0.0,              # DAPO 常去掉 KL(纯 RLVR 时不贴参考模型)
    "rollout_engine": "vllm", "async_rollout": True,  # 异步 rollout 防 GPU 空转
}
```

## 面试高频

**Q1:Agentic RL 和经典 RLHF 差在哪?**
标准答:RLHF 解决**对齐**(输出符合人类偏好),"一条轨迹"≈对一个 prompt 的**一次回答**;Agentic RL 训模型在**真实任务环境里多步决策并最终成功**(写代码、解数学、多步检索),"一条轨迹"是**一整段 agent↔环境的多步交互**,奖励常要**整段跑完**(代码过没过测试)才给。追问"这放大了哪两个老难题":**长程信用分配**(成功了功劳算哪步)+ **奖励从哪来**(谁判多步轨迹好坏)。

**Q2:GRPO 相比 PPO 省了什么?为什么能省?**
答:PPO 是 actor-critic,要额外训 **critic(价值网络)** 估每步期望回报,四模型(policy+reference+critic+reward model)一起转,重。GRPO **砍掉 critic**:对同一 prompt 采**一组**(典型 16 条),用**组内相对优势**(每条减组均值除标准差)当 advantage,省掉价值网络;再叠 **RLVR** 连奖励模型也省,塌成两模型(policy+reference)。追问 DAPO:GRPO 的工业改良(解耦 clip、动态采样、token-level 损失、软超长惩罚)。

**Q3:RLVR 是什么?为什么 2025 起这么火?**
答:**可验证奖励**——用确定性工具判对错(代码跑测试通过=1、数学校验对=1,否则 0),**客观、不可钻、免奖励模型免人标**。火的原因:DeepSeek-R1-Zero 证明仅 GRPO+RLVR、**不做 SFT** 就能让推理能力暴涨(AIME 15.6%→77.9%),把"用可验证奖励训长推理"变成行业主线,且成本远低于训奖励模型。

**Q4:Agentic RL 四大坑?**
答:① **reward hacking**(找捷径骗高分,头号敌人,奖励鲁棒性 > 算法);② **长程信用分配**(几十步只有最后 0/1 信号,中间难归因;PRM 缓解但自身易钻);③ **rollout 成本**(每步真跑环境,慢且贵,靠异步/解耦/vLLM 扛);④ **训练不稳 + 灾难性遗忘**(KL/组大小/学习率没调好易崩,猛训某类任务损伤通用能力,需混数据+KL+通用评测)。

**Q5:DSPy 和 Agentic RL 怎么选?(对照高频)**
答:**DSPy 调嘴(prompt 层,不改权重,便宜可逆,受底座封顶),RL 改脑(权重层,贵慢不可逆,能突破底座)**。次序:先穷尽 prompt/上下文工程(GEPA 等),确认瓶颈真在模型策略,且任务能批量自动验证奖励、有算力,再上 RL。陷阱:有人一上来就想 RL——绝大多数项目应先把便宜的做透。

**Q6(陷阱):Reflexion 和 Agentic RL 都靠"失败反馈改进",一样吗?**
答:**不一样**。[[13 Reflection 与 Reflexion|Reflexion]] 是**推理时**用语言反思改下一次尝试(在 prompt/记忆层,**不改权重**);Agentic RL 是**训练时**用奖励改**权重**(参数层)。可类比(都靠失败反馈→改进),但一个在推理时一个在训练时,一个改文本一个改参数。

## 知识拓展

- **后训练 2026 全景**:GRPO 已是事实标配,DAPO 成工业主力,RLVR 是奖励默认;前沿在**长程信用分配、过程奖励、多轮工具 RL、把 RL 接进真实软件/终端/GUI 环境**。标志性工作:DeepSeek-R1(RLVR+GRPO)、DAPO(字节)、SWE-RL(Meta,真实软件演化数据+规则奖励训代码 agent)、ProRL Agent(NVIDIA,rollout-as-a-service)、VerlTool(工具用 RL)。
- **"RL 训推理"的本质 = test-time compute**:RLVR 训出的长思维链,本质是教模型在推理时**多花算力多想几步**——与 Deep Research(见 [[29 Deep Research Agent|Deep Research Agent]])多搜多读、与 [[LLM/089 推理模型与 RL：o1、R1 的长 CoT 与自我反思|推理模型]] 多想,是同一哲学(用更多 token 换质量)的不同载体。
- **轻量替代,别动辄全量 RL**:① **拒绝采样 / best-of-N + SFT**——采样多条只留通过验证的做 SFT,常作冷启动;② **GEPA 等反思式 prompt 优化**——论文(arXiv 2507.19457)显示样本效率高 100×(10 样例 vs 10000+ rollout),很多任务 prompt 层就够,不必上 RL(见 [[31 Agent 提示词优化(DSPy)|DSPy]])。判断是否上 RL 的关键:基线是否真见顶 + 奖励是否可批量自动验证 + 是否有算力。
- **环境是新瓶颈**:2026 agentic RL 的竞争力越来越取决于**高质量、可并行、奖励可验证的环境**(SkyRL 的 skyrl-gym、各类工具/代码/检索/SQL 沙箱)——好算法人人有,好环境才稀缺。
- **边界 / 反模式**:① 任务无法自动判好坏硬上 RL(只能靠奖励模型,又贵又易钻);② 没搭好评估/奖励就训(没可靠奖励 RL 必崩);③ 小团队没 GPU 集群啃 RL(优先 DSPy/上下文工程/蒸馏);④ 只盯单类可验证任务猛训(灾难性遗忘,通用能力退化);⑤ 奖励函数不做鲁棒化(被 reward hacking 钻穿)。
- **相关兄弟**:[[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]](不改权重那条路,本篇对照面)、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](推理时的"失败反馈→改进")、[[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]] 与 [[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]](agentic RL 的典型环境)、[[38 Agent 评估与可观测性|Agent 评估与可观测性]](奖励/评估的来源)、[[09 ReAct|ReAct]](rollout 里跑的多步循环)。
