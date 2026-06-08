[[080 后训练总览：SFT 到 RM 到 RLHF]]:预训练只让模型「会接龙」,**后训练(post-training)**才让它「听话、有用、无害」;经典三段管线为 **① SFT 指令微调 → ② 训练奖励模型 RM → ③ 用 RLHF(PPO)优化策略**,由 InstructGPT(Ouyang 2022,arXiv 2203.02155)确立,是今天所有对话模型的标准范式。

## 直觉:预训练给「知识」,后训练给「品行」

[[036 GPT 系列：自回归与规模化|预训练]] 的模型读了半个互联网,知识极广,但目标只是**预测下一个词**。你问它「怎么煮蛋」,它可能续写成「……这道题出自某教材第 3 章」——因为网上这种文本更常见。它**懂很多,却不知道你要它干嘛**。

后训练做三件事,逐级加码:

1. **SFT(对齐格式与意图)**:给它看几万条「人写的好答案」,教它「被问问题时,就该直接好好回答」。这步最便宜、收益最大。
2. **RM(把人的偏好压成一个分数)**:人很难写出「完美答案」,但很容易**比较**两个答案谁更好。于是让人排序,训一个 [[083 奖励模型 RM|奖励模型]] 学会给任意回答打分。
3. **RLHF(让模型去追这个分数)**:用强化学习([[084 策略梯度与 PPO 基础|PPO]])让模型生成的回答**最大化 RM 分数**,同时用 KL 约束别跑太偏。

一句话:**SFT 教「怎么说」,RM 学「哪个好」,RLHF 让模型「自己练到更好」。**

## 例子:同一个 prompt 在三段里的不同角色

prompt:`「写一句鼓励考研失败的人的话」`

- **预训练模型**:可能续成「考研失败的人占比……」(在写百科,不在安慰)。
- **SFT 后**:`「一次没考上不代表你不行,调整方向再来,你已经很努力了。」`——格式对了,会安慰了。
- **RM**:对比 SFT 的 4 个采样回答,人排序「真诚具体 > 套话 > 空洞 > 跑题」,RM 学到「真诚具体」该给高分。
- **RLHF 后**:模型主动往「真诚、具体、不说教」的方向漂移,因为那样 RM 分更高。

三段的**数据形态**完全不同(这是面试爱考的点):

| 阶段 | 数据 | 规模(InstructGPT) | 学习范式 |
|---|---|---|---|
| ① SFT | (prompt, **人写答案**) | ~13k 示范 | 监督学习 |
| ② RM | (prompt, **答案排序**) | ~33k 比较 | 偏好排序(BT) |
| ③ RLHF | **仅 prompt**(答案模型自生成) | ~31k prompt | 强化学习(PPO) |

注意第三段**不再需要人写答案**——模型自己生成,RM 打分当奖励,人力成本骤降。

![[post-three-stage-pipeline.png]]

## 原理:三个目标函数,一脉相承

**① SFT** —— 在「答案」token 上做最大似然(本质是 [[060 训练目标与 loss 实现|loss 实现]] 里的交叉熵,只是加了 mask):

$$
\mathcal{L}_{\text{SFT}}=-\mathbb{E}_{(x,y)\sim D_{\text{demo}}}\Big[\sum_{t\in \text{答案}}\log \pi_\theta(y_t\mid x,y_{<t})\Big]
$$

**② RM** —— 在偏好对上学 Bradley-Terry 打分(见 [[083 奖励模型 RM|RM]]):

$$
\mathcal{L}_{\text{RM}}=-\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]
$$

**③ RLHF** —— 最大化奖励,减去对 SFT 参考策略的 [[31 KL 散度与 JS 散度|KL]] 惩罚(防奖励黑客,详见 [[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF 全流程]]):

$$
\max_\theta\ \mathbb{E}_{x,\,y\sim\pi_\theta}\Big[\underbrace{r_\phi(x,y)}_{\text{RM 奖励}}-\ \beta\,\underbrace{\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{SFT}}(\cdot\mid x)\big)}_{\text{别跑偏}}\Big]
$$

三步是**递进**的:RM 用 SFT 模型初始化,RLHF 的策略也从 SFT 出发、并以 SFT 为「锚」。所以 SFT 质量是整条链的地基。

## 对齐到底对齐什么:3H 与对齐税

后训练的目标常概括为 **3H**(Anthropic 提法):
- **Helpful(有用)**:真的解决用户问题、遵循指令。
- **Honest(诚实)**:不编造(减少幻觉)、不知道就说不知道。
- **Harmless(无害)**:拒绝有害请求、不输出危险内容。

三者**互相打架**:Helpful 和 Harmless 冲突(详细回答危险问题=有用但有害),Honest 和 Helpful 冲突(诚实说"我不确定"不如自信编一个"有用"看起来好)。所以偏好标注必须给**优先级**(见 [[082 偏好数据与标注|偏好数据]])。还有**对齐税(alignment tax)**:对齐后模型在某些纯能力 benchmark 上可能掉点——RLHF 让模型更"听话"但也更"保守/啰嗦"。InstructGPT 用混入预训练梯度(PPO-ptx)缓解。

## 三段的成本与瓶颈对比

| | 算力成本 | 人力成本 | 主要瓶颈 | 显存 |
|---|---|---|---|---|
| SFT | 低(几千~几万样本,1-3 epoch) | 中(写示范贵) | 数据质量与多样性 | 低(可 LoRA) |
| RM | 低(1 epoch,易过拟合) | 中(排序比写答案便宜) | 标注一致性(~72-77% 上限) | 中 |
| RLHF/PPO | **高**(在线采样 + 四模型) | 低(只需 prompt) | 训练不稳、奖励黑客、显存 | **高(actor/critic/RM/ref 四份)** |

要点:**人力成本 SFT>RM>RLHF(RLHF 不用写答案),算力/工程成本反过来 RLHF≫SFT**。这个倒挂正是 DPO 等"省掉 PPO"方法受欢迎的原因。

## 现代变体:经典三段之后的演化

经典 InstructGPT 三段是 2022 年的配方,之后的主流改进(面试高频"现在还用 PPO 吗"):
- **DPO**(见 [[086 DPO 直接偏好优化(推导)|DPO]]):把 RM+PPO 合并成一个直接在偏好对上的监督损失,**去掉显式 RM 和在线采样**,只需 policy+ref 两模型。稳、省显存,2023 后开源主流。
- **拒绝采样 / Best-of-N(RAFT、RFT)**:用 RM 给 SFT 模型的 N 个采样打分,**只把高分回答拿回去再做 SFT**。简单稳定,介于 SFT 和 RLHF 之间;Llama-2 用它做迭代式对齐。
- **RLAIF / 宪法 AI**(见 [[082 偏好数据与标注|偏好数据]]):用强模型(而非人)生成偏好标注,降本可规模化。
- **GRPO**(见 [[088 GRPO 与可验证奖励|GRPO]]):去掉 critic,用一组采样的**组内相对奖励**当优势,省显存,DeepSeek-R1 等推理模型用它配可验证奖励。
- **在线 vs 离线**:PPO/GRPO 是**在线**(边训边采新数据);DPO/拒绝采样多为**离线**(固定偏好集)。在线探索能力强但贵,离线稳省但受限于数据集分布。

## 代码:三段管线的伪代码骨架

```python
# ❌ 误区:以为「微调=直接拿 (问题,答案) 一直训」,跳过 RM/RLHF
# 只做 SFT 也能用,但学不会「偏好层面的好坏」,易啰嗦、爱编造。

# ✅ 标准三段(伪代码)
# --- 阶段 1:SFT ---
sft_model = pretrained.clone()
for x, y in demo_data:                       # (prompt, 人写答案)
    loss = ce_on_answer_only(sft_model, x, y)  # 只对答案 token 算 loss
    loss.backward(); opt.step()

# --- 阶段 2:奖励模型 RM ---
rm = sft_model.clone_with_scalar_head()      # 复用 SFT 骨干,换标量头
for x, y_w, y_l in pref_data:                # (prompt, 好, 坏)
    loss = -log_sigmoid(rm(x, y_w) - rm(x, y_l))  # Bradley-Terry
    loss.backward(); opt.step()

# --- 阶段 3:RLHF(PPO)---
policy, ref = sft_model.clone(), sft_model.frozen()
for x in prompt_only_data:                    # 只有 prompt
    y = policy.generate(x)                    # 模型自己生成回答
    reward = rm(x, y) - beta * kl(policy, ref, x, y)  # RM 分 − KL 惩罚
    ppo_update(policy, critic, x, y, reward)  # 见 084
```

## 面试高频

- **后训练三段分别解决什么?** SFT 解决**格式与意图对齐**(会听话);RM 解决**偏好量化**(把「好」变成可优化的分数);RLHF 解决**超越示范上限**(模型能比人写的示范更好,因为它在优化偏好而非模仿单条答案)。
- **为什么不只做 SFT?** SFT 是**模仿**单条「正确答案」,但好答案不唯一,且人写的示范有上限。RLHF 优化的是**相对偏好**,能探索出比示范更好的回答;还能压低有害/编造内容。
- **为什么需要 RM,不直接让人给 RL 当奖励?** 人没法在 RL 训练里**实时**给百万次 rollout 打分,太慢太贵。RM 是「人类偏好的可微、可批量代理」,一次训好反复用。
- **这套和现在的 DPO 关系?** [[086 DPO 直接偏好优化(推导)|DPO]] 把「RM + RLHF」两步合并成一个直接在偏好对上的监督损失,**省掉显式 RM 和 PPO**,更稳更省显存,是 2023 后的主流替代;但理解经典三段是理解 DPO 的前提。
- **后训练对齐什么?** 3H:Helpful(有用)、Honest(诚实)、Harmless(无害)。三者互相冲突(有用 vs 无害、诚实 vs 有用),标注要给优先级;过度对齐有"对齐税"(纯能力 benchmark 掉点),InstructGPT 用 PPO-ptx 混预训练梯度缓解。
- **现在还用 PPO 吗?** 开源主流已大量转 DPO(去掉 RM+PPO)、拒绝采样/Best-of-N(Llama-2)、GRPO(去 critic,DeepSeek-R1);PPO 仍在线探索强、可验证奖励场景占优。经典三段是理解所有变体的基础。
- **三段的成本瓶颈?** 人力 SFT>RM>RLHF(RLHF 不用写答案);算力/工程 RLHF≫SFT(在线采样+四模型);这个倒挂是 DPO 流行的原因。
- **跨域联系**:这套「示范→偏好→RL」的范式也用于 Agent 训练,见 [[32 Agentic RL 与训练|Agentic RL]](把环境反馈/可验证奖励接到第三段)。

## 关键事实

- Ouyang et al.(OpenAI), *Training language models to follow instructions with human feedback*(InstructGPT),2022,arXiv **2203.02155**:确立 SFT→RM→PPO 三段;**1.3B 的 InstructGPT 输出被人类偏好于 175B 的 GPT-3**,证明对齐 ≫ 单纯放大。
- 数据规模(原文):SFT 约 13k 示范、RM 约 33k 比较、PPO 约 31k prompt。
- RM 用 6B 规模即可(原文用比策略小的 RM);策略用 PPO 优化「RM 奖励 − KL」。
- 3H(Helpful/Honest/Harmless)对齐目标与对齐税出自 Anthropic(Askell et al., 2021, arXiv:2112.00861)与 InstructGPT(PPO-ptx 混预训练梯度缓解掉点)。
- 现代变体:DPO(Rafailov et al., 2023, arXiv:2305.18290)去 RM+PPO;拒绝采样/Best-of-N 用于 Llama-2(Touvron et al., 2023, arXiv:2307.09288);GRPO(Shao et al., 2024, arXiv:2402.03300)去 critic;RLAIF/宪法 AI(Bai et al., 2022, arXiv:2212.08073)用 AI 标注。
- 关联:[[081 指令微调 SFT 与数据构造|SFT]]、[[083 奖励模型 RM|RM]]、[[084 策略梯度与 PPO 基础|PPO]]、[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF 全流程]]、[[086 DPO 直接偏好优化(推导)|DPO]]、[[088 GRPO 与可验证奖励|GRPO]]。
- **最新进展(2025-2026)**:后训练已从「一套 RLHF 走天下」转为**按任务选栈的模块化流程**——SFT 管指令遵循,偏好优化(DPO/SimPO/KTO)管风格/语气对齐,**RLVR(可验证奖励 RL,GRPO/DAPO)** 管数学/代码/工具调用等可客观判定的推理任务;2025 年几乎每个前沿模型(R1、Nemotron、GPT-5 系)都用不同的后训练组合,而非统一三段(post-training 综述,2025-2026)。
