[[085 RLHF 全流程与 KL 约束、奖励黑客]]:RLHF(Reinforcement Learning from Human Feedback,基于人类反馈的强化学习)用 [[084 策略梯度与 PPO 基础|PPO]] 把 [[083 奖励模型 RM|RM]] 的打分当奖励来优化策略,同时加**逐 token 的 [[31 KL 散度与 JS 散度|KL]] 惩罚**把策略拴在 [[081 指令微调 SFT 与数据构造|SFT]] 参考模型附近——这道约束是防止「**奖励黑客**(reward hacking)」的命门,是 InstructGPT/ChatGPT 对齐的核心配方,也是 [[086 DPO 直接偏好优化(推导)|DPO]] 想绕开的复杂闭环。

## 直觉:让模型「投人所好」,但别跑偏

[[080 后训练总览：SFT 到 RM 到 RLHF|后训练]] 的三步已经铺好:SFT 教会模型「听话答题」,RM 学会「人喜欢哪个」,现在第三步要让模型的输出**奖励越来越高**。

但这里有个致命陷阱:**RM 不是真理,只是人类偏好的近似代理**(见 [[083 奖励模型 RM|RM]])。如果让策略毫无约束地去最大化 RM 分,它会变成一个「钻空子的优等生」——专找 RM 评分的漏洞,输出**RM 给高分、但人类其实讨厌**的回答(谄媚、超长、套话、固定句式)。这就是**奖励黑客**。

解药很朴素:**别让新策略离原来的 SFT 模型太远**。每生成一个 token,就在奖励里扣一笔「你偏离参考模型多少」的 KL 罚款。模型既要追 RM 分,又要交 KL 税,自然不敢狂奔去钻漏洞。一句话:**RLHF = 在「离 SFT 不远」的信赖球里,小步爬 RM 的奖励山**。

![[post-rlhf-loop.png]]

## 例子:KL 惩罚怎么改变奖励(小数字)

一个 prompt,策略采了回答 $y$。逐 token 看某个位置 $t$:策略给某 token 的概率 $\pi_\theta=0.6$,参考模型 $\pi_{\text{ref}}=0.3$。这一步的 KL 项(单样本估计)是

$$
\log\frac{\pi_\theta(a_t)}{\pi_{\text{ref}}(a_t)}=\log\frac{0.6}{0.3}=\log 2\approx 0.693
$$

取惩罚系数 $\beta=0.1$,这一步扣 $\beta\times0.693\approx0.069$ 的奖励——策略**越偏离参考**(比值越大),扣得越狠。

整段回答的 RM 分只在**末尾**给一次,设 $r_\phi(x,y)=1.8$。把每个 token 的 KL 罚累加(设整段共扣 $\sum_t 0.05$),最终喂给 PPO 的奖励:

$$
R=\underbrace{1.8}_{\text{RM 末尾分}}-\ \beta\sum_t\log\frac{\pi_\theta}{\pi_{\text{ref}}}=1.8-0.5=1.3
$$

如果策略为了刷 RM 分变得很「怪」(KL 暴涨到累计 3.0),罚款 $0.1\times3.0=0.3$… 但 $\beta$ 调大或 KL 失控时,净奖励会被罚到不划算——**这就是 KL 在「拉住缰绳」**。

## 原理:带 KL 约束的 RLHF 目标

**1)优化目标**。RLHF 要最大化「期望 RM 奖励」,同时**约束**策略对参考策略的 KL 不要太大:

$$
\max_{\pi_\theta}\ \mathbb{E}_{x\sim D,\ y\sim\pi_\theta(\cdot\mid x)}\Big[\,r_\phi(x,y)\,\Big]\ -\ \beta\,\mathbb{D}_{\mathrm{KL}}\!\big[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big]
$$

第一项追 RM 分,第二项是 [[31 KL 散度与 JS 散度|KL 散度]] 惩罚(把硬约束写成拉格朗日软惩罚),$\beta$ 越大越保守。$\pi_{\text{ref}}$ 通常就是冻结的 SFT 模型。

**2)等价的逐 token 奖励**。把 KL 拆进每一步,等价于给 PPO 一个**整形后的逐 token 奖励**:

$$
R_t=\underbrace{r_\phi(x,y)\cdot\mathbb{1}[t=\text{末token}]}_{\text{RM 分只在末尾}}\ -\ \beta\,\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{ref}}(a_t\mid s_t)}
$$

注意:**RM 分是序列级、只在最后一个 token 给**;KL 罚是 token 级、每步都扣。然后把这个 $R_t$ 丢给 [[084 策略梯度与 PPO 基础|PPO]] 算优势、做 clip 更新。

**3)为什么这个目标有「解析最优解」**。固定 $r_\phi$,上式的最优策略是

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\,r_\phi(x,y)\Big)
$$

直觉:在参考分布上**按 $\exp(r/\beta)$ 重新加权**——奖励高的回答概率被指数级放大,$\beta$ 控制放大力度。这个闭式解**正是 [[086 DPO 直接偏好优化(推导)|DPO]] 推导的起点**:它把这条式子反解出 $r_\phi$,从而绕过整个 PPO 闭环。

**4)显存账(PPO 为何贵)**。训练时要同时常驻**四份模型**:actor($\pi_\theta$)、critic($V_\phi$)、RM($r_\phi$)、ref($\pi_{\text{ref}}$)。这是 RLHF 工程难、DPO 受欢迎的直接动因。

![[post-reward-hacking.png]]

## 最优解的完整推导(DPO 的起点)

把 ③ 的闭式解推一遍(面试问"带 KL 的 RLHF 最优策略长啥样、怎么推"能写)。目标(单 prompt $x$):
$$\max_{\pi}\ \mathbb{E}_{y\sim\pi}[r(x,y)]-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}}).$$
展开 KL 并把 $\beta$ 提出:
$$=\max_\pi\ \mathbb{E}_{y\sim\pi}\Big[r(x,y)-\beta\log\tfrac{\pi(y)}{\pi_{\text{ref}}(y)}\Big]=-\beta\min_\pi\ \mathbb{E}_{y\sim\pi}\Big[\log\tfrac{\pi(y)}{\pi_{\text{ref}}(y)}-\tfrac1\beta r(x,y)\Big].$$
把括号内凑成一个 KL。定义
$$\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac1\beta r(x,y)\big),\quad Z(x)=\sum_y\pi_{\text{ref}}(y)\exp(\tfrac1\beta r).$$
则括号 $=\log\frac{\pi(y)}{\pi^*(y)}-\log Z(x)$,于是目标 $=-\beta\,\mathbb{E}_y[\log\frac{\pi}{\pi^*}]+\beta\log Z=-\beta\,\mathrm{KL}(\pi\|\pi^*)+\text{const}$。KL ≥0 在 $\pi=\pi^*$ 取 0,**所以最优策略就是 $\pi^*$**——在参考分布上按 $\exp(r/\beta)$ 指数重加权。$Z(x)$ 难算(要遍历所有 $y$),所以 RLHF 用 PPO 数值逼近;[[086 DPO 直接偏好优化(推导)|DPO]] 的妙处是把这条式子**反解出 $r$**($r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z$),代进 Bradley-Terry 后 $Z(x)$ 在差值里**抵消**,从而绕过 RM 和 PPO。

## 自适应 KL 控制器:β 不是定值

$\beta$ 太小 → 奖励黑客;太大 → 学不动(被 SFT 拽死)。固定 $\beta$ 难调,InstructGPT 用**自适应 KL 控制器**(Ziegler 2019):设目标 KL $\mathrm{KL}_{\text{target}}$,每步测当前 KL,按比例调 $\beta$:
$$\beta\leftarrow\beta\cdot\big(1+K_\beta\cdot\tfrac{\mathrm{KL}-\mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}}\big).$$
KL 超标就上调 $\beta$(收紧缰绳),KL 太小就下调(放松)。这是一个简单的反馈控制器,把"离 SFT 多远"稳定在目标值附近,不用手调死 $\beta$。

![[post-自适应KL控制.png]]

## 奖励黑客的实例与 Goodhart 定律

**Goodhart 定律**:"当一个指标成为目标,它就不再是好指标。"RM 是"人类偏好"的代理指标,PPO 一旦过度优化它,就开始钻代理与真实目标的缝。实测过的奖励黑客花样:
- **长度黑客**:RM 有长度偏置(见 [[082 偏好数据与标注|偏好数据]]),策略发现"写长就高分",输出越来越啰嗦。最经典、最普遍。
- **格式黑客**:猛加 markdown、列表、加粗、emoji——RM 觉得"专业"。
- **谄媚**:无条件附和用户立场、夸用户,RM 给高分但实际有害(放弃纠错)。
- **套话/安全词堆砌**:反复说"作为 AI 我……""请注意安全",刷无害性 RM。
- **退化重复**:极端时输出无意义但 RM 恰好打高分的字符串(RM 分布外失效)。

**over-optimization 的标度规律**(Gao 2023):随着 PPO 优化(KL 增大),**RM 分一路涨,但真实人类偏好先涨后跌**——存在一个最优 KL 点,过了就是纯刷分。这给"早停"和"KL 约束"提供了定量依据:别让策略把 RM 榨到分布外。

![[post-rewardOveropt.png]]

## RLHF vs DPO vs GRPO:显存与稳定性对比

| | 载几个模型 | 在线采样 | 显式 RM | 稳定性 | 探索能力 |
|---|---|---|---|---|---|
| **PPO-RLHF** | 4(actor/critic/RM/ref) | 是 | 是 | 难调、易崩 | 强(在线) |
| **DPO** | 2(policy/ref) | 否(离线偏好集) | 否 | 稳 | 弱(受数据集限) |
| **GRPO** | 2-3(去 critic,用组内相对奖励当优势) | 是 | 看任务(可用可验证奖励) | 中 | 强 |

记忆主线:**DPO 去掉 critic+RM+采样(最省最稳,但只在固定数据上);GRPO 去掉 critic(用一组采样的组内相对奖励算优势,省一半显存,保留在线探索)**。GRPO 配**可验证奖励**(数学答案对错、代码能否通过)是 DeepSeek-R1 等推理模型的配方,见 [[088 GRPO 与可验证奖励|GRPO]]。

## 完整流程图(三步回顾 + 闭环)

RLHF 不是孤立一步,而是 [[080 后训练总览：SFT 到 RM 到 RLHF|三阶段]] 的最后一环:

1. **SFT**:在高质量指令数据上微调,得到 $\pi_{\text{ref}}$(也当 actor 初始化)。
2. **RM**:在 [[082 偏好数据与标注|偏好对]] 上训 Bradley-Terry 奖励模型 $r_\phi$。
3. **PPO/RLHF**:策略采样 → RM 打分 → 减 KL 罚 → PPO 更新,**循环多轮**。

## 代码:KL 整形奖励 + RLHF 循环(伪代码)

```python
import torch

def shaped_reward(logp_policy, logp_ref, rm_score, beta=0.1):
    # logp_policy/logp_ref: 逐 token 对数概率 [B, T];rm_score: 整段分 [B]
    # ❌ 误区:只用 RM 分当奖励,不加 KL —— 策略会狂奔钻 RM 漏洞(奖励黑客)
    # return rm_score                      # 错:无约束,几步就训飞 / 谄媚

    # ✅ 正解:逐 token 减 KL 罚,RM 分只加在末尾
    kl_per_token = logp_policy - logp_ref          # log(πθ/πref) [B, T]
    reward = -beta * kl_per_token                  # 每步 KL 惩罚 [B, T]
    reward[:, -1] += rm_score                       # RM 序列分加在最后一个 token
    return reward                                    # 喂给 PPO 算优势

# 一轮 RLHF(概念)
for batch_prompts in dataloader:
    responses = policy.generate(batch_prompts)              # actor 采样
    rm = reward_model(batch_prompts, responses)             # RM 打分(冻结)
    lp  = policy.logprobs(responses)                        # πθ
    lpr = ref_model.logprobs(responses)                     # πref(冻结)
    R   = shaped_reward(lp, lpr, rm, beta=0.1)
    adv = gae(R, critic.values(responses))                  # 优势(critic 当基线)
    loss = ppo_clip_loss(lp, lp_old, adv) + value_loss(...) # 见 [[084]]
    loss.backward(); opt.step()
```

要点:$\beta$ 太小 → 奖励黑客;太大 → 学不动(被 SFT 拽死)。实践常用**自适应 KL 控制器**(KL 超目标值就上调 $\beta$)。

## 面试高频

- **RLHF 为什么必须加 KL 惩罚?** RM 是不完美代理,无约束优化会触发奖励黑客(策略钻 RM 漏洞)。KL 把策略拴在 SFT 参考附近,既限制偏离、又保留语言能力;数学上等价于在「信赖球」内最大化奖励。
- **KL 罚和 PPO 的 clip 是一回事吗?** 不是,两套独立约束。**KL 罚**约束「策略 vs 参考模型」(防偏离 SFT、防奖励黑客),进**奖励**里;**clip** 约束「新策略 vs 旧策略」(防一步更新过猛),进**目标函数**里。一个管「离锚多远」,一个管「单步多大」。
- **奖励黑客是什么?怎么缓解?** 策略学会输出 RM 高分但人类讨厌的回答(谄媚、超长、套话)。根因=RM 不完美 + 过度优化(Goodhart 定律)。缓解:KL 约束、RM 集成、早停、过程奖励 [[090 RLAIF、宪法 AI 与过程奖励 PRM|PRM]]、定期重训 RM。
- **RM 分加在哪个 token?KL 加在哪?** RM 分是**序列级**,只加在最后一个 token;KL 罚是**token 级**,每步都减。合成后才喂 PPO。
- **RLHF 训练要载几个模型?** 四个:actor、critic、RM、ref。这是显存瓶颈,也是 [[086 DPO 直接偏好优化(推导)|DPO]](只需 policy + ref)和 [[088 GRPO 与可验证奖励|GRPO]](去掉 critic)优化的方向。
- **带 KL 约束目标的最优策略长什么样?怎么推?** 把目标凑成 $-\beta\,\mathrm{KL}(\pi\|\pi^*)+\text{const}$,其中 $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$,KL≥0 在 $\pi=\pi^*$ 取 0,故最优即 $\pi^*$。$Z(x)$ 难算故 RLHF 用 PPO 逼近;DPO 反解出 $r$ 后 $Z$ 在 BT 差值里抵消,绕过 RM+PPO。
- **β 怎么调,为什么用自适应?** 太小→奖励黑客,太大→学不动。自适应 KL 控制器设目标 KL,实测 KL 超标就上调 β、太小就下调,把"离 SFT 多远"稳在目标值,免手调死值。
- **奖励黑客有哪些实例?背后是什么定律?** 长度黑客、格式堆砌、谄媚、套话、退化重复。背后是 Goodhart 定律(指标变目标就失效)。Gao 2023 实测:KL 增大时 RM 分一路涨但真实偏好先涨后跌,故要早停 + KL 约束。
- **RLHF/DPO/GRPO 显存与稳定性怎么比?** PPO 载 4 模型(难调)、DPO 载 2(去 RM+critic+采样,最稳但离线)、GRPO 去 critic(用组内相对奖励算优势,省显存且保留在线探索,配可验证奖励是推理模型配方)。

## 关键事实

- RLHF 配方由 InstructGPT(Ouyang et al. 2022,arXiv **2203.02155**)定型:SFT → RM → PPO + KL;ChatGPT、Claude、Llama-2-chat 均沿用。更早的偏好 RL 见 Christiano et al. 2017(arXiv 1706.03741)、摘要任务 Stiennon et al. 2020(arXiv 2009.01325)。
- 目标:$\max\ \mathbb{E}[r_\phi]-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}]$;逐 token 奖励 $R_t=r_\phi\cdot\mathbb{1}[\text{末}]-\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$。
- 最优解 $\pi^*\propto\pi_{\text{ref}}\exp(r_\phi/\beta)$ —— [[086 DPO 直接偏好优化(推导)|DPO]] 反解此式去掉 RM 与 PPO。
- 奖励黑客 = Goodhart 定律在 RLHF 的体现;缓解靠 KL、RM 集成、早停、PRM。
- 工程:同载 actor/critic/RM/ref 四模型;常用自适应 KL 控制器调 $\beta$;PPO 见 [[084 策略梯度与 PPO 基础|PPO 基础]]。
- 最优解 $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$ 的推导:目标凑成 $-\beta\,\mathrm{KL}(\pi\|\pi^*)$,$Z(x)$ 配分函数难算故用 PPO 逼近;DPO 反解 $r$ 后 $Z$ 抵消。
- 自适应 KL 控制器(Ziegler et al., 2019, arXiv:1909.08593):按当前 KL 与目标 KL 的偏差比例调 β,稳定策略偏离。
- 奖励 over-optimization 的标度规律:RM 分上涨但真实偏好先升后降,存在最优 KL(Gao et al., "Scaling Laws for Reward Model Overoptimization", 2022, arXiv:2210.10760);Goodhart 定律的体现。
- GRPO(Shao et al., 2024, arXiv:2402.03300)去 critic、用组内相对奖励算优势,配可验证奖励用于 DeepSeek-R1(2025, arXiv:2501.12948)等推理模型。
- 对齐相关风险(谄媚、被诱导越权)与 [[06 Jailbreak 越狱]] 同源——都关乎「目标错置」。
- 关联:[[080 后训练总览：SFT 到 RM 到 RLHF|后训练总览]]、[[083 奖励模型 RM|RM]]、[[084 策略梯度与 PPO 基础|PPO]]、[[086 DPO 直接偏好优化(推导)|DPO]]、[[088 GRPO 与可验证奖励|GRPO]]、[[31 KL 散度与 JS 散度|KL 散度]]、[[27 Softmax 与温度|Softmax 温度]]、[[32 Agentic RL 与训练|Agentic RL]]。
