[[088 GRPO 与可验证奖励|GRPO]]:GRPO（Group Relative Policy Optimization，组相对策略优化）是 [[084 策略梯度与 PPO 基础|PPO]] 在大模型推理任务上的一种省显存做法——它不再训练 critic，而是对同一题采样一组答案，用组内相对奖励做更新；当奖励来自可判真的答案检查、单元测试或终态验证时，就进入 [[088 GRPO 与可验证奖励|RLVR]] 的范畴。

## 直觉：不用 critic，不等于“天然无偏”

[[084 策略梯度与 PPO 基础|PPO]] 想降低方差，常见办法是训练一个价值网络 $V_\phi(s)$ 当基线。但大模型 RL 里，critic 本身要占显存、要训稳定，还容易在长文本上估偏。

GRPO 的工程直觉是：同一题一次采 $G$ 个完整答案，直接拿这一组奖励做相对比较，不再额外养 critic。这样做很省事，但有个常见误区要纠正：**“组均值天然是严格无偏 baseline”不对**。对 REINFORCE 来说，严格无偏的 baseline 必须在条件于当前动作时与该动作独立；而 GRPO 的组均值里包含当前样本自己的奖励，属于**低成本近似基线**，不是教科书式的无偏控制变量。若要严格满足这条，应该用 leave-one-out baseline。

一句话：**GRPO 的价值在于“去 critic + 组内对比 + 配合可验证奖励”，不是“它在理论上等价于无偏 REINFORCE”**。

![[post-grpo-group.png]]

## 小数字手算：组均值 baseline 和 leave-one-out baseline 差在哪

题目「$3+5\times2=?$」，正确答案是 13。对同一个 prompt 采 $G=4$ 个答案，得到奖励

$$
r=\{1,0,1,0\}.
$$

先算 **GRPO 常见写法**里的组均值：

$$
\mu=\frac{1+0+1+0}{4}=0.5.
$$

如果用**总体标准差**（不是 PyTorch 默认的样本标准差）：

$$
\sigma=\sqrt{\frac{(1-0.5)^2+(0-0.5)^2+(1-0.5)^2+(0-0.5)^2}{4}}
=\sqrt{0.25}=0.5.
$$

于是标准化优势是

$$
\hat A_i^{\text{GRPO}}=\frac{r_i-\mu}{\sigma}
\Rightarrow
\{+1,-1,+1,-1\}.
$$

再看 **leave-one-out** baseline。第 1 个样本的 baseline 不该包含它自己，所以

$$
b_1^{\text{LOO}}=\frac{0+1+0}{3}=\frac13,\qquad
A_1^{\text{LOO}}=1-\frac13=\frac23.
$$

类似地：

$$
b_2^{\text{LOO}}=\frac{1+1+0}{3}=\frac23,\qquad
A_2^{\text{LOO}}=0-\frac23=-\frac23.
$$

这一组的 leave-one-out 优势就是

$$
\Big\{\frac23,-\frac23,\frac23,-\frac23\Big\}.
$$

你会发现它和“只减组均值、不做标准化”的结果 $\{+0.5,-0.5,+0.5,-0.5\}$ 方向相同，但数值不同。对 $G=4$ 而言：

$$
A_i^{\text{mean}}=r_i-\mu=\frac{G-1}{G}\,A_i^{\text{LOO}}=\frac34 A_i^{\text{LOO}}.
$$

这说明：**组均值基线和 leave-one-out 基线在有限组下只差一个缩放，不代表前者就成了严格无偏控制变量；再叠加标准差归一、PPO clip 与长度归一后，更不能把它说成“精确无偏梯度”**。

## 公式推导：GRPO 为什么好用，以及它的边界

对同一题 $x$，旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个完整答案 $\{y_1,\dots,y_G\}$，每个答案对应一个序列级奖励 $r_i$。

GRPO 的常见组内标准化优势写法是

$$
\hat A_i^{\text{GRPO}}=
\frac{r_i-\mu}{\sigma},
\qquad
\mu=\frac1G\sum_{j=1}^G r_j,\quad
\sigma=\sqrt{\frac1G\sum_{j=1}^G(r_j-\mu)^2}.
$$

如果改成严格 leave-one-out baseline，则

$$
b_i^{\text{LOO}}=\frac{1}{G-1}\sum_{j\neq i} r_j,
\qquad
A_i^{\text{LOO}}=r_i-b_i^{\text{LOO}}.
$$

两者关系可以直接展开：

$$
\mu=\frac1G r_i+\frac{G-1}{G}b_i^{\text{LOO}}
\Rightarrow
r_i-\mu=\frac{G-1}{G}(r_i-b_i^{\text{LOO}}).
$$

也就是

$$
A_i^{\text{mean}}=\frac{G-1}{G}A_i^{\text{LOO}}.
$$

所以在**不做标准差归一**时，组均值版本和 leave-one-out 版本方向一致，只差一个常数缩放；但因为组均值含有当前样本自己的 $r_i$，它不是“动作无关”的 baseline。GRPO 在实践上仍然有效，是因为它把 RL 目标改写成了一个好优化、低内存、与多采样推理任务匹配的 surrogate objective，而不是因为它完美复刻了无偏 REINFORCE。

其 PPO-clip 目标可写成

$$
\mathcal{J}_{\text{GRPO}}
=
\mathbb{E}\Bigg[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}
\sum_t
\min\Big(
\rho_{i,t}\hat A_i,\,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i
\Big)
\Bigg]
-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}],
$$

其中

$$
\rho_{i,t}=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}.
$$

这里再叠加了两层工程近似：一是**同一序列的所有 token 共享同一个序列级优势**，二是**长度归一 $\frac{1}{|y_i|}$**。这正是 Dr.GRPO（2025，COLM）指出偏差来源的地方：标准差归一会引入难度偏差，长度归一会引入长度偏差。

![[post-GRPO显存对比.png]]

![[post-DrGRPO归一化.png]]

## 可验证奖励（RLVR）：真正要防的是“骗过 verifier”

GRPO 在推理任务里常与可验证奖励绑定使用：

- 数学：抽取最终答案，与标准答案精确或符号等价匹配。
- 代码：在干净环境中运行公开测试与隐藏测试，按通过情况计分。
- 终态任务：检查文件、数据库、API 调用、日志或环境终态是否满足规范。

这里也有一个常见误区：**“可验证奖励天然不会被钻空子”不对**。更准确的说法是，它通常比神经 RM 更抗作弊，但仍取决于 verifier 是否足够强。典型失败面有三类：

1. **抽取错误**：答案明明对了，但解析器没抓到最终框或没做 canonicalization。
2. **局部过拟合**：代码只过了公开单测，真实终态仍错。
3. **格式投机**：模型学会迎合某个弱 verifier 的模板，而不是完成任务本身。

因此，2025–2026 年更稳妥的做法不是“只看最后一句”，而是做**terminal-state verifier**：在隔离环境里重放执行，检查最终状态是否满足任务要求，必要时配隐藏测试、幂等检查和副作用审计。对代码 Agent 来说，**“我已经修好了”不是奖励信号，测试通过且终态正确才是**。

## 代码/配置：`torch.std` 默认值就是一个坑

```python
import torch

def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    rewards = rewards.float()

    # ❌ 错误 1：把 PyTorch 默认 std() 当成总体标准差
    # 它默认 unbiased=True，算的是样本标准差；
    # 对 [1,0,1,0] 会得到 0.577...，和手算 0.5 不一致。
    # bad_std = rewards.std()

    # ❌ 错误 2：把“组均值 baseline”说成严格无偏控制变量
    # 组均值包含当前样本自己的 reward，只是工程上常用近似。

    # ✅ 若要复现 GRPO 常见写法，手算对应的是总体标准差
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    if sigma < 1e-6:
        return torch.zeros_like(rewards)
    return (rewards - mu) / (sigma + 1e-6)


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    rewards = rewards.float()
    group_size = rewards.numel()
    loo_baseline = (rewards.sum() - rewards) / (group_size - 1)
    return rewards - loo_baseline


def terminal_state_reward(task, completion):
    # ❌ 只信模型自述，容易被“口头完成”骗过
    # return 1.0 if "fixed" in completion else 0.0

    # ✅ 真正执行 verifier：解析最终答案 / 跑隐藏测试 / 检查终态
    result = run_isolated_verifier(task, completion)
    return float(result.passed)
```

要点：

- 手算若按 $\frac{1}{G}$ 定义方差，代码里就该用 `std(unbiased=False)`。
- 若你要讲“严格无偏 baseline”，应讲 **RLOO / leave-one-out**，不是组均值。
- 若你要讲“可靠奖励”，应讲 **verifier 的终态覆盖面**，不是只说“可验证”三个字。

## 面试高频

- 题库路由：[[LLM 面试题库]]
- **GRPO 和 PPO 的核心区别？** GRPO 去掉 critic，用同题多采样后的组内相对优势替代价值网络；clip 与 KL 约束仍保留。
- **为什么说“组均值天然无偏 baseline”不严谨？** 因为组均值里含当前样本自己的奖励，不满足“baseline 对当前动作独立”的严格条件；严格无偏应使用 leave-one-out baseline。
- **组均值版和 leave-one-out 版到底差多少？** 对 raw reward 而言，$A_i^{\text{mean}}=\frac{G-1}{G}A_i^{\text{LOO}}$，方向相同但有限组下不是同一个估计器；加上标准差归一、clip 后差异进一步扩大。
- **GRPO 为什么在工程上仍然常用？** 它省掉 critic，显存和实现复杂度都更低，而且天然适配“同题多采样 + 序列级奖励”的推理训练。
- **GRPO 的已知偏差是什么？** Dr.GRPO 指出两类：标准差归一带来难度偏差，长度归一带来长度偏差，尤其会让错误回答虚胖变长。
- **什么是 RLVR？** 用可程序化判真的 verifier 给奖励，如数学答案匹配、代码测试、终态检查；它通常比神经 RM 更抗奖励黑客，但 verifier 自己仍可能被钻空子。
- **为什么现在强调 terminal-state verifier？** 因为“答对最后一句”不足以保证任务真正完成；代码和 Agent 任务要看最终环境状态、隐藏测试与副作用约束。
- **GRPO 的优势是 token 级还是序列级？** 常见实现里是序列级：同一条 completion 的所有 token 共享同一个序列级优势。
- **什么时候不用 GRPO？** 当奖励开放、主观、难以程序验证时，往往需要 [[083 奖励模型 RM|RM]]、[[090 RLAIF、宪法 AI 与过程奖励 PRM|RLAIF/PRM]] 或更复杂的混合奖励。

## 关键事实

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024，arXiv **2402.03300**：提出 GRPO，并在数学推理场景中使用组相对优势替代 critic。
- Ahmadian et al., *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*, 2024，arXiv **2402.14740**：RLOO 明确采用 leave-one-out baseline，适合作为“严格无偏控制变量”对照项。
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025，arXiv **2503.20783**，COLM 2025：指出 GRPO 的标准差归一和长度归一会引入优化偏差，并提出 Dr.GRPO 修正。
- DeepSeek-R1（技术报告 2025，arXiv **2501.12948**）的推理 RL 依赖可验证奖励；但“可验证”不代表“不可作弊”，verifier 设计仍是可靠性核心。
- 代码实现里 `torch.std()` 默认 `unbiased=True`，会给出样本标准差；若笔记手算按总体标准差，应显式写 `std(unbiased=False)`。
- 对代码/Agent 任务，2025–2026 更可靠的奖励对象是**终态验证**而不是模型自述；这也是防奖励黑客的关键工程边界。
- 关联：[[084 策略梯度与 PPO 基础|PPO]]、[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF/奖励黑客]]、[[083 奖励模型 RM|RM]]、[[089 推理模型与 RL：o1、R1 的长 CoT 与自我反思|推理模型]]、[[090 RLAIF、宪法 AI 与过程奖励 PRM|PRM]]、[[31 KL 散度与 JS 散度|KL]]、[[32 Agentic RL 与训练|Agentic RL]]
