[[32 Agentic RL 与训练|Agentic RL：从 RLVR 到 GRPO]]：让模型在环境中采样完整轨迹、根据奖励更新权重；RLVR 提供可执行的奖励来源，GRPO 是不用单独 critic 的一类策略优化方法，二者不是同义词。

## 直觉：驾驶练习场而不是改导航口令

`[[31 Agent 提示词优化(DSPy)|DSPy]]` 是不动发动机、换导航指令；Agentic RL 是在练习场中反复驾驶，让“看到状态后采取什么动作”的策略本身改变。一次 rollout 可以是一段 `[[09 ReAct|ReAct]]` 轨迹：模型调用工具、读取观察、继续行动，最后由环境给分。奖励越接近真实任务，训练越可能学到有用策略；验证器有漏洞时，模型也可能学会利用漏洞。

RLVR（reinforcement learning with verifiable rewards）是奖励由测试、符号校验器或受控执行环境自动给出的训练设定。它不是“绝不会被钻”的保证，也不要求只能给 0/1 分。DeepSeekMath 在 2024 年把 GRPO 描述为 PPO 的变体，并强调以组相对比较减少 critic 的内存负担。[DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)

同一批可验证任务，先把三条路线放在一起比较，才不会把“有验证器”误当成“必须上 RL”：

| 路线 | 如何使用通过/失败信号 | 改变什么 | 主要边界 |
|---|---|---|---|
| prompt / `[[31 Agent 提示词优化(DSPy)\|DSPy]]` | metric 选择指令、示例或流程 | 不改权重 | 仍受底座策略能力限制 |
| 拒绝采样 + SFT | 多采样，只保留验证通过的轨迹做监督学习 | 增加正例的似然 | 失败轨迹只被丢弃，未直接形成负向策略梯度 |
| RLVR + GRPO | 同题采样组由验证器给回报，再用组相对优势更新 | 改权重 | 需要多条 rollout、稳健验证器与训练控制 |

拒绝采样可作为冷启动/基线：若 $\mathcal S=\{\tau_i\mid v(\tau_i)=1\}$，它最小化 $-\sum_{\tau\in\mathcal S}\log\pi_\theta(\tau)$；GRPO 则同时利用相对正、负优势。不要把提示优化相对 RL 的 sample-efficiency 写成无来源的“100 倍”：GEPA 论文只在其六个任务和比较设置中报告，相对 GRPO 最多 $35\times$ 少 rollout，且相对 GRPO 的平均/最高质量差异为 $+6\%$/$+20\%$；具体路线是否更好仍是实验问题。[DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)；[GEPA, 2025；修订于 2026-02](https://arxiv.org/abs/2507.19457)

## 小数字手算：一组成功/失败轨迹如何变成优势

同一题采样 $G=8$ 条轨迹，受控测试给出 $R=[1,1,0,1,0,0,1,0]$。组内有 4 条通过，故

$$
\mu=\frac{4}{8}=0.5,\qquad
\sigma=\sqrt{\frac{1}{8}\sum_{i=1}^{8}(R_i-0.5)^2}=0.5.
$$

以 $\hat A_i=(R_i-\mu)/(\sigma+\varepsilon)$ 归一化，$\varepsilon$ 很小，则通过轨迹为 $+1$，失败轨迹为 $-1$：

$$
\hat A=[+1,+1,-1,+1,-1,-1,+1,-1].
$$

组内全对或全错时 $\sigma=0$，按上式没有区分信息；实践中必须显式处理这种组，不能把除零或随机噪声误当学习信号。

## 公式推导：从可验证回报到 GRPO 更新

对输入 $x$，当前策略 $\pi_{\theta_{old}}$ 采样 $G$ 条轨迹 $\tau_i$，验证器给 $R_i=v(\tau_i)$。用上面的组相对优势近似“这条轨迹相对同题其他尝试是否更好”。一种常见的裁剪目标可写为

$$
\max_\theta\;\mathbb E\!\left[\frac{1}{G}\sum_{i=1}^{G}
\min\!\left(\rho_i(\theta)\hat A_i,
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)\hat A_i\right)
-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})\right],
$$

其中 $\rho_i(\theta)=\pi_\theta(\tau_i|x)/\pi_{\theta_{old}}(\tau_i|x)$。第一项提高高优势轨迹的概率，裁剪限制单步变化；$\beta$ 与 KL 项是否使用、以及优势/长度处理，随具体 GRPO 变体与实现而异。GRPO 的关键不是“奖励来自 RLVR”，而是用同题组内回报构造优势、避免训练独立 critic。[DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)

DeepSeek-R1 论文报告大规模 RL 能诱导推理行为，但该结果属于其模型、数据、奖励和训练配方，不能推出任何任务只要接 GRPO 就会提升。[DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948)

## 图：奖励来源与策略更新是两条轴

![[Agentic RL 与训练.png]]

![[Agentic RL 与训练-奖励范式演进.png]]

图中的“结果奖励/过程奖励”描述的是何时给分；“人类、AI、验证器”描述的是谁/什么产生分数。二者可组合，不能画成单一必经历史路线。

## 可运行代码：❌ 生搬回报 vs ✅ 组内标准化并处理退化组

```python
# Python 3 标准库；演示 GRPO 的“组相对优势”计算，不执行模型训练。
from math import sqrt

def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = sqrt(variance)
    if std == 0.0:                 # 全对/全错：没有相对排序信息
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]

rewards = [1, 1, 0, 1, 0, 0, 1, 0]

# ❌ 直接把 0/1 当“优势”，忽略同题难度和组内相对比较。
raw_advantages = rewards

# ✅ 同一 prompt 的同组轨迹减均值、除标准差；无需单独 critic 估基线。
advantages = group_advantages(rewards)
assert raw_advantages == [1, 1, 0, 1, 0, 0, 1, 0]
assert advantages == [1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
assert group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]
print("advantages=", advantages)
```

要把这段计算接到真实训练，仍需隔离 rollout 环境、版本化验证器、记录轨迹/奖励/KL/长度，并保留未参与训练的任务集做评估。

## 面试高频

**RLVR 与 GRPO 分别解决什么？** RLVR 是“如何得到奖励”的设定：由可执行验证器打分；GRPO 是“如何利用一组回报更新策略”的方法。可以组合，但不应互相替代。

**GRPO 为什么能不训练 critic？** 它以同一输入采样组的均值与方差构造相对优势；代价是每题需要多条 rollout，且全同回报组没有相对信号。[DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)

**何时不该上 Agentic RL？** 没有稳健验证器、没有隔离环境/回滚 checkpoint、或 prompt 与 `[[20 上下文工程|上下文工程]]` 仍有明显空间时。先用评估证明瓶颈，再决定是否为改权重付出 rollout 与训练成本。

## 关键事实

- GRPO 首见于 DeepSeekMath（2024），论文将其定位为 PPO 变体；论文结果应按其数学任务、模型和资源条件理解。[DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1（2025）是“RL 可形成推理行为”的一个具体训练报告，不是通用的零样本配方或所有 Agent 环境的效果保证。[DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948)
- 安全与质量责任不由算法接管：验证器要防奖励投机，工具要在沙箱和最小权限下运行，高风险动作要接 `[[24 沙箱、最小权限与人审闸门|人审闸门]]`；训练后还需对未见任务回归评估。
