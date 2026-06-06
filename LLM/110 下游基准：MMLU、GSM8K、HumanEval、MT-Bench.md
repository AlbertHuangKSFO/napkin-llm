[[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench]]:**四个最常被引用的下游基准,各测一种能力——MMLU(知识多选)、GSM8K(数学推理)、HumanEval(代码 pass@k)、MT-Bench(开放对话,LLM 当裁判),配合 few-shot 提示来量化模型的实际本事**。

## 直觉

[[109 语言模型评估：困惑度与 bits-per-byte|困惑度]] 只告诉你「模型对文本拟合得多好」,但拟合好 ≠ 会答题、会推理、会写代码。**下游基准(downstream benchmark)** 换个问法:给模型一堆有标准答案的真实任务,看它答对多少。

四个基准刻意覆盖不同维度,合起来才是「全貌」——任何单一基准都偏:

- **MMLU**:57 个学科的多选题,测「世界知识 + 跨领域问题求解」。
- **GSM8K**:小学数学应用题,测「多步算术推理」。
- **HumanEval**:Python 编程题,跑单元测试,测「能不能写出真能跑的代码」。
- **MT-Bench**:开放式多轮对话,没标准答案,用强模型(GPT-4)当裁判打分,测「对话质量与指令遵从」。

前三个有客观答案、可自动评分;MT-Bench 是开放式,得靠 [[112 人评、LLM-as-judge 与 Arena|人评或 LLM-as-judge]]。

**这四个是「经典款」,但已部分饱和——要知道新一代。** 前沿模型把 MMLU、GSM8K、HumanEval 都刷到 85–95%(天花板效应 + 数据污染),区分度下降,于是 2024 起出现更难、更抗污染的接班人:
- **MMLU-Pro**:MMLU 的加难版,选项从 4 个扩到 **10 个**(随机基线降到 10%)、剔除噪声题、加更多推理题;比 MMLU 难,准确率掉 16–33%,且对 prompt 敏感度更低。
- **GPQA(Google-Proof QA)**:研究生级物理/化学/生物难题,**PhD 专家约 65%、能上网的非专家仅约 34%**——刻意做到「光搜索搜不出来,得真懂」。
- **MATH / AIME**:竞赛数学(MATH 5 个难度级;AIME 是高中数学竞赛),测深层多步推理,是推理模型(o 系、R 系)的主战场。
- **SWE-bench**:给真实 GitHub issue,模型要提交能通过测试的补丁——测「能不能改真代码库」,比 HumanEval 的单函数难得多,是 Agent/编程能力的硬指标。
- **LiveCodeBench / LiveBench**:**只收模型训练截止日之后**的新题(竞赛、新闻),**天然抗数据污染**(见 [[111 评估陷阱：数据污染与格式敏感|数据污染]])。

记法:**老四件套测基础能力但易饱和/被污染,新一代(MMLU-Pro/GPQA/MATH/SWE-bench/Live*)更难更抗污染**——面试问「现在用什么基准」别只答 MMLU。

![[eval-基准雷达图.svg]]

## 例子

**MMLU(小数字)。** 4 选 1,随机基线 = 25%。论文里早期小模型几乎就是随机(约 25%),最大的 GPT-3 才比随机高约 20 个百分点(约 43.9%),距离专家水平仍远。今天前沿模型已到 85%+。报告时通常是 **5-shot**(prompt 里先给 5 道带答案的范例)。

**GSM8K(小数字)。** 题如:「小明有 5 个苹果,买了 3 袋每袋 4 个,共几个?」答案 $5+3\times4=17$。每题需 2–8 步基本四则运算。关键发现:**加思维链(让模型一步步写推理)后准确率大涨**;Cobbe 2021 还提出训练一个 verifier 给多个候选解打分、选最高者,显著超过直接取第一个解。

**HumanEval(小数字)。** 164 道题,给函数签名 + docstring,模型补全函数体,跑单元测试判对错。Codex 原论文:**pass@1 = 28.8%**(一次采样就对的比例),**pass@100 = 70.2%**(采 100 个候选只要有 1 个对就算过)。GPT-3 当时是 0%。

**MT-Bench(小数字)。** 80 道两轮开放题(写作、角色扮演、推理、数学、代码等 8 类),GPT-4 裁判按 1–10 给每个回答打分,取平均。强模型一致率与人类偏好 > 80%,所以能当人评的廉价近似。

**pass@k 再算一组(吃透公式)。** 设某题采 $n=10$ 个候选,其中 $c=3$ 个通过测试:
- $\text{pass@1}=1-\frac{\binom{10-3}{1}}{\binom{10}{1}}=1-\frac{7}{10}=0.30$ —— 随机抽 1 个,30% 概率对(= $c/n$,符合直觉)。
- $\text{pass@5}=1-\frac{\binom{7}{5}}{\binom{10}{5}}=1-\frac{21}{252}\approx 0.917$ —— 抽 5 个里至少 1 对,91.7%。
- $\text{pass@10}=1-\frac{\binom{7}{10}}{\binom{10}{10}}$,$\binom{7}{10}=0$(错解不够 10 个)$\to 1.0$ —— 把 10 个全采了必含正确解。

关键直觉:**$k$ 越大 pass@k 越高**(更多机会碰到对的),但**产品体验更接近 pass@1**(用户通常只看一次结果)。pass@1 与 pass@100 差距大 = 模型「有解但不稳」,这正是采样 + verifier/重排(见 GSM8K 的 verifier)能涨分的地方。

![[eval-fewshot与passatk.svg]]

## 原理

**few-shot 是「上下文学习」,不是训练。** $k$-shot 指 prompt 里塞 $k$ 个「问题+答案」范例,再接真问题,模型靠续写给出答案。权重一字不改:

$$\text{prompt}=\underbrace{(x_1,y_1),\dots,(x_k,y_k)}_{k\text{ 个范例}},\ x_{\text{test}}\ \longrightarrow\ \text{模型续写 } \hat y$$

$0$-shot 最难(只给问题),$5$-shot 是 MMLU 惯例。**报分必须注明 shot 数**——shot 数不同分数不同,这也是评估陷阱之一(见 [[111 评估陷阱：数据污染与格式敏感|评估陷阱]])。

**MMLU 怎么打分(多选)。** 不是让模型自由生成,而是看它对四个选项字母 A/B/C/D 的条件概率,取最大者:

$$\hat a=\arg\max_{a\in\{A,B,C,D\}} p(a\mid \text{prompt}),\qquad \text{准确率}=\frac{\#\{\hat a=a^\star\}}{\#\text{题}}$$

**HumanEval 的 pass@k 无偏估计(重点公式)。** 直接「采 $k$ 个看有没有对的」方差太大。改成:每题采 $n\ge k$ 个($n=200$),数出 $c$ 个通过测试的,用组合数算「随机抽 $k$ 个里至少 1 个对」的期望:

$$\boxed{\ \text{pass@}k=\mathbb{E}_{\text{题}}\Big[\,1-\dfrac{\binom{n-c}{k}}{\binom{n}{k}}\,\Big]\ }$$

含义:$\binom{n-c}{k}/\binom{n}{k}$ 是「抽的 $k$ 个全是错解」的概率,$1-$ 它就是「至少 1 个对」。当 $c=0$ 该题 pass@k = 0;$c\ge n-k+1$ 时必中 = 1。

**为什么不直接 $1-(1-\hat p)^k$(常见错法)。** 有人用「单次成功率 $\hat p=c/n$,则 $k$ 次至少一对 = $1-(1-\hat p)^k$」。这是**有偏的**:它假设 $k$ 次独立同分布且 $\hat p$ 已知,但 $\hat p$ 本身是估计、且采样是**无放回**地从 $n$ 个里看。组合数版 $1-\binom{n-c}{k}/\binom{n}{k}$ 是从 $n$ 个采样里精确算「抽 $k$ 个的超几何概率」,无偏且方差小。Chen 2021 特意强调这点——这是面试陷阱题。

**自动判分的边界(代码题特有的坑)。** HumanEval 跑单元测试判对错,但:① 单元测试**覆盖不全**——能过测试 ≠ 真正确(可能恰好绕过未测的边界);② 跑代码有**安全风险**(模型生成的代码要沙箱执行,否则可能删文件、联网);③ 「功能对但风格差/超时」怎么算?所以代码基准要配沙箱 + 超时 + 完整测试集。这也是 SWE-bench 比 HumanEval 难评的原因——真实仓库的测试更复杂。

**MT-Bench 用 LLM-as-judge。** 没有 $y^\star$,改成裁判模型 $J$ 给回答打分 $s=J(\text{prompt},\text{answer})\in[1,10]$,取多题平均。裁判有位置/冗长/自我增强等偏置,需去偏(详见 [[112 人评、LLM-as-judge 与 Arena|112]])。

![[eval-基准雷达图.svg]]

## 代码

```python
from math import comb

# ❌ 错:用一次采样的命中率近似 pass@k —— 高方差、且 k 变了得重采
def pass_at_k_bad(samples_correct, k):
    # samples_correct: 长度 k 的 0/1 列表;只采了 k 个,样本少时极不稳
    return 1.0 if any(samples_correct[:k]) else 0.0

# ✅ 对:采 n≥k 个、数出 c 个正确,用无偏估计(Chen et al. 2021）
def pass_at_k(n, c, k):
    # n: 每题总采样数; c: 其中通过单元测试的个数; k: 评估的 k
    if n - c < k:           # 错解不足 k 个 → 抽 k 个必含正确解
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

print(pass_at_k(n=200, c=10, k=1))    # ≈ 0.05   一次就对的概率
print(pass_at_k(n=200, c=10, k=100))  # ≈ 0.99   采 100 个挑一个
```

```python
# MMLU 多选:取选项字母概率最大者(伪代码,model 返回各 token 的对数概率)
def mmlu_predict(model, prompt, choices=("A", "B", "C", "D")):
    logps = {c: model.score(prompt + f"\n答案:{c}") for c in choices}
    return max(logps, key=logps.get)        # ✅ 比较 A/B/C/D 的概率,不让模型自由生成

# MT-Bench:把两答案交给裁判模型打分(开放式,无标准答案)
def mt_bench_score(judge, question, answer):
    rubric = "请对回答按有用性/正确性给 1–10 分,只输出数字。"
    return int(judge.generate(f"{rubric}\n问题:{question}\n回答:{answer}\n分数:"))
```

## 面试高频

- **「few-shot 会更新参数吗?」** 不会。它把范例放进上下文做「in-context learning」,权重不变;shot 数和范例顺序都会影响分数,所以报分要写清 $k$。
- **「pass@k 为什么要无偏估计?」** 直接采 $k$ 个算命中率方差大;改成采 $n\gg k$、数出 $c$ 个正确,用 $1-\binom{n-c}{k}/\binom{n}{k}$ 解析地算期望,稳定且 $k$ 可任意取。
- **「pass@1 和 pass@100 差很多说明什么?」** 说明模型「有解但不稳定」——多采样能碰到对的,但单次成功率低。真实产品体验更接近 pass@1。
- **「MMLU 是生成题吗?」** 不是,是多选;标准评法是比较四个选项的条件概率取最大,而非让模型自由作答(避免答案抽取噪声)。
- **「为什么要四个基准一起看?」** 各测一维:知识(MMLU)、数学推理(GSM8K)、代码(HumanEval)、开放对话(MT-Bench)。单看一个会偏,且单一公开基准易被「刷榜」(见 111 数据污染)。
- **「GSM8K 怎么提分?」** 思维链(逐步推理)显著有效;Cobbe 2021 还用 verifier 对多个候选解打分选优,优于直接取第一个解——这也是后来 [[079 Scaling Law 与 Chinchilla 最优|test-time compute]] 与「088 GRPO 与可验证奖励」思路的雏形。
- **「老基准都刷到 90%+ 了,还用吗?」** 区分度下降(天花板效应 + 污染),改用更难更抗污染的接班人:MMLU-Pro(10 选项)、GPQA(研究生难题)、MATH/AIME(竞赛数学)、SWE-bench(改真代码库)、LiveCodeBench/LiveBench(只收训练截止后新题、抗污染)。
- **「pass@1 远低于 pass@k 怎么利用?」** 说明模型有解但不稳;用「采样多个 + verifier/重排/多数投票」把高 pass@k 的潜力兑现成可用答案(test-time compute 的核心思路)。
- **「为什么 MMLU 用 5-shot 而不是 0-shot?」** few-shot 给格式范例,稳定输出格式、降低答案抽取噪声,也帮模型「进入答题状态」;5-shot 是 MMLU 约定俗成,报分必须注明 shot 数(见 [[111 评估陷阱：数据污染与格式敏感|格式敏感]])。

## 关键事实

- **MMLU**:Hendrycks et al. 2020,arXiv 2009.03300(ICLR 2021);57 个学科多选题,涵盖 STEM、人文、社科,难度从小学到专业级,标准 5-shot;随机基线 25%,最大 GPT-3 约 43.9%。
- **GSM8K**:Cobbe et al. 2021,arXiv 2110.14168(《Training Verifiers to Solve Math Word Problems》);8.5K 道小学数学应用题(7.5K 训练 + 1K 测试),每题 2–8 步基本四则运算;提出 verifier 选优显著提分。
- **HumanEval**:Chen et al. 2021,arXiv 2107.03374(Codex 论文);164 道手写 Python 题、跑单元测试;指标 pass@k 用无偏估计 $1-\binom{n-c}{k}/\binom{n}{k}$;Codex pass@1=28.8%、pass@100=70.2%,GPT-3=0%。
- **MT-Bench**:Zheng et al. 2023,arXiv 2306.05685(与 Chatbot Arena 同一论文,NeurIPS 2023 D&B);80 道多轮开放题、8 类,GPT-4 裁判 1–10 打分;强裁判与人类偏好一致率 > 80%。
- 这些基准是「外在评估」,与 [[109 语言模型评估：困惑度与 bits-per-byte|困惑度等内在指标]] 互补;真实部署还需关注数据污染与格式敏感(见 111)及人类偏好(见 112)。
- **MMLU-Pro**:Wang et al. 2024,arXiv 2406.01574(NeurIPS 2024);MMLU 加难版,选项 4→10(随机基线 10%)、剔噪声题、增推理题,准确率较 MMLU 掉 16–33%,prompt 敏感度从 4–5% 降到约 2%。
- **GPQA**(Google-Proof Q&A):Rein et al. 2023,arXiv 2311.12022;研究生级理科难题,PhD 专家约 65%、可上网非专家仅约 34%(刻意做到搜不出来)。
- **LiveCodeBench**:Jain et al. 2024,arXiv 2403.07974;只收训练截止后(≥2023-11)的竞赛新题,天然抗污染,覆盖生成/自修复/执行预测;SWE-bench(Jimenez et al. 2023,arXiv 2310.06770)测真实 GitHub issue 补丁,存在 solution leakage 等污染争议。
- 老四件套已部分饱和(前沿 85–95%),区分度靠新一代(MMLU-Pro/GPQA/MATH/AIME/SWE-bench/Live*)维持;pass@k 随 k 单调增,产品体验近 pass@1。
