[[112 人评、LLM-as-judge 与 Arena]]:**开放式任务没有标准答案,只能靠偏好评估——人评是金标准但贵且慢,LLM-as-judge(用强模型当裁判,如 MT-Bench、AlpacaEval)便宜可规模化但有偏置,Chatbot Arena 用真实用户成对投票 + Elo/Bradley-Terry 排名,兼顾真实分布与抗污染**。

## 直觉

[[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|下游基准]] 里前三个有标准答案、能自动判分。但「写封得体的道歉邮件」「这两个回答哪个更好」这类**开放式任务没有唯一正解**,只能比较「人(或代理)更喜欢哪个」。这就是偏好评估,三种做法各有取舍:

- **人评(human eval)**:招标注员看回答打分或两两比较。**金标准**——最接近真实用户感受,但**贵、慢、难复现**(标注员之间也有分歧)。
- **LLM-as-judge**:用一个强模型(常是 GPT-4)当裁判,代替人去打分或选优。**便宜、快、可规模化**,与人类偏好一致率能 > 80%,所以是人评的廉价近似。代价是裁判**有系统性偏置**(下文)。代表:MT-Bench(1–10 打分)、AlpacaEval(对参考模型算胜率)。
- **Chatbot Arena**:让真实用户对两个**匿名**模型的回答投票,海量投票用 Bradley-Terry 拟合成 Elo 排名。真实用户、真实分布,题目动态不公开 → **天然抗 [[111 评估陷阱：数据污染与格式敏感|数据污染]]**。

![[eval-llm裁判流程.png]]

## 例子

**LLM-as-judge 的三大偏置(小数字)。** Zheng 2023 在 MT-Bench 上量化:

- **位置偏置**:把同一对回答 (A, B) 喂给 GPT-4,它倾向选**排在前面**的那个;交换成 (B, A) 后结论可能翻转。缓解:两种顺序各评一次,**两次都赢才算赢**,否则算平。
- **冗长偏置**:裁判偏爱更长、更啰嗦的回答,哪怕信息量不更高。AlpacaEval 后来专门做**长度去偏(length-controlled)**版本。
- **自我增强偏置**:裁判偏爱「和自己同源/自己写」的回答(GPT-4 裁判略偏 GPT 系)。缓解:换不同厂商裁判或多裁判投票。

**Chatbot Arena 的 Elo(小数字)。** 锚定基准 1500。若模型甲 Elo=1287、模型乙=1265,差 22 分,代入逻辑斯蒂:$P(\text{甲胜})=\frac{1}{1+10^{-(1287-1265)/400}}=\frac{1}{1+10^{-0.055}}\approx 0.53$,即甲对乙约 53% 胜率——分差小则胜率接近五五。排行榜还带置信区间(如 ±5),投票越多区间越窄。

**MT-Bench 一致性(小数字)。** GPT-4 裁判与人类偏好一致率 > 80%,接近「人类标注员之间」的一致率,所以可作人评的自动化替身——但高风险结论仍要人评兜底。

**Arena 的「风格作弊」与 style control(关键更新)。** LMSYS 后来发现:Arena 偏好里掺了**风格因素**——更长、markdown 排版漂亮、分点列表多的回答更讨喜,哪怕内容不更对。于是上线 **style-controlled Elo**:在 Bradley-Terry 回归里把「回答长度、markdown 元素数」等风格特征当**协变量**控制掉,得到「剔除风格后的纯能力排名」。一个模型在原始榜和 style-control 榜上的位次差异,直接量化了它「靠风格吃分」的程度。这是面试高频追问:**Arena 测的是偏好,偏好里混了风格,需 style control 去偏。**

**Arena 与静态基准会打架(重要现象)。** 同一个模型,MMLU 很高但 Arena 排名一般,或反之——这很常见。原因:MMLU 测知识/解题(单轮、有标准答案),Arena 测真实交互偏好(多轮、开放、含风格)。两者**测的根本不是一回事**,出现分歧恰恰说明「单一指标不够」。一个会考试但答得干巴巴的模型 MMLU 高 Arena 低;一个知识一般但对话舒服的模型反之。完整画像要内在指标(PPL)+ 客观基准(MMLU 等)+ 偏好(Arena/人评)三者并看。

![[eval-arena排名.png]]

## 原理

**Bradley-Terry 模型(Arena 排名的数学核)。** 给每个模型 $i$ 一个隐含「强度」参数 $\beta_i$,两两胜负概率由 logistic 给出:

$$\boxed{\ P(i\ \text{胜}\ j)=\frac{1}{1+e^{-(\beta_i-\beta_j)}}=\sigma(\beta_i-\beta_j)\ }$$

把几十万张成对投票当作伯努利观测,用**最大似然(逻辑回归)**一次性估计所有 $\beta_i$。这与在线 Elo 的区别:Elo 是逐场增量更新、受对局顺序影响;Bradley-Terry 假设强度不变、集中式拟合,**顺序无关、更稳**。再换算到熟悉的 Elo 刻度(锚定 1500):

$$\text{Elo}_i=400\cdot\log_{10}\!\frac{\text{strength}_i}{\text{strength}_{\text{anchor}}}+1500$$

由 $\sigma$ 可反推:Elo 差 $400$ → 约 $10:1$ 胜率;差 $0$ → 五五开。

![[eval-bradley-terry-elo.png]]

**位置偏置的去偏(形式化)。** 设裁判对顺序敏感,定义「A 在前判 A 胜」为 $w_{AB}$、「B 在前判 A 胜」为 $w_{BA}$。稳健判定:只有 $w_{AB}\wedge w_{BA}$ 都成立才记 A 胜,只有都不成立才记 B 胜,否则记平局——这样消掉一阶位置效应。

**AlpacaEval 的胜率 + 长度去偏(另一主流 judge 基准)。** 不同于 MT-Bench 的 1–10 绝对打分,AlpacaEval 让裁判把候选模型的回答和一个**固定参考模型**(如 GPT-4-turbo)的回答两两比,算**胜率(win rate)**。问题同样是冗长偏置——长回答更易赢。于是出 **length-controlled(LC)版本**:用回归把「长度」对胜率的影响剥离,得到「同等长度下的真实胜率」。LC-AlpacaEval 与 Arena 人类偏好的相关性显著更高,成了离线 judge 的常用指标。这和 Arena 的 style control 是同一思想:**把风格混淆因子从能力信号里回归掉**。

**评估光谱(怎么选)。**

$$\underbrace{\text{人评}}_{\text{最准、最贵}}\ \longleftrightarrow\ \underbrace{\text{Arena 真实用户投票}}_{\text{真实分布、抗污染、慢}}\ \longleftrightarrow\ \underbrace{\text{LLM-as-judge}}_{\text{便宜快、有偏}}\ \longleftrightarrow\ \underbrace{\text{自动基准}}_{\text{最廉、最易污染}}$$

实践:日常迭代用 LLM-as-judge 快筛,关键里程碑用 Arena/人评定论。注意偏好评估测的是**人类偏好**,不直接等于事实正确——裁判数学/专业题上可能「自信地判错」,别盲信。

**怎么把人评做得可信(标注方法论)。** 人评虽是金标准,做不好一样不可靠:① **明确 rubric**(给清晰的评分维度:有用性/正确性/安全/格式),别让标注员凭感觉;② **成对比较优于打分**——「A 和 B 哪个好」比「给 A 打 7 分」更稳定(人对绝对分尺度不一致,对相对好坏判断更一致);③ **多标注员 + 一致性度量**(如 Cohen's κ / Krippendorff's α),低一致说明 rubric 模糊或任务主观;④ **盲评 + 随机化顺序**消除品牌偏好与位置偏置;⑤ **校准题**抓不认真的标注员。这些和 LLM-as-judge 的去偏是同一套思想——只是把「裁判」从模型换成人。

**LLM-as-judge 的第四个坑:能力上限。** 裁判**不可能可靠评判超出自己能力的回答**——让 GPT-4 评一道它自己做错的数学题,它可能把错的判成对的。所以 judge 应**比被评模型更强或至少同级**,且高风险/专业领域(医疗、法律、前沿数学)别全交给 LLM 裁判。这是「LLM-as-judge 能完全替代人评吗」的标准否定理由之一。

![[eval-llm裁判流程.png]]

## 代码

```python
import numpy as np

# ❌ 错:LLM 裁判只按一个顺序评一次 —— 吃满位置偏置,顺序一换结论就翻
def judge_bad(judge, q, ans_a, ans_b):
    verdict = judge(f"问题:{q}\nA:{ans_a}\nB:{ans_b}\n哪个更好?只答 A 或 B:")
    return verdict   # 偏向排在前面的 A

# ✅ 对:交换顺序各评一次,两次一致才算数,否则判平(消一阶位置偏置）
def judge_robust(judge, q, ans_a, ans_b):
    def ask(first, second):  # 返回裁判选了"第一个(first)"还是"第二个"
        v = judge(f"问题:{q}\n回答1:{first}\n回答2:{second}\n哪个更好?答 1 或 2:")
        return v.strip()
    r1 = ask(ans_a, ans_b)   # A 在前
    r2 = ask(ans_b, ans_a)   # B 在前
    a_wins = (r1 == "1") and (r2 == "2")   # 两种顺序都判 A 好
    b_wins = (r1 == "2") and (r2 == "1")   # 两种顺序都判 B 好
    return "A" if a_wins else "B" if b_wins else "tie"
```

```python
# Bradley-Terry:从成对胜负拟合各模型强度,再换算 Elo（锚定 1500)
def bradley_terry_elo(battles, n_models, iters=200, lr=0.1, anchor=1500.0):
    # battles: [(i, j, winner)]，winner=i 表示 i 胜 j
    beta = np.zeros(n_models)
    for _ in range(iters):                      # 简化的梯度上升做 MLE
        grad = np.zeros(n_models)
        for i, j, w in battles:
            p_i = 1 / (1 + np.exp(-(beta[i] - beta[j])))   # P(i 胜)
            y = 1.0 if w == i else 0.0
            grad[i] += (y - p_i); grad[j] -= (y - p_i)
        beta += lr * grad
    elo = 400 / np.log(10) * (beta - beta.mean()) + anchor  # 换算到 Elo 刻度
    return elo
```

## 面试高频

- **「为什么不能全靠自动基准?」** 开放式任务无标准答案;且自动基准易 [[111 评估陷阱：数据污染与格式敏感|被污染/刷榜]]。要补人类偏好信号(人评 / Arena)。
- **「LLM-as-judge 有哪些偏置?怎么缓解?」** 位置偏置(交换顺序两次都赢才算赢)、冗长偏置(长度去偏)、自我增强偏置(换不同家裁判/多裁判)。还有推理能力有限,数学/专业题可能自信判错。
- **「Chatbot Arena 怎么排名?」** 真实用户对**匿名**模型对的回答投票;海量投票用 **Bradley-Terry** logistic 回归拟合强度,换算成 Elo(锚 1500),带置信区间。
- **「Elo 和 Bradley-Terry 区别?」** Elo 逐场增量更新、受顺序影响;Bradley-Terry 假设强度不变、集中式 MLE 拟合、顺序无关、更稳。Arena 用后者。
- **「Arena 为什么抗数据污染?」** 题目来自真实用户、实时产生且不公开,模型不可能预先「背过」;对比公开静态基准是关键优势。
- **「Arena 有什么弱点?」** 测的是主观偏好(可能偏好风格/讨好语气而非正确性);可被刷票、被「话痨/排版好看」带偏;长尾专业能力覆盖不足。
- **「裁判一致率 > 80% 意味着可以完全替代人评吗?」** 不行。它是廉价近似、适合规模化初筛;高风险或专业结论仍需人评兜底。裁判还有能力上限——评不可靠地判超出自己能力的回答,故 judge 应≥被评模型。
- **「Arena 偏好里的风格作弊怎么处理?」** 偏好掺了长度、markdown 排版等风格因素;LMSYS 用 style-controlled Elo,把风格特征当协变量在 Bradley-Terry 回归里控制掉,得到剔除风格的纯能力排名。原始榜与 style-control 榜的位次差 = 靠风格吃分的程度。
- **「MMLU 高但 Arena 低,矛盾吗?」** 不矛盾。MMLU 测知识/解题(单轮有标答),Arena 测交互偏好(多轮开放含风格),测的不是一回事。分歧恰说明要内在(PPL)+ 客观基准 + 偏好三者并看。
- **「人评怎么做才可信?」** 明确 rubric、成对比较优于打分、多标注员 + 一致性度量(κ/α)、盲评 + 随机顺序、校准题抓水标注——和 LLM-judge 去偏同一套思想。

## 关键事实

- MT-Bench 与 Chatbot Arena 出自同一论文:Zheng et al. 2023,《Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena》,arXiv 2306.05685(NeurIPS 2023 Datasets & Benchmarks)。
- 论文记录 LLM 裁判三大偏置:位置(position)、冗长(verbosity)、自我增强(self-enhancement),并提出顺序交换等缓解法;强裁判(GPT-4)与人类偏好一致率 > 80%,接近人类标注员间一致率。
- Chatbot Arena 由 LMSYS 运营(2023-05 上线),用真实用户成对匿名投票;排名由 **Bradley-Terry** 模型拟合,$P(i\succ j)=\sigma(\beta_i-\beta_j)$,再换算 Elo 刻度,锚定 1500、带置信区间(LMSYS 博客 2023-12 更新方法)。
- AlpacaEval 是另一主流 LLM-as-judge 基准(对参考模型算 win rate),因对长度敏感后推出 length-controlled 版本去偏。
- 偏好评估测「人类偏好」而非「事实正确」;与 [[109 语言模型评估：困惑度与 bits-per-byte|内在指标]] 和 [[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|客观基准]] 互补,完整评估需三者结合。
- Chatbot Arena 偏好掺风格因素(长度、markdown),LMSYS 推出 style-controlled Elo,把风格特征作为协变量在 Bradley-Terry 回归中控制,得到剔除风格的能力排名(LMSYS 博客,2024)。
- LLM-as-judge 有能力上限:无法可靠评判超出自身能力的回答,故裁判应 ≥ 被评模型;高风险/专业领域(医疗、法律、前沿数学)需人评兜底。
- 人评可信性靠方法论:成对比较优于绝对打分、明确 rubric、多标注员 + 一致性度量(Cohen's κ / Krippendorff's α)、盲评 + 随机顺序、校准题;MMLU 与 Arena 排名分歧是常态,印证单一指标不足。
