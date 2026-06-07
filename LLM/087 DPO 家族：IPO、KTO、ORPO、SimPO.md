[[087 DPO 家族：IPO、KTO、ORPO、SimPO]]:[[086 DPO 直接偏好优化(推导)|DPO]] 火了之后衍生出一族变体,各自修补 DPO 的一个毛病——**IPO** 换平方损失抗过拟合、**KTO** 用单点「好/坏」标签免成对、**ORPO** 把 SFT 与对齐合一且**免参考模型**、**SimPO** 长度归一 + 目标 margin 且免参考模型;它们共享「直接用偏好监督训策略、不走 [[085 RLHF 全流程与 KL 约束、奖励黑客|PPO]]」的思路。

## 直觉:DPO 不是终点,每个变体治一个病

[[086 DPO 直接偏好优化(推导)|DPO]] 简单好用,但有几处痛:① 容易**过拟合**偏好集、把好坏答案概率一起压低;② 必须有**成对**偏好数据 $(y_w,y_l)$,贵;③ 必须载**参考模型** $\pi_{\text{ref}}$,占显存,且通常还得先单独跑一遍 SFT;④ 不做长度归一,**偏爱长答案**(对数概率求和随长度增长)。

四个变体各砍一刀:

- **IPO**:把 DPO 的 sigmoid 分类损失换成**平方损失**,加入显式正则,缓解过拟合(尤其偏好对几乎确定时 DPO 会无限拉大间隔)。
- **KTO**:借**前景理论**(Kahneman-Tversky),把目标从「成对偏好似然」改成「单条回答的效用」,于是**只要单点的『好/坏』标签**,无需成对——数据便宜得多。
- **ORPO**:把 SFT 损失和一个**赔率比(odds ratio)**惩罚揉成一个损失,**一步完成 SFT + 对齐,且不需要参考模型**。
- **SimPO**:用**长度归一的平均对数概率**当隐式奖励、加一个**目标奖励 margin $\gamma$**,**完全不用参考模型**,最省。

**一图记忆「治什么病」**:DPO 的四个毛病 → 四个变体各对应一刀。① 间隔无限拉大/过拟合 → IPO 用平方损失锚到有限目标;② 必须成对、数据贵 → KTO 改单点好/坏标签;③ 必须 ref + 常需先 SFT → ORPO 把 SFT 和对齐合一、免 ref;④ 偏长 + ref 占显存 → SimPO 长度归一 + 免 ref。记法:**IPO 治「过拟合」、KTO 治「要成对」、ORPO 治「要两步两模型」、SimPO 治「偏长 + 要 ref」**。

![[post-dpo-family.png]]

## 例子:同一偏好对,四种损失各算什么(小数字)

偏好对 $(x,y_w,y_l)$。设 $y_w$ 长 10 token、$y_l$ 长 5 token。$\log\pi_\theta(y_w)=-12$(平均 $-1.2$/tok),$\log\pi_\theta(y_l)=-6$(平均 $-1.2$/tok)。参考模型对应 $\log\pi_{\text{ref}}(y_w)=-13,\ \log\pi_{\text{ref}}(y_l)=-6.5$。

- **DPO**(求和、有 ref):隐式奖励差 $\beta[(-12+13)-(-6+6.5)]=\beta[1-0.5]=0.5\beta$。问题:$y_w$ 更长,**总对数概率天然更负**,DPO 没归一,长度成了混淆变量。
- **SimPO**(平均、无 ref、带 margin):奖励用**平均** $\frac{1}{|y|}\log\pi_\theta$。两者平均都 $-1.2$,差值 $=0$,再减目标 margin $\gamma$ → logits $=-\gamma<0$,**强制好答案的平均对数概率比坏答案至少高出 $\gamma/\beta$** 才算赢。长度被归一掉,不再偏长。具体代入数字:取 $\beta=2.0,\gamma=1.0$,奖励 $r_w=2.0\times(-1.2)=-2.4$、$r_l=2.0\times(-1.2)=-2.4$,logits $=r_w-r_l-\gamma=0-1.0=-1.0$,loss $=-\log\sigma(-1.0)\approx1.31$,**非零**——SimPO 会继续推高 $y_w$ 的平均对数概率,直到它比 $y_l$ 高出至少 $\gamma/\beta=0.5$/token。对比 DPO 在「两者总对数概率差为正」时就满足了,SimPO 的 margin 强制了更大的区分度。
- **KTO**:不看 $y_l$ 相对 $y_w$,而是各自相对参考的**效用**:$y_w$ 标「好」就推高其效用(损失偏好不利的失败),$y_l$ 标「坏」就压低——**两条可以来自不同 prompt、无需配对**。
- **ORPO**:在标准 SFT(最大化 $\log\pi_\theta(y_w)$)之外,加 $\lambda\cdot$赔率比惩罚,压低 $y_l$ 的赔率,**不需要 $\pi_{\text{ref}}$**。

## 原理:四个损失函数

记 DPO 隐式奖励 $\hat r(y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$,$h=\hat r(y_w)-\hat r(y_l)$。

**DPO(基线,回顾)**:$\mathcal{L}_{\text{DPO}}=-\log\sigma(h)$。

**IPO**(Azar et al.,arXiv 2310.12036,*A General Theoretical Paradigm…*)。把目标改成**回归到固定 margin** 的平方损失,$\tau$ 是正则强度:

$$
\mathcal{L}_{\text{IPO}}=\Big(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}-\frac{1}{2\tau}\Big)^2
$$

直觉:DPO 在偏好近乎确定时会把间隔 $h$ 推向 $+\infty$(过拟合);IPO 把它**锚到一个有限目标值** $\frac{1}{2\tau}$,到了就停,**天然带正则**。

**KTO**(Ethayarajh et al. 2024,arXiv 2402.01306,*Model Alignment as Prospect Theoretic Optimization*)。基于前景理论的「人类感知效用」(HALO),对**单条**回答 $y$(带二元标签 desirable/undesirable)定义效用,$z_{\text{ref}}$ 是一个 KL 基准项:

$$
\mathcal{L}_{\text{KTO}}=\mathbb{E}\big[\,w(y)\cdot\big(1-\sigma\big(\,\text{sign}\cdot(\hat r(y)-z_{\text{ref}})\big)\big)\big]
$$

好样本 sign$=+1$(推高其奖励),坏样本 sign$=-1$。$w(y)$ 对「期望」与「厌恶」用**不同权重** $\lambda_D$(desirable)与 $\lambda_U$(undesirable)——对应人类**损失厌恶**(坏体验的痛 > 好体验的爽)。关键收益:**无需成对偏好,单点标签即可**,且对类别不平衡稳健。

**$\lambda_D,\lambda_U$ 怎么治类别不平衡**(KTO 的实用核心)。设好样本数 $n_D$、坏样本数 $n_U$,KTO 用 $\lambda_D,\lambda_U$ 平衡两类的**有效梯度质量**,实务上保持 $\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,\tfrac43]$ 附近。若你的反馈里坏样本远多于好样本(常见,点踩比点赞多),就调 $\lambda$ 把两边拉平;若**更在意压制坏行为**(如毒性、越狱),故意令 $\lambda_U n_U>\lambda_D n_D$,让「避免坏」的权重更大。$z_{\text{ref}}$ 是一个用 batch 内样本估计的 KL 基准(把奖励锚在参考附近),保证效用有意义的零点。这套不对称损失厌恶 + 单点标签,正是 KTO 比 DPO 更贴合「线上稀疏二元反馈」场景的原因。

**ORPO**(Hong et al. 2024,arXiv 2403.07691,*Monolithic Preference Optimization without Reference Model*)。把 SFT 与偏好对齐合成**一个损失,无 $\pi_{\text{ref}}$**。定义赔率 $\text{odds}_\theta(y)=\frac{\pi_\theta(y)}{1-\pi_\theta(y)}$:

$$
\mathcal{L}_{\text{ORPO}}=\underbrace{-\log\pi_\theta(y_w\mid x)}_{\text{SFT:学好答案}}\ +\ \lambda\cdot\underbrace{\Big(-\log\sigma\Big(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\Big)\Big)}_{\text{赔率比惩罚:压低坏答案}}
$$

直觉:边学好答案(SFT)、边用赔率比拉开好坏差距,**一步到位,不用先 SFT 再 DPO,也不用载参考模型**。

**SimPO**(Meng et al. 2024,NeurIPS 2024,arXiv 2405.14734,*Simple Preference Optimization with a Reference-Free Reward*)。隐式奖励用**长度归一的平均对数概率**($|y|$ 是 token 数),并加**目标 margin $\gamma$**:

$$
\mathcal{L}_{\text{SimPO}}=-\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)-\gamma\Big)
$$

两个改动:**(a) 除以长度** → 隐式奖励与生成时的对数似然解码目标对齐,且**不再偏好长答案**;**(b) margin $\gamma>0$** → 要求好答案至少领先 $\gamma$ 才算赢,提升区分度。**无 $\pi_{\text{ref}}$**,最省显存。超参经验(原论文):$\beta\in[2.0,2.5]$、$\gamma\in[0.5,1.5]$,batch size 固定 128,三个要调的超参是 learning_rate、$\beta$、$\gamma$;$\gamma$ 太大会要求过强区分、训练不稳,太小则退化回无 margin。

**为什么去掉 ref 还能稳?** DPO 用「相对参考的对数比」当奖励,参考提供了一个隐式锚点防止概率乱跑;SimPO 去掉参考后,**长度归一的平均对数概率自身**就和解码时真正用的「平均 token 似然」对齐——奖励直接等于「生成这句话有多顺」,锚点由这个绝对量纲提供,加上 margin 防止退化。这就是 SimPO 能 reference-free 还保持稳定的关键。

## 对比表

| 方法 | 损失核心 | 需成对? | 需参考模型 $\pi_{\text{ref}}$? | 治的病 |
|---|---|---|---|---|
| **DPO** | $-\log\sigma(\beta\,\Delta\log\frac{\pi_\theta}{\pi_{\text{ref}}})$ | 是 | 是 | (基线) |
| **IPO** | 平方损失锚到固定 margin | 是 | 是 | DPO 过拟合 / 间隔爆炸 |
| **KTO** | 前景理论效用,单点好坏 | **否** | 是 | 拿不到成对偏好;类别不平衡 |
| **ORPO** | SFT + 赔率比,合一 | 是 | **否** | 省一步 SFT、省 ref |
| **SimPO** | 长度归一平均 logp + margin $\gamma$ | 是 | **否** | 偏长答案、ref 占显存 |

## 选型决策树

- **数据是干净的成对偏好** $(y_w,y_l)$,且偏好近乎确定(标注一致性高)→ 怕 DPO 间隔爆炸,用 **IPO**(平方损失锚定);否则 **DPO** 仍是稳妥默认。
- **只有单条「好/坏」二元反馈**(点赞/点踩、人工审核标 pass/fail),拿不到成对 → **KTO**;尤其类别不平衡或更在意压制坏行为时,调 $\lambda_D,\lambda_U$。
- **想省一步、跳过单独的 SFT 阶段,且不想载参考模型** → **ORPO**(SFT + odds-ratio 一体)。适合从基座直接一步对齐。
- **答案啰嗦、被长度偏置困扰,且想省掉参考模型的那份显存/前向** → **SimPO**(长度归一 + margin,reference-free,最省)。
- 注意:**ORPO 和 SimPO 都免 ref,但定位不同**——ORPO 把 SFT 揉进来(替代「先 SFT 再 DPO」两步),SimPO 假设你已有 SFT 模型、只想更省地对齐。

**共同前提**:这一族全是**离线、无在线采样**的监督式优化,适合「已有现成偏好/反馈数据」;要在线探索、用可验证奖励冲推理上限,得换 PPO/[[088 GRPO 与可验证奖励|GRPO]]。

## 代码:SimPO vs DPO 的核心差异(❌ vs ✅)

```python
import torch.nn.functional as F

# 已有:整段对数概率 logp_w / logp_l(policy);长度 len_w / len_l
# DPO 还需 ref_logp_w / ref_logp_l

def dpo(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    # DPO:对数比求和 + 需要 ref
    return -F.logsigmoid(beta * ((logp_w - ref_w) - (logp_l - ref_l))).mean()

def simpo(logp_w, logp_l, len_w, len_l, beta=2.0, gamma=1.0):
    # ❌ 误区:照搬 DPO 但偷偷去掉 ref —— 那不是 SimPO,缺了长度归一和 margin,会偏长 + 不稳
    # return -F.logsigmoid(beta * (logp_w - logp_l)).mean()

    # ✅ SimPO:奖励 = 长度归一的平均对数概率(无 ref),再减目标 margin γ
    r_w = beta * logp_w / len_w          # 平均每 token,消除长度偏置
    r_l = beta * logp_l / len_l
    return -F.logsigmoid(r_w - r_l - gamma).mean()   # γ:好答案要领先这么多才算赢
```

要点:**SimPO/ORPO 不传 ref**(省一份模型);SimPO 必须**除以长度**且带 margin;KTO 改用**单点标签**的接口(不接收成对)。

## 面试高频

- **DPO 有哪些缺陷,催生了这些变体?** ① 过拟合 / 间隔无限拉大(→ IPO 平方损失正则);② 需成对偏好,贵(→ KTO 单点标签);③ 需 ref + 常需先 SFT(→ ORPO 合一免 ref);④ 偏好长答案、ref 占显存(→ SimPO 长度归一 + 免 ref)。
- **哪些是「免参考模型(reference-free)」的?为什么省?** ORPO 和 SimPO。不载 $\pi_{\text{ref}}$ → 少一份显存与一次前向;SimPO 用平均对数概率自身当奖励,ORPO 用赔率比 + SFT。
- **KTO 最大的实用价值?** 把数据需求从「成对偏好」降到「单条好/坏标签」,数据更易得;且对正负样本不平衡稳健,适合只有点赞/点踩这类反馈的场景。
- **SimPO 为什么要长度归一?** DPO 用对数概率**求和**,长答案天然更负,使奖励与长度纠缠、偏好啰嗦;SimPO 除以 token 数 → 奖励与解码目标对齐、消除长度偏置,$\gamma$ 再保证区分度。
- **ORPO 凭什么不要参考模型?** 它不依赖「相对参考的对数比」当奖励,而是用**赔率比** $\frac{\text{odds}(y_w)}{\text{odds}(y_l)}$ 直接拉开好坏,并把 SFT 项一起优化,锚点由 SFT 项隐式提供。
- **这些方法和 PPO/GRPO 的本质区别?** 全是**离线、无在线采样**的监督式偏好优化(DPO 谱系);PPO/[[088 GRPO 与可验证奖励|GRPO]] 是**在线 RL**,能用可验证/外部奖励、做探索,上限更高但更贵。
- **IPO 的平方损失为什么能抗过拟合?** DPO 的 $-\log\sigma(h)$ 在偏好确定时把间隔 $h$ 推向 $+\infty$(梯度不饱和、无止境拉大);IPO 把目标改成 $(h-\frac1{2\tau})^2$,**回归到一个有限目标值**,到了就停、超了反而被惩罚,$\tau$ 直接控制正则强度,因此不会无限拉大间隔。
- **KTO 的 $\lambda_D,\lambda_U$ 是干嘛的?** 控制对「好/坏」两类样本的不对称损失厌恶权重,用来平衡类别不平衡($\frac{\lambda_D n_D}{\lambda_U n_U}$ 拉到 1 附近),或在更在意压制坏行为时故意加大 $\lambda_U$。这是 KTO 适配真实稀疏二元反馈的关键旋钮。
- **SimPO 的超参怎么调?** 主要调 $\beta\in[2.0,2.5]$、$\gamma\in[0.5,1.5]$ 和学习率;$\gamma$ 控制要求好答案领先多少,过大不稳、过小退化。注意 SimPO 的 $\beta$ 比 DPO 的 $0.1$ 大很多,因为奖励是「平均对数概率」量纲不同。
- **ORPO 一步对齐相比「SFT→DPO」省在哪?** 省一个独立的 SFT 阶段(SFT 项内含)、省一份参考模型(无 ref)、少一次训练流水。代价:把两个目标耦合进一个损失,$\lambda$ 需要调好以平衡「学好答案」与「拉开好坏」。

## 关键事实

- IPO:Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*,arXiv **2310.12036**(2023,AISTATS 2024);平方损失锚定有限 margin,抗 DPO 过拟合。
- KTO:Ethayarajh et al. 2024,*KTO: Model Alignment as Prospect Theoretic Optimization*,arXiv **2402.01306**;HALO 家族、前景理论效用、**单点好坏标签**。
- ORPO:Hong et al. 2024,*ORPO: Monolithic Preference Optimization without Reference Model*,arXiv **2403.07691**(EMNLP 2024);SFT + odds-ratio 一体、**免 ref**。
- SimPO:Meng, Xia, Chen 2024,*SimPO: Simple Preference Optimization with a Reference-Free Reward*,arXiv **2405.14734**(NeurIPS 2024);长度归一平均 logp + 目标 margin $\gamma$、**免 ref**;借鉴 ORPO 的长度归一思想。
- 选型:有干净成对→DPO/IPO;只有单点标签→KTO;省显存/跳 SFT→ORPO;免 ref+控长度→SimPO。
- 关联:[[086 DPO 直接偏好优化(推导)|DPO]]、[[083 奖励模型 RM|RM/Bradley-Terry]]、[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF]]、[[081 指令微调 SFT 与数据构造|SFT]]、[[088 GRPO 与可验证奖励|GRPO]]、[[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|MT-Bench/Arena-Hard 评测]]、[[30 交叉熵与负对数似然|交叉熵]]。
