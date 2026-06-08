[[100 解码策略：贪心与 Beam|解码策略：贪心与 Beam]] 是自回归生成里**确定性**地从概率分布选下一个 token 的两种搜索方法:**贪心(greedy)**每步只取概率最大的那个、一步锁死;**Beam 搜索**每步保留 beam 宽度条最优部分序列、回头救看似次优的早期选择,以更高代价逼近"整句联合概率最大",再配**长度惩罚**纠正它天然偏短偏保守的毛病。

## 直觉

模型每步给出词表上的概率分布,我们要把"逐步选词"拼成一整句。两种思路:

- **贪心**:每步无脑取最大概率的 token,选完不回头。快、省、确定,但**局部最优常 ≠ 整句最优**——第一步一个稍高概率的词,可能把后面更好的路全堵死,而你回不去。
- **Beam(集束)搜索**:每步同时维护 **$k$ 条**(beam 宽度)累积概率最高的候选序列;每条都扩展所有可能 token,再从所有扩展里**全局保留前 $k$ 条**。相当于"留几手"地往前探,能救回第一步看似次优、但整句更优的路径。$k=1$ 时退化成贪心。

但 Beam 有个系统性偏差:**序列越长,各步 log 概率累加越负 → 总分越低**,于是它**偏爱短句**。所以要除以一个长度归一项(长度惩罚)。还有个有名现象:Beam 让输出**偏保守、偏重复、偏"安全废话"**,因为它一味追高概率,而人类语言其实带"惊喜度";这正是开放式生成转向**采样**([[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样解码]])的原因。

## 例子

词表只有几个词,看第一步贪心怎么吃亏(数字示意)。

- 第 1 步概率:`A 0.45`,`The 0.40`。**贪心选 A**。
- 接 A 后,最好的续写 `dog` 只有 `0.50` → 整句 "A dog" 概率 `0.45×0.50 = 0.225`。
- 接 The 后,续写 `sun` 高达 `0.90` → 整句 "The sun" 概率 `0.40×0.90 = 0.36`。
- **贪心被第一步的 0.45 锁死**,拿到 0.225;**Beam(宽 2)** 第一步把 A 和 The 都留着,第二步发现 "The sun" 0.36 更高 → **救回来了**。

**长度惩罚**为什么必要。"The sun"(2 词)总 log 概率 $\ln 0.36\approx-1.02$;若有个 3 词句各步都还不错,累加却可能到 $-1.5$,**只因它更长就被判输**。用 GNMT 长度归一 $lp(L)=\big(\tfrac{5+L}{6}\big)^{\alpha}$ 把总分除以它($\alpha$ 越大越鼓励长句),消除"短句天然占便宜"。

**把长度惩罚代进数字看效果**。两个候选:A="The sun"(2 词,总 logp $-1.02$),B="A dog runs fast"(4 词,总 logp $-1.50$)。
- $\alpha=0$(不归一):score_A $=-1.02$,score_B $=-1.50$ → 选 A(偏短)。
- $\alpha=0.7$:$lp(2)=(\frac{7}{6})^{0.7}\approx1.11$,$lp(4)=(\frac{9}{6})^{0.7}\approx1.33$。score_A $=-1.02/1.11=-0.919$,score_B $=-1.50/1.33=-1.128$ → 仍选 A,但差距缩小。
- $\alpha=1.5$:$lp(2)\approx1.18$,$lp(4)\approx1.65$。score_A $=-0.864$,score_B $=-0.909$ → 几乎拉平,$\alpha$ 再大就翻转选 B。

可见 **$\alpha$ 是一个连续旋钮**:从「只看总概率(偏短)」平滑过渡到「强烈奖励长句」。HF 的 `length_penalty` 即此 $\alpha$,>1 鼓励长、<1 鼓励短。

![[infer-贪心vsBeam搜索树.png]]

## 原理

**目标**。理想是找联合概率最大的序列 $y^*=\arg\max_y \prod_{t} p(y_t\mid y_{<t},x)$。这是 $V^L$ 的指数搜索空间,精确解不可行,贪心与 Beam 都是近似。

**贪心**:$y_t=\arg\max_{w} p(w\mid y_{<t},x)$,每步 $O(V)$,全程 $O(LV)$,无回溯。

**Beam 搜索**(宽度 $k$)。维护候选集合 $\mathcal B_t$($|\mathcal B_t|=k$),每条记其累积对数概率 $s(y_{\le t})=\sum_{\tau\le t}\ln p(y_\tau\mid y_{<\tau})$。每步:对 $\mathcal B_{t}$ 中每条扩展全部 $V$ 个 token、算新分,从 $k\cdot V$ 个里**取前 $k$** 作 $\mathcal B_{t+1}$。复杂度 $O(L\cdot k\cdot V)$。

**长度归一与长度惩罚**(Wu et al. 2016, GNMT)。最终排序用

$$
\text{score}(y) = \frac{\sum_{t}\ln p(y_t\mid y_{<t},x)}{lp(L)},\qquad lp(L)=\Big(\frac{5+L}{6}\Big)^{\alpha}
$$

$\alpha=0$ 即不归一(偏短);$\alpha$ 增大越奖励长句。HuggingFace 的 `length_penalty` 即此 $\alpha$。机器翻译里还常加 **coverage penalty** 防漏译/欠翻。

**为什么贪心/Beam 偏保守**。它们直接最大化概率,而高概率 token 往往是高频"安全"词,导致输出**平淡、易重复**;Holtzman 等指出最大化似然在开放式生成里反而产生退化文本(neural text degeneration),这正是 [[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样解码]] 要解决的问题。实践:翻译/摘要等"有标准答案"任务爱用 Beam;闲聊/创作用采样。

**一个深刻的反直觉:最可能的句子未必是最好的句子**。人类语言天然带「惊喜度」(信息论上不会每个词都挑最可能的),而 Beam 一味追联合概率最大,反而落进「高频安全词」的低信息区,产出空洞重复。Holtzman 实测:人写文本的 token 概率是**起伏**的(有高有低),而 Beam 输出的概率曲线**平坦地贴在高位**——这正是「机器味」的来源。所以开放式生成的目标不是「最大化似然」,而是「采到既合理又有信息量的文本」,这是从搜索([[100 解码策略：贪心与 Beam|本篇]])转向采样([[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|下一篇]])的根本动机。

**Beam 的兄弟:contrastive search / diverse beam**(扩展知识)。① **diverse beam search** 给不同 beam 组加多样性惩罚,避免 $k$ 条候选高度雷同;② **contrastive search**(2022)在「模型置信度」和「与已生成内容的不相似度」间权衡,兼顾连贯与不重复,是介于 Beam 与采样之间的确定性方法。但工程上开放式生成仍以温度 + top-p 采样为主。

## 代码

```python
import numpy as np

def greedy_decode(step_fn, start, max_len, eos):
    """step_fn(seq)->下一步概率分布(V,)。贪心:每步取 argmax。"""
    seq = list(start)
    for _ in range(max_len):
        p = step_fn(seq)
        nxt = int(np.argmax(p))          # ❌ 只看当前最大,无回溯,可能整句次优
        seq.append(nxt)
        if nxt == eos: break
    return seq

def beam_search(step_fn, start, max_len, eos, k=4, alpha=0.7):
    """✅ 保留 k 条最优;用 GNMT 长度归一再排序,纠正偏短。"""
    beams = [(list(start), 0.0)]                      # (序列, 累积 logprob)
    finished = []
    for _ in range(max_len):
        cand = []
        for seq, score in beams:
            if seq and seq[-1] == eos:
                finished.append((seq, score)); continue
            logp = np.log(step_fn(seq) + 1e-12)
            for tok in np.argsort(logp)[-k:]:         # 每条只需扩展 top-k 个就够
                cand.append((seq + [int(tok)], score + float(logp[tok])))
        # 从所有扩展里全局保留 k 条(关键:跨候选比较,才能救回早期次优)
        beams = sorted(cand, key=lambda x: x[1], reverse=True)[:k]
    finished += beams
    lp = lambda L: ((5 + L) / 6) ** alpha             # 长度归一,消除偏短
    return max(finished, key=lambda x: x[1] / lp(len(x[0])))[0]
```

## 面试高频

- **贪心和 Beam 的本质区别?** 贪心每步取最大、一步锁死;Beam 每步保留 $k$ 条累积最优、能回头救早期次优,逼近整句联合概率最大。$k=1$ 时 Beam = 贪心。
- **Beam 宽度越大越好吗?** 否。宽度增大算力线性涨,且过大反而更易产出**平淡、重复、偏短**的退化文本;翻译常用 4–8 即够,再大收益递减甚至变差。
- **为什么 Beam 偏短?怎么修?** 各步 logprob 累加为负,句越长总分越低,故偏短;用长度归一 $((5+L)/6)^\alpha$ 把总分除掉,$\alpha$ 越大越鼓励长句(HF 的 `length_penalty`)。
- **为什么开放式生成不用 Beam 用采样?** Beam/贪心最大化似然 → 偏高频"安全"词,输出保守、重复、缺多样性;创作/闲聊改用 [[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样]](温度 + top-k/top-p)引入随机性与多样性。
- **贪心和 Beam 都是确定的吗?** 是,给定模型与输入,输出可复现(无随机);采样才引入随机性(可设温度 0 退化为贪心)。
- **Beam 的复杂度?** $O(L\cdot k\cdot V)$ 量级;每步实现上只需各候选扩展 top-$k$ 再全局取前 $k$。
- **为什么「最可能的句子」未必最好?** 人类语言带惊喜度(概率有起伏),Beam 追联合概率最大反而落进高频安全词的低信息区,输出平淡重复;Holtzman 实测人写文本概率曲线起伏、Beam 输出曲线平贴高位——这是开放式生成转向采样的根本动机。
- **Beam 用 KV-Cache 有什么特别?** 每条 beam 是不同前缀,需各自维护 K/V;beam 分裂/淘汰时要相应复制/释放缓存,显存随 $k$ 倍增。这也是大 $k$ 在长序列上变贵的原因之一。

![[infer-beamKV.png]]
- **diverse beam / contrastive search 是什么?** diverse beam 给候选加多样性惩罚防雷同;contrastive search 在置信度与不相似度间权衡,兼顾连贯与不重复,是介于 Beam 和采样之间的确定性方法。
- **温度 0 的采样和贪心一样吗?** 一样。采样设 $T\to0$ 时分布塌成 one-hot,等价 argmax = 贪心;所以贪心可看作采样的极限特例(见 [[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样]])。

## 关键事实

- 贪心是 Beam 宽度 $k=1$ 的特例;两者都是 $V^L$ 指数搜索空间的近似,非精确最优。
- Beam 长度归一与覆盖惩罚出自 Wu et al.《Google's Neural Machine Translation System》(2016,arXiv:1609.08144),长度归一 $lp(L)=((5+L)/6)^\alpha$。
- 最大化似然在开放式生成里产生退化(重复、平淡)文本,见 Holtzman et al.(2019,arXiv:1904.09751),促成 [[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|nucleus 采样]]。
- 经验:翻译/摘要/代码等约束强、有参考答案的任务用 Beam(宽 4–8);对话/创作用采样。`length_penalty`>1 鼓励长句,<1 鼓励短句。
- 解码与 [[102 KV-Cache|KV-Cache]] 协同:每步只前向新 token、复用缓存的历史 K/V,使逐步解码代价大幅下降。
