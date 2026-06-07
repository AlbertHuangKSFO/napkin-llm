[[066 训练不稳定：loss spike 与对策|训练不稳定：loss spike 与对策]] 讲大模型预训练里突然出现的 **loss spike(损失尖峰)**:loss 平稳下降时猛地飙高,处理不当会发散、整轮训练报废。成因多为数值溢出、坏数据、学习率过大;对策从轻到重是梯度裁剪 → 跳过坏 batch → 回滚 checkpoint → 降 lr。梯度裁剪部分接 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]]。

## 直觉

训练曲线本该平滑下降,但大规模预训练里常见这样一幕:某一步 loss 突然窜高(几倍甚至几十倍),梯度范数(grad norm)同时暴涨。如果不管,优化器会被这一脚踹离正常轨道,要么慢慢爬回来,要么直接**发散**(loss 一路涨到 nan),几天的算力打水漂。

为什么会 spike?几类成因:

- **数值问题**:[[064 混合精度 FP16、BF16 与 FP8|FP16/BF16]] 下某处溢出(注意力 logits 过大、$e^x$ 爆掉)、[[061 优化器与超参(AdamW)|AdamW]] 的二阶矩 $v$ 被一个异常梯度突然拉小,导致自适应步长瞬间放大。
- **坏数据**:某个 batch 撞上重复内容、乱码、超长拼接样本,产生异常大的梯度。
- **学习率过大**:lr 偏高时,在损失面陡壁处一步迈过头,冲上高地。
- **早期信号**:常常是**浅层的梯度范数先飙**,再传导成 loss spike——可作为提前告警。

对策的核心思路:**别让单步坏更新毁掉整轮**。从轻到重——

1. **梯度裁剪**:把梯度范数截到阈值(如 1.0),第一道防线,日常常开。
2. **跳过(skip)坏 batch**:检测到 grad norm 异常就**不更新**这一步。
3. **回滚 checkpoint**:发散了就回到 spike 之前的存档,跳过那段坏数据重训。
4. **降 lr / 加长 warmup**:从源头降低越壁概率。

一句话:**spike 是单步坏更新引爆的;裁剪压幅度、skip/回滚止损、降 lr 治本**。

## 例子

**梯度裁剪救一步**。某步因坏样本,梯度范数 $\lVert g\rVert=50$,正常约 $0.5$。设裁剪阈值 $c=1.0$:

$$g \leftarrow g\cdot\frac{c}{\lVert g\rVert} = g\cdot\frac{1.0}{50} = g\cdot 0.02$$

梯度方向不变,幅度被压到范数 1,这一步的破坏力从「踹飞」降成「正常一步」,spike 被掐灭。若不裁剪,$50$ 倍大的更新极可能让 loss 当场炸上去。

**回滚流程**。监控发现第 12000 步 loss 从 2.1 突然飙到 8.5、grad norm 破 100:

1. 停止训练,加载第 11900 步的 checkpoint(spike 前最近存档)。
2. 定位并**跳过**引发问题的 batch(或那段数据)。
3. 略**降学习率**、确认梯度裁剪开启,从 11900 继续。
4. loss 平稳回到 2.0 附近并继续下降——一次成功恢复。这正是 PaLM 训练里采用的手段:回滚 + 跳过坏 batch。

![[train-loss-spike与恢复.png]]

## 原理

**梯度裁剪(clip by global norm)**。把所有参数梯度拼成一个大向量,算其 L2 范数 $\lVert g\rVert$,超过阈值 $c$ 就整体缩放:

$$\hat g = g\cdot\min\!\left(1,\ \frac{c}{\lVert g\rVert}\right)$$

关键:**按全局范数缩放,方向不变、只压幅度**;只有 $\lVert g\rVert>c$ 时才动手,正常步不受影响。$c$ 常取 $1.0$。详见 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]]。

**spike 与 AdamW 的关系**。AdamW 步长 $\propto \hat m/(\sqrt{\hat v}+\varepsilon)$。若某步梯度异常大,$\hat m$ 升、$\hat v$ 也升,本可自我抑制;但若坏梯度让 $\hat v$ 估计失真(尤其 $\beta_2=0.999$ 长窗口时反应迟钝),自适应步长可能瞬间放大 ⇒ 这也是 LLM 取 $\beta_2=0.95$(短窗口、对突变更敏感)的原因之一,见 [[061 优化器与超参(AdamW)|优化器]]。

**为何 spike 偏爱大模型**。模型越大、序列越长,某些激活/注意力 logits 越易触达 [[064 混合精度 FP16、BF16 与 FP8|FP16]] 的范围上限而溢出;且 Pre-LN 深层残差累积、个别坏样本被放大——故大规模训练几乎必配监控 + 自动 skip。

**结构性防护**(治本):

- **梯度裁剪**(必开)+ 合理 lr、足够长 [[062 学习率调度：warmup 加 cosine 与 WSD|warmup]]。
- **BF16 替 FP16**:范围等同 FP32,从源头减少溢出。
- **归一化加固**:embedding/输出做归一化、QK-Norm(对 Q、K 归一化压 logits)、[[048 路由稳定性：router z-loss|z-loss]](惩罚过大 logits,MoE/输出层常用)。
- **数据清洗**:去重、过滤乱码与异常长样本。

**为什么 spike 偏爱深层 + 大模型(机制细化)。** Pre-LN 残差流里,每层把输出加回主干,**激活范数随深度单调增长**;层数越多、增长越夸张,深层的注意力 logits $q\cdot k$ 越易触达 [[064 混合精度 FP16、BF16 与 FP8|FP16]] 上限或让 $\exp$ 溢出。再叠加个别坏样本被逐层放大,就引爆 spike。**QK-Norm**(对 Q、K 各自做 RMSNorm 再算注意力)直接把 $q\cdot k$ 的尺度钉住,从源头压 logits;Gemma-2、部分新模型用它。**embedding/output 归一化**(如 σ-reparam、嵌入乘常数后归一)同理稳住两端。

**spike 的「确定性 vs 随机性」诊断。** 回滚后从同一 checkpoint **用同一数据顺序重跑**:若 spike **复现**在同一步 → 多半是**特定坏数据**触发(skip 那个 batch 即可);若**不复现** → 多半是数值随机性(如某次 reduce 顺序、非确定 kernel)或临界状态,降 lr / 换 BF16 更对症。所以**数据顺序要可复现**(固定 seed + 记录 dataloader 状态),否则连「是不是数据问题」都判不了。

**自适应裁剪(ZClip 思路)。** 固定阈值 $c=1.0$ 简单但一刀切:训练后期正常 grad norm 本就更小,固定阈值可能既漏掉中等 spike 又偶尔误伤。自适应裁剪维护 grad norm 的**滑动均值 $\mu$ 和标准差 $\sigma$**,把阈值设成 $\mu+z\sigma$(如 $z=3$),即「超过历史 3σ 才算异常并裁/skip」。这随训练自动收紧,对 spike 更敏感、对正常步更宽容(ZClip 2025)。

## 代码

```python
import torch

# ✅ 第一道防线:全局梯度裁剪(几乎所有 LLM 训练都开)
def train_step(model, opt, batch, clip=1.0, skip_thresh=20.0):
    loss = model(batch).mean()
    loss.backward()

    # 算裁剪前的全局梯度范数(也作监控/告警信号)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # ✅ 跳过坏 batch:范数离谱(或 loss 出 nan)就不更新这一步
    if not torch.isfinite(total_norm) or total_norm > skip_thresh:
        opt.zero_grad()                 # 丢弃这步梯度,不 step
        return float("nan"), float(total_norm)   # 记录、告警

    opt.step()
    opt.zero_grad()
    return loss.item(), float(total_norm)

# ❌ 错误一:完全不裁剪 —— 一个坏 batch 的巨大梯度直接把模型踹飞
#   loss.backward(); opt.step()        # 没有任何护栏

# ❌ 错误二:spike 后只是「继续硬训」,指望它自己恢复
#   实践中常一路发散到 nan;应回滚到 spike 前 checkpoint 并跳过坏数据

# ✅ 正确:定期存 checkpoint,便于回滚
#   if step % save_every == 0:
#       torch.save({"model": model.state_dict(),
#                   "opt": opt.state_dict(), "step": step}, f"ckpt_{step}.pt")
```

```python
import collections, torch
# —— 自适应裁剪/skip:按 grad norm 的滑动 μ+zσ 设阈值(ZClip 思路) ——
class AdaptiveClipper:
    def __init__(self, window=100, z=3.0, hard_cap=1.0):
        self.hist = collections.deque(maxlen=window); self.z = z; self.cap = hard_cap
    def step(self, model, opt):
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cap)  # 先硬裁到 cap
        gn = float(gn)
        if len(self.hist) >= 10:
            import statistics
            mu = statistics.mean(self.hist); sd = statistics.pstdev(self.hist) + 1e-8
            if gn > mu + self.z * sd:        # 超历史 3σ → 判为 spike,跳过本步
                opt.zero_grad(); self.hist.append(min(gn, mu + self.z*sd))
                return False, gn             # skipped
        self.hist.append(gn)
        opt.step(); opt.zero_grad()
        return True, gn
# 固定阈值会随训练后期 grad norm 变小而失配;自适应阈值随历史自动收紧

# —— spike 诊断:复现 = 坏数据,不复现 = 数值/随机 ——
# 1) 固定 seed + 记录 dataloader 状态 → 回滚后同序重跑
# 2) 同步复现 → skip 那个 batch;不复现 → 降 lr / 换 BF16 / 关非确定 kernel
```

## 面试高频

- **什么是 loss spike?** 训练中 loss 突然飙高(常伴 grad norm 暴涨);不处理可能发散到 nan,整轮训练作废。
- **常见成因?** 数值溢出(FP16、注意力 logits 爆)、坏数据(重复/乱码/超长样本)、学习率过大;常表现为浅层 grad norm 先飙。
- **从轻到重有哪些对策?** ① 梯度裁剪(clip norm=1.0,第一道防线)② 跳过 grad 异常的 batch ③ 回滚到 spike 前 checkpoint 并跳过坏数据 ④ 降 lr / 加长 warmup ⑤ 结构加固(BF16、QK-Norm、z-loss、归一化、数据清洗)。
- **梯度裁剪怎么工作?** 按全局 L2 范数缩放,只在范数超阈值时压幅度、不改方向;详见 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]]。
- **为什么 PaLM 那类训练要人工回滚?** 大规模训练 spike 难完全避免,工程上靠监控 + 回滚 checkpoint + 跳过坏 batch 来止损(PaLM 技术报告即如此)。
- **怎么提前发现?** 监控 loss 和 grad norm,设阈值告警;grad norm 突增往往早于 loss spike。
- **AdamW 超参与 spike 的关系?** $\beta_2=0.95$(而非 0.999)让二阶矩对梯度突变更敏感,降低 spike 后失稳风险,见 [[061 优化器与超参(AdamW)|优化器]]。
- **为什么深层/大模型更容易 spike?** Pre-LN 残差流激活范数随深度增长,深层注意力 logits $q\cdot k$ 易溢出或让 $\exp$ 爆;坏样本逐层放大引爆。QK-Norm(对 Q、K 归一化)、嵌入/输出归一化能从源头压 logits。
- **怎么判断 spike 是数据问题还是数值问题?** 回滚后用同一数据顺序重跑:同步复现 → 特定坏数据(skip 即可);不复现 → 数值/随机性(降 lr、换 BF16、关非确定 kernel)。前提是数据顺序可复现(固定 seed)。
- **固定裁剪阈值有什么不足,怎么改进?** 固定 1.0 一刀切,训练后期 grad norm 本就更小会失配。自适应裁剪按 grad norm 滑动 $\mu+z\sigma$(如 3σ)设阈值,随训练自动收紧(ZClip 2025)。
- **AdamW 超参与 spike 的关系?** $\beta_2=0.95$(而非 0.999)让二阶矩对梯度突变更敏感,降低 spike 后失稳风险。

## 关键事实

- 梯度裁剪(clip global norm,常用阈值 1.0)是预训练标配的第一道稳定性防线;见 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]]。
- PaLM(Chowdhery et al., 2022,arXiv:2204.02311)报告:loss spike 出现时,从最近 checkpoint 回滚并跳过问题 batch 即可恢复;说明 spike 常由特定数据 × 特定状态触发。
- OPT-175B(Zhang et al., 2022,arXiv:2205.01068)训练日志详细记录了反复的发散、回滚与降 lr 干预。
- BF16(范围等同 FP32)相比 FP16 显著减少溢出型 spike;QK-Norm、z-loss 等归一化手段在新模型中常用以压制过大 logits。
- 近期自适应防护:ZClip(2025,arXiv:2504.02507)按梯度范数统计动态调裁剪阈值;SPAM(2025,arXiv:2501.06842)用动量重置 + spike 感知裁剪缓解尖峰。
