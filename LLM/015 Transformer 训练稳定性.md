[[015 Transformer 训练稳定性|Transformer 训练稳定性]] 讲的是:深而宽的 Transformer 天生难训——残差流会逐层累加尺度,Post-LN 在初始化时可能出现不利的梯度尺度,开局大学习率就可能发散。一套组合拳让它更稳:**warmup + 合适初始化 + Pre-LN(或残差缩放)+ 梯度裁剪**,外加 bf16、AdamW 等。它们是风险控制旋钮,不是任一配置下的稳定性保证。

## 直觉

为什么「越深越难训」?把 Transformer 想成 N 段串联的放大器,每段都有残差 `x = x + f(x)`。**每加一层,信号方差就往上叠一点**;堆几十层后,深处的激活可能膨胀到数量级失控,梯度也跟着失控(见 [[44 梯度消失、爆炸与梯度裁剪|梯度爆炸]])。再加上注意力里有 softmax、有 `1/√dk` 这些对尺度敏感的操作,初始化稍微不对、学习率稍微大点,**头几百步就能把模型「炸」掉**(loss 飙到 NaN)。

四个旋钮各治一种病:

- **初始化**:别让起点就太大太小——给每层一个「方差守恒」的随机起点(见 [[41 权重初始化(Xavier、He、正交)|权重初始化]])。
- **Pre-LN / 残差缩放**:Pre-LN 让每个子层看到归一化输入、改善梯度路径；残差缩放直接限制新注入残差的尺度。两者都要配合监控 residual stream 的尺度(见 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]])。
- **warmup**:开局别用峰值学习率,**先从 0 慢慢升温**几百~几千步,等梯度统计稳了再加速,然后 cosine 退火(见 [[40 学习率调度与 warmup、cosine|warmup]])。
- **梯度裁剪**:万一某步冒出一个巨大梯度,**按比例缩回**,别让单步毁掉权重(见 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]])。

一句话:**初始化定起点、Pre-LN 改善梯度路径、残差缩放管注入尺度、warmup 控节奏、裁剪兜底——一起降低训练翻车概率。**

## 例子

**warmup 的具体数字**。设峰值 $\eta_{\max}=3\times10^{-4}$、warmup $T_w=2000$ 步。第 $t$ 步学习率:

- $t=0$:$\eta=0$(从零起步,不冲)。
- $t=500$:$\eta=3\times10^{-4}\times\frac{500}{2000}=7.5\times10^{-5}$。
- $t=2000$:$\eta=3\times10^{-4}$(到峰值)。
- 之后 cosine 退火,到训练末降回接近 0。

如果**跳过 warmup**、第 0 步就上 $3\times10^{-4}$:在 Post-LN 网络里,初始化时近输出层的梯度本就偏大,这一大步可能直接把权重推到坏区域,loss 在前几十步飙成 NaN。

**Post-LN vs Pre-LN 的梯度**。Xiong 等(2020)的初始化分析表明,Post-LN 的输出层附近可出现随深度放大的梯度尺度；Pre-LN 的该梯度路径更平稳。这说明 Pre-LN 通常更容易起训，**不等于所有数据、深度、学习率下都可以删除 warmup**。

**残差方差膨胀(小数字)**。设每层新增分支 $f_\ell(\mathrm{LN}(x))$ 近似独立、方差为 $0.1$，且初始 residual stream 方差为 $1$。则 $
\mathrm{Var}(x^{(10)})\approx1+10\times0.1=2$。Pre-LN 使**进入下一子层的输入**先被归一化，但残差流本身仍是累加的；它不会自动把这个 $2$ 变回 $1$。若希望残差流方差也有界，需控制分支初始化/缩放或使用经验证的架构配方。

![[tf-PreLN与warmup.png]]

## 原理

**1. 残差导致方差线性增长**。Pre-LN 形式下 $x^{(\ell)}=x^{(\ell-1)}+f_\ell(\mathrm{LN}(x^{(\ell-1)}))$。若各子层输出近似独立、方差为 $\sigma_f^2$,则

$$\mathrm{Var}(x^{(\ell)})\approx \mathrm{Var}(x^{(0)})+\sum_{k=1}^{\ell}\sigma_{f,k}^2 = O(\ell)$$

在上述独立近似下，残差流方差随深度**线性增长**。两种不同作用的手段:① **残差缩放**——令 $x+\alpha f(x)$；若各层方差近似相同，取 $\alpha=O(1/\sqrt N)$ 可使累加项维持 $O(1)$（GPT-2 对残差输出投影做深度相关缩放）；② **Pre-LN**——每个子层输入先归一，使 $f$ 面对的输入尺度稳定、并改善梯度路径，**但不等价于 residual stream 方差不再累加**。

**2. 初始化要「方差守恒」**。一层线性 $y=Wx$,$W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$,要让 $\mathrm{Var}(y)\approx\mathrm{Var}(x)$,需

$$\mathrm{Var}(W_{ij})=\frac{1}{d_{\text{in}}}\ (\text{Xavier})\quad\text{或}\quad\frac{2}{d_{\text{in}}}\ (\text{He, 配 ReLU})$$

否则信号一层层放大/缩小,几十层后崩。深网常额外按层数缩放输出投影(见 [[41 权重初始化(Xavier、He、正交)|权重初始化]])。

**3. warmup 为何有用(Post-LN 视角)**。Xiong 等(ICML 2020)分析了 Post-LN 初始化时输出层附近的不利梯度尺度；Adam 早期二阶矩估计也尚未稳定。warmup 把 $\eta_t=\eta_{\max}\cdot\min(1,t/T_w)$ 从 0 升起，降低第一批更新过大的风险。Pre-LN 往往允许使用更温和的 warmup 配置，但 warmup 长度仍应由模型深度、有效 batch、数据与目标学习率的消融决定。

**4. 梯度裁剪兜底**。即便上述都做了,数据噪声偶尔会引出超大梯度,造成 loss 尖峰。全局范数裁剪:

$$g\leftarrow g\cdot\min\!\Big(1,\ \frac{c}{\lVert g\rVert_2}\Big)$$

当总梯度范数 $\lVert g\rVert_2$ 超阈值 $c$(常 1.0)时按比例缩回,**方向不变、长度封顶**(见 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]])。

**5. 数值精度与优化器**。bf16 的指数位与 fp32 相同，动态范围大于 fp16，通常更不易发生表示范围溢出；但 matmul 累加、softmax、数据异常和不合适学习率仍可产生 inf/NaN，不能把它当作“不会溢出”。fp16 常需要 loss scaling 来减轻梯度下溢。优化器可用 **AdamW**(解耦权重衰减，见 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|AdamW]])；z-loss、QK-Norm 等是依模型验证的附加稳定化手段。

**fp16 vs bf16 的位分配(为什么动态范围决定一切)**:
- fp16:1 符号 + **5 指数** + 10 尾数 → 最大约 65504,容易上溢成 inf;需要 loss scaling(把 loss 放大再反向、更新前缩回)绕过下溢。
- bf16:1 符号 + **8 指数** + 7 尾数 → 指数位与 fp32 相同,动态范围约 $10^{38}$；代价是尾数精度更低。它降低范围相关风险，不能替代数值监控与稳定的优化配置。
一句话:**训练失稳可能来自范围、累计误差、数据、实现或优化超参；bf16 只改善其中的表示范围问题。**

![[tf-fp16-bf16位域.png]]

**bf16 时归一化层和 softmax 仍用 fp32 累加**:LayerNorm 的均值方差、softmax 的 $\sum e^z$、loss 累加这些"归约"操作在 bf16 下误差累积明显,实现里常强制用 fp32 算(混合精度的细节),否则也会不稳。

![[tf-训练稳定工具箱.png]]

## 代码

```python
import torch, torch.nn as nn, math

# ① 初始化:深层按 1/sqrt(2N) 缩放残差投影(GPT-2 风格)
def init_weights(model, n_layers):
    for name, p in model.named_parameters():
        if name.endswith('proj.weight'):                 # 残差分支的输出投影
            nn.init.normal_(p, mean=0.0,
                            std=0.02 / math.sqrt(2 * n_layers))  # ← 按层数缩小
        elif 'weight' in name and p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)

# ③ warmup + cosine 调度
def lr_lambda(step, warmup=2000, total=100000):
    if step < warmup:
        return step / warmup                              # 线性升温
    prog = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))           # cosine 退火

# 训练 step(④ 梯度裁剪 + bf16)
def train_step(model, opt, sched, x, y, scaler=None):
    with torch.autocast('cuda', dtype=torch.bfloat16):    # ⑤ bf16 自动混合精度
        loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ④ 裁剪
    opt.step(); sched.step(); opt.zero_grad()
    return loss.item()
```

```python
# ❌ 易翻车配方:Post-LN + 无 warmup + 一上来峰值 LR + 默认初始化 + 无裁剪
#   现象:前几十步 loss 直接 NaN
opt = torch.optim.Adam(model.parameters(), lr=3e-4)   # 第 0 步就峰值
# (没有 warmup、没有 clip_grad_norm、没有按层缩放初始化)

# ✅ 稳定配方:Pre-LN + warmup + 按层缩放初始化 + 梯度裁剪 + AdamW + bf16
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1,
                        betas=(0.9, 0.95))            # 大模型常用 β2=0.95
```

## 面试高频

- **Q:Transformer 为什么需要 warmup?** A:训练初期梯度统计不稳、Adam 二阶矩没校准好;尤其 Post-LN 初始化时近输出层梯度 $\propto\sqrt{N}$ 偏大,一上来用峰值学习率会把权重冲垮 → 发散。warmup 让 LR 从 0 慢慢升,给模型平稳起步。
- **Q:Pre-LN 和 Post-LN 哪个更稳?各有什么取舍?** A:Pre-LN 将 LN 放在子层输入侧，初始化梯度路径通常更平稳、较容易调参；Post-LN 更依赖合适 warmup 与配方。不能从这一初始化结论推出“Pre-LN 一定不需要 warmup”或任意任务的最终质量排序。(出处:Xiong 等,ICML 2020)
- **Q:残差为什么会让深层不稳?怎么治?** A:在分支近似独立时，residual stream 是逐层加和，方差可随层数累积。Pre-LN 稳定的是子层看到的输入与梯度路径；要直接限制 residual stream 的新增尺度，使用深度相关初始化/残差缩放，并监控激活、梯度与 loss。
- **Q:训练突然 loss 飙 NaN,排查清单?** A:① 是否漏 warmup / LR 太大;② 是否 fp16 溢出(换 bf16 或调 loss scale);③ 是否没做梯度裁剪;④ 初始化是否按层缩放;⑤ 数据里有无异常 batch;⑥ 是否 Post-LN 但当 Pre-LN 调参。
- **Q:为什么大模型偏好 bf16 而非 fp16?** A:bf16 的 8 位指数与 fp32 相同，表示范围远大于 fp16 的 5 位指数，因此通常较少出现范围溢出/下溢；是否还需要混合精度保护取决于硬件、算子和训练配方。
- **Q:fp16 和 bf16 的位分配差在哪,为什么大模型常选 bf16?** A:fp16 是 5 指数 + 10 尾数；bf16 是 8 指数 + 7 尾数。bf16 用尾数精度换指数范围，适合许多深度学习训练 workload，但不是对所有算子或硬件的无条件优解。
- **Q:用了 bf16 还有哪些地方要保 fp32?** A:LayerNorm 的均值/方差、softmax 的 $\sum e^z$、loss 累加等归约操作,bf16 下误差累积明显,常强制 fp32。
- **Q:z-loss 和 QK-Norm 各防什么?** A:z-loss 惩罚 logits 的 $\log\sum e^z$ 漂移,防输出 logits 整体爆;QK-Norm 对 Q、K 归一,防注意力 logits 量级失控(配合 $\sqrt{d_k}$ 缩放,见 004)。
- **Q:DeepNorm / sandwich norm 解决什么?** A:让 Post-LN 也能训很深(DeepNorm 给残差乘放大系数 + 缩小初始化,可训到上千层);sandwich norm 在子层前后都加 LN 进一步稳输出方差(Gemma 用)。
- **追问:如何证明 Pre-LN 真解决了训练问题?** A:固定数据、有效 batch、优化器与总 token，只改变 norm 位置/残差缩放；记录每层 residual RMS、grad norm、溢出率、loss 曲线和终态验证指标，并对多个 seed 重复。不要只凭一次未 NaN 下结论。
- **陷阱**:梯度裁剪是**兜底**不是根治——若每步都触发裁剪,应回查 LR/初始化/数据；别把裁剪阈值调得过小；别把“Pre-LN 可弱化 warmup”说成“Pre-LN 自动控制 residual 方差或无需 warmup”。

## 关键事实

- Xiong 等《On Layer Normalization in the Transformer Architecture》(ICML 2020,arXiv:2002.04745)分析了 Post-LN 初始化的梯度尺度问题及 Pre-LN 的更平稳梯度路径；它不是“所有 Pre-LN 配置均可取消 warmup”或“Pre-LN 消除 residual stream 方差累积”的定理。
- warmup 作为 Transformer 标配始于原始论文(Vaswani 等,2017,arXiv:1706.03762),其 $\eta$ 调度即「先升后按 $1/\sqrt{\text{step}}$ 降」。
- GPT-2(Radford 等,2019)对残差分支权重做 $1/\sqrt{N}$ 缩放初始化,以控深层方差;现代实践(GPT-3、LLaMA)普遍用 Pre-LN 或 RMSNorm 变体。
- 全局范数梯度裁剪(典型 $c=1.0$)是大模型训练标配,源自 Pascanu 等(2013,arXiv:1211.5063)对 RNN 梯度爆炸的处理。
- 关联:warmup/cosine [[40 学习率调度与 warmup、cosine|学习率调度]];初始化 [[41 权重初始化(Xavier、He、正交)|初始化]];归一化位置 [[010 层归一化：Pre-LN 与 Post-LN|Pre/Post-LN]] 与 [[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]];裁剪 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]];优化器 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|AdamW]];残差本身 [[009 残差连接与梯度流|残差连接]]。
