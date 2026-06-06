[[096 AWQ 与 SmoothQuant|AWQ 与 SmoothQuant]]:两种用**逐通道缩放**而非混合精度来对付离群值的量化方法。**AWQ**(Activation-aware Weight Quantization,激活感知权重量化)发现「不是所有权重一样重要」——按**激活幅度**找出约 1% 的关键权重,用缩放把它们抬高后再统一 int4 量化,精度接近无损又**全程纯整数、硬件友好**。**SmoothQuant** 则把**激活的量化难度数学等价地"迁移"到权重上**,让权重和激活**都能 int8**(W8A8),无需 [[094 LLM.int8 与离群值|LLM.int8]] 的运行时分解。二者都建立在 [[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]上,是 [[095 GPTQ|GPTQ]] 之外 int4 推理与 W8A8 部署的主流方案。

## 直觉:既然离群让量化变难,那就「缩放搬家」,别破例留 fp16

[[094 LLM.int8 与离群值|LLM.int8]] 的办法是把难量化的离群列**破例留 fp16**(混合精度)。这虽然零损失,但**硬件不友好**:要在运行时检测离群列、拆成两路矩阵乘再合并,实现复杂、对 int4 也不够快。AWQ 和 SmoothQuant 换了思路:**用一个数学上等价的逐通道缩放,把"难"重新分配,使所有数据在统一的低比特格点下都好量化**——全程纯整数,没有破例。

**AWQ 的洞察**:权重不是平权的。一个权重重不重要,**看它乘的那个激活通道有多大**——和「大激活」相乘的权重,量化错一点点,对输出的影响就被放大很多。所以按激活幅度,约 **1% 的权重是关键的**。但 LLM.int8() 式「把关键的留 fp16」会引入混合精度。AWQ 改为:**把关键权重所在的通道乘上一个 $s>1$ 放大,对应激活除以 $s$**(两者抵消、输出不变);放大后的关键权重相对量化误差变小,于是**统一 int4 也能把它们量准**,无需任何 fp16。

**SmoothQuant 的洞察**:LLM 里**激活有离群尖峰、难量化,而权重很平坦、好量化**。既然 $Wx$ 里激活和权重相乘,就可以**把激活的"尖"按通道除掉一点($x/s$),同时把权重乘上来($Ws$)**——数学上 $(Ws)(x/s)=Wx$ 完全不变,但**激活的尖峰被压平、权重略变陡**,结果**两边都落进 int8 的舒适区**。等于把量化难度从「激活独扛」**平摊**到激活和权重之间。

一句话:**AWQ 让 int4 权重量化避开关键权重塌陷;SmoothQuant 让激活也能 int8。两者都靠"等价缩放"而非混合精度。**

![[quant-awq-激活感知保护.svg]]

## 例子:缩放如何救回关键权重 / 压平激活尖峰(小数字)

**AWQ**。设一个关键权重 $w=0.18$,int4 对称量化、scale $s_q=0.1$(格距 0.1)。直接量化:$0.18\to\text{round}(1.8)=2\to0.20$,相对误差 $\frac{|0.18-0.20|}{0.18}\approx11\%$。若先把这一列乘 $s=2$:$w\to0.36$,激活对应 $\div2$;量化 $0.36\to\text{round}(3.6)=4\to0.40$,再 $\div s=2$ 等价回 $0.20$?——关键在于**搜索一个让整组关键权重量化误差最小的 $s$**:AWQ 按「激活幅度」网格搜 $s$,使被保护通道的相对量化误差显著下降,实测把困惑度损失压到接近 fp16,而**没有一个权重用 fp16**。

**SmoothQuant**。某激活通道值 $[0.5,\,30.0,\,0.4]$(中间是离群尖峰,$\max=30$,极难 int8),对应权重通道幅度都约 $1.0$(平坦,易量化)。取迁移系数把该通道激活除以 $s=\sqrt{30/1}\approx5.5$、权重乘以 $5.5$:激活变 $[0.09,\,5.45,\,0.07]$(尖峰从 30 压到 5.45,好量化多了),权重从 $1.0$ 变 $5.5$(略大,但权重本就平坦、仍易量化)。$\max$ 从 30 降到约 5.5,**激活的量化分辨率大幅改善**,而输出 $\sum (x/s)(Ws)=\sum xW$ 不变。

![[quant-smoothquant-迁移难度.svg]]

## 原理:等价缩放变换 + 各自的 scale 搜索

两法共享同一个**数学恒等式**:对一个线性层 $y=Wx$,插入逐通道对角缩放 $\text{diag}(s)$:

$$
y = W x = \big(W\,\text{diag}(s)\big)\big(\text{diag}(s)^{-1} x\big)
$$

即「权重某通道乘 $s_j$、激活对应通道除 $s_j$」,**输出严格不变**。缩放是**离线**做的(把 $s$ 吸收进权重和前一层),**推理时零额外计算**。区别在 $s$ 的目标和取法。

**1)AWQ**(目标:让 int4 权重量化误差最小)。

- **重要性度量**:权重的重要性由其对应**激活通道的幅度**衡量(用校准集统计 $\mathbb{E}[|x_j|]$),不是看权重自身大小,也不是看 Hessian。
- **per-channel 缩放**:对激活幅度大的通道,把权重放大($s_j>1$),相对量化误差随之缩小;放大倍率由一个**简单网格搜索**得到——令 $s_j=(\,\overline{|x_j|}\,)^\alpha$,搜超参 $\alpha\in[0,1]$ 使该层量化后输出 MSE 最小。
- **全程纯 int4**:没有任何 fp16 列,统一量化、统一矩阵乘,**硬件友好、推理快**;且不需反向传播,只靠激活统计,对校准集**比 GPTQ 更鲁棒**(不易过拟合校准数据)。

**为什么按「激活幅度」而非「权重大小」或 Hessian 选关键权重?** AWQ 做过消融:① 按**权重自身幅度**选(留大权重)——几乎无改善,说明「权重大」不等于「重要」;② 按**激活幅度**选(保护那些乘大激活的权重列)——效果接近无损。原因回到输出 $y=Wx$:某权重对输出的贡献是 $w\cdot x$,**乘了大激活的权重,量化错一点点会被这个大激活放大成大输出误差**,所以它才「重要」。这比 GPTQ 算 Hessian 更轻(只需激活幅度的一阶统计),也更鲁棒。关键不是「留 fp16 保护」,而是**用缩放把这些列放大**,使它们在统一 int4 下相对量化误差变小——缩放是数学等价的,无混合精度。

**2)SmoothQuant**(目标:让激活和权重都能 int8,即 W8A8)。

- **迁移系数**:对每个通道 $j$,取

$$
s_j = \frac{(\max|x_j|)^{\alpha}}{(\max|W_j|)^{1-\alpha}}
$$

$\alpha\in[0,1]$ 是**迁移强度**:$\alpha$ 大 → 把更多难度从激活搬到权重;典型 $\alpha=0.5$。它让激活除 $s$ 后峰值降下来、权重乘 $s$ 后峰值升一点,使**两者量化难度均衡**。

- **效果**:把原本「激活离群、无法 per-tensor int8」变成「激活和权重都能 int8」,于是可做 **W8A8**(权重 8-bit + 激活 8-bit)推理,享受 int8 Tensor Core 的全部算力,而**不需要 LLM.int8() 的运行时混合精度分解**。

- **$\alpha$ 怎么选(网格搜)**:$\alpha$ 太小(偏 0)→ 几乎不迁移,激活照旧离群、难 int8;$\alpha$ 太大(偏 1)→ 难度全压给权重,反而把权重撑出离群、权重难量化。$\alpha=0.5$ 是「平摊」的对称点:让迁移后激活和权重的**逐通道峰值大致相当**,两边量化难度均衡。实务对不同层/模型在 $[0.5,0.9]$ 附近搜一个最小化整体误差的 $\alpha$(离群越重的模型 $\alpha$ 越大,把更多难度搬给权重)。

**3)三者定位对比**(都为对付离群值,手段不同):

| 方法 | 量化对象 | 处理离群手段 | 是否混合精度 | 典型用途 |
|---|---|---|---|---|
| [[094 LLM.int8 与离群值|LLM.int8]] | 权重+激活 int8 | 离群列拎出走 fp16 | **是**(运行时) | 零退化 8-bit、省显存 |
| SmoothQuant | 权重+激活 int8(W8A8) | 离线缩放把激活难度迁给权重 | 否 | 高吞吐 int8 推理 |
| AWQ | **仅权重 int4** | 缩放保护激活感知的关键权重 | 否 | 单卡 int4、低延迟 |
| [[095 GPTQ|GPTQ]] | **仅权重 int4** | 逐列二阶误差补偿 | 否 | 单卡 int4 |

实务里 SmoothQuant(管激活)与 AWQ/GPTQ(管 int4 权重)目标不同,**可分别使用**;AWQ 和 GPTQ 都是 int4 权重方案,精度接近、二选一。

## 代码:AWQ 缩放保护 + SmoothQuant 迁移(❌ vs ✅)

```python
import torch

W = torch.randn(256, 512)          # 权重 [out, in]
X = torch.randn(512, 128)          # 校准激活 [in, n]

def quant_int4(w):                  # 对称 int4,per-channel(逐 out 行)
    s = w.abs().amax(dim=1, keepdim=True) / 7
    return torch.clamp(torch.round(w / s), -7, 7) * s

# ❌ 错:直接 int4,激活感知的关键权重塌陷,困惑度上升
W_naive = quant_int4(W)
err_naive = ((W @ X - W_naive @ X) ** 2).mean()

# ✅ AWQ:按激活幅度给权重列做 per-channel 缩放保护,再统一 int4
act_scale = X.abs().mean(dim=1)                 # 每个输入通道的激活幅度
alpha = 0.5
s = act_scale.clamp(min=1e-4) ** alpha          # 缩放系数(可网格搜 alpha)
s = s / s.mean()                                # 归一,避免整体漂移
W_scaled = W * s.unsqueeze(0)                    # 权重列 × s(放大关键列)
X_scaled = X / s.unsqueeze(1)                    # 激活 ÷ s(数学等价)
W_awq = quant_int4(W_scaled)
err_awq = ((W @ X - W_awq @ X_scaled) ** 2).mean()
print(f"朴素 int4: {err_naive:.4f}   AWQ: {err_awq:.4f}")  # AWQ 更小

# ✅ SmoothQuant:把激活离群难度迁移到权重,让两者都能 int8
act_max = X.abs().amax(dim=1)                    # 激活逐通道峰值
w_max = W.abs().amax(dim=0)                      # 权重逐输入通道峰值
alpha_sq = 0.5
s_sq = (act_max ** alpha_sq) / (w_max.clamp(min=1e-4) ** (1 - alpha_sq))
s_sq = s_sq.clamp(min=1e-4)
X_smooth = X / s_sq.unsqueeze(1)                 # 激活峰值被压平 → 好 int8
W_smooth = W * s_sq.unsqueeze(0)                 # 权重略变陡 → 仍好 int8
# 现在 X_smooth、W_smooth 都可做 per-tensor int8(W8A8),输出 = 原 W@X
```

## 面试高频

- **AWQ 一句话原理?为什么叫"激活感知"?** 按**激活幅度**判定权重重要性(约 1% 关键),用 per-channel 缩放把关键权重放大后再统一 int4 量化,使其量化误差变小;"激活感知"指**用激活分布而非权重本身**来决定保护谁。
- **AWQ 和 LLM.int8() 都处理离群,差在哪?** LLM.int8() 把关键列**留 fp16**(混合精度、运行时分解、硬件不友好);AWQ **不留任何 fp16**,靠等价缩放让关键权重在统一 int4 下也量得准,硬件友好、更快。
- **SmoothQuant 解决什么?核心变换?** 解决**激活离群导致激活无法 int8** 的问题;核心是恒等变换 $(W\,\text{diag}(s))(\text{diag}(s)^{-1}x)=Wx$,用 $s=\frac{(\max|x|)^\alpha}{(\max|W|)^{1-\alpha}}$ 把激活峰值迁一部分给权重,使**两者都能 int8(W8A8)**。
- **AWQ vs SmoothQuant 目标区别?** AWQ 解决**int4 权重**量化(仅权重);SmoothQuant 解决**激活**量化让 W8A8 可行。一个管权重低比特、一个管激活,可分别使用。
- **AWQ vs GPTQ?** 都做 int4 仅权重;[[095 GPTQ|GPTQ]] 用二阶 Hessian 逐列误差补偿(更"数学"、稍慢、对校准更敏感),AWQ 用激活感知缩放(更轻量、对校准更鲁棒、推理 kernel 友好)。精度接近,实务都常见。
- **这些缩放会增加推理开销吗?** 不会。缩放是**离线**吸收进权重和相邻层的,数学等价,**推理时零额外计算**。
- **SmoothQuant 的 α 怎么理解?** 迁移强度:$\alpha$ 越大,从激活搬到权重的难度越多;$\alpha=0.5$ 是常用折中,使激活和权重量化难度均衡。太大反把权重撑出离群、太小迁移不够,实务在 $[0.5,0.9]$ 搜。
- **AWQ 为什么不用 Hessian?和 GPTQ 比省在哪?** AWQ 只需激活幅度的一阶统计来定缩放,不算 Hessian、不逐列优化,所以更轻、对校准更鲁棒、推理 kernel 友好;GPTQ 用二阶 Hessian 逐列误差补偿,更「数学」、稍慢、对校准更敏感。二者 int4 精度接近。
- **AWQ 消融证明了什么?** 按权重自身大小选关键权重几乎无改善,按激活幅度选才有效——证明「重要」由「乘的激活多大」决定(贡献 $w\cdot x$ 被大激活放大),而非权重本身大小。
- **三条离群处理路线一句话区分?** LLM.int8 = 离群列**留 fp16**(混合精度、运行时分解);SmoothQuant = 把激活离群难度**迁给权重**(W8A8、离线缩放);AWQ = 用缩放**放大激活感知的关键权重**再统一 int4(仅权重)。都靠等价缩放或分解对付离群。
- **AWQ/SmoothQuant 能叠加吗?** 可以分别用:SmoothQuant 管激活让 W8A8 可行,AWQ/GPTQ 管 int4 权重;但 AWQ 与 GPTQ 都是 int4 仅权重方案,通常二选一。

## 关键事实

- AWQ:Lin, Tang, Tang, et al.《AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration》(MIT,2023,arXiv:2306.00978,MLSys 2024 最佳论文)。核心:约 **1%** 显著权重由**激活幅度**确定,per-channel 缩放保护,无混合精度;配套 TinyChat 推理 kernel。
- SmoothQuant:Xiao, Lin, Seznec, et al.《SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models》(MIT/NVIDIA,2022,arXiv:2211.10438,ICML 2023)。实现 **W8A8** 无损 int8 推理,OPT-175B 上约 1.5× 加速、2× 省显存;迁移系数 $s_j=(\max|x_j|)^\alpha/(\max|W_j|)^{1-\alpha}$,默认 $\alpha=0.5$。
- 共同数学基础:逐通道等价缩放 $Wx=(W\,\text{diag}(s))(\text{diag}(s)^{-1}x)$,离线吸收、推理零开销;均依赖少量校准集统计激活分布。
- 定位:AWQ/GPTQ = **int4 仅权重**(单卡省显存、低延迟);SmoothQuant = **W8A8 权重+激活 int8**(高吞吐);LLM.int8() = 混合精度 int8(零退化但运行时分解)。
- 工程:AWQ 集成于 `autoawq`、vLLM、TensorRT-LLM、llama.cpp;SmoothQuant 集成于 TensorRT-LLM 等 [[108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|推理引擎]]。
- 关联:[[094 LLM.int8 与离群值|LLM.int8(离群值)]]、[[095 GPTQ|GPTQ]]、[[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]、[[093 PTQ 与 QAT|PTQ]]、[[097 NF4 与 QLoRA 4-bit|NF4]]。
