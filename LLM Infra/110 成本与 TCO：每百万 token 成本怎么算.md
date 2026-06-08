这是 [[LLM Infra]] 域把所有性能优化「翻译成钱」的总账。**每百万 token 成本(cost per million tokens)**=把一张(或一组)GPU 的租用/折旧单价,按它每秒实际产出的 token 数摊下去。一句话:服务做得好不好,最终不看 TFLOPS,看 **$/Mtok**。它把 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标]]、[[016 batch 如何改变算术强度|算术强度]]、量化、[[077 多 token 预测 MTP(DeepSeek)|MTP]] 全部收敛到一个可比的经济量,也是 [[105 SLO、SLA 设计：为推理定指标|SLO 设计]] 和定价的底座。

类比:开网约车。车的**租金按小时算**(GPU 时价),但你赚钱是**按里程(token)**算的。同一辆车,空驶率越高(利用率越低),每公里分摊的租金越贵;一次多拉几个顺路乘客(batch),每公里成本就摊薄。换更省油的开法(量化)等于同样租金跑更多里程。**时价是固定的,单位成本由「有效吞吐」决定**。

小数字手算(70B 在 H100 上):取整卡聚合输出吞吐 ≈ 2000 tok/s(连续批 + FP8),H100 租价取 $2.5/GPU·h。每小时产出 $2000\times3600=7.2\times10^6$ tok = 7.2 Mtok,成本 $2.5。于是 $\$2.5/7.2 \approx \$0.35$/Mtok——但这是**利用率=100%** 的理想值。真实平均利用率常只有 ~50%(白天高峰、夜间空转),空转的卡照样付钱,有效产出减半 → 实际 ≈ **$0.69/Mtok**。这解释了为何 Llama-3.1-70B 类自托管能压到 ~$0.8/Mtok 量级,而 API 报价往往 $2–5/Mtok(含毛利、冗余、长尾)。

$$
\text{Cost}_{/\text{Mtok}}=\frac{P_{\text{gpu/h}}\times N_{\text{gpu}}}{\text{Thr}_{\text{tok/s}}\times 3600}\times 10^{6}\times\frac{1}{u},\qquad
u=\frac{\text{实际吞吐}}{\text{峰值吞吐}}
$$

分母里真正能动的是 $\text{Thr}\times u$=**有效吞吐**。量化 FP8 让 $\text{Thr}$ 翻倍(不加卡)→ 成本砍半;增大 batch 把 [[016 batch 如何改变算术强度|算术强度]]从访存受限推向计算受限,$\text{Thr}$ 和 $u$ 同涨(代价是 TPOT 变差,见 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标]]);投机解码在高接受率时再叠一档。注意 prefill 与 decode 单价不同:输入 token 便宜(并行)、输出 token 贵(逐个生成),报价常拆成 input/output 两档。

![[obs-每百万token成本.png]]

![[obs-110降本杠杆.png]]

更长视角看 **TCO(总拥有成本)**:租价只是「别人替你算好的 TCO+毛利」。自建时 GPU 采购仅占 3 年 TCO 的 ~35–50%,电/冷却 10–20%、网络 10–15%、机房+人力+软件 ~20%;运维让总成本在硬件价上再 **+50–70%**。单卡 H100 采购 $25k–$40k,8 卡 HGX 服务器 >$216k,整系统功耗 ≈1.4 kW/卡。GPU 3–4 年折旧、新代 2–3× 性能,经验阈值:**持续 >1 万 GPU·h/月** 自建才比租划算。

![[obs-推理TCO拆解.png]]

注意两种百分比口径别混:"GPU 占 35–50%"是**占比**(各项÷总 TCO,加起来=100%),"运维 +50–70%"是**加成**(在裸成本上乘的放大系数,可叠加、和可 &gt;100%)——算 $/Mtok 时先定裸成本再乘加成,别把加成塞进饼图当占比:

![[orch-110TCO占比加成.png]]

```python
# 每百万 token 成本计算脚本(含利用率)
def cost_per_mtok(price_gpu_hr, n_gpu, thr_tok_s, util=1.0):
    eff_thr = thr_tok_s * util          # 有效吞吐:空转的卡照付钱
    tok_per_hr = eff_thr * 3600
    return price_gpu_hr * n_gpu / tok_per_hr * 1e6

# 70B / H100,FP8,整卡聚合 2000 tok/s,$2.5/GPU·h
print(round(cost_per_mtok(2.5, 1, 2000, util=1.0), 3))  # 0.347  理想
print(round(cost_per_mtok(2.5, 1, 2000, util=0.5), 3))  # 0.694  现实

# FP8 把吞吐翻倍 → 单价直接砍半(不加卡)
print(round(cost_per_mtok(2.5, 1, 4000, util=0.5), 3))  # 0.347
```

```text
❌ 只比 GPU 时价选硬件:"A100 比 H100 便宜就用 A100"
   → H100 时价高,但 FP8 吞吐 ≈2×,$/Mtok 反而更低;且空转利用率没人管
✅ 永远比 $/Mtok = 时价 ÷ (有效吞吐×3600):先压利用率与吞吐,再谈换卡
```

## 面试高频
- **每百万 token 成本怎么算?** $\text{时价}\times N \div(\text{吞吐}\times3600)\times10^6\div\text{利用率}$;核心是「时价固定,单价由有效吞吐定」。
- **为什么不能只看 GPU 时价?** 单价取决于吞吐;H100 时价比 A100 高但 FP8 吞吐翻倍,$/Mtok 更低。利用率低则空转烧钱。
- **三大降本杠杆?** 量化(吞吐×2)、增大 batch(提算术强度与利用率,代价 TPOT)、投机解码;都作用于分母「有效吞吐」。
- **input 和 output token 单价为何不同?** prefill 并行、便宜;decode 逐 token 生成、贵。报价常拆 input/output 两档。
- **租 vs 自建怎么决策?** 自建 3 年 TCO 中 GPU 仅 ~35–50%,运维再 +50–70%;持续 >~1 万 GPU·h/月 自建才划算,否则租。

## 关键事实
- H100 租价 2025–2026 区间 $1.5–$4/GPU·h(boutique 低至 ~$1.5,超大厂 >$4);1 年期合约 2026 年 3 月回升到 ~$2.35/h(**2026**,SemiAnalysis/IntuitionLabs)。
- 公式:$\text{Cost}_{/\text{Mtok}}=P_{\text{gpu/h}}/(\text{tok/s}\times3600)\times10^6$;有效吞吐 = 吞吐 × 利用率。
- 自托管 70B(如 Llama-3.1-70B)可压到 ~$0.8/Mtok 量级;API 报价常 $2–5/Mtok(**2026**)。
- 3 年 TCO:GPU 35–50%、电/冷 10–20%、网络 10–15%、机房+人力+软件 ~20%;运维在硬件价上 +50–70%(**2025–2026**,Introl/GMI)。
- 单卡 H100 采购 $25k–$40k,8 卡 HGX >$216k,整系统 ≈1.4 kW/卡;GPU 3–4 年折旧。
