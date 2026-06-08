[[107 基准：MLPerf Inference 与 InferenceMAX|推理基准]]是把不同硬件/软件栈的性能放到同一把尺子上量。两个主流基准代表两种哲学:**MLPerf Inference(MLCommons)** 是定期发版、审核制、严格可比的「行业标准考试」,固定模型/数据集/精度规则,分 **Server 场景**(泊松到达 + 延迟约束,≈ 开环)和 **Offline 场景**(请求一次给齐,测最大批吞吐)。**InferenceMAX / InferenceX(SemiAnalysis)** 是开源、多厂商、**持续(每日)重跑**的基准,扫多组输入/输出长度与并发,画吞吐×延迟×成本/token 的 Pareto 前沿,捕捉软件优化的实时收益。关键反直觉:**两个 tok/s 数字相同也可能完全不可比**——序列长度(prefill/decode 比例)、精度/量化(FP8 vs BF16)、有无投机解码、单位口径(tok/s/GPU vs 整机、offline 峰值 vs server 达标)任一不同,排名都会翻转。这是 [[106 压测方法：开环 vs 闭环、并发模型|压测方法]]在标准化层面的落地,数字最终要回到 [[105 SLO、SLA 设计：为推理定指标|SLO]]和 [[110 成本与 TCO：每百万 token 成本怎么算|成本 TCO]]去解读,底层性能由 [[LLM/078 推理算力、吞吐与延迟、Roofline|Roofline]]决定。

## 直觉

把基准想成**两种汽车测试**。

- **MLPerf = 官方排放/油耗认证**:规定路况、规定油品、规定测法,半年出一版,有人审核。好处是**跨厂商可比、权威**;坏处是出版那一刻就成快照,跟不上软件每天的优化。
- **InferenceMAX = 赛道实时计时器**:同一批车天天上赛道重跑,软件一更新(新版 vLLM/SGLang)立刻重测,画出「速度 vs 油耗 vs 每公里成本」的最优边界。好处是**反映当下真实水平**;坏处是每天变,引用必须注明日期和配置。

而「可比性陷阱」就像比油耗却不说路况:市区还是高速(短序列还是长序列)、加 92 还是 95 号油(BF16 还是 FP8)——不对齐这些,数字大小毫无意义。

## 例子

**MLPerf Inference v5.1(2025-09 发布)** 部分结果:

| 模型 / 系统 | 场景 | 吞吐 | 备注 |
|---|---|---|---|
| Llama 3.1-8B,单 H100 | Offline | ~5777 tok/s | 单卡峰值吞吐 |
| Llama 3.1-8B,单 H100 | Server | ~5103 tok/s | 带延迟约束,低于 offline |
| Llama 3.1-8B,单 L40S | Offline | ~1642 tok/s | 弱卡对照 |
| DeepSeek-R1,GB300 NVL72 | Offline | ~5842 tok/s/GPU | Blackwell Ultra,比 Blackwell +45% |

- **Server < Offline**:同模型同卡,Server(开环 + 延迟约束)吞吐比 Offline(无约束批处理)低 ~12%(5103 vs 5777)——这就是延迟 SLO 的代价,**两个数不能混着比**。
- v5.1 新增 **DeepSeek-R1 推理基准、Whisper-v3 语音、Llama 3.1 8B 小模型**,累计 90000+ 提交结果。

**InferenceMAX(2025-10)** 趋势:Blackwell 比 Hopper 在**相近延迟下吞吐约 4×**(gpt-oss 120B、Llama 3.3 70B);AMD MI300X→MI355X(ROCm 7)约 **5× 代际提升**。

- 同样是「快几倍」,**必须连着说清在哪个延迟点、哪个序列长度、哪个精度**——脱离前沿点的「4×」是无意义的。

## 原理

**MLPerf 场景定义**:
- **Offline**:全部样本一次性可见,系统自由批处理,指标 = 峰值吞吐(samples/s 或 tok/s)。对应 [[017 吞吐与延迟：根本性张力|吞吐]]上限。
- **Server**:请求按**泊松到达**注入,且必须满足延迟约束(如 TTFT/TPOT 百分位),指标 = 满足约束下的最大达标吞吐 ≈ [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]]。本质是开环压测。
- 分 **Closed Division**(锁死模型/精度,严格可比)与 **Open Division**(允许改模型/量化,展示创新但弱可比)。

**InferenceMAX 的 Pareto 思想**:不报单点,而报前沿。对每个配置 $c$(GPU×软件×精度),扫并发得到 $(\text{latency}, \text{throughput})$ 散点,取**非支配集**:

$$
\text{Pareto} = \{c : \nexists\, c'\ \text{s.t.}\ \text{tput}(c') \ge \text{tput}(c)\ \wedge\ \text{lat}(c') \le \text{lat}(c)\}
$$

即「没有别的点能在不牺牲延迟的前提下更高吞吐」。这避免了单点数字的误导。

**三点小例:谁支配谁。** 设三个配置(延迟越低越好、吞吐越高越好):
- $A$:延迟 50ms,吞吐 3000 tok/s
- $B$:延迟 80ms,吞吐 3000 tok/s
- $C$:延迟 80ms,吞吐 5000 tok/s

$A$ 和 $B$ 吞吐相同,但 $A$ 延迟更低 → **$A$ 支配 $B$**($B$ 被支配,出局)。$A$ 与 $C$ 谁也不支配谁:$A$ 延迟更低、$C$ 吞吐更高,是一对 trade-off。于是 **Pareto 前沿 = $\{A, C\}$**,$B$ 不在其中。读法:要 50ms 内必选 $A$(只能拿 3000);能放宽到 80ms 就选 $C$(拿到 5000)。光看「峰值吞吐 5000」吹 $C$、却不说它要 80ms,就是典型的单点误导——前沿强制你把延迟一并报出来。

**可比性的三道关**(对齐才可比):

$$
\text{comparable} \iff (\text{seq\_len}_{in/out}\ \text{同}) \wedge (\text{精度/量化}\ \text{同}) \wedge (\text{口径}\ \text{同})
$$

口径含:tok/s/GPU vs 整机、是否含投机解码、offline 峰值 vs server 达标、是否带延迟 SLO。任一不齐,排名可翻转。

## 图

![[obs-基准对比.png]]

![[obs-107MLPerf场景.png]]

![[obs-107Pareto前沿.png]]

## 代码

复现 + 报告可比口径(用 vLLM 自带基准,显式钉死所有变量):

```bash
# ✅ 复现:把所有影响可比性的变量显式钉死,并随结果一起记录
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \   # 序列长度:必报
  --request-rate 8 \                                    # 开环到达率(server 口径)
  --num-prompts 2000 \
  --percentile-metrics ttft,tpot,itl,e2el               # 报百分位,非平均
# 另记录:GPU 型号、精度(--quantization fp8)、引擎版本、是否投机解码
```

```python
# ❌ 反模式:跨配置直接比裸 tok/s
def compare_BAD(a_tps, b_tps):
    return "A 更快" if a_tps > b_tps else "B 更快"
    # 若 A 是 128/128 短序列、B 是 1024/1024 长序列,或精度/口径不同 → 结论无效

# ✅ 先校验口径一致再比,并以 Pareto 点(同延迟下吞吐)对比
def comparable(a, b):
    keys = ("in_len","out_len","precision","spec_decode","unit","slo")
    return all(a[k] == b[k] for k in keys)   # 全对齐才允许比较
```

`❌` 跨序列长度/精度/口径直接比 tok/s 是面试与生产里最常见的造假来源;`✅` 把序列长度、精度、引擎版本、是否投机解码、单位口径全部钉死并随数字公布,且以同延迟点(Pareto)而非孤立峰值对比。MLPerf 用审核制强制这点,InferenceMAX 用「报前沿 + 标日期/配置」实现这点。

## 面试高频

- **Q:MLPerf 的 Server 和 Offline 场景区别?** Offline 请求一次给齐、自由批处理、测峰值吞吐;Server 按泊松到达 + 延迟约束、测达标吞吐(≈ goodput,本质开环)。同模型 Server 吞吐低于 Offline,差额就是延迟 SLO 的代价。
- **Q:MLPerf 和 InferenceMAX 定位差异?** MLPerf 定期发版、审核制、严格可比、是行业标准快照;InferenceMAX/InferenceX 开源、多厂商、每日持续重跑、报吞吐×延迟×成本 Pareto 前沿,捕捉软件优化实时收益但每日变动。
- **Q:为什么两个 tok/s 数字可能不可比?** 序列长度(prefill/decode 比例)、精度/量化(FP8 vs BF16)、有无投机解码、单位口径(tok/s/GPU vs 整机、offline 峰值 vs server 达标、有无 SLO)任一不同,排名都会翻转。
- **Q:Closed 和 Open Division?** Closed 锁死模型/精度规则,严格可比;Open 允许改模型/量化,展示创新但可比性弱。比厂商性能要看 Closed。
- **Q:为什么报 Pareto 前沿而非单点?** 单点(如「峰值吞吐」)会误导,因为它可能在延迟极差处取得;Pareto 前沿同时呈现吞吐与延迟权衡,选型时按目标延迟读对应吞吐才公平。
- **Q:引用 InferenceMAX 数字要注意什么?** 它每日重跑,每个数字是某天某配置的快照;引用必须注明日期、GPU、软件版本、序列长度、精度,否则不可复现也不可比。

## 关键事实

- **MLPerf Inference v5.1(2025-09,MLCommons)** 新增 DeepSeek-R1 推理、Whisper-v3 语音、Llama 3.1 8B 三个基准,累计 90000+ 结果;Server(开环+延迟约束)与 Offline(峰值吞吐)是两套不可混比的口径,NVIDIA Blackwell Ultra(GB300)登顶。
- **InferenceMAX / InferenceX(SemiAnalysis,2025)** 是开源、多厂商、每日持续重跑的基准,报吞吐×延迟×成本/token Pareto;2025-10 数据显示 Blackwell 相近延迟下比 Hopper 吞吐约 4×,AMD MI300X→MI355X(ROCm7)约 5× 代际提升;获 OpenAI、Microsoft、Meta、vLLM、SGLang 等背书。
- **可比性三要素**:序列长度(输入/输出)、精度/量化、单位口径(含是否投机解码、是否带 SLO)必须对齐;不齐则 tok/s 排名可翻转——这是面试与厂商对比中最高频的陷阱(2025–2026 仍如此)。
