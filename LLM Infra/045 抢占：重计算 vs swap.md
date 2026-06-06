[[045 抢占：重计算 vs swap|抢占的两条路]]回答一个具体问题:当 [[044 调度器设计：waiting、running 队列与抢占|调度器]]决定抢占一个请求腾出 [[LLM/102 KV-Cache|KV-Cache]] 块时,被踢出的请求的 KV 怎么办?两个选择——**recompute(重计算)**:直接丢弃它的 KV 块,恢复时把已生成的 token 当新 prompt **重跑一遍 [[013 Prefill 阶段：计算受限|prefill]]**;**swap(换出)**:把 KV 块经 PCIe **拷到 CPU 内存**保留,恢复时再拷回显存接着 [[014 Decode 阶段：访存受限|decode]]。本质是一道经典取舍:**recompute 拿计算换显存,swap 拿带宽+CPU 内存换计算**。vLLM V1(2025)默认选 recompute。这是显存墙(见 [[010 显存墙与 LLM 推理的本质约束|显存墙]])下不得不做的折中。

## 直觉

被抢占 = **一个正在用餐的客人被请暂时离席,稍后请回来续上**。他桌上已经吃了一半的菜(已生成 token 的 KV)怎么处理?

- **recompute**:**直接撤桌、菜倒掉**。客人回来时,厨房**照着他已点的重新做一遍**到当前进度。好处:撤桌瞬间完成、不占别处空间;坏处:回来要重做(重算),序列越长重做越贵。
- **swap**:**把半成品打包进冰箱(CPU 内存)**。客人回来时,从冰箱**取出来加热**接着吃。好处:不用重做;坏处:打包和取出都要走一段路(PCIe 拷贝),且冰箱占地方(CPU 内存)。

哪个划算取决于:**重做一遍贵,还是来回搬两趟贵?** 序列短 → 重做便宜,选 recompute;序列长 + PCIe 快 → 来回搬更值,选 swap。

## 例子

一个已生成 800 token 的请求被抢占(prompt 200 + 输出 600),模型 13B,H100。

**recompute 代价**:恢复时把这 800 token 当新 prompt 重跑 prefill。
- FLOPs ≈ $2 \times 13\text{e}9 \times 800 \approx 2.1\times10^{13}$ ≈ 21 TFLOP,H100 ~1 PFLOP/s → 理论 ~21 ms,实际 ~30–50 ms 计算。
- 不占 PCIe、不占 CPU 内存;若有 [[042 chunked prefill：切块融合|prefix caching]] 命中公共前缀,只需重算尾部,更便宜。

**swap 代价**:换出 + 换回 = 两次拷贝。
- 800 token 的 KV(13B,GQA 后)≈ 几十到上百 MB;PCIe Gen5 ~64 GB/s 单向。
- 假设 200 MB:换出 ~3 ms + 换回 ~3 ms ≈ 6 ms 带宽,加占用 CPU 内存。

这个例子里 **swap(~6 ms)比 recompute(~30+ ms)快**——因为序列较长、重算成本超过两次拷贝。但若序列只有 50 token,重算 ~3 ms < 拷贝 ~6 ms,recompute 反而赢。**交叉点随序列长度与 PCIe 速度移动**,这正是 vLLM V1 在其架构下默认选 recompute 的原因:V1 重算开销更低、实现更简单,且避免了 CPU 内存压力与拷贝同步开销。

## 原理

被抢占请求当前序列长 $L$,设:
- recompute 成本(计算):$T_\text{recomp} \approx \dfrac{2 N_\text{params} L}{\text{FLOP/s}\cdot u}$(随 $L$ 线性,prefill 利用率 $u$ 高)。
- swap 成本(带宽):$T_\text{swap} \approx \dfrac{2\,\text{KVbytes}(L)}{\text{BW}_\text{PCIe}}$(换出+换回两趟,$\text{KVbytes}(L)$ 也随 $L$ 线性)。

二者都随 $L$ 线性,谁更陡取决于常数:

$$
\frac{T_\text{recomp}}{T_\text{swap}} \approx \frac{2 N_\text{params}/(\text{FLOP/s}\cdot u)}{2\,(\text{bytes/token}_\text{KV})/\text{BW}_\text{PCIe}}
$$

- 模型大($N_\text{params}$ 大)、PCIe 快 → 比值大 → **swap 划算**。
- 模型小、PCIe 慢、显存紧张(没 CPU 内存可换)→ **recompute 划算**。

注意还有隐性成本:swap 占 CPU 内存且拷贝需与计算流同步(可能阻塞);recompute 在抢占瞬间零成本(直接丢块),代价全部推迟到恢复时。vLLM V1 重构后 recompute 路径更轻,故设为默认;swap 作为可选。

## 图

![[sched-抢占重计算vs换出.svg]]

## 代码

两种抢占的实现轮廓:

```python
def preempt(req, mode="recompute"):
    if mode == "recompute":
        # ✅ 抢占瞬间零成本：直接丢 KV 块、释放显存
        free_kv_blocks(req.kv_blocks)
        req.kv_blocks = None
        req.prefilled = 0                 # 标记需从头重算
        req.state = "WAITING_RECOMPUTE"   # 退回 waiting 队头，恢复时整段重跑 prefill
    elif mode == "swap":
        # 换出：KV 经 PCIe 拷到 CPU（占带宽 + CPU 内存）
        req.cpu_kv = copy_to_cpu(req.kv_blocks)   # H2D 反向，异步但需同步点
        free_kv_blocks(req.kv_blocks)
        req.kv_blocks = None
        req.state = "SWAPPED"

def resume(req):
    if req.state == "WAITING_RECOMPUTE":
        run_prefill(req, req.generated_tokens)    # ✅ 把已生成 token 当 prompt 重算
    elif req.state == "SWAPPED":
        req.kv_blocks = copy_to_gpu(req.cpu_kv)   # 换回显存，接着 decode

# ❌ 反模式：长序列 + 慢 PCIe 仍硬选 swap
#    换出/换回拷贝时间 > 重算时间，还白占 CPU 内存 → 恢复延迟更差
# ❌ 反模式：超长序列 + 快 PCIe 仍硬选 recompute
#    每次抢占都重跑整段 prefill，长上下文下重算成本爆炸
```

`❌` 选错模式会让恢复延迟翻倍:长序列该 swap 却 recompute(重算爆炸),或短序列/慢 PCIe 该 recompute 却 swap(拷贝白费)。`✅` 按序列长度与 PCIe 带宽选;不确定时跟随 vLLM V1 默认的 recompute(架构下更轻)。


![[sched-045交叉点曲线.svg]]

## 面试高频

- **Q:抢占时 recompute 和 swap 各做什么?** recompute 丢弃 KV、恢复时把已生成 token 当 prompt 重跑 prefill;swap 把 KV 拷到 CPU 内存、恢复时拷回。
- **Q:本质取舍是什么?** recompute 拿计算换显存(省 CPU 内存与带宽,代价是重算),swap 拿 PCIe 带宽 + CPU 内存换计算(省重算,代价是两次拷贝)。
- **Q:什么时候选哪个?** 序列短 / PCIe 慢 / 显存紧 → recompute;序列长 / PCIe 快 / CPU 内存富余 → swap。交叉点随序列长与带宽移动。
- **Q:vLLM 默认哪个,为什么?** V1(2025)默认 recompute——重构后重算路径更轻、实现更简单,且避免 swap 的 CPU 内存占用与拷贝同步开销;有 prefix caching 时重算还能复用前缀只算尾部。
- **Q:抢占和 OOM 是一回事吗?** 不是。抢占是显存不够时的**主动**腾挪(踢出请求保系统稳定);不抢占放任 KV 增长才会 OOM 崩溃。抢占是 OOM 的预防机制。

## 关键事实

- 两种抢占模式 **recompute(重计算)** 与 **swap(换出到 CPU)** 是 vLLM 应对 KV 显存不足的标准手段;**V1(2025)默认 RECOMPUTE**,因其在新架构下开销低于 swap。
- recompute 成本随被抢占序列长度线性增长(= 一次 prefill);有 **prefix caching** 命中时可只重算未缓存的尾部,大幅降本。
- swap 成本 = 两次 PCIe 拷贝(换出 + 换回),受 PCIe 带宽(Gen5 ~64 GB/s/方向)与 CPU 内存容量约束;长序列、高带宽场景下可能优于 recompute。
