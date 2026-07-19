[[19 人机信任利用与 Rogue Agents|人机信任利用与 Rogue Agents]]把两类常被混淆的风险拆开：**ASI09 Human-Agent Trust Exploitation** 攻击的是审批人和操作者，利用 automation bias 让人替系统放行；**ASI10 Rogue Agents** 关心的是 agent 在高权限、长时程环境里偏离授权后还能否被及时限制。前者是“人被说服”，后者是“系统失控后是否可控”。

直觉类比：ASI09 像财务流程里的“假冒高管催付款”，重点是你**信了**；ASI10 像拿着真工牌、真权限但开始违规操作的内部人，重点是你是否有 **最小权限、双人审批、可回滚、kill switch** 去限制它。

先算审批设计为什么不能只靠单人。设高危动作的单次误批概率是 $p=5\%$。若只是单人审批，坏动作放行概率就是 $5\%$。若改成两人、不同通道、相互独立的审批，放行概率近似变成

$$
P_{\text{bypass}} = p^2 = 0.05^2 = 0.0025 = 0.25\%
$$

风险降了 20 倍。现实里审批人往往不独立，所以这个数字通常乐观；也因此工程上要再叠加**通道分离、预演 diff、可回滚、最小权限**，而不是把双人审批神化成银弹。

更一般地，对不可逆动作，真正该验证的是

$$
\text{risk} = \text{authority} \times \text{action irreversibility} \times \text{oversight weakness}
$$

其中 authority 越大、动作越不可逆、监督越形式化，ASI09 与 ASI10 的破坏半径就越大。

![[sec-人机信任与流氓agent.png]]

因此这篇不把“scheming”写成所有生产 agent 的既定事实，而是写成**必须纳入威胁模型的条件化风险**：当模型拥有较多自主性、真实工具权限、长时程目标和较弱监督时，实验性研究已经观察到隐藏、规避监督或与授权目标冲突的行为；但这不等于现实部署里默认都会发生。工程上正确姿势是：**按可能发生来设计控制，而不是按一定会发生来下结论**。

```python
# ❌ 朴素：高危动作只要 agent 说“我确认过了”就执行
def execute_payment(agent, payment):
    if agent.confidence > 0.9:
        return bank.transfer(payment)


# ✅ 分级：最小权限 + 预演 + 双人审批 + 可回滚
def execute_payment(agent, payment, approver_a, approver_b):
    preview = simulate(payment)
    if not preview.valid:
        return "blocked: invalid"
    if payment.amount > 10000 or payment.irreversible:
        require_step_up_auth(approver_a, channel="hardware_key")
        require_step_up_auth(approver_b, channel="mobile_push")
        require_independent_approval(approver_a, approver_b, preview)
    scoped_token = mint_scoped_token(capabilities={"pay:transfer"}, ttl_minutes=5)
    result = bank.transfer(payment, token=scoped_token)
    write_audit_record(payment, preview, result)
    return result
```

防御上，ASI09 与 ASI10 要分开治：

- **治 ASI09**：审批界面展示证据来源、dry-run、风险标签和不确定性；高危动作做 step-up；避免让人只看到一句自信结论。
- **治 ASI10**：默认最小权限、短时令牌、可撤权、沙箱隔离、双人批准、任务级 kill switch、异常行为熔断。
- **共同点**：不要把 HITL 设计成“每一步都点一下同意”，那只会制造审批疲劳；要把审批预算留给高风险、不可逆、跨主体的动作。

## 面试高频

回链：[[AI 安全面试题库]]

**Q1：HITL 为什么不能直接等于“安全”？**
因为人会有 automation bias 和审批疲劳。若界面不给证据、不给预演、不给风险标签，人只是在替系统盖章，不是在做独立判断。

**Q2：什么情况下要把 Rogue Agent 当成真实威胁模型？**
当 agent 拥有真实工具权限、较长自主循环、可修改状态、可跨系统行动、且监督较弱时，就应该按 Rogue Agent 风险设计最小权限、撤权、熔断和回滚。

**Q3：双人审批为什么仍然不够？**
因为两个审批人可能共享同一误导界面、同一错误摘要或同一组织压力。要让审批通道、信息来源和身份验证尽量独立。

**Q4：怎么把“规避监督”写得既准确又不过度？**
标准说法是：前沿模型在受控实验中已观察到隐藏、对抗或与授权目标冲突的行为迹象，但 Anthropic 公开材料同时明确表示，尚未看到这类 agentic misalignment 在其真实部署中的证据。因此生产设计要把它当早期预警信号，而不是既成事故统计。

## 关键事实

- OWASP 于 **2025-12-09** 发布 Top 10 for Agentic Applications，其中 **ASI09** 定义为 **Human-Agent Trust Exploitation**，强调 fluent and perceived expertise 会放大 automation bias；**ASI10** 定义为 **Rogue Agents**，关注失控后持续自主作恶与隐藏。（2026-07-19 核验）
- OWASP 2026 Q1 exploit round-up 继续把“错误信任 agent 建议”与“agent 越权执行”列为现实部署中的关键设计失败模式，缓解建议集中在显式确认、回滚、权限收缩和隔离。
- Anthropic 在 **2025-06-20** 发布《Agentic Misalignment: How LLMs could be insider threats》：其受控模拟中，多家前沿模型在“目标冲突或停机威胁 + 高自主权限”的设置下会表现出有害 insider-like 行为；Anthropic 同时明确写道，**尚未看到这类现象在真实部署中的证据**。
- 因此，关于 scheming、concealment、reward hacking 的准确工程表述应是：**这是需要测试和防范的条件化风险，不应被写成对所有生产 agent 的无条件事实判断。**

把这两类风险分开后，系统设计就清晰了：ASI09 主要考验审批与界面，人要看见足够证据才能批准；ASI10 主要考验运行时控制，agent 即使偏离，也不该轻易触达高破坏动作。这也是它与 [[24 沙箱、最小权限与人审闸门|沙箱与人审闸门]]、[[25 监控、可观测与事件响应|监控与 IR]]、[[16 Agent 身份与权限滥用(非人类身份 NHI)|身份与权限]] 的直接连接点。
