[[25 Agent Skills(SKILL.md)|Agent Skills]] 的本质：把某类任务的程序性知识、脚本和参考资料封装成一个可版本化目录；agent 先看到它适用的简短描述，真正匹配任务时才加载详细指令与按需资源。

## 直觉：目录卡、操作手册与工具箱

医院前台的科室目录只写「皮肤科：处理皮疹和皮肤病」；挂号后医生才拿到完整诊疗路径，必要时再取检验单和器械。Skill 的 `name`/`description` 是目录卡，`SKILL.md` 正文是操作手册，`scripts/`、`references/`、`assets/` 是工具箱。它教 agent **怎么组织已有能力**，并不自动授予数据库、浏览器或部署权限。

因此它与 [[17 MCP 模型上下文协议|MCP]]、[[16 工具设计与工具层|工具层]]互补：MCP/工具定义「可调用什么」，Skill 记录「面对某种任务应按什么流程调用、验证和交付」。一个 Skill 可以指向 MCP 工具，但运行时仍须由 [[23 Agent Harness 概览|harness]] 的权限策略审查。

## 小数字手算：渐进式披露仍有线性发现成本

设已安装 $N=50$ 个 skills，平均 description 为 $d=40$ token；本次激活 1 个 skill，正文 $b=800$ token、一个参考文件 $r=2{,}000$ token。发现层的成本是：

$$
C_{\text{discover}}=N\bar d=50\times40=2{,}000\text{ tokens}
$$

若这次确实需要正文和一个引用，当前任务的总加载量近似：

$$
C_{\text{task}}=N\bar d+b+r=2{,}000+800+2{,}000=4{,}800
$$

这比把每个 skill 的所有内容一开始都塞入上下文小得多；但当 $N$ 从 50 翻到 100，光 description 就从 2,000 翻到 4,000。故发现层是 **$O(N)$，不是常数**。这些 token 数仅为算例；不同 client 的注入方式、缓存和资源大小会改变实测成本。

## 推导：三层加载与可移植边界

设第 $i$ 个 skill 的描述、正文和可选资源大小为 $(d_i,b_i,r_i)$，激活集合为 $A$，实际读取资源集合为 $R_i$。渐进式披露近似为：

$$
C=\sum_{i=1}^{N}d_i+\sum_{i\in A}b_i+\sum_{i\in A}\sum_{j\in R_i}r_{ij}
$$

相比完全预加载：

$$
C_{\text{eager}}=\sum_{i=1}^{N}(d_i+b_i+\sum_j r_{ij})
$$

它把昂贵正文和资源从「每次都付」改为「命中时才付」，但第一项仍随安装数线性增长。因此好的 description 要短、判别性强；大的 API 文档、模板和可执行脚本应下沉，避免每次激活都读进一大段无关内容。

[Agent Skills 官方概览](https://agentskills.io/home)定义三阶段：启动时读 `name`/`description`，匹配时读完整 `SKILL.md`，执行时再按需读引用文件或运行脚本。不同 client 对路径、额外 frontmatter、缓存和授权的支持并不完全相同，写可移植 skill 时应以 [规范](https://agentskills.io/specification) 为基线，并把 client 特性隔离为可选扩展。

![[Agent Skills(SKILL.md)-目录结构.png]]

![[Agent Skills(SKILL.md)-渐进式披露.png]]

## 可运行代码：不要在发现阶段加载全部正文

❌ 朴素加载会让每个任务都付出全部手册成本。

```python
def eager_load(skills):
    return "\n".join(skill["description"] + skill["body"] for skill in skills)
```

✅ 此最小示例先只保留 description，再按匹配读取正文；真实 harness 还要进行路径校验、版本锁定、脚本授权与内容审计。

```python
skills = [
    {"name": "pdf", "description": "填写 PDF 表单", "body": "inspect -> map -> write"},
    {"name": "release", "description": "按发布清单部署", "body": "test -> approve -> deploy"},
]

def discover(skills):
    return [{"name": s["name"], "description": s["description"]} for s in skills]

def activate(skills, name):
    return next(s["body"] for s in skills if s["name"] == name)

cards = discover(skills)
assert cards == [{"name": "pdf", "description": "填写 PDF 表单"},
                 {"name": "release", "description": "按发布清单部署"}]
assert activate(skills, "pdf") == "inspect -> map -> write"
print(cards)
```

一个可移植目录的最小形态是：

```text
invoice-fill/
├── SKILL.md        # frontmatter + 主流程
├── references/     # 长文档，正文按需指向
├── scripts/        # 显式执行才运行的代码
└── assets/         # 模板、样例、静态资源
```

## 设计准则与安全边界

- **description 写触发条件，不写广告语**：例如「当用户要将结构化数据填入 PDF 表单时使用」，让模型能判断相关性。
- **正文留主干，细节下沉**：正文写决策点、输入输出、验收与风险；长参考资料通过相对路径按需加载。
- **脚本是供应链代码**：安装第三方 Skill 不等于可信；锁定来源/版本，审阅 diff，在 sandbox 里以最小权限运行，敏感外部动作另设人审。
- **不要把 Skill 当实时数据库**：频繁变化的事实应经检索、工具或 [[19 Agent 记忆系统|记忆系统]]获取；Skill 适合稳定的流程知识。

⚠️ **开放标准不等于生态已经稳定。** 规范提供互操作的共同起点，但各 client 的发现目录、扩展字段、脚本执行模型、权限与管理面仍在演进；发布前要在目标 client 上做安装、触发、拒绝权限和升级测试。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**问：Skill 与 MCP 的区别？**

答：Skill 是可复用的流程知识和资源包；MCP/工具是访问外部数据与动作的接口。Skill 可以教 agent 何时、如何调用工具，但不能替代认证、参数校验和权限控制。

**问：渐进式披露为什么仍可能变慢？**

答：每个可发现 skill 的 description 都要被枚举或注入，成本随 $N$ 线性增长；description 太长、skills 太碎或无关资源在正文里都会增加延迟/上下文压力。要测量真实 token 与召回质量，不声称常数成本。

**问：如何验证一个 Skill？**

答：至少覆盖四类：匹配任务能触发；不相关任务不误触发；脚本/引用路径可用且权限拒绝有效；升级后输出仍通过领域验收。对高风险 Skill，再加供应链审查与 sandbox 逃逸测试。

## 关键事实

- [Anthropic，2025-10-16，后续更新](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)：介绍由 `SKILL.md`、脚本和资源组成的 Agent Skills，并在页面更新中注明 **2025-12-18** 已将 Agent Skills 作为跨平台可移植的开放标准发布。
- [Claude by Anthropic，2025-12-18](https://claude.com/blog/organization-skills-and-directory)：官方产品公告将 Agent Skills 公开为开放标准，并讨论组织级管理与目录；这是日期与开放标准状态的一手来源。
- [Agent Skills specification](https://agentskills.io/specification)：要求 skill 是带 `SKILL.md` 的目录，核心元数据至少包括 `name` 和 `description`；生态仍在形成，具体 client 能力要以其文档与测试为准。
