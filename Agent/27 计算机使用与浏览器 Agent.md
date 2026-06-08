[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]] 让 agent 像人一样操作 GUI 与浏览器:看一眼屏幕(截图或 DOM)→ 决定点哪里、敲什么字 → 执行 → 再看一眼,如此闭环,直到任务完成。它本质就是把 [[03 Agent 核心循环|Agent 核心循环]] 的"行动"接到了真实的鼠标键盘上,而不是接到一组干净的 API。

## 本质:为什么要让模型去"点屏幕"

绝大多数软件没有给 AI 留 API。报销系统、老旧 ERP、银行网银、政府网站、桌面客户端——它们只有一套给人用的图形界面。如果想让 agent 自动完成"在这个内网系统里提交一张报销单",你要么等它出 API(永远不会),要么让 agent 学会像人一样操作这套 GUI。计算机使用 / 浏览器 Agent 就是后一条路:把"人能看屏幕、能动鼠标键盘"这件事交给模型。

这条路的代价很大(慢、贵、脆),但它的价值在于**通用性**:只要是人能在屏幕上完成的事,理论上 agent 都能做,不需要每个系统单独写集成。这跟 [[15 Function Calling 工具调用|Function Calling 工具调用]] / [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 形成互补——后者是"系统主动暴露干净接口",前者是"系统什么都不暴露时的兜底"。能用 API 就别用 GUI:GUI 操作是工具层的最后选项,见 [[16 工具设计与工具层|工具设计与工具层]]。

## 机制:截图 → 动作 闭环(分步)

![[计算机使用与浏览器 Agent.png]]

一个标准的 computer-use 回合按下面四步走,然后循环:

1. **观察(Observation)**:抓取当前环境状态。视觉路线抓一张**屏幕截图**(PNG,常压到固定分辨率如 1280×800);DOM 路线抓**精简后的 DOM 或可访问性树(accessibility tree)**,把可交互元素标号。这一步决定了模型"看到什么"。
2. **推理 + 出动作(Action)**:多模态模型读入观察,推理出下一步,输出一个**结构化动作**。视觉路输出的是坐标动作,例如 `click(x=412, y=188)`、`type("zhangsan")`、`scroll(dy=300)`、`key("Enter")`、`wait()`;DOM 路输出的是 `click(element_index=7)` 或选择器动作。
3. **执行(Execution)**:执行器把抽象动作翻译成真实事件。视觉路把坐标喂给操作系统级的鼠标/键盘注入(或虚拟机里的 xdotool 之类);DOM 路调用 Playwright / Puppeteer 的 `page.click()` / `page.fill()`。
4. **反馈**:动作执行后页面/窗口变了,**重新截图回灌**。注意这一步无法省略——GUI 状态不可预测(弹窗、加载、跳转),agent 必须每步都重新"看",不能盲操作。

整个循环和 [[09 ReAct|ReAct]] 同构:观察=截图/DOM,思考=模型推理,行动=GUI 动作,只是 observation 从文本变成了图像/页面结构。失败时配合 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 重看截图纠错。

### 视觉路 vs DOM 路:两条技术路线

![[计算机使用与浏览器 Agent-视觉路vsDOM路.png]]

这是这个领域最核心的分歧,务必分清:

- **视觉 / 像素路(Computer Use 流派)**:输入纯截图,输出坐标。代表是 Anthropic Computer Use、OpenAI Operator/CUA、Google Project Mariner。优点是**通用**——桌面 app、远程桌面、Canvas 游戏都能操作,不依赖能不能拿到 DOM;缺点是**慢且贵**(每步一张大图,视觉 token 成本高)、坐标容易点偏、对分辨率和小字体敏感。
- **DOM / 可访问性树路(浏览器 Agent 流派)**:输入精简后的 DOM 结构(把按钮、输入框、链接抽出来编号),输出元素 index 或选择器。代表是 browser-use、Stagehand。优点是**快、便宜、定位准**(直接拿元素句柄,不靠猜坐标),还能把成功的动作脚本缓存复用;缺点是**只限浏览器**(桌面 app 无效),且对 Canvas、iframe、Shadow DOM "失明",重 JS 页面的 DOM 噪声大。

实战里两条路常**混合**:DOM 为主拿结构,模型看不懂或元素拿不到时,回退到截图用视觉定位(hybrid)。browser-use 等库就走这种 DOM + 截图标注的混合方案。

**成本手算(为什么 DOM 路便宜得多)。** 视觉 token 量级粗估按 $\frac{w\times h}{750}$ 算,一张 $1280\times800$ 截图 $\approx\frac{1024000}{750}\approx1365$ 视觉 token;DOM 路一段精简后的可访问性树约 $600$ 文本 token。**单步**视觉路约是 DOM 路的 $1365/600\approx2.3$ 倍观察成本。放到一个 $30$ 步的「订机票」任务上:视觉路光观察就 $1365\times30\approx4.1\times10^4$ token,DOM 路 $600\times30\approx1.8\times10^4$ token——单这一项差 $2.3\times$;再叠加视觉 token 单价通常更高、且大图还拖慢首 token 延迟,整体差距进一步放大。这就是规模化首选「**DOM 为主、视觉兜底**」的硬账。

## 来源:产品与时间线

- **Anthropic Computer Use**(2024-10):Claude 3.5 Sonnet 首次开放 computer-use 能力的公开 API,纯视觉路线——模型看截图、输出坐标动作。是第一个把"通用桌面操作"做成产品级 API 的。后续演进为 Claude 的屏幕交互底座(消费端 Claude Cowork 即建于其上)。
- **OpenAI Operator / CUA**(2025-01):OpenAI 的 Computer-Using Agent,随 ChatGPT Pro 上线,跑在云端虚拟浏览器里,主打消费级网页任务(订餐、订票、填表)。
- **Google Project Mariner**(2024-12 起):Google 的 computer-use 入口,深度集成 Chrome,偏浏览器原生任务,支持云端虚拟机上并发多任务。
- 三家在 2025-2026 形成"agent 军备竞赛":都在比 WebVoyager / WebArena / OSWorld 等基准上的成功率,也都在补安全护栏。

## 可跑最小代码:browser-use(DOM 路)

```python
# pip install browser-use && playwright install chromium
import asyncio
from browser_use import Agent
from langchain_openai import ChatOpenAI

async def main():
    agent = Agent(
        task="打开 news.ycombinator.com,找出当前排名第一的帖子标题并返回",
        llm=ChatOpenAI(model="gpt-4o"),   # 任意多模态/强 LLM
    )
    result = await agent.run()   # 内部:截图+DOM → 模型出动作 → Playwright 执行 → 循环
    print(result)

asyncio.run(main())
```

### 视觉路伪码(Computer Use 风格)

```python
# 视觉路:你自己拿截图、把动作回放给虚拟机
screenshot = grab_screen()                 # 1. 观察:截图
while not done:
    action = model.act(                     # 2. 模型出坐标动作
        system="你能操作电脑。每步先看图再给一个动作。",
        image=screenshot,
        tools=["click", "type", "scroll", "key", "screenshot"],
    )
    if action.name == "click":
        os_mouse_click(action.x, action.y)  # 3. 执行:注入真实点击
    elif action.name == "type":
        os_keyboard_type(action.text)
    screenshot = grab_screen()              # 4. 反馈:重新截图回灌
```

关键点:模型每一步都**只能基于最新截图**做决策,所以循环里必须不断重抓屏幕。

## 对比表

| 维度 | 视觉 / 像素路 | DOM / a11y 树路 |
|---|---|---|
| 输入 | 屏幕截图(图像) | 精简 DOM / 可访问性树(文本) |
| 输出动作 | 坐标 `click(x,y)`、`type` | 元素 index / 选择器动作 |
| 适用范围 | 任意 GUI:桌面+浏览器+游戏 | 仅浏览器 |
| 速度 / 成本 | 慢、贵(大图 token) | 快、便宜(文本) |
| 定位精度 | 易点偏,分辨率敏感 | 准,拿元素句柄 |
| 盲区 | 几乎无(看得见就能操作) | Canvas / iframe / Shadow DOM |
| 代表 | Anthropic Computer Use、Operator、Mariner | browser-use、Stagehand |
| 底层执行 | OS 级鼠标键盘注入 / 虚拟机 | Playwright / Puppeteer |

## 何时用 / 坑

**何时用**:目标系统**没有 API**、且任务能由人在屏幕上完成时;一次性自动化、跨多个无接口系统的流程、UI 回归测试。如果系统有 API 或 MCP,优先走 [[15 Function Calling 工具调用|Function Calling 工具调用]],别用 GUI——GUI 慢一两个数量级。

**坑(逐条,这是这个方向最痛的地方)**:

- **慢**:每步要截图+推理+执行,一个"订机票"任务可能几十步、几分钟,成本和延迟都高。优化思路见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]:DOM 优先省截图、缓存动作脚本、能并行的子任务并行。
- **脆**:页面改版、A/B 实验、弹窗、Cookie 横幅、验证码,任何意料外的 UI 都会让 agent 卡死或乱点。需要强 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 和超时/重试护栏。
- **易错**:视觉路坐标点偏、DOM 路元素 index 错位,都会做出破坏性误操作。高风险动作(付款、删除、提交)应设确认关卡(此处涉及人在回路与 [[24 沙箱、最小权限与人审闸门|权限护栏]])。
- **安全风险大**:agent 操作真实账号、能点任意按钮,一旦被恶意页面诱导([[05 Prompt Injection 提示注入|提示注入]])就可能泄露凭据或执行危险操作。这块属于 [[01 AI 安全总览与三层栈|Agent 安全]]、[[24 沙箱、最小权限与人审闸门|权限护栏]]范畴,实践上要沙箱化、最小权限、敏感操作人工确认。
- **状态不可观测**:agent 看不到后台真正发生了什么(网络请求成没成),只能靠下一张截图推断,容易"以为成功了其实没成"。

## 关键事实

- WebVoyager(643 任务、15 个真实网站)上,browser-use 报告约 **89.1%** 成功率,Skyvern 2.0 约 **85.85%**;这类数字证明浏览器 Agent 在真实网页上已相当能打,但离"无人值守可靠"仍有距离。
- WebArena 是**自托管、可复现**的离线网页环境(812 任务,涵盖电商/论坛/CMS/地图等),专为可复现评测设计;WebVoyager 则跑在**真实在线网站**上。两者常一起用,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- 视觉路的瓶颈往往不是"看不懂",而是**坐标精度**和**token 成本**;DOM 路的瓶颈是**覆盖范围**(非浏览器、Canvas 失明)。
- 与 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]] 的区别:代码 Agent 操作的是文件系统+终端+测试(文本世界),浏览器 Agent 操作的是像素/DOM(视觉世界),但二者都是 agentic loop,且都重度依赖"执行→观察反馈→再行动"。

## 主流开源实现 / Python 库

> 均已 web 核验为 2026 年活跃项目。

- **browser-use**(`browser-use/browser-use`,pip `browser-use`):2026 年最火的浏览器 Agent 库之一(GitHub 50k+ 星),DOM+截图混合,给任意 LLM 套上完整浏览器操作循环,WebVoyager 约 89%。首选。
- **Playwright**(pip `playwright`,微软)/ **Puppeteer**(npm,Google):底层浏览器驱动,几乎所有浏览器 Agent 都建在它们之上;不带 AI,负责真实点击/输入/导航。
- **Skyvern**(`Skyvern-AI/skyvern`,pip `skyvern`):Playwright 扩展 + LLM + 计算机视觉,主打表单填写类工作流自动化,WebVoyager 约 85.85%。
- **Stagehand**(`browserbase/stagehand`,主要为 TypeScript SDK):Browserbase 出品,"确定性优先"——扩展而非替换 Playwright,可用自然语言也可写精确脚本,适合需要可维护、可缓存的生产流程。
- **LaVague**(`lavague-ai/LaVague`,pip `lavague`):自然语言网页自动化框架,处理跨布局的元素识别("点那个绿色按钮"它能找到)。
- **WebArena**(`web-arena-x/webarena`)/ **WebVoyager**:评测基准而非执行库——前者自托管可复现(812 任务),后者跑真实在线网站(643 任务),用于衡量浏览器 Agent 成功率。

链接:[[03 Agent 核心循环|Agent 核心循环]]、[[16 工具设计与工具层|工具设计与工具层]]、[[09 ReAct|ReAct]]、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]、[[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]、[[38 Agent 评估与可观测性|Agent 评估与可观测性]]、[[15 Function Calling 工具调用|Function Calling 工具调用]]、[[17 MCP 模型上下文协议|MCP 模型上下文协议]]。

## 工业界实践

浏览器/computer-use Agent 在 2026 年是「军备竞赛」级别的赛道,但生产落地有一条铁律:**它是没有 API 时的兜底,不是默认方案**——能走 API/MCP 一定走,GUI 操作慢一两个数量级。

**主流服务与产品(具体定位)。**
- **Anthropic Computer Use**(纯视觉):Claude 屏幕交互底座,消费端 Claude Cowork 建于其上;面向「任意 GUI 通用操作」。
- **OpenAI Operator / CUA**:云端虚拟浏览器,主打消费级网页任务(订餐订票填表)。
- **Google Project Mariner**:深度集成 Chrome,云端 VM 并发多任务。
- **Browserbase**(基础设施):托管的无头浏览器云,提供会话录制、stealth/反检测、代理池——生产部署浏览器 Agent 的「云端浏览器即服务」,Stagehand 即其官方 SDK。
- 开源执行层:**browser-use**(DOM+截图混合,WebVoyager ~89%,首选)、**Skyvern**(表单填写工作流,~85.85%)、**Stagehand**(确定性优先,可缓存脚本)、**Playwright/Puppeteer**(底层驱动)。

**典型生产架构。**
```
任务编排(agent loop)
  └── 浏览器会话(Browserbase / 本地 Playwright,隔离容器/VM)
        ├── 观察:DOM 精简 + a11y 树(主)→ 看不懂时回退截图(hybrid)
        ├── 动作:Playwright page.click/fill(DOM 路)或坐标注入(视觉路)
        ├── 护栏:高风险动作(付款/删除/提交)人审闸门
        └── 反检测/代理:stealth、住宅代理、验证码服务
  └── 可观测:每步截图 + DOM 快照 + 动作日志(轨迹回放)
```
**规模化、成本与延迟。**
- **成本结构**:视觉路每步一张大图 = 高视觉 token;DOM 路每步一段精简 DOM = 便宜得多。规模化首选 **DOM 为主、视觉兜底**。
- **延迟**:一个「订机票」任务可能几十步、几分钟。优化(见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]):DOM 优先省截图、**缓存成功的动作脚本**(Stagehand 的卖点:第一次用 LLM 推断、之后回放确定性脚本)、可并行子任务并行、降图像分辨率。
- **可靠性瓶颈**:真实生产成功率远低于 benchmark 的 ~89%,因为线上有反爬、登录态、A/B、验证码、风控。

**可观测与运维。** 必须做**全轨迹回放**:每步存截图 + DOM 快照 + 输出动作 + 执行结果,失败时能逐帧复盘「它当时看到什么、为什么点错」。监控指标:每任务步数、点击命中率、重试次数、人审触发率、单任务成本。

**踩坑与最佳实践。**
- **状态不可观测最坑**:agent 只能靠下一张截图推断后台成没成,容易「以为成功了其实没成」。对关键步骤加显式校验(等待特定元素出现/读取确认文案)。
- **脆**:Cookie 横幅、弹窗、验证码、改版任意一个都能卡死;配强 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] + 超时/重试 + 兜底退出。
- **安全是头号风险**:agent 操作真实账号、能点任意按钮,恶意页面可用 [[05 Prompt Injection 提示注入|提示注入]] 诱导泄露凭据/执行危险操作。生产硬性要求:沙箱化、最小权限([[24 沙箱、最小权限与人审闸门|权限护栏]])、敏感操作人审、独立的低权限浏览器 profile。
- **反检测的灰度**:住宅代理 + stealth 绕风控,但触及目标站点 ToS,生产要评估合规边界。

## 面试高频

**Q1:视觉/像素路 vs DOM/可访问性树路,区别与取舍?**
标准答:**视觉路**输入纯截图、输出坐标动作(`click(x,y)`),代表 Anthropic Computer Use / Operator / Mariner;优点是**通用**(桌面 app、远程桌面、Canvas 游戏都能操作,不依赖能否拿到 DOM),缺点是**慢、贵(大图视觉 token)、坐标易点偏、对分辨率敏感**。**DOM 路**输入精简 DOM/a11y 树、输出元素 index/选择器,代表 browser-use / Stagehand;优点是**快、便宜、定位准(拿元素句柄)、可缓存脚本**,缺点是**只限浏览器,且对 Canvas/iframe/Shadow DOM 失明**。实战常 **hybrid**:DOM 为主,拿不到时回退视觉。

**Q2:computer-use 的一个回合分哪几步?和 ReAct 什么关系?**
标准答:**观察(截图/DOM)→ 推理出结构化动作 → 执行(OS 注入 / Playwright)→ 反馈(重新截图回灌)**,然后循环。它和 [[09 ReAct|ReAct]] 同构:observation 从文本变成图像/页面结构,thought=模型推理,action=GUI 动作。
- 追问「为什么每步都要重新截图?」:GUI 状态不可预测(弹窗、加载、跳转),不能盲操作,必须每步重新「看」。

**Q3:既然有 API,为什么还要让 agent 去点屏幕?**
标准答:绝大多数软件(老 ERP、网银、政府/内网系统、桌面客户端)**没有给 AI 留 API**。GUI 路的价值是**通用性**——人能在屏幕上做的事 agent 都能做,无需为每个系统单独写集成。但代价是慢、贵、脆,所以**能用 API/MCP 就别用 GUI**,GUI 是工具层最后选项。

**Q4(陷阱):浏览器 Agent 报 89% 成功率,是不是已经能无人值守了?**
标准答:不是。89%(WebVoyager,browser-use)是**精选 benchmark** 上的数字;WebVoyager 跑真实在线网站、WebArena 是自托管可复现离线环境。真实生产有反爬/登录态/验证码/风控,成功率显著下降,且 11% 失败里可能含破坏性误操作,离「无人值守可靠」仍有距离。

**Q5:这类 Agent 的安全风险有哪些?怎么防?**
标准答:① **提示注入**——恶意页面文案诱导 agent 执行危险动作/泄露凭据;② 高风险动作误操作(付款/删除)。防御:沙箱化、最小权限、独立低权限账号、敏感操作人审闸门、把页面内容视为不可信输入。属 [[01 AI 安全总览与三层栈|Agent 安全]] / [[24 沙箱、最小权限与人审闸门|权限护栏]] 范畴。

**Q6:WebVoyager 和 WebArena 区别?**
标准答:**WebVoyager** 跑**真实在线网站**(643 任务、15 个站),贴近真实但不可复现;**WebArena** 是**自托管、可复现的离线环境**(812 任务,电商/论坛/CMS/地图),专为可复现评测设计。常一起用。

## 知识拓展

**为什么坐标精度是视觉路的真瓶颈。** 视觉路的失败往往不是「看不懂界面」,而是**把动作落到正确像素**——小字体、高分辨率缩放、动态布局都让 `click(x,y)` 点偏。这催生了 **set-of-marks(SoM)** 提示法:先给截图里的可交互元素叠加编号标注,让模型输出「点编号 7」而非裸坐标,把视觉路的定位问题转成离散选择问题——本质是借了 DOM 路「元素编号」的思路。browser-use 的 DOM+截图标注混合就吃这个红利。

**GUI grounding 与专用模型。** 2024-2025 出现一批专做「截图→坐标」的 grounding 模型(如 SeeClick、UGround、OS-Atlas 等学术工作),把通用 VLM 的弱定位能力专门强化;趋势是底层多模态模型直接内化更强的屏幕 grounding,减少对 SoM 标注的依赖。

**边界与反模式。**
- **反模式:有 API 还硬走 GUI**——慢一两个数量级、脆、贵,纯属自找麻烦。
- **反模式:不做轨迹记录**——GUI Agent 失败几乎无法离线复盘,必须存截图+DOM+动作。
- **反模式:把页面文本当可信指令**——这是提示注入的入口,页面内容一律视为不可信数据。
- **边界:状态不可观测是结构性缺陷**——agent 看不到后台真实结果,关键步骤必须加显式确认,不能假定「点了就成了」。

**前沿与时间线(带年份)。**
- **2024-10** Anthropic Computer Use 首个产品级通用桌面操作 API;**2024-12** Google Project Mariner;**2025-01** OpenAI Operator/CUA。
- **2024** OSWorld 基准发布,把评测从「纯浏览器」扩到「真实桌面操作系统任务」,是 computer-use(非纯浏览器)的标尺;与 WebArena/WebVoyager 共同构成评测三件套(见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]])。
- **2025-2026** 三大厂在 WebVoyager/WebArena/OSWorld 上的成功率竞赛持续,同时补安全护栏(防注入、沙箱、人审)。

延伸链接:[[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]](文本世界 vs 视觉世界的姊妹篇)、[[03 Agent 核心循环|Agent 核心循环]]、[[09 ReAct|ReAct]]、[[05 Prompt Injection 提示注入|Prompt Injection 提示注入]]、[[24 沙箱、最小权限与人审闸门|沙箱、最小权限与人审闸门]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]、[[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
