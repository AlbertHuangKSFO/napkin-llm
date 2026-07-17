[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]] 把 [[03 Agent 核心循环|Agent 核心循环]] 接到 GUI：观察当前界面或页面结构，选择一个可逆动作，执行后再次观察并验证状态。它是没有 API 时的适配层，不是 API/MCP 的替代品。

## 直觉 / 生活类比

浏览器自动化像两种导航：DOM/可访问性树路线拿到“门牌号”，能直接走到按钮；视觉路线只看街景，需要凭像素找到门。前者通常更确定，后者覆盖范围更广——远程桌面、Canvas 和原生应用没有可靠 DOM，仍需要截图与坐标。

但“有 DOM 就一定能点到”是误解。Canvas 绘制内容只是位图，通常没有逐个对象的语义节点；跨源或复杂嵌套 iframe 需要切换 frame；Shadow DOM 的可见性又受开放/封闭根、驱动和自动化库实现影响。W3C WebDriver 标准定义了 frame 与 shadow-root 操作，**不保证每个 driver、页面架构和安全边界都以同样方式暴露它们**。[WebDriver，2025](https://www.w3.org/TR/webdriver2/)

## 小数字手算

一个表单任务有 $6$ 个普通页面，每页“读结构一次、动作后验证一次”，另有 $3$ 个 Canvas 控件必须视觉确认。观察次数为：

$$
6\times2+3=15
$$

若所有页面都只靠截图且每步都重试一次，最坏为 $15\times2=30$ 次观察。这个例子不是 token 价格估算；它说明可靠性预算应按**可观察状态转换数**来算。DOM/a11y 让普通页面的动作更可验证，视觉回退覆盖没有语义结构的 $3$ 个控件。

## 公式推导

把浏览器状态记为 $x_t$，观察器为 $o$，动作执行器为 $e$，目标谓词为 $G$：

$$
y_t=o(x_t),\quad a_t=\pi(y_t),\quad x_{t+1}=e(x_t,a_t)
$$

正确的终止条件不是“已经点击”，而是：

$$
\operatorname{stop}\iff G\bigl(o(x_{t+1})\bigr)=\text{true}
$$

因此付款、删除、提交等动作应拆为“预览 → 人审/策略检查 → 执行 → 读取确认页”。页面文字、Agent Card、下载文件都应视为不可信输入，避免 [[05 Prompt Injection 提示注入|提示注入]] 把观察内容伪装成指令。

安全分层见 [[AI 安全/01 AI 安全总览与三层栈]]。

## 手绘图

![[计算机使用与浏览器 Agent-视觉路vsDOM路.png]]

## 可运行代码 / 配置

安装并运行：`python -m pip install playwright && playwright install chromium && python browser_safe.py`。示例只打开 `example.com` 并验证标题，不会登录或提交表单。

```python
# browser_safe.py
import asyncio
from playwright.async_api import async_playwright

async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://example.com", wait_until="domcontentloaded")

        # ❌ Canvas 内画出的“按钮”不一定是 DOM button；不要盲猜选择器。
        # await page.locator("canvas button").click()

        # ✅ 对可访问的真实元素，先定位、再检查、最后做可逆动作。
        heading = page.get_by_role("heading", name="Example Domain")
        await heading.wait_for()
        assert await heading.text_content() == "Example Domain"
        print("验证通过：", await page.title())
        await browser.close()

asyncio.run(main())
```

iframe 需要明确切换上下文，例如 `page.frame_locator("iframe[name='payment']")`；开放 Shadow DOM 可由部分驱动穿透，而封闭根、跨源/动态 iframe、浏览器策略与 driver 版本都必须实测。Playwright 的 frame API 是一个库级例子，并不改变这些边界。[Playwright Frames 文档](https://playwright.dev/docs/frames)

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：DOM 路和视觉路如何取舍？**

答：优先 DOM/a11y，因为元素定位、等待条件和状态校验更确定；遇 Canvas、原生 GUI 或无法可靠暴露的元素时，以截图/坐标作为受限回退。混合不是“多看一眼”，而是对每个动作选择可验证的观察通道。

**Q：为什么 iframe 和 Shadow DOM 是风险点？**

答：它们引入浏览上下文与封装边界。WebDriver 有相应标准命令，但实际可访问性仍受同源策略、嵌套层级、开放/封闭 shadow root 和 driver 实现影响；必须在目标站点与目标浏览器组合上验证。

**Q：高风险 GUI 动作的最低护栏？**

答：隔离 profile/容器、最小权限、允许动作清单、关键步骤人工确认、执行后读取独立成功证据，以及可回放的截图/DOM/动作日志。

## 关键事实

- **Canvas 不是语义 DOM**：`<canvas>` 自身是位图绘制面，画出的对象不会自动成为可访问的 HTML 元素；应提供替代内容或走视觉验证。[MDN Canvas，更新于 2025](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/canvas)
- **标准与驱动需区分**：WebDriver 的正式规范包含“切换 frame”和“获取 Shadow Root”命令（现行文档遵循 2025 W3C Process），实际支持范围依赖具体浏览器与驱动。[W3C WebDriver，2025](https://www.w3.org/TR/webdriver2/)
- **接口优先**：有 [[15 Function Calling 工具调用|Function Calling]]、MCP 或稳定 HTTP API 时，应优先调用接口；GUI 自动化更慢、更脆弱，也更需要 [[24 沙箱、最小权限与人审闸门|人审护栏]]。
- 代码仓库的文本—终端闭环见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]；深度网页证据任务见 [[29 Deep Research Agent|Deep Research Agent]]。
