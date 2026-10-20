"""TodoListMiddleware 演示：让 Agent 先列待办，再按待办逐步修复代码。

案例：
1. 在 demo_workspace 中准备一个有多个 bug 的 my_add.py 和对应测试；
2. 给 Agent 注册文件读写、运行测试等工具，并挂载 TodoListMiddleware；
3. 让 Agent 修复 my_add.py。

TodoListMiddleware 的作用：
- 自动向 Agent 注入名为 write_todos 的工具；
- 向系统提示词追加「如何拆解任务、如何更新待办」的说明；
- 在 Agent 状态中维护 todos 字段，保存最新的待办列表。

运行方式（在项目根目录，使用项目虚拟环境）：
    .venv/bin/python ch08/todolist/main.py

输出只保留三条主线，方便观察中间件的作用：
- 每次 write_todos 之后发生变化的待办；
- Agent 调用了哪些其他工具；
- 最终 state['todos'] 和回答。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# demo_workspace 每次运行都会重新生成，保证 bug 一定存在，方便反复演示
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "demo_workspace"

# 故意写错的源码：三个函数各有一个 bug，让任务真正需要多步完成
BUGGY_MY_ADD = '''def add(a: int, b: int) -> int:
    """返回两个整数的和"""
    return a - b


def multiply(a: int, b: int) -> int:
    """返回两个整数的积"""
    return a + b


def is_even(n: int) -> bool:
    """判断一个整数是否为偶数"""
    return n % 2 == 1
'''

TEST_MY_ADD = '''from my_add import add, is_even, multiply


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(10, -5) == 5


def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0


def test_is_even():
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True
'''

STATUS_ICON = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


def _safe_path(relative_path: str) -> Path:
    """把相对路径解析到工作区内，防止访问工作区外的文件。"""
    workspace_root = WORKSPACE.resolve()
    target = (WORKSPACE / relative_path).resolve()
    if not target.is_relative_to(workspace_root):
        raise ValueError("只允许访问 demo_workspace 内的文件")
    return target


@tool
def list_files(path: str = ".") -> str:
    """列出工作区指定目录下的文件和子目录，path 必须是相对路径。"""
    try:
        target = _safe_path(path)
    except ValueError as exc:
        return f"错误：{exc}"

    if not target.exists():
        return f"错误：目录不存在: {path}"
    if not target.is_dir():
        return f"错误：不是目录: {path}"

    items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not items:
        return f"目录为空: {path}"

    lines = []
    workspace_root = WORKSPACE.resolve()
    for item in items:
        relative = item.relative_to(workspace_root).as_posix()
        kind = "[DIR]" if item.is_dir() else "[FILE]"
        lines.append(f"{kind} {relative}")
    return "\n".join(lines)


@tool
def read_file(path: str) -> str:
    """读取工作区中的文本文件内容，path 必须是相对路径。"""
    try:
        file_path = _safe_path(path)
    except ValueError as exc:
        return f"错误：{exc}"

    if not file_path.exists():
        return f"错误：文件不存在: {path}"
    if not file_path.is_file():
        return f"错误：不是文件: {path}"
    return file_path.read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """把内容写入工作区中的文本文件，path 必须是相对路径。"""
    try:
        file_path = _safe_path(path)
    except ValueError as exc:
        return f"错误：{exc}"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"已写入文件: {path}"


@tool
def run_tests() -> str:
    """在工作区运行测试，返回是否通过以及 pytest 的完整输出。"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (
            f"returncode={result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
    except Exception as exc:  # noqa: BLE001 - 工具应把异常转换成文本返回给模型
        return f"运行测试失败: {exc}"


def prepare_workspace() -> None:
    """重新生成演示用的工作区，确保每次运行都从同一组 bug 开始。"""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "my_add.py").write_text(BUGGY_MY_ADD, encoding="utf-8")
    (WORKSPACE / "test_my_add.py").write_text(TEST_MY_ADD, encoding="utf-8")


def _message_text(content: Any) -> str:
    """兼容 str 和 content blocks 两种消息内容格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def format_tool_call(tool_call: dict[str, Any]) -> str:
    """把工具调用格式化成一行，避免把大段文件内容打印出来。"""
    args = tool_call.get("args") or {}
    if not args:
        return tool_call["name"]

    parts = []
    for key, value in args.items():
        # 文件内容是噪音，只保留文件名等关键参数
        if key == "content":
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"{key}={text!r}")
    return f"{tool_call['name']}({', '.join(parts)})"


def print_todo_changes(
    todos: list[dict[str, Any]],
    index: int,
    previous: list[dict[str, Any]],
) -> None:
    """只打印相对上一次新增或状态变化的待办，避免整张表反复刷屏。"""
    print(f"\n[待办更新 {index}]")
    printed = False
    for position, todo in enumerate(todos):
        old = previous[position] if position < len(previous) else None
        if (
            old is not None
            and old.get("content") == todo.get("content")
            and old.get("status") == todo.get("status")
        ):
            continue
        status = todo.get("status", "pending")
        print(f"  {STATUS_ICON.get(status, '[?]')} {todo.get('content', '')}")
        printed = True
    if not printed:
        print("  (无变化)")


def print_final_todos(todos: list[dict[str, Any]]) -> None:
    """打印 state['todos'] 中的最终待办。"""
    print("\n最终待办 (state['todos']):")
    for todo in todos:
        status = todo.get("status", "pending")
        print(f"  {STATUS_ICON.get(status, '[?]')} {todo.get('content', '')}")

    completed = sum(1 for todo in todos if todo.get("status") == "completed")
    print(f"完成: {completed}/{len(todos)}")


def build_agent(model: Any):
    """创建挂载了 TodoListMiddleware 的代码修复 Agent。"""
    return create_agent(
        model=model,
        # 注意：这里没有 write_todos，它由 TodoListMiddleware 自动注入
        tools=[list_files, read_file, write_file, run_tests],
        # 挂载后，Agent 获得 write_todos 工具，并产生 state["todos"]
        middleware=[TodoListMiddleware()],
        system_prompt=(
            "你是一个严谨的代码修复助手。\n"
            "遇到需要多个步骤才能完成的任务时，必须先调用 write_todos 制定计划；"
            "每完成一步，立刻调用 write_todos 更新该步骤的状态，然后再继续下一步。\n"
            "修复完成后，用 run_tests 验证结果。\n"
            "最终回答只写 2-3 句话：修复了什么、测试是否通过；"
            "不要分点、不要代码块、不要复述待办列表。"
        ),
    )


def run_demo() -> None:
    load_dotenv(override=True)

    model = init_chat_model(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model="deepseek:deepseek-flash",
    )

    prepare_workspace()
    agent = build_agent(model)

    print("=" * 64)
    print("TodoListMiddleware Demo")
    print("用法: create_agent(..., middleware=[TodoListMiddleware()])")
    print("作用: 自动注入 write_todos 工具，并把待办保存到 state['todos']")
    print("=" * 64)
    print("任务: 修复 demo_workspace/my_add.py 中的多个 bug")

    question = (
        "工作区中的 my_add.py 有多个测试不通过，请定位并修复其中的 bug。"
        "第一步必须先调用 write_todos 制定修复计划（至少拆成 3 步），"
        "在制定计划之前不要调用其他工具；"
        "之后按计划逐步执行，每完成一步都要调用 write_todos 更新该步骤的状态。"
        "要求：不要修改 test_my_add.py，只修复 my_add.py，最后运行测试确认全部通过。"
    )

    final_todos: list[dict[str, Any]] = []
    final_answer = ""
    todo_updates = 0
    previous_todos: list[dict[str, Any]] = []
    planned_todos: list[dict[str, Any]] = []

    # stream_mode="updates" 按节点返回增量更新：
    # - 模型节点：新增的 AIMessage，其中包含本轮的工具调用；
    # - 工具节点：工具执行结果，以及 write_todos 通过 Command 更新的 todos。
    for chunk in agent.stream(
        {"messages": [HumanMessage(question)]},
        config={"recursion_limit": 50},
        stream_mode="updates",
    ):
        for _node, update in chunk.items():
            if not isinstance(update, dict):
                continue

            for message in update.get("messages") or []:
                if isinstance(message, AIMessage):
                    for tool_call in message.tool_calls:
                        if tool_call["name"] == "write_todos":
                            # 直接展示 write_todos 的参数，也就是 Agent 写下的新待办
                            todos = (tool_call.get("args") or {}).get("todos")
                            if isinstance(todos, list):
                                todo_updates += 1
                                print_todo_changes(todos, todo_updates, previous_todos)
                                previous_todos = todos
                                planned_todos = todos
                        else:
                            print(f"[工具] {format_tool_call(tool_call)}")

                    text = _message_text(message.content)
                    if text and not message.tool_calls:
                        final_answer = text

                elif isinstance(message, ToolMessage):
                    if getattr(message, "status", "success") == "error":
                        error = _message_text(message.content)[:200]
                        print(f"[工具错误] {message.name}: {error}")

            # 工具节点返回的 todos 才是真正写入 Agent 状态的值
            todos = update.get("todos")
            if todos is not None:
                final_todos = todos

    print("\n" + "=" * 64)
    if final_todos:
        print_final_todos(final_todos)
    elif planned_todos:
        print("结果: 生成了计划，但没有从 state['todos'] 收到更新")
    else:
        print("结果: Agent 没有调用 write_todos（任务可能过于简单）")

    if final_answer:
        print(f"\n最终回答:\n{final_answer}")


if __name__ == "__main__":
    run_demo()
