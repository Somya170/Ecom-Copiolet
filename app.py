import uuid
from typing import Any, Dict, List

import chainlit as cl
from langgraph.types import Command

from graph import get_copilot_graph

WELCOME_MESSAGE = """###  Internal E-Commerce Operations Copilot

Ask me about live orders, stock, and customers (SQLite), or historical sales and
category performance (DuckDB). I generate SQL automatically and route it to the
right engine.

Any write operation or unbounded analytical scan will require your approval before
it runs — you'll see exactly what will change before anything happens.

**Try asking:**
- *"What is the current stock for Fitness Smartwatch?"*
- *"Show me total revenue by category for the last quarter."*
- *"Mark all pending orders older than 20 days as cancelled."*
"""


def _format_result_table(columns: List[str], rows: List[Any]) -> str:
    if not columns or not rows:
        return ""
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, divider] + body_lines)


def _build_approval_card(payload: Dict[str, Any]) -> str:
    engine_label = (
        "SQLite (Live Transactional DB)"
        if payload.get("engine") == "sqlite"
        else "DuckDB (Historical Analytics DB)"
    )
    reasons = payload.get("risk_reasons") or []
    reasons_md = "\n".join(f"- {reason}" for reason in reasons)
    impact_summary = payload.get("impact_summary", "").strip()
    sql_query = payload.get("sql", "")
    code_fence = "```"

    return (
        f"### ⚠️ Approval Required — {engine_label}\n\n"
        f"**Plain-English Impact Summary:**\n\n"
        f"{impact_summary}\n\n"
        f"**Why this was flagged:**\n\n"
        f"{reasons_md}\n\n"
        f"<details>\n"
        f"<summary>🔍 View technical SQL</summary>\n\n"
        f"{code_fence}sql\n"
        f"{sql_query}\n"
        f"{code_fence}\n\n"
        f"</details>\n\n"
        f"Please review the impact above before approving this operation."
    )


async def _run_graph(input_or_command: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    graph = await get_copilot_graph()
    result = await graph.ainvoke(input_or_command, config=config)
    return result


async def _handle_graph_result(result: Dict[str, Any]) -> None:
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value
        cl.user_session.set("awaiting_approval", True)

        actions = [
            cl.Action(
                name="confirm_apply",
                label="✅ Confirm & Apply Changes",
                payload={},
            ),
            cl.Action(
                name="cancel_operation",
                label="❌ Cancel Operation",
                payload={},
            ),
        ]
        card_content = _build_approval_card(payload)
        await cl.Message(content=card_content, actions=actions).send()
        return

    cl.user_session.set("awaiting_approval", False)
    final_message = result.get("final_message", "Done.")
    execution_result = result.get("execution_result") or {}
    table_md = _format_result_table(
        execution_result.get("columns", []), execution_result.get("rows", [])
    )

    content = final_message
    if table_md:
        content = f"{final_message}\n\n{table_md}"

    await cl.Message(content=content).send()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    cl.user_session.set("awaiting_approval", False)
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    if cl.user_session.get("awaiting_approval"):
        await cl.Message(
            content="⏳ Please confirm or cancel the pending operation above before sending a new request."
        ).send()
        return

    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    thinking_msg = cl.Message(content="🧠 Analyzing your request and generating SQL...")
    await thinking_msg.send()

    try:
        result = await _run_graph({"user_query": message.content}, config)
    except Exception as exc:  # noqa: BLE001
        thinking_msg.content = f"**Something went wrong.**\n\n`{exc}`"
        await thinking_msg.update()
        return

    await thinking_msg.remove()
    await _handle_graph_result(result)


@cl.action_callback("confirm_apply")
async def on_confirm_apply(action: cl.Action):
    await action.remove()
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    processing_msg = cl.Message(content="⏳ Applying approved changes...")
    await processing_msg.send()

    try:
        result = await _run_graph(Command(resume={"approved": True}), config)
    except Exception as exc:  # noqa: BLE001
        processing_msg.content = f"**Execution failed after approval.**\n\n`{exc}`"
        await processing_msg.update()
        cl.user_session.set("awaiting_approval", False)
        return

    await processing_msg.remove()
    await _handle_graph_result(result)


@cl.action_callback("cancel_operation")
async def on_cancel_operation(action: cl.Action):
    await action.remove()
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await _run_graph(Command(resume={"approved": False}), config)
    except Exception as exc:  # noqa: BLE001
        await cl.Message(content=f"**Error while cancelling.**\n\n`{exc}`").send()
        cl.user_session.set("awaiting_approval", False)
        return

    await _handle_graph_result(result)