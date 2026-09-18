import operator
import os
import re
import sqlite3
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import aiosqlite
import duckdb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ecommerce_oltp.db")
DUCKDB_DB_PATH = os.getenv("DUCKDB_DB_PATH", "ecommerce_analytics.duckdb")
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "langgraph_checkpoints.db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RESULT_ROWS = 50

# --------------------------------------------------------------------------
# Database schema descriptions (grounding context for LLM)
# --------------------------------------------------------------------------
SQLITE_SCHEMA = """
SQLite database `ecommerce_oltp.db` (OLTP - live transactional data):

Table products (product_id INTEGER PK, name VARCHAR, category VARCHAR,
    unit_price FLOAT, stock_quantity INTEGER)

Table customers (customer_id INTEGER PK, name VARCHAR, email VARCHAR,
    city VARCHAR, tier VARCHAR CHECK IN ('REGULAR','GOLD','PLATINUM'))

Table orders (order_id INTEGER PK, customer_id INTEGER FK -> customers,
    order_date TIMESTAMP, order_status VARCHAR
    CHECK IN ('PENDING','PROCESSING','SHIPPED','CANCELLED'), total_amount FLOAT)

Table order_items (item_id INTEGER PK, order_id INTEGER FK -> orders,
    product_id INTEGER FK -> products, quantity INTEGER, line_total FLOAT)
""".strip()

DUCKDB_SCHEMA = """
DuckDB database `ecommerce_analytics.duckdb` (OLAP - historical analytics):

Table historical_sales_daily (date DATE, category VARCHAR, units_sold INTEGER,
    gross_revenue DOUBLE, return_count INTEGER)

Table category_margins (category VARCHAR, profit_margin_percentage DOUBLE,
    target_quarterly_growth DOUBLE)
""".strip()

ROUTER_SYSTEM_PROMPT = f"""You are a senior data engineer that routes natural-language
requests from internal e-commerce operations staff to the correct database engine and
writes a single, correct, engine-compliant SQL statement to answer the request.

There are two engines available:

1. SQLite ("sqlite") - the live transactional (OLTP) database. Use this for:
   - Looking up or updating individual products, customers, or orders
   - Checking live stock, order status, or customer details
   - Any INSERT / UPDATE / DELETE against live transactional records

{SQLITE_SCHEMA}

2. DuckDB ("duckdb") - the historical analytics (OLAP) database. Strictly READ-ONLY. Use this for:
   - Historical trend analysis, aggregations, category-level performance
   - Margin, growth-target, or multi-day/multi-month analytics

{DUCKDB_SCHEMA}

Critical Rules:
- FIRST, decide if the request is actually answerable using the two databases below.
  Assume the user can ask absolutely anything — general knowledge questions, jokes,
  personal opinions, requests unrelated to e-commerce, or gibberish. If the request is
  NOT something the two schemas below can answer, output engine="sqlite", sql="", and
  in rationale write a short, friendly message explaining you can only help with live
  orders/stock/customers or historical sales analytics. Do NOT invent a SQL query for
  an out-of-scope request just to produce something.
- Choose exactly one engine per request.
- Never invent tables or columns that are not listed above.
- When calculating aggregates like SUM(), AVG(), or margins, always wrap the expression in ROUND(..., 2) so values stay clean with 2 decimal places.
- NEVER attempt to directly "increase", "decrease", or "modify" derived business metrics such as revenue, profit, or returns. Databases do not create revenue out of thin air. If the user asks to "increase revenue", "boost sales", or perform an impossible operational update, output sql="" with an explanation in rationale.
- DuckDB is strictly READ-ONLY. Never produce INSERT, UPDATE, or DELETE statements for DuckDB.
- When an operator requests to "delete", "remove", or "cancel" an order, ALWAYS write an UPDATE query setting `order_status = 'CANCELLED'`. Never run hard DELETE on `orders` because it violates the foreign key relationship with `order_items`.
- Date comparison accuracy:
  * "orders pending from past X days" or "older than X days" means: order_date <= datetime('now', '-X days')
  * "within the last X days" or "recent X days" means: order_date >= datetime('now', '-X days')
- CONVERSATION CONTEXT & PRONOUN RESOLUTION:
  * When the user refers to "their", "them", "this customer", or "this order", inspect the provided Conversation History.
  * If the previous query or result was focused on a specific entity (e.g. customer_id = 10), filter explicitly using that entity ID.
  * Example: If the previous query fetched customer ID 10, and user asks "Show me their latest order", generate:
    SELECT * FROM orders WHERE customer_id = 10 ORDER BY order_date DESC LIMIT 1;
- Prefer adding WHERE or LIMIT clauses to analytical queries when appropriate.
- Do not wrap the SQL in markdown fences.
"""

IMPACT_SUMMARY_SYSTEM_PROMPT = """You write short, plain-English "blast radius" impact
summaries for a non-technical operations manager who is about to approve or reject a
database operation. You are given the target engine, the SQL statement, and the
automated reasons it was flagged as risky.

Respond with EXACTLY two bullet points, each starting with "- ", written in plain
English with no SQL syntax or jargon. The first bullet should explain what data will
be changed or scanned. The second bullet should explain the practical consequence or
risk if this is approved. Keep each bullet under 25 words.
"""

DML_PATTERN = re.compile(r"\b(UPDATE|DELETE|INSERT|DROP|ALTER|TRUNCATE|CREATE)\b", re.IGNORECASE)
WHERE_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)

# --------------------------------------------------------------------------
# LLM setup
# --------------------------------------------------------------------------
class SQLGeneration(BaseModel):
    engine: Literal["sqlite", "duckdb"] = Field(
        description="The target database engine for this request."
    )
    sql: str = Field(
        description="A single, engine-compliant SQL statement, or an empty string if the request cannot be executed via SQL."
    )
    rationale: str = Field(
        description="A short explanation of why this engine/query was chosen, or why the request is impossible."
    )


def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set.")
        return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key)

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=api_key)

    raise RuntimeError(f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'.")


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    user_query: str
    conversation_history: Annotated[List[str], operator.add]
    engine: Literal["sqlite", "duckdb"]
    sql: str
    rationale: str
    is_risky: bool
    risk_reasons: List[str]
    impact_summary: str
    approved: Optional[bool]
    execution_result: Dict[str, Any]
    final_message: str
    error: Optional[str]


# --------------------------------------------------------------------------
# Node: Dynamic Router & SQL Generator
# --------------------------------------------------------------------------
CAPABILITY_MESSAGE = """I can help you with two kinds of things:

1. **Live store operations** (orders, stock, customers) — e.g. *"What is the current stock for Fitness Smartwatch?"* or *"Mark all pending orders older than 20 days as cancelled."*
2. **Historical sales analytics** — e.g. *"Show me total revenue by category for the last quarter."*

Any write operation (like cancelling orders) will always ask for your approval first, with a plain-English explanation of what it will change.

I can't help with things unrelated to this store's orders, inventory, customers, or sales data — for those, I'll let you know it's outside what I can do."""

CAPABILITY_TRIGGERS = [
    "what can you do",
    "what can i ask",
    "what kind of question",
    "what questions can i ask",
    "what can you help",
    "what do you do",
    "how do you work",
    "what are you",
    "your capabilities",
    "what can you answer",
]


def _is_capability_question(query: str) -> bool:
    q = query.strip().lower()
    if any(trigger in q for trigger in CAPABILITY_TRIGGERS):
        return True
    # Broader heuristic: catches paraphrases like "what type of question can I
    # ask", "suggest me some questions", "give me example questions" that don't
    # match the fixed phrase list above.
    mentions_question_word = any(w in q for w in ["question", "ask"])
    mentions_meta_intent = any(
        w in q for w in ["suggest", "example", "sample", "type of", "kind of", "should i"]
    )
    return mentions_question_word and mentions_meta_intent


def router_node(state: GraphState) -> Dict[str, Any]:
    query = state["user_query"].strip()

    if query.lower() in ["hi", "hello", "hey", "hola", "namaste"]:
        return {
            "engine": "sqlite",
            "sql": "",
            "rationale": "Greeting",
            "error": None,
            "final_message": "Hello! I am your E-Commerce Operations Copilot. How can I help you with live store orders, inventory, or analytics today?",
            "execution_result": {},
            "approved": None,
        }

    if _is_capability_question(query):
        return {
            "engine": "sqlite",
            "sql": "",
            "rationale": "Capability question",
            "error": None,
            "final_message": CAPABILITY_MESSAGE,
            "execution_result": {},
            "approved": None,
        }

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SQLGeneration)

        history = state.get("conversation_history", [])
        history_context = ""
        if history:
            history_context = "Recent Conversation History:\n" + "\n".join(history[-4:]) + "\n\n"

        prompt_content = (
            f"{history_context}Current User Request: {query}\n\n"
            f"Generate the exact SQL statement."
        )

        response: SQLGeneration = structured_llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
        )

        sql_clean = response.sql.strip()

        if not sql_clean:
            return {
                "engine": response.engine,
                "sql": "",
                "rationale": response.rationale,
                "error": None,
                "final_message": response.rationale,
                "execution_result": {},
                "approved": None,
            }

        return {
            "engine": response.engine,
            "sql": sql_clean,
            "rationale": response.rationale,
            "error": None,
            "final_message": "",
            "execution_result": {},
            "approved": None,
        }
    except Exception as exc:  # noqa: BLE001
        err_str = str(exc)

        if "failed_generation" in err_str:
            match = re.search(r"'failed_generation':\s*['\"]([^'\"]+)['\"]", err_str)
            if match:
                return {
                    "engine": "sqlite",
                    "sql": "",
                    "rationale": "Conversational reply",
                    "error": None,
                    "final_message": match.group(1),
                    "execution_result": {},
                    "approved": None,
                }

        return {
            "engine": "sqlite",
            "sql": "",
            "rationale": "",
            "error": f"Failed to process request: {exc}",
            "final_message": "",
            "execution_result": {},
            "approved": None,
        }


# --------------------------------------------------------------------------
# Node: Guardrail & Impact Analyzer
# --------------------------------------------------------------------------
def guardrail_node(state: GraphState) -> Dict[str, Any]:
    if state.get("error") or not state.get("sql"):
        return {"is_risky": False, "risk_reasons": [], "impact_summary": ""}

    sql = state["sql"]
    engine = state["engine"]
    reasons: List[str] = []

    if DML_PATTERN.search(sql):
        reasons.append(
            "Query contains a write/DDL operation (INSERT, UPDATE, DELETE, DROP, "
            "ALTER, TRUNCATE, or CREATE) against the database."
        )

    if engine == "duckdb" and not WHERE_PATTERN.search(sql) and not LIMIT_PATTERN.search(sql):
        reasons.append(
            "Query is an unbounded analytical scan with no WHERE or LIMIT clause, "
            "which will scan the full historical dataset."
        )

    is_risky = len(reasons) > 0
    impact_summary = ""

    if is_risky:
        try:
            llm = get_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=IMPACT_SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Engine: {engine}\n"
                            f"SQL: {sql}\n"
                            f"Automated risk reasons: {'; '.join(reasons)}"
                        )
                    ),
                ]
            )
            impact_summary = response.content.strip()
        except Exception as exc:  # noqa: BLE001
            impact_summary = (
                "- Automated check flagged this query as risky.\n"
                f"- Reasons: {'; '.join(reasons)}"
            )

    return {
        "is_risky": is_risky,
        "risk_reasons": reasons,
        "impact_summary": impact_summary,
    }


def route_after_guardrail(state: GraphState) -> str:
    if state.get("error") or not state.get("sql"):
        return "execute"
    return "human_approval" if state.get("is_risky") else "execute"


# --------------------------------------------------------------------------
# Node: Interrupt-based Human-in-the-Loop approval
# --------------------------------------------------------------------------
def human_approval_node(state: GraphState) -> Dict[str, Any]:
    decision = interrupt(
        {
            "engine": state["engine"],
            "sql": state["sql"],
            "risk_reasons": state["risk_reasons"],
            "impact_summary": state["impact_summary"],
        }
    )
    approved = False
    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
    return {"approved": approved}


# --------------------------------------------------------------------------
# Node: Database execution
# --------------------------------------------------------------------------
def format_val(val: Any) -> Any:
    if isinstance(val, float):
        return f"₹{val:,.2f}"
    return val


def execute_sql(engine: str, sql: str) -> Dict[str, Any]:
    clean_sql = sql.strip().rstrip(";")
    is_select = clean_sql.lower().startswith(("select", "with", "pragma"))

    if engine == "sqlite":
        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            cur = conn.cursor()
            cur.execute(clean_sql)
            if is_select:
                columns = [d[0] for d in cur.description] if cur.description else []
                raw_rows = cur.fetchmany(MAX_RESULT_ROWS)
                rows = [[format_val(cell) for cell in row] for row in raw_rows]
                return {"columns": columns, "rows": rows, "rowcount": len(rows), "engine": engine}
            conn.commit()
            return {"columns": [], "rows": [], "rowcount": cur.rowcount, "engine": engine}
        finally:
            conn.close()

    if engine == "duckdb":
        conn = duckdb.connect(DUCKDB_DB_PATH)
        try:
            result = conn.execute(clean_sql)
            if is_select:
                columns = [d[0] for d in result.description] if result.description else []
                raw_rows = result.fetchmany(MAX_RESULT_ROWS)
                rows = [[format_val(cell) for cell in row] for row in raw_rows]
                return {"columns": columns, "rows": rows, "rowcount": len(rows), "engine": engine}
            return {"columns": [], "rows": [], "rowcount": -1, "engine": engine}
        finally:
            conn.close()

    raise ValueError(f"Unknown engine: {engine}")


def execute_node(state: GraphState) -> Dict[str, Any]:
    if state.get("final_message"):
        return {"final_message": state["final_message"], "execution_result": {}}

    if state.get("error"):
        return {
            "final_message": f"I could not process that request.\n\n**Error:** {state['error']}",
            "execution_result": {},
        }

    if state.get("is_risky") and not state.get("approved", False):
        return {
            "final_message": (
                "**Operation cancelled.** No changes were made to any database, as requested."
            ),
            "execution_result": {},
        }

    if not state.get("sql"):
        return {
            "final_message": state.get(
                "rationale", "No executable database action could be determined for this request."
            ),
            "execution_result": {},
        }

    try:
        result = execute_sql(state["engine"], state["sql"])
        if result["columns"]:
            preview = f"Returned {result['rowcount']} row(s)"
            if result["rowcount"] == MAX_RESULT_ROWS:
                preview += f" (truncated to the first {MAX_RESULT_ROWS})"
        else:
            preview = (
                "Statement executed successfully."
                if result["rowcount"] == -1
                else f"{result['rowcount']} row(s) affected."
            )

        history_entry = (
            f"User query: {state['user_query']} | "
            f"Engine: {state['engine']} | "
            f"Executed SQL: {state['sql']}"
        )

        return {
            "final_message": preview,
            "execution_result": result,
            "conversation_history": [history_entry],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "final_message": f"**Execution failed.** The database returned an error:\n\n`{exc}`",
            "execution_result": {},
            "error": str(exc),
        }


# --------------------------------------------------------------------------
# Async Graph Assembly (Disk Checkpointing via AsyncSqliteSaver)
# --------------------------------------------------------------------------
_copilot_graph = None


async def get_copilot_graph():
    global _copilot_graph
    if _copilot_graph is None:
        conn = await aiosqlite.connect(CHECKPOINT_DB_PATH)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

        builder = StateGraph(GraphState)
        builder.add_node("router", router_node)
        builder.add_node("guardrail", guardrail_node)
        builder.add_node("human_approval", human_approval_node)
        builder.add_node("execute", execute_node)

        builder.add_edge(START, "router")
        builder.add_edge("router", "guardrail")
        builder.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {"human_approval": "human_approval", "execute": "execute"},
        )
        builder.add_edge("human_approval", "execute")
        builder.add_edge("execute", END)

        _copilot_graph = builder.compile(checkpointer=checkpointer)

    return _copilot_graph