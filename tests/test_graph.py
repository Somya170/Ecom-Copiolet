"""
Test suite for the E-Commerce Ops Copilot LangGraph pipeline.

Run with:
    pytest -v tests/test_graph.py

Requires (add to a requirements-dev.txt or install directly):
    pytest
    pytest-asyncio
"""
import os
import sqlite3
import sys
from unittest.mock import MagicMock, patch

import duckdb
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph as g  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: isolated throwaway databases so tests never touch real data
# ---------------------------------------------------------------------------
@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_oltp.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
        "order_date TEXT, order_status TEXT, total_amount REAL)"
    )
    conn.execute(
        "INSERT INTO orders VALUES (1, 10, '2024-01-01', 'PENDING', 500.0)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(g, "SQLITE_DB_PATH", str(db_path))
    return str(db_path)


@pytest.fixture
def duckdb_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_analytics.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE historical_sales_daily (date DATE, category VARCHAR, "
        "units_sold INTEGER, gross_revenue DOUBLE, return_count INTEGER)"
    )
    conn.execute(
        "INSERT INTO historical_sales_daily VALUES "
        "('2024-01-01', 'Electronics', 10, 999.99, 1)"
    )
    conn.close()
    monkeypatch.setattr(g, "DUCKDB_DB_PATH", str(db_path))
    return str(db_path)


def make_llm_response(engine="sqlite", sql="", rationale="ok"):
    """Build a fake SQLGeneration-shaped object returned by structured_output LLM."""
    resp = MagicMock()
    resp.engine = engine
    resp.sql = sql
    resp.rationale = rationale
    return resp


# ---------------------------------------------------------------------------
# router_node — greeting shortcut (no LLM call should happen at all)
# ---------------------------------------------------------------------------
class TestRouterGreeting:
    @pytest.mark.parametrize("text", ["hi", "hello", "hey", "hola", "namaste"])
    def test_exact_greeting_skips_llm(self, text):
        with patch.object(g, "get_llm") as mock_get_llm:
            result = g.router_node({"user_query": text})
            mock_get_llm.assert_not_called()
            assert result["sql"] == ""
            assert "Copilot" in result["final_message"]

    def test_whitespace_padded_greeting_is_still_recognized(self):
        """The code strips whitespace before comparing, so "  hello  " correctly
        matches the greeting list and never reaches the LLM."""
        with patch.object(g, "get_llm") as mock_get_llm:
            g.router_node({"user_query": "  hello  "})
            mock_get_llm.assert_not_called()

    @pytest.mark.parametrize("text", ["Hii", "HELLO!", "yo"])
    def test_greeting_variants_fall_through_to_llm(self, text):
        """
        EDGE CASE / KNOWN GAP: the greeting check is an exact, case-sensitive
        match against a fixed list (only after stripping whitespace). Capitalized
        greetings, trailing punctuation, or synonyms ("yo") all fall through to a
        full LLM call instead of being recognized as greetings. Not a crash, just
        an unnecessary LLM round-trip — flagging so it's a conscious tradeoff,
        not an oversight.
        """
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.with_structured_output.return_value.invoke.return_value = (
                make_llm_response(rationale="Hello there!")
            )
            mock_get_llm.return_value = fake_llm
            g.router_node({"user_query": text})
            mock_get_llm.assert_called_once()


# ---------------------------------------------------------------------------
# router_node — empty / whitespace-only / off-topic input
# ---------------------------------------------------------------------------
class TestRouterEdgeInputs:
    def test_empty_query_does_not_crash(self):
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.with_structured_output.return_value.invoke.return_value = (
                make_llm_response(rationale="I need a question to help with.")
            )
            mock_get_llm.return_value = fake_llm
            result = g.router_node({"user_query": ""})
            assert result["error"] is None

    def test_off_topic_question_returns_no_sql(self):
        """
        Simulates the LLM correctly following the new out-of-scope rule:
        a question like "what's the weather today" must come back with
        sql="" and a rationale, never an invented/hallucinated query.
        """
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.with_structured_output.return_value.invoke.return_value = (
                make_llm_response(
                    sql="",
                    rationale="I can only help with orders, stock, customers, or sales analytics.",
                )
            )
            mock_get_llm.return_value = fake_llm
            result = g.router_node({"user_query": "what's the weather today?"})
            assert result["sql"] == ""
            assert result["final_message"] == (
                "I can only help with orders, stock, customers, or sales analytics."
            )

    @pytest.mark.parametrize(
        "text",
        [
            "what kind of question can I ask?",
            "What can you help with?",
            "what can you do",
            "how do you work?",
        ],
    )
    def test_capability_question_answered_without_llm_call(self, text):
        """
        This is the exact scenario the manager tested: a meta-question about
        the assistant's own capabilities. It must be answered directly with
        examples — not sent to the SQL-generating LLM, and not treated as
        "out of scope" like a genuinely unrelated question would be.
        """
        with patch.object(g, "get_llm") as mock_get_llm:
            result = g.router_node({"user_query": text})
            mock_get_llm.assert_not_called()
            assert result["sql"] == ""
            assert "orders" in result["final_message"].lower()
            assert "analytics" in result["final_message"].lower()

    def test_llm_raises_generic_exception_is_caught(self):
        with patch.object(g, "get_llm") as mock_get_llm:
            mock_get_llm.side_effect = RuntimeError("network timeout")
            result = g.router_node({"user_query": "show me all orders"})
            assert result["error"] is not None
            assert "network timeout" in result["error"]

    def test_missing_api_key_raises_before_llm_call(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(g, "LLM_PROVIDER", "groq")
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            g.get_llm()

    def test_unsupported_provider_raises(self, monkeypatch):
        monkeypatch.setattr(g, "LLM_PROVIDER", "some_unknown_provider")
        with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
            g.get_llm()

    def test_conversation_history_used_for_pronoun_resolution(self):
        """History is passed into the prompt; router shouldn't crash on it,
        and should only take the last 4 entries (older ones are dropped)."""
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_structured = fake_llm.with_structured_output.return_value
            fake_structured.invoke.return_value = make_llm_response(
                sql="SELECT * FROM orders WHERE customer_id = 10", rationale="ok"
            )
            mock_get_llm.return_value = fake_llm
            long_history = [f"turn {i}" for i in range(10)]
            g.router_node(
                {"user_query": "show their latest order", "conversation_history": long_history}
            )
            sent_prompt = fake_structured.invoke.call_args[0][0][1].content
            assert "turn 9" in sent_prompt
            assert "turn 0" not in sent_prompt  # only last 4 kept


# ---------------------------------------------------------------------------
# guardrail_node — pure logic, no LLM needed unless risky
# ---------------------------------------------------------------------------
class TestGuardrail:
    def test_safe_select_is_not_risky(self):
        state = {"sql": "SELECT * FROM products WHERE product_id = 1", "engine": "sqlite"}
        result = g.guardrail_node(state)
        assert result["is_risky"] is False

    def test_update_statement_flagged_risky(self):
        state = {"sql": "UPDATE orders SET order_status='CANCELLED' WHERE order_id=1", "engine": "sqlite"}
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.invoke.return_value.content = "- changes 1 row\n- irreversible"
            mock_get_llm.return_value = fake_llm
            result = g.guardrail_node(state)
        assert result["is_risky"] is True
        assert any("write/DDL" in r for r in result["risk_reasons"])

    def test_unbounded_duckdb_scan_flagged_risky(self):
        state = {"sql": "SELECT * FROM historical_sales_daily", "engine": "duckdb"}
        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.invoke.return_value.content = "- scans everything\n- slow query"
            mock_get_llm.return_value = fake_llm
            result = g.guardrail_node(state)
        assert result["is_risky"] is True
        assert any("unbounded" in r for r in result["risk_reasons"])

    def test_bounded_duckdb_scan_not_risky(self):
        state = {"sql": "SELECT * FROM historical_sales_daily LIMIT 10", "engine": "duckdb"}
        result = g.guardrail_node(state)
        assert result["is_risky"] is False

    def test_no_sql_short_circuits_without_llm_call(self):
        with patch.object(g, "get_llm") as mock_get_llm:
            result = g.guardrail_node({"sql": "", "engine": "sqlite"})
            mock_get_llm.assert_not_called()
            assert result["is_risky"] is False

    def test_impact_summary_llm_failure_has_fallback(self):
        """If the LLM call for the plain-English summary itself fails,
        the user must still get a readable fallback, not a crash."""
        state = {"sql": "DELETE FROM orders", "engine": "sqlite"}
        with patch.object(g, "get_llm") as mock_get_llm:
            mock_get_llm.side_effect = RuntimeError("llm down")
            result = g.guardrail_node(state)
        assert result["is_risky"] is True
        assert "Automated check flagged" in result["impact_summary"]

    def test_case_insensitive_dml_detection(self):
        state = {"sql": "update orders set order_status='CANCELLED'", "engine": "sqlite"}
        with patch.object(g, "get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value.content = "- x\n- y"
            result = g.guardrail_node(state)
        assert result["is_risky"] is True


# ---------------------------------------------------------------------------
# route_after_guardrail — the conditional edge function
# ---------------------------------------------------------------------------
class TestRouting:
    def test_error_state_routes_to_execute(self):
        assert g.route_after_guardrail({"error": "boom", "sql": "SELECT 1"}) == "execute"

    def test_no_sql_routes_to_execute(self):
        assert g.route_after_guardrail({"sql": ""}) == "execute"

    def test_risky_routes_to_human_approval(self):
        assert g.route_after_guardrail({"sql": "DELETE FROM x", "is_risky": True}) == "human_approval"

    def test_safe_routes_to_execute(self):
        assert g.route_after_guardrail({"sql": "SELECT 1", "is_risky": False}) == "execute"


# ---------------------------------------------------------------------------
# execute_sql — real, isolated DB integration tests
# ---------------------------------------------------------------------------
class TestExecuteSQLIntegration:
    def test_sqlite_select(self, sqlite_db):
        result = g.execute_sql("sqlite", "SELECT * FROM orders")
        assert result["rowcount"] == 1
        assert result["columns"] == ["order_id", "customer_id", "order_date", "order_status", "total_amount"]

    def test_sqlite_update_cancel(self, sqlite_db):
        result = g.execute_sql(
            "sqlite", "UPDATE orders SET order_status='CANCELLED' WHERE order_id=1"
        )
        assert result["rowcount"] == 1
        conn = sqlite3.connect(sqlite_db)
        status = conn.execute("SELECT order_status FROM orders WHERE order_id=1").fetchone()[0]
        assert status == "CANCELLED"

    def test_sqlite_syntax_error_raises(self, sqlite_db):
        with pytest.raises(Exception):
            g.execute_sql("sqlite", "SELEKT * FROM orders")

    def test_duckdb_select(self, duckdb_db):
        result = g.execute_sql("duckdb", "SELECT * FROM historical_sales_daily")
        assert result["rowcount"] == 1
        assert result["columns"][1] == "category"

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            g.execute_sql("mongodb", "SELECT 1")

    def test_money_formatting(self, sqlite_db):
        result = g.execute_sql("sqlite", "SELECT total_amount FROM orders WHERE order_id=1")
        assert result["rows"][0][0] == "₹500.00"


# ---------------------------------------------------------------------------
# execute_node — behavior branches (approved / cancelled / error / greeting)
# ---------------------------------------------------------------------------
class TestExecuteNode:
    def test_returns_final_message_directly_if_already_set(self):
        state = {"final_message": "Hello!"}
        result = g.execute_node(state)
        assert result["final_message"] == "Hello!"

    def test_returns_error_message_if_error_present(self):
        state = {"error": "boom", "final_message": ""}
        result = g.execute_node(state)
        assert "boom" in result["final_message"]

    def test_risky_not_approved_cancels_without_executing(self):
        state = {"is_risky": True, "approved": False, "sql": "DELETE FROM orders"}
        with patch.object(g, "execute_sql") as mock_exec:
            result = g.execute_node(state)
            mock_exec.assert_not_called()
        assert "cancelled" in result["final_message"].lower()

    def test_no_sql_falls_back_to_rationale(self):
        state = {"sql": "", "rationale": "Out of scope request."}
        result = g.execute_node(state)
        assert result["final_message"] == "Out of scope request."

    def test_execute_sql_exception_is_caught(self, sqlite_db):
        state = {"sql": "SELEKT bad syntax", "engine": "sqlite", "user_query": "x"}
        result = g.execute_node(state)
        assert "Execution failed" in result["final_message"]
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Full graph — human-in-the-loop interrupt/resume flow (integration)
# ---------------------------------------------------------------------------
class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_risky_operation_interrupts_then_resumes(self, tmp_path, sqlite_db, monkeypatch):
        monkeypatch.setattr(g, "_copilot_graph", None)
        monkeypatch.setattr(g, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))

        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.with_structured_output.return_value.invoke.return_value = make_llm_response(
                engine="sqlite",
                sql="UPDATE orders SET order_status='CANCELLED' WHERE order_id=1",
                rationale="cancel order 1",
            )
            fake_llm.invoke.return_value.content = "- cancels 1 order\n- cannot be undone"
            mock_get_llm.return_value = fake_llm

            graph = await g.get_copilot_graph()
            config = {"configurable": {"thread_id": "test-thread-1"}}

            result = await graph.ainvoke({"user_query": "cancel order 1"}, config=config)
            assert "__interrupt__" in result, "risky write should pause for approval"

            from langgraph.types import Command

            final = await graph.ainvoke(Command(resume={"approved": True}), config=config)
            assert "__interrupt__" not in final
            assert final["execution_result"]["rowcount"] == 1

    @pytest.mark.asyncio
    async def test_rejecting_approval_makes_no_db_change(self, tmp_path, sqlite_db, monkeypatch):
        monkeypatch.setattr(g, "_copilot_graph", None)
        monkeypatch.setattr(g, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints2.db"))

        with patch.object(g, "get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_llm.with_structured_output.return_value.invoke.return_value = make_llm_response(
                engine="sqlite",
                sql="UPDATE orders SET order_status='CANCELLED' WHERE order_id=1",
                rationale="cancel order 1",
            )
            fake_llm.invoke.return_value.content = "- cancels 1 order\n- cannot be undone"
            mock_get_llm.return_value = fake_llm

            graph = await g.get_copilot_graph()
            config = {"configurable": {"thread_id": "test-thread-2"}}
            await graph.ainvoke({"user_query": "cancel order 1"}, config=config)

            from langgraph.types import Command

            final = await graph.ainvoke(Command(resume={"approved": False}), config=config)
            assert "cancelled" in final["final_message"].lower()

        conn = sqlite3.connect(sqlite_db)
        status = conn.execute("SELECT order_status FROM orders WHERE order_id=1").fetchone()[0]
        assert status == "PENDING", "rejected operation must not touch the database"