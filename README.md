# Internal E-Commerce Operations Copilot

Text-to-SQL over two engines (SQLite OLTP + DuckDB OLAP), with a LangGraph
`interrupt`-based human-in-the-loop approval gate for any risky operation.

## Architecture

```
User message
    -> router_node        (LLM: natural language -> {engine, sql})
    -> guardrail_node      (regex-based DML / unbounded-scan detection
                             + LLM-written plain-English impact summary)
    -> [risky?] --yes--> human_approval_node (interrupt(), waits for
                             Command(resume={"approved": bool}))
              --no ---> execute_node (runs immediately)
    -> execute_node        (runs the SQL against SQLite or DuckDB)
```

State is checkpointed in-memory per Chainlit session (`thread_id`), so the
graph can pause indefinitely at `human_approval_node` and resume exactly
where it left off once the user clicks a button in the UI.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (or switch LLM_PROVIDER=openai + OPENAI_API_KEY)

# Point the app at your pre-generated databases (see generate_mock_data.py)
export SQLITE_DB_PATH=./ecommerce_oltp.db
export DUCKDB_DB_PATH=./ecommerce_analytics.duckdb

chainlit run app.py -w
```

Open http://localhost:8000.

## Docker

```bash
mkdir -p data
cp /path/to/ecommerce_oltp.db data/
cp /path/to/ecommerce_analytics.duckdb data/
cp .env.example .env   # fill in your API key

docker compose up --build
```

## Files

| File | Purpose |
|---|---|
| `graph.py` | LangGraph `StateGraph`: router, guardrail, HITL interrupt, execution nodes, `MemorySaver` checkpointer |
| `app.py` | Chainlit lifecycle hooks, approval-card rendering, `@cl.action_callback` handlers |
| `.chainlit/config.toml` | Dark enterprise theme, custom CSS path |
| `public/custom.css` | Executive dashboard styling |
| `Dockerfile` / `docker-compose.yml` | Containerized deployment, mounts `./data/*.db` |
| `requirements.txt` | Pinned dependencies |

## Safety model

- Any SQL containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
  `TRUNCATE`, or `CREATE` is flagged automatically.
- Any DuckDB query with no `WHERE` and no `LIMIT` clause is flagged as a
  heavy/unbounded analytical scan.
- Flagged queries never touch the database until a human clicks
  **✅ Confirm & Apply Changes** in the Chainlit UI. Clicking
  **❌ Cancel Operation** aborts the graph run with zero side effects.
- `MemorySaver` is in-memory only — state does not survive an app restart.
  Swap in `langgraph-checkpoint-sqlite` or `langgraph-checkpoint-postgres`
  for a durable checkpointer in a real production deployment.
