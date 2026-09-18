import asyncio
import sqlite3
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = "langgraph_checkpoints.db"

async def verify():
    # Step 1: SQLite se sabse latest checkpoint ka thread_id nikalo
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        print("No checkpoints found in database yet.")
        return

    latest_thread_id = row[0]

    # Step 2: Us thread ki persisted state load karo
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        config = {"configurable": {"thread_id": latest_thread_id}}
        checkpoint_tuple = await saver.aget_tuple(config)
        
        if not checkpoint_tuple:
            print(f"No checkpoint found for thread_id={latest_thread_id}")
            return

        state = checkpoint_tuple.checkpoint.get("channel_values", {})
        next_nodes = checkpoint_tuple.checkpoint.get("next", ())

        print("\n================ PERSISTENCE REPORT ================")
        print(f"Thread ID Found    : {latest_thread_id}")
        print(f"Next Pending Node  : {next_nodes}")
        print(f"Last Query Executed: {state.get('last_user_query') or state.get('user_query')}")
        print(f"Generated SQL      : {state.get('generated_sql') or state.get('sql')}")
        print(f"Risk Level         : {state.get('risk_level')}")
        print(f"History Length     : {len(state.get('history', []))}")
        print("====================================================\n")

if __name__ == "__main__":
    asyncio.run(verify())