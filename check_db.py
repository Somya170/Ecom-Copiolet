import sqlite3

conn = sqlite3.connect('langgraph_checkpoints.db')
cursor = conn.cursor()

tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print('Tables in Checkpoint DB:', tables)

try:
    count = cursor.execute('SELECT COUNT(*) FROM checkpoints;').fetchone()[0]
    print('Total Checkpoints Stored:', count)
except Exception as e:
    print('Could not query checkpoints table:', e)

conn.close()
