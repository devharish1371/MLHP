import sqlite3
from datetime import datetime, timezone

DB_PATH = "clicks.db"


def _ensure_schema(conn):
    """Create or migrate the clicks table to the current schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clicks (
            timestamp TEXT,
            query TEXT,
            url TEXT,
            position INTEGER,
            session_id TEXT,
            user_id TEXT
        )
    """)

    # Migrate old 4-column table → add new columns if missing
    cols = {row[1] for row in conn.execute("PRAGMA table_info(clicks)").fetchall()}
    if "session_id" not in cols:
        conn.execute("ALTER TABLE clicks ADD COLUMN session_id TEXT")
    if "user_id" not in cols:
        conn.execute("ALTER TABLE clicks ADD COLUMN user_id TEXT")


def logger(query, url, position, session_id=None, user_id=None):
    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_schema(conn)

        conn.execute(
            """
            INSERT INTO clicks (timestamp, query, url, position, session_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, query, url, position, session_id, user_id)
        )
