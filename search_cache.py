"""
Search-result feature cache.

When the Streamlit app displays results it writes each document's feature
vector into an SQLite table keyed by a session ID.  When the click server
later receives a click it reads the cached features to perform a Bayesian
posterior update without recomputing features.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone

DB_PATH = "clicks.db"


def generate_session_id():
    """Return a unique search-session identifier."""
    return str(uuid.uuid4())


def cache_results(session_id, query, results_with_features):
    """
    Store search results and their feature vectors.

    Parameters
    ----------
    session_id : str
    query : str
    results_with_features : list of dict
        Each dict must have keys:
          url      – canonical URL
          position – display position (1-indexed)
          features – list/array of float feature values
          title    – document title
          link     – original (non-canonical) link
    """
    ts = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                session_id TEXT,
                query      TEXT,
                url        TEXT,
                position   INTEGER,
                features   TEXT,
                title      TEXT,
                link       TEXT,
                timestamp  TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sc_session "
            "ON search_cache(session_id)"
        )

        conn.executemany(
            """INSERT INTO search_cache
                   (session_id, query, url, position, features, title, link, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    session_id,
                    query,
                    r["url"],
                    r["position"],
                    json.dumps([float(v) for v in r["features"]]),
                    r.get("title", ""),
                    r.get("link", ""),
                    ts,
                )
                for r in results_with_features
            ],
        )


def get_session_results(session_id):
    """
    Retrieve all cached results for a search session.

    Returns a list of dicts (url, position, features, title, link)
    sorted by display position, or an empty list if the session is unknown.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM search_cache WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()

    return [
        {
            "url": row["url"],
            "position": row["position"],
            "features": json.loads(row["features"]),
            "title": row["title"],
            "link": row["link"],
        }
        for row in rows
    ]


def cleanup_old_cache(max_age_hours=48):
    """Delete cache entries older than *max_age_hours*."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM search_cache "
            "WHERE timestamp < datetime('now', ? || ' hours')",
            (str(-max_age_hours),),
        )
