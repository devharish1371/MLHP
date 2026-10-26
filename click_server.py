"""
Click logging & redirect proxy with online Bayesian model updates.

When a user clicks a search result:
  1. Log the click to SQLite (as before).
  2. Look up the cached feature vectors for that search session.
  3. Do pairwise Bayesian posterior updates (clicked doc vs docs above it).
  4. Persist the updated model.
  5. Redirect the user to the target page.
"""

from flask import Flask, request, redirect, abort
from urllib.parse import urlparse
import numpy as np

from click_logger import logger
from search_cache import get_session_results
from model_store import load_model, save_model

app = Flask(__name__)


@app.get("/click")
def click():
    query      = request.args.get("q", "")
    url        = request.args.get("url", "")
    position   = request.args.get("pos", type=int)
    session_id = request.args.get("sid", "")
    user_id    = request.args.get("uid", "")

    # ── Validation ──────────────────────────────────────────────────────
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        abort(400)

    if not query or position is None or position < 1:
        abort(400)

    # ── 1. Log the click ────────────────────────────────────────────────
    logger(query, url, position,
           session_id=session_id or None,
           user_id=user_id or None)

    # ── 2. Online Bayesian update ───────────────────────────────────────
    if session_id:
        try:
            cached = get_session_results(session_id)
            if cached:
                clicked_features = None
                skipped_features = []

                for r in cached:
                    if r["link"] == url:
                        clicked_features = np.array(r["features"])
                    elif r["position"] < position:
                        skipped_features.append(np.array(r["features"]))

                if clicked_features is not None and skipped_features:
                    model = load_model()
                    model.update_on_click(
                        user_id or None,
                        clicked_features,
                        skipped_features,
                    )
                    save_model(model)
        except Exception as e:
            # Model update is best-effort; never block the redirect
            print(f"[click_server] model update failed: {e}")

    # ── 3. Redirect to target page ──────────────────────────────────────
    return redirect(url)


if __name__ == "__main__":
    app.run(port=5001, debug=False)
