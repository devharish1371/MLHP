#!/usr/bin/env python
"""
Batch-retrain the global Bayesian ranking model from all accumulated clicks.

Reads click events and cached feature vectors from SQLite, replays every
pairwise preference through the model, and saves the updated state.

Usage
─────
    python retrain.py                          # incremental (continue from saved model)
    python retrain.py --from-scratch           # train from the prior only
    python retrain.py --model-dir models/      # custom model directory
"""

import argparse
import sqlite3
import json
import numpy as np

from user_model import HierarchicalModel
from model_store import load_model, save_model
from bayesian_ranker import FEATURE_NAMES

DB_PATH = "clicks.db"


def get_click_pairs():
    """
    Join clicks with cached feature vectors to build pairwise preferences.

    Returns
    -------
    list of (user_id, clicked_features, [skipped_features, ...])
    """
    pairs = []

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        clicks = conn.execute("""
            SELECT c.query, c.url, c.position, c.session_id, c.user_id,
                   sc.features AS clicked_features
              FROM clicks c
              JOIN search_cache sc
                ON c.session_id = sc.session_id
               AND c.url       = sc.link
             WHERE c.session_id IS NOT NULL
             ORDER BY c.timestamp
        """).fetchall()

    for click in clicks:
        if not click["clicked_features"]:
            continue

        x_click = np.array(json.loads(click["clicked_features"]))

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            skipped_rows = conn.execute("""
                SELECT features FROM search_cache
                 WHERE session_id = ? AND position < ?
                 ORDER BY position
            """, (click["session_id"], click["position"])).fetchall()

        x_skipped = [np.array(json.loads(r["features"])) for r in skipped_rows]

        if x_skipped:
            pairs.append((click["user_id"], x_click, x_skipped))

    return pairs


def retrain(from_scratch=False, model_dir="models"):
    pairs = get_click_pairs()

    if not pairs:
        print("No click-pair data found. Nothing to retrain.")
        return

    model = HierarchicalModel() if from_scratch else load_model(model_dir)

    print(f"Retraining on {len(pairs)} click events …")

    for uid, x_click, x_skipped in pairs:
        model.update_on_click(uid, x_click, x_skipped)

    model.refresh_user_priors()
    save_model(model, model_dir)

    mu = model.global_ranker.mu
    print("\nRetrain complete.")
    print(f"  Global updates:       {model.global_ranker.n_updates}")
    print(f"  Personalized users:   {len(model.user_rankers)}")
    print()
    print(f"  {'Feature':<25}  {'Weight':>8}")
    print("  " + "-" * 36)
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<25}  {mu[i]:>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch-retrain ranking model from click data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args()
    retrain(from_scratch=args.from_scratch, model_dir=args.model_dir)
