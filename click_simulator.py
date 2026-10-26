#!/usr/bin/env python
"""
Click simulator for bootstrapping the Bayesian ranking model.

Generates synthetic search sessions, simulates user clicks with a cascade
click model, and feeds the resulting pairwise preferences into the
Bayesian ranker so the model starts with meaningful weights before any
real user traffic arrives.

Usage
─────
    python click_simulator.py                        # 200 sessions, fresh prior
    python click_simulator.py --n-sessions 500       # more sessions
    python click_simulator.py --no-from-scratch      # continue from saved model
"""

import argparse
import numpy as np

from bayesian_ranker import BayesianRanker, N_FEATURES, FEATURE_NAMES
from user_model import HierarchicalModel
from model_store import load_model, save_model


# ── Hidden "true" relevance weights (what we hope to recover) ───────────

TRUE_WEIGHTS = np.array([
    3.5,   # rank_google
    4.0,   # top1_google
    2.5,   # top10_google
    1.0,   # top50_google
    3.0,   # rank_bing
    3.5,   # top1_bing
    2.0,   # top10_bing
    0.8,   # top50_bing
    4.0,   # top1count
    2.5,   # top10count
    1.0,   # top50count
    2.5,   # query_url_cosine
    3.0,   # query_title_cosine
    1.5,   # domain_name_in_query
    -0.3,  # url_length
    0.3,   # abstract_contains_home
    -0.5,  # url_contains_tilde
])


# ── Synthetic data generation ───────────────────────────────────────────

def generate_synthetic_features(n_docs=10):
    """Generate a realistic (n_docs × d) feature matrix."""
    X = np.zeros((n_docs, N_FEATURES))

    for i in range(n_docs):
        g_rank = np.random.randint(1, 50) if np.random.random() > 0.3 else None
        b_rank = np.random.randint(1, 50) if np.random.random() > 0.3 else None

        if g_rank:
            X[i, 0] = max(0, (100 - g_rank) / 100)
            X[i, 1] = int(g_rank == 1)
            X[i, 2] = int(g_rank <= 10)
            X[i, 3] = int(g_rank <= 50)
        if b_rank:
            X[i, 4] = max(0, (100 - b_rank) / 100)
            X[i, 5] = int(b_rank == 1)
            X[i, 6] = int(b_rank <= 10)
            X[i, 7] = int(b_rank <= 50)

        X[i, 8]  = X[i, 1] + X[i, 5]            # top1count
        X[i, 9]  = X[i, 2] + X[i, 6]            # top10count
        X[i, 10] = X[i, 3] + X[i, 7]            # top50count

        X[i, 11] = np.random.beta(2, 5)          # query_url_cosine
        X[i, 12] = np.random.beta(3, 4)          # query_title_cosine
        X[i, 13] = int(np.random.random() > 0.7) # domain_name_in_query

        X[i, 14] = np.random.exponential(1.0)    # url_length (normalized)
        X[i, 15] = int(np.random.random() > 0.8) # abstract_contains_home
        X[i, 16] = int(np.random.random() > 0.9) # url_contains_tilde

    return X


def true_relevance(features):
    """Score documents using the hidden true weights + noise, centred around 0."""
    raw = features @ TRUE_WEIGHTS + np.random.normal(0, 0.5, len(features))
    # Centre so sigmoid gives a useful mix of clicks / skips
    return raw - np.mean(raw)


# ── Cascade click model ────────────────────────────────────────────────

def cascade_click_model(relevance_scores, max_depth=7):
    """
    Simulate user clicks under a cascade examination model.

    The user scans results top-to-bottom:
      • P(examine position k) = 0.95^k  (gentle decay — users scan several results)
      • P(click | examine)    = σ(relevance)
      • After a click, P(stop) = 0.3     (often keeps scanning)

    Returns a list of clicked positions (0-indexed).
    """
    clicked = []
    for pos in range(min(len(relevance_scores), max_depth)):
        if np.random.random() > 0.95 ** pos:
            break  # stopped examining
        p_click = 1.0 / (1.0 + np.exp(-relevance_scores[pos]))
        if np.random.random() < p_click:
            clicked.append(pos)
            if np.random.random() < 0.3:
                break  # satisfied
    return clicked


# ── Main simulation loop ───────────────────────────────────────────────

def simulate(n_sessions=200, model_dir="models", from_scratch=True):
    model = HierarchicalModel() if from_scratch else load_model(model_dir)
    ranker = model.global_ranker

    total_clicks = 0
    weight_history = [ranker.mu.copy()]

    for s in range(n_sessions):
        # 1. Generate synthetic documents
        X = generate_synthetic_features(n_docs=10)

        # 2. Rank with Thompson Sampling
        w = ranker.sample_weights()
        order = np.argsort(X @ w)[::-1]
        X_ranked = X[order]

        # 3. Compute "true" relevance for the displayed order
        rel = true_relevance(X_ranked)

        # 4. Simulate clicks
        clicks = cascade_click_model(rel)

        # 5. Update model with pairwise preferences
        for cp in clicks:
            skipped = [X_ranked[j] for j in range(cp) if j not in clicks]
            if skipped:
                model.update_on_click(None, X_ranked[cp], skipped)
                total_clicks += 1

        weight_history.append(ranker.mu.copy())

        if (s + 1) % 50 == 0:
            cos = _cosine(ranker.mu, TRUE_WEIGHTS)
            print(f"  session {s+1:>4}/{n_sessions}  |  clicks {total_clicks:>5}  |  cos(μ, w*) = {cos:.4f}")

    # ── Save & report ──────────────────────────────────────────────────
    save_model(model, model_dir)

    print("\n" + "=" * 64)
    print(" SIMULATION COMPLETE")
    print("=" * 64)
    print(f" Sessions:           {n_sessions}")
    print(f" Total click events: {total_clicks}")
    print()

    mu = ranker.mu
    print(f" {'Feature':<25}  {'Learned':>8}  {'True':>8}  {'|Err|':>8}")
    print(" " + "-" * 55)
    for i, name in enumerate(FEATURE_NAMES):
        print(f" {name:<25}  {mu[i]:>8.3f}  {TRUE_WEIGHTS[i]:>8.3f}  {abs(mu[i]-TRUE_WEIGHTS[i]):>8.3f}")

    cos = _cosine(mu, TRUE_WEIGHTS)
    print(f"\n Cosine similarity(μ, w*): {cos:.4f}")
    return weight_history


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bootstrap Bayesian ranker with simulated clicks")
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--no-from-scratch", dest="from_scratch", action="store_false", default=True)
    args = ap.parse_args()
    simulate(n_sessions=args.n_sessions, model_dir=args.model_dir, from_scratch=args.from_scratch)
