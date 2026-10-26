"""
Hierarchical Bayesian user model for MetaSearch.

Architecture
────────────
  Global model:  w_global ~ N(μ_g, Σ_g)       ← all click data
  Per-user model: w_user  ~ N(μ_g, Σ_g)       ← init from global posterior
                            ↓ user clicks
                  w_user  ~ N(μ_u, Σ_u)        ← personal posterior

Cold-start users (< MIN_CLICKS_FOR_PERSONAL clicks) use the global model.
"""

import numpy as np
from bayesian_ranker import BayesianRanker


class HierarchicalModel:
    """Manages a global ranker and per-user rankers."""

    MIN_CLICKS_FOR_PERSONAL = 5

    def __init__(self, global_ranker=None):
        self.global_ranker = global_ranker or BayesianRanker()
        self.user_rankers = {}        # user_id → BayesianRanker
        self.user_click_counts = {}   # user_id → int

    # ── Ranker selection ────────────────────────────────────────────────

    def get_ranker(self, user_id=None):
        """
        Return the appropriate ranker for a user.

        - Anonymous / cold-start  → global ranker
        - Warm user               → personal ranker (initialized lazily)
        """
        if user_id is None:
            return self.global_ranker

        if self.user_click_counts.get(user_id, 0) < self.MIN_CLICKS_FOR_PERSONAL:
            return self.global_ranker

        if user_id not in self.user_rankers:
            self.user_rankers[user_id] = self._spawn_user_ranker()

        return self.user_rankers[user_id]

    def _spawn_user_ranker(self):
        """Create a fresh user ranker whose prior = current global posterior."""
        child = BayesianRanker(v_sq=self.global_ranker.v_sq)
        child.B = self.global_ranker.B.copy()
        child.f = self.global_ranker.f.copy()
        child._dirty = True
        return child

    # ── Click updates ───────────────────────────────────────────────────

    def update_on_click(self, user_id, x_clicked, x_skipped_list):
        """
        Process a click event.

        Parameters
        ----------
        user_id : str | None
        x_clicked : array (d,)
            Feature vector of the clicked document.
        x_skipped_list : list of arrays (d,)
            Feature vectors of documents displayed *above* the clicked one
            that were NOT clicked (implicit negative preference).
        """
        # 1) Always update the global model
        for x_skip in x_skipped_list:
            self.global_ranker.update_pairwise(x_clicked, x_skip)

        if user_id is None:
            return

        # 2) Track per-user click volume
        self.user_click_counts[user_id] = (
            self.user_click_counts.get(user_id, 0) + 1
        )

        # 3) Promote to personal model if threshold reached
        if (self.user_click_counts[user_id] >= self.MIN_CLICKS_FOR_PERSONAL
                and user_id not in self.user_rankers):
            self.user_rankers[user_id] = self._spawn_user_ranker()

        # 4) Update personal model if it exists
        if user_id in self.user_rankers:
            for x_skip in x_skipped_list:
                self.user_rankers[user_id].update_pairwise(x_clicked, x_skip)

    # ── Maintenance ─────────────────────────────────────────────────────

    def refresh_user_priors(self):
        """
        Re-seed every user's prior from the latest global posterior.
        Call after a global batch retrain to propagate improvements.
        """
        for uid in list(self.user_rankers.keys()):
            self.user_rankers[uid] = self._spawn_user_ranker()

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self):
        return {
            "global": self.global_ranker.to_dict(),
            "users": {
                uid: r.to_dict() for uid, r in self.user_rankers.items()
            },
            "click_counts": self.user_click_counts,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        obj.global_ranker = BayesianRanker.from_dict(data["global"])
        obj.user_rankers = {
            uid: BayesianRanker.from_dict(rd)
            for uid, rd in data.get("users", {}).items()
        }
        obj.user_click_counts = data.get("click_counts", {})
        return obj
