"""
Bayesian Thompson Sampling ranker for MetaSearch.

Model:  relevance(doc) = w · x(doc)
Prior:  w ~ N(μ₀, v² B₀⁻¹)
After n observations:
        w | data ~ N(Bₙ⁻¹ fₙ,  v² Bₙ⁻¹)
where   Bₙ = B₀ + Σ xₜ xₜᵀ / v²
        fₙ = B₀ μ₀ + Σ rₜ xₜ / v²

Thompson Sampling: sample w̃ ~ posterior, rank by w̃ · x.
"""

import numpy as np

# ── Feature name registry (must match features.py output order) ─────────────

FEATURE_NAMES = [
    # Per-engine rank features (google, then bing)
    "rank_google", "top1_google", "top10_google", "top50_google",
    "rank_bing",   "top1_bing",   "top10_bing",   "top50_bing",
    # Aggregate rank features
    "top1count", "top10count", "top50count",
    # Content features
    "query_url_cosine", "query_title_cosine", "domain_name_in_query",
    # Popularity features
    "url_length", "abstract_contains_home", "url_contains_tilde",
]

N_FEATURES = len(FEATURE_NAMES)  # 17


# ── Heuristic prior ────────────────────────────────────────────────────────

def default_prior():
    """
    Build an informed prior:
      • Rank features  ~3×  (strong signal)
      • Content features ~2× (moderate signal)
      • Popularity      ~1×  (weak / noisy signal)

    Returns (μ₀, σ₀²_diag) — prior mean and diagonal variance.
    """
    mu = np.zeros(N_FEATURES)
    var = np.ones(N_FEATURES)

    # ── Rank features (indices 0-10) ──
    mu[0]  = 3.0;  mu[4]  = 3.0   # rank_google, rank_bing
    mu[1]  = 3.0;  mu[5]  = 3.0   # top1_google, top1_bing
    mu[2]  = 2.0;  mu[6]  = 2.0   # top10_*
    mu[3]  = 1.0;  mu[7]  = 1.0   # top50_*
    mu[8]  = 3.0                   # top1count
    mu[9]  = 2.0                   # top10count
    mu[10] = 1.0                   # top50count
    var[0:11] = 2.0                # moderate prior uncertainty

    # ── Content features (indices 11-13) ──
    mu[11] = 2.0                   # query_url_cosine
    mu[12] = 2.5                   # query_title_cosine (title match is a strong cue)
    mu[13] = 1.5                   # domain_name_in_query
    var[11:14] = 3.0               # less certain → more exploration

    # ── Popularity features (indices 14-16) ──
    mu[14] = -0.5                  # url_length (longer = deeper = weaker authority)
    mu[15] = 0.5                   # abstract_contains_home
    mu[16] = -0.5                  # url_contains_tilde (~user dirs)
    var[14:17] = 4.0               # most uncertain → heavy exploration

    return mu, var


# ── BayesianRanker ──────────────────────────────────────────────────────────

class BayesianRanker:
    """
    Linear Thompson Sampling ranker.

    Internal state
    ──────────────
    B  : (d, d) precision matrix (scaled inverse covariance)
    f  : (d,)   reward-weighted feature accumulator
    v² : scalar observation-noise variance

    Posterior
    ────────
    μ  = B⁻¹ f
    Σ  = v² B⁻¹
    w | data ~ N(μ, Σ)
    """

    def __init__(self, v_sq=1.0):
        self.d = N_FEATURES
        self.v_sq = v_sq
        self.n_updates = 0

        mu_0, var_0 = default_prior()

        # B₀ = diag(1 / σ²₀)  — precision of the prior
        self.B = np.diag(1.0 / var_0)
        # f₀ = B₀ μ₀
        self.f = self.B @ mu_0

        self._dirty = True
        self._mu_cache = None
        self._Sigma_cache = None

    # ── Posterior access ────────────────────────────────────────────────

    def _recompute(self):
        if self._dirty:
            self._Sigma_cache = np.linalg.inv(self.B)
            self._mu_cache = self._Sigma_cache @ self.f
            self._dirty = False

    @property
    def mu(self):
        """Posterior mean (MAP estimate)."""
        self._recompute()
        return self._mu_cache

    @property
    def Sigma(self):
        """Posterior covariance (unscaled by v²)."""
        self._recompute()
        return self._Sigma_cache

    # ── Sampling / scoring ──────────────────────────────────────────────

    def sample_weights(self):
        """Thompson Sampling: draw w̃ ~ N(μ, v² Σ)."""
        self._recompute()
        return np.random.multivariate_normal(
            self._mu_cache, self.v_sq * self._Sigma_cache
        )

    def map_weights(self):
        """Return the MAP estimate (posterior mean)."""
        self._recompute()
        return self._mu_cache.copy()

    def rank_documents(self, feature_vectors, explore=True):
        """
        Score and rank documents.

        Parameters
        ----------
        feature_vectors : list of (d,) arrays
        explore : bool
            True  → Thompson Sampling (stochastic)
            False → MAP estimate (deterministic)

        Returns
        -------
        indices : list[int]   — document indices sorted best-first
        scores  : list[float] — corresponding scores
        """
        w = self.sample_weights() if explore else self.map_weights()
        scores = [float(np.dot(w, x)) for x in feature_vectors]
        order = list(np.argsort(scores)[::-1])
        return order, [scores[i] for i in order]

    # ── Online Bayesian updates ─────────────────────────────────────────

    def update(self, x, reward):
        """
        Pointwise update: observe (feature_vector, scalar reward).

        For click data:  reward = 1.0 (click) or 0.0 (skip).
        """
        x = np.asarray(x, dtype=float)
        self.B += np.outer(x, x) / self.v_sq
        self.f += reward * x / self.v_sq
        self.n_updates += 1
        self._dirty = True

    def update_pairwise(self, x_clicked, x_skipped):
        """
        Pairwise preference update.

        The user preferred doc(x_clicked) over doc(x_skipped).
        Equivalent to observing (x_clicked − x_skipped) with reward 1.
        """
        diff = np.asarray(x_clicked, dtype=float) - np.asarray(x_skipped, dtype=float)
        self.update(diff, 1.0)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self):
        return {
            "B": self.B.tolist(),
            "f": self.f.tolist(),
            "v_sq": self.v_sq,
            "d": self.d,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        obj.d = data["d"]
        obj.v_sq = data["v_sq"]
        obj.B = np.array(data["B"])
        obj.f = np.array(data["f"])
        obj.n_updates = data.get("n_updates", 0)
        obj._dirty = True
        obj._mu_cache = None
        obj._Sigma_cache = None
        return obj
