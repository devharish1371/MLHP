import streamlit as st
import numpy as np
import uuid
import pandas as pd
from urllib.parse import urlencode

from metasearch import search, rank_with_features
from model_store import load_model
from search_cache import generate_session_id, cache_results
from bayesian_ranker import FEATURE_NAMES, N_FEATURES

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MetaSearch",
    page_icon=":material/search:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session init ───────────────────────────────────────────────────────────

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

# ── Load model ─────────────────────────────────────────────────────────────

model = load_model()

# ═══════════════════════════════════════════════════════════════════════════
#  USER-FACING SEARCH INTERFACE
#  Clean, minimal — feels like a normal search engine.
#  Bayesian personalisation runs silently in the background.
# ═══════════════════════════════════════════════════════════════════════════

st.title("MetaSearch", icon=":material/travel_explore:")
st.caption("Search smarter across the web")

query = st.text_input(
    "Search",
    placeholder="Search the web…",
    label_visibility="collapsed",
)

search_clicked = st.button(
    "Search",
    icon=":material/search:",
    type="primary",
    width="stretch",
)

# ── Search logic ───────────────────────────────────────────────────────────

if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.", icon=":material/warning:")
    else:
        engines = ["google", "bing"]
        clean_query = query.strip()

        with st.spinner("Searching…"):
            try:
                results = {engine: search(engine, clean_query) for engine in engines}
                session_id = generate_session_id()

                # Pick ranking strategy from dev panel (defaults to Thompson Sampling)
                dev_ranker = st.session_state.get("dev_ranker", "Explore")
                user_ranker = model.get_ranker(st.session_state["user_id"])

                if dev_ranker == "Exploit":
                    w = user_ranker.map_weights()
                elif dev_ranker == "Baseline":
                    w = np.ones(N_FEATURES)
                else:
                    w = user_ranker.sample_weights()

                ranked = rank_with_features(clean_query, results, engines, w)

                # Cache features so the click server can do Bayesian updates
                cache_entries = [
                    {
                        "url": url,
                        "position": i,
                        "features": feats,
                        "title": item["doc"]["title"],
                        "link": item["doc"]["link"],
                    }
                    for i, (url, item, score, feats) in enumerate(ranked, 1)
                ]
                cache_results(session_id, clean_query, cache_entries)

                st.session_state["ranked"] = ranked
                st.session_state["query"] = clean_query
                st.session_state["session_id"] = session_id
                st.session_state["engine_stats"] = {
                    "google": len(results.get("google", [])),
                    "bing": len(results.get("bing", [])),
                }

            except Exception as e:
                st.error(f"Search failed: {e}", icon=":material/error:")

# ── Display results ────────────────────────────────────────────────────────

ranked = st.session_state.get("ranked", [])

if ranked:
    st.caption(f"{len(ranked)} results")

    for i, (url, item, score, _feats) in enumerate(ranked, 1):
        doc = item["doc"]

        params = urlencode({
            "q": st.session_state["query"],
            "url": doc["link"],
            "pos": i,
            "sid": st.session_state.get("session_id", ""),
            "uid": st.session_state["user_id"],
        })
        click_url = f"http://127.0.0.1:5001/click?{params}"

        with st.container(border=True):
            st.markdown(f"**[{doc['title']}]({click_url})**")
            st.caption(doc["link"])
            if doc.get("snippet"):
                st.markdown(doc["snippet"])

elif st.session_state.get("query"):
    st.info("No results found.", icon=":material/search_off:")


# ═══════════════════════════════════════════════════════════════════════════
#  DEVELOPER PANEL
#  Hidden by default. Shows every knob of the Bayesian pipeline.
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("For developers", icon=":material/code:", expanded=False):

    dev_tabs = st.tabs([
        ":material/model_training: Model",
        ":material/bar_chart: Posterior weights",
        ":material/analytics: Result analysis",
        ":material/info: How it works",
    ])

    # ── Tab 1: Model overview ──────────────────────────────────────────

    with dev_tabs[0]:
        st.subheader("Model overview", icon=":material/model_training:")

        overview_cols = st.columns(3)
        overview_cols[0].metric("Bayesian updates", f"{model.global_ranker.n_updates:,}")
        overview_cols[1].metric("Personalized users", len(model.user_rankers))

        user_clicks = model.user_click_counts.get(st.session_state["user_id"], 0)
        remaining = max(0, model.MIN_CLICKS_FOR_PERSONAL - user_clicks)
        overview_cols[2].metric("Your clicks", user_clicks)

        user_ranker = model.get_ranker(st.session_state["user_id"])
        if user_ranker is not model.global_ranker:
            st.badge("Personalized model active", icon=":material/person:", color="green")
        else:
            st.caption(
                f":material/public: Using global model — "
                f"{remaining} more click{'s' if remaining != 1 else ''} until personalization"
            )

        st.subheader("Ranking strategy", icon=":material/tune:")
        st.segmented_control(
            "Ranking strategy",
            options=["Explore", "Exploit", "Baseline"],
            default="Explore",
            key="dev_ranker",
            label_visibility="collapsed",
        )
        strategy_info = {
            "Explore": "**Thompson Sampling** — Samples weight vector from the Bayesian posterior each query. Naturally balances exploration (learning) with exploitation (ranking quality).",
            "Exploit": "**MAP Estimate** — Uses the posterior mean (most likely weights). Best ranking quality given current knowledge, but no exploration.",
            "Baseline": "**Uniform weights** — All 17 features weighted equally. Ignores learned model entirely.",
        }
        st.caption(strategy_info.get(st.session_state.get("dev_ranker", "Explore"), ""))

    # ── Tab 2: Posterior weights ───────────────────────────────────────

    with dev_tabs[1]:
        st.subheader("Posterior mean weights", icon=":material/bar_chart:")
        st.caption("Learned importance of each feature after Bayesian updates")

        mu = model.global_ranker.mu
        weights_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Weight": mu,
            "Abs. weight": np.abs(mu),
            "Category": (
                ["Rank"] * 4 + ["Rank"] * 4 + ["Rank"] * 3
                + ["Content"] * 3
                + ["Popularity"] * 3
            ),
        }).sort_values("Abs. weight", ascending=False)

        st.dataframe(
            weights_df[["Feature", "Weight", "Category"]],
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="large"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.4f"),
                "Category": st.column_config.TextColumn("Category"),
            },
            hide_index=True,
            height=min(35 * len(weights_df) + 38, 600),
            alt="Feature weight table sorted by absolute magnitude",
        )

        # Weight distribution bar chart
        chart_df = weights_df[["Feature", "Weight"]].set_index("Feature")
        st.bar_chart(
            chart_df,
            horizontal=True,
            color="#6366F1",
            alt="Bar chart of posterior feature weights",
        )

    # ── Tab 3: Result analysis ─────────────────────────────────────────

    with dev_tabs[2]:
        st.subheader("Result analysis", icon=":material/analytics:")

        if ranked:
            st.caption(f"Detailed breakdown for {len(ranked)} results")

            for i, (url, item, score, feats) in enumerate(ranked, 1):
                doc = item["doc"]
                ranks = item.get("ranks", {})

                with st.container(border=True):
                    detail_header = st.container(horizontal=True)
                    with detail_header:
                        st.badge(f"#{i}", color="blue")
                        st.badge(f"Score: {score:.4f}", color="violet")
                        for eng, r in ranks.items():
                            st.badge(f"{eng.title()} rank #{r}", color="gray")

                    st.markdown(f"**{doc['title']}**")
                    st.caption(url)

                    with st.expander("Feature vector", icon=":material/data_array:"):
                        feat_df = pd.DataFrame({
                            "Feature": FEATURE_NAMES,
                            "Value": feats,
                            "Weight": mu,
                            "Contribution": np.array(feats) * mu,
                        })
                        st.dataframe(
                            feat_df,
                            column_config={
                                "Value": st.column_config.NumberColumn(format="%.4f"),
                                "Weight": st.column_config.NumberColumn(format="%.4f"),
                                "Contribution": st.column_config.NumberColumn(format="%.4f"),
                            },
                            hide_index=True,
                            alt=f"Feature breakdown for result #{i}",
                        )
        else:
            st.info("Run a search to see result analysis.", icon=":material/search:")

    # ── Tab 4: How it works ────────────────────────────────────────────

    with dev_tabs[3]:
        st.subheader("How MetaSearch learns", icon=":material/school:")
        st.markdown("""
        MetaSearch uses **Bayesian Thompson Sampling** to learn the optimal ranking
        of search results from implicit user feedback (clicks).

        #### The pipeline

        1. **Feature extraction** — Each (query, document) pair is represented by a
           17-dimensional feature vector: engine ranks, TF-IDF cosine similarities,
           and URL heuristics.

        2. **Bayesian prior** — The weight vector starts with an informed heuristic:
           rank features weighted ~3×, content ~2×, popularity ~1×.

        3. **Thompson Sampling** — Each query, a weight vector is *sampled* from
           the Bayesian posterior. This naturally explores uncertain feature weights
           while exploiting known good ones.

        4. **Click feedback** — When you click result #3 but skip #1 and #2, the
           model infers a *pairwise preference*: doc₃ > doc₁ and doc₃ > doc₂.
           The posterior is updated accordingly.

        5. **Personalization** — After 5+ clicks, you get your own Bayesian posterior
           initialized from the global model. Your subsequent clicks refine your
           personal ranking.

        #### Key files

        | File | Purpose |
        |:-----|:--------|
        | `bayesian_ranker.py` | Thompson Sampling linear bandit |
        | `user_model.py` | Hierarchical global → per-user model |
        | `features.py` | 17-dim feature extraction |
        | `click_server.py` | Online Bayesian update on each click |
        | `click_simulator.py` | Bootstrap model before real traffic |
        | `retrain.py` | Batch retrain from accumulated clicks |
        """)

        engine_stats = st.session_state.get("engine_stats", {})
        if engine_stats:
            st.subheader("Engine contribution", icon=":material/dns:")
            eng_cols = st.columns(2)
            eng_cols[0].metric("Google results", engine_stats.get("google", 0))
            eng_cols[1].metric("Bing results", engine_stats.get("bing", 0))
