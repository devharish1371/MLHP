import streamlit as st
import numpy as np
import uuid
from urllib.parse import urlencode

from metasearch import search, rank_with_features
from model_store import load_model
from search_cache import generate_session_id, cache_results
from bayesian_ranker import FEATURE_NAMES, N_FEATURES

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MetaSearch",
    page_icon=":material/travel_explore:",
    layout="wide",
)

# ── Persistent user identity ───────────────────────────────────────────────

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

# ── Load model ─────────────────────────────────────────────────────────────

model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("MetaSearch", icon=":material/travel_explore:")
    st.caption("Bayesian meta-search engine")

    st.subheader("Ranking method", icon=":material/tune:")

    ranker_choice = st.segmented_control(
        "Ranking strategy",
        options=["Explore", "Exploit", "Baseline"],
        default="Explore",
        label_visibility="collapsed",
    )

    with st.expander("What do these mean?", icon=":material/help:"):
        st.markdown("""
        - **Explore** — Thompson Sampling draws a random weight vector
          from the Bayesian posterior each query.
          Naturally balances learning vs ranking quality.
        - **Exploit** — Uses the MAP estimate (posterior mean).
          Best ranking quality given current knowledge, no exploration.
        - **Baseline** — Uniform weights (all features equal).
          Ignores learned model entirely.
        """)

    st.subheader("Model status", icon=":material/model_training:")

    mu = model.global_ranker.mu
    col1, col2 = st.columns(2)
    col1.metric("Updates", f"{model.global_ranker.n_updates:,}")
    col2.metric("Users", len(model.user_rankers))

    user_ranker = model.get_ranker(st.session_state["user_id"])
    if user_ranker is not model.global_ranker:
        st.badge("Personalized", icon=":material/person:", color="green")
    else:
        clicks_so_far = model.user_click_counts.get(st.session_state["user_id"], 0)
        remaining = max(0, model.MIN_CLICKS_FOR_PERSONAL - clicks_so_far)
        st.caption(f":material/public: Global model · {remaining} clicks to personalize")

    with st.expander("Posterior weights", icon=":material/bar_chart:"):
        # Sort by absolute weight magnitude
        sorted_indices = np.argsort(np.abs(mu))[::-1]
        for idx in sorted_indices:
            name = FEATURE_NAMES[idx]
            val = mu[idx]
            sign = "+" if val >= 0 else ""
            st.caption(f"`{name}` → **{sign}{val:.3f}**")

# ── Main content ───────────────────────────────────────────────────────────

st.title("MetaSearch", icon=":material/travel_explore:")
st.caption("Search across Google & Bing · Re-ranked with Bayesian Thompson Sampling · Learns from your clicks")

query = st.text_input(
    "Search the web",
    placeholder="e.g. learning to rank, information retrieval, …",
    label_visibility="collapsed",
)

search_clicked = st.button(
    "Search",
    icon=":material/search:",
    type="primary",
    width="stretch",
)

if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.", icon=":material/warning:")
    else:
        engines = ["google", "bing"]
        clean_query = query.strip()

        with st.spinner("Searching Google & Bing…"):
            try:
                results = {engine: search(engine, clean_query) for engine in engines}
                session_id = generate_session_id()

                # Select weights
                user_ranker = model.get_ranker(st.session_state["user_id"])

                if ranker_choice == "Explore":
                    w = user_ranker.sample_weights()
                elif ranker_choice == "Exploit":
                    w = user_ranker.map_weights()
                else:
                    w = np.ones(N_FEATURES)

                # Rank with features
                ranked = rank_with_features(clean_query, results, engines, w)

                # Cache features for click-time Bayesian update
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

                # Count engines contributing
                total_google = len(results.get("google", []))
                total_bing = len(results.get("bing", []))
                st.session_state["engine_stats"] = {
                    "google": total_google,
                    "bing": total_bing,
                }

            except Exception as e:
                st.error(f"Search failed: {e}", icon=":material/error:")

# ── Display results ────────────────────────────────────────────────────────

ranked = st.session_state.get("ranked", [])

if ranked:
    engine_stats = st.session_state.get("engine_stats", {})
    stat_cols = st.columns(3)
    stat_cols[0].metric("Results", len(ranked))
    stat_cols[1].metric("From Google", engine_stats.get("google", 0))
    stat_cols[2].metric("From Bing", engine_stats.get("bing", 0))

    for i, (url, item, score, _feats) in enumerate(ranked, 1):
        doc = item["doc"]
        ranks = item.get("ranks", {})

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
            st.caption(url)
            if doc.get("snippet"):
                st.markdown(doc["snippet"], help=f"Score: {score:.4f}")

            tag_row = st.container(horizontal=True)
            with tag_row:
                st.badge(f"#{i}", color="blue")
                st.badge(f"Score {score:.3f}", color="violet")
                for eng, r in ranks.items():
                    st.badge(f"{eng.title()} #{r}", color="gray")

elif st.session_state.get("query"):
    st.info("No results found.", icon=":material/search_off:")
