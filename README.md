# MetaSearch

A **Bayesian meta-search engine** that aggregates results from Google and Bing, re-ranks them using **Thompson Sampling**, and continuously learns from user clicks through **hierarchical Bayesian online learning**.

Built as a Machine Learning for Hyperlink Prediction (MLHP) project.

---

## Idea

Traditional search engines return a single, fixed ranking. **MetaSearch** takes a different approach:

1. **Aggregate** — Query multiple search engines (Google, Bing via SerpApi) and merge their results.
2. **Extract features** — Compute a 17-dimensional feature vector for every (query, document) pair: engine ranks, TF-IDF similarity scores, and URL heuristics.
3. **Rank with Thompson Sampling** — Instead of fixed weights, sample from a Bayesian posterior over the weight vector each query. This naturally balances **exploration** (trying uncertain rankings) with **exploitation** (using the best-known ranking).
4. **Learn from clicks** — Every click triggers an online Bayesian update: if a user clicks result #3 but skipped #1 and #2, the model learns that result #3's features correlate with relevance.
5. **Personalize** — After enough clicks, each user gets their own posterior (initialized from the global model). The global model improves for everyone; the personal model adapts to individual preferences.

### The Bayesian Pipeline

```
Prior: w ~ N(μ₀, v² B₀⁻¹)     ← Heuristic initialization
                                   (rank features ~3×, content ~2×, popularity ~1×)
        │
        ▼
Thompson Sampling: w̃ ~ N(μ, v² B⁻¹)  ← Sample weights each query
        │
        ▼
Score: relevance(doc) = w̃ · x(doc)   ← Linear scoring
        │
        ▼
Click feedback: user prefers doc_i over doc_j
        │
        ▼
Pairwise update:
    B ← B + (xᵢ - xⱼ)(xᵢ - xⱼ)ᵀ / v²
    f ← f + (xᵢ - xⱼ) / v²           ← Bayesian posterior update
        │
        ▼
New posterior: w | data ~ N(B⁻¹f, v² B⁻¹)
```

---

## Architecture

```
User query
   │
   ▼
app.py ──► metasearch.py ──► SerpApi (Google + Bing)
   │              │
   │      features.py (17-dim feature vector)
   │              │
   │      bayesian_ranker.py (Thompson Sampling / MAP)
   │      user_model.py (hierarchical: global → per-user)
   │              │
   │      search_cache.py (cache features in SQLite)
   │
   ▼ user clicks a result
click_server.py
   ├── click_logger.py (log to clicks.db)
   ├── search_cache.py (read cached features)
   ├── model_store.py  (load model)
   ├── user_model.py   (pairwise Bayesian update)
   └── model_store.py  (save updated model)
         │
         ▼ redirect to target page
```

---

## Features

### 17-Dimensional Feature Vector

Each (query, document) pair is represented by 17 features:

| Category | Features | Description |
|:---------|:---------|:------------|
| **Rank features** (per engine) | `rank_{engine}`, `top1_{engine}`, `top10_{engine}`, `top50_{engine}` | Normalized rank position and top-K indicators |
| **Aggregate rank** | `top1count`, `top10count`, `top50count` | How many engines rank the doc in their top-K |
| **Content** | `query_url_cosine`, `query_title_cosine`, `domain_name_in_query` | TF-IDF cosine similarity between query and URL/title |
| **Popularity** | `url_length`, `abstract_contains_home`, `url_contains_tilde` | URL heuristics for authority estimation |

### Bayesian Thompson Sampling

- **Exploration vs Exploitation**: Thompson Sampling automatically explores uncertain rankings early on and converges to the optimal ranking as more click data is collected.
- **Online learning**: The model updates in real-time — every click immediately improves the posterior.
- **No retraining needed**: The Bayesian approach incrementally incorporates new evidence without expensive batch retraining (though batch retrain is available for recovery).

### Hierarchical Personalization

- **Global model**: Learns from all users' clicks — improves ranking for everyone.
- **Per-user model**: After 5+ clicks, each user gets a personalized posterior initialized from the global model.
- **Cold-start**: New users seamlessly use the global model until enough data is collected.

### Click Logging & Redirect Proxy

- Every click passes through a Flask proxy that logs `(timestamp, query, url, position, session_id, user_id)` to SQLite.
- The proxy simultaneously performs the Bayesian update before redirecting the user to the target page.
- Full click history is preserved for offline analysis and batch retraining.

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/devharish1371/MLHP.git
cd MLHP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure SerpApi key

Create a `.env` file in the project root:

```bash
SERPAPI_KEY=your_api_key_here
```

Get your key from [serpapi.com](https://serpapi.com/).

### 3. Bootstrap the model (recommended)

Run the click simulator to initialize the Bayesian ranker with meaningful weights:

```bash
python click_simulator.py --n-sessions 500
```

This creates a `models/` directory with the trained model state.

### 4. Run the application

Open two terminals:

**Terminal 1 — Click logging server:**
```bash
python click_server.py
```

**Terminal 2 — Streamlit app:**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
MLHP/
├── app.py                  # Streamlit search interface
├── metasearch.py           # SerpApi client, result merging, ranking
├── features.py             # URL canonicalization, 17-feature extraction
├── bayesian_ranker.py      # Thompson Sampling linear bandit
├── user_model.py           # Hierarchical Bayesian user model
├── model_store.py          # JSON persistence with auto-backup
├── search_cache.py         # SQLite feature cache (app ↔ click server)
├── click_server.py         # Flask redirect proxy + online Bayesian updates
├── click_logger.py         # SQLite click event storage
├── click_simulator.py      # Cascade click model for bootstrapping
├── retrain.py              # Batch retrain from accumulated clicks
├── inspect_clicklog.ipynb  # Jupyter notebook for click data inspection
├── requirements.txt        # Python dependencies
├── .env                    # SerpApi key (not committed)
├── .gitignore              # Excludes .env, .venv, models/, clicks.db
├── .streamlit/
│   └── config.toml         # Dark theme configuration
└── models/                 # Trained model state (auto-created)
```

---

## Usage

### Ranking Methods

| Method | Behavior |
|:-------|:---------|
| **Explore** (Thompson Sampling) | Samples weights from the posterior — balances learning and quality |
| **Exploit** (MAP Estimate) | Uses the posterior mean — best quality, no exploration |
| **Baseline** (Uniform) | All feature weights = 1 — ignores learned model |

### Batch Retrain

After collecting click data, retrain the global model:

```bash
python retrain.py                  # Incremental (keeps existing model)
python retrain.py --from-scratch   # Full retrain from prior
```

### Inspect Click Data

```bash
jupyter notebook inspect_clicklog.ipynb
```

---

## How Learning Works

### Phase 1: Bootstrap (before real users)

```bash
python click_simulator.py --n-sessions 1000
```

The simulator generates synthetic search sessions, simulates user clicks using a cascade model (geometric examination decay + sigmoid click probability), and feeds pairwise preferences into the Bayesian ranker.

### Phase 2: Deploy & Collect

Once real users start searching, every click automatically triggers:
1. A pairwise comparison: "user preferred result #3 over #1 and #2"
2. A Bayesian posterior update on the weight vector
3. Model persistence to disk

### Phase 3: Personalization

After 5+ clicks, a user gets their own posterior initialized from the global model. Their subsequent clicks update only their personal model, allowing personalized rankings.

### Phase 4: Global Optimization

Run `python retrain.py` periodically to rebuild the global model from all accumulated click data. This propagates collective learning and refreshes all user priors.

---

## Technical Details

### URL Canonicalization

Different engines return different URLs for the same page. MetaSearch normalizes URLs by:
- Lowercasing hostnames and stripping `www.`
- Removing tracking parameters (`utm_*`, `gclid`, `fbclid`, `msclkid`)
- Normalizing protocol to HTTPS
- Removing trailing slashes

### Feature Engineering

Features are extracted using:
- **scikit-learn** `TfidfVectorizer` for query-document cosine similarity
- Per-engine rank normalization: `(100 - rank) / 100`
- Binary indicators for top-1, top-10, top-50 presence
- Cross-engine agreement counts

### Model Persistence

The hierarchical model (global + per-user posteriors) is serialized as JSON with:
- Precision matrix `B` (d×d)
- Reward accumulator `f` (d-vector)
- Noise variance `v²`
- Per-user click counts and model states

Atomic writes (write to temp file → rename) prevent corruption. The last 5 backups are automatically retained.

---

## Dependencies

- **Streamlit** — Search interface and model dashboard
- **Flask** — Click redirect proxy
- **NumPy** — Linear algebra for Bayesian updates
- **scikit-learn** — TF-IDF feature extraction
- **SerpApi** — Google & Bing search API
- **python-dotenv** — Secure API key management

---

## License

This project is for educational purposes as part of the MLHP coursework.
