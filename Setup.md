# MetaSearch — Setup Instructions

## 1. Install dependencies

Run:

```bash
python -m pip install streamlit flask numpy serpapi scikit-learn
```

## 2. Get your SerpApi API key

1. Create an account at [SerpApi](https://serpapi.com/).
2. Get your API key from your dashboard.
3. Open `metasearch.py` and find:

```python
api_key = "YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual SerpApi key.

## 3. Bootstrap the model (optional but recommended)

Run the click simulator to initialize the Bayesian ranker with meaningful
weights before any real user traffic:

```bash
python click_simulator.py --n-sessions 200
```

This creates a `models/` directory with the trained model state.

## 4. Run the application

Open two terminals in the project directory.

**Terminal 1 — Start the click logging server:**

```bash
python click_server.py
```

**Terminal 2 — Start the Streamlit app:**

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal to use MetaSearch.

## 5. Inspect click data

Clicking a result title logs the search query, destination URL, displayed position, and timestamp to `clicks.db`.

Run the inspection script to view the recorded click data:

```bash
python inspect_clicklog.py
```

## 6. Batch retrain (periodic)

After collecting click data, retrain the global model:

```bash
python retrain.py                  # incremental (keeps existing model)
python retrain.py --from-scratch   # full retrain from prior
```

## Project files

### Core search
- `app.py` — Streamlit search interface with model status sidebar.
- `metasearch.py` — Search retrieval, result merging, and ranking.
- `features.py` — URL canonicalization and 17-feature extraction.

### Bayesian ranking
- `bayesian_ranker.py` — Thompson Sampling linear bandit with heuristic prior.
- `user_model.py` — Hierarchical Bayes: global prior → per-user posterior.
- `model_store.py` — JSON persistence with auto-backup for model state.

### Click pipeline
- `click_server.py` — Flask redirect proxy with online Bayesian updates.
- `click_logger.py` — Stores click events in SQLite.
- `search_cache.py` — Caches feature vectors for click-time model updates.

### Training & simulation
- `click_simulator.py` — Cascade click model for bootstrapping.
- `retrain.py` — Batch retrain from accumulated click data.
- `inspect_clicklog.ipynb` — Displays the click database.
- `click_sim.ipynb` — Notebook for click simulation experiments.

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
   ├──► click_logger.py (log to clicks.db)
   ├──► search_cache.py (read cached features)
   ├──► model_store.py  (load model)
   ├──► user_model.py   (pairwise Bayesian update)
   └──► model_store.py  (save updated model)
         │
         ▼ redirect to target page
```

## Current status

- Google and Bing search through SerpApi.
- 17-dimensional feature extraction (rank, content, popularity).
- Bayesian Thompson Sampling ranker with heuristic prior initialization.
- Hierarchical personalization (global → per-user posterior).
- Online Bayesian posterior update on every click.
- Click simulator for bootstrapping before real traffic.
- Batch retrain script for periodic model refresh.
- Click logging and SQLite storage with session tracking.

**Note:** Keep your API key private and never commit it to a public repository.
