import os
import json
import hashlib
import serpapi
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import numpy as np
from dotenv import load_dotenv
from features import *

load_dotenv()  # reads .env file

_api_key = os.environ.get("SERPAPI_KEY", "")
if not _api_key:
    raise RuntimeError("SERPAPI_KEY not set. Add it to .env or export it as an env var.")
client = serpapi.Client(api_key=_api_key)

def search(engine, q, k=10):
    return client.search({"engine": engine, "q": q}).get("organic_results", [])[:k]
    
def union(results):
    V = {}

    for engine, docs in results.items():
        for i, r in enumerate(docs, 1):
            url = canonicalize(r["link"])
            doc = {
                "title": r.get("title", ""),
                "link": r["link"],
                "snippet": r.get("snippet", "")
            }
            if url not in V:
                V[url] = {"doc": doc, "ranks": {}}

            V[url]["ranks"][engine] = r.get("position", i)

    return V    

def rank(query, results, engines, w):
    V = union(results)
    scores = []

    for url, doc in V.items():
        x = compute_feature_vector(query, url, doc["doc"]["title"], results, engines)
        score = np.dot(w, x)
        scores.append((url, doc, score))

    return sorted(scores, key=lambda x: x[2], reverse=True)

def rank_with_features(query, results, engines, w):
    """Like rank(), but also returns each document's feature vector."""
    V = union(results)
    scored = []

    for url, doc in V.items():
        x = compute_feature_vector(query, url, doc["doc"]["title"], results, engines)
        score = float(np.dot(w, x))
        scored.append((url, doc, score, x.tolist()))

    return sorted(scored, key=lambda t: t[2], reverse=True)

