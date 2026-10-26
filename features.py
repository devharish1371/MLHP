import os
import json
import hashlib
import serpapi
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRACKING = {"gclid", "fbclid", "msclkid", "ref", "ref_src"}

def canonicalize(url):
    p = urlparse(url)
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    params = sorted(
        (k, v) for k, v in parse_qsl(p.query)
        if not k.startswith("utm_") and k not in TRACKING
    )

    return urlunparse((
        "https",                      
        host,
        p.path.rstrip("/") or "/",
        "",                           
        urlencode(params),
        "",                           
    ))

def rank_features(doc_url, results, engines):
    features = {}

    for engine in engines:
        ranks = [
            i for i, d in enumerate(results.get(engine, []), 1)
            if canonicalize(d["link"]) == doc_url
        ]
        r = min(ranks) if ranks else None

        features[f"rank_{engine}"] = max(0, (100 - r) / 100) if r else 0
        features[f"top1_{engine}"] = int(r == 1)
        features[f"top10_{engine}"] = int(r is not None and r <= 10)
        features[f"top50_{engine}"] = int(r is not None and r <= 50)

    features["top1count"] = sum(features[f"top1_{e}"] for e in engines)
    features["top10count"] = sum(features[f"top10_{e}"] for e in engines)
    features["top50count"] = sum(features[f"top50_{e}"] for e in engines)

    return features

def content_features(query, url, title):
    url_words = urlparse(url).path.replace("/", " ").replace("-", " ")
    domain = urlparse(url).hostname or ""
    domain = domain.removeprefix("www.").split(".")[0]

    vectorizer = TfidfVectorizer()
    texts = [query, url_words, title]

    X = vectorizer.fit_transform(texts)

    return {
        "query_url_cosine": float(cosine_similarity(X[0], X[1])[0, 0]),
        "query_title_cosine": float(cosine_similarity(X[0], X[2])[0, 0]),
        "domain_name_in_query": int(domain.lower() in query.lower())
    }

def popularity_features(url, title):
    return {
        "url_length": len(url) / 30,
        "abstract_contains_home": int(
            "home" in (url + " " + title).lower()
        ),
        "url_contains_tilde": int("~" in url)
    }

def compute_feature_vector(query, url, title, results, engines):
    features = {}
    features.update(rank_features(canonicalize(url), results, engines))
    features.update(content_features(query, url, title))
    features.update(popularity_features(url, title))

    return np.array(list(features.values()), dtype=float)


def get_feature_names(engines):
    """Return the ordered list of feature names produced by compute_feature_vector."""
    names = []
    for engine in engines:
        names.extend([
            f"rank_{engine}", f"top1_{engine}",
            f"top10_{engine}", f"top50_{engine}",
        ])
    names.extend(["top1count", "top10count", "top50count"])
    names.extend(["query_url_cosine", "query_title_cosine", "domain_name_in_query"])
    names.extend(["url_length", "abstract_contains_home", "url_contains_tilde"])
    return names