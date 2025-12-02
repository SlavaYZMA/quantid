from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
import json
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем spaCy модель (лёгкая, 50 МБ)
nlp = spacy.load("ru_core_news_sm")

# Словарь идентичностей (загружаем из файла)
try:
    with open('vocab_id.json', 'r', encoding='utf-8') as f:
        VOCAB = json.load(f)
except:
    VOCAB = []

@app.post("/ontology")
async def ontology(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "").lstrip("@").strip()
        if not username:
            return JSONResponse({"error": "нет username"}, status_code=400)

        # Apify
        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
            json={"usernames": [username], "resultsLimit": 50},
            timeout=60
        )
        profile_data = resp.json()

        if not profile_data or not profile_data[0]:
            return JSONResponse({"error": "приватный или не найден"}, status_code=404)

        profile = profile_data[0]
        texts = []
        if profile.get("biography"):
            texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:40]:
            if post.get("caption"):
                texts.append(post["caption"])

        if len(texts) < 3:
            return JSONResponse({"error": "мало текстов"}, status_code=400)

        # TF-IDF + SVD для эмбеддингов (без torch, 50 МБ)
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,3))
        tfidf = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=100)
        embeddings = svd.fit_transform(tfidf)

        k = max(5, min(12, len(texts)//3))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for i in range(k):
            idxs = [j for j, l in enumerate(labels) if l == i]
            cluster_text = " ".join(texts[j] for j in idxs).lower()
            words = re.findall(r'[а-яё]+', cluster_text)
            freq = Counter(w for w in words if len(w) > 3)
            top = freq.most_common(5)
            name = " ".join(word for word, _ in top[:3]) if top else f"версия {i+1}"
            weight = round(len(idxs) / len(texts) * 100, 1)
            clusters.append({"name": name, "weight": weight})

        return {
            "username": username,
            "clusters": sorted(clusters, key=lambda x: -x["weight"]),
            "total_posts": len(texts)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
