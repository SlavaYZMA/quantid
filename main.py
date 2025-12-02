from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
from sklearn.cluster import KMeans
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Бесплатный HF Inference API (можно свой токен вставить, но и без него работает)
HF_API_URL = "https://api-inference.huggingface.co/models/sergeyzh/rubert-tiny-turbo"
HEADERS = {"Authorization": "Bearer hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"}  # можно оставить пустым

def get_embeddings(texts):
    response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": texts})
    if response.status_code != 200:
        raise Exception(f"HF error: {response.text}")
    return np.array(response.json())

@app.get("/")
def root():
    return {"msg": "QuantID online (free version)"}

@app.post("/ontology")
async def ontology(request: Request):
    data = await request.json()
    username = data.get("username", "").lstrip("@").strip()

    # Apify (твой токен)
    resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
        params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
        json={"usernames": [username], "resultsLimit": 60},
        timeout=90
    )
    profile = resp.json()[0]

    texts = []
    if profile.get("biography"): texts.append(profile["biography"])
    for post in profile.get("latestPosts", [])[:50]:
        if post.get("caption"): texts.append(post["caption"])

    if len(texts) < 4:
        return {"error": "мало текстов"}

    # Эмбеддинги через HF API
    embeddings = get_embeddings(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    k = max(5, min(14, len(texts)//3))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        cluster_text = " ".join(texts[j] for j in idxs).lower()
        words = re.findall(r'[а-яё]{4,}', cluster_text)
        top = [w for w, _ in __import__("collections").Counter(words).most_common(7)]
        name = " ".join(top[:4]).capitalize() if top else f"кластер {i+1}"
        weight = round(len(idxs)/len(texts)*100, 1)
        clusters.append({"name": name, "weight": weight})

    return {
        "artist": f"@{username}",
        "identities": sorted(clusters, key=lambda x: -x["weight"]),
        "posts": len(texts),
        "mode": "free tier (HF Inference API)"
    }
