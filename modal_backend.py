import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re
from typing import Dict, List

app = modal.App("quantum-ontology-v4")

image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn[standard]", "requests", "sentence-transformers[torch]", "scikit-learn"
)

class UsernameRequest(BaseModel):
    username: str

@app.function(image=image, timeout=600)
@modal.fastapi_endpoint()
def fastapi_app():
    api = FastAPI()

    @api.post("/ontology")
    async def ontology(req: UsernameRequest):
        username = req.username.lstrip("@")
        if not username:
            raise HTTPException(status_code=400, detail="нет username")

        APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"

        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": APIFY_TOKEN},
            json={"usernames": [username], "resultsLimit": 50},
            timeout=60
        )
        profile_data = resp.json()

        if not profile_data or not profile_data[0]:
            raise HTTPException(status_code=404, detail="приватный или не найден")

        profile = profile_data[0]
        texts = []
        if profile.get("biography"):
            texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:40]:
            if post.get("caption"):
                texts.append(post["caption"])

        if len(texts) < 3:
            raise HTTPException(status_code=400, detail="мало текстов")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)

        k = max(5, min(15, len(texts)//3))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for i in range(k):
            idxs = [j for j, l in enumerate(labels) if l == i]
            cluster_text = " ".join(texts[j] for j in idxs).lower()
            words = re.findall(r'[а-яёa-z]+', cluster_text)
            freq = {}
            for w in words:
                if len(w) > 3:
                    freq[w] = freq.get(w, 0) + 1
            top_words = sorted(freq, key=freq.get, reverse=True)[:5]
            name = " ".join(top_words[:3]) if top_words else f"версия {i+1}"
            weight = round(len(idxs) / len(texts) * 100, 1)
            clusters.append({"name": name, "weight": weight})

        return {
            "username": username,
            "clusters": sorted(clusters, key=lambda x: -x["weight"]),
            "total_posts": len(texts)
        }

    return api
