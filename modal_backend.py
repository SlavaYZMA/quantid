from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re
import uvicorn
import os

app = FastAPI()

# CORS для Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Твой Apify токен
APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"

# Загружаем словарь (из той же папки)
try:
    with open('vocab_id.json', 'r', encoding='utf-8') as f:
        VOCAB = json.load(f)
except:
    VOCAB = []  # fallback

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
            params={"token": APIFY_TOKEN},
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

        # Zero-shot
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
            weight = len(idxs) / len(texts) * 15
            clusters.append({"name": name, "weight": round(weight, 2)})

        return {
            "username": username,
            "clusters": sorted(clusters, key=lambda x: -x["weight"]),
            "total_posts": len(texts)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
