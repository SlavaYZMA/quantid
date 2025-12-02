from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
import numpy as np
from sklearn.cluster import KMeans
import os

app = FastAPI()

# ← ЭТО САМОЕ ГЛАВНОЕ: РАЗРЕШАЕМ ВСЕМ (Netlify и т.д.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # или конкретно "https://quantid.netlify.app"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://api-inference.huggingface.co/models/sergeyzh/rubert-tiny-turbo"
HEADERS = {}  # если у тебя есть HF токен — вставь сюда "Bearer hf_..."

def get_embeddings(texts):
    resp = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": texts}, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"HF error {resp.status_code}: {resp.text}")
    return np.array(resp.json())

@app.get("/")
async def root():
    return {"msg": "QuantID онлайн"}

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
            json={"usernames": [username], "resultsLimit": 60},
            timeout=120
        )
        profile_data = resp.json()
        if not profile_data or not profile_data[0]:
            return JSONResponse({"error": "приватный профиль или не найден"}, status_code=404)

        profile = profile_data[0]
        texts = []
        if profile.get("biography"): texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:50]:
            if post.get("caption"): texts.append(post["caption"])

        if len(texts) < 4:
            return JSONResponse({"error": "мало текстов"}, status_code=400)

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
            "identities": sorted(clusters, key=lambda x: -x["weight"]),
            "posts": len(texts)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ← ЭТО ЧТОБЫ RENDER НЕ УБИВАЛ ПРОЦЕСС СРАЗУ (антиспячка)
@app.on_event("startup")
async def startup_event():
    print("QuantID готов к работе")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), workers=1)
