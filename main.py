from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
import numpy as np
from sklearn.cluster import KMeans
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_MODEL = "sergeyzh/rubert-tiny-turbo"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {}  # если поставишь свой токен — будет быстрее и без лимитов

def chunk_texts(texts: List[str], max_per_request: int = 8) -> List[List[str]]:
    """Разбиваем на чанки, чтобы не превышать лимит HF"""
    chunks = []
    chunks = []
    for i in range(0, len(texts), max_per_request):
        chunks.append(texts[i:i + max_per_request])
    return chunks

def get_embeddings_safely(texts: List[str]) -> np.ndarray:
    all_embeddings = []
    for chunk in chunk_texts(texts, 8):
        for _ in range(3):  # retry до 3 раз
            try:
                resp = requests.post(HF_URL, headers=HF_HEADERS, json={"inputs": chunk}, timeout=45)
                if resp.status_code == 200:
                    emb = np.array(resp.json())
                    all_embeddings.append(emb)
                    break
                elif resp.status_code == 503:
                    # модель грузится — ждём
                    import time
                    time.sleep(8)
                    continue
                else:
                    raise Exception(f"HF {resp.status_code}: {resp.text}")
            except:
                import time
                time.sleep(5)
        else:
            raise Exception("HuggingFace не отвечает после 3 попыток")
    return np.vstack(all_embeddings)

@app.get("/")
def root():
    return {"status": "QuantID 81 онлайн · 2025", "artist": "Влада Садик"}

@app.post("/ontology")
async def ontology(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "").lstrip("@").strip()
        if not username:
            return JSONResponse({"error": "укажи username"}, status_code=400)

        # Apify
        apify_resp = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
            json={"usernames": [username], "resultsLimit": 60},
            timeout=120
        )
        profile_data = apify_resp.json()

        if not profile_data:
            return JSONResponse({"error": "профиль не найден или приватный"}, status_code=404)

        profile = profile_data[0]
        texts = []
        if profile.get("biography"):
            texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:50]:
            if caption := post.get("caption"):
                texts.append(caption.strip())

        if len(texts) < 4:
            return JSONResponse({"error": "слишком мало текстов для анализа"}, status_code=400)

        # ← САМЫЙ ВАЖНЫЙ БЛОК — безопасные эмбеддинги
        embeddings = get_embeddings_safely(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        k = max(5, min(15, len(texts)//3))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for i in range(k):
            idxs = np.where(labels == i)[0]
            cluster_text = " ".join(texts[j] for j in idxs).lower()
            words = re.findall(r'[а-яё]{4,}', cluster_text)
            top_words = [w for w, _ in __import__("collections").Counter(words).most_common(8)]
            name = " ".join(top_words[:4]).capitalize() if top_words else f"версия {i+1}"
            weight = round(len(idxs) / len(texts) * 100, 1)
            clusters.append({"name": name, "weight": weight})

        return {
            "artist": f"@{username}",
            "identities": sorted(clusters, key=lambda x: -x["weight"]),
            "posts": len(texts),
            "model": "sergeyzh/rubert-tiny-turbo",
            "status": "success"
        }

    except Exception as e:
        return JSONResponse({"error": f"внутренняя ошибка: {str(e)}"}, status_code=500)
