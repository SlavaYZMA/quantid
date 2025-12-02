from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
import numpy as np
from sklearn.cluster import KMeans
import time
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
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HF_HEADERS = {}  # ← сюда можно вставить свой HF токен, если сделаешь (тогда будет мгновенно)

def get_embeddings_safe(texts: List[str]) -> np.ndarray:
    chunks = [texts[i:i+7] for i in range(0, len(texts), 7)]
    all_emb = []

    for i, chunk in enumerate(chunks):
        for attempt in range(6):  # максимум 6 попыток = ~2 минуты
            try:
                print(f"Запрос эмбеддингов чанк {i+1}/{len(chunks)}, попытка {attempt+1}")
                resp = requests.post(
                    HF_URL,
                    headers=HF_HEADERS,
                    json={"inputs": chunk, "options": {"wait_for_model": True}},
                    timeout=90
                )
                if resp.status_code == 200:
                    emb = np.array(resp.json())
                    all_emb.append(emb)
                    print(f"Чанк {i+1} получен успешно")
                    break
                elif resp.status_code == 503:
                    print("Модель ещё грузится на HF, ждём 12 сек...")
                    time.sleep(12)
                    continue
                else:
                    print(f"HF ошибка {resp.status_code}: {resp.text}")
                    raise Exception(f"HF {resp.status_code}")
            except Exception as e:
                print(f"Ошибка запроса: {e}")
                if attempt == 5:
                    raise
                time.sleep(10)
        else:
            raise Exception("Не удалось получить эмбеддинги после всех попыток")

    return np.vstack(all_emb)

@app.get("/")
def root():
    return {"msg": "Квантовая онтология 81 · Влада Садик · жива"}

@app.post("/ontology")
async def ontology(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "").lstrip("@").strip()

        if not username:
            return JSONResponse({"error": "укажи username"}, status_code=400)

        # === Apify ===
        print(f"Запуск Apify для @{username}")
        apify = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
            json={"usernames": [username], "resultsLimit": 60},
            timeout=120
        ).json()

        if not apify:
            return JSONResponse({"error": "профиль не найден"}, status_code=404)

        profile = apify[0]
        texts = []
        if profile.get("biography"):
            texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:50]:
            if caption := post.get("caption"):
                texts.append(caption.strip())

        if len(texts) < 4:
            return JSONResponse({"error": "мало текста"}, status_code=400)

        print(f"Собрано {len(texts)} текстов, начинаю эмбеддинги...")

        # === Эмбеддинги с ожиданием загрузки модели ===
        embeddings = get_embeddings_safe(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        k = max(5, min(15, len(texts)//3))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for i in range(k):
            idxs = [j for j, l in enumerate(labels) if l == i]
            cluster_text = " ".join(texts[j] for j in idxs).lower()
            words = re.findall(r'[а-яё]{4,}', cluster_text)
            top = [w for w, _ in __import__("collections").Counter(words).most_common(7)]
            name = " ".join(top[:4]).capitalize() if top else f"архетип {i+1}"
            weight = round(len(idxs)/len(texts)*100, 1)
            clusters.append({"name": name, "weight": weight})

        print(f"Успешно! @{username} — {len(clusters)} архетипов")

        return {
            "artist": f"@{username}",
            "identities": sorted(clusters, key=lambda x: -x["weight"]),
            "posts": len(texts),
            "model": "sergeyzh/rubert-tiny-turbo"
        }

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        print(error_msg)
        return JSONResponse({"error": error_msg}, status_code=500)
