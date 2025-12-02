import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # на всякий случай
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests, re, json, os

# Отключаем лишние предупреждения и параллелизм токенизаторов
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # гарантируем, что torch не ищет GPU

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Самая лёгкая русскоязычная модель — весит ~90 МБ, стартует за 2 секунды
model = SentenceTransformer("cointegrated/rubert-tiny2")

@app.post("/ontology")
async def ontology(request: Request):
    try:
        data = await request.json()
        username = data.get("username", "").lstrip("@").strip()
        if not username: return JSONResponse({"error": "нет username"}, 400)

        # Apify (оставляем твой токен)
        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
            json={"usernames": [username], "resultsLimit": 50},
            timeout=90
        )
        profile_data = resp.json()
        if not profile_data: return JSONResponse({"error": "не найден или приватный"}, 404)

        texts = []
        profile = profile_data[0]
        if profile.get("biography"): texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:40]:
            if post.get("caption"): texts.append(post["caption"])

        if len(texts) < 3: return JSONResponse({"error": "мало текстов"}, 400)

        # Кластеризация
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=False)
        k = max(5, min(12, len(texts)//4))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for i in range(k):
            idxs = [j for j, l in enumerate(labels) if l == i]
            cluster_text = " ".join(texts[j] for j in idxs).lower()
            words = re.findall(r'[а-яё]+', cluster_text)
            freq = {}
            for w in words:
                if len(w) > 3: freq[w] = freq.get(w, 0) + 1
            top = sorted(freq, key=freq.get, reverse=True)[:4]
            name = " ".join(top) if top else f"кластер {i+1}"
            weight = round(len(idxs)/len(texts)*100, 1)
            clusters.append({"name": name, "weight": weight})

        return {
            "username": username,
            "clusters": sorted(clusters, key=lambda x: -x["weight"]),
            "total_posts": len(texts)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
