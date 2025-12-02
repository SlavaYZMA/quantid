import modal
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ← МОНТИРУЕМ ФАЙЛ ПРАВИЛЬНО (Modal 2025 способ)
image = modal.Image.debian_slim().pip_install(
    "requests",
    "sentence-transformers[torch]",
    "scikit-learn"
).copy_local_file("vocab_id.json", "/vocab_id.json")

app = modal.App("quantum-ontology-final", image=image)

@app.function(image=image, timeout=300)
@modal.web_endpoint(method="POST")
async def ontology(data: dict):
    username = data.get("username", "").lstrip("@").strip()
    if not username:
        return {"error": "нет username"}, 400

    # ТВОЙ APIFY ТОКЕН
    APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"

    try:
        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
            params={"token": APIFY_TOKEN},
            json={"usernames": [username], "resultsLimit": 50},
            timeout=60
        )
        profile_data = resp.json()

        if not profile_data or not profile_data[0]:
            return {"error": "приватный профиль"}, 404

        profile = profile_data[0]
        texts = []
        if profile.get("biography"):
            texts.append(profile["biography"])
        for post in profile.get("latestPosts", [])[:40]:
            if post.get("caption"):
                texts.append(post["caption"])

        if len(texts) < 3:
            return {"error": "мало текстов"}, 400

        # Читаем твой словарь
        with open("/vocab_id.json", "r", encoding="utf-8") as f:
            VOCAB = json.load(f)

        model = SentenceTransformer('intfloat/multilingual-e5-large')
        post_vectors = model.encode(texts)

        # Прототипы из словаря
        prototypes = []
        for item in VOCAB:
            phrases = item.get("phrases_ru", []) + item.get("phrases_en", [])
            if phrases:
                proto_vec = model.encode(phrases).mean(axis=0)
                prototypes.append({
                    "name": item["name_ru"],
                    "vector": proto_vec
                })

        if not prototypes:
            return {"error": "словарь пуст"}, 500

        # Считаем близость
        results = []
        for vec in post_vectors:
            scores = []
            for proto in prototypes:
                sim = cosine_similarity([vec], [proto["vector"]])[0][0]
                if sim > 0.55:
                    scores.append({"name": proto["name"], "score": float(sim)})
            if scores:
                best = max(scores, key=lambda x: x["score"])
                results.append(best["name"])

        # Агрегация
        if not results:
            return {"clusters": [{"name": "неопределённая идентичность", "weight": 100.0}]}

        from collections import Counter
        count = Counter(results)
        total = len(results)
        clusters = []
        for name, cnt in count.most_common():
            clusters.append({
                "name": name,
                "weight": round(cnt / total * 100, 1)
            })

        return {
            "username": username,
            "clusters": clusters,
            "total_posts": len(texts)
        }

    except Exception as e:
        return {"error": str(e)}, 500