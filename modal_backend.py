import modal
import requests
import json
from pathlib import Path
from transformers import pipeline
from typing import List, Dict

app = modal.App("quantum-ontology-v2")

# ← Теперь всё берётся из репозитория автоматически
image = modal.Image.debian_slim().pip_install(
    "requests",
    "torch",
    "transformers",
    "sentencepiece",
    "accelerate"
)

# Загружаем твой словарь 81 идентичности
VOCAB_PATH = Path("vocab_id.json")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    IDENTITIES = json.load(f)

# Zero-shot классификатор (лучший мультиязычный 2025 года)
classifier = None

@app.function(image=image, timeout=600, gpu="any")
@modal.web_endpoint(method="POST")
async def ontology(data: Dict):
    global classifier
    username = data.get("username", "").lstrip("@")
    if not username:
        return {"error": "нет username"}, 400

    # 1. Apify — берём тексты
    APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"
    resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
        params={"token": APIFY_TOKEN},
        json={"usernames": [username], "resultsLimit": 60},
        timeout=90
    )
    profile_data = resp.json()
    if not profile_data or not profile_data[0]:
        return {"error": "приватный профиль или не найден"}, 404

    texts: List[str] = []
    profile = profile_data[0]
    if profile.get("biography"):
        texts.append(profile["biography"])
    for post in profile.get("latestPosts", [])[:50]:
        if caption := post.get("caption"):
            texts.append(caption)

    if len(texts) < 3:
        return {"error": "мало текстов"}, 400

    # 2. Инициализируем zero-shot один раз
    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=0 if modal.is_local() else -1,  # GPU в Modal, CPU локально
            batch_size=8
        )

    # 3. Готовим гипотезы
    candidate_labels = [item["name_ru"] for item in IDENTITIES]
    hypotheses = [item["hypothesis_ru"] for item in IDENTITIES]

    # 4. Классифицируем каждый пост
    scores_per_identity = [0.0] * len(IDENTITIES)

    # Делаем батчами, чтобы не убить память
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        results = classifier(batch, candidate_labels, hypothesis_template="{}", multi_label=True)
        for res in results:
            for label, score in zip(res["labels"], res["scores"]):
                idx = candidate_labels.index(label)
                scores_per_identity[idx] += score

    # 5. Нормализуем в проценты
    total = sum(scores_per_identity)
    if total == 0:
        total = 1

    clusters = []
    for i, score in enumerate(scores_per_identity):
        percent = round(score / total * 100, 1)
        if percent >= 1.0:  # отсекаем шум
            identity = IDENTITIES[i]
            clusters.append({
                "name": identity["name_ru"],
                "weight": percent,
                "color": identity["color"],
                "valence": identity["valence"],
                "core_fear": identity["core_fear"],
                "core_desire": identity["core_desire"]
            })

    # Сортируем по убыванию
    clusters.sort(key=lambda x: -x["weight"])

    # Определяем доминирующую валентность и цвет для пены
    dominant = clusters[0] if clusters else IDENTITIES[0]
    dominant_color = dominant["color"]

    return {
        "username": username,
        "clusters": clusters[:12],           # отдаём топ-12
        "dominant_color": dominant_color,
        "total_posts": len(texts),
        "message": "квантовая онтология v2 готова"
    }
