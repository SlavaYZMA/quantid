# generate_ontology.py — запускаешь один раз на любом @username
import json
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from datetime import datetime

# === НАСТРОЙКИ ===
APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"
USERNAME = input("Введи @username (без @): ").strip().replace("@", "")

# 1. Получаем посты
url = "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items"
payload = {"usernames": [USERNAME], "resultsLimit": 50}
headers = {"Content-Type": "application/json"}

print("Читаю Instagram...")
data = requests.post(url + "?token=" + APIFY_TOKEN, json=payload).json()

if not data or not data[0]:
    print("Профиль приватный или ошибка Apify")
    exit()

profile = data[0]
bio = profile.get("biography", "")
posts = [p["caption"] for p in profile.get("latestPosts", []) if p.get("caption")]
texts = [bio] + posts if bio else posts

if not texts:
    print("Нет текстов")
    exit()

print(f"Найдено {len(texts)} текстов")

# 2. Векторизация
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# 3. Кластеризация
best_k = 5
best_score = -1
for k in range(5, min(16, len(texts)//2 + 1)):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)
    score = silhouette_score(vectors, labels)
    if score > best_score:
        best_score = score
        best_k = k

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(vectors)

# 4. Названия кластеров
clusters = []
for i in range(best_k):
    idxs = np.where(labels == i)[0]
    words = " ".join([texts[j] for j in idxs]).lower()
    words = re.findall(r'[а-яё]+', words)
    top_words = [w for w, c in Counter(words).most_common(6)]
    name = " ".join(top_words[:3]) if top_words else f"кластер {i+1}"
    weight = len(idxs) / len(texts)
    clusters.append({"name": name, "weight": float(weight), "size": len(idxs)})

# 5. Сохраняем
result = {
    "username": USERNAME,
    "generated_at": datetime.now().isoformat(),
    "total_posts": len(texts),
    "clusters_count": best_k,
    "clusters": clusters
}

filename = f"ontologies/{USERNAME}_ontology.json"
os.makedirs("ontologies", exist_ok=True)
with open(filename, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nГОТОВО! Онтология сохранена: {filename}")
print(f"   → {best_k} версий себя")
for c in clusters:
    print(f"   • {c['name']} ({c['size']} постов, {c['weight']*100:.1f}%)")
print(f"\nТеперь положи этот файл на Netlify в папку /data/")
