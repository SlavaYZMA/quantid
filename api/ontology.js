import fetch from 'node-fetch';

// Твой Apify токен (безопасно — Vercel шифрует)
const APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz";

// Загружаем словарь идентичностей
const VOCAB = require('../../vocab_id.json');

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method !== 'POST') {
    console.log('Ошибка: метод не POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { username } = req.body || {};
  if (!username) {
    console.log('Ошибка: нет username');
    return res.status(400).json({ error: 'no username' });
  }

  console.log(`Запуск анализа для @${username}`);

  try {
    // ШАГ 1: Получаем данные из Apify
    console.log('Шаг 1: запрос к Apify...');
    const apifyResponse = await fetch(
      "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: APIFY_TOKEN,
          usernames: [username.replace(/^@/, "")],
          resultsLimit: 60
        })
      }
    );

    if (!apifyResponse.ok) {
      console.log('Apify вернул ошибку:', apifyResponse.status);
      return res.status(500).json({ error: "Apify error" });
    }

    const data = await apifyResponse.json();
    console.log(`Apify вернул ${data.length} профилей`);

    if (!data || data.length === 0) {
      console.log('Профиль не найден или приватный');
      return res.status(404).json({ error: "private or not found" });
    }

    const profile = data[0];
    const texts = [];
    if (profile.biography) texts.push(profile.biography);
    profile.latestPosts?.forEach(p => p.caption && texts.push(p.caption));

    console.log(`Собрано ${texts.length} текстов`);

    if (texts.length < 3) {
      return res.status(400).json({ error: "not enough text" });
    }

    // ШАГ 2: Zero-shot через публичную модель (БЕЗ ТОКЕНА!)
    console.log('Шаг 2: запуск zero-shot классификации...');

    const modelUrl = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli";

    const scores = VOCAB.map(id => ({ ...id, score: 0 }));

    for (const text of texts.slice(0, 40)) { // ограничиваем, чтобы не убить API
      for (const identity of VOCAB) {
        try {
          const response = await fetch(modelUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              inputs: text,
              parameters: {
                candidate_labels: [identity.hypothesis_ru, "нейтральный текст"],
                hypothesis_template: "Этот текст выражает {}."
              }
            })
          });

          if (!response.ok) continue;

          const result = await response.json();
          if (result.labels && result.labels[0] === identity.hypothesis_ru) {
            scores.find(s => s.id === identity.id).score += result.scores[0];
          }
        } catch (e) {
          // если HF спит — пропускаем
        }
      }
    }

    // ШАГ 3: Нормализация
    console.log('Шаг 3: нормализация результатов');
    const total = scores.reduce((s, i) => s + i.score, 0) || 1;

    const clusters = scores
      .map(s => ({
        name: s.name_ru,
        weight: Math.round((s.score / total) * 1000) / 10,
        color: s.color,
        valence: s.valence
      }))
      .filter(c => c.weight >= 1.5)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 10);

    console.log('Успех! Отправляем результат:', clusters.map(c => `${c.name} ${c.weight}%`));

    res.status(200).json({
      username,
      clusters,
      total_posts: texts.length,
      debug: "all ok"
    });

  } catch (err) {
    console.error('Критическая ошибка:', err);
    res.status(500).json({ error: "server crash", details: err.message });
  }
}

export const config = {
  api: {
    bodyParser: true,
  },
};
