import fetch from 'node-fetch';

const APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz";
const VOCAB = await fetch('https://raw.githubusercontent.com/SlavaYZMA/quantid/main/vocab_id.json').then(r => r.json());

export default async function handler(req, res) {
  console.log('→ Запрос получен');

  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({error: 'POST only'});

  const { username } = req.body || {};
  if (!username) return res.status(400).json({error: 'no username'});

  console.log(`1. Анализируем @${username}`);

  try {
    // Apify
    const apify = await fetch("https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items?token=" + APIFY_TOKEN, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ usernames: [username.replace('@','')], resultsLimit: 50 })
    });

    if (!apify.ok) throw new Error('Apify failed');
    const data = await apify.json();
    if (!data[0]) return res.json({error: "private or empty"});

    const texts = [];
    if (data[0].biography) texts.push(data[0].biography);
    data[0].latestPosts?.forEach(p => p.caption && texts.push(p.caption));
    console.log(`2. Собрано ${texts.length} текстов`);

    if (texts.length < 4) return res.json({clusters: [], error: "not enough text"});

    // Zero-shot через бесплатный публичный endpoint (работает без токена!)
    const model = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli";
    const scores = VOCAB.map(v => ({...v, score: 0}));

    for (const text of texts.slice(0, 35)) {
      for (const id of VOCAB) {
        try {
          const r = await fetch(model, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              inputs: text.length > 500 ? text.slice(0,497)+'...' : text,
              parameters: { candidate_labels: [id.hypothesis_ru, "нейтральный текст"] }
            })
          });
          if (!r.ok) continue;
          const json = await r.json();
          if (json.labels?.[0] === id.hypothesis_ru) {
            scores.find(s => s.id === id.id).score += json.scores[0];
          }
        } catch(e) {}
      }
    }

    const total = scores.reduce((a,b) => a + b.score, 0) || 1;
    const clusters = scores
      .map(s => ({
        name: s.name_ru,
        weight: Math.round(s.score / total * 1000)/10,
        color: s.color
      }))
      .filter(c => c.weight >= 2)
      .sort((a,b) => b.weight - a.weight)
      .slice(0, 9);

    console.log('УСПЕХ! Кластеры:', clusters.map(c => `${c.name} ${c.weight}%`));
    res.json({ clusters });

  } catch (err) {
    console.error('ОШИБКА:', err.message);
    res.status(500).json({error: err.message});
  }
}

export const config = { api: { bodyParser: true }};
