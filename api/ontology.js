import { createClient } from '@supabase/supabase-js'
import fetch from 'node-fetch'

// Твой Apify токен (можно оставить тут — Vercel безопасен)
const APIFY_TOKEN = "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"

// Загружаем словарь идентичностей
const VOCAB = require('../vocab_id.json')

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({error: 'Method not allowed'})
  
  const { username } = req.body
  if (!username) return res.status(400).json({error: 'no username'})

  try {
    // 1. Получаем данные из Apify
    const apifyRes = await fetch(
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
    )
    
    const data = await apifyRes.json()
    if (!data || data.length === 0) return res.status(404).json({error: "profile not found or private"})

    const profile = data[0]
    const texts = []
    if (profile.biography) texts.push(profile.biography)
    profile.latestPosts?.forEach(p => p.caption && texts.push(p.caption))
    
    if (texts.length < 3) return res.status(400).json({error: "not enough text"})

    // 2. Zero-shot классификация через Hugging Face (бесплатно и мгновенно)
    const HF_TOKEN = "hf_YOUR_TOKEN_IF_NEEDED" // можно без токена для публичных моделей
    
    const scores = VOCAB.map(id => ({...id, score: 0}))
    
    for (const text of texts) {
      for (const identity of VOCAB) {
        const response = await fetch(
          "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
          {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${HF_TOKEN}`,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              inputs: text,
              parameters: { candidate_labels: [identity.hypothesis_ru, "нейтральный текст"] }
            })
          }
        )
        
        const result = await response.json()
        if (result.labels && result.labels[0] === identity.hypothesis_ru) {
          scores.find(s => s.id === identity.id).score += result.scores[0]
        }
      }
    }

    // Нормализуем в проценты
    const total = scores.reduce((s, i) => s + i.score, 0) || 1
    const clusters = scores
      .map(s => ({
        name: s.name_ru,
        weight: Math.round((s.score / total) * 1000) / 10,
        color: s.color,
        valence: s.valence
      }))
      .filter(c => c.weight > 1)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 12)

    res.status(200).json({ username, clusters, total_posts: texts.length })

  } catch (err) {
    console.error(err)
    res.status(500).json({error: "server error"})
  }
}

export const config = { api: { bodyParser: true } }
