---
title: Analyse de sentiment de tweets
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Analyse de sentiment de tweets : démo en ligne

Tapez ou collez un tweet, ou cliquez sur l'un des exemples (sarcasme, négation, avis mitigé…). La démo affiche le sentiment (négatif / neutre / positif), les probabilités des trois classes et le temps d'inférence. Un mode « lot » classe jusqu'à 50 lignes d'un coup.

**Le modèle analyse des textes en anglais** : il a été entraîné sur des tweets anglais, les exemples sont donc en anglais.

- Modèle : `cardiffnlp/twitter-roberta-base-sentiment-latest`, fine-tuné sur le jeu public Tweet Sentiment Extraction, exporté en ONNX et exécuté avec ONNX Runtime sur CPU.
- Les métriques affichées viennent de `training/metrics.json` (copié ici sous le nom `metrics.json` pour le Space). Le protocole d'évaluation est décrit dans le README du dépôt.
- Code : <https://github.com/elliotbch/Sentiment-analysis>

## API

| Méthode | Chemin | Corps | Réponse |
|---|---|---|---|
| GET | `/` | | la page de démo |
| GET | `/health` | | `{"status": "ok", "model_loaded": true, ...}` |
| GET | `/api/examples` | | textes d'exemple |
| GET | `/api/metrics` | | contenu de `training/metrics.json` |
| POST | `/api/predict` | `{"text": "..."}` (1 000 caractères au maximum) | classe, probabilités, `inference_ms` |
| POST | `/api/predict-batch` | `{"texts": ["...", "..."]}` (50 lignes non vides au maximum) | un résultat par ligne |

Les étiquettes renvoyées par l'API restent `negative`, `neutral`, `positive` ; la page les affiche en français.
Sans poids de modèle, l'application démarre quand même : `/health` indique `model_loaded: false` et les endpoints de prédiction répondent **503** avec une explication. Rien n'est jamais simulé.

## Lancer en local

Depuis la racine du dépôt, après l'export du modèle (voir `training/README.md`) :

```bash
python -m venv .venv-demo && source .venv-demo/bin/activate
pip install -r demo/requirements.txt
cd demo && uvicorn app:app --port 7860
# ouvrir http://localhost:7860
```

L'application cherche les poids dans cet ordre : `MODEL_PATH` (un fichier `.onnx` avec `tokenizer.json` et `sentiment_meta.json` à côté, ou leur dossier), puis `../models/sentiment_onnx/`, puis le dépôt du Hub Hugging Face indiqué par `HF_MODEL_REPO` (valeur par défaut dans le Dockerfile : `elliotb/sentiment-twitter-roberta`, pas encore publié).

Avec Docker :

```bash
docker build -t sentiment-demo demo/
docker run --rm -p 7860:7860 -v "$PWD/models/sentiment_onnx:/models:ro" \
  -e MODEL_PATH=/models sentiment-demo
```

Tests : `pip install pytest httpx && pytest demo/tests`.
