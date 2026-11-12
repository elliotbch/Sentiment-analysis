# Entraînement et export du modèle de la démo

Scripts :

- `data.py` : téléchargement des données publiques et découpage reproductible (ou split du notebook pour les données officielles du challenge) ;
- `train.py` : fine-tuning (recette du notebook `--head notebook`, ou `AutoModelForSequenceClassification`), choix de l'epoch sur la validation, évaluation unique sur le test, métriques fusionnées dans `training/metrics.json` ;
- `export_onnx.py` : choix du meilleur run sur la validation, export ONNX FP32, quantification INT8 dynamique, mesure de la perte sur le même test, fichiers de la démo dans `models/sentiment_onnx/`.

## 1. Machine

Runs publiés : MacBook Apple M2 Pro, 16 Go de RAM partagée, PyTorch 2.15 sur le backend `mps` (Python 3.11).
Alternative : n'importe quel GPU CUDA (`--device cuda`, détecté automatiquement) ou un CPU (compter plusieurs heures par run). Les résultats sur MPS ne sont pas strictement déterministes au bit près, même à graine fixée.

## 2. Préparation

```bash
python3.11 -m venv .venv-train && source .venv-train/bin/activate
pip install "torch>=2.6" "transformers>=4.44,<5" scikit-learn pandas onnx onnxruntime huggingface_hub safetensors
export HF_HOME="$PWD/models/.hf_cache"   # caches dans models/, ignoré par git
```

Données : rien à faire, `train.py` télécharge les deux CSV publics (`raw_data/train_csv`, `raw_data/test_csv`) du dépôt Hub `SetFit/tweet_sentiment_extraction`, révision figée dans `data.py`, vers `data/raw/setfit/` (3,6 Mo, ignoré par git). Protocole :

- **test** = jeu de test officiel étiqueté de *Tweet Sentiment Extraction* (3 534 tweets), utilisé une seule fois par run ;
- **train / validation** = jeu d'entraînement officiel (27 480 tweets non vides), moins les 23 tweets dont le texte figure aussi dans le test, découpé 90 / 10 stratifié avec la graine 42 : 24 707 / 2 746 ;
- l'epoch retenu (et le run exporté) est choisi sur le **F1 macro de validation**, jamais sur le test.

## 3. Commandes d'entraînement

```bash
# a) Zéro-shot (évaluation seule, sans entraînement) du modèle Twitter pré-entraîné
python training/train.py --run-name twitter_roberta_zero_shot \
  --model cardiffnlp/twitter-roberta-base-sentiment-latest --head auto --normalize-twitter --max-len 128 --epochs 0

# b) Baseline : recette du notebook du challenge (bert-base-cased, texte brut, max_len 120, lr 2e-5, batch 16, 4 époques)
python training/train.py --run-name bert_base_notebook --model bert-base-cased --head notebook \
  --optimizer notebook --max-len 120 --batch-size 16 --lr 2e-5 --epochs 4

# c) Nouveau modèle : Twitter RoBERTa fine-tuné (run publié)
python training/train.py --run-name twitter_roberta_ft \
  --model cardiffnlp/twitter-roberta-base-sentiment-latest --head auto --normalize-twitter \
  --max-len 128 --batch-size 32 --lr 2e-5 --epochs 3 --warmup-ratio 0.06 --weight-decay 0.01 \
  --time-limit-min 25 --save-dir models/twitter_roberta_ft

# d) Export ONNX FP32 + INT8 du meilleur candidat (F1 macro de validation), mesure de la perte, fichiers de la démo
python training/export_onnx.py --candidates twitter_roberta_ft
cp training/metrics.json demo/metrics.json   # copie embarquée dans le Space
```

Arguments importants : `--max-len` (troncature ; le padding est dynamique par batch, arrondi au multiple de 32 supérieur pour limiter la mémoire MPS, sans effet sur les prédictions), `--normalize-twitter` (mentions -> `@user`, URL -> `http`, prétraitement de la fiche cardiffnlp, appliqué aussi par la démo), `--time-limit-min` (arrêt de sécurité, 44 min par défaut), `--max-steps N` (mesure de vitesse, n'écrit que `speed_tests`).

Écart assumé avec le notebook pour la baseline : l'optimiseur `transformers.AdamW(correct_bias=False)` n'existe plus, il est réimplémenté à l'identique (`AdamWNoBiasCorrection`, eps 1e-6, sans weight decay) ; le checkpoint est choisi sur le F1 macro de validation (le notebook gardait l'accuracy de validation et évaluait le dernier epoch).

### Variante « données officielles du challenge »

Le jour où le jeton Kaggle est installé (`~/.kaggle/kaggle.json`, à faire soi-même) et l'accès à la compétition privée du cours accordé :

```bash
pip install kaggle
kaggle competitions download -c eurecom-aml-2021-challenge-3 -p data/challenge
unzip data/challenge/*.zip -d data/challenge && rm data/challenge/*.zip
python training/train.py --data challenge --challenge-csv data/challenge/train.csv \
  --run-name bert_base_notebook_challenge --model bert-base-cased --head notebook \
  --optimizer notebook --max-len 120 --batch-size 16 --lr 2e-5 --epochs 4
python training/train.py --data challenge --challenge-csv data/challenge/train.csv \
  --run-name twitter_roberta_ft_challenge --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --head auto --normalize-twitter --max-len 128 --batch-size 32 --lr 2e-5 --epochs 3
```

`--data challenge` reprend exactement le découpage du notebook (95 / 2,5 / 2,5, `random_state=42`) : 23 495 / 618 / 619. Les chiffres obtenus seront alors comparables au résultat historique (77,4 % d'accuracy, F1 macro 0,78), à la non-déterminisme GPU/MPS près.

## 4. Suivi

Lancer en arrière-plan avec un journal, puis suivre les lignes `epoch … done` (perte, accuracy et F1 de validation) :

```bash
mkdir -p training/logs
nohup python training/train.py ... > training/logs/<run>.log 2>&1 &
tail -f training/logs/<run>.log
```

## 5. Durée observée et ressources (M2 Pro, MPS)

Mesures du 12/11/2026, MacBook M2 Pro 16 Go, MPS, chiffres de `training/metrics.json` :

| Run | Durée d'entraînement | Durée totale | Remarque |
|---|---:|---:|---|
| `twitter_roberta_zero_shot` | 0 | 0,3 min | évaluation seule |
| `bert_base_notebook` | 18,5 min (4 époques, 4,4 à 5,1 min par époque) | 18,9 min | |
| `twitter_roberta_ft` | 13,8 min (3 époques, 4,6 min par époque, environ 2,8 it/s) | 14,2 min | mémoire MPS stable : 1,4 Go alloués, 6,25 Go réservés par le pilote |
| `twitter_roberta_ft_lr2e-5_truncated` | 46 min | 46,3 min | **raté**, voir ci-dessous |
| export ONNX + évaluation FP32 et INT8 | | environ 2 min | CPU |

Run raté, conservé dans `metrics.json` pour la traçabilité : la première tentative RoBERTa (padding dynamique non arrondi) a vu son empreinte mémoire monter à 8,6 Go alors que la machine était déjà chargée. Le Mac s'est mis à swapper (0,24 it/s) et le garde-fou de 44 min a arrêté le run pendant la 1re époque. Depuis, le padding est arrondi à un multiple de 32, `torch.mps.empty_cache()` est appelé après chaque époque et la mémoire est journalisée tous les 50 pas. Si la vitesse tombe sous 2 it/s, arrêter le run.

Compter environ 15 min pour un run RoBERTa de 3 époques et 20 min pour la baseline BERT, avec environ 6 Go de mémoire unifiée. Il faut aussi environ 2,5 Go de disque dans `models/` (cache Hugging Face, checkpoint, ONNX).

## 6. Métriques

`training/metrics.json` est la seule source des chiffres du README et de la page de démo :
`runs.<nom>` (hyperparamètres, historique par epoch, validation, test : accuracy, F1 macro, rapport par classe, matrice de confusion, date, graine, machine, durée), `export` (ONNX FP32 / INT8 : accuracy, F1, accord, latence, taille, modèle servi), `speed_tests`.

## Publication des poids sur le Hub (non exécutée)

```bash
hf auth login   # jeton personnel, à faire soi-même
hf repo create elliotb/sentiment-twitter-roberta --repo-type model
# seul le modèle servi (INT8), le tokenizer et les métadonnées : le FP32 (476 Mo) reste en local
hf upload elliotb/sentiment-twitter-roberta models/sentiment_onnx . --exclude "model_fp32.onnx"
```

Le Space lit ensuite le modèle via la variable `HF_MODEL_REPO=elliotb/sentiment-twitter-roberta` (valeur par défaut du `demo/Dockerfile`).
