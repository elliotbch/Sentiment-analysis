# 🐦 Analyse de Sentiment des Tweets avec BERT

Ce projet utilise des techniques avancées de traitement du langage naturel pour classifier les sentiments exprimés dans les tweets. En utilisant le modèle BERT et d'autres approches, nous avons atteint une précision de **78%** pour la classification en trois catégories : positif, neutre et négatif.

## 📋 Table des matières

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prétraitement](#prétraitement)
4. [Modèles](#modèles)
5. [Résultats](#résultats)
6. [Discussion](#discussion)
7. [Conclusion](#conclusion)
8. [Contributeurs](#contributeurs)

## 📖 Introduction

Dans l'ère numérique actuelle, l'analyse des sentiments sur les réseaux sociaux est devenue cruciale pour comprendre l'opinion publique. Ce projet vise à développer un modèle capable de classifier avec précision les sentiments exprimés dans les tweets, contribuant ainsi à une meilleure compréhension des discussions en ligne.

## 📊 Dataset

Le dataset provient de la bibliothèque "Data for Everyone" de Figure Eight, spécialement conçu pour l'analyse de sentiments des tweets.

- **Composition** :
  - Ensemble d'entraînement : 25,732 tweets labellisés
  - Ensemble de test : Tweets non labellisés
- **Caractéristiques** :
  - 3 classes de sentiment : positif, neutre, négatif
  - Déséquilibre des classes : 10,018 neutres, 8,711 positifs, 7,003 négatifs

![Distribution des sentiments](https://cdn.mathpix.com/cropped/2025_03_18_6fd9357c09fc794ee3a7g-1.jpg?height=570&width=835&top_left_ 🧹 Prétraitement

Nous avons appliqué plusieurs techniques de prétraitement pour améliorer la qualité des données :

1. 🚫 Suppression des mots vides
2. 🔄 Lemmatisation avec Spacy
3. 🔗 Suppression des URLs
4. 😀 Conversion des emojis et émoticônes en mots
5. ✍️ Correction orthographique
6. 🔤 Normalisation des mots avec répétitions de lettres
7. 💬 Conversion des abréviations de chat
8. 🗑️ Suppression des handles Twitter et des données numériques

## 🧠 Modèles

### TextBlob et VADER

Nous avons d'abord testé deux approches basées sur le lexique :

- **TextBlob** : Utilise un dictionnaire de mots avec des scores de sentiment prédéfinis.
- **VADER** : Spécialement conçu pour l'analyse de sentiment des médias sociaux.

### BERT

Notre modèle principal utilise BERT (Bidirectional Encoder Representations from Transformers) :

- Architecture : Transformers avec attention bidirectionnelle
- Prétraitement : Tokenisation spécifique à BERT
- Fine-tuning : Adapté à notre tâche de classification de sentiments

## 🏆 Résultats

| Modèle   | Macro F1-Score |
|----------|----------------|
| TextBlob | 0.60           |
| VADER    | 0.64           |
| BERT     | 0.78           |

BERT surpasse significativement les autres modèles, démontrant sa capacité à capturer les nuances complexes du langage.

![Matrice de confusion BERT](https://cdn.mathpix.com/cropped/2025_03_18_6fd9357c09fc794ee3a7g-5.jpg?height=586&width=795&top_left_ 💡 Discussion

- BERT excelle dans la compréhension du contexte et des subtilités linguistiques.
- Les modèles basés sur le lexique (TextBlob, VADER) sont plus rapides mais moins précis.
- Défis : sarcasme, dialectes, contexte manquant.

## 🎓 Conclusion

Notre étude démontre l'efficacité  de BERT pour l'analyse de sentiment des tweets, tout en soulignant l'importance de choisir le modèle approprié selon les contraintes spécifiques de l'application.

## 🤝 Contributeurs

Ce projet a été réalisé par :
- Victor Mayaud
- Elliot Bouchy
- Aymeric Legros
- Naveen Johnson Vallavanatt

Sous la supervision du Professeur Pietro Michiardi, EURECOM, France.
