# Réalisez une analyse de sentiments grâce au Deep Learning

---

# AirParadis – Analyse de sentiment des tweets & déploiement MLOps

Ce projet met en place un prototype complet d’analyse de sentiment pour la compagnie aérienne **Air Paradis**.

Objectif : prédire automatiquement si un tweet est **positif** ou **négatif**, comparer plusieurs approches de modélisation, puis déployer un **service de prédiction** utilisable via une API et une interface Streamlit, en appliquant une démarche inspirée **MLOps** (tracking, versionning, tests, CI, monitoring, alertes).

---

## 1. Architecture globale

Le projet est organisé autour de trois briques :

1. **Modélisation**

   - Baseline : TF-IDF + Régression Logistique (modèle déployé).
   - Modèle avancé : réseau de neurones avec word embeddings (GloVe + un autre embedding).
   - Modèle avancé BERT (ModernBERT) pour évaluer l’apport des Transformers.

2. **Industrialisation / MLOps**

   - Tracking des expériences avec **MLflow**.
   - Sérialisation des modèles (artifacts).
   - Tests unitaires avec **pytest**.
   - CI via **GitHub Actions**.

3. **Mise en production & Monitoring**
   - API REST avec **FastAPI** pour exposer le modèle.
   - Interface **Streamlit** pour tester l’API et remonter du feedback.
   - Logging des mauvaises prédictions dans un fichier JSON.
   - Compteurs de performance + seuil d’alerte.
   - Envoi d’email via **Mailtrap** lorsqu’il y a trop de prédictions erronées.

---

## 2. Organisation du dépôt

Arborescence principale :

```text
.
├── api/                    # API FastAPI (modèle de scoring + endpoints)
│   ├── main.py             # Entrée FastAPI (routes, monitoring, alertes)
│   ├── model_loader.py     # Chargement du modèle TF-IDF + pipeline de prédiction
│   └── schemas.py          # Schémas Pydantic (entrées / sorties)
│
├── app/
│   └── streamlit_app.py    # Interface Streamlit (prédiction + monitoring)
│
├── notebooks/
│   ├── 1_exploration.ipynb       # EDA (exploration des données)
│   ├── 2_preprocessing.ipynb     # Analyse des différents prétraitements NLTK
│   ├── 3_modele_simple.ipynb     # TF-IDF + Régression Logistique (+ MLflow)
│   ├── 4_modele_avance.ipynb     # Réseau de neurones + embeddings (+ MLflow)
│   ├── 5_modele_bert.ipynb       # ModernBERT / Transformers (+ MLflow)
│   └── 6_comparaison.ipynb       # Comparaison des modèles / résultats
│
├── scripts/
│   └── preprocessing.py          # Fonctions de nettoyage/lemmatisation
│
├── models/
│   └── tfidf_logreg.joblib       # Modèle TF-IDF + LogReg sérialisé (modèle déployé)
│
├── tests/
│   ├── test_api.py               # Tests unitaires de l’API FastAPI
│   ├── test_model_loader.py      # Tests de chargement du modèle / prédiction
│   └── test_preprocessing.py     # Tests du prétraitement NLTK
│
├── logs/
│   └── feedback.log              # Logs JSON des mauvaises prédictions + alertes
│
├── data/
    ├── embeddings/               # Source pour Fastext (wiki-news-300d-1M-subword.vec) et Glove (glove.twitter.27B.200d.txt)
│   └── training.1600000.processed.noemoticon.csv (*non versionné* -> trop volumineux -> check at : https://www.kaggle.com/datasets/kazanova/sentiment140/data)
│
├── run.sh                   # Script de lancement (FastAPI + Streamlit + NLTK)(uniquement sur Replit)
├── requirements.txt         # Dépendances du projet
├── .gitignore               # Exclusion des données brutes, env, etc.
│
└── .github/workflows/...
    └── ci.yml               # CI GitHub Actions (installation + pytest)

```

---

Les gros fichiers (dataset CSV, venv, artefacts temporaires, etc.) sont exclus du dépôt via .gitignore.

---

## 3. Données utilisées

Nous utilisons le dataset de tweets annotés binaire :

- Fichier attendu : data/training.1600000.processed.noemoticon.csv

- Contenu : 1,6 million de tweets, avec un label binaire (0 = négatif, 4 = positif).

- Avant modélisation, les labels sont remappés en 0 (négatif) / 1 (positif).

Pour faire tourner les notebooks, il faut placer le CSV dans le dossier data/ à la racine du projet.

---

## 4. Installation locale

### 4.1. Prérequis

Python 3.11

git

environnement virtuel type env/venv

### 4.2. Cloner le dépôt et installer les dépendances

git clone git@github.com:El-GuiGui/Realisez-une-analyse-de-sentiments-grace-au-Deep-Learning.git
cd <"votre dossier">

-> Création d'un environnement virtuel
python -m venv env

Activer :
source env/bin/activate # Linux / macOS
-> ou
env\Scripts\activate # Windows

# Installation des dépendances

pip install -r requirements.txt

### 4.3. Télécharger les ressources NLTK

Pour que le prétraitement fonctionne (et les tests aussi), il faut télécharger les ressources NLTK utilisées :

python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

Pour certaines plateformes (Replit, CI GitHub), ce téléchargement est fait automatiquement via script (run.sh pour replit par exemple ou direct via workflow GitHub Actions).

---

## 5. Entraîner les modèles et suivre les expériences (MLflow)

### 5.1. Baseline : TF-IDF + Régression Logistique

1. Ouvrir le notebook : notebooks/3_modele_simple.ipynb.

2. Exécuter les cellules :

- chargement des données,

- prétraitement simple via scripts/preprocessing.py,

- split train/test (split fixe partagé par tous les modèles),

- entraînement du pipeline TF-IDF + LogReg,

- logging des métriques dans MLflow,

- sérialisation du modèle dans models/tfidf_logreg.joblib.

Ce fichier .joblib est celui qui sera utilisé par l’API.

### 5.2. Modèle avancé (embeddings + réseau de neurones)

1. Ouvrir notebooks/4_modele_avance.ipynb.

2. Utiliser le prétraitement “avancé”.

3. Charger les embeddings (GloVe + second embedding).

4. Entraîner le modèle (LSTM ici présent).

5. Logger dans MLflow : hyperparamètres, métriques, figures (courbes d’accuracy, matrices de confusion).

### 5.3. Modèle ModernBERT / Transformers

1. Ouvrir notebooks/5_modele_bert.ipynb.

2. Tokenisation avec le tokenizer ModernBERT.

3. Entraînement sur un sous-échantillon (pour respecter les contraintes de ressources ou contrainte de l'environnement local).

4. Logging des expériences dans MLflow.

### 5.4. Visualiser les runs MLflow

Depuis la racine du projet (dans la console):

mlflow ui

Puis ouvrir l’URL indiquée (http://127.0.0.1:5000) pour comparer les expériences (baseline, embeddings, BERT).

---

## 6. Lancer l’API de prédiction

### 6.1. Vérifier le modèle sérialisé

Assurez-vous que le fichier suivant existe (généré par le notebook 3) :

models/tfidf_logreg.joblib

C’est ce fichier que api/model_loader.py charge au démarrage.

### 6.2. Lancer FastAPI en local

Depuis la racine du projet (environnement virtuel activé) :

uvicorn api.main:app --reload

L’API est alors disponible par défaut sur :

http://127.0.0.1:8000

La documentation interactive est accessible à :

http://127.0.0.1:8000/docs

---

## 7. Lancer l’interface Streamlit

Dans app/streamlit_app.py, veillez à ce que l’URL de l’API soit bien locale si vous travaillez en local :

API_BASE_URL = "http://127.0.0.1:8000"
(commenter la ligne replit)

Puis lancer Streamlit :

streamlit run app/streamlit_app.py

Accès local :

http://localhost:8501/

L’interface web permet :

- D’entrer un tweet,

- De lancer une prédiction (appel à l’API /predict),

- De voir le label (positif / négatif) et la probabilité,

- De donner un feedback (👍 / 👎) qui sera envoyé à /feedback pour le monitoring.

---

## 8. Endpoints principaux de l’API

'GET /health'

- Vérifie que l’API est démarrée.

- Réponse : { "status": "ok" }.

'POST /predict'

- Entrée :

{ "text": "I love this airline, best flight ever!" }

- Sortie :

{
"label": 1,
"label_str": "positive",
"proba": 0.93
}

Le texte est prétraité via preprocess_simple, puis passé dans le pipeline TF-IDF + LogReg chargé en mémoire.

'POST /feedback'

- Entrée :

{
"text": "tweet original",
"prediction": 1,
"proba": 0.93,
"is_correct": false
}

- Si is_correct est false, l’API log une mauvaise prédiction dans logs/feedback.log et met à jour les compteurs/alertes.

- Sortie :

{ "status": "received" }

### Endpoints de monitoring (si activés dans main.py)

'GET /stats'
→ retourne le nombre total de prédictions, le nombre de prédictions jugées erronées, et le taux d’erreur global.

'GET /wrong_feedbacks'
→ retourne les derniers tweets signalés comme mal prédits (texte, label prédit, proba, timestamp).

Ces endpoints sont consommés par l’onglet “Monitoring” de l’interface Streamlit.

---

## 9. Monitoring & alertes

### 9.1. Logging structuré

Chaque mauvaise prédiction signalée par un utilisateur est enregistrée comme une ligne JSON dans :

'logs/feedback.log'

Exemple :

{
"timestamp": "2025-01-01T10:15:32Z",
"type": "WRONG_PREDICTION",
"text": "Nice airline but it's not a good airline company",
"prediction": 1,
"proba": 0.73
}

Une entrée de type ALERT est ajoutée lorsqu’un seuil est franchi.

### 9.2. Seuil d’alerte

- Si 3 mauvaises prédictions ou plus sont enregistrées sur une fenêtre de 5 minutes, alors une alerte est déclenchée :

  - Écriture d’un log ALERT dans feedback.log,

  - Envoi d’un email selon la configuration SMTP avec les informations essentielles.

### 9.3. Configuration des emails (Mailtrap)

Le projet utilise des variables d’environnement pour l’alerte email :

- ALERT_EMAIL_ENABLED (True / False)

- ALERT_EMAIL_FROM

- ALERT_EMAIL_TO

- ALERT_EMAIL_SMTP (par ex. sandbox.smtp.mailtrap.io)

- ALERT_EMAIL_PORT (par défaut 587)

- ALERT_EMAIL_PASSWORD (token ou mot de passe SMTP)

En local, on peut les définir via un fichier .env (non versionné) ou directement dans l’environnement du système.

---

## 10. Intégration continue (CI)

Le dépôt contient un workflow GitHub Actions qui :

- Installe Python et les dépendances.

- Télécharge les ressources NLTK nécessaires (stopwords, punkt, etc.).

- Lance pytest sur le dossier tests/.

Objectif : s’assurer que :

- L’API démarre correctement,

- Le modèle est bien chargeable,

- Le prétraitement se comporte comme attendu,

- Les modifications futures ne cassent pas la chaîne de prédiction.

---

## 11. Limites et pistes d’amélioration

Quelques axes possibles si le projet devait aller plus loin :

- Déployer un modèle plus avancé (embeddings ou ModernBERT) sur une infra cloud avec GPU.

- Remplacer le logging fichier par une stack de monitoring plus robuste (type Application Insights, Prometheus + Grafana, ou un APM managé).

- Mettre en place une vraie boucle de réentraînement basée sur les feedbacks collectés.

- Ajouter une gestion des versions de modèles plus fine (tagging de modèles, rollback, etc.).

- Gérer d’autres langues ou d’autres réseaux sociaux (Instagram, Facebook, avis sites tiers, etc.).

---

```

```
