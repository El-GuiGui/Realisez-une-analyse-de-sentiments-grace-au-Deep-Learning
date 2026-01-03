import streamlit as st
import requests

# API_BASE_URL = "https://27738be4-0a39-402b-bbe1-fff5639a6dff-00-3n63iojx4smy7.riker.replit.dev:8000"
API_BASE_URL = "http://127.0.0.1:8000"


PREDICT_URL = f"{API_BASE_URL}/predict"
FEEDBACK_URL = f"{API_BASE_URL}/feedback"
STATS_URL = f"{API_BASE_URL}/stats"
WRONG_FEEDBACKS_URL = f"{API_BASE_URL}/wrong_feedbacks"

st.set_page_config(
    page_title="AirParadis - Sentiment sur tweets",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("✈️ AirParadis - Détection de sentiment sur les tweets")

mode = st.sidebar.radio("Onglet :", ["Prédiction", "Monitoring"])

if mode == "Prédiction":
    st.markdown(
        """
Cette interface permet de **tester l'API de prédiction** et de **remonter du feedback** :

1. Vous entrez un tweet.
2. L'API renvoie un sentiment (positif / négatif).
3. Vous indiquez si la prédiction est correcte.
4. En cas d'erreur, un feedback est envoyé à l'API (et logué pour le monitoring).
"""
    )
else:
    st.markdown(
        """
Cet onglet permet de **suivre le modèle en production** :

- Nombre total de prédictions,
- Nombre de prédictions jugées incorrectes,
- Taux d'erreur,
- Liste des derniers tweets mal prédits.
"""
    )

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def call_predict_api(text: str):
    try:
        response = requests.post(PREDICT_URL, json={"text": text})
        if response.status_code != 200:
            st.error(f"Erreur API /predict : {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None


def call_feedback_api(text: str, prediction: int, proba: float, is_correct: bool):
    try:
        payload = {
            "text": text,
            "prediction": prediction,
            "proba": proba,
            "is_correct": is_correct,
        }
        response = requests.post(FEEDBACK_URL, json=payload)
        if response.status_code != 200:
            st.error(f"Erreur API /feedback : {response.status_code}")
        else:
            data = response.json()
            st.success(f"Feedback envoyé (status: {data.get('status', 'unknown')})")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API (feedback) : {e}")


# PRÉDICTION
if mode == "Prédiction":
    text_input = st.text_area(
        "Entrez un tweet :",
        height=150,
        placeholder="Ex : I love this airline, best flight ever! ✈️",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        predict_btn = st.button("Prédire le sentiment", type="primary")

    if predict_btn:
        if not text_input.strip():
            st.warning("Merci d'entrer un texte avant de prédire.")
        else:
            with st.spinner("Appel à l'API de prédiction..."):
                result = call_predict_api(text_input.strip())

            if result is not None:
                st.session_state.last_prediction = {
                    "text": text_input.strip(),
                    "label": result["label"],
                    "label_str": result["label_str"],
                    "proba": result["proba"],
                }

                st.subheader("Résultat de la prédiction")
                st.write(
                    f"**Sentiment prédit :** `{result['label_str']}` "
                    f"(label = {result['label']}, proba = {result['proba']:.3f})"
                )

    st.markdown("---")
    st.subheader("Votre avis sur la prédiction")

    if st.session_state.last_prediction is None:
        st.info("Faites d'abord une prédiction pour pouvoir donner un feedback.")
    else:
        pred = st.session_state.last_prediction

        st.write(
            f"Texte analysé :\n\n> _{pred['text']}_\n\n"
            f"Prédiction actuelle : **{pred['label_str']}** "
            f"(proba = {pred['proba']:.3f})"
        )

        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button("👍 Prédiction correcte"):
                call_feedback_api(
                    text=pred["text"],
                    prediction=pred["label"],
                    proba=pred["proba"],
                    is_correct=True,
                )

        with col_no:
            if st.button("👎 Prédiction incorrecte"):
                call_feedback_api(
                    text=pred["text"],
                    prediction=pred["label"],
                    proba=pred["proba"],
                    is_correct=False,
                )

# MONITORING
else:
    st.header("Monitoring du modèle :")

    # 1) Stats globales
    try:
        resp = requests.get(STATS_URL, timeout=5)
        if resp.status_code == 200:
            stats = resp.json()
            total = stats["total_predictions"]
            wrong = stats["total_wrong_predictions"]
            error_rate = stats["error_rate"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Prédictions totales", total)
            col2.metric("Prédictions erronées", wrong)
            col3.metric("Taux d'erreur", f"{error_rate:.1%}")
        else:
            st.error(f"Erreur API /stats : {resp.status_code}")
    except Exception as e:
        st.error(f"Impossible de récupérer les stats : {e}")

    st.markdown("---")
    st.subheader("Derniers tweets jugés mal prédits")

    # 2) Liste des derniers feedbacks négatifs
    try:
        resp_fb = requests.get(f"{WRONG_FEEDBACKS_URL}?limit=20", timeout=5)
        if resp_fb.status_code == 200:
            data = resp_fb.json()
            if data:
                rows = []
                for item in data:
                    txt = item["text"]
                    if len(txt) > 120:
                        txt = txt[:120] + "…"
                    rows.append(
                        {
                            "Horodatage": item["timestamp"],
                            "Texte": txt,
                            "Label prédit": item["predicted_label"],
                            "Proba": round(item["proba"], 3),
                        }
                    )
                st.table(rows)
            else:
                st.info("Aucun feedback négatif pour le moment.")
        else:
            st.error(f"Erreur API /wrong_feedbacks : {resp_fb.status_code}")
    except Exception as e:
        st.error(f"Impossible de récupérer les feedbacks : {e}")

    st.caption(
        "Onglet monitoring : "
        "statistiques globales, erreurs récentes et base pour analyser les dérives du modèle."
    )
