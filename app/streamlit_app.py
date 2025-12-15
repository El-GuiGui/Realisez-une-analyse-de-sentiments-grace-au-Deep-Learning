import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
FEEDBACK_URL = f"{API_BASE_URL}/feedback"


st.set_page_config(
    page_title="AirParadis - Sentiment sur tweets",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ AirParadis - Détection de sentiment sur les tweets")
st.markdown(
    """
Cette interface permet de **tester l'API de prédiction** et de **remonter du feedback** :

1. Vous entrez un tweet.
2. L'API renvoie un sentiment (positif / négatif).
3. Vous indiquez si la prédiction est correcte.
4. En cas d'erreur, un feedback est envoyé à l'API (et logué pour le monitoring).
"""
)

text_input = st.text_area(
    "Entrez un tweet :",
    height=150,
    placeholder="Ex : I love this airline, best flight ever! ✈️",
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

            st.subheader("📊 Résultat de la prédiction")
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
