import streamlit as st
import requests

st.title("Breast Cancer Prediction API")
st.write("Choisis la version du modèle et entre les features pour la prédiction.")

# Choix de la version du modèle
model_version = st.radio("Sélectionne la version du modèle :", ("v1", "v2"))

# Input des features
features_input = st.text_area(
    "Entrer les 30 features séparées par des virgules",
    "20.57,17.77,132.9,1326.0,0.08474,0.07864,0.0869,0.07017,0.05433,0.4564,1.075,3.425,48.55,0.005903,0.03731,0.0438,0.01241,0.01619,0.0034,24.99,23.41,158.8,1956.0,0.1238,0.1866,0.2416,0.186,0.275,0.08902,0.05"
)

# Bouton pour prédire
if st.button("Prédire"):

    try:
        # Convertir la string en liste de floats
        features = [float(x.strip()) for x in features_input.split(",")]
        if len(features) != 30:
            st.error("Il faut exactement 30 features.")
        else:
            # Définir l'URL selon la version choisie
            url = f"https://fastapi-app.yellowwater-2f47f3a8.francecentral.azurecontainerapps.io/api/{model_version}/predict"

            payload = {"features": features}
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction: {data['prediction']}")
                # Affiche la probabilité avec tous les chiffres après la virgule
                st.info(f"Probability: {data['probability']}")
                st.write(f"Model version used: {model_version}")
            else:
                st.error(f"Erreur API: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Erreur: {e}")
