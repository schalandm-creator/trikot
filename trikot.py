# trikot.py   ← oder wie immer deine Datei heißt

import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# ────────────────────────────────────────────────
# 1. TensorFlow laden – mit klarer Fehlermeldung
# ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    st.caption(f"TensorFlow {tf.__version__}")
except ImportError:
    st.error("TensorFlow fehlt.\n\n**Streamlit Cloud Lösung:**\n"
             "1. Settings → Advanced settings\n"
             "2. Python-Version auf **3.11** oder **3.12** stellen\n"
             "3. requirements.txt muss tensorflow-cpu==2.15.0 enthalten")
    st.stop()

# ────────────────────────────────────────────────
# 2. Modell + Labels laden (einmalig)
# ────────────────────────────────────────────────
@st.cache_resource
def lade_modell():
    try:
        modell = tf.keras.models.load_model("keras_Model.h5", compile=False)

        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [zeile.strip() for zeile in f if zeile.strip()]

        if not labels:
            raise ValueError("labels.txt ist leer")

        return modell, labels

    except Exception as e:
        st.error(f"Modell / Labels konnten nicht geladen werden:\n{e}\n\n"
                 "• Dateien keras_Model.h5 + labels.txt im Root?\n"
                 "• Python-Version 3.11 oder 3.12?\n"
                 "• tensorflow-cpu==2.15.0 in requirements.txt?")
        st.stop()

modell, klassen = lade_modell()

# ────────────────────────────────────────────────
# 3. Streamlit App
# ────────────────────────────────────────────────
st.title("📷 Teachable Machine Klassifikator")
st.markdown("Bild hochladen → Modell sagt, was es sieht.")

bild_datei = st.file_uploader("Bild (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if bild_datei is not None:
    # Bild anzeigen
    bild = Image.open(bild_datei).convert("RGB")
    st.image(bild, caption="Hochgeladenes Bild", use_column_width=True)

    # Preprocessing (Teachable Machine Standard: 224×224, -1 bis +1)
    groesse = (224, 224)
    bild = ImageOps.fit(bild, groesse, Image.Resampling.LANCZOS)

    bild_array = np.asarray(bild)
    normalisiert = (bild_array.astype(np.float32) / 127.5) - 1
    eingabe = np.expand_dims(normalisiert, axis=0)         # (1, 224, 224, 3)

    # Vorhersage
    with st.spinner("Analysiere ..."):
        vorhersage = modell.predict(eingabe)
        index = np.argmax(vorhersage[0])
        klasse = klassen[index]
        sicherheit = vorhersage[0][index]

    # Ergebnis
    st.success("Ergebnis")
    st.markdown(f"**Klasse:** {klasse}")
    st.markdown(f"**Sicherheit:** {sicherheit:.1%}")

    st.progress(float(sicherheit))

    if st.checkbox("Alle Wahrscheinlichkeiten anzeigen"):
        for i, w in enumerate(vorhersage[0]):
            st.write(f"{klassen[i]:<30} {w:.1%}")

else:
    st.info("Bitte ein Bild hochladen ↑")

st.markdown("---")
st.caption("Dateien: keras_Model.h5 + labels.txt müssen im gleichen Ordner wie diese .py-Datei liegen")
