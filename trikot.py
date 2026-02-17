import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# ────────────────────────────────────────────────
# TensorFlow / Keras Import mit sehr robuster Fehlerbehandlung
# ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    st.caption(f"TensorFlow Version: {tf.__version__} – Python kompatibel")
except ImportError as e:
    st.error(
        "TensorFlow konnte nicht importiert werden.\n\n"
        "**Lösungsschritte (Streamlit Cloud):**\n"
        "1. Gehe zu App → Settings → Advanced settings\n"
        "2. Python-Version auf **3.11** oder **3.12** ändern (nicht 3.13!)\n"
        "3. requirements.txt mit tensorflow-cpu==2.15.0 verwenden\n"
        "4. App rebooten oder neu pushen\n\n"
        f"Fehler: {e}"
    )
    st.stop()

# ────────────────────────────────────────────────
# App-Konfiguration
# ────────────────────────────────────────────────
st.set_page_config(page_title="Teachable Machine Bilderkennung", layout="centered")

st.title("📷 Teachable Machine – Bild-Klassifikation")
st.markdown("Lade ein Bild hoch. Das Modell sagt dir, was es sieht.\n\n"
            "Dateien `keras_Model.h5` + `labels.txt` müssen im Repository-Root liegen.")

# Modell + Labels laden (cached → nur einmal laden)
@st.cache_resource(show_spinner="Modell wird geladen … (kann 10–30 Sekunden dauern)")
def load_teachable_model():
    try:
        # Moderner Import: tensorflow.keras
        model = tf.keras.models.load_model("keras_Model.h5", compile=False)
        
        # Labels sauber laden
        with open("labels.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        
        if not class_names:
            raise ValueError("labels.txt ist leer oder fehlerhaft")
        
        return model, class_names
    except Exception as e:
        st.error(
            f"Modell konnte nicht geladen werden:\n{e}\n\n"
            "**Häufige Ursachen:**\n"
            "- Falsche TensorFlow-Version (versuche 2.12–2.16)\n"
            "- Python-Version in Cloud nicht 3.11/3.12\n"
            "- Dateien fehlen oder falscher Name/Pfad\n"
            "- .h5 ist zu groß (>800 MB) oder korrupt"
        )
        st.stop()

model, class_names = load_teachable_model()

# ────────────────────────────────────────────────
# Bild-Upload
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Bild hochladen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein hochgeladenes Bild", use_column_width=True)

    # Preprocessing (genau wie Teachable Machine Standard)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    with st.spinner("Analysiere Bild …"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence = float(prediction[0][index])

    # Ergebnis schön darstellen
    st.success("Vorhersage abgesch
