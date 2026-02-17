
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io

# ────────────────────────────────────────────────
# TensorFlow / Keras Import mit Fallback & besserer Fehlermeldung
# ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    st.success("TensorFlow erfolgreich importiert (Version: {})".format(tf.__version__))
except ImportError as e:
    st.error(
        "TensorFlow / Keras konnte **nicht** gefunden werden.\n\n"
        "**Lösung lokal:**\n"
        "pip install tensorflow-cpu\n\n"
        "**Lösung auf Streamlit Cloud:**\n"
        "1. In requirements.txt → tensorflow-cpu==2.15.0 oder tensorflow-cpu==2.16.*\n"
        "2. Python-Version auf 3.10 oder 3.11 stellen (Settings → Advanced)\n"
        "3. App neu bauen / rebooten\n\n"
        f"Original-Fehler: {e}"
    )
    st.stop()

# ────────────────────────────────────────────────
# Konfiguration
# ────────────────────────────────────────────────
st.set_page_config(page_title="Teachable Machine – Bildklassifikation", layout="centered")

st.title("📸 Teachable Machine Klassifikator")
st.markdown("Lade ein Bild hoch – das Modell sagt dir, was es erkennt.\n\nModell & labels.txt müssen im gleichen Ordner liegen.")

# Modell und Labels laden (cached!)
@st.cache_resource(show_spinner="Modell wird geladen …")
def load_classifier():
    try:
        # Moderne Art: über tensorflow.keras
        model = tf.keras.models.load_model("keras_Model.h5", compile=False)
        
        with open("labels.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        return model, class_names
    except Exception as e:
        st.error(f"Modell oder labels.txt konnten nicht geladen werden:\n{e}\n\n"
                 "• Dateien im Root-Ordner? (keras_Model.h5 + labels.txt)\n"
                 "• Dateinamen exakt gleich?\n"
                 "• Dateigröße < 500–800 MB? (Cloud-Limit)")
        st.stop()

model, class_names = load_classifier()

# ────────────────────────────────────────────────
# Bild hochladen
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Bild auswählen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild laden & anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Preprocessing – exakt wie Teachable Machine
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    with st.spinner("Modell analysiert das Bild …"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

    # Ergebnis
    st.success("Fertig!")
    
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**Erkannte Klasse:** {class_name}")
    col2.markdown(f"**Sicherheit:** {confidence_score:.4f}")

    st.progress(confidence_score)

    if st.checkbox("Alle Klassen & Wahrscheinlichkeiten zeigen"):
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]:<35} {prob:.4f}")

else:
    st.info("Bitte ein Bild hochladen ↑")
