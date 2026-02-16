import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io

# ────────────────────────────────────────────────
# Hinweis: TensorFlow muss installiert sein!
# ────────────────────────────────────────────────
try:
    from keras.models import load_model
except ImportError:
    st.error("keras / tensorflow nicht gefunden. Bitte installiere tensorflow oder tensorflow-cpu.")
    st.stop()

# ────────────────────────────────────────────────
# Konfiguration
# ────────────────────────────────────────────────
st.set_page_config(page_title="Teachable Machine – Bildklassifikation", layout="centered")

st.title("📸 Teachable Machine Klassifikator")
st.markdown("Lade ein Bild hoch – das Modell sagt dir, was es erkennt.")

# Modell und Labels laden (einmalig beim Start)
@st.cache_resource
def load_classifier():
    try:
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        class_names = [line.strip() for line in class_names]  # bereinigen
        return model, class_names
    except Exception as e:
        st.error(f"Modell oder Labels konnten nicht geladen werden:\n{e}")
        st.stop()

model, class_names = load_classifier()

# ────────────────────────────────────────────────
# Bild hochladen
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Bild auswählen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Preprocessing – genau wie bei Teachable Machine
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # In Array umwandeln & normalisieren (-1 bis +1)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Batch-Dimension hinzufügen → Shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    with st.spinner("Modell denkt ..."):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

    # Ergebnis anzeigen
    st.success("Vorhersage abgeschlossen!")
    
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**Klasse:** {class_name}")
    col2.markdown(f"**Konfidenz:** {confidence_score:.4f}")

    # Balken für bessere Visualisierung
    st.progress(confidence_score)
    
    # Alle Klassen mit Wahrscheinlichkeiten (optional)
    if st.checkbox("Alle Klassen & Wahrscheinlichkeiten anzeigen"):
        probs = prediction[0]
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]:<30} {prob:.4f}")

else:
    st.info("Bitte lade ein Bild hoch ↑")
