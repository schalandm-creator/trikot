# app.py   ← speichere genau so (oder trikot.py, aber passe dann den Dateinamen im Cloud-Dashboard an)

import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# ────────────────────────────────────────────────
# TensorFlow / Keras laden – mit sehr klarer Fehlermeldung für Cloud
# ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    st.caption(f"TensorFlow geladen: Version {tf.__version__}")
except ImportError as e:
    st.error(
        "TensorFlow konnte **nicht** importiert werden.\n\n"
        "**Lösung für Streamlit Cloud (wichtig!):**\n"
        "1. Gehe zu deiner App → Settings (Zahnrad oben rechts)\n"
        "2. Scrolle zu **Advanced settings**\n"
        "3. Stelle **Python version** auf **3.11** oder **3.12** (nicht 3.13!)\n"
        "4. Speichern → App rebooten oder neu pushen\n\n"
        "requirements.txt muss enthalten: tensorflow-cpu==2.15.0\n\n"
        f"Fehler: {e}"
    )
    st.stop()

# ────────────────────────────────────────────────
# Modell + Labels laden (nur einmal – dank Cache)
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Modell laden ... (kann 10–60 Sekunden dauern)")
def lade_modell_und_labels():
    try:
        # Model laden (compile=False wie in deinem Original)
        modell = tf.keras.models.load_model("keras_Model.h5", compile=False)

        # Labels laden & bereinigen (entfernt \n und leere Zeilen)
        with open("labels.txt", "r", encoding="utf-8") as f:
            klassen = [zeile.strip() for zeile in f if zeile.strip()]

        if not klassen:
            raise ValueError("labels.txt ist leer oder fehlerhaft")

        st.success(f"{len(klassen)} Klassen geladen")
        return modell, klassen

    except Exception as e:
        st.error(
            f"Modell oder labels.txt konnten **nicht** geladen werden:\n{e}\n\n"
            "• Liegen keras_Model.h5 + labels.txt wirklich im Root-Ordner des Repos?\n"
            "• Python-Version in Cloud auf 3.11/3.12?\n"
            "• tensorflow-cpu==2.15.0 (oder 2.12.0 / 2.16.1) in requirements.txt?\n"
            "• Modell mit neuerer TF-Version gespeichert? → Neu exportieren!"
        )
        st.stop()

modell, klassen = lade_modell_und_labels()

# ────────────────────────────────────────────────
# Streamlit Oberfläche
# ────────────────────────────────────────────────
st.title("📸 Teachable Machine – Bildklassifikation")
st.markdown("Lade ein Bild hoch – das Modell sagt dir die Klasse und Sicherheit.")

bild = st.file_uploader("Bild hochladen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if bild is not None:
    # Bild laden & anzeigen
    img = Image.open(bild).convert("RGB")
    st.image(img, caption="Dein hochgeladenes Bild", use_column_width=True)

    # Preprocessing – genau wie in deinem Originalcode
    groesse = (224, 224)
    img = ImageOps.fit(img, groesse, Image.Resampling.LANCZOS)

    img_array = np.asarray(img)
    normalisiert = (img_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalisiert

    # Vorhersage
    with st.spinner("Berechne Vorhersage ..."):
        vorhersage = modell.predict(data)
        index = np.argmax(vorhersage[0])
        klasse = klassen[index]
        konfidenz = vorhersage[0][index]

    # Ergebnis anzeigen
    st.success("Vorhersage abgeschlossen!")
    st.markdown(f"**Klasse:** {klasse.strip()}")
    st.markdown(f"**Konfidenz:** {konfidenz:.2%} ({konfidenz:.4f})")
    st.progress(float(konfidenz))

    if st.checkbox("Alle Klassen mit Wahrscheinlichkeiten zeigen"):
        for i, wert in enumerate(vorhersage[0]):
            st.write(f"{klassen[i].strip():<35} {wert:.2%}")

else:
    st.info("Lade bitte ein Bild hoch ↑")

st.markdown("---")
st.caption("Dein Originalcode wurde angepasst für Streamlit: Upload statt fester Pfad, Fehlerbehandlung, Caching.")
