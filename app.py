import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# =========================
# DOWNLOAD STOPWORDS
# =========================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =========================
# LOAD MODEL & FILES
# =========================
model = load_model("mental_health_rnn_model.keras")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as handle:
    le = pickle.load(handle)

# =========================
# CONSTANTS
# =========================
MAX_LEN = 100

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([
        word for word in text.split()
        if word not in stop_words
    ])
    return text

# =========================
# PREDICTION FUNCTION
# =========================
def predict_emotion(text):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    emotion = le.inverse_transform(
        [predicted_index]
    )[0]

    return emotion, confidence, prediction[0]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

# =========================
# SECTION 1 — HEADER
# =========================
st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# =========================
# SECTION 2 — ABOUT PROJECT
# =========================
st.markdown("---")

st.header("📘 About the Project")

st.write("""
This AI-powered Mental Health Sentiment Monitoring System
uses Natural Language Processing (NLP) and Simple Recurrent
Neural Networks (RNN) to analyze emotional sentiment from
user text messages.

### Importance of Emotional AI
- Helps monitor emotional well-being
- Detects negative emotional patterns
- Supports early intervention for counselors
- Encourages mental wellness awareness

### NLP Applications
- Sentiment Analysis
- Mental Health Monitoring
- Chatbots
- Emotion Detection Systems

### Role of RNN in Sequence Learning
Recurrent Neural Networks (RNNs) are powerful for text
analysis because they learn patterns in sequential data
such as sentences and conversations.
""")

# =========================
# SECTION 3 — TEXT INPUT
# =========================
st.markdown("---")

st.header("✍️ User Text Input Area")

st.write("### Sample Sentences")

samples = [
    "I feel lonely and emotionally exhausted.",
    "Today was amazing and I feel very happy.",
    "I am constantly worried about my future.",
    "Everything feels hopeless lately."
]

for sample in samples:
    st.write("•", sample)

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=180
)

# =========================
# SECTION 4 — PREDICTION BUTTON
# =========================
analyze = st.button("🔍 Analyze Emotion")

# =========================
# SECTION 5 — PREDICTION OUTPUT
# =========================
if analyze:

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        st.markdown("---")

        st.header("📊 Prediction Output")

        st.success(f"Emotion Detected: {emotion}")

        st.info(f"Confidence Score: {confidence * 100:.2f}%")

        # Emotional status
        if emotion.lower() in ['depression', 'anxiety', 'stress', 'suicidal']:
            status = "⚠️ Negative Emotional State Detected"
        else:
            status = "✅ Emotion Appears Stable"

        st.write(f"### Emotional Status: {status}")

        # =========================
        # SECTION 6 — VISUALIZATION
        # =========================
        st.markdown("---")

        st.header("📈 Sentiment Confidence Graph")

        class_names = le.classes_

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(class_names, probabilities)

        ax.set_xlabel("Emotion Classes")
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Prediction Probabilities")

        st.pyplot(fig)

        # =========================
        # SECTION 7 — GUIDANCE AREA
        # =========================
        st.markdown("---")

        st.header("💡 Emotional Wellness Guidance")

        guidance = {

            "anxiety":
            "Take a short break, practice deep breathing, and talk with someone you trust.",

            "depression":
            "Remember that support is available. Try engaging in a small positive activity today.",

            "stress":
            "Consider relaxation techniques, meditation, or light physical exercise.",

            "suicidal":
            "Please reach out to a trusted person or mental health professional immediately.",

            "bipolar":
            "Maintaining a healthy routine and seeking emotional support can help stabilize emotions.",

            "personality disorder":
            "Mindfulness exercises and professional counseling may provide support.",

            "normal":
            "Keep maintaining healthy habits and positive social connections."
        }

        emotion_lower = emotion.lower()

        if emotion_lower in guidance:
            st.success(guidance[emotion_lower])
        else:
            st.success(
                "Stay positive and continue taking care of your mental wellness."
            )

        st.write("### 🌿 Positive Activities")
        st.write("""
        - Go for a short walk
        - Listen to calming music
        - Talk with friends or family
        - Practice meditation
        - Maintain healthy sleep habits
        """)

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption(
    "Developed using Streamlit, TensorFlow, NLP, and Simple RNN"
)