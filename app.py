import os
import textwrap
import joblib
import pandas as pd
import streamlit as st

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Page configuration
st.set_page_config(
    page_title="Emotion Predictor", 
    page_icon="😊", 
    layout="centered",  # Switched back to centered for safer structural rendering
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for a cleaner UI
st.markdown("""
    <style>
    /* Hide default Streamlit header and footer for a clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Improve the text area styling */
    .stTextArea textarea {
        font-size: 18px !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and vectorizer with caching
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, "emotions.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectoriser.pkl")
    return joblib.load(model_path), joblib.load(vectorizer_path)

model, vectorizer = load_assets()

# Mappings and Colors
emotion_mapping = {
    0: "Sadness 😢",
    1: "Anger 😡",
    2: "Love ❤️",
    3: "Surprise 😲",
    4: "Fear 😨",
    5: "Joy 😄",
}

emotion_colors = {
    "Sadness 😢": "#5DADE2",
    "Anger 😡": "#E74C3C",
    "Love ❤️": "#FF69B4",
    "Surprise 😲": "#F39C12",
    "Fear 😨": "#8E44AD",
    "Joy 😄": "#FFD700",
}

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1782/1782914.png", width=100) # Decorative icon
    st.title("🧠 About the App")
    st.markdown("""
    Welcome to the **Emotion Prediction Dashboard by Bhavesh**! ❤️
    
    This application analyzes your text and detects the underlying emotion using Natural Language Processing (NLP).
    
    ### 🛠️ How it works:
    * **TF-IDF Vectorizer**: Converts your text into numerical features based on word importance.
    * **Logistic Regression**: A trained machine learning model classifies the text into one of six emotions.
    
    ### 📊 Detectable Emotions:
    * 😢 **Sadness**
    * 😡 **Anger**
    * ❤️ **Love**
    * 😲 **Surprise**
    * 😨 **Fear**
    * 😄 **Joy**
    
    ---
    *Built with Python and Streamlit.*
    """)
# Sidebar - Attractive About Section
with st.sidebar:
    st.title("🧠 About the App")
    st.markdown("""
    Welcome to the **Emotion Prediction Dashboard**! ✨
    
    This application analyzes your text and detects the underlying emotion using Natural Language Processing (NLP).
    
    ### 🛠️ How it works:
    * **TF-IDF Vectorizer**: Converts your text into numerical features based on word importance.
    * **Logistic Regression**: A trained machine learning model classifies the text into one of six emotions.
    
    ### 📊 Detectable Emotions:
    * 😢 **Sadness** | 😡 **Anger** | ❤️ **Love**
    * 😲 **Surprise** | 😨 **Fear** | 😄 **Joy**
    
    ---
    *Built with Python and Streamlit.*
    """)

# Main App UI
st.title("✨ Emotion Prediction Dashboard")
st.markdown("Type a sentence below, and our machine learning model will figure out the emotion behind it!")
st.markdown("---")

# User input
text = st.text_area("Enter your text here:", height=100, placeholder="Example: I am feeling absolutely fantastic today!")

# Prediction Button
# ... (User input text_area is above this)

# Prediction Button
if st.button("Predict Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Convert text into TF-IDF features
            text_vector = vectorizer.transform([text])

            # Model prediction
            prediction_number = model.predict(text_vector)[0]
            emotion = emotion_mapping.get(prediction_number, "Unknown")
            color = emotion_colors.get(emotion, "#95A5A6")

            st.markdown("### 🎯 Result")
            
            # Render HTML card safely
            html_card = textwrap.dedent(f"""
                <div style="
                    background-color: {color};
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    font-size: 32px;
                    font-weight: bold;
                    color: #1A252F;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 25px;">
                    {emotion}
                </div>
            """)
            st.markdown(html_card, unsafe_allow_html=True)

            # ==========================================================
            # PASTE THE SAFE BREAKDOWN LOGIC HERE (REPLACING THE OLD SNIPPET)
            # ==========================================================
            if hasattr(model, "predict_proba"):
                st.markdown("### 🔍 Analysis Breakdown")
                
                probabilities = model.predict_proba(text_vector)[0]
                
                # Loop through each emotion and display a safe native progress bar
                for i, class_idx in enumerate(model.classes_):
                    emo_name = emotion_mapping.get(class_idx, "Unknown")
                    prob_val = float(probabilities[i])
                    
                    # Display emotion text and percentage
                    st.write(f"**{emo_name}** ({prob_val * 100:.1f}%)")
                    st.progress(prob_val)
            # ==========================================================

