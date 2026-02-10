import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CONFIG ---
MODEL_PATH = "beauty_model_v2.h5"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_beauty_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_beauty_model()

# --- SIDEBAR FOR CORRECTIONS ---
st.sidebar.title("Settings & Debug")
override_gender = st.sidebar.checkbox("Manually Correct Gender?")
manual_gender = st.sidebar.radio("Select Correct Gender:", ["Male", "Female"]) if override_gender else None

# --- MAIN UI ---
st.title("✨ AI Beauty & Persona Analyzer")

if model is None:
    st.error("Model file not found!")
else:
    uploaded_file = st.file_uploader("Upload photo...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        # 1. PRE-PROCESS
        img_input = np.array(img.resize(IMG_SIZE)) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # 2. PREDICT
        with st.spinner("Analyzing..."):
            gen_pred, score_pred = model.predict(img_input)
        
        # 3. GENDER LOGIC
        raw_val = float(gen_pred[0][0])
        
        # AI Detection
        ai_detected_female = raw_val > 0.5 
        
        # Determine final gender based on AI or Manual Override
        if override_gender:
            is_female = (manual_gender == "Female")
            st.sidebar.warning(f"AI originally detected: {'Female' if ai_detected_female else 'Male'}")
        else:
            is_female = ai_detected_female

        # 4. SET LABELS
        if is_female:
            gender_tag = "Female"
            aesthetic_label = "Cuteness"
        else:
            gender_tag = "Male"
            aesthetic_label = "Handsomeness"

        # 5. SCORE CALCULATION
        raw_beauty = float(score_pred[0][0])
        final_score = round(max(0, min(10, raw_beauty * 2)), 1)

        # 6. DISPLAY
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader(f"Identity: {gender_tag}")
            st.metric(f"{aesthetic_label} Index", f"{final_score} / 10")
            st.progress(final_score / 10)
            
            # Show the raw AI confidence for debugging
            st.write(f"---")
            st.write(f"**AI Raw Confidence:** {raw_val:.4f}")
            st.caption("Note: Values > 0.5 usually indicate Female detection.")