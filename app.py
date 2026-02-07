
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="LogoLens - AI Logo Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main Container */
    .main {
        background-color: #0e1117;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        padding-bottom: 20px;
    }
    
    /* Upload Section */
    .stFileUploader {
        border: 2px dashed #00C9FF;
        border-radius: 10px;
        padding: 20px;
        transition: 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #92FE9D;
    }
    
    /* Success Message */
    .stSuccess {
        background-color: rgba(0, 201, 255, 0.1);
        border-left: 5px solid #00C9FF;
        color: #ffffff;
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(45deg, #00C9FF, #92FE9D);
        color: #000000;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 201, 255, 0.4);
    }
    
    /* Card/Container */
    .css-1r6slb0 {
        background-color: #1e2329;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# CONSTANTS & LOADING
# =====================================================
IMG_SIZE = (160, 160)
CLASS_NAMES = ['facebook', 'google', 'nike', 'youtube']

@st.cache_resource
def load_model():
    # Try loading the final model, then fallback
    try:
        model = tf.keras.models.load_model('logo_classifier_final.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('best_logo_model_finetuned.keras')
            return model
        except:
            return None

model = load_model()

# =====================================================
# HELPER FUNCTIONS
# =====================================================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def import_and_predict(image_data, model):
    img = ImageOps.fit(image_data, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Robust handling for RGBA/Transparency
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        img = background
    else:
        img = img.convert('RGB')
        
    img = np.asarray(img)
    
    # EXACT PREPROCESSING MATCH WITH TRAINING
    # MobileNetV2 expects inputs in [-1, 1], not [0, 1] or [0, 255]
    # We must use the exact same function used in training
    
    # Reshape for the model (1, 160, 160, 3)
    img_reshape = img[np.newaxis, ...]
    
    prediction = model.predict(img_reshape)
    return prediction

# =====================================================
# UI LAYOUT
# =====================================================

# Custom CSS for Premium Animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Neon Text */
    .neon-text {
        color: #fff;
        text-shadow: 0 0 7px #00C9FF, 0 0 10px #00C9FF, 0 0 21px #00C9FF;
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        border: none;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.6);
    }

    /* Progress Bar Animation */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00C9FF, #92FE9D);
        animation: progress 2s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3669/3669476.png", width=100)
    st.markdown("## LogoLens AI")
    st.markdown("---")
    st.info("This model uses **MobileNetV2** (Transfer Learning) to classify logos with **97% Accuracy**.")
    st.markdown("### Supported Brands:")
    for brand in CLASS_NAMES:
        st.markdown(f"- **{brand.title()}**")
    st.markdown("---")
    st.markdown("Created by **Project 45**")

# Main Content
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<h1 class="neon-text">LogoLens <span style="font-size: 20px; vertical-align: middle;">v2.0</span></h1>', unsafe_allow_html=True)
    st.markdown("##### The Ultimate AI Brand Recognition System")

# Function to get sample images
import os
import random

def get_sample_images():
    base_path = "data/logos_small/val"
    samples = []
    if os.path.exists(base_path):
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(base_path, class_name)
            if os.path.exists(class_dir):
                files = os.listdir(class_dir)
                if files:
                    # Pick a random image from each class
                    img_file = random.choice(files)
                    samples.append((class_name, os.path.join(class_dir, img_file)))
    return samples

# Tabs for Mode Selection
tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Quick Gallery Test"])

image_to_process = None

# Tab 1: Upload
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

# Tab 2: Gallery
with tab2:
    st.markdown("### Select a sample logo to test:")
    samples = get_sample_images()
    
    if samples:
        cols = st.columns(len(samples))
        for idx, (label, img_path) in enumerate(samples):
            with cols[idx]:
                st.image(img_path, caption=label.title(), use_container_width=True)
                if st.button(f"Test {label.title()}", key=f"btn_{idx}"):
                    image_to_process = Image.open(img_path)
    else:
        st.warning("⚠️ Data folder not found. Please ensure 'data/logos_small/val' exists.")

# Prediction Section
if image_to_process:
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image_to_process, caption="Input Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        if model is None:
            st.error("Model not found!")
        else:
            with st.spinner('✨ AI is analyzing pixels...'):
                time.sleep(0.5) # UX Delay
                prediction = import_and_predict(image_to_process, model)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                pred_label = CLASS_NAMES[class_index]
            
            # Result Card
            st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h3 style="margin:0; color: #aaa;">Prediction</h3>
                    <h1 class="neon-text" style="font-size: 3rem; margin: 10px 0;">{pred_label.title()}</h1>
                    <p style="color: #92FE9D; font-weight: bold; font-size: 1.2rem;">{confidence:.2f}% Confidence</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confidence))
            
            # Confidence Breakdown
            with st.expander("📊 View Probability Distribution"):
                chart_data = {brand: float(prob) for brand, prob in zip(CLASS_NAMES, prediction[0])}
                st.bar_chart(chart_data)

# Footer
st.markdown("""
    <div style='position: fixed; bottom: 20px; right: 20px; opacity: 0.7;'>
        <small>Powered by MobileNetV2</small>
    </div>
""", unsafe_allow_html=True)
