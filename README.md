# LogoLens 🔍 - AI Logo Classifier

![Project Banner](https://img.shields.io/badge/Status-Active-success) ![Accuracy](https://img.shields.io/badge/Accuracy-97.6%25-brightgreen) ![Tech](https://img.shields.io/badge/Tech-TensorFlow%20%7C%20MobileNetV2%20%7C%20Streamlit-blue)

A professional, high-accuracy Deep Learning application capable of identifying brand logos from images. Built using **Transfer Learning** with MobileNetV2 architecture to achieve exceptional performance even with small datasets.

---

## 🚀 Features

- **High Accuracy (97%+)**: Utilizing the power of pre-trained ImageNet weights.
- **Robustness**: Trained with aggressive data augmentation to handle various angles, lighting, and orientations.
- **Micro-Dataset Learning**: Capable of learning robust features from as few as ~15 images per class using 2-stage training (Frozen Base -> Fine Tuning).
- **Interactive UI**: A stunning dark-mode web application built with Streamlit for real-time predictions.

## 🧠 Model Architecture

We moved away from a custom "from-scratch" CNN to a **Transfer Learning** approach to solve the problem of limited training data.

1.  **Base Model**: MobileNetV2 (Pre-trained on ImageNet), frozen initially.
2.  **Head**:
    -   `GlobalAveragePooling2D` (Preserves spatial info better than Flatten)
    -   `BatchNormalization` (Stabilizes training)
    -   `Dropout (0.5)` (Prevents overfitting)
    -   `Dense (Softmax)` (Final classification)
3.  **Training Strategy**:
    -   **Stage 1**: Train only the head (30 epochs, LR=1e-3).
    -   **Stage 2**: Unfreeze top 30 layers of MobileNetV2 and fine-tune (15 epochs, LR=1e-5).

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "project 45 logo CNN"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure

```
project 45 logo CNN/
├── app.py                  # Main Streamlit Application
├── main_new.ipynb         # Improved Training Notebook (Transfer Learning)
├── main.ipynb             # Original Training Experiment
├── logo_classifier_final.keras  # Saved Trained Model
├── requirements.txt       # Project Dependencies
└── data/                  # Dataset Directory
    └── logos_small/       # Train/Val Split
```

## 📊 Supported Brands
- Facebook
- Google
- Nike
- YouTube

## 🤝 Contributing
Feel free to open issues or submit pull requests if you want to add more brands to the dataset!
