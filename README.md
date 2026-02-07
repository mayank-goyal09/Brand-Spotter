# 🔍🧠 BRAND-SPOTTER: AI Logo Recognition 🧠🔍

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=00C9FF&center=true&vCenter=true&width=1000&lines=Transfer+Learning+Logo+Classifier;MobileNetV2+%2B+Fine-Tuning+Pipeline;Only+~50+Images+Per+Class!;97%25+Accuracy+on+Micro-Dataset;A+Journey+of+Problem-Solving+%26+AI-Powered+Development)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://brand-spotter-projecttt.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/Brand-Spotter?style=for-the-badge)](https://github.com/mayank-goyal09/Brand-Spotter/stargazers)

---

<p align="center">
  <img src="assets/image.png" alt="AI Logo Recognition Banner" width="800"/>
</p>

---

### 🔍 **Classify brand logos with 97%+ accuracy using just ~50 images per class** 🎯

### 🧠 Transfer Learning × MobileNetV2 = **Micro-Data Mastery** 💪

---

## 📖 **THE STORY: A JOURNEY OF CHALLENGES & SOLUTIONS** 📖

> **"This project wasn't just about building a model—it was about overcoming real-world ML obstacles and discovering how AI can accelerate development."**

This README tells the authentic story of how `Brand-Spotter` was built, the **hardships faced**, the **lessons learned**, and how **AI-powered coding (Antigravity)** transformed the development process.

---

## 🚧 **THE PROBLEM VS THE SOLUTION** 🚧

<p align="center">
  <img src="assets/image copy 2.png" alt="Challenge vs Solution" width="700"/>
</p>

<table>
<tr>
<td width="50%">

### 😰 **The Initial Challenge**

When I started this project, I had a simple goal:
> *Build a CNN to classify logos (Facebook, Google, Nike, YouTube).*

But there was a **massive problem**:

| 🔴 Challenge | Impact |
|-------------|--------|
| **Only ~14 images per class** | Not enough data for a CNN to learn |
| **Custom CNN from scratch** | Overfitting within 5 epochs |
| **Validation accuracy stuck at ~60%** | Model was guessing, not learning |

**The harsh reality**: Deep learning typically needs **thousands** of images per class. I had **14**. 💀

</td>
<td width="50%">

### ❓ **Why It Failed**

My first approach was a **simple custom CNN**:

```python
# ❌ MY ORIGINAL (FAILED) APPROACH
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, activation="relu")(x)
x = layers.Flatten()(x)  # ← MISTAKE!
x = layers.Dense(128, activation="relu")(x)
```

**What went wrong:**
- ❌ Training from scratch with 14 images
- ❌ Using `Flatten()` (huge parameters, no spatial info)
- ❌ Weak data augmentation
- ❌ No callbacks (early stopping, LR scheduling)
- ❌ Single-stage training

**Result**: Model memorized training data → failed on validation.

</td>
</tr>
</table>

---
