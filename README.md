# Urolithiasis Classification: Deep Learning Architectures for Kidney Stone Detection

## Project Overview

This repository contains the implementation and comparative analysis of deep learning architectures for classifying kidney stones in medical images.  
The study evaluates three neural network models — **Deep Neural Network (DNN)**, **Multi-Layer Perceptron (MLP)**, and **Autoencoder-based DNN (AE-DNN)** — on a dataset of **9,416 kidney images** from **Kaggle**.

---

## Key Results

| Model | Accuracy | Specificity | Notes |
|:------|:----------|:-------------|:------|
| **AE-DNN** | 99.47% | 98.87% | Hybrid unsupervised-supervised model |
| **MLP** | 99.67% | — | Optimal computational efficiency |
| **DNN** | 98.95% | — | Robust generalization |
| **All Models** | ROC AUC = 0.987 | — | Consistent performance |

---

## Model Architectures

### 1. Deep Neural Network (DNN)
- High-capacity design with two hidden layers (512 → 256 neurons)  
- ReLU activation, Batch Normalization, and Dropout  
- ~6.5 million parameters  

### ⚙️ 2. Multi-Layer Perceptron (MLP)
- Compact and computationally efficient (128 → 64 neurons)  
- Optimal balance between performance and speed  
- ~1.6 million parameters  

### 🔄 3. Autoencoder-based DNN (AE-DNN)
- Hybrid model combining **unsupervised feature learning** (autoencoder) and **supervised classification**  
- Encoder-decoder for deep feature extraction  
- ~7.2 million parameters  

---

## Dataset

- **Source**: [Kidney Stone Classification and Object Detection (Kaggle)](https://www.kaggle.com/datasets)  
- **Total Images**: 9,416  
  - 4,708 Normal  
  - 4,708 Stone  
- **Image Resolution**: 64×64 pixels (RGB)  
- **Split**:  
  - 80% Training  
  - 20% Validation  
  - 20% Test  

---
## Train Models
 - Train DNN python main.py --model dnn --epochs 10 --batch_size 32
 - Train MLP python main.py --model mlp --epochs 10 --batch_size 32
 - Train AE-DNN python main.py --model ae_dnn --epochs 10 --batch_size 32

---
## Project Strcture

urolithiasis_classification/
├── dataset/                 # Kidney stone images
├── saved_models/            # Trained model weights
├── figs/                    # Results and visualizations
├── main.py                  # Main training script
├── models.py                # Model architectures
├── data_loader.py           # Data preprocessing
├── evaluation.py            # Performance metrics
├── visualization.py         # Plotting functions
├── app.py                   # Web interface
├── config.py                # Configuration settings
├── utils.py                 # Utility functions
└── requirements.txt         # Dependencies

---

## Performance Metrics

| Model  | Accuracy | Sensitivity | Specificity | Precision | Training Time / Epoch |
|:-------|:----------:|:-------------:|:-------------:|:------------:|:----------------:|
| **AE-DNN** | 99.47% | 99.47% | 98.87% | 99.48% | 8.42s |
| **MLP** | 99.67% | 99.67% | 99.44% | 99.67% | 0.58s |
| **DNN** | 98.95% | 98.95% | 97.75% | 98.97% | 6.34s |



