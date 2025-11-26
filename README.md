# Urolithiasis Classification: Deep Learning Architectures for Kidney Stone Detection

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](Kidney_Stone_Short_V1.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Official Repository
This is the official repository for the research paper:

**"Deep Learning Architectures for Urolithiasis Classification: A Comparative Analysis of DNN, MLP, and Autoencoder-based Models"**

---

## 🎯 Project Overview
This repository provides the implementation and comparative analysis of deep learning models for **kidney stone classification** using medical imaging.

The study evaluates three architectures:

- **Deep Neural Network (DNN)**
- **Multi-Layer Perceptron (MLP)**
- **Autoencoder-based DNN (AE-DNN)**

A total of **9,416 CT kidney images** from Kaggle were used for the experiments.

---

## 🏆 Key Results

| Model | Accuracy | Specificity | Notes |
|------|----------|-------------|-------|
| **AE-DNN** | 99.47% | 98.87% | Hybrid unsupervised–supervised model |
| **MLP** | 99.67% | 99.44% | Fastest training & highly efficient |
| **DNN** | 98.95% | 97.75% | Strong generalization |
| **All Models** | ROC AUC = 0.987 | — | Consistent overall performance |

---

## 🏗️ Model Architectures

### **1. Deep Neural Network (DNN)**
- Two hidden layers (512 → 256 neurons)  
- ReLU + BatchNorm + Dropout  
- ~6.5M parameters  

### **2. Multi-Layer Perceptron (MLP)**
- Lightweight (128 → 64 neurons)  
- Excellent balance of speed & accuracy  
- ~1.6M parameters  

### **3. Autoencoder-based DNN (AE-DNN)**
- Encoder–decoder for deep feature extraction  
- Supervised classifier after latent layer  
- ~7.2M parameters  

---

## 📊 Dataset

- **Source:** Kaggle — *CT Kidney Dataset: Normal vs Stone*  
- **Total Images:** 9,416  
  - 4,708 Normal  
  - 4,708 Stone  
- **Resolution:** 64×64 RGB  
- **Data Split:**  
  - 64% Training  
  - 16% Validation  
  - 20% Test  

---


