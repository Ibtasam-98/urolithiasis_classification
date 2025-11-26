# Urolithiasis Classification: Deep Learning Architectures for Kidney Stone Detection

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](Kidney_Stone_Short_V1.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Official Repository
This is the official repository for the research paper:

**"Deep Learning Architectures for Urolithiasis Classification: A Comparative Analysis of DNN, MLP, and Autoencoder-based Models"**

---

## Project Overview
This repository provides the implementation and comparative analysis of deep learning models for **kidney stone classification** using medical imaging.

The study evaluates three architectures:

- **Deep Neural Network (DNN)**
- **Multi-Layer Perceptron (MLP)**
- **Autoencoder-based DNN (AE-DNN)**

A total of **9,416 CT kidney images** from Kaggle were used for the experiments.

---

## Key Results

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

## Dataset

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

# Train DNN
python main.py --model dnn --epochs 10 --batch_size 32

# Train MLP
python main.py --model mlp --epochs 10 --batch_size 32

# Train AE-DNN
python main.py --model ae_dnn --epochs 10 --batch_size 32

# Model Architectures Overview

This document outlines the design and key specifications for the various deep learning models used in this project.

---

## 1. Deep Neural Network (DNN)

A high-capacity design chosen for its potential to capture complex, high-dimensional features.

| Feature | Specification |
| :--- | :--- |
| **Architecture** | Standard Deep Neural Network |
| **Hidden Layers** | 2 |
| **Layer 1 Size** | 512 neurons |
| **Layer 2 Size** | 256 neurons |
| **Activation** | ReLU (Rectified Linear Unit) |
| **Regularization** | Batch Normalization and Dropout |
| **Parameter Count** | Approximately **6.5 million** |

---

## 2. Multi-Layer Perceptron (MLP)

A compact and computationally efficient model designed to achieve an optimal balance between performance and inference speed.

| Feature | Specification |
| :--- | :--- |
| **Architecture** | Simple Multi-Layer Perceptron |
| **Hidden Layers** | 2 |
| **Layer 1 Size** | 128 neurons |
| **Layer 2 Size** | 64 neurons |
| **Activation** | Standard (e.g., ReLU or Tanh, specify if known) |
| **Parameter Count** | Approximately **1.6 million** |

---

## 3. Autoencoder-based DNN (AE-DNN)

A hybrid model that leverages **unsupervised feature learning** for robust feature extraction before supervised classification. 


### Core Components

* **Autoencoder (AE):** Used as an initial component for **deep feature extraction**.
    * Consists of an **Encoder** and a **Decoder**.
    * The Encoder output (bottleneck) serves as the compressed, high-level features passed to the final classification layers.
* **DNN Classifier:** The subsequent fully connected layers for the final supervised task.

| Feature | Specification |
| :--- | :--- |
| **Architecture** | Hybrid: Autoencoder + DNN |
| **Primary Goal** | Feature Learning followed by Supervised Classification |
| **Parameter Count** | Approximately **7.2 million** |


