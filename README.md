# 🖼️ Image Caption Generation using DenseNet201 + Bidirectional LSTM

## 📌 Project Overview

This project implements an end-to-end Image Caption Generation system
combining Computer Vision and Natural Language Processing techniques.

The system extracts high-level visual features using a pre-trained
DenseNet201 CNN backbone and generates descriptive captions using a
Bidirectional LSTM-based sequence model.

------------------------------------------------------------------------

## 🏗️ System Architecture

### 1️⃣ Visual Feature Extraction (Encoder)

-   Backbone: DenseNet201 (pre-trained on ImageNet)
-   Input size: 224x224
-   Output: 1920-dimensional feature vector
-   Feature extraction layer: Second last layer of DenseNet

Images are: - Resized - Normalized - Passed through DenseNet201 -
Features stored for training

------------------------------------------------------------------------

### 2️⃣ Custom Data Generator

Implemented a custom Keras Sequence-based generator with:

-   Dynamic batch loading
-   Caption tokenization
-   Padding to fixed max length
-   One-hot encoding of target words
-   Feature-level noise injection (30% probability)
-   Shuffle at epoch end

Noise injection improves generalization and prevents overfitting.

------------------------------------------------------------------------

### 3️⃣ Multimodal Fusion Model

#### Image Feature Branch

-   Dense(512) + L2 regularization
-   Batch Normalization
-   Dropout (0.3)
-   Reshape for fusion

#### Text Branch

-   Embedding layer (512-dim)
-   Dropout (0.3)

#### Fusion & Sequence Modeling

-   Concatenation of image + text features
-   Bidirectional LSTM (256 units)
-   Residual connection with image features
-   Fully connected layers with heavy regularization

#### Model architecture
Complete Architecture Summary

-   The final model consists of:
-   CNN Encoder (DenseNet201) → Extract visual embeddings
-   Feature Projection Layer → Dense + Regularization
-   Embedding Layer → Word vector representation
-   Bidirectional LSTM Decoder → Context modeling
-   Residual Fusion Layer → Combine image + text features
-   Fully Connected Layers → Classification head
-   Softmax Output Layer → Predict next word

#### Output

-   Dense(vocab_size)
-   Softmax activation
-   Loss: Categorical Crossentropy
-   Optimizer: Adam (learning rate = 0.0003)

------------------------------------------------------------------------

## ⚙️ Training Strategy

### Callbacks Used

-   ModelCheckpoint (save best model)
-   EarlyStopping (patience=7, restore best weights)
-   ReduceLROnPlateau (factor=0.5, patience=3)

### Regularization Techniques

-   Dropout (0.3 -- 0.5)
-   L2 Regularization
-   Batch Normalization
-   Feature Noise Injection
-   Adaptive Learning Rate Scheduling

------------------------------------------------------------------------
### Observations:

-   Training loss decreases steadily
-   Validation loss decreases and stabilizes
-   Slight generalization gap indicates mild overfitting
-   Early stopping prevents degradation

Model convergence observed around 30--35 epochs.

------------------------------------------------------------------------

## 🔬 Technical Highlights

-   Multimodal Deep Learning (Vision + Text)
-   Pretrained CNN Feature Extraction
-   Bidirectional LSTM for sequence modeling
-   Residual feature fusion
-   Custom batch generator implementation
-   Structured regularization strategy
-   Training stability optimization

------------------------------------------------------------------------

## 📜 Conclusion

This project demonstrates a complete deep learning pipeline for image
caption generation using multimodal fusion and advanced training
strategies. It showcases an understanding of CNN feature extraction,
sequence modeling, regularization, and optimization techniques in deep
learning.
