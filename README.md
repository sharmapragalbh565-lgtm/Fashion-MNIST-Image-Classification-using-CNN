# Fashion-MNIST Image Classification using CNN
Deep Learning with TensorFlow, Keras & Data Visualization

This project builds a Convolutional Neural Network (CNN) to classify grayscale clothing images from the Fashion-MNIST dataset using TensorFlow Keras.

In addition to model training, this implementation includes:

1. Image visualization

2. Training & validation accuracy plots

3. Loss curve visualization

4. Label decoding

# Dataset Overview

Fashion-MNIST was created by Zalando as a more challenging replacement for MNIST digits.

60,000 training images

10,000 testing images

28x28 grayscale images

10 clothing categories

# Class Labels
Label	Category
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

# Project Workflow
Load Dataset (CSV)
        ↓
Preprocessing
  - Normalize (0–255 → 0–1)
  - Reshape to 28×28×1
  - One-hot encode labels
        ↓
Train/Validation Split (80/20)
        ↓
Build CNN Model
        ↓
Compile (Adam + CrossEntropy)
        ↓
Train (15 Epochs)
        ↓
Visualize:
  - Sample Images
  - Accuracy Curve
  - Loss Curve
   
# Data Preprocessing
1️. Normalization

Pixel values are scaled to improve training stability:

X = X / 255.0

2️. Reshaping for CNN
X = X.reshape(-1, 28, 28, 1)

Adds a channel dimension for convolution layers.

3️. One-Hot Encoding
y = to_categorical(y, 10)

Converts integer labels into categorical vectors.

# CNN Architecture
Conv2D (32 filters, 3×3) → ReLU
MaxPooling2D (2×2)
Dropout (25%)

Conv2D (64 filters, 3×3) → ReLU
MaxPooling2D (2×2)
Dropout (25%)

Flatten
Dense (128) → ReLU
Dropout (50%)
Dense (10) → Softmax

# Why This Architecture?

Convolution Layers → Extract image features

Pooling Layers → Reduce dimensionality

Dropout → Prevent overfitting

Dense Layers → Perform classification

# Model Compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy


# Accuracy & Loss Curves

The project plots:

Training Accuracy vs Validation Accuracy

Training Loss vs Validation Loss

These graphs help:

Detect overfitting

Monitor convergence

Analyze learning stability

# Expected Results

Typical performance:

1. 88–92% Validation Accuracy

2. Smooth convergence after ~10 epochs

3. Slight generalization gap due to dropout

# Installation
git clone https://github.com/yourusername/fashion-mnist-cnn.git
cd fashion-mnist-cnn
pip install -r requirements.txt
python train.py

# Requirements
pandas
numpy
matplotlib
scikit-learn
tensorflow

# Key Concepts Demonstrated

Convolutional Neural Networks (CNN)

Image preprocessing

Overfitting prevention with Dropout

Multi-class classification

Training history visualization

Model performance analysis

Validation-based evaluation

# Possible Improvements

Add EarlyStopping

Add ModelCheckpoint

Add Data Augmentation

Use BatchNormalization

Add Confusion Matrix

Convert to Transfer Learning (ResNet / EfficientNet)

Deploy as Web App (Streamlit / Flask)

Convert to PyTorch

# Real-World Applications

E-commerce product classification

Visual recommendation engines

Automated inventory systems

Retail AI analytics
