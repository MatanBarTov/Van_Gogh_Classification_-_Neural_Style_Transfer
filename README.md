# Van Gogh Classification and Neural Style Transfer

A deep learning project that combines transfer learning, generative modeling, hyperparameter optimization, and interpretability in a single end-to-end computer vision pipeline.

The project has two goals:

1. Classify whether a painting is by Vincent van Gogh or not
2. Generate Van Gogh-style images using neural style transfer, then evaluate those outputs with trained classifiers

This project was built to demonstrate practical deep learning skills relevant to Computer Vision and Deep Learning Engineer roles: model design, training, tuning, evaluation, benchmarking, and interpretation.

## Project Overview

The classification stage fine-tunes two pretrained CNNs, AlexNet and VGG-19, on a filtered Post-Impressionism subset of WikiArt. The style transfer stage implements a generic neural style transfer pipeline and uses a trained classifier as a quantitative "judge" of Van Gogh-like outputs.

The project also includes:

- Optuna hyperparameter search
- Weights & Biases experiment tracking
- CPU vs GPU benchmarking
- Grad-CAM-based interpretability analysis

## Why This Project Matters

This repository is more than a course submission. It shows how I approach an applied computer vision problem end to end:

- Define a practical modeling objective
- Compare multiple architectures instead of relying on a single model
- Tune hyperparameters systematically
- Analyze failure modes, not just final metrics
- Connect discriminative and generative methods in one workflow
- Use interpretability tools to understand what the model actually learned

## Key Results

- **AlexNet:** AUC-ROC = **0.9739**, Precision = **0.9306**
- **VGG-19:** Accuracy = **0.9473**, F1 = **0.8154**, Recall = **0.7513**

In practice:

- **VGG-19** was stronger at recovering Van Gogh paintings
- **AlexNet** was more conservative and more precise

In the neural style transfer comparison:

- **VGG-19-based NST** generally preserved image structure more coherently
- **AlexNet-based NST** often produced stronger but noisier stylization

Grad-CAM analysis further suggested that both classifiers relied mainly on local texture cues such as brushstrokes, swirling patterns, edge density, and color-texture combinations rather than object-level semantics.

## Technical Highlights

### 1. Transfer Learning for Fine-Grained Art Classification

- Fine-tuned **AlexNet** and **VGG-19** with ImageNet-pretrained weights
- Froze the convolutional feature extractor and replaced the final classifier layer for binary prediction
- Used a unified **224×224** preprocessing pipeline
- Applied **ImageNet normalization**
- Used training augmentations including:
  - `RandomResizedCrop`
  - `RandomHorizontalFlip`
  - `ColorJitter`

### 2. Systematic Hyperparameter Optimization

- Used **Optuna** to search over:
  - learning rate
  - weight decay
  - batch size
- Logged runs and learning curves with **Weights & Biases**
- AlexNet tuning used **4-fold cross-validation**
- VGG-19 tuning used a **train/validation split** due to runtime constraints

### 3. Neural Style Transfer as an Optimization Problem

- Implemented a generic style transfer function for both **AlexNet** and **VGG-19** backbones
- Used **content loss + Gram-matrix style loss**
- Supported configurable content/style layers and per-layer style weights
- Normalized Gram matrices and style-layer weights for more stable optimization
- Tuned style-transfer hyperparameters with Optuna using a classifier-as-judge objective

### 4. Model Interpretation with Grad-CAM

- Used Grad-CAM to visualize which image regions most influenced Van Gogh predictions
- Compared true positives and false positives across AlexNet and VGG-19
- Identified a texture-driven learned signature rather than a semantic one

### 5. Hardware Benchmarking

- Included a CPU vs GPU benchmark to verify actual GPU usage
- Reported training on an **NVIDIA GeForce RTX 4090**
- Measured a single epoch at:
  - **204.739s on GPU**
  - **215.475s on CPU**

This relatively small speedup likely reflects preprocessing and data-loading bottlenecks rather than pure model compute.

## Repository Structure

```text
.
├── main_notebook.ipynb
└── Project_Report.pdf
