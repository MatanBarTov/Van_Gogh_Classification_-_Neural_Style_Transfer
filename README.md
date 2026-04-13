# Van Gogh Classification and Neural Style Transfer

This repository contains my final project for the Introduction to Deep Learning course (Tel Aviv University, 2025/2026).

The project combines **transfer learning, neural style transfer, hyperparameter optimization, and model interpretability** in one end-to-end computer vision pipeline. The goal was to classify Van Gogh paintings versus other Post-Impressionist works, and then use the trained classifiers to evaluate generated Van Gogh-style images.

* * *

## Project Overview

This project is divided into two connected parts:

- **Part 1:** Fine-tune pretrained CNNs (**AlexNet** and **VGG-19**) to classify whether a painting was created by **Vincent van Gogh**
- **Part 2:** Implement a generic **Neural Style Transfer** pipeline and generate Van Gogh-style outputs from new images
- **Evaluation:** Use the trained classifiers as quantitative judges of how “Van Gogh-like” the generated images are
- **Bonus:** Use **Grad-CAM** to understand which visual regions the classifiers relied on

This project was designed to demonstrate practical skills relevant to **Computer Vision** and **Deep Learning** roles: model training, hyperparameter search, evaluation, benchmarking, and interpretability.

* * *

## Methodology

The project follows a full deep learning workflow:

1. **Dataset Preparation**
   - Filtered a Post-Impressionism subset from WikiArt
   - Built a binary target: **Van Gogh** vs **Not Van Gogh**
   - Applied train / validation / test splitting with class balance preservation

2. **Transfer Learning**
   - Fine-tuned **AlexNet** and **VGG-19** with ImageNet-pretrained weights
   - Froze convolutional feature extractors and replaced the final classifier head
   - Used a unified **224×224** preprocessing pipeline with ImageNet normalization

3. **Data Augmentation**
   - `RandomResizedCrop`
   - `RandomHorizontalFlip`
   - `ColorJitter`

4. **Hyperparameter Optimization**
   - Used **Optuna** to tune:
     - learning rate
     - batch size
     - weight decay
   - Logged all experiments with **Weights & Biases**
   - Used different search strategies for AlexNet and VGG-19 based on runtime constraints

5. **Neural Style Transfer**
   - Implemented a generic style transfer function for both **AlexNet** and **VGG-19**
   - Used **content loss + Gram-matrix style loss**
   - Supported configurable content/style layers and style-layer weights
   - Tuned style-transfer hyperparameters using a classifier-as-judge objective

6. **Interpretability & Benchmarking**
   - Applied **Grad-CAM** to inspect classifier attention
   - Compared **CPU vs GPU** runtime to verify actual hardware usage

* * *

## Main Results

- **AlexNet**
  - AUC-ROC = **0.9739**
  - Precision = **0.9306**

- **VGG-19**
  - Accuracy = **0.9473**
  - F1-score = **0.8154**
  - Recall = **0.7513**

### Key takeaways

- **VGG-19** was stronger at recovering Van Gogh paintings and achieved better recall / F1
- **AlexNet** was more conservative and achieved higher precision
- In style transfer, **VGG-19-based NST** generally preserved structure more coherently
- **AlexNet-based NST** often produced stronger but noisier stylization
- Grad-CAM analysis suggested that both models relied mainly on **local texture cues** such as:
  - brushstroke-like patterns
  - swirling structures
  - edge density
  - color-texture combinations


* * *

## Repository Structure

- `main_notebook.ipynb`  
  Full implementation, including data preparation, model training, Optuna tuning, style transfer, evaluation, benchmarking, and Grad-CAM analysis

- `Project_Report.pdf`  
  Full project report with methodology, experiments, results, visualizations, and discussion

* * *

## Technologies & Tools

- **Python**
- **PyTorch**
- **torchvision**
- **Optuna**
- **Weights & Biases**
- **Matplotlib**

* * *

## Files in This Repository

- `main_notebook.ipynb`
- `Project_Report.pdf`
- `README.md`

* * *

## Authors

- **Matan Bar Tov**
- **Omri Yarkoni**

* * *

This project reflects my interest in building strong, technically grounded computer vision and deep learning systems, with emphasis on both **model performance** and **understanding what the model actually learned**.
