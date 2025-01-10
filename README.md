# Transformer Models on CIFAR-10 Image Classification

This project implements and compares three prominent transformer architectures designed for image classification: Vision Transformer (ViT), Swin Transformer, and Transformer in Transformer (TNT). The models are trained and evaluated on the CIFAR-10 dataset, focusing on performance metrics such as accuracy, precision, recall, F1-score, and AUC.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Models Implemented](#models-implemented)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Hyperparameters](#hyperparameters)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [References](#references)

## Project Overview

Transformers, originally developed for Natural Language Processing (NLP), have been successfully adapted for vision tasks. This project compares three transformer architectures:
- Vision Transformer (ViT)
- Swin Transformer
- Transformer in Transformer (TNT)

The goal is to evaluate their classification performance, computational complexity, and training efficiency on the CIFAR-10 dataset. The code trains these models and provides evaluation metrics such as accuracy, precision, recall, F1-score, and AUC.

## Models Implemented

### Vision Transformer (ViT)
ViT splits images into patches and treats each patch as a token in the transformer model. The model applies self-attention mechanisms to extract features from the patches and uses a global classification token for final predictions.

### Swin Transformer
Swin Transformer introduces a hierarchical vision architecture with shifted windows. It processes images in patches using window-based attention, allowing for local and global feature extraction across multiple stages.

### Transformer in Transformer (TNT)
TNT enhances ViT by adding an additional layer of tokenization. It processes inner-patch tokens, allowing finer-grained information extraction and better performance on image classification tasks.

## Dataset

We use the **CIFAR-10** dataset, a widely-used benchmark for image classification tasks. The dataset contains 60,000 images across 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

- Training Set: 50,000 images
- Test Set: 10,000 images

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/transformer-cifar10.git
   cd transformer-cifar10
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that your system has TensorFlow with GPU support to speed up training:
   ```bash
   pip install tensorflow-gpu
   ```

## Usage

1. **Data Preprocessing and Augmentation:**
   The dataset is preprocessed by normalizing pixel values between 0 and 1. Data augmentation techniques such as random rotations, horizontal flips, and width/height shifts are applied to increase data diversity.

2. **Training the Models:**
   Each model (ViT, Swin, TNT) can be trained using the provided scripts. You can modify the training configurations in the code if needed.
   ```bash
   python Vit.py       # For training Vision Transformer
   python Swin.py      # For training Swin Transformer
   python TNT.py       # For training Transformer in Transformer
   ```

3. **Evaluating the Models:**
   After training, the models are evaluated on the CIFAR-10 test set. The performance metrics such as accuracy, precision, recall, F1-score, AUC, and confusion matrix are computed and visualized.

## Hyperparameters

All models are trained with the following hyperparameters:

- **Batch size:** 64
- **Learning rate:** 0.001 with exponential decay
- **Optimizer:** Adam optimizer
- **Dropout rate:** 0.1 to mitigate overfitting
- **Epochs:** 50 (with early stopping if validation accuracy does not improve for 10 epochs)

You can modify the hyperparameters in the code as needed.

## Evaluation Metrics

The models are evaluated using the following metrics:
- **Accuracy**: Percentage of correct classifications.
- **Precision**: Positive predictive value for each class.
- **Recall**: Sensitivity or true positive rate for each class.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC**: Area Under the Receiver Operating Characteristic Curve.
- **Confusion Matrix**: Visualizes misclassification patterns.

## Results

The following table summarizes the performance of the three transformer models on the CIFAR-10 test set:

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  | Training Time | Inference Time |
|---------------------|----------|-----------|--------|----------|------|---------------|----------------|
| Vision Transformer  | 85.20%   | 0.86      | 0.85   | 0.85     | 0.93 | 12 hours      | 12 ms          |
| Swin Transformer    | 87.90%   | 0.88      | 0.88   | 0.88     | 0.94 | 10 hours      | 10 ms          |
| Transformer in TNT  | 89.70%   | 0.90      | 0.90   | 0.90     | 0.95 | 14 hours      | 13 ms          |

## References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.
2. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners.
3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
4. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030.
5. Han, K., Xiao, A., Wu, E., Guo, J., Xu, C., & Wang, Y. (2021). Transformer in transformer. arXiv preprint arXiv:2103.00112.
6. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images: CIFAR-10 dataset.

---
