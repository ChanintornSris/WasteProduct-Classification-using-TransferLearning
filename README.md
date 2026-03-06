# Waste Products Classification

This project automates waste product classification using Transfer Learning. By leveraging Keras' advanced functions for data generation and fine-tuning a pre-trained VGG16 model, we optimized performance to accurately separate organic and recyclable waste.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training &amp; Fine-Tuning](#training--fine-tuning)
- [File Structure](#file-structure)

## Overview

Proper waste segregation is a critical step towards effective recycling and environmental sustainability. This project utilizes deep learning (a Convolutional Neural Network based on the VGG16 architecture) to classify images into two categories:

- **O (Organic)**
- **R (Recyclable)**

## Dataset

The project makes use of the "Waste Classification Dataset" from Kaggle, structured into a standard `train` and `test` split under the `o-vs-r-split` directory. The dataset features raw images of organic and recyclable waste products.

## Technologies

- **Python**
- **TensorFlow / Keras** (v2.17.0)
- **NumPy**
- **Matplotlib**
- **scikit-learn**

## Methodology

### Data Preprocessing

Data pipelines were built using Keras' `ImageDataGenerator` to efficiently load batches of images. The images were resized and rescaled (pixel values between 0 and 1). To prevent overfitting, the training data was augmented with real-time transformations such as:

- Width shifts
- Height shifts
- Horizontal flips

### Model Architecture

The core classification model is built upon **VGG16**, a convolutional neural network architecture pre-trained on the ImageNet dataset. Transfer learning was employed because it yields robust feature extraction capabilities without having to train a massive network from scratch.

### Training & Fine-Tuning

The training was conducted in two primary stages:

1. **Feature Extraction:** The pre-trained VGG16 base layers were frozen. A custom classification head was added on top, and this head was trained to recognize the features extracted by VGG16 for our specific waste classification dataset.
2. **Fine-Tuning:** To push performance further, specific deeper layers of the VGG16 base (`block5_conv3` and `block5_pool` in addition to the fully connected layers) were unfrozen. The model was then fine-tuned, allowing the network to subtly adjust its pre-learned features to better suit the organic vs. recyclable problem. Loss and accuracy metrics were logged and visualized over multiple epochs.

## File Structure

- `food_waste_classification_using_tensorflow.ipynb`: Jupyter Notebook containing the full pipeline (data augmentation, model generation, training, and evaluation).
- `O_R_tlearn_vgg16.keras`: The model saved after the initial feature extraction stage.
- `O_R_tlearn_fine_tune_vgg16.keras`: The final, fine-tuned VGG16 model.
- `o-vs-r-split/`: Directory containing the categorized waste images.
