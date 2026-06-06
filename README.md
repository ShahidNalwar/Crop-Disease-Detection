# Crop Disease Detection

A deep learning project using Convolutional Neural Networks (CNN)
trained on the PlantDoc dataset to detect diseases from crop leaf
images across multiple crop types.

## Problem

Farmers often cannot identify crop diseases early enough to prevent
yield loss. Manual identification requires agricultural expertise
that is not available in rural areas.

## Solution

A CNN model trained on the PlantDoc dataset that classifies crop
leaf images into healthy or diseased categories across 17 crop
types and 27 disease classes.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | TensorFlow / Keras |
| Model | Custom CNN |
| Dataset | PlantDoc |
| Environment | Jupyter Notebook |
| Visualization | Matplotlib |

## Architecture
Crop leaf image input
|
Image preprocessing
(resize, normalize, augment)
|
CNN Feature Extraction
(Conv layers + MaxPooling)
|
Dense Classification layers
|
Disease class prediction

## Dataset

PlantDoc dataset — 2,598 images across 17 crop species and
27 disease categories. Available on Kaggle.

## Key Features

- Multi-crop disease detection (tomato, potato, corn, wheat, etc.)
- Data augmentation (rotation, flipping, zoom) to handle limited data
- Transfer learning experimentation
- Confidence score output per prediction

## Design Tradeoffs

- Limited dataset size — used aggressive data augmentation
  (rotation, horizontal flip, zoom, brightness adjustment)
- Chose custom CNN over ResNet for learning purposes, though
  ResNet would give better accuracy in production
- Jupyter Notebook format for reproducibility and sharing

## What I Would Do Differently

- Deploy as a mobile app (Flutter) for field use by farmers
- Use transfer learning with EfficientNet for better accuracy
- Add bounding box detection to localize disease on the leaf
- Build a multilingual interface (Hindi, Marathi) for rural farmers

## Setup

```bash
pip install tensorflow keras numpy matplotlib jupyter
jupyter notebook "Crop_Disease_Detection_Final - Copy.ipynb"
```

Download PlantDoc dataset from Kaggle before running.

## Results

Trained CNN achieves multi-class classification across 27 disease
categories with data augmentation improving generalization on the
limited PlantDoc dataset.
