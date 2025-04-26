---
tags:
- vision
- image-classification
- flowchart-detection
- pytorch
- VisionTransformer
---

# Flowchart Classification Model

## Model Description

This is a Vision Transformer (ViT) model fine-tuned for binary classification of images into **flowchart** or **non-flowchart** categories. The model achieves **~85% accuracy** on validation data.

- **Model type:** Vision Transformer (ViT-Base)
- **Input:** 224x224 RGB images
- **Output:** Binary classification (flowchart/non-flowchart)
- **Training data:** Custom dataset of 10,000+ flowchart/non-flowchart images

## Intended Uses

This model is designed for:

- Classifying if given image is a flowchart or not


## Huggingface
[Link](https://huggingface.co/abirmoy/flowchart_vit.pth)
