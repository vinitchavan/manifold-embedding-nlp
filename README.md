# README.md

# Manifold-Constrained Sentence Embeddings

This repository contains code for our research on generating sentence embeddings constrained to lie on geometric manifolds (Sphere, Torus, Möbius Strip). The method is trained using triplet loss and evaluated on benchmark datasets for clustering and classification tasks.

## 🔍 Overview

Traditional sentence embeddings are trained in unconstrained Euclidean spaces. This project introduces a novel approach to learn embeddings directly on differentiable manifolds using geometric projection layers and contrastive learning.

## ✨ Features
- Sphere, Torus, and Möbius strip manifold projections
- Triplet loss-based training
- Silhouette Score and accuracy evaluation
- 3D visualization of learned embeddings

## 📁 Project Structure
```
models/
    manifold_embedding.py     # Custom manifold projection class
train_model.py                # Triplet loss training
evaluate_model.py             # Classification and clustering evaluation
visualize_embeddings.py       # 3D embedding visualization
notebook/
    your_colab_notebook.ipynb # Colab notebook with experiments
requirements.txt              # Required packages
```

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🧪 Running Experiments
Customize the `train_model.py` and `evaluate_model.py` scripts to train your model and evaluate on your dataset.

## 📊 Example Visualization
Use `visualize_embeddings.py` to plot your embeddings in 3D space.

## 🧠 Datasets
- [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [MBTI Personality Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)

## 📄 Citation
Coming soon — arXiv preprint submission.

---

📬 Contact: vinitchavan83@gmail.com


MIT License
