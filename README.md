# NLP Content Moderation with Transformers
COMP 9130 Mini Project 9

## Problem Description and Motivation
Online platforms need reliable systems to identify between hate speech, offensive language, and acceptable content.

Missing hate speech creates safety, ethical, and legal risks. Over censoring offensive but non hateful content can reduce user trust and damage platform credibility. It can be annoying for some users.

This project builds a 3 class text classifier for moderation and compares:
- TF-IDF + Logistic Regression as the baseline
- DistilBERT as the transformer model

The goal is to evaluate which approach performs better for real moderation use, especially under strong class imbalance.

## Dataset Description
- **Dataset**: Hate Speech and Offensive Language Dataset
- **Source**: https://arxiv.org/abs/1703.04009
- **Size**: 24,783 tweets

### Class Distribution
- **Hate Speech**: 5.77%
- **Offensive Language**: 77.43%
- **Neither**: 16.80%

The dataset is highly imbalanced, so we used macro F1  since it will be more informative than accuracy alone.

### Preprocessing
The text was cleaned using the following steps:
- Lowercasing
- Replacing URLs with `URL`
- Replacing mentions with `USER`
- Converting hashtags into words
- Removing non-ASCII characters
- Collapsing extra whitespace

## Setup Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
torch
transformers
scikit-learn
pandas
numpy
matplotlib
tqdm
```

### How to Run
1. Open the notebook:
   ```bash
   mini_project_9_all_in_one_colab.ipynb
   ```

2. Run all cells in order

## Results Summary

### Baseline vs Transformer Comparison

| Metric       | TF-IDF + Logistic Regression | DistilBERT |
|--------------|------------------------------|------------|
| Accuracy     | 0.8779                       | 0.9147     |
| Macro F1     | 0.7387                       | 0.7890     |
| Weighted F1  | 0.8849                       | 0.9162     |

### Summary
The transformer outperformed the baseline on all metrics.

DistilBERT handled contextual language better than the TF-IDF baseline, but both models still struggled with:
- tone
- slang
- quoted speech
- borderline cases between hate speech and offensive language

## 5. Team Member Contributions
- Vibhor: Code, Analysis, Report
- Bryan: Analysis, Git, Report

## 6. References
- Hugging Face Transformers Documentation
- PyTorch Documentation
- scikit-learn Documentation
