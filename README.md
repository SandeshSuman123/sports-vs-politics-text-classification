# Sports vs Politics Text Classification

## Project Overview

This project implements a binary text classification system that classifies documents into:

-  **Sports**
-  **Politics**

The objective is to compare multiple machine learning algorithms and feature representations to determine which approach performs best for this task.

This project was developed as part of:

**Course:** CSL 7640 – Natural Language Understanding  
**Author:** Sandesh Suman  
**Roll Number:** M25CSA034  

---

## Dataset Description

- Total Documents: **120**
- Sports: **60**
- Politics: **60**
- Balanced Dataset 
- 80% Training / 20% Testing Split

The dataset was manually curated in a structured news-report style.  
To avoid trivial classification, ambiguous samples were introduced:

---

## Text Preprocessing

The following preprocessing steps were applied:

- Lowercasing
- URL removal
- Removal of special characters
- Tokenization (whitespace-based)
- Stopword removal

---

## Feature Representation

Three feature extraction techniques were used:

1. **Bag of Words (BoW)**
2. **TF-IDF**
3. **Bigrams (n-grams)**

TF-IDF assigns higher weight to informative words, improving classification performance.

---

## Machine Learning Algorithms

Three machine learning techniques were implemented from scratch:

1. **Naive Bayes (Multinomial)**
2. **Logistic Regression (Gradient Descent + L2 Regularization)**
3. **Support Vector Machine (Linear SVM-style implementation)**

---

## Experimental Results

| Model | Feature | Accuracy | F1 Score |
|--------|----------|------------|------------|
| Naive Bayes | BoW | 90% | 87.50% |
| Naive Bayes | TF-IDF | 90% | 87.50% |
| Logistic Regression | TF-IDF | **95%** | **94.74%** |
| SVM | TF-IDF | 90% | 87.50% |

### Best Performing Model
**Logistic Regression with TF-IDF** achieved the highest accuracy of **95%**.




