# Emotion Classification with Naive Bayes 🧠

A 2024 University of Groningen group project classifying seven emotions in short English essays with TF-IDF features and Complement Naive Bayes.

> **Archive note**
>
> This README was refreshed in 2026 to document the project more clearly. The implementation and results remain from the original coursework.

## 🔎 Pipeline

1. Lowercase the essays and remove punctuation, stop words, and non-alphabetic tokens.
2. Split the course-provided training data into training and validation sets.
3. Convert the cleaned text into count vectors and apply TF-IDF weighting.
4. Train a Complement Naive Bayes classifier across anger, disgust, fear, joy, neutral, sadness, and surprise.
5. Evaluate the saved model on a held-out test set of 270 essays.

## 📊 Results

| Metric | Test result |
| --- | ---: |
| Accuracy | 47.0% |
| Weighted F1 | 44.3% |
| Weighted precision | 44.7% |
| Weighted recall | 47.0% |

The class distribution is uneven, and sadness is both the largest class and the model's strongest category. The results are best read as an interpretable baseline and a practical lesson in evaluating classifiers under class imbalance.

## 🛠️ Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python preprocessing/preprocessing.py
python model/NB_Train.py
python model/NB_Test.py
```

The preprocessing step downloads the NLTK English stop-word list and rewrites the prepared CSV files. Training stores the classifier, vectorizer, and TF-IDF transformer in `results/model_nb.pkl`.

## 📁 Repository Guide

- `data/raw/` contains the original course data
- `data/preprocessed/` contains the cleaned training, validation, and test splits
- `preprocessing/` contains the text-cleaning pipeline
- `model/` contains training and evaluation scripts
- `results/` contains the serialized model artifact

## 👥 Team

Built by [Joris Postmus](https://github.com/jorispos), [Leon Tanis](https://github.com/Tanis1304), and [Joris Suurmeijer](https://github.com/Jorissuurmeijer) for the University of Groningen Natural Language Processing course.
