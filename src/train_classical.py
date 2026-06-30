import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_PATH, TEST_PATH,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    RANDOM_SEED
)


def load_data(train_path, test_path):
    print("Loading train/test data...")
    train = pd.read_csv(train_path, encoding='utf-8-sig')
    test  = pd.read_csv(test_path,  encoding='utf-8-sig')

    train = train.dropna(subset=['Sentence', 'Region'])
    test  = test.dropna(subset=['Sentence', 'Region'])

    print(f"  Train: {len(train):,} samples")
    print(f"  Test:  {len(test):,} samples\n")
    return train, test


#TF-IDF vectorization 
def build_tfidf(train_sentences, test_sentences):
    print("Building word-level TF-IDF features...")
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_sentences)
    X_test  = vectorizer.transform(test_sentences)
    print(f"  TF-IDF matrix shape: {X_train.shape} (train)")
    print(f"  TF-IDF matrix shape: {X_test.shape}  (test)\n")
    return X_train, X_test, vectorizer


#Train and evaluate one model 
def train_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    print(f"Training {model_name}...")

    # Training
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 4)

    # Inference
    t1 = time.time()
    y_pred = model.predict(X_test)
    inference_time = round(time.time() - t1, 4)

    # Metrics
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1       = round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4)

    print(f"  Accuracy:       {accuracy}")
    print(f"  Macro F1:       {f1}")
    print(f"  Training time:  {train_time}s")
    print(f"  Inference time: {inference_time}s\n")

    return {
        'model':          model_name,
        'y_true':         y_test.tolist(),
        'y_pred':         y_pred.tolist(),
        'accuracy':       accuracy,
        'f1':             f1,
        'train_time':     train_time,
        'inference_time': inference_time
    }

def run(train=None, test=None):
    if train is None or test is None:
        train, test = load_data(TRAIN_PATH, TEST_PATH)

    X_train, X_test, vectorizer = build_tfidf(
        train['Sentence'].values,
        test['Sentence'].values
    )

    y_train = train['Region'].values
    y_test  = test['Region'].values

    models = [
        (
            MultinomialNB(),
            'Naive Bayes'
        ),
        (
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver='lbfgs'
            ),
            'Logistic Regression'
        ),
        (
            LinearSVC(
                max_iter=2000,
                random_state=RANDOM_SEED
            ),
            'Linear SVM'
        )
    ]

    results = []
    for model, name in models:
        result = train_evaluate(
            model, name,
            X_train, X_test,
            y_train, y_test
        )
        results.append(result)

    print("Classical model training complete.")
    return results


if __name__ == '__main__':
    results = run()