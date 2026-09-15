"""
Text Classification Pipeline

Practice:
- TF-IDF text vectorization
- Logistic Regression for text classification
- Train/test split
- Model evaluation
- Prediction probabilities
- Reusable inference function
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. Sample Dataset
# ---------------------------------------------------------

data = pd.DataFrame(
    {
        "text": [
            "I love this product",
            "This is an amazing experience",
            "The service was excellent",
            "Absolutely fantastic product",
            "I am very happy with this purchase",
            "This works really well",
            "I hate this product",
            "This was a terrible experience",
            "The service was awful",
            "Absolutely disappointing product",
            "I am very unhappy with this purchase",
            "This does not work at all",
            "The product exceeded my expectations",
            "Really good quality",
            "Very useful and easy to use",
            "I would definitely recommend this",
            "Worst product I have ever bought",
            "Completely useless and disappointing",
            "Very poor quality",
            "I regret buying this product",
        ],
        "sentiment": [
            "positive",
            "positive",
            "positive",
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "negative",
            "negative",
            "negative",
            "positive",
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "negative",
        ],
    }
)


print("Dataset:")
print(data)


# ---------------------------------------------------------
# 2. Separate Features and Target
# ---------------------------------------------------------

X = data["text"]
y = data["sentiment"]


# ---------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ---------------------------------------------------------
# 4. Build Text Classification Pipeline
# ---------------------------------------------------------
# TF-IDF converts text into numerical features.
#
# Logistic Regression then learns to classify those
# numerical representations.

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train,
)

print("\nModel trained successfully.")


# ---------------------------------------------------------
# 6. Make Test Predictions
# ---------------------------------------------------------

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test.to_numpy())


# ---------------------------------------------------------
# 7. Evaluate Model
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 8. Inspect TF-IDF Features
# ---------------------------------------------------------

vectorizer = model.named_steps["tfidf"]

feature_names = vectorizer.get_feature_names_out()

print("\nNumber of TF-IDF Features:")
print(len(feature_names))

print("\nFirst 20 TF-IDF Features:")

for feature in feature_names[:20]:
    print(feature)


# ---------------------------------------------------------
# 9. Prediction Probabilities
# ---------------------------------------------------------

probabilities = model.predict_proba(X_test)

classes = model.classes_

print("\nPrediction Probabilities:")

for text, prediction, probability in zip(
    X_test,
    predictions,
    probabilities,
):
    confidence = np.max(probability)

    print(
        f"\nText: {text}"
        f"\nPrediction: {prediction}"
        f"\nConfidence: {confidence:.2%}"
    )

    for class_name, class_probability in zip(
        classes,
        probability,
    ):
        print(
            f"  {class_name}: "
            f"{class_probability:.2%}"
        )


# ---------------------------------------------------------
# 10. Reusable Inference Function
# ---------------------------------------------------------

def predict_sentiment(text):
    """
    Predict sentiment for a single new text.
    """

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = np.max(probabilities)

    return {
        "text": text,
        "sentiment": prediction,
        "confidence": float(confidence),
    }


# ---------------------------------------------------------
# 11. Test New Inputs
# ---------------------------------------------------------

new_texts = [
    "This product is fantastic and very useful",
    "I completely regret buying this",
    "The quality is really good",
    "This is absolutely terrible",
    "I am satisfied with the purchase",
]


print("\nNew Text Predictions:")

for text in new_texts:

    result = predict_sentiment(text)

    print(
        f"\nText: {result['text']}"
        f"\nSentiment: {result['sentiment']}"
        f"\nConfidence: {result['confidence']:.2%}"
    )


# ---------------------------------------------------------
# 12. Batch Inference
# ---------------------------------------------------------

batch_predictions = model.predict(new_texts)

batch_probabilities = model.predict_proba(new_texts)

batch_results = pd.DataFrame(
    {
        "text": new_texts,
        "prediction": batch_predictions,
        "confidence": np.max(
            batch_probabilities,
            axis=1,
        ),
    }
)

print("\nBatch Inference Results:")
print(batch_results)


# ---------------------------------------------------------
# 13. Understanding the Pipeline
# ---------------------------------------------------------

print(
    """
    
AI Text Classification Flow:

Raw Text
    ↓
Lowercase + Tokenization
    ↓
TF-IDF Vectorization
    ↓
Numerical Feature Vectors
    ↓
Logistic Regression
    ↓
Class Prediction
    ↓
Probability / Confidence

"""
)


# ---------------------------------------------------------
# 14. Important AI Engineering Takeaways
# ---------------------------------------------------------

print("AI Engineering Takeaways:")

print(
    "1. ML models cannot directly understand raw text."
)

print(
    "2. TF-IDF converts text into numerical features."
)

print(
    "3. A Pipeline keeps preprocessing and prediction together."
)

print(
    "4. predict() returns the predicted class."
)

print(
    "5. predict_proba() provides class probabilities."
)

print(
    "6. Confidence should be interpreted carefully; "
    "it is not a guarantee of correctness."
)

print(
    "7. The same preprocessing used during training "
    "must be used during inference."
)
