"""
Text Classification Pipeline

Practice:
- Text data preparation
- TF-IDF vectorization
- Logistic Regression
- Train/Test Split
- Model evaluation
- Prediction probabilities
- Reusable inference function
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Create a Small Text Dataset
# ---------------------------------------------------------

data = pd.DataFrame(
    {
        "text": [
            "I love this product",
            "This is an amazing experience",
            "The service was excellent",
            "I am very happy with this purchase",
            "This product is fantastic",
            "Absolutely loved it",
            "I hate this product",
            "This was a terrible experience",
            "The service was horrible",
            "I am very disappointed",
            "This product is useless",
            "Absolutely hated it",
            "The product works perfectly",
            "I am satisfied with the service",
            "Very good quality",
            "Worst purchase ever",
            "I would not recommend this",
            "The experience was great",
            "The quality is excellent",
            "I regret buying this",
        ],
        "label": [
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
            "negative",
            "negative",
            "positive",
            "positive",
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
y = data["label"]


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 4. TF-IDF Vectorization
# ---------------------------------------------------------
# TF-IDF converts text into numerical features.
#
# TF  = Term Frequency
# IDF = Inverse Document Frequency

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)


# ---------------------------------------------------------
# 5. Inspect Important Vocabulary
# ---------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

print("\nVocabulary:")
print(feature_names)


# ---------------------------------------------------------
# 6. Train Classification Model
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
)

model.fit(
    X_train_tfidf,
    y_train,
)

print("\nModel training completed.")


# ---------------------------------------------------------
# 7. Make Predictions
# ---------------------------------------------------------

predictions = model.predict(
    X_test_tfidf
)

print("\nPredictions:")
print(predictions)

print("\nActual Labels:")
print(y_test.to_numpy())


# ---------------------------------------------------------
# 8. Evaluate Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

print(f"\nAccuracy: {accuracy:.2f}")


# ---------------------------------------------------------
# 9. Classification Report
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------------

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "positive"],
)

print("\nConfusion Matrix:")
print(matrix)


# ---------------------------------------------------------
# 11. Prediction Probabilities
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test_tfidf
)

classes = model.classes_

print("\nPrediction Probabilities:")

for text, prediction, probability in zip(
    X_test,
    predictions,
    probabilities,
):
    confidence = probability.max()

    print(
        f"\nText: {text}"
        f"\nPrediction: {prediction}"
        f"\nConfidence: {confidence:.2f}"
    )


# ---------------------------------------------------------
# 12. Reusable Inference Function
# ---------------------------------------------------------

def predict_sentiment(text):
    """
    Predict the sentiment of new text.
    """

    transformed_text = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        transformed_text
    )[0]

    probabilities = model.predict_proba(
        transformed_text
    )[0]

    confidence = probabilities.max()

    return prediction, confidence


# ---------------------------------------------------------
# 13. Test New Text
# ---------------------------------------------------------

new_texts = [
    "The product quality is amazing",
    "I am extremely disappointed",
    "This service was very good",
    "I absolutely hate this experience",
]

print("\nNew Text Predictions:")

for text in new_texts:

    prediction, confidence = predict_sentiment(
        text
    )

    print(
        f"\nText: {text}"
        f"\nPrediction: {prediction}"
        f"\nConfidence: {confidence:.2f}"
    )


# ---------------------------------------------------------
# 14. Complete Pipeline Flow
# ---------------------------------------------------------

print(
    """
\nComplete AI Pipeline:

Raw Text
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Probability / Confidence
   ↓
Evaluation
"""
)


# ---------------------------------------------------------
# 15. Important ML Rule
# ---------------------------------------------------------

print(
    "\nImportant Rule:"
)

print(
    "Fit the vectorizer only on training data."
)

print(
    "Use transform() for test and new data."
)

print(
    "Never fit the TF-IDF vectorizer on the test set."
)
