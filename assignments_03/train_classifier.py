"""Assignment 03, Part 2: train, compare, and save a weather classifier."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Task 1: Fetch one year of weather data ---
# City: Los Angeles, California.
LATITUDE = 34.0522
LONGITUDE = -118.2437
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=364)

response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": FEATURES,
        "timezone": "America/Los_Angeles",
    },
    timeout=30,
)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])

print("City: Los Angeles")
print("Date range:", start_date, "to", end_date)
print("Data shape:", df.shape)
print("Number of rows:", len(df))
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df[FEATURES].isna().sum())

# Avoid training on rows missing one of the four required measurements.
df = df.dropna(subset=FEATURES).copy()

# --- Task 2: Engineer the label ---
# Good=1 only when all four weather conditions are satisfied.
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print("\nFirst 10 labeled days:")
print(df.head(10))
print("\nClass distribution (0=Skip, 1=Good):")
print(df["good_for_running"].value_counts().sort_index())
print("\nClass percentages:")
print(df["good_for_running"].value_counts(normalize=True).sort_index() * 100)

good_count = int(df["good_for_running"].sum())
good_fraction = good_count / len(df)
print(f"Good days: {good_count}/{len(df)} ({good_fraction:.1%})")

# Task 2 reflection: In an earlier 365-day run, 153 days (about 41.9%)
# were labeled Good. This seems plausible for Los Angeles using my rule:
# days that are too hot, rainy, or windy are excluded. The precise fraction
# may change because this script fetches a rolling year of weather data.
# These are labels from my thresholds, not actual runners' preferences.

X = df[FEATURES]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training class distribution:\n", y_train.value_counts())
print("Testing class distribution:\n", y_test.value_counts())

# --- Task 3: Build and evaluate KNN ---
k_values = [1, 3, 5, 7, 9]
cv_results = {}
print("\nKNN cross-validation (F1):")
for k in k_values:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring="f1")
    cv_results[k] = scores.mean()
    print(f"K={k}, Mean CV F1={scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)
print("Best K:", best_k)
print("Best K CV F1:", round(cv_results[best_k], 4))

best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k)),
])
best_knn.fit(X_train, y_train)

# --- Task 4: Build and evaluate Logistic Regression ---
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
logistic_model.fit(X_train, y_train)

results = {}
for name, model in [("KNN", best_knn), ("Logistic Regression", logistic_model)]:
    predictions = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
    }
    print(f"\n{name} test results:")
    print("Accuracy:", round(results[name]["accuracy"], 4))
    print("Precision:", round(results[name]["precision"], 4))
    print("Recall:", round(results[name]["recall"], 4))
    print("F1:", round(results[name]["f1"], 4))
    print("Confusion matrix (labels 0=Skip, 1=Good):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(y_test, predictions))

# Task 4: Show how each standardized weather feature affects log-odds of Good.
coefficients = logistic_model.named_steps["classifier"].coef_[0]
print("\nLogistic Regression coefficients (standardized features):")
for feature, coefficient in zip(FEATURES, coefficients):
    print(f"{feature}: {coefficient:+.4f}")

negative_effects = sorted(
    ((name, float(coef)) for name, coef in zip(FEATURES, coefficients) if coef < 0),
    key=lambda item: item[1],
)
print("Features pushing most toward Skip:", negative_effects[:2])
# Task 4 reflection: Negative coefficients push predictions toward Skip
# when the feature increases and the others stay fixed. Wind and rainfall
# are plausible strong negative influences on whether a day is suitable
# for running. The printed coefficients tell us which features actually
# have the strongest negative effects in this trained model.

# --- Task 5: Compare and select which model to deploy ---
knn_cv_f1 = cv_results[best_k]
logistic_cv_f1 = cross_val_score(
    logistic_model, X_train, y_train, cv=5, scoring="f1"
).mean()

print("\nModel selection (training CV F1):")
print("KNN CV F1:", round(knn_cv_f1, 4))
print("Logistic Regression CV F1:", round(logistic_cv_f1, 4))

if knn_cv_f1 >= logistic_cv_f1:
    selected_model = best_knn
    selected_name = "KNN"
else:
    selected_model = logistic_model
    selected_name = "Logistic Regression"

# Task 5 reflection:
# 1. In my previous test run, KNN and Logistic Regression had the same
#    test accuracy (about 98.63%). Their errors differed, so accuracy
#    alone did not tell the whole story. Current scores are printed above.
# 2. For a running app, I care particularly about precision for Good:
#    a false positive recommends running on a day my rule says to skip.
#    Recall also matters, since false negatives miss good opportunities.
# 3. I choose the model with higher *training cross-validation F1* to
#    avoid choosing it based on test performance. Previously that was
#    KNN (CV F1 about 0.9835 versus 0.9464 for Logistic Regression).
#    KNN is simple and worked well here, but it stores the training
#    examples and distance-based predictions may cost more as data grows.
#    Logistic Regression is smaller, usually fast at prediction, and its
#    coefficients are easier to interpret. Both pipelines provide
#    predict_proba, but their probabilities may need calibration.

# --- Task 6: Save selected fitted pipeline and metadata ---
model_dir = Path(__file__).resolve().parent / "models"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "weather_classifier.pkl"
metadata_path = model_dir / "weather_classifier_metadata.json"

joblib.dump(selected_model, model_path)
selected_predictions = selected_model.predict(X_test)
metadata = {
    "model_name": selected_name,
    "city": "Los Angeles",
    "features": FEATURES,
    "label_rule": (
        "Good=1 when 7 <= temperature_2m_max <= 26, "
        "temperature_2m_min >= 0, precipitation_sum < 3, "
        "wind_speed_10m_max < 30; otherwise Skip=0"
    ),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "test_accuracy": accuracy_score(y_test, selected_predictions),
    "test_f1": f1_score(y_test, selected_predictions),
}
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print("\nSelected model:", selected_name)
print("Model saved:", model_path)
print("Metadata saved:", metadata_path)
print("Model and metadata saved successfully.")
