
import requests
import pandas as pd
from datetime import date, timedelta

# Los Angeles coordinates
LATITUDE = 34.0522
LONGITUDE = -118.2437

# Historical weather data: previous 365 complete days
end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=364)

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["daily"])

print("Number of rows:", len(df))
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())


# Create classification labels
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print("\nFirst 10 labeled days:")
print(df.head(10))

print("\nClass distribution:")
print(df["good_for_running"].value_counts())

print("\nClass percentages:")
print(df["good_for_running"].value_counts(normalize=True) * 100)


from sklearn.model_selection import train_test_split

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

print("\nKNN Cross-Validation:")

k_values = [1, 3, 5, 7, 9]
cv_results = {}

for k in k_values:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        knn,
        X_train,
        y_train,
        cv=5,
        scoring="f1"
    )

    cv_results[k] = scores.mean()

    print(f"K={k}, Mean CV F1={scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)

print("\nBest K:", best_k)
print("Best CV F1:", round(cv_results[best_k], 4))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# KNN with the best K found through cross-validation
best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k))
])

# Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train both models
best_knn.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)

# Evaluate both models on the same test data
for name, model in [
    ("KNN", best_knn),
    ("Logistic Regression", logistic_model)
]:
    y_pred = model.predict(X_test)

    print(f"\n{name} Test Results:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1-score:", round(f1_score(y_test, y_pred), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


import json
import sys
import joblib
import sklearn
from pathlib import Path

# Select the best model using Training Cross-Validation

knn_cv_f1 = cross_val_score(
    best_knn,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
).mean()

logistic_cv_f1 = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
).mean()

print("\nModel Selection by CV F1:")
print("KNN CV F1:", round(knn_cv_f1, 4))
print("Logistic CV F1:", round(logistic_cv_f1, 4))

if knn_cv_f1 >= logistic_cv_f1:
    selected_model = best_knn
    selected_name = "KNN"
else:
    selected_model = logistic_model
    selected_name = "Logistic Regression"

selected_pred = selected_model.predict(X_test)

# Save the selected model
#model_dir = Path("models")
model_dir = Path(__file__).resolve().parent / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(selected_model, model_dir / "weather_classifier.pkl")

metadata = {
    "model_name": selected_name,
    "city": "Los Angeles",
    "features": FEATURES,
    "label_rule": (
        "Good=1 when 7 <= temperature_2m_max <= 26, "
        "temperature_2m_min >= 0, "
        "precipitation_sum < 3, "
        "wind_speed_10m_max < 30; otherwise Skip=0"
    ),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "test_accuracy": accuracy_score(y_test, selected_pred),
    "test_f1": f1_score(y_test, selected_pred),
}

with open(
    model_dir / "weather_classifier_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print("\nSelected model:", selected_name)
print("Model and metadata saved successfully.")
