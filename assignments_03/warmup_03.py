
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib


# --- Setup ---

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "weather_classification.csv")

numeric = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# PART 1: PREPROCESSING
# ============================================================

# --- Preprocessing Q1: StandardScaler ---

scaler = StandardScaler()

# Learn scaling parameters from training data only.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same parameters to the test data.
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing Q1:")
print("Scaled training means:")
print(X_train_scaled.mean(axis=0))

# We fit the scaler only on X_train to prevent information
# from the test set from leaking into model training.


# --- Preprocessing Q2: One-Hot Encoding ---

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("\nPreprocessing Q2:")
print("One-Hot Encoding:")
print("Shape:", season_encoded.shape)
print("Feature names:", encoder.get_feature_names_out())

# The single season column becomes four binary columns
# because the dataset contains four unique seasons:
# fall, spring, summer, and winter.


# --- Preprocessing Q3: ColumnTransformer and Pipeline ---

X_all = df[numeric + ["season"]]

X_train_all, X_test_all, y_train_all, y_test_all = (
    train_test_split(
        X_all, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    (
        "season",
        OneHotEncoder(handle_unknown="ignore"),
        ["season"],
    ),
])

preprocessing_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

preprocessing_pipeline.fit(X_train_all, y_train_all)

preprocessing_accuracy = preprocessing_pipeline.score(
    X_test_all, y_test_all
)

print("\nPreprocessing Q3:")
print(
    "Pipeline test accuracy:",
    round(preprocessing_accuracy, 4),
)

# The pipeline automatically fits the numeric scaler
# and season encoder on training data, transforms the
# features, and passes them to LogisticRegression.
# During prediction, it reuses the learned transformations.


# ============================================================
# PART 2: K-NEAREST NEIGHBORS
# ============================================================

# --- KNN Q1: KNN with k=5 ---

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

knn_pipeline.fit(X_train, y_train)

knn_predictions = knn_pipeline.predict(X_test)

print("\nKNN Q1:")
print(
    "Test accuracy:",
    accuracy_score(y_test, knn_predictions),
)
print("Classification Report:")
print(classification_report(y_test, knn_predictions))


# --- KNN Q2: Choose k using cross-validation ---

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

print("\nKNN Q2: Cross-Validation")

for k in k_values:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=k),
        ),
    ])

    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    cv_results[k] = scores.mean()

    print(
        f"K={k}, "
        f"Average CV Accuracy={scores.mean():.4f}"
    )

best_k = max(cv_results, key=cv_results.get)

print("Best K:", best_k)
print(
    "Best CV Accuracy:",
    round(cv_results[best_k], 4),
)

# Based on the current dataset, I chose k=11 because
# it had the highest mean 5-fold CV accuracy (0.9417).
# I used training data for this selection, not test data.


# --- KNN Q3: Compare without scaling and with scaling ---

raw_knn = KNeighborsClassifier(n_neighbors=5)

raw_knn.fit(X_train, y_train)
raw_accuracy = raw_knn.score(X_test, y_test)

scaled_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

scaled_knn.fit(X_train, y_train)
scaled_accuracy = scaled_knn.score(X_test, y_test)

print("\nKNN Q3:")
print(f"Without scaling: {raw_accuracy:.4f}")
print(f"With scaling:    {scaled_accuracy:.4f}")

# Scaling improved test accuracy from 0.9250 to 0.9583.
# KNN measures distances, so features with larger
# numerical scales may dominate the calculation.
# StandardScaler helps make the feature scales comparable.


# ============================================================
# PART 3: LOGISTIC REGRESSION
# ============================================================

# --- Logistic Regression Q1: Train and evaluate ---

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logreg_pipeline.fit(X_train, y_train)

logreg_pred = logreg_pipeline.predict(X_test)

print("\nLogistic Regression Q1:")
print("Accuracy:", accuracy_score(y_test, logreg_pred))
print("Precision:", precision_score(y_test, logreg_pred))
print("Recall:", recall_score(y_test, logreg_pred))
print("F1:", f1_score(y_test, logreg_pred))
print("Classification Report:")
print(classification_report(y_test, logreg_pred))


# --- Logistic Regression Q2: Coefficients ---

coefficients = logreg_pipeline.named_steps[
    "classifier"
].coef_[0]

print("\nLogistic Regression Q2: Coefficients")

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Wind speed (-2.4438) and precipitation (-1.8891)
# had the strongest negative coefficients in my run.
# Increasing these features pushes predictions toward
# Skip, holding the other features constant.
# This matches my intuition because windy or rainy
# weather is usually less suitable for running.
# Coefficients refer to standardized input features.


# --- Logistic Regression Q3: Probabilities ---

probabilities = logreg_pipeline.predict_proba(
    X_test.iloc[:5]
)

good_index = list(logreg_pipeline.classes_).index(1)

print("\nLogistic Regression Q3: Probabilities")

for i, probability in enumerate(probabilities, start=1):
    print(
        f"Day {i}: "
        f"P(Good)={probability[good_index]:.2f}"
    )

# A probability close to 0.5 means the model is near
# its decision boundary and has lower confidence.
# In my run, Day 2 had P(Good)=0.47.


# ============================================================
# PART 4: EVALUATION
# ============================================================

# --- Evaluation Q1: Logistic Regression Confusion Matrix ---

cm = confusion_matrix(
    y_test,
    logreg_pred,
    labels=[0, 1],
)

print("\nEvaluation Q1: Confusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")
plt.savefig(
    OUTPUT_DIR / "logreg_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False positives:", false_positives)
print("False negatives:", false_negatives)

# The model made 9 false positives and 7 false
# negatives in my test results.
# False positives matter more for my running app
# because they recommend running on unsuitable days.
# False negatives mean missing an otherwise good day.


# ============================================================
# PART 5: JOBLIB
# ============================================================

# --- Joblib Q1: Save and load fitted pipeline ---

warmup_model_path = MODEL_DIR / "warmup_model.pkl"

joblib.dump(
    logreg_pipeline,
    warmup_model_path,
)

loaded = joblib.load(warmup_model_path)

assert (
    loaded.predict(X_test)
    == logreg_pipeline.predict(X_test)
).all()

print("\nJoblib Q1: Predictions match.")
print("Saved model:", warmup_model_path)

# Saving only the LogisticRegression estimator would
# leave out the fitted StandardScaler.
# Raw input data would not be transformed to the scale
# used during training, causing unreliable predictions.
