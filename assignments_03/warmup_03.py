import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
     accuracy_score , precision_score , recall_score , f1_score,
     confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import joblib


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1: StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print(f"\nscaler: {scaler.mean_} , {scaler.scale_}\n")

#for name, mean, std in zip(
#    X_train.columns,
#    scaler.mean_,
#    scaler.scale_
#):
#    print(f"{name}: Mean={mean:.4f}, Std={std:.4f}")


print("Scaled training means:")
print(X_train_scaled.mean(axis=0))

# Fit only on training data to prevent data leakage
# from the test set.

# Q2: One-Hot Encoding

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])


#print(encoder.categories_)
#print(season_encoded[:5])
#print(type(season_encoded))
#print(season_encoded.shape)


print("\nOne-Hot Encoding:")
print("Shape:", season_encoded.shape)
print("Feature names:", encoder.get_feature_names_out())

# One season column becomes four binary columns
# because the dataset contains four unique seasons.

# Q3: ColumnTransformer and Pipeline

# Include numeric features and season

X_all = df[numeric+["season"]]

#print(f"X_all: {X_all}")

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# Apply different preprocessing to different columns

preprocessor = ColumnTransformer([
   ("num", StandardScaler(), numeric),
   ("season", OneHotEncoder(handle_unknown="ignore"),["season"])
])

# Combine preprocessing and classification
pipeline =  Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

#print(f"pipeline : \n {pipeline}")

# Train the full pipeline
pipeline.fit(X_train_all, y_train_all)

# Evaluate on the test set
accuracy = pipeline.score(X_test_all, y_test_all)

print("\nPreprocessing Q3:")
print("Pipeline test accuracy:", round(accuracy, 4))

# The pipeline automatically applies scaling and encoding
# before training or predicting, using transformations
# learned only from the training data.


# --- KNN ---
# Q1: K-Nearest Neighbors with k=5

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)

knn_predictions = knn_pipeline.predict(X_test)

print("\nKNN Q1:")
print("Test accuracy:", accuracy_score(y_test, knn_predictions))
print("\nClassification Report:")
print(classification_report(y_test, knn_predictions))


#from inspection_utils import export_preprocessing_inspection
#export_preprocessing_inspection(
#    pipeline,
#    X_test_all,
#    y_test_all,
#    numeric
#)

k_values = [1, 3, 5, 7, 9]

for k in k_values:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(f"K={k}, Average CV Accuracy={scores.mean():.4f}")


best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=7))
])

best_knn.fit(X_train, y_train)

test_accuracy = best_knn.score(X_test, y_test)

print("K=7 Test Accuracy:", test_accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict using the selected KNN model
y_pred = best_knn.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nKNN K=7 Confusion Matrix:")
print(cm)

# Display and save the matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Skip", "Good"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("KNN K=7 - Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

logistic_pred = pipeline.predict(X_test_all)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_all, logistic_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test_all, logistic_pred))

import joblib
from pathlib import Path

Path("models").mkdir(exist_ok=True)

joblib.dump(best_knn, "models/warmup_model.pkl")

loaded_model = joblib.load("models/warmup_model.pkl")

original_predictions = best_knn.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

print("\nModel saved and loaded successfully.")
print(
    "Predictions identical:",
    (original_predictions == loaded_predictions).all()
)