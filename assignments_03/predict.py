
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load the saved model and metadata
model = joblib.load(MODEL_DIR / "weather_classifier.pkl")

with open(
    MODEL_DIR / "weather_classifier_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

# Four hypothetical weather days
days = pd.DataFrame(
    [
        [20, 10, 0, 12],
        [35, 20, 0, 15],
        [18, 8, 8, 10],
        [22, 12, 0, 35],
    ],
    columns=metadata["features"]
)

# Predict with the saved model
predictions = model.predict(days)

# Probability for class Good = 1
good_index = list(model.classes_).index(1)
probabilities = model.predict_proba(days)[:, good_index]

print("Model:", metadata["model_name"])
print("\nPredictions:")

for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities),
    start=1
):
    label = "Good" if prediction == 1 else "Skip"

    print(
        f"Day {i}: {label}, "
        f"Good probability = {probability:.2%}"
    )

# Compare model predictions against the labeling rule
rule_labels = (
    days["temperature_2m_max"].between(7, 26)
    & (days["temperature_2m_min"] >= 0)
    & (days["precipitation_sum"] < 3)
    & (days["wind_speed_10m_max"] < 30)
).astype(int)

print("\nModel vs Rule Comparison:")

for i, (prediction, rule) in enumerate(
    zip(predictions, rule_labels),
    start=1
):
    print(
        f"Day {i}: Model={prediction}, "
        f"Rule={rule}, "
        f"Match={prediction == rule}"
    )
