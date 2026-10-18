"""Assignment 03, Part 2: load a trained model and predict new weather days."""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
model_path = MODEL_DIR / "weather_classifier.pkl"
metadata_path = MODEL_DIR / "weather_classifier_metadata.json"

# --- Task 7: Load model and metadata (no fitting or API calls) ---
if not model_path.is_file() or not metadata_path.is_file():
    raise SystemExit(
        "Missing trained model or metadata. Run 'python train_classifier.py' "
        "from the assignments_03 directory first."
    )

model = joblib.load(model_path)
with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Model:", metadata["model_name"])
print("City:", metadata["city"])
print("Features (in order):", ", ".join(metadata["features"]))
print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
print(f"Test F1: {metadata['test_f1']:.4f}")

# Units: temperatures in C, precipitation in mm, wind speed in km/h.
# Day 5 is near several limits in the rule (borderline).
days = pd.DataFrame(
    [
        [20.0, 10.0, 0.0, 12.0],  # Clearly Good
        [35.0, 20.0, 0.0, 15.0],  # Too hot
        [18.0, 8.0, 8.0, 10.0],   # Too rainy
        [22.0, 12.0, 0.0, 35.0],  # Too windy by the rule
        [25.5, 1.0, 2.5, 28.0],   # Borderline: close to four limits
    ],
    columns=metadata["features"],
)

predictions = model.predict(days)
good_index = list(model.classes_).index(1)
probabilities = model.predict_proba(days)[:, good_index]

print("\nPredictions (four weather inputs and P(Good)):")
for i, (row, prediction, probability) in enumerate(
    zip(days.to_dict(orient="records"), predictions, probabilities), start=1
):
    label = "Good" if prediction == 1 else "Skip"
    print(f"Day {i}: inputs={row}, predicted={label}, P(Good)={probability:.2%}")

# Optional comparison: predictions can disagree with the simple labeling rule.
rule_labels = (
    days["temperature_2m_max"].between(7, 26)
    & (days["temperature_2m_min"] >= 0)
    & (days["precipitation_sum"] < 3)
    & (days["wind_speed_10m_max"] < 30)
).astype(int)
print("\nModel vs. labeling rule:")
for i, (prediction, rule) in enumerate(zip(predictions, rule_labels), start=1):
    print(f"Day {i}: model={prediction}, rule={rule}, match={prediction == rule}")

print(f"\nBorderline Day 5: P(Good)={probabilities[4]:.2%}")

# --- Task 8: Reflection ---
# 1. My borderline day is Day 5: maximum temperature 25.5 C, minimum 1 C,
#    precipitation 2.5 mm, and wind speed 28 km/h. The program prints its
#    actual P(Good) just above; I should use that value rather than guess it,
#    because it depends on the trained model. A vote fraction from KNN, even
#    when high, is not guaranteed to be a well-calibrated probability.
#    If the model gave P(Good)=0.52, I would treat it as uncertain and
#    check the real weather and threshold rule rather than confidently
#    recommend running.
# 2. If this script runs before train_classifier.py, the .pkl and metadata
#    files may not exist, so loading would fail. The file check above gives
#    a helpful instruction to run training first.
# 3. For Week 10, I would replace these hard-coded example days with current
#    forecast data fetched automatically inside a cloud workflow, while
#    keeping the same four input names/order and handling failed API calls.
