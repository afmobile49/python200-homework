
# --- Mini-Project: Predicting the Daily High ---
# Task 1: Fetch real weather data

import requests
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# City: Los Angeles, California
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 34.05,
    "longitude": -118.24,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
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

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)
print(df.head())

# --- Task 2: Look at the Features ---

print("\nDataset statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nCorrelation with daily high:")

correlations = df.select_dtypes(include="number").corr()[
    "temperature_2m_max"
].sort_values(ascending=False)

print(correlations)


# Create output folder
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
    alpha=0.6
)

plt.title("Los Angeles Daily High vs Low - 2023")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")
plt.grid(True)

plt.savefig(output_dir / "la_high_vs_low_2023.png")
plt.show()

# The scatter plot shows a positive relationship between
# daily minimum and maximum temperatures.
# Higher minimum temperatures generally correspond to
# higher maximum temperatures.
# However, there is some variation, suggesting that
# other weather features may improve the predictions.

# --- Task 3: Engineer a Feature ---

# Create a binary summer indicator.
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("\nSummer vs non-summer days:")
print(df["is_summer"].value_counts())

# I expect is_summer to help predict daily high temperatures
# because summer days are generally warmer in Los Angeles.
# This feature may capture seasonal temperature patterns
# that are not fully explained by the minimum temperature.

# --- Task 4: Baseline Model ---


X = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

print("Baseline slope:", baseline_model.coef_[0])

y_pred_baseline = baseline_model.predict(X_test)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

baseline_r2 = baseline_model.score(X_test, y_test)

print("Baseline RMSE:", baseline_rmse)
print("Baseline R²:", baseline_r2)

# The baseline RMSE is about 4.13°C on the test set.
# This describes the scale of prediction errors, with larger
# errors receiving more weight in the RMSE calculation.

# --- Task 5: Full Model ---

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer"
]

X_full = df[feature_cols].values
y_full = df["temperature_2m_max"].values

print("Full X shape:", X_full.shape)
print("Full y shape:", y_full.shape)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42
)


# Create and train the full model
full_model = LinearRegression()
full_model.fit(X_train_full, y_train_full)

# Predict on the test set
y_pred_full = full_model.predict(X_test_full)

# Evaluate the full model
train_r2 = full_model.score(X_train_full, y_train_full)
test_r2 = full_model.score(X_test_full, y_test_full)

full_rmse = np.sqrt(
    mean_squared_error(y_test_full, y_pred_full)
)

print("\nFull Model Results:")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Test RMSE:", full_rmse)

# Compare against the baseline
print("\nModel Comparison:")
print("Baseline Test R²:", baseline_r2)
print("Full Model Test R²:", test_r2)
print("R² improvement:", test_r2 - baseline_r2)

print("\nFeature Coefficients:")

for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Minimum temperature (+0.992) and summer status (+1.292)
# increase predicted highs, holding other features fixed.
# Rainfall (-0.175) and wind speed (-0.010) decrease them.
# A surprising sign is the slightly negative wind coefficient,
# although wind has a slightly positive univariate correlation.
# Raw coefficients cannot be ranked fairly across different units.
# The min-temperature contribution over its observed 21°C range
# is about 20.8°C, so it has a large practical effect here.
# Train R² (0.6942) is higher than test R² (0.5934), showing
# some generalization gap on this split.


# --- Task 6: Evaluate and Summarize ---

# Create a predicted vs actual plot.
plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred_full,
    y_test_full,
    alpha=0.7
)

# Calculate the limits for the reference line.
line_min = min(y_pred_full.min(), y_test_full.min())
line_max = max(y_pred_full.max(), y_test_full.max())

# Draw the perfect-prediction line.
plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    "r--",
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Actual Temperature (°C)")

plt.legend()
plt.grid(True)

plt.savefig(output_dir / "predicted_vs_actual_high.png")
plt.show()

# The plot shows errors across the temperature range.
# Some of the largest visible errors occur at the low end,
# but error is not restricted to one temperature range.

# --- Task 6: Final Summary ---
# The dataset contains 365 Los Angeles daily observations from 2023,
# with 73 days held out for testing. No missing values were found.
# The full model achieved RMSE = 3.78°C and test R² = 0.5934.
# An RMSE of 3.78°C gives the approximate scale of test errors;
# it gives extra weight to larger errors and is not the MAE.
# Minimum temperature is an important predictor: increasing it
# by 1°C changes the fitted high by about +0.992°C when the
# other predictors are held constant. Its observed range gives
# a larger fitted temperature change than the summer indicator.
# A surprising result was the negative adjusted coefficient for
# wind speed despite a slightly positive simple correlation.
