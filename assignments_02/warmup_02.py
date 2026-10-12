# --- scikit-learn API ---
# Q1: Create -> Fit -> Predict

import numpy as np
from sklearn.linear_model import LinearRegression

# --- Linear Regression ---
# Generate synthetic weather data

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

tmp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8 , 10, 15, 18, 23, 27])

#step 1: Create a regression model
model = LinearRegression()

# Step 2: Train the model
model.fit(tmp_min, temp_max)

# Step 3: Predict high temperatures for lows of 6 and 18 degrees.

new_temps = np.array([6,18]).reshape(-1,1)


predictions = model.predict(new_temps)

# Step 4: Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for low 6°C:", predictions[0])
print("Predicted high for low 18°C:", predictions[1])

# --- scikit-learn API ---
# Q2: Array shapes

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped array:")
print(x_2d)
print("New shape:", x_2d.shape)

# scikit-learn expects a 2D feature array because
# rows represent samples and columns represent features.

rng = np.random.default_rng(42)

num_days = 120

temp_min = rng.uniform(-5, 22 , num_days)

is_rainy = rng.integers(0 , 2, num_days).astype(float)

temp_max = (
    1.0 * temp_min
    + 6
    - 4 * is_rainy
    + rng.normal(0 , 2, num_days)
)


# --- Linear Regression ---
# Q1: Explore synthetic weather data

from pathlib import Path

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

plt.scatter(
    temp_min,
    temp_max,
    c=is_rainy,
    cmap="coolwarm"
)

plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")

plt.savefig(output_dir / "high_vs_low.png")
plt.show()

# Rainy and dry days show different patterns.
# At similar low temperatures, rainy days generally
# have lower high temperatures.
# This suggests that is_rainy may be a useful
# feature for predicting the daily high.

# Q2: Train/test split

x = temp_min.reshape(-1,1)
y = temp_max
X_train , X_test , y_train, y_test = train_test_split( x , y, test_size=0.2, random_state=42 )

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --- Linear Regression ---
# Q3: Fit and evaluate a regression model


model_simple = LinearRegression()

model_simple.fit(X_train , y_train)

print("Slope:", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)

y_pred = model_simple.predict(X_test)

print("Predictions:", y_pred)
print("Actual values:", y_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("RMSE:" , rmse)

r2 = model_simple.score(X_test,y_test)
print("R2",r2)

r2_check = r2_score(y_test, y_pred)
print("R² from r2_score:", r2_check)

# Interpretation:
# The slope estimates how much the predicted daily high
# changes when the daily low increases by 1 degree Celsius,
# using only the daily low as a predictor.

# --- Linear Regression ---
# Q4: Multiple regression with two features

# Combine temperature and rain status into two columns.
X_full = np.column_stack([temp_min, is_rainy])

# Split the dataset.
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    temp_max,
    test_size=0.2,
    random_state=42
)

# Create and train the full model.
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

# Evaluate the model.
r2_full = model_full.score(X_test_full, y_test_full)

print("Simple model R²:", r2)
print("Full model R²:", r2_full)

# Print coefficients.
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Holding minimum temperature constant, rainy days are predicted
# to have daily highs about 2.92°C lower than dry days.


# --- Linear Regression ---
# Q5: Predicted vs Actual Plot

# Predict using the two-feature model
y_pred_full = model_full.predict(X_test_full)

# Create the scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)

# Calculate reference line limits
line_min = min(y_pred_full.min(), y_test_full.min())
line_max = max(y_pred_full.max(), y_test_full.max())

# Perfect prediction reference line
plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    "r--",
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Actual Temperature (°C)")
plt.legend()
plt.grid(True)

# Save to the required outputs directory
plt.savefig(output_dir / "predicted_vs_actual_warmup.png")
plt.show()

# A point above the diagonal indicates underprediction:
# the actual high is greater than the predicted high.
# A point below the diagonal indicates overprediction:
# the predicted high is greater than the actual high.
