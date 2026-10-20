
"""Predict running conditions using the saved weather model."""

from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Load the saved classifier and predict four weather scenarios."""

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "weather_classifier.pkl"

    classifier = WeatherClassifier(model_path)

    days = [
        {
            "temperature_2m_max": 20,
            "temperature_2m_min": 10,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 35,
            "temperature_2m_min": 20,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 15,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 8,
            "precipitation_sum": 8,
            "wind_speed_10m_max": 10,
        },
        {
            "temperature_2m_max": 25.5,
            "temperature_2m_min": 1,
            "precipitation_sum": 2.5,
            "wind_speed_10m_max": 28,
        },
    ]

    predictions = classifier.predict(days)

    print("Weather Classifier Predictions")
    print("-" * 65)

    for index, (day, prediction) in enumerate(
        zip(days, predictions), start=1
    ):
        print(
            f"Day {index}: "
            f"max={day['temperature_2m_max']}C, "
            f"min={day['temperature_2m_min']}C, "
            f"rain={day['precipitation_sum']}mm, "
            f"wind={day['wind_speed_10m_max']}km/h "
            f"-> {prediction.label.upper()}, "
            f"P(good)={prediction.probability:.2%}"
        )


# The main guard prevents predictions from running automatically
# when another Python script imports this module to reuse a helper.
if __name__ == "__main__":
    main()


# ============================================================
# Task 6: Reflection
# ============================================================

# Q1: Why does predict() select features in a fixed order?
#
# The trained model expects features in the same order used
# during training. If the caller sends dictionary keys in a
# different order and we use that order directly, precipitation
# could be passed as temperature, producing incorrect predictions.
# Selecting the required columns explicitly prevents this bug.


# Q2: Why do tests train a tiny model in a fixture?
#
# A small test model makes the tests independent of the real
# weather_classifier.pkl file. The tests can run on another
# computer or in CI without needing the original model file.
# It also makes the tests faster and more predictable.


# Q3: How is the component ready for Week 10?
#
# WeatherClassifier has a reusable predict() method that accepts
# batches of weather records. It loads the model only once and
# can be imported into a cloud pipeline.
#
# Before production use, I would add stronger input validation,
# including numeric type checks, missing-value checks, and
# validation of reasonable weather measurement ranges.
