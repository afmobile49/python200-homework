
"""Reusable components for weather classification."""

from dataclasses import dataclass


@dataclass
class Prediction:
    """Store a weather prediction and its probability.

    Attributes:
        label: Predicted class, either "good" or "skip".
        probability: Probability that the day is good for running,
            represented as a value between 0.0 and 1.0.
    """

    label: str
    probability: float


from pathlib import Path

import joblib
import pandas as pd


class WeatherClassifier:
    """Load and reuse a trained weather classification pipeline.

    The model is loaded once when the object is created.
    """

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str | Path) -> None:
        """Load a saved model.

        Args:
            model_path: Path to the saved .pkl model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}"
            )

        self.model = joblib.load(path)

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict running conditions for multiple days.

        Args:
            days: List of weather dictionaries containing four
                required numeric features.

        Returns:
            A list of Prediction objects in input order.

        Raises:
            ValueError: If any day has missing required features.
        """
        if not days:
            return []

        for index, day in enumerate(days):
            missing = [
                feature
                for feature in self.FEATURES
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Day {index + 1} missing features: "
                    f"{', '.join(missing)}"
                )

        frame = pd.DataFrame(days)[self.FEATURES]

        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]

        return [
            Prediction(
                label="good" if label == 1 else "skip",
                #label="skip",  # TEMPORARY BUG — testing test effectiveness
                probability=float(probability),
            )
            for label, probability in zip(labels, probabilities)
        ]

        # Deliberate failure verification:
        # Temporarily changed all predicted labels to "skip".
        # test_predict_expected_label and test_predict_three_days_in_order
        # failed and detected the bug.
        # Restored the correct label mapping and all 7 tests passed.
