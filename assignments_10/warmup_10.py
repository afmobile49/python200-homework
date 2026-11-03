
"""
Code the Dream - Python 200
Week 10: LLMs in Pipelines
Assignment: Warmup 10
"""

import time


# ============================================================
# Part 1: ML vs. LLM in Pipelines
# ============================================================

# ML/LLM Question 1
#
# The ML classifier predicts whether a day is good for running
# and returns a probability score.
#
# The LLM uses the weather data and ML prediction to generate
# a natural-language recommendation.
#
# The ML classifier is trained to make consistent predictions
# from structured numerical data.
# The LLM is better suited for generating readable explanations.
#
# If we swapped their roles, the LLM might produce inconsistent
# binary predictions, while the ML classifier could not generate
# meaningful natural-language recommendations.


# ML/LLM Question 2
#
# 1. Convert a date string to a day of the week:
# Deterministic code, because date conversion follows fixed rules.
#
# 2. Classify a job posting by seniority:
# An LLM, because it can interpret freeform job descriptions
# and identify the experience level from the text.
#
# 3. Predict customer churn:
# A trained ML model, because we have numerical features
# and labeled training data.
#
# 4. Normalize inconsistent city names:
# Deterministic code, using a mapping of known aliases
# to standard city names.
#
# 5. Sum a column of revenue:
# Deterministic code, because addition is an exact operation.


# ML/LLM Question 3
#
# Incremental processing means processing only new records
# that have not already been enriched.
#
# It is important because it saves time and API costs.
#
# If all 365 records were processed on every run, the pipeline
# would repeatedly call the LLM for the same dates.
#
# This would increase costs and processing time.
# It could also overwrite previous results with different
# LLM-generated summaries, reducing consistency.


# ============================================================
# Part 2: Prompt Design
# ============================================================

# Prompt Question 1
#
# Alternative system prompt:
#
# You are a running-weather assistant.
# Use only the provided weather data and ML prediction.
# Write exactly two sentences.
# The first sentence must state whether the day is good
# for running according to the ML model.
# The second sentence must explain the prediction using
# the provided weather conditions.
# Do not invent weather information.
# Do not change the ML prediction.
#
# Validation changes:
# The validation logic should require exactly two sentences
# instead of one.
# It should also verify that the first sentence matches
# the ML prediction and that the second sentence explains
# the reasoning using the supplied weather features.
# Invalid responses should be rejected or regenerated.


# Prompt Question 2
#
# This retry function is useful in production when an API
# request fails temporarily because of a network problem,
# timeout, or rate limit.
#
# Retries can make the pipeline more reliable.
# The function makes at most max_retries attempts and waits
# two seconds between failed attempts.
#
# In a production system, I would normally retry only
# transient errors and use exponential backoff.


def call_with_retry(client, messages, max_retries=3):
    """Call the OpenAI Chat Completions API with retries."""

    if max_retries < 1:
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return response

        except Exception as exc:
            print(
                f"API attempt {attempt + 1} failed: "
                f"{type(exc).__name__}"
            )

            if attempt < max_retries - 1:
                time.sleep(2)

    return None


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Week 10 Warmup")
    print("Conceptual answers are included as comments.")
    print("Retry function is defined and ready for testing.")
