
"""
Code the Dream - Python 200
Week 10: LLMs in Pipelines

Video link:
https://drive.google.com/file/d/1YiAv9Pmco9rrQwWhMuGA8HydNcJfNo--/view?usp=sharing
"""


import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task
from supabase import create_client

from weather_model import WeatherClassifier


ASSIGNMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = ASSIGNMENT_DIR.parent
MODEL_PATH = ASSIGNMENT_DIR / "models" / "weather_classifier.pkl"

FEATURES = WeatherClassifier.FEATURES
PAGE_SIZE = 500
BATCH_SIZE = 50
FALLBACK = "Weather recommendation unavailable due to an API error."


SYSTEM_PROMPT = """
You are a running-weather assistant.
Write exactly one short English sentence.
Use only the provided weather features and ML prediction.
Follow the ML prediction without changing its label.
Mention significant precipitation, strong wind, or extreme
temperatures when present, even if the ML prediction is good.
Do not invent facts or claim that conditions are perfectly safe.
Do not add a second sentence.
""".strip()



def get_supabase():
    load_dotenv(ROOT_DIR / ".env")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_BACKEND_KEY")

    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_BACKEND_KEY"
        )

    return create_client(url, key)


def fetch_all(client, table, columns="*"):
    """Read every page, avoiding API row limits."""
    records = []
    start = 0

    while True:
        response = (
            client.table(table)
            .select(columns)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        page = response.data or []
        records.extend(page)

        if len(page) < PAGE_SIZE:
            break

        start += PAGE_SIZE

    return records



@task
def read_weather(limit=None):
    """Find new dates and records needing LLM recovery."""
    client = get_supabase()

    raw = fetch_all(client, "weather_raw")
    enriched = fetch_all(
        client,
        "weather_enriched",
        "date,llm_summary",
    )

    fallback = (
        "Weather recommendation unavailable due to an API error."
    )

    completed_dates = {
        row["date"]
        for row in enriched
        if row.get("llm_summary")
        and row["llm_summary"].strip()
        and row["llm_summary"] != fallback
    }

    pending = [
        row for row in raw
        if row["date"] not in completed_dates
    ]

    pending.sort(key=lambda row: row["date"])

    print(f"Raw records: {len(raw)}")
    print(f"Already enriched: {len(enriched)}")
    print(f"Completed recommendations: {len(completed_dates)}")
    print(f"Pending or retry records: {len(pending)}")

    if limit is not None:
        pending = pending[:limit]
        print(f"Selected for this run: {len(pending)}")

    return pending


@task
def classify_weather(days):
    """Run the saved Week 4 ML classifier."""
    if not days:
        print("No new records to classify.")
        return []

    classifier = WeatherClassifier(MODEL_PATH)

    for day in days:
        for feature in FEATURES:
            if day.get(feature) is None:
                raise ValueError(
                    f"Missing value: {day['date']} / {feature}"
                )

    predictions = classifier.predict(days)

    records = []
    for day, prediction in zip(days, predictions):
        probability = float(prediction.probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Invalid model probability")

        records.append({
            "date": day["date"],
            "good_for_running": prediction.label == "good",
            # Week 10 requires Prediction.probability.
            # This is P(good), not necessarily P(predicted label).
            "confidence": probability,
            "_weather": {
                feature: float(day[feature])
                for feature in FEATURES
            },
        })

    good = sum(r["good_for_running"] for r in records)
    probabilities = [r["confidence"] for r in records]

    print(f"Good days: {good}")
    print(f"Skip days: {len(records) - good}")
    print(
        "Probability range: "
        f"{min(probabilities):.4f} - {max(probabilities):.4f}"
    )

    return records


def build_user_message(record):
    weather = record["_weather"]
    label = "good" if record["good_for_running"] else "skip"

    return (
        f"Date: {record['date']}\n"
        f"Maximum temperature: {weather['temperature_2m_max']} C\n"
        f"Minimum temperature: {weather['temperature_2m_min']} C\n"
        f"Precipitation: {weather['precipitation_sum']} mm\n"
        f"Maximum wind speed: {weather['wind_speed_10m_max']} km/h\n"
        f"ML prediction: {label}\n"
        f"ML probability of good: {record['confidence']:.4f}"
    )



def valid_summary(text):
    """Validate a short, single-sentence recommendation."""
    import re

    if not text:
        return False

    value = text.strip()

    if not 10 <= len(value) <= 350:
        return False

    if not value.endswith((".", "!", "?")):
        return False

    # Count sentence-ending punctuation, not decimal points.
    endings = re.findall(r"[.!?](?=\s|$)", value)

    return len(endings) == 1



@task
def enrich_with_llm(records, execute=False):
    """Generate one-sentence recommendations."""
    if not execute:
        print("DRY RUN: OpenAI API calls skipped.")
        for record in records[:3]:
            print(build_user_message(record))
        return records

    if not records:
        return records

    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY")

    client = OpenAI(max_retries=0)

    for index, record in enumerate(records, start=1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_user_message(record),
                    },
                ],
            )

            summary = (
                response.choices[0].message.content or ""
            ).strip()

            record["llm_summary"] = (
                summary if valid_summary(summary) else FALLBACK
            )

        except Exception as exc:
            print(
                f"LLM error on {record['date']}: "
                f"{type(exc).__name__}"
            )
            record["llm_summary"] = FALLBACK

        if index % 50 == 0:
            print(f"LLM progress: {index}/{len(records)}")

    print(f"LLM records processed: {len(records)}")
    return records


@task
def load_enriched(records, execute=False):
    """Upsert records using date as the primary key."""
    if not execute:
        print("DRY RUN: Database writes skipped.")
        return 0

    if not records:
        print("Nothing to upsert.")
        return 0

    client = get_supabase()
    total = 0

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        payload = [
            {
                "date": row["date"],
                "good_for_running": row["good_for_running"],
                "confidence": row["confidence"],
                "llm_summary": row["llm_summary"],
            }
            for row in batch
        ]

        client.table("weather_enriched").upsert(
            payload,
            on_conflict="date",
        ).execute()

        total += len(payload)

    print(f"Rows upserted: {total}")
    return total


def verify_database():
    """Verify saved rows independently of Prefect tasks."""
    client = get_supabase()
    rows = fetch_all(client, "weather_enriched")

    print(f"Total enriched rows: {len(rows)}")

    good = sum(row["good_for_running"] is True for row in rows)
    print(f"Total good days: {good}")

    for row in sorted(rows, key=lambda r: r["date"])[:5]:
        print({
            key: row.get(key)
            for key in (
                "date",
                "good_for_running",
                "confidence",
                "llm_summary",
            )
        })


@flow(log_prints=True)
def weather_transform_flow(execute=False, limit=None):
    """Run the four-stage Week 10 pipeline."""
    days = read_weather(limit)
    classified = classify_weather(days)
    enriched = enrich_with_llm(classified, execute)
    loaded = load_enriched(enriched, execute)

    print(f"Flow completed. Rows written: {loaded}")

    if execute:
        verify_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable real LLM calls and database writes.",
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")

    # Default: safe dry run with only three records.
    selected_limit = args.limit
    if not args.execute and selected_limit is None:
        selected_limit = 3

    weather_transform_flow(
        execute=args.execute,
        limit=selected_limit,
    )
