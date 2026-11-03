# Assignment 11 Video: https://drive.google.com/file/d/12t8g0-NhRUDO1Gfc_DNo8if59iXX0j_5/view?usp=sharing
"""Code the Dream - Python 200, Week 11: Cloud ETL Capstone.

Video link: https://drive.google.com/file/d/12t8g0-NhRUDO1Gfc_DNo8if59iXX0j_5/view?usp=sharing

Safe default: fetches public Open-Meteo data only, does not write to
Supabase or call OpenAI. Use --execute after verifying city and settings.
"""

import argparse
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, get_run_logger, task
from supabase import create_client

from weather_model import WeatherClassifier

ASSIGNMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = ASSIGNMENT_DIR.parent
MODEL_PATH = ASSIGNMENT_DIR / "models" / "weather_classifier.pkl"
FEATURES = WeatherClassifier.FEATURES
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FALLBACK = "Weather recommendation unavailable due to an API error."
PAGE_SIZE = 500
BATCH_SIZE = 50

SYSTEM_PROMPT = """You are a running-weather assistant.
Write exactly one short English sentence.
Use only the weather features and ML prediction provided.
Follow the ML prediction without changing its label.
Mention important rain, wind or extreme temperatures when present.
Do not invent facts or claim that conditions are perfectly safe.
Do not add a second sentence."""


def get_supabase():
    load_dotenv(ROOT_DIR / ".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_BACKEND_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_BACKEND_KEY in .env")
    return create_client(url, key)


def fetch_all(client, table, columns="*"):
    """Read a table using stable pagination."""
    rows = []
    start = 0
    while True:
        page = (
            client.table(table)
            .select(columns)
            .order("date")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data or []
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def valid_summary(text):
    if not text:
        return False
    value = text.strip()
    return (
        10 <= len(value) <= 350
        and value.endswith((".", "!", "?"))
        and len(re.findall(r"[.!?](?=\s|$)", value)) == 1
    )


def build_user_message(row, probability):
    label = "good" if row["good_for_running"] else "skip"
    weather = row["_weather"]
    return (
        f"Date: {row['date']}\n"
        f"Maximum temperature: {weather['temperature_2m_max']} C\n"
        f"Minimum temperature: {weather['temperature_2m_min']} C\n"
        f"Precipitation: {weather['precipitation_sum']} mm\n"
        f"Maximum wind speed: {weather['wind_speed_10m_max']} km/h\n"
        f"ML prediction: {label}\n"
        f"ML probability of good: {probability:.4f}"
    )


@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float, timezone: str = "America/Los_Angeles"):
    """Extract 2023 daily weather and turn Open-Meteo column arrays into rows."""
    logger = get_run_logger()
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": ",".join(FEATURES),
            "timezone": timezone,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily") or {}
    dates = daily.get("time")
    if not isinstance(dates, list) or len(dates) != 365:
        raise ValueError("Expected 365 days of daily weather data for 2023")
    for feature in FEATURES:
        if not isinstance(daily.get(feature), list) or len(daily[feature]) != len(dates):
            raise ValueError(f"Missing or mismatched daily series: {feature}")
    records = []
    for index, date in enumerate(dates):
        row = {"date": date}
        for feature in FEATURES:
            value = daily[feature][index]
            if value is None:
                raise ValueError(f"Missing {feature} for {date}")
            row[feature] = float(value)
        records.append(row)
    if len({r["date"] for r in records}) != len(records):
        raise ValueError("Duplicate dates returned from archive API")
    logger.info("Extracted %d weather records", len(records))
    print(f"Extracted {len(records)} weather records.")
    return records


@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list[dict], execute: bool = False):
    """Upsert extracted weather data with date as the unique key."""
    if not execute:
        print(f"DRY RUN: would upsert {len(records)} raw rows; no database write.")
        return 0
    if not records:
        print("No raw records to upsert.")
        return 0
    client = get_supabase()
    for start in range(0, len(records), BATCH_SIZE):
        client.table("weather_raw").upsert(
            records[start:start + BATCH_SIZE], on_conflict="date"
        ).execute()
    get_run_logger().info("Upserted %d raw records", len(records))
    print(f"Upserted {len(records)} raw records.")
    return len(records)


@task
def transform(records: list[dict], execute: bool = False):
    """Incrementally classify and create one-sentence LLM recommendations."""
    if not records:
        print("No weather records to transform.")
        return []
    logger = get_run_logger()
    if execute:
        saved = fetch_all(get_supabase(), "weather_enriched", "date,llm_summary")
        completed = {
            row["date"] for row in saved
            if row.get("llm_summary")
            and row["llm_summary"].strip()
            and row["llm_summary"].strip() != FALLBACK
        }
    else:
        # Dry run reads no private database. It intentionally previews 3 rows.
        completed = set()
    pending = [r for r in records if r["date"] not in completed]
    logger.info("Already complete: %d; pending: %d", len(records) - len(pending), len(pending))
    print(f"Already complete: {len(records) - len(pending)}; pending: {len(pending)}")
    if not pending:
        print("All source dates are already enriched.")
        return []

    for row in pending:
        for feature in FEATURES:
            if row.get(feature) is None:
                raise ValueError(f"Missing {feature} on {row['date']}")
    classifier = WeatherClassifier(MODEL_PATH)
    predictions = classifier.predict(pending)
    if len(predictions) != len(pending):
        raise ValueError("Classifier returned unexpected number of predictions")
    if execute:
        load_dotenv(ROOT_DIR / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY in .env")
        llm_client = OpenAI(max_retries=0, timeout=45)
    else:
        llm_client = None

    enriched = []
    for index, (raw, pred) in enumerate(zip(pending, predictions), start=1):
        probability = float(pred.probability)
        if pred.label not in ("good", "skip") or not (0 <= probability <= 1):
            raise ValueError(f"Invalid classifier output for {raw['date']}")
        record = {
            "date": raw["date"],
            "good_for_running": pred.label == "good",
            "confidence": probability,  # P(good), not P(predicted label)
            "_weather": {key: raw[key] for key in FEATURES},
        }
        if not execute:
            record["llm_summary"] = "DRY RUN - OpenAI was not called."
        else:
            try:
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(record, probability)},
                    ],
                )
                summary = (response.choices[0].message.content or "").strip()
                record["llm_summary"] = summary if valid_summary(summary) else FALLBACK
                if record["llm_summary"] == FALLBACK:
                    logger.warning("LLM response invalid for %s", raw["date"])
            except Exception as exc:
                logger.warning("LLM error on %s: %s", raw["date"], type(exc).__name__)
                record["llm_summary"] = FALLBACK
        enriched.append(record)
        if index % 50 == 0:
            print(f"Transform progress: {index}/{len(pending)}")
    print(f"Transformed {len(enriched)} records.")
    return enriched


@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list[dict], execute: bool = False):
    """Upsert enriched rows; omit internal _weather field."""
    if not enrichment_records:
        print("Nothing to load into weather_enriched.")
        return 0
    if not execute:
        print(f"DRY RUN: would upsert {len(enrichment_records)} enriched rows.")
        return 0
    client = get_supabase()
    total = 0
    for start in range(0, len(enrichment_records), BATCH_SIZE):
        batch = enrichment_records[start:start + BATCH_SIZE]
        payload = [
            {key: row[key] for key in ("date", "good_for_running", "confidence", "llm_summary")}
            for row in batch
        ]
        client.table("weather_enriched").upsert(payload, on_conflict="date").execute()
        total += len(payload)
    get_run_logger().info("Upserted %d enrichment records", total)
    print(f"Upserted {total} enriched records.")
    return total


@flow(log_prints=True)
def weather_etl_flow(latitude: float, longitude: float, timezone: str, execute: bool = False):
    """Run the required four tasks in sequence."""
    raw = extract(latitude, longitude, timezone)
    raw_count = load_raw(raw, execute)
    enriched = transform(raw, execute)
    enriched_count = load_enriched(enriched, execute)
    print(f"Flow completed. Raw upserted: {raw_count}; enriched upserted: {enriched_count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 11 full Prefect ETL flow")
    parser.add_argument("--latitude", type=float, default=34.0522,
                        help="Default Los Angeles; must match Week 9 city before --execute")
    parser.add_argument("--longitude", type=float, default=-118.2437)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--execute", action="store_true",
                        help="Enable real Supabase upserts and OpenAI calls")
    args = parser.parse_args()
    if args.execute:
        print("EXECUTE: real database writes and possible paid OpenAI requests enabled.")
    else:
        print("DRY RUN: no database writes or OpenAI calls.")
    weather_etl_flow(args.latitude, args.longitude, args.timezone, args.execute)


