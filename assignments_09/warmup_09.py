
# --- Supabase Connection ---
# Q1
#
# Supabase needs two values to connect:
# the project URL and the API key.
#
# We can find them in the Supabase dashboard
# under the project's API settings.
#
# We should store these values in environment
# variables instead of writing them directly
# in our Python code.
#
# This helps protect credentials and prevents
# accidentally sharing them on GitHub.


# --- Supabase Connection ---
# Q2

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


def get_client() -> Client:
    # Find the .env file in the project root
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL is missing")

    if not key:
        raise ValueError("SUPABASE_KEY is missing")

    return create_client(url, key)


# --- Supabase Connection ---
# Q3
#
# Row Level Security (RLS) controls which rows
# a user can read or modify in a database.
#
# We disabled RLS for this course to simplify
# our exercises with public weather data.
#
# In a real application, such as a banking app,
# RLS should be enabled to protect user data.
# Each user should only access authorized rows.


# --- supabase-py CRUD ---
# Q1

from datetime import date


def insert_test_record(supabase):
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 25.0,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.0,
    }

    response = (
        supabase.table("weather_raw")
        .insert(record)
        .execute()
    )

    print("Inserted record:", response.data)

    return response.data


# Running insert twice with the same date
# will cause a primary key conflict.
#
# To make it safe to run multiple times,
# we can use upsert with on_conflict="date".


# --- supabase-py CRUD ---
# Q2

def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .order("date")
        .execute()
    )

    return response.data


# --- supabase-py CRUD ---
# Q3
#
# INSERT adds a new row to the database.
# If the primary key already exists, it fails.
#
# UPSERT inserts a new row or updates an existing
# row when a conflict occurs.
#
# I would use INSERT to add a new unique record.
# I would use UPSERT in a weather data pipeline
# that may run more than once for the same dates.


def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    count = len(response.data)
    print("Rows upserted:", count)

    return response.data


# --- Idempotency ---
# Q1
#
# Idempotency means that running an operation
# multiple times produces the same final result
# as running it once.
#
# This is important for data pipelines because
# a pipeline may fail and need to restart.
#
# For example, if a pipeline crashes after
# inserting 50 out of 100 records, restarting
# with INSERT may cause primary key conflicts.
#
# Using UPSERT lets us safely process the same
# dates again without creating duplicate rows.


if __name__ == "__main__":
    supabase = get_client()

    test_record = {
        "date": "2026-11-01",
        "temperature_2m_max": 27.0,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.0,
    }

    print("First UPSERT:")
    safe_upsert(supabase, [test_record])

    print("\nSecond UPSERT:")
    safe_upsert(supabase, [test_record])

    records = get_records_by_date_range(
        supabase,
        "2026-11-01",
        "2026-11-01"
    )

    print("\nRecords found:", len(records))
    print("Temperature:", records[0]["temperature_2m_max"])
