

# Week 9 - Extract + Load Pipeline
# Video: https://drive.google.com/file/d/1qAV8lmDpIKQG6Zdip4XMMsarMOgM7AHq/view?usp=sharing


import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": 34.0522,
    "longitude": -118.2437,
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


def extract_weather():
    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    daily = data["daily"]

    print("City: Los Angeles")
    print("Timezone:", data.get("timezone"))
    print("Number of days:", len(daily["time"]))
    print("First date:", daily["time"][0])
    print("Last date:", daily["time"][-1])

    return data


def transform_weather(data):
    daily = data["daily"]

    dates = daily["time"]
    max_temps = daily["temperature_2m_max"]
    min_temps = daily["temperature_2m_min"]
    precipitation = daily["precipitation_sum"]
    wind_speeds = daily["wind_speed_10m_max"]

    lengths = [
        len(dates),
        len(max_temps),
        len(min_temps),
        len(precipitation),
        len(wind_speeds),
    ]

    if len(set(lengths)) != 1:
        raise ValueError("Weather arrays have different lengths")

    records = []

    for i in range(len(dates)):
        record = {
            "date": dates[i],
            "temperature_2m_max": max_temps[i],
            "temperature_2m_min": min_temps[i],
            "precipitation_sum": precipitation[i],
            "wind_speed_10m_max": wind_speeds[i],
        }

        records.append(record)

    print("Transformed records:", len(records))

    if records:
        print("First record:", records[0])
        print("Last record:", records[-1])

    return records



from warmup_09 import get_client


def load_weather(records):
    if not records:
        raise ValueError("No records to load")

    supabase = get_client()

    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print("Rows upserted:", len(response.data))

    return supabase


def verify_weather(supabase):
    table = supabase.table("weather_raw")

    count_result = (
        table.select("date", count="exact", head=True)
        .execute()
    )

    earliest = (
        table.select("date")
        .order("date")
        .limit(1)
        .execute()
    )

    latest = (
        table.select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    july_4 = (
        table.select("*")
        .eq("date", "2023-07-04")
        .execute()
    )

    year_count = (
        table.select("date", count="exact", head=True)
        .gte("date", "2023-01-01")
        .lte("date", "2023-12-31")
        .execute()
    )

    print("\n--- Database Verification ---")
    print("Total rows:", count_result.count)
    print("Rows in 2023:", year_count.count)
    print("Earliest date:", earliest.data[0]["date"])
    print("Latest date:", latest.data[0]["date"])
    print("July 4 record:", july_4.data)


if __name__ == "__main__":
    data = extract_weather()

    records = transform_weather(data)

    print("Expected records: 365")
    print("Actual records:", len(records))

    assert len(records) == 365, "Unexpected record count"

    supabase = load_weather(records)

    verify_weather(supabase)
