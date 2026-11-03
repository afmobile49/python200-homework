-- Assignment 11 - Cloud ETL Capstone
-- SQL Verification Report
-- City: Los Angeles
-- Successful Prefect run: bipedal-warthog
-- Date: November 2, 2026 (Los Angeles)

-- Query 1: Raw table summary
SELECT
    COUNT(*) AS total_rows,
    MAX(loaded_at) AS latest_loaded_at
FROM public.weather_raw;

-- Actual result:
-- total_rows: 366
-- latest_loaded_at: 2026-11-01 19:53:21.613037+00


-- Query 2: Enriched table summary
SELECT
    COUNT(*) AS total_rows,
    MAX(enriched_at) AS latest_enriched_at,
    ROUND(
        (
            SELECT COUNT(*)
            FROM public.weather_enriched
            WHERE good_for_running = TRUE
        ) * 100.0 / NULLIF(COUNT(*), 0),
        1
    ) AS good_days_percentage
FROM public.weather_enriched;

-- Actual result:
-- total_rows: 366
-- latest_enriched_at: 2026-11-03 03:08:15.071795+00
-- good_days_percentage: 60.4


-- Timestamp Reflection:
-- These timestamps are not from the latest pipeline run.
-- The latest run upserted 365 raw records, but the loaded_at
-- column kept its original value because DEFAULT now()
-- applies on insertion, not automatically on updates.
-- The transform task found all 365 dates already enriched,
-- so no enriched records were updated during this run.

-- Running Conditions Reflection:
-- About 60.4% of the days were classified as good for running.
-- This seems reasonable for Los Angeles because the city
-- usually has mild weather, although some days can be
-- too hot, rainy, or windy.
-- The model's predictions are estimates, not safety guarantees.

-- Data note:
-- The tables contain 365 dates from 2023 plus one
-- additional test record dated 2026-11-01.
