
-- Week 9 - SQL Verification
-- Code the Dream - Python 200

-- Query 1: Row count and date range
SELECT
    COUNT(*) AS total_rows,
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date
FROM public.weather_raw;

-- Result:
-- total_rows | earliest_date | latest_date
-- 366        | 2023-01-01    | 2026-11-01


-- Query 2: Missing weather values
SELECT
    COUNT(*) AS rows_with_null
FROM public.weather_raw
WHERE temperature_2m_max IS NULL
   OR temperature_2m_min IS NULL
   OR precipitation_sum IS NULL
   OR wind_speed_10m_max IS NULL;

-- Result:
-- rows_with_null
-- 0


-- Verification Notes:
-- The SQL results match the Python verification.
-- The pipeline loaded 365 daily weather records
-- for the full year 2023.
--
-- One additional test record was inserted during
-- the warmup exercise on 2026-11-01.
-- This explains the total of 366 rows and
-- the latest date being 2026-11-01.
--
-- Running the pipeline again did not create
-- duplicate dates because UPSERT uses
-- on_conflict="date".
--
-- None of the 366 rows have NULL values
-- in the four required weather columns.
