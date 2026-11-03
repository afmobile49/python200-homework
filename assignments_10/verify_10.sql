
-- Code the Dream - Python 200
-- Assignment 10: LLMs in Pipelines
--
-- Query 1: Find raw dates without enrichment.
--
-- BEFORE:
-- Missing rows: 366
-- First missing date: 2023-01-01
-- Last missing date: 2026-11-01
--
-- AFTER:
-- Missing rows: 0
-- First missing date: NULL
-- Last missing date: NULL

SELECT
    r.date
FROM public.weather_raw AS r
LEFT JOIN public.weather_enriched AS e
    ON r.date = e.date
WHERE e.date IS NULL
ORDER BY r.date;


-- Query 2: Count good and skip predictions.
--
-- AFTER:
-- good_for_running = true: 221
-- good_for_running = false: 145

SELECT
    good_for_running,
    COUNT(*) AS total_days
FROM public.weather_enriched
GROUP BY good_for_running
ORDER BY good_for_running DESC;


-- Additional validation:
-- Expected: 366 rows, 0 fallback, 0 empty.

SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (
        WHERE llm_summary =
        'Weather recommendation unavailable due to an API error.'
    ) AS fallback_count,
    COUNT(*) FILTER (
        WHERE llm_summary IS NULL
           OR TRIM(llm_summary) = ''
    ) AS empty_count
FROM public.weather_enriched;
