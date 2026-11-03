"""Assignment 11 - Cloud ETL Capstone Warmup."""

from prefect import task, get_run_logger


# PART 1: PREFECT ORCHESTRATION

# Prefect Question 1
# A @task is one unit of work inside a workflow.
# A @flow organizes tasks and controls their execution.
# I would not use @task for a simple Celsius-to-Fahrenheit
# conversion because it is a fast calculation without I/O.
# A regular Python function is enough.


# Prefect Question 2
@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass


# Prefect Question 3
# I would open the failed flow run in the Prefect UI.
# Then I would select the failed transform task and read
# its logs and exception traceback.
# I would check the error message, task inputs, and retries.
# The load_enriched task did not run because transform failed.


# PART 2: PRODUCTION PATTERNS

# Production Question 1
# raise_for_status() raises an HTTPError for HTTP error responses.
# With a 500 error, the task fails and Prefect can retry it.
# If all attempts fail, downstream tasks do not run normally.
# Printing an error does not stop execution by itself.
# The pipeline could continue with invalid or missing data.


# Production Question 2
# Upsert inserts new rows or updates existing rows using
# date as the unique key.
# If the pipeline crashes and restarts, upsert prevents
# duplicate dates and primary-key conflicts.
# Plain insert would fail on dates already stored.


# Production Question 3
@task
def log_enrichment_count(enrichment_records: list):
    logger = get_run_logger()
    logger.info(
        "Upserted %d enrichment records.",
        len(enrichment_records),
    )


# Production Question 4
# Incremental processing skips dates already enriched.
# This makes repeated runs faster and reduces LLM API costs.
# Without it, all 365 days would be classified and enriched
# again, even when their results already exist.
# This wastes time and money and may overwrite good summaries.
# Checking completed records helps keep repeated runs consistent.


if __name__ == "__main__":
    print("Assignment 11 warmup is ready.")
