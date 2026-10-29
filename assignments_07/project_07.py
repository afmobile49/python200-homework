"""Assignment 07, Part 2: World Happiness Agent -- Task 1 (local tools).

Place in assignments_07/project_07.py. API calls are controlled by run flags.
"""
from pathlib import Path
import os
import json

# Prevent Tkinter GUI/thread warnings when agents generate Matplotlib charts.
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "happiness_project"
df = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load and combine annual World Happiness CSV files into memory.

    Returns:
        Dictionary with dataset shape and column names, or an error.
    """
    global df
    paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    if not paths:
        return {"error": f"No yearly CSV files found in {DATA_DIR}"}

    frames = []
    for path in paths:
        try:
            year_df = pd.read_csv(path, sep=";", decimal=",")
            year_df = year_df.rename(columns=COLUMN_NAMES)
            year_df["year"] = int(path.stem.split("_")[-1])
            frames.append(year_df)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            return {"error": f"Could not read {path.name}: {exc}"}

    combined = pd.concat(frames, ignore_index=True)
    required = {"country", "region", "happiness_score", "gdp_per_capita", "year"}
    missing = sorted(required - set(combined.columns))
    if missing:
        return {"error": f"Required columns missing after merge: {missing}"}
    df = combined
    return {"shape": list(df.shape), "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return summary statistics for a column of the loaded dataset.

    Args:
        column: Name of the column to summarize.

    Returns:
        Descriptive statistics dictionary or an error.
    """
    if df is None:
        return {"error": "No happiness data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Unknown column: {column}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and p-value between two numeric columns.

    Args:
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        Dictionary containing column names, pearson_r and p_value, or an error.
    """
    if df is None:
        return {"error": "No happiness data loaded. Call load_happiness_data first."}
    missing = [name for name in (col1, col2) if name not in df.columns]
    if missing:
        return {"error": f"Unknown columns: {missing}"}
    try:
        pairs = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pairs) < 2 or pairs[col1].nunique() < 2 or pairs[col2].nunique() < 2:
            return {"error": "At least two nonconstant numeric observation pairs are required."}
        r, p = pearsonr(pairs[col1], pairs[col2])
        return {
            "col1": col1, "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top countries for a metric in a selected year.

    Args:
        column: Numeric column used for descending ranking.
        year: Year of the annual happiness report.
        n: Number of countries to return, default 5.

    Returns:
        A dictionary with the top ranked country records or an error.
    """
    if df is None:
        return {"error": "No happiness data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Unknown column: {column}"}
    if n < 1:
        return {"error": "n must be at least 1."}
    rows = df[df["year"] == year].copy()
    if rows.empty:
        return {"error": f"No data found for year {year}."}
    rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=[column])
    if rows.empty:
        return {"error": f"No numeric values for {column} in {year}."}
    records = rows.sort_values(column, ascending=False).head(n)[["country", column]]
    return {"countries": records.to_dict(orient="records")}


# Task 2 — build the CodeAgent (do not call the API until enabled).
TOOLS = [
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries,
]

RUN_AGENT_TEST = False  # Task 2 has already been tested successfully.
RUN_GUIDED_QUERIES = False  # Already completed; avoid repeat API calls.
RUN_CUSTOM_QUERIES = False  # Set True once for the two new Task 4 queries.

AGENT_INSTRUCTIONS = (
    "You are an assistant analyzing World Happiness CSV datasets in happiness_project/. "
    "Use the provided tools for loading, summary statistics, correlation, and country rankings. "
    "Do not invent data. If you generate charts, label axes and save them into outputs/. "
    "Be transparent about data coverage and statistical limitations. "
    "The load_happiness_data tool returns a dict with keys shape and columns, "
    "not a DataFrame. For custom analysis and plots the loaded DataFrame is "
    "available as happiness_df in the Python executor. "
    "When saving plots, use matplotlib's savefig and close, not plt.show(), "
    "because GUI windows may fail in the agent executor. "
    "Save chart outputs into outputs/."
)


def create_happiness_agent():
    """Create but do not run a CodeAgent; constructing it does not call the API."""
    load_dotenv(dotenv_path=BASE_DIR.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in root .env or environment.")

    model = OpenAIServerModel(model_id="gpt-4o-mini", api_key=api_key)
    return CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=AGENT_INSTRUCTIONS,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )


GUIDED_QUERIES = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


def run_guided_queries():
    """Run all five official guided queries with conversation context preserved."""
    global df
    chart_dir = BASE_DIR / "outputs"
    chart_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        loaded = load_happiness_data()
        if "error" in loaded:
            raise RuntimeError(str(loaded))

    agent = create_happiness_agent()
    previous_dir = Path.cwd()
    try:
        # CodeAgent's generated relative file paths now target assignments_07/.
        os.chdir(BASE_DIR)
        for i, query in enumerate(GUIDED_QUERIES, 1):
            print(f"\\n--- Task 3 / Query {i}: {query} ---", flush=True)
            try:
                response = agent.run(
                    query,
                    reset=False,
                    additional_args={"happiness_df": df},
                )
                print(f"Query {i} response: {response}", flush=True)
            except Exception as exc:
                print(f"Query {i} failed: {type(exc).__name__}: {exc}", flush=True)
                print("Stopped here to avoid unnecessary follow-up API calls.")
                break
        chart_path = chart_dir / "happiness_by_region.png"
        print(f"\\nChart saved: {chart_path.exists()} | {chart_path}")
    finally:
        os.chdir(previous_dir)


# --- Task 4: Two independently chosen questions ---
CUSTOM_QUERIES = [
    # Compare how the highest-ranked countries changed since the first year.
    "Load the World Happiness data and identify the five countries with the "
    "highest happiness_score in 2015. Include their scores and rank them.",
    # Explore another hypothesized relationship using the existing tool.
    "Load the World Happiness data and compute the Pearson correlation "
    "between Healthy life expectancy and happiness_score. Explain the "
    "direction and approximate strength, and interpret the p-value carefully.",
]


def run_custom_queries():
    """Run only Task 4 with fresh agent state for each question."""
    agent = create_happiness_agent()
    for i, query in enumerate(CUSTOM_QUERIES, 1):
        print(f"\n--- Task 4 / Custom query {i} ---", flush=True)
        try:
            # No DataFrame injection: these questions use the four provided tools.
            # reset=True avoids carrying all previous tokens into the next query.
            result = agent.run(query, reset=True)
            print(f"Custom query {i} answer: {result}", flush=True)
        except Exception as exc:
            print(f"Custom query {i} failed: {type(exc).__name__}: {exc}")
            break


# --- Task 5: Reflection ---
# 1. Correlation result and interpretation (guided query 3):
#    The CodeAgent called compute_correlation for gdp_per_capita and
#    happiness_score. The Pearson coefficient was r = 0.6313, indicating
#    a moderately strong positive linear association. The reported
#    p_value = 0.0 is the value rounded to four decimals, not a literal
#    zero probability. This supports statistical significance but does
#    not establish that higher GDP causes greater happiness.
#
# 2. Unexpected agent behavior and lessons learned:
#    During the initial Task 2 test, CodeAgent first treated the result
#    of load_happiness_data() as a DataFrame and attempted to read .shape.
#    That failed with AttributeError because the tool actually returns
#    a dictionary with 'shape' and 'columns' keys. The agent inspected
#    the dictionary and recovered, but the extra steps cost API tokens.
#    During guided query 5, the agent generated plotting code itself,
#    grouped scores by year and region, and saved happiness_by_region.png.
#    Matplotlib nevertheless emitted Tkinter GUI-thread warnings when
#    the process exited; the generated image was reported as saved.
#
# 3. Suggested improvement / additional tool:
#    I would add a plot_happiness_by_region(output_path: str) tool that
#    groups the dataframe by year and region, calculates the mean score,
#    saves a labeled multi-line chart using a noninteractive backend,
#    and returns the file path or an error. This would make plotting
#    repeatable, reduce unnecessary code generation and API usage, and
#    avoid GUI-thread issues.
#
# Task 4 extra observation:
#    The custom 2015 ranking returned Switzerland, Iceland, Denmark,
#    Norway and Canada. The additional correlation between Healthy life
#    expectancy and happiness_score was r = 0.6617, another statistically
#    significant positive association (rounded p_value = 0.0).


if __name__ == "__main__":
    print("--- Task 1: Local test (no OpenAI API) ---")
    result = load_happiness_data()
    print("Load:", json.dumps(result, indent=2, default=str))
    if "error" not in result:
        print("Summary:", summarize_column("happiness_score"))
        print("Correlation:", compute_correlation("gdp_per_capita", "happiness_score"))
        print("Top 5 (2020):", get_top_n_countries("happiness_score", 2020, 5))

    if RUN_AGENT_TEST:
        print("\n--- Task 2: Live CodeAgent test (uses OpenAI API) ---")
        agent = create_happiness_agent()
        answer = agent.run("Load the World Happiness data and report the number of rows and columns.")
        print("Agent answer:", answer)

    if RUN_GUIDED_QUERIES:
        run_guided_queries()

    if RUN_CUSTOM_QUERIES:
        run_custom_queries()
