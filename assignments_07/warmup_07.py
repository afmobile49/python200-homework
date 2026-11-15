"""Code the Dream Python 200 â€” Assignment 07 warmup (Q1â€“Q3)."""

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dotenv import load_dotenv
from openai import OpenAI


# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# Q1: Define a Python tool and its JSON schema.

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}آ°C is {fahrenheit}آ°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a temperature from Celsius to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius.",
                },
            },
            "required": ["celsius"],
            "additionalProperties": False,
        },
    },
}


# Q2: Minimal agent with get_current_time as its only available tool.
# Prediction (written before running):
# The model probably will not request get_current_time because a clock
# does not help with converting Celsius to Fahrenheit.
# I predict one API call if it answers directly; two if it requests a tool.

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Keep this list limited to the time tool until Q2 has been tested.
# Q2: Only the time tool
tools_q2 = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]

# Q3: Both tools
tools_q3 = tools_q2 + [celsius_to_fahrenheit_schema]


def run_agent(user_prompt: str, available_tools=None) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""
    if available_tools is None:
        available_tools = tools_q2
    system_prompt = (
        "You are a simple assistant that can tell the current time. "
        "Use the tool get_current_time whenever a user asks about the time."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Create the client when the agent runs, not when the module is imported.
    load_dotenv()
    client = OpenAI()

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=available_tools,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message
    print("First response received from model...")
    print("Tool calls requested:", [call.function.name for call in (first_message.tool_calls or [])])
    messages.append(first_message.model_dump(exclude_none=True))

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(
                    arguments["celsius"]
                )

            else:
                tool_result = f"Error: unknown tool {function_name}."
            print("Tool called:", function_name)
            print("Tool result:", tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        final_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content or ""

    return first_message.content or ""


# Q2 was already tested successfully with the time-only tool.
# Set RUN_Q2=True when you want to repeat that test (it incurs API cost).
RUN_Q2 = False
RUN_Q3 = False  # Q3 has already passed; avoid repeating API calls


if __name__ == "__main__":
    print("--- Q1: Celsius to Fahrenheit ---")
    for temperature in (0, 100, -40):
        print(celsius_to_fahrenheit(temperature))

    if RUN_Q2:
        print("\n--- Q2: Minimal ReAct Agent (time tool only) ---")
        result = run_agent(
            "Convert 100 degrees Celsius to Fahrenheit",
            available_tools=tools_q2,
        )
        print("Q2 Result:", result)
        # Earlier observed Q2 result: no tool call; one API call.
        # Compare any new run to the prediction above.

    if RUN_Q3:
        print("\n--- Q3: Multi-Tool Agent ---")
        response_a = run_agent(
            "What is 37 degrees Celsius in Fahrenheit?",
            available_tools=tools_q3,
        )
        print("Response A:", response_a)
        # Observed: called celsius_to_fahrenheit; returned 98.6 F.

        response_b = run_agent(
            "What is the boiling point of water in plain English?",
            available_tools=tools_q3,
        )
        print("Response B:", response_b)
        # Observed: answered directly with no tool call.


# --- Lesson 03: Multi-Tool Agent (Q4-Q6) ---
# The original class, schema and ReAct function below come from 03_analysis_agent.md,
# with only the new correlation tool and resource path adjustments.

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    # Q4: New Pearson correlation tool
    def compute_correlation(self, col1: str, col2: str):
        """Compute Pearson correlation and p-value for two numeric CSV columns."""
        error = self._ensure_loaded()
        if error:
            return error

        missing = [col for col in (col1, col2) if col not in self.df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}"}

        try:
            pair = self.df[[col1, col2]].apply(pd.to_numeric, errors="raise").dropna()
            if len(pair) < 2:
                return {"error": "At least 2 complete numeric rows are required."}
            if pair[col1].nunique() < 2 or pair[col2].nunique() < 2:
                return {"error": "Pearson correlation requires non-constant columns."}
            r, p = pearsonr(pair[col1], pair[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4),
            }
        except (TypeError, ValueError) as exc:
            return {"error": f"Could not compute correlation: {exc}"}


RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation coefficient and p-value between two numeric columns of the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First numeric column name."},
                    "col2": {"type": "string", "description": "Second numeric column name."},
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

# Local Q4 check requires no OpenAI API request.
RUN_Q4_LOCAL_TEST = False
RUN_Q5_Q6 = False  # Change to True only when ready to pay for the agent API run.


if __name__ == "__main__":
    if RUN_Q4_LOCAL_TEST:
        print("\n--- Q4: Local correlation tool verification (no API) ---")
        print(csv_backend.load_csv("bike_commute.csv"))
        print(csv_backend.compute_correlation("avg_traffic_density", "avg_speed_kmh"))

    if RUN_Q5_Q6:
        print("\n--- Q5: Agent computing correlation ---")
        load_dotenv()
        client = OpenAI()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        result = run_agent_cycle(
            messages,
            "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
        )
        print("Q5 Result:", result)

        # Q6: system=instructions, user=request, assistant=decisions/tool requests,
        # tool=Python function output observed by the model in the ReAct loop.
        print("\n--- Q6: Full message history ---")
        print(json.dumps(messages, indent=2, default=str))


# --- Lesson 04: smolagents ---
# Q7: Wrap the existing CsvManager methods as smolagents tools.
# No OpenAI request is needed to define these tools or print their descriptions.

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool
import os

# Reuse the CsvManager instance that the earlier questions created.
csv_manager = csv_backend


@tool
def list_csv_files() -> dict:
    """List CSV files found in the resources directory.

    Returns:
        A dictionary listing available filenames or reporting no files.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file as the active dataset.

    Args:
        filename: Filename in resources, such as bike_commute.csv.

    Returns:
        A dictionary with the load status and column names, or an error.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Get the column names in the active dataset.

    Returns:
        A list of columns or an error dictionary if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return descriptive statistics for selected or all columns.

    Args:
        columns: Optional list of column names. None selects all columns.

    Returns:
        A dictionary of summary statistics or an error dictionary.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Return descriptive statistics for one column.

    Args:
        column: Name of the column to describe.

    Returns:
        A dictionary of statistics or an error dictionary.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot the active CSV using a simple scatter or line plot.

    Args:
        y: Name of the column plotted on the vertical axis.
        x: Optional name of the horizontal-axis column.
        plot_type: Either scatter or line.

    Returns:
        A status message or an error after attempting the plot.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and a p-value for two numeric CSV columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary with col1, col2, pearson_r and p_value rounded to
        four decimals, or an error dictionary for invalid inputs.
    """
    return csv_manager.compute_correlation(col1, col2)


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# Q7 comparison:
# smolagents generates a tool description/schema from the function signature,
# type hints and Google-style docstring, rather than requiring the manually
# authored JSON schema used in Q4. Clear argument names, types and descriptions
# are needed so the model knows what the tool accepts and when to call it.

# Q8: Run only after inspecting the Q7 output.
RUN_Q7 = True
RUN_Q8 = False  # Set True for actual API-based tool-vs-code comparison.


def run_q8():
    """Compare ToolCallingAgent and CodeAgent on the same user prompt."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment/.env.")

    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    tool_instructions = (
        "You are a small CSV analysis assistant. Use the provided tools "
        "for data tasks; never claim unsupported plot styling succeeded."
    )
    code_instructions = """
You are a helpful CSV analysis assistant.
Prefer provided tools for simple data operations.
If the user asks for plot styling the plot_data tool cannot support
(e.g. dot colors), write matplotlib code directly rather than claiming
the tool produced the requested styling.
Only claim successful actions when their outputs confirm success.
The active dataset is in csv_manager.df after loading a CSV.
"""

    tool_agent = ToolCallingAgent(
        tools=TOOLS, model=model, instructions=tool_instructions, max_steps=8
    )
    code_agent = CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=code_instructions,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
        max_steps=8,
    )

    prompt = (
        "Load bike_commute.csv. Plot avg_heart_rate vs duration_min "
        "as a scatter plot with green dots."
    )

    print("\\n--- Q8: ToolCallingAgent ---")
    response_tool = tool_agent.run(prompt)
    print("ToolCallingAgent response:", response_tool)

    print("\\n--- Q8: CodeAgent ---")
    response_code = code_agent.run(
        prompt, additional_args={"csv_manager": csv_manager}
    )
    print("CodeAgent response:", response_code)


    # Q8 - ToolCallingAgent vs CodeAgent
    #
    # Both agents successfully loaded bike_commute.csv.
    #
    # ToolCallingAgent:
    # The agent called load_csv and plot_data.
    # It created a scatter plot with the correct x and y columns,
    # but the points were blue instead of the requested green.
    # The plot_data tool does not support a color parameter.
    # The agent did not acknowledge this limitation.
    #
    # CodeAgent:
    # The agent generated custom matplotlib code using
    # plt.scatter(..., color="green").
    # The resulting plots had green points, as requested.
    #
    # The first CodeAgent attempt encountered a Tkinter threading
    # error and a timeout when calling plt.show().
    # The agent retried, sampled 100 rows, and saved a green
    # scatter plot to avg_heart_rate_vs_duration.png.
    #
    # Conclusion:
    # ToolCallingAgent is limited by its predefined tools.
    # CodeAgent can generate code for unsupported functionality.




# Q9 - Reflection
#
# A good use case for ToolCallingAgent is a CSV analysis
# assistant that performs predefined operations, such as
# loading files, listing columns, and computing correlations.
# These operations are predictable and easier to audit.
#
# CodeAgent is more flexible because it can generate and
# execute Python code. However, this introduces security
# risks, such as unintended file access, data modification,
# or execution of harmful code.
#
# CodeAgent should therefore run with restricted permissions
# and in an isolated environment when appropriate.
#
# In Q8, CodeAgent also encountered a GUI threading issue.
# This shows that generated code can introduce execution
# problems even when its logic is correct.


if __name__ == "__main__":
    if RUN_Q7:
        print("\\n--- Q7: smolagents tool description (no API) ---")
        print(compute_correlation.description)

    if RUN_Q8:
        run_q8()
