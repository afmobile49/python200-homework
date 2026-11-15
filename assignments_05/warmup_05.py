"""Assignment 05 – Warmup: OpenAI API, prompting, and local Ollama.

Install: pip install openai python-dotenv
Set OPENAI_API_KEY in a .env file that is excluded from Git.
Run this file after uncommenting the exercise you want in __main__.

NOTE: The AIRHub feedback does not quote the original three reviews or the
original delimiter passages. REVIEW_TEXTS, TOOL_REVIEW, and STEP_PASSAGE below
use clear example inputs. Replace with the *exact course-provided texts* before
submission if those are different. Do not label these examples as official.
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = "gpt-4o-mini"

# IMPORTANT: Substitute the exact three reviews from the Lesson 5 handout.
# Keep the SAME three reviews in Q1, Q2 and Q3, in this order.
REVIEW_TEXTS = [
    "The app is easy to use and works perfectly. I love it!",
    "The app keeps crashing, and customer support was unhelpful.",
    "The app has useful features, but it is slow and sometimes freezes.",
]
LABELS = "positive, negative, or mixed"

# IMPORTANT: Substitute the exact tool review from the handout if different.
TOOL_REVIEW = (
    "The tool is easy to set up and saves me time, but its reports are slow "
    "and it sometimes crashes."
)

# IMPORTANT: Substitute the exact passages from the handout if different.
STEP_PASSAGE = (
    "To prepare tea, first boil water. Next, place a tea bag in a cup. "
    "Pour the hot water into the cup, steep for three minutes, "
    "and then remove the tea bag."
)
NON_STEP_PASSAGE = (
    "The weather is warm and sunny. Trees are green, and birds are singing."
)


def get_client():
    """Construct the cloud client only when making a real API request."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from the environment/.env")
    return OpenAI()


def ask(messages, *, max_tokens=150, **options):
    return get_client().chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, **options
    )


def answer_text(response):
    return (response.choices[0].message.content or "").strip()


def api_question_1():
    """API Q1: display text, model and token count."""
    response = ask([{"role": "user", "content": "Why is Python good for beginners?"}], max_tokens=40)
    print("Answer:", answer_text(response))
    print("Model:", response.model)
    print("Total tokens:", response.usage.total_tokens)


def api_question_2():
    """API Q2: compare creativity across three temperatures."""
    for temperature in [0, 0.7, 1.5]:
        response = ask([{"role": "user", "content": (
            "Suggest one creative name for a Python weather prediction app. "
            "Reply with only the name."
        )}], temperature=temperature, max_tokens=40)
        print(f"\nTemperature: {temperature}")
        print("Answer:", answer_text(response))
    # Observation: A higher temperature can increase variation, but does not
    # guarantee a higher-quality answer.
    # I would choose temperature=0 for more consistent and reproducible results.

def api_question_3():
    """API Q3: get three completions in one API call."""
    response = ask([{"role": "user", "content": (
        "Suggest a short name for a Python weather prediction app."
    )}], n=3, max_tokens=30)
    for index, choice in enumerate(response.choices, start=1):
        print(f"Choice {index}:", choice.message.content)
    print("Total tokens:", response.usage.total_tokens)


def api_question_4():
    """API Q4: demonstrate truncated output with a small token budget."""
    response = ask([{"role": "user", "content": (
        "Explain five advantages of learning Python programming in detail."
    )}], max_tokens=15)
    print("Answer:", answer_text(response))
    print("Finish reason:", response.choices[0].finish_reason)
    print("Total tokens:", response.usage.total_tokens)
    # A finish_reason of 'length' means output reached its token budget.
    # In a real application, max_tokens helps limit response length,
    # control API costs, and prevent unnecessarily long outputs.

def system_question_1():
    """System Q1: compare two system personalities."""
    question = "Explain what a Python function is."
    personas = [
        ("Beginner Teacher", "You are a friendly Python teacher. Explain in simple English with a real-world example."),
        ("Senior Developer", "You are a senior Python developer. Use precise technical terminology."),
    ]
    for label, instruction in personas:
        response = ask([
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ], max_tokens=120)
        print("\nPersona:", label)
        print("Answer:", answer_text(response))
        print("Tokens:", response.usage.total_tokens)


    # Reflection: Both personas answered the same question about
    # Python functions, but their explanations had different styles.
    # The Beginner Teacher used friendly, simple language and
    # a real-world example, making the answer easier for beginners.
    # The Senior Developer used more precise technical terminology,
    # making the explanation more suitable for experienced programmers.
    # This shows how a system prompt can change the tone and
    # level of detail without changing the user's question.



def system_question_2():
    """System Q2: demonstrate explicitly supplied message history."""
    messages = [
        {"role": "system", "content": "You are a friendly assistant. Answer briefly and accurately."},
        {"role": "user", "content": "My name is Jordan, and I am learning Python programming."},
        {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great language to learn."},
        {"role": "user", "content": "What is my name, and what programming language am I learning?"},
    ]
    response = ask(messages, max_tokens=80)
    print("Answer:", answer_text(response))
    print("Total tokens:", response.usage.total_tokens)
    # API requests are stateless: Jordan's name is available because prior
    # conversation messages were explicitly supplied in this request.


def classify_reviews(title, examples):
    """Run the identical set of three reviews, displaying a label per number."""
    review_block = "\n".join(
        f"Review {number}: {review}" for number, review in enumerate(REVIEW_TEXTS, start=1)
    )
    prompt = (
        "Classify each of the three product reviews as exactly one of "
        f"these categories: {LABELS}.\n"
        "Write exactly three lines in this format:\n"
        "Review 1: positive|negative|mixed\n"
        "Review 2: positive|negative|mixed\n"
        "Review 3: positive|negative|mixed\n"
        "Use one label per line, with no explanation.\n\n"
        f"{examples}\n\n{review_block}"
    )
    response = ask([{"role": "user", "content": prompt}], max_tokens=110, temperature=0)
    print(f"\n=== {title} ===")
    print(answer_text(response))
    print("Total tokens:", response.usage.total_tokens)
    return answer_text(response)


def prompt_question_1():
    """Prompt Q1: zero-shot classification of ALL three required reviews."""
    return classify_reviews("Zero-shot", "")



def prompt_question_2():
    """Prompt Q2: one-shot classification of the SAME reviews."""
    example = "Example (different from the three test reviews):\nReview: The camera works great.\nSentiment: positive"

    # Observation: Both zero-shot (Q1) and one-shot (Q2)
    # produced the same labels: positive, negative, and mixed.
    # Adding one example did not change the output format or
    # classification results in this test. However, the example
    # made the expected label format clearer and may help
    # improve consistency with less familiar inputs.

    return classify_reviews("One-shot", example)


def prompt_question_3():
    """Prompt Q3: few-shot classification of the SAME reviews."""
    examples = (
        "Labeled examples (not part of the three test reviews):\n"
        "Review: It is fast, reliable, and helpful.\nSentiment: positive\n"
        "Review: I regret buying this unreliable product.\nSentiment: negative\n"
        "Review: Great design, but the battery drains too quickly.\nSentiment: mixed"
    )
    result = classify_reviews("Few-shot", examples)
    # Compared with zero-shot, labeled examples demonstrate the intended
    # categories and output style, but use additional input tokens.
    return result


def prompt_question_4():
    """Prompt Q4: request clear calculation steps."""
    prompt = (
        "An employee earns $85,000 per year. They receive a 12% salary increase. "
        "After the increase, they receive a $7,500 bonus. "
        "Calculate the total annual compensation in clear, numbered steps, "
        "and give the final answer in US dollars."
    )
    response = ask([{"role": "user", "content": prompt}], max_tokens=250)
    print("=== Step-by-step Prompting ===")
    print(answer_text(response))
    print("Total tokens:", response.usage.total_tokens)
    # Expected arithmetic: 85,000 * 1.12 = 95,200; plus 7,500 = 102,700.
    # Breaking a calculation into visible intermediate steps can help check
    # arithmetic and catch errors, but it does not guarantee correctness.


def prompt_question_5():
    """Prompt Q5: parse required sentiment JSON with error handling."""
    prompt = (
        "Analyze the following tool review and return ONLY a valid JSON object. "
        "It must have exactly these keys: 'sentiment' (positive, negative, or mixed), "
        "'confidence' (a number between 0 and 1), and 'reason' (a brief string). "
        f"\n\nTool review: {TOOL_REVIEW}"
    )
    response = ask([{"role": "user", "content": prompt}],
                   response_format={"type": "json_object"}, max_tokens=160)
    raw = answer_text(response)
    print("=== Structured JSON Output ===")
    try:
        result = json.loads(raw)
        if (not isinstance(result, dict)
                or set(result) != {"sentiment", "confidence", "reason"}
                or result["sentiment"] not in ("positive", "negative", "mixed")
                or not isinstance(result["reason"], str)
                or not isinstance(result["confidence"], (float, int))
                or isinstance(result["confidence"], bool)
                or not 0 <= result["confidence"] <= 1):
            raise ValueError("JSON does not match the required schema")
        print("Parsed data:", result)
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
        print("Reason:", result["reason"])
    except (json.JSONDecodeError, ValueError) as exc:
        print("JSON parse/validation failed:", exc)
        print("Raw response:", raw)
        return None
    print("Total tokens:", response.usage.total_tokens)
    return result


def prompt_question_6():
    """Prompt Q6: two delimiter-protected passages and required fallback."""
    instruction = (
        "Read ONLY the text between triple backticks. If it contains a series "
        "of actions or instructions, rewrite them as a numbered list (1., 2., ...). "
        "If the passage does NOT contain steps, respond with EXACTLY "
        "the single sentence 'No steps provided.' (without quotation marks). "
        "Do not add any explanation.\n\n"
    )
    for label, passage in [
        ("Instructional passage", STEP_PASSAGE),
        ("Non-instructional passage", NON_STEP_PASSAGE),
    ]:
        response = ask([{"role": "user", "content": instruction + "```\n" + passage + "\n```"}],
                       max_tokens=180, temperature=0)
        output = answer_text(response)
        print(f"\n=== {label} ===")
        print(output)
        if label == "Non-instructional passage":
            print("Matches required fallback exactly:", output == "No steps provided.")
    # Delimiters separate data from instructions; the second call specifically
    # tests that prose without actionable steps triggers the exact fallback.


def ollama_comparison():
    """Explain how to run a fair local-versus-cloud comparison."""
    print("Compare local qwen3:0.6b with cloud gpt-4o-mini using the SAME prompt.")
    print("Run ollama_question_1() while the Ollama service is running.")

# Ollama Question 1 - Actual terminal output
#
# Command:
# ollama run qwen3:0.6b "What is one advantage of Python for beginners? Answer in one sentence."
#
# The terminal displayed a Thinking section, followed by:
#
# ...done thinking.
#
# Python is easy to learn for beginners, making it accessible
# to a wide range of users.
#
# The following comparison was also tested using the same prompt:
# Local model: qwen3:0.6b
# Local answer: Python is easy to learn, allowing beginners
# to quickly get started with programming tasks.
#
# Cloud model: gpt-4o-mini-2024-07-18
# Cloud answer: One advantage of Python for beginners is its
# simple and readable syntax, which makes it easier to
# learn and understand programming concepts.
# Cloud tokens: 45
#
# Observation:
# Both models gave relevant answers to the same prompt.
# Ollama runs locally, while the OpenAI model uses a cloud API.


def ollama_question_1():
    """Ollama Q1: make both calls with identical messages and compare results."""
    prompt = "What is one advantage of Python for beginners? Answer in one sentence."
    messages = [{"role": "user", "content": prompt}]
    local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    local = local_client.chat.completions.create(
        model="qwen3:0.6b", messages=messages,
        max_tokens=300, 
    )
    cloud = ask(messages, max_tokens=120)
    print("=== Same Prompt: Ollama vs. OpenAI ===")
    print("Prompt:", prompt)
    print("Local model:", local.model)
    print("Local answer:", answer_text(local))
    print("Cloud model:", cloud.model)
    print("Cloud answer:", answer_text(cloud))
    print("Cloud tokens:", cloud.usage.total_tokens)
    # Compare factuality, clarity, length, speed, and resource/cost tradeoffs.
    # Ollama runs locally; the cloud API requires a key and may incur charges.

if __name__ == "__main__":
    # Uncomment exactly ONE desired exercise. Each API call may incur charges.
    # api_question_1()
    # api_question_2()
    # api_question_3()
    # api_question_4()
    # system_question_1()
    # system_question_2()
    # prompt_question_1()
    # prompt_question_2()
    # prompt_question_3()
    # prompt_question_4()
    # prompt_question_5()
    # prompt_question_6()
    ollama_comparison()  # Instruction text only; no network request.
    # ollama_question_1()  # Requires the local Ollama server and OpenAI API key.
