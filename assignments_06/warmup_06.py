
# ============================================================
# Code the Dream - Python 200
# Week 6: Augmenting AI - From Memory to Retrieval
# Assignment 06 - Part 1: Warmup Exercises
# ============================================================

import os
import string
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------

assignment_dir = Path(__file__).resolve().parent
project_root = assignment_dir.parent
load_dotenv(project_root / ".env")
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY was not found."
print("API key loaded successfully.")


# ============================================================
# SECTION 1: RAG Concepts
# ============================================================

# ------------------------------------------------------------
# Concepts Question 1
# ------------------------------------------------------------

# Scenario A:
# RAG is the best approach because the legal team has
# hundreds of internal policy documents that change regularly.
# RAG can retrieve updated information without retraining
# the language model.

# Scenario B:
# Fine-tuning is a good choice because the startup has
# 3,000 examples of its unique brand voice.
# These examples can help the model learn the writing style.

# Scenario C:
# Prompt engineering with context injection is the simplest
# choice because the report is only two pages long.
# The analyst can include the report directly in the prompt.


# ------------------------------------------------------------
# Concepts Question 2
# ------------------------------------------------------------

# A confidently wrong answer is more harmful because people
# may trust it and make decisions without checking the facts.
#
# For example, an AI assistant might give an incorrect company
# security policy, causing an employee to make an unsafe decision.
#
# Saying "I am not sure" helps users understand that the
# information needs verification.
#
# The tone matters because a confident answer can sound
# trustworthy even when the information is incorrect.


# ------------------------------------------------------------
# Concepts Question 3
# ------------------------------------------------------------

# Correct order of the RAG pipeline:

# Step 1: Extract text from source documents.
# Read text from PDFs and other source files.

# Step 2: Split text into chunks.
# Divide long documents into smaller, manageable pieces.

# Step 3: Convert text chunks into embeddings.
# Represent each chunk as a numerical vector.

# Step 4: Receive the user's query.
# Accept the question that the user wants answered.

# Step 5: Embed the user's query.
# Convert the question into a numerical vector.

# Step 6: Retrieve the most relevant chunks.
# Find text chunks that are most similar to the query.

# Step 7: Inject retrieved chunks into the prompt.
# Add the relevant information to the LLM prompt.

# Step 8: Generate a response from the LLM.
# Produce an answer using the retrieved context.


# ============================================================
# SECTION 2: Keyword RAG
# ============================================================

# ------------------------------------------------------------
# Keyword Retrieval Function
# Copied from the official assignment.
# ------------------------------------------------------------

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# ------------------------------------------------------------
# Sample Documents from the Assignment
# ------------------------------------------------------------

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# ------------------------------------------------------------
# Keyword Question 1
# ------------------------------------------------------------

query = "What are your hours on weekends?"

result = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

print("\nSelected document:", result[0][0])

# Reflection:
# The function selected loyalty.txt instead of hours.txt.
#
# The word "hours" appears in the filename hours.txt,
# but not in its content, so it does not increase the score.
#
# The word "your" is not included in the stopwords list.
# Both hiring.txt and loyalty.txt match this word.
#
# Three documents receive a score of 1.
# Because the function sorts the tuples in descending order,
# loyalty.txt is selected first when the scores are tied.
#
# This shows a limitation of keyword-based retrieval:
# matching words does not always mean matching meaning.


# ------------------------------------------------------------
# Keyword Question 2
# ------------------------------------------------------------

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

print("\nSelected document:", result[0][0])


# Keyword Question 2 - Reflection

# The function returned "None found" because none of the
# query keywords matched the words in the documents.
#
# Keyword retrieval worked as designed, but it could not
# find a useful answer to the customer's question.
#
# Semantic retrieval could perform better when relevant
# information is expressed using different words.
#
# However, the documents do not confirm that any
# caffeine-free drinks are available. The assistant
# should not invent information that is not documented.


# ------------------------------------------------------------
# Keyword Question 3
# ------------------------------------------------------------

# Prediction:
# I expect the function to return "None found".
# The question uses "rewards" and "sign up", but the
# loyalty document uses words like "loyalty", "points",
# and "redeem". Keyword matching does not understand
# that these expressions can be related.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

print("\nSelected document:", result[0][0])


# Keyword Question 3 - Reflection

# My prediction was correct. The function returned
# "None found" because there were no matching keywords.
#
# The loyalty.txt document contains information about
# earning points and redeeming rewards, but the query
# uses different words such as "sign up" and "rewards".
#
# This shows that keyword retrieval can miss a relevant
# document when the query and document use different words.
#
# Semantic search could improve retrieval by comparing
# the meanings of texts instead of exact keywords.



# ============================================================
# SECTION 3: Semantic RAG
# ============================================================

# ------------------------------------------------------------
# Semantic Question 1
# ------------------------------------------------------------

# 1. What is a vector embedding?
#
# A vector embedding is a numerical representation of text.
# It helps a computer compare the meanings of words,
# sentences, or documents using numbers.


# 2. What do similarity scores of 0.85 and 0.30 mean?
#
# A score of 0.85 indicates higher cosine similarity
# between the query and a text chunk than a score of 0.30.
#
# Therefore, the chunk with a score of 0.85 is more likely
# to be relevant to the query.
#
# These scores do not represent the probability that
# an answer is factually correct.


# 3. Why can semantic search find documents that
#    share no exact keywords with the query?
#
# Semantic search compares the meaning of text using
# embeddings instead of matching only exact words.
#
# For example, "rewards" and "loyalty points" are
# related concepts even though the words are different.
#
# This helps semantic search retrieve relevant documents
# that keyword-based retrieval might miss.



# ------------------------------------------------------------
# Semantic Question 2
# ------------------------------------------------------------

# Comparison: Keyword Retrieval vs. Semantic Retrieval
#
# | Feature             | Keyword RAG                | Semantic RAG           |
# |---------------------|----------------------------|------------------------|
# | What is compared?   | Exact word overlap         | Embedding similarity   |
# | What is retrieved?  | Full documents (this task) | Text chunks (typical)  |
# | Handles synonyms?   | No                         | Often yes              |
# | Storage format      | Plain text dictionary      | Vector index           |
# | Relevance score     | Count of shared keywords   | Vector similarity      |

# 1. Comparison method
#
# Keyword retrieval compares exact words or tokens.
# Semantic retrieval compares numerical embeddings
# that represent the meaning of text.


# 2. Search unit
#
# In our keyword example, the entire document is searched.
# Semantic RAG commonly divides documents into smaller
# chunks and retrieves the most relevant chunks.
#
# Both approaches can work with either documents or chunks.


# 3. Synonyms and related concepts
#
# Keyword retrieval may miss relevant information when
# the query and document use different words.
#
# Semantic retrieval can recognize related meanings,
# such as "rewards" and "loyalty points".


# 4. Storage and indexing
#
# Our keyword retrieval uses a simple Python dictionary.
#
# Semantic retrieval uses vector embeddings and usually
# a vector index to search efficiently.


# 5. Relevance scoring
#
# Our keyword retrieval counts overlapping words.
#
# Semantic retrieval uses vector similarity metrics,
# such as cosine similarity, to rank relevant text.

# ============================================================
# SECTION 4: LlamaIndex Warmup
# Load BrightLeaf PDF Documents
# ============================================================

pdf_dir = assignment_dir / "resources" / "brightleaf_pdfs"
assert pdf_dir.is_dir(), f"PDF directory not found: {pdf_dir}"

brightleaf_documents = SimpleDirectoryReader(
    input_dir=str(pdf_dir),
    required_exts=[".pdf"]
).load_data()

print("\n--- BrightLeaf Documents ---")
print("Loaded document objects:", len(brightleaf_documents))

for i, doc in enumerate(brightleaf_documents, start=1):
    print(f"\nDocument {i}")
    print("File:", doc.metadata.get("file_name"))
    print("Text preview:", doc.text[:150])


# ============================================================
# SECTION 4: LlamaIndex Warmup
# LlamaIndex Question 1
# ============================================================


# Step 1: Configure the embedding model.
# It converts document text and questions into vectors.

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# Step 2: Configure the language model.
# It generates answers using the retrieved documents.

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Step 3: Build the vector index.
# LlamaIndex chunks the documents and creates embeddings.

print("\nBuilding BrightLeaf vector index...")

index = VectorStoreIndex.from_documents(
    brightleaf_documents
)

print("Vector index built successfully.")


# Step 4: Create the query engine.
# Retrieve the top 3 most similar chunks.

query_engine = index.as_query_engine(
    similarity_top_k=3
)


# Step 5: Define the two assignment questions.

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


# Step 6: Ask each question and inspect the sources.

for question_number, question in enumerate(questions, start=1):

    print(f"\n{'=' * 60}")
    print(f"LlamaIndex Question 1 - Query {question_number}")
    print(f"{'=' * 60}")

    print("\nQuestion:", question)

    response = query_engine.query(question)

    print("\nModel answer:")
    print(str(response))

    print("\nRetrieved source nodes:")

    for i, source_node in enumerate(response.source_nodes, start=1):

        score = source_node.score
        filename = source_node.node.metadata.get(
            "file_name",
            "Unknown"
        )

        chunk_text = source_node.node.get_content()

        print(f"\nSource {i}")
        print("Filename:", filename)
        print("Similarity score:", score)
        print("Text preview:", chunk_text[:150])


# ------------------------------------------------------------
# LlamaIndex Question 1 - Reflection
# ------------------------------------------------------------

# Query 1: Employee Benefits
#
# The most relevant source was employee_benefits.pdf
# with a similarity score of approximately 0.765.
#
# The model gave a detailed and confident answer about
# insurance, retirement, parental leave, flexible work,
# and professional development.
#
# The retrieval of partnerships.pdf and
# mission_statement.pdf was somewhat unexpected because
# these documents are not directly about employee benefits.
#
# The answer sounded confident and specific, but confidence
# alone does not prove that every detail is correct.


# Query 2: Security Policies
#
# The most relevant source was security_policy.pdf
# with a similarity score of approximately 0.676.
#
# The model gave a detailed answer covering authentication,
# encryption, incident response, access control,
# employee training, and compliance.
#
# The retrieval of employee_benefits.pdf and
# mission_statement.pdf was less directly relevant.
#
# The model did not express much uncertainty.
# This shows why retrieved sources should be inspected
# rather than trusting the model's tone alone.


# ------------------------------------------------------------
# LlamaIndex Question 2
# Compare similarity_top_k=1 and similarity_top_k=5
# ------------------------------------------------------------

comparison_question = (
    "What employee benefits does BrightLeaf offer?"
)

for top_k in [1, 5]:

    print(f"\n{'=' * 60}")
    print(f"LlamaIndex Question 2 - top_k={top_k}")
    print(f"{'=' * 60}")

    comparison_engine = index.as_query_engine(
        similarity_top_k=top_k
    )

    response = comparison_engine.query(
        comparison_question
    )

    print("\nQuestion:", comparison_question)
    print("\nModel answer:")
    print(str(response))

    print("\nRetrieved sources:")

    for i, source_node in enumerate(
        response.source_nodes,
        start=1
    ):
        filename = source_node.node.metadata.get(
            "file_name",
            "Unknown"
        )

        print(
            f"Source {i}: "
            f"{filename}, "
            f"score={source_node.score}"
        )


# ------------------------------------------------------------
# LlamaIndex Question 2 - Reflection
# ------------------------------------------------------------

# I compared the same employee benefits question using
# similarity_top_k=1 and similarity_top_k=5.
#
# With top_k=1, the system retrieved employee_benefits.pdf
# with a similarity score of approximately 0.765.
# The model provided a detailed and relevant answer.
#
# With top_k=5, the system retrieved five documents,
# including partnerships.pdf, mission_statement.pdf,
# earnings_report.pdf, and security_policy.pdf.
#
# The main information in the model's answer did not
# change significantly between the two runs.
#
# More retrieved context is not always better.
# Additional documents may introduce irrelevant
# information, increase token usage, and make
# the response less focused.
#
# However, a larger top_k can be useful when the
# answer requires information from multiple documents.


# ------------------------------------------------------------
# LlamaIndex Question 3
# Test a difficult or unsupported question
# ------------------------------------------------------------

difficult_question = (
    "What is BrightLeaf's official policy for "
    "employee travel expenses in 2026?"
)

print(f"\n{'=' * 60}")
print("LlamaIndex Question 3 - Difficult Query")
print(f"{'=' * 60}")

print("\nQuestion:", difficult_question)

response = query_engine.query(difficult_question)

print("\nModel answer:")
print(str(response))

print("\nAll retrieved source nodes:")

for i, source_node in enumerate(response.source_nodes, start=1):

    filename = source_node.node.metadata.get(
        "file_name",
        "Unknown"
    )

    print(f"\nSource {i}")
    print("Filename:", filename)
    print("Similarity score:", source_node.score)
    print("Text preview:", source_node.node.get_content()[:200])


# ------------------------------------------------------------
# LlamaIndex Question 3 - Reflection
# ------------------------------------------------------------

# I asked about BrightLeaf's official employee travel
# expense policy for 2026 because I expected this
# information to be missing from the provided documents.
#
# The system retrieved employee_benefits.pdf,
# partnerships.pdf, and mission_statement.pdf.
# None of the displayed source previews directly
# addressed the requested travel expense policy.
#
# The model responded that the provided information
# did not include details about this policy.
#
# This was a good result because the model did not
# invent unsupported rules or reimbursement amounts.
#
# The model's tone was cautious and acknowledged
# the limits of the available information.
#
# To improve the system, I would add an official
# travel expense policy document if one is available.
# I would also consider relevance thresholds and
# instructions to avoid unsupported answers.


# ------------------------------------------------------------
# LlamaIndex Question 4
# Evaluate RAG responses using LLM-as-a-Judge
# ------------------------------------------------------------

# Use a separate LLM as the evaluator.
judge_llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0
)

faithfulness_evaluator = FaithfulnessEvaluator(
    llm=judge_llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=judge_llm
)

evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is BrightLeaf's official policy for employee travel expenses in 2026?",
]

for question_number, question in enumerate(
    evaluation_questions, start=1
):

    print(f"\n{'=' * 60}")
    print(f"LlamaIndex Question 4 - Evaluation {question_number}")
    print(f"{'=' * 60}")

    print("\nQuestion:", question)

    # Generate a new RAG response.
    rag_response = query_engine.query(question)

    print("\nModel answer:")
    print(str(rag_response))

    # Evaluate whether the answer is supported
    # by the retrieved source documents.
    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=question,
        response=rag_response
    )

    # Evaluate whether the answer is relevant
    # to the user's original question.
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=rag_response
    )

    print("\nFaithfulness:")
    print("Score:", faithfulness_result.score)
    print("Passing:", faithfulness_result.passing)

    print("\nRelevancy:")
    print("Score:", relevancy_result.score)
    print("Passing:", relevancy_result.passing)

    print("\nFaithfulness feedback:")
    print(faithfulness_result.feedback)

    print("\nRelevancy feedback:")
    print(relevancy_result.feedback)


# ------------------------------------------------------------
# LlamaIndex Question 4 - Reflection
# ------------------------------------------------------------

# A faithfulness score of 1.0 means the evaluator judged
# the answer to be supported by the retrieved context.
#
# A score of 0.0 means the response did not pass the
# evaluator's faithfulness check. It does not necessarily
# mean every statement in the response was incorrect.
#
# Relevancy measures whether the response addresses
# the user's question. Faithfulness measures whether
# the response is grounded in the retrieved sources.
#
# For the employee benefits question, both evaluators
# returned 1.0 and passing=True.
#
# For the employee travel expense policy question,
# both evaluators returned 0.0 and passing=False.
#
# The second response correctly acknowledged that
# the provided information did not contain the policy.
# However, the evaluators still rejected the response.
#
# This shows that evaluation scores require interpretation.
# A refusal to invent missing facts can be appropriate
# even when an automated evaluator gives a low score.
#
# LLM-as-a-judge uses a language model to evaluate
# generated answers based on context and relevance.
#
# It is useful because RAG responses are open-ended
# and may have several valid phrasings, making simple
# exact-match accuracy insufficient.
#
# However, LLM judges can also make mistakes,
# so human review remains important.

print("\nFaithfulness feedback:")
print(faithfulness_result.feedback)

print("\nRelevancy feedback:")
print(relevancy_result.feedback)