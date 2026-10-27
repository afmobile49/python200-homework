
# ============================================================
# Code the Dream - Python 200
# Week 6 - Groundwork Coffee Co. Q&A Assistant
# Part 2: Mini-Project
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# ------------------------------------------------------------
# Step 1: Setup
# ------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

env_path = project_root / ".env"

load_dotenv(env_path)

assert os.getenv("OPENAI_API_KEY"), (
    "OPENAI_API_KEY was not found in the environment."
)

print("API key loaded successfully.")


# Locate Groundwork documents.
assignment_dir = Path(__file__).resolve().parent

docs_dir = assignment_dir / "resources" / "groundwork_docs"

assert docs_dir.is_dir(), (
    f"Document directory not found: {docs_dir}"
)

print("Groundwork documents directory found.")
print("Directory:", docs_dir)


# ------------------------------------------------------------
# Step 2: Load the Documents
# ------------------------------------------------------------

print("\n--- Loading Groundwork Documents ---")

reader = SimpleDirectoryReader(
    input_dir=str(docs_dir),
    required_exts=[".txt"]
)

documents = reader.load_data()

print("Number of documents loaded:", len(documents))

for i, doc in enumerate(documents, start=1):
    filename = doc.metadata.get("file_name", "Unknown")

    print(f"\nDocument {i}")
    print("Filename:", filename)
    print("Characters:", len(doc.text))
    print("Text preview:", doc.text[:200])


# ------------------------------------------------------------
# Step 3: Build the Index and Query Engine
# ------------------------------------------------------------

# Configure the embedding model.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# Configure the language model.
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Build the vector index from the loaded documents.
print("\nBuilding Groundwork vector index...")

index = VectorStoreIndex.from_documents(documents)

print("Index built successfully.")

# Create the query engine.
query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Ready to answer questions.")


# ------------------------------------------------------------
# Step 4: Query the Groundwork Coffee Assistant
# ------------------------------------------------------------

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question_number, question in enumerate(questions, start=1):

    print(f"\n{'=' * 60}")
    print(f"Question {question_number}")
    print(f"{'=' * 60}")

    print("\nQuestion:", question)

    # Retrieve relevant information and generate an answer.
    response = query_engine.query(question)

    print("\nModel answer:")
    print(str(response))

    # Reuse the first query's sources for the diagnostic below.
    if question_number == 1:
        weekend_source_nodes = response.source_nodes

    # Examine the highest-ranked retrieved source.
    if response.source_nodes:

        top_source = response.source_nodes[0]

        filename = top_source.node.metadata.get(
            "file_name",
            "Unknown"
        )

        score = top_source.score
        chunk_text = top_source.node.get_content()

        print("\nTop retrieved source:")
        print("Filename:", filename)
        print("Similarity score:", score)
        print("Text preview:", chunk_text[:200])

    else:
        print("\nNo source nodes were retrieved.")


# ------------------------------------------------------------
# Step 4 - Reflection
# ------------------------------------------------------------

# The assistant answered all five customer questions
# with detailed and confident responses.
#
# The founding story and wholesale/catering questions
# retrieved the expected documents as their top sources.
#
# However, the weekend hours question unexpectedly
# ranked our_story.txt as the top source, even though
# faq.txt contains the opening hours.
#
# The dairy-free milk question ranked
# seasonal_specials.txt first.
#
# The loyalty question retrieved faq.txt, but its
# top similarity score was relatively low.
#
# These results show that a confident and apparently
# correct answer does not guarantee that the highest-
# ranked retrieved document is the best source.
#
# I would inspect all retrieved source nodes and compare
# the generated answers with the original documents
# before treating the responses as fully verified.

# Diagnostic: inspect all sources from Question 1 without
# making an additional API request.

print("\n--- Weekend Hours Source Diagnostics ---")

for i, source_node in enumerate(
    weekend_source_nodes,
    start=1
):
    print(f"\nSource {i}")
    print("Filename:", source_node.node.metadata.get("file_name"))
    print("Score:", source_node.score)
    print("Text:", source_node.node.get_content()[:350])


# ------------------------------------------------------------
# Step 5: Find a Failure
# ------------------------------------------------------------

difficult_question = (
    "Can I reserve a private room for 50 people "
    "next Saturday, and how much will it cost?"
)

print(f"\n{'=' * 60}")
print("Step 5 - Difficult Question")
print(f"{'=' * 60}")

print("\nQuestion:", difficult_question)

response = query_engine.query(difficult_question)

print("\nFull model answer:")
print(str(response))

print("\nAll retrieved sources:")

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
# Step 6: Final Reflection
# ------------------------------------------------------------

# Question 1: What is the value of using LlamaIndex?
#
# LlamaIndex allowed me to build the main RAG pipeline
# using only a few high-level operations.
#
# SimpleDirectoryReader loaded the documents, while
# VectorStoreIndex.from_documents() handled document
# processing, chunking, embedding, and indexing.
#
# The query engine handled retrieval and answer generation.
#
# Without a framework, I would need to implement or
# integrate these components separately.
#
# This shows that frameworks reduce repetitive code,
# simplify development, and let developers focus on
# application logic and response quality.


# Question 2: Describe another business use case.
#
# A digital marketing agency could use RAG to build
# an internal assistant for Google Ads and SEO projects.
#
# The assistant could retrieve information from campaign
# reports, internal procedures, client documentation,
# and troubleshooting guides.
#
# This would help employees find relevant information
# without manually searching many separate documents.
#
# It would be especially useful when the agency's
# documentation changes frequently.


# Question 3: What limitation can RAG not fully prevent?
#
# Even when retrieval works correctly, the language model
# can misinterpret the retrieved information or generate
# unsupported conclusions.
#
# For example, a document may describe catering services
# without confirming private-room availability.
#
# The assistant must not assume that catering means
# a private room can be reserved.
#
# Source checking, clear instructions, evaluation,
# and human review remain important.

# In this project, configuring the two models, creating the
# vector index, and creating the query engine took roughly
# 10-15 lines of Python code.
#
# Loading documents and querying the system required
# additional code, mostly for displaying results.
#
# Implementing document chunking, embeddings, vector
# similarity search, and prompt construction manually
# would require significantly more code.