# 🎓 Study Assistant Agent — Agentic L2 AI System

An autonomous, privacy-preserving **AI Study Assistant Agent** built using local LLMs, Model Context Protocol (MCP) client/server architecture, ChromaDB vector storage, bounded LLM tool loops, post-generation reflection, and prompt-injection security.

---

## 📌 1. Problem Statement

Students often struggle with large volumes of study material, understanding complex topics, creating realistic study schedules, generating practice quizzes, and tracking their study progress.

The **Study Assistant Agent** solves this problem by providing a private, local, and grounded learning system that:
* Extracts, chunks, and indexes study materials (PDF, TXT, DOCX) into ChromaDB.
* Uses an LLM-driven decision loop to call specialized tools.
* Generates personalized study schedules aligned with exam countdowns and daily hour capacity.
* Produces material-grounded summaries, practice quizzes (MCQs, short answer, conceptual), and progress metrics.
* Validates draft responses via a Reflection stage to prevent hallucinations, profile constraint violations, and unsupported claims.
* Sanitizes untrusted content to prevent prompt injection.

---

## 🏗️ 2. Updated Project Architecture

```text
                               ┌─────────────────────────┐
                               │         STUDENT         │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │       STREAMLIT UI      │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   LOCAL LLM AGENT LOOP  │
                               │   (Max 5 Bounded Steps) │
                               └────────────┬────────────┘
                                            │
                                     JSON Tool Action
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │        MCP CLIENT       │
                               │  - Schema Validation    │
                               │  - Security Whitelist   │
                               └────────────┬────────────┘
                                            │
                                       MCP Transport
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │        MCP SERVER       │
                               └────────────┬────────────┘
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          │                                 │                                 │
          ▼                                 ▼                                 ▼
┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
│     RAG Tool      │             │   Planner Tool    │             │     Quiz Tool     │
└─────────┬─────────┘             └─────────┬─────────┘             └─────────┬─────────┘
          │                                 │                                 │
          ▼                                 ▼                                 ▼
┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
│     ChromaDB      │             │   Progress Data   │             │   Progress Data   │
└─────────┬─────────┘             └─────────┬─────────┘             └─────────┬─────────┘
          │                                 │                                 │
          └─────────────────────────────────┼─────────────────────────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   EXPLICIT OBSERVATION  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │     LOCAL LLM DRAFT     │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │    REFLECTION AGENT     │
                               │  - Claim Verification   │
                               │  - Profile Constraint   │
                               │  - Citation Check       │
                               └────────────┬────────────┘
                                            │
                                ┌───────────┴───────────┐
                                │                       │
                            SUPPORTED               UNSUPPORTED
                                │                       │
                                ▼                       ▼
                          Final Answer          Constrain/Correct
                                                        │
                                                        ▼
                                                  Final Answer
                                                        │
                                                        ▼
                                                 Source Citations
```

---

## ⚡ 3. Agent Flow

```text
User
 ↓
Agent
 ↓
LLM decides action
 ↓
MCP Client
 ↓
MCP Server
 ↓
Tool
 ↓
Observation
 ↓
LLM
 ↓
Draft
 ↓
Reflection
 ↓
Verified Answer
 ↓
Citation
 ↓
User
```

---

## ⚠️ 4. Important Conceptual Distinction: MCP vs. Vector Database

> [!IMPORTANT]
> **MCP ≠ Vector Database**
> 
> * **Model Context Protocol (MCP)**: A standardized client-server protocol layer that handles tool discovery, typed input schema validation, security authorization, and execution request dispatching.
> * **ChromaDB**: A vector database layer that stores text chunk embeddings and performs fast cosine similarity vector search over uploaded study materials.
> 
> MCP exposes the RAG retriever, Planner, Quiz Generator, Summarizer, and Progress Tracker as standardized tools that the LLM agent can discover and invoke safely.

---

## 🛠️ 5. Key System Components

1. **LLM-Driven Bounded Tool Loop (`src/agent_loop.py`)**: Replaces rigid keyword matching with an LLM decision loop capped at `MAX_AGENT_STEPS = 5`.
2. **Real MCP Integration (`src/mcp/`)**:
   * `server.py`: Runs `mcp.server.Server` exposing tools with JSON Schema definitions.
   * `client.py`: Discovers tools (`session.list_tools()`), validates parameters using Pydantic, checks security whitelists, and executes requests.
3. **Pydantic Typed Validation (`src/tool_schemas.py`)**: Enforces typed constraints, default values, and rejects unexpected arguments (`extra="forbid"`).
4. **Explicit Tool Observations (`src/observations.py`)**: Formats all tool outputs into structured observation text (`OBSERVATION\nTool: ...\nStatus: ...\nResult: ...`).
5. **Reflection Stage (`src/reflection.py`)**: Performs post-generation verification to detect hallucinations, profile limit breaches (e.g. daily study hours), and unsupported claims, modifying the final answer accordingly.
6. **Prompt Injection Defense (`src/prompt_injection.py`)**: Intercepts prompt overrides and wraps untrusted retrieved data inside XML boundaries.

---

## 🧪 6. Test Suite Overview

The project includes a comprehensive pytest test suite located in `tests/`:

* `test_basic.py`: Core baseline unit tests.
* `test_tool_selection.py`: Verifies LLM-driven tool decisions.
* `test_invalid_model_actions.py`: Verifies rejection of unauthorized tools (`delete_database`), bad argument types, missing required parameters, and extra forbidden args.
* `test_missing_tool_data.py`: Verifies graceful handling when tools return null or missing fields without hallucinating.
* `test_partial_tool_data.py`: Verifies profile constraint enforcement when partial data is returned.
* `test_mcp_client.py`: Verifies tool discovery and typed parameter validation in MCP Client.
* `test_mcp_server.py`: Verifies tool request dispatching in MCP Server.
* `test_mcp_integration.py`: Verifies end-to-end MCP Client-Server communication over transport streams.
* `test_reflection.py`: Verifies claim verification and answer constraint logic.
* `test_rag_grounding.py`: Verifies document ingestion, embedding, retrieval relevancy, and citation support.
* `test_prompt_injection.py`: Verifies interception of system instruction override attempts.

---

## 🚀 7. Installation & Run Instructions

### Prerequisites
* Python 3.10+
* (Optional) [Ollama](https://ollama.ai/) running locally with `llama3.2:3b` model.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pytest Test Suite
```bash
python -m pytest
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🎓 8. Mentor Review Q&A Preparation Guide

When presenting to your mentor, use these concise answers:

1. **Why did you replace keyword routing?**
   * *Answer:* Keyword routing is fragile and fails on complex or rephrased queries. Replacing it with an LLM-driven tool loop allows the agent to reason dynamically about intent, pick appropriate tools based on JSON schema, and multi-step solve student requests.
2. **What is the LLM-driven tool loop?**
   * *Answer:* An iterative decision loop where the LLM evaluates the query and previous tool observations to either invoke an MCP tool or return the final answer, bounded by `MAX_AGENT_STEPS = 5`.
3. **What is an observation?**
   * *Answer:* An explicit, structured text block representing a tool's output (`Tool`, `Status`, `Result`) so the LLM can clearly distinguish context from previous actions.
4. **What is MCP?**
   * *Answer:* Model Context Protocol is an open standard connecting AI models to tools and data sources with schema discovery and typed parameter enforcement.
5. **How does MCP client communicate with MCP server?**
   * *Answer:* Over `anyio` memory streams / transport sessions using JSON-RPC request-response messages.
6. **How does tool discovery work?**
   * *Answer:* The client sends `tools/list` request to the MCP server, receiving a manifest of tool names, descriptions, and input schemas.
7. **How do you validate tool arguments?**
   * *Answer:* Using Pydantic models with `extra="forbid"` to check argument types, required fields, and range boundaries before execution.
8. **What is reflection?**
   * *Answer:* A post-generation auditing stage where an agent inspects the draft answer against context, observations, and profile constraints.
9. **How does reflection prevent hallucination?**
   * *Answer:* By checking whether factual claims are grounded in context and explicitly constraining or correcting unsupported text.
10. **How does RAG work?**
    * *Answer:* Text is extracted, split into chunks, embedded with SentenceTransformers, stored in ChromaDB, and retrieved via cosine similarity.
11. **Where is ChromaDB used?**
    * *Answer:* In the vector store manager (`src/vector_store.py`) to index and search study document embeddings.
12. **How do you handle errors?**
    * *Answer:* Specific exception catching and Python logging without swallowing exceptions blindly or exposing private data.
13. **What tests did you write?**
    * *Answer:* 11 test suites covering tool selection, invalid actions, MCP client/server integration, reflection, grounding, prompt injection, and missing data.
14. **How do you prevent prompt injection?**
    * *Answer:* By regex pattern scanning and wrapping untrusted document text in XML delimiters with explicit instructions to ignore embedded overrides.
15. **How do you verify citation correctness?**
    * *Answer:* By checking if the cited document chunk actually contains the content words present in the answer claim via groundedness scoring.#   S m a r t - s t u d y _ a g e n t  
 