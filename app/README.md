# Backend Developer's Guide

This document provides a comprehensive overview of the backend architecture, design decisions, and development guidelines for the Agentic CSV QA platform.

## 1. Architecture Overview

The backend is built with **FastAPI**, a modern, high-performance Python web framework. It is designed to be asynchronous from the ground up to handle long-running tasks like file processing and LLM-based query analysis without blocking the main application thread.

### 1.1. Technology Choices & Trade-offs

-   **Web Framework: FastAPI**
    -   **Why?**: FastAPI was chosen for its exceptional performance, asynchronous support (which is critical for our use case), automatic OpenAPI documentation, and Pydantic integration for robust data validation.
    -   **Trade-offs**: While powerful, its async nature requires careful management of database sessions and other I/O operations to avoid blocking the event loop.

-   **Database: SQLAlchemy (Sync) + SQLite/PostgreSQL**
    -   **Why Sync?**: For this project's scope, a synchronous SQLAlchemy setup is simpler to manage and debug than its async counterpart (`asyncio-sqlalchemy`). It provides a good balance of performance and development simplicity.
    -   **Why SQLite?**: It's used in development for its zero-configuration setup, making it easy for new developers to get started.
    -   **Production Path**: The application is designed to seamlessly switch to a more robust database like **PostgreSQL** for production environments with minimal code changes.

-   **LLM Integration: LangChain**
    -   **Why?**: LangChain provides a powerful and flexible framework for building applications with Large Language Models (LLMs). It simplifies the process of creating "agents" that can use tools to interact with their environment. Our `HybridCSVAgent` is a prime example of this, combining the reasoning power of an LLM with custom tools for secure data analysis.
    -   **Trade-offs**: LangChain can have a steep learning curve and sometimes adds a layer of abstraction that can be complex to debug. However, its benefits in structuring agentic workflows outweigh these challenges for our use case.

### 1.2. Core Components

-   **`main.py`**: The entry point of the FastAPI application. It initializes the app, includes the API routers, and sets up middleware.
-   **`api/`**: This directory contains the API endpoints.
    -   `upload.py`: Handles file uploads, validation, and kicks off the background processing task.
    -   `query.py`: Manages query requests, session history, and initiates the query analysis task.
-   **`services/`**: This is where the core business logic resides.
    -   `csv_processor.py`: A robust service for validating CSV files and extracting detailed structural metadata. This metadata is vital for the LLM to understand the context of the data it's working with.
    -   `hybrid_agent.py`: The heart of the application. This service uses LangChain to create an agent that can understand natural language queries, generate Python code, and execute it in a secure environment.
    -   `sandbox_executor.py` & `security_validator.py`: These components work together to create a secure sandbox for executing the LLM-generated Python code, preventing any malicious or unintended actions.
-   **`core/`**: Contains core application settings and database configuration.
-   **`models.py`**: Defines the SQLAlchemy ORM models, providing a Python-native way to interact with the database.

## 2. Database Schema

The database is designed to track files, user sessions, individual queries, and background jobs.

```python
# In app/models.py

class File(Base):
    """Tracks uploaded CSV files, their status, and metadata."""
    id: str  # UUID
    filename: str
    original_filename: str
    file_size: int
    status: str  # "processing", "completed", "error"
    file_metadata: dict  # JSON blob with column info, stats, etc.
    created_at: datetime
    processed_at: datetime

class Session(Base):
    """Manages multi-turn conversation context."""
    id: str  # UUID
    file_id: str
    conversation_history: list  # Chronological list of user/assistant messages
    active_tables: dict  # References to data generated in the conversation
    analysis_context: dict # Current state of the analysis
    created_at: datetime
    updated_at: datetime

class Query(Base):
    """Tracks individual queries, their results, and performance."""
    id: str  # UUID
    session_id: str
    query_text: str
    result: dict  # The structured result (e.g., table data, plot spec)
    reasoning: str # LLM's reasoning process
    execution_plan: dict # Structured plan from the LLM
    llm_metadata: dict # LLM token usage, model info
    execution_time: float
    status: str  # "processing", "completed", "error"
    created_at: datetime

class Job(Base):
    """Monitors the status of background tasks."""
    id: str  # UUID
    job_type: str  # "file_upload", "query_processing"
    status: str  # "queued", "processing", "completed", "failed"
    progress: int  # 0-100
    result: dict
    error_message: str
    created_at: datetime
    completed_at: datetime
```

## 3. LangChain Agent Architecture

Our `HybridCSVAgent` is a sophisticated component that leverages LangChain to provide intelligent data analysis.

### 3.1. Agent Structure

The agent is built using `langchain_experimental.agents.create_pandas_dataframe_agent`. This is a specialized agent that is optimized for interacting with a pandas DataFrame. It is designed to:

1.  **Understand the DataFrame**: It inspects the DataFrame's columns and data types.
2.  **Reason and Plan**: Given a user's query, it determines the steps needed to answer it.
3.  **Generate Code**: It writes Python code (using the pandas library) to execute its plan.
4.  **Execute Code**: It uses the provided tools to run the code and get a result.

### 3.2. Custom Tools

The agent's true power comes from its custom tools, which allow us to control its capabilities and ensure security.

-   **`SecureExecutionTool`**: This is the primary tool. When the agent wants to run Python code, it must use this tool. The tool passes the code to our `SandboxExecutor`, which first validates it for safety with the `CodeSecurityValidator` before executing it. This is a critical security boundary.
-   **`SessionManagementTool`**: Allows the agent to access the conversation history, providing context for multi-turn dialogues.
-   **`SmartSamplingTool`**: If the agent needs to analyze a very large dataset, this tool can provide a smaller, representative sample to avoid overwhelming the LLM's context window.
-   **`DataQualityTool`**: Provides the agent with tools to perform on-the-fly data quality assessments.

## 4. Development Guidelines

### 4.1. Code Standards

-   **Type Hinting**: All function definitions must include type hints.
-   **Docstrings**: All modules, classes, and functions should have comprehensive docstrings explaining their purpose.
-   **Logging**: Use structured logging to provide clear, actionable information for debugging and monitoring.

### 4.2. Security Best Practices

-   **Input Validation**: All data coming from the user (file uploads, query text) is validated using FastAPI's Pydantic integration.
-   **Secure Code Execution**: All LLM-generated code is treated as untrusted and is executed only within the secure sandbox. The `SecurityValidator` explicitly forbids dangerous operations like file system access or network calls.
-   **Error Sanitization**: API error messages should be generic and not expose internal implementation details or stack traces.

### 4.3. Performance

-   **Asynchronous Operations**: All potentially blocking I/O operations are handled in background tasks to keep the API responsive.
-   **Database Queries**: Be mindful of database query performance. Use appropriate indexing on database models.
-   **LLM Context Management**: The `HybridCSVAgent` is designed to provide the LLM with only the necessary context (metadata, data samples) rather than entire datasets to optimize performance and cost.
