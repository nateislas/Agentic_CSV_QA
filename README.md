# Agentic_CSV_QA
## Overview

An intelligent web application that enables users to upload large CSV files and interact with them through natural language queries. Built with production-grade architecture to handle real-world data complexity while maintaining fast response times.

## Core Features

- **File Upload API**: Handle large, messy CSV files with robust preprocessing.
- **Query Processing API**: Multi-turn conversations with LLM-driven analysis.
- **Background Job Processing**: Asynchronous file processing and query execution.
- **Secure Code Execution**: LLM-generated code is executed in a secure sandbox.
- **Conversation History**: Supports multi-turn conversations through session management.

## Technical Architecture

### Technology Stack

- **Backend**: FastAPI (async support, automatic documentation)
- **Data Processing**: Pandas
- **LLM Integration**: LangChain + OpenAI API (agentic reasoning with custom tools)
- **Database**: SQLite (development) / PostgreSQL (production) with SQLAlchemy ORM.

### System Design

The system is designed around a FastAPI backend that handles two main user flows: uploading a CSV file and querying it. Both operations are handled asynchronously using background tasks to prevent blocking the API.

1.  **File Upload**: The user uploads a CSV file, which is validated and saved. A background task is triggered to process the file using the `csv_processor` service. This service extracts structural metadata and saves it to the database.
2.  **Query Processing**: The user submits a natural language query. A background task calls the `hybrid_agent` service, which uses a LangChain agent powered by an OpenAI LLM. The agent generates Python (pandas) code to answer the query, which is then executed in a secure sandbox. Conversation history is maintained for multi-turn dialogue.

### User Flow Examples

The following screenshots demonstrate the agentic CSV analysis system in action:

![Step 1: File Upload and Processing](figures/Screenshot%202025-08-04%20at%2010.21.36%20PM.png)

![Step 2: Initial Query Processing](figures/Screenshot%202025-08-04%20at%2010.21.50%20PM.png)

![Step 3: Multi-turn Conversation](figures/Screenshot%202025-08-04%20at%2010.22.03%20PM.png)

![Step 4: Complex Data Analysis](figures/Screenshot%202025-08-04%20at%2010.22.34%20PM.png)

![Step 5: Results and Insights](figures/Screenshot%202025-08-04%20at%2010.23.03%20PM.png)

![Step 6: Filtering by Category](figures/Screenshot%202025-08-04%20at%2010.26.53%20PM.png)

## API Endpoints

### 1. File Upload Endpoint

```
POST /api/upload
Content-Type: multipart/form-data

Response:
{
  "job_id": "uuid",
  "status": "processing",
  "file_info": {
    "filename": "data.csv",
    "size": 1048576
  }
}
```

### 2. LLM Query Endpoint

```
POST /api/query
Content-Type: application/json

Request:
{
  "file_id": "uuid",
  "query": "create a table that shows average sales by region",
  "session_id": "uuid" // for multi-turn conversations
}

Response:
{
  "query_id": "uuid",
  "status": "completed",
  "result": {
    "type": "table",
    "data": [...],
    "metadata": {...}
  },
  "execution_time": 0.8
}
```

### Supporting Endpoints

```
GET /api/status/{job_id}        # Check file processing status
GET /api/session/{session_id}     # Get session history
```

## Development Setup

### Prerequisites

```bash
Python 3.9+
OpenAI API Key
```

### Installation

```bash
git clone <repository>
cd Agentic_CSV_QA
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root and add your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=your_key_here
```

### Running the Application

```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

The application will be available at `http://localhost:8000`.

## Project Structure

```
app/
├── api/
│   ├── upload.py           # Handles file uploads and status checks
│   └── query.py            # Handles query processing and session management
├── core/
│   ├── config.py           # Application configuration
│   └── database.py         # Database connection and session management
├── services/
│   ├── csv_processor.py    # CSV validation and metadata extraction
│   ├── hybrid_agent.py     # Core agent logic with LangChain
│   ├── sandbox_executor.py # Secure code execution environment
│   └── security_validator.py # Validates LLM-generated code
├── main.py                 # FastAPI application entry point
└── models.py               # SQLAlchemy database models
```
