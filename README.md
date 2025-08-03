# Agentic_CSV_QA
## Overview

An intelligent web application that enables users to upload large CSV files and interact with them through natural language queries. Built with production-grade architecture to handle real-world data complexity while maintaining fast response times.

## Core Features

- **File Upload API**: Handle large, messy CSV files with robust preprocessing
- **Query Processing API**: Multi-turn conversations with LLM-driven analysis
- **Background Job Processing**: Async file processing and query execution
- **Smart Data Sampling**: Handle 100 columns × 10k rows without overwhelming LLM context
- **Real-time Updates**: WebSocket progress tracking for long-running operations

## Technical Architecture

### Technology Stack

- **Backend**: FastAPI (async support, automatic documentation)
- **Frontend**: Single-page React app (components, hooks, real-time updates)
- **Real-time Communication**: WebSockets for live updates
- **Job Queue**: Redis + RQ (background processing)
- **Database**: SQLite (development) / PostgreSQL (production)
- **Data Processing**: Pandas + Polars (performance optimization)
- **LLM Integration**: LangChain + OpenAI API (agentic reasoning with custom tools)
- **Caching**: Redis (shared with job queue)

### System Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Web Interface  │    │   API Layer     │    │ Background Jobs │
│                 │    │                 │    │                 │
│ • File Upload   │───▶│ • Upload API    │───▶│ • CSV Process   │
│ • Chat UI       │    │ • Query API     │    │ • Data Clean    │
│ • Results View  │    │ • WebSocket     │    │ • Index Build   │
│ • Progress      │◄───│ • Status API    │◄───│ • Job Updates  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File System   │    │     Redis       │    │    Database     │
│                 │    │                 │    │                 │
│ • Raw CSV       │    │ • Job Queue     │    │ • Metadata      │
│ • Parquet Data  │    │ • Cache         │    │ • Results       │
│ • Temp Files    │    │ • Sessions      │    │ • Job Status    │
└─────────────────┘    └─────────────────┘    └─────────────────┘

```

## API Endpoints (Core Assignment)

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
    "size": 1048576,
    "estimated_rows": 10000,
    "estimated_columns": 95
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
WebSocket /ws/{session_id}       # Real-time updates

```

## Example Multi-Turn Query Flow

```
User: "create a table that shows average sales by region"
→ LLM analyzes data structure and creates aggregation
→ Returns: Table with regions and average sales

User: "now add the top 3 products for each region"
→ LLM references previous result and joins with product data
→ Returns: Enhanced table with top products per region

User: "merge this with the customer data table"
→ LLM identifies relationships and performs join operation
→ Returns: Combined table with sales, products, and customer info

```

### Upload Flow

1. **File Validation**: Size, format, basic structure checks
2. **Immediate Response**: Return job ID for tracking
3. **Background Processing**:
    - Encoding detection and conversion
    - CSV parsing with error recovery
    - Data type inference and validation
    - Statistical summary generation
    - Parquet conversion for performance
    - Metadata indexing

### Query Flow

1. **Intent Analysis**: Parse natural language with LLM
2. **Context Building**: Generate relevant data samples
3. **Operation Planning**: Break complex queries into steps
4. **Execution**: Run operations on optimized data
5. **Response Formatting**: Return structured results

## Handling Large CSV Files

### The Challenge

- **Scale**: 100 columns × 10,000 rows = 1M+ data points
- **LLM Limits**: Cannot fit entire dataset in context window
- **Performance**: Must maintain <1 second response times

### Solution Strategy

### Smart Data Sampling

```python
# Instead of sending full dataset to LLM
Original CSV (100MB) → Data Fingerprint (5KB)
                    ↓
┌─────────────────────────────────────┐
│ • Column schema (names, types)      │
│ • Statistical summaries             │
│ • Representative samples (50 rows)  │
│ • Relationship maps                 │
│ • Quality indicators               │
└─────────────────────────────────────┘

```

### Multi-Step Query Processing

```
User: "Create a table showing average sales by region"

LLM Planning:
1. Identify columns: 'sales', 'region'
2. Validate data types and handle nulls
3. Group by region
4. Calculate averages
5. Format output table

Execution Engine:
→ Validates against actual data
→ Runs pandas operations
→ Returns formatted results

```

## Real-World CSV Challenges

### Data Quality Issues

- **Missing Values**: Scattered nulls, empty strings, "N/A" variants
- **Type Inconsistency**: Numbers as strings, mixed date formats
- **Structural Problems**: Extra/missing columns, malformed rows
- **Encoding Issues**: UTF-8, Latin-1, mixed character sets
- **Delimiter Confusion**: Commas, semicolons, tabs, mixed usage

### Robust Parsing Strategy

```python
# Multi-pass parsing approach
Pass 1: Quick structure detection
    → Estimate rows, columns, delimiters
    → Detect encoding and basic format

Pass 2: Progressive parsing
    → Handle malformed rows gracefully
    → Infer data types with confidence scoring
    → Track and report quality issues

Pass 3: Optimization
    → Convert to efficient storage format
    → Build searchable indexes
    → Generate statistical summaries

```

## Multi-Turn Conversation Support

### Context Management

```python
Session State:
{
  "session_id": "uuid",
  "file_id": "uuid",
  "conversation_history": [
    {
      "query": "Show sales by region",
      "result_id": "uuid",
      "timestamp": "..."
    }
  ],
  "active_tables": {
    "sales_by_region": "result_uuid"
  }
}

```

### Example Conversation Flow

```
User: "Create a table showing average sales by region"
→ Agent creates sales_summary_table

User: "Now add the top 3 products for each region"
→ Agent references previous table
→ Joins with product data
→ Updates existing table

User: "Show only regions with sales > $10,000"
→ Agent filters previous result
→ Maintains conversation context

```

## Performance Optimization

### Response Time Targets

- **Simple queries**: <1 second
- **Complex operations**: <5 seconds with progress updates
- **File processing**: <30 seconds for 10k row files

### Optimization Strategies

### Caching Layers

```python
L1: In-memory results cache (recent queries)
L2: Redis cache (session data, summaries)
L3: Parquet files (optimized data format)
L4: Database metadata (column info, statistics)

```

### Async Processing

```python
# Non-blocking API responses
Upload → Immediate job ID → Background processing
Query → Immediate acknowledgment → Async execution
Large Results → Streaming response → Progressive loading

```

### Data Structure Optimization

```python
# Efficient storage formats
Raw CSV → Parquet (columnar, compressed)
Metadata → SQLite indexes (fast lookups)
Cache → Redis (in-memory speed)
Results → JSON streaming (progressive loading)

```

## Error Handling & Recovery

### File Processing Errors

- **Parse failures**: Fallback strategies, partial recovery
- **Memory issues**: Chunked processing, streaming
- **Type conflicts**: Best-effort conversion, quality reports
- **Encoding problems**: Auto-detection, manual override

### Query Processing Errors

- **Ambiguous queries**: Clarification prompts
- **Invalid operations**: Clear error messages
- **Timeout issues**: Partial results, resume capability
- **LLM failures**: Fallback to structured parsing

### Production Monitoring

```python
Metrics to Track:
- Query response times
- File processing success rates
- Error frequencies and types
- Memory and CPU usage
- API rate limits and costs

```

## Development Setup

### Prerequisites

```bash
Python 3.9+
Redis Server
OpenAI API Key

```

### Installation

```bash
git clone <repository>
cd csv-agent
pip install -r requirements.txt

```

### Configuration

```bash
# Environment variables
OPENAI_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///./data.db
UPLOAD_DIR=./uploads

```

### Running the Application

```bash
# Start Redis server
redis-server

# Start background worker
python jobs.py

# Start FastAPI server
uvicorn main:app --reload --port 8000

# Access web interface
open http://localhost:8000

```

## Project Structure

```
csv-agent/
├── main.py                    # FastAPI app with all routes
├── models.py                  # Database models (Files, Sessions, Queries, Jobs)
├── csv_processor.py           # Core CSV processing logic
├── llm_service.py             # OpenAI integration
├── jobs.py                    # Background job processing
├── database.py                # DB connection setup
├── static/
│   ├── index.html             # Single React app
│   └── app.js                 # All React components
├── uploads/                   # File storage
├── requirements.txt
└── README.md

```

## Core Implementation Focus

### What This Assignment Tests

- **CSV Processing Complexity**: Handling messy real-world data at scale
- **LLM Integration Skills**: Context management without overwhelming token limits
- **System Architecture**: Background jobs, caching, real-time updates
- **Code Quality**: Clean, maintainable, production-ready code

### Key Technical Challenges

1. **100 columns × 10k rows** - Cannot fit in LLM context, need smart sampling
2. **Multi-turn conversations** - Context preservation and state management
3. **1-second response target** - Async processing and caching strategies
4. **Real-world CSV mess** - Robust parsing with graceful error handling

## Deployment Considerations

### Local Development

```bash
# Single machine setup
SQLite + Redis + Local files
Simple process management
Development debugging tools

```

### Production Migration

```bash
# Scalable deployment
PostgreSQL + Redis Cluster
S3/MinIO object storage
Kubernetes/Docker orchestration
Monitoring and alerting

```

### Performance Tuning

```python
# Key optimization areas
Database connection pooling
Redis memory optimization
File system optimization
LLM API rate limiting
Background job scaling

```

## Security & Privacy

### Data Protection

- File encryption at rest
- Secure temporary file handling
- Session-based access control
- Automatic data cleanup

### API Security

- Request rate limiting
- Input validation and sanitization
- Error message sanitization
- Audit logging

## Future Enhancements

*Note: These are intentionally out of scope for the assignment*

### Advanced Features

- **Data Visualization**: Chart generation from query results
- **Statistical Analysis**: Advanced analytics and ML predictions
- **Multi-user Support**: Authentication and shared sessions

## Contributing

### Development Workflow

1. Fork repository and create feature branch
2. Implement changes with comprehensive tests
3. Update documentation for new features
4. Submit pull request with detailed description

### Code Standards

- Python PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- 90%+ test coverage
