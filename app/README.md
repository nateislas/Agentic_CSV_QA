# Backend Development Guide

## Architecture Overview

### Technology Stack Decisions

**Database: SQLAlchemy (Sync) + SQLite**
- ✅ **Why Sync SQLAlchemy?** Simpler for this project scope, easier debugging
- ✅ **Why SQLite?** Zero setup, file-based, perfect for development
- ✅ **Migration Path:** Easy upgrade to PostgreSQL for production

**Async Components:**
- ✅ **File Uploads:** Large file handling with progress tracking
- ✅ **WebSocket Connections:** Real-time updates for job status
- ✅ **Background Jobs:** Non-blocking CSV processing

**LangChain Integration:**
- ✅ **Full Agent Implementation:** Custom tools for CSV operations
- ✅ **Multi-turn Conversations:** Session-based context management
- ✅ **Smart Data Sampling:** Handle large datasets without overwhelming LLM

## Development Roadmap

### Phase 1: Foundation (Week 1)
**Goal:** Establish core infrastructure and data models

#### 1.1 Database Setup
- [ ] `app/core/database.py` - SQLAlchemy connection and session management
- [ ] `app/core/config.py` - Environment configuration and settings
- [ ] `app/models.py` - Database models (File, Session, Query, Job)

#### 1.2 Core Services
- [ ] `app/services/csv_processor.py` - CSV parsing and validation
- [ ] `app/services/llm_service.py` - OpenAI integration wrapper

### Phase 2: LangChain Agent (Week 1-2)
**Goal:** Implement intelligent CSV analysis agent

#### 2.1 Agent Architecture
- [ ] `app/services/agent_service.py` - Main LangChain agent setup
- [ ] `app/services/tools/` - Custom tools for CSV operations
  - [ ] `data_sampling_tool.py` - Smart data sampling
  - [ ] `pandas_tool.py` - Pandas operations wrapper
  - [ ] `statistics_tool.py` - Statistical analysis tools

#### 2.2 Agent Tools
- [ ] Data exploration tools (column info, sample data)
- [ ] Aggregation tools (group by, pivot tables)
- [ ] Filtering tools (where clauses, data filtering)
- [ ] Visualization tools (basic charts, summaries)

### Phase 3: API Layer (Week 2)
**Goal:** RESTful API endpoints with proper error handling

#### 3.1 Core Endpoints
- [ ] `app/api/upload.py` - File upload with validation
- [ ] `app/api/query.py` - Natural language query processing
- [ ] `app/api/status.py` - Job status tracking

#### 3.2 FastAPI App
- [ ] `app/main.py` - Main FastAPI application
- [ ] CORS configuration for frontend
- [ ] Error handling middleware
- [ ] Request/response logging

### Phase 4: Real-time Features (Week 2-3)
**Goal:** WebSocket integration and background processing

#### 4.1 Background Jobs
- [ ] `app/jobs.py` - Redis + RQ job processing
- [ ] CSV processing workers
- [ ] Job status tracking

#### 4.2 WebSocket Integration
- [ ] `app/core/websocket.py` - WebSocket connection manager
- [ ] Real-time job progress updates
- [ ] Session-based message handling

### Phase 5: Production Features (Week 3)
**Goal:** Production-ready features and optimizations

#### 5.1 Performance & Security
- [ ] Rate limiting
- [ ] File size limits
- [ ] Input validation and sanitization
- [ ] Error logging and monitoring

#### 5.2 Testing & Documentation
- [ ] Unit tests for all services
- [ ] Integration tests for API endpoints
- [ ] API documentation with OpenAPI
- [ ] Deployment guide

## Database Schema Design

### Core Models

```python
# File Management
class File(Base):
    id: str = Column(String, primary_key=True)  # UUID
    filename: str = Column(String, nullable=False)
    original_filename: str = Column(String, nullable=False)
    file_size: int = Column(BigInteger, nullable=False)
    status: str = Column(String, default="processing")  # processing, completed, error
    metadata: dict = Column(JSON)  # Column info, data types, statistics
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    processed_at: datetime = Column(DateTime, nullable=True)

# Session Management
class Session(Base):
    id: str = Column(String, primary_key=True)  # UUID
    file_id: str = Column(String, ForeignKey("files.id"))
    conversation_history: list = Column(JSON, default=list)
    active_tables: dict = Column(JSON, default=dict)  # Reference to previous results
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Query Tracking
class Query(Base):
    id: str = Column(String, primary_key=True)  # UUID
    session_id: str = Column(String, ForeignKey("sessions.id"))
    query_text: str = Column(Text, nullable=False)
    result: dict = Column(JSON, nullable=True)
    execution_time: float = Column(Float, nullable=True)
    status: str = Column(String, default="processing")  # processing, completed, error
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

# Job Tracking
class Job(Base):
    id: str = Column(String, primary_key=True)  # UUID
    job_type: str = Column(String, nullable=False)  # file_upload, query_processing
    status: str = Column(String, default="queued")  # queued, processing, completed, failed
    progress: int = Column(Integer, default=0)  # 0-100
    result: dict = Column(JSON, nullable=True)
    error_message: str = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    completed_at: datetime = Column(DateTime, nullable=True)
```

## LangChain Agent Architecture

### Agent Structure

```python
# Three-tier agent approach:

1. Query Understanding Agent
   - Parse natural language into structured operations
   - Identify data columns and operations needed
   - Generate execution plan

2. Data Analysis Agent  
   - Execute pandas operations
   - Handle data sampling for large datasets
   - Perform statistical analysis

3. Response Formatting Agent
   - Structure results for frontend display
   - Generate natural language summaries
   - Format tables and visualizations
```

### Custom Tools

```python
# Core CSV Analysis Tools:

1. DataSamplingTool
   - Smart sampling for large datasets
   - Maintain data distribution
   - Handle 100+ columns efficiently

2. PandasAnalysisTool
   - Group by operations
   - Aggregations (sum, mean, count)
   - Filtering and sorting
   - Pivot tables

3. StatisticsTool
   - Descriptive statistics
   - Correlation analysis
   - Data quality assessment

4. VisualizationTool
   - Basic chart generation
   - Table formatting
   - Summary statistics
```

## API Endpoint Design

### File Upload Flow

```python
POST /api/upload
Request: multipart/form-data
Response: {
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

### Query Processing Flow

```python
POST /api/query
Request: {
  "file_id": "uuid",
  "query": "create a table showing average sales by region",
  "session_id": "uuid"
}
Response: {
  "query_id": "uuid",
  "status": "completed",
  "result": {
    "type": "table",
    "data": {...},
    "metadata": {...}
  },
  "execution_time": 0.8
}
```

### Job Status Tracking

```python
GET /api/status/{job_id}
Response: {
  "job_id": "uuid",
  "status": "processing",
  "progress": 75,
  "result": {...},
  "error": null
}
```

## WebSocket Events

### Event Types

```python
# File Processing Events
file_progress: {"job_id": "uuid", "progress": 50, "message": "Processing CSV..."}
file_complete: {"job_id": "uuid", "file_id": "uuid", "metadata": {...}}
file_error: {"job_id": "uuid", "error": "Invalid CSV format"}

# Query Processing Events  
query_progress: {"query_id": "uuid", "message": "Analyzing data..."}
query_complete: {"query_id": "uuid", "result": {...}}
query_error: {"query_id": "uuid", "error": "Column not found"}
```

## Development Guidelines

### Code Standards

1. **Type Hints**: All functions must have type hints
2. **Docstrings**: Comprehensive docstrings for all classes and methods
3. **Error Handling**: Proper exception handling with meaningful messages
4. **Logging**: Structured logging for debugging and monitoring
5. **Testing**: Unit tests for all business logic

### Performance Considerations

1. **Data Sampling**: Never send full dataset to LLM
2. **Caching**: Cache processed file metadata and common queries
3. **Async Operations**: Use async for I/O operations (file uploads, API calls)
4. **Memory Management**: Process large files in chunks
5. **Database Optimization**: Proper indexing and query optimization

### Security Measures

1. **File Validation**: Strict CSV format validation
2. **Size Limits**: Maximum file size restrictions
3. **Input Sanitization**: Clean all user inputs
4. **Rate Limiting**: Prevent abuse
5. **Error Sanitization**: Don't expose internal errors

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual service methods
2. **Integration Tests**: API endpoint testing
3. **End-to-End Tests**: Full user workflows
4. **Performance Tests**: Large file processing
5. **Security Tests**: Input validation and sanitization

### Test Data

- Small CSV files (10-100 rows) for unit tests
- Medium CSV files (1k-10k rows) for integration tests
- Large CSV files (100k+ rows) for performance tests
- Malformed CSV files for error handling tests

## Deployment Considerations

### Development Environment

```bash
# Local setup
SQLite database
Redis for job queue
Local file storage
Hot reload for development
```

### Production Environment

```bash
# Production setup
PostgreSQL database
Redis cluster
S3/MinIO object storage
Docker containerization
Kubernetes orchestration
```

## Next Steps

1. **Start with Phase 1**: Database setup and core models
2. **Implement incrementally**: Each phase builds on the previous
3. **Test thoroughly**: Write tests as you build
4. **Document as you go**: Keep this README updated
5. **Iterate based on feedback**: Adjust architecture as needed

## Success Metrics

- [ ] Handle 100 columns × 10k rows efficiently
- [ ] <1 second response time for simple queries
- [ ] <30 seconds for file processing
- [ ] 99%+ uptime for API endpoints
- [ ] Comprehensive test coverage (>90%)
- [ ] Production-ready security and performance 