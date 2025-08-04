# LLM Reasoning Capture Implementation Plan

## Overview

This document outlines the implementation plan for adding LLM reasoning capture to the CSV analysis backend. The goal is to capture the LLM's thought process and planning steps for debugging and quality control purposes, without exposing this information to end users.

## Current System Analysis

### What We Have
- ✅ **Query Processing**: `agent_service.py` handles natural language queries
- ✅ **Code Generation**: `llm_service.py` generates Python code for data analysis
- ✅ **Execution**: `sandbox_executor.py` runs code in secure environment
- ✅ **Response Formatting**: Results are formatted and returned to frontend
- ❌ **Missing**: Explicit reasoning capture and planning steps

### What We Need
- 🔧 **Reasoning Capture**: LLM's understanding and planning process
- 🔧 **Structured Planning**: Step-by-step execution plans
- 🔧 **Debug Information**: Backend access to reasoning for troubleshooting
- 🔧 **Quality Control**: Ability to review and improve LLM performance

## Modern AI Agent Architecture

### Industry Standard Approaches

#### 1. **Chain of Thought (CoT) Pattern**
```
User Query → Understanding → Reasoning → Planning → Execution → Response
```

#### 2. **ReAct Pattern (Reasoning + Acting)**
```
Loop until completion:
1. Think: Analyze current state and plan next action
2. Act: Execute the planned action  
3. Observe: Check results and update state
4. Repeat
```

#### 3. **Three-Tier Agent Architecture**
```
1. Query Understanding Agent
   - Parse intent and extract entities
   - Identify required data sources
   - Generate execution plan

2. Data Analysis Agent
   - Execute operations (pandas, SQL, etc.)
   - Handle data sampling for large datasets
   - Perform statistical analysis

3. Response Formatting Agent
   - Structure results for display
   - Handle different output formats
   - Add metadata and explanations
```

## Implementation Plan

### Phase 1: LLM Prompt Engineering

#### 1.1 Update LLM Service Prompt
**File**: `app/services/llm_service.py`

**Current Prompt Structure**:
```python
system_message = """
You are a Python code generator for data analysis. Generate clean, efficient Python code that:
1. Loads the data first: df = pd.read_csv(data_path)
2. Uses pandas for data manipulation (pd is already available)
3. Handles the data safely and efficiently
4. Returns results using create_result() function
5. Focuses on the specific analysis requested
"""
```

**New Prompt Structure**:
```python
system_message = """
You are a data analysis agent. For each query, follow this structure:

1. REASONING: Explain your understanding of the query and your plan
2. CODE: Generate the Python code to execute your plan

Example:
# REASONING:
# The user wants the top borough by crime count. I need to:
# - Group by 'borough' column
# - Sum the 'value' column (crime counts)
# - Sort in descending order
# - Return the top result

# CODE:
df = pd.read_csv(data_path)
...
"""
```

#### 1.2 Add Reasoning Extraction
**File**: `app/services/code_generation_service.py`

**New Method**:
```python
def _extract_reasoning_and_code(self, llm_response: str) -> Tuple[str, str]:
    """Extract reasoning and code from LLM response."""
    reasoning = ""
    code = llm_response
    
    # Look for reasoning section
    if "# REASONING:" in llm_response:
        parts = llm_response.split("# CODE:", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("# REASONING:", "").strip()
            code = parts[1].strip()
    
    return reasoning, code
```

### Phase 2: Database Schema Updates

#### 2.1 Update Query Model
**File**: `app/models.py`

**Add New Fields**:
```python
class Query(Base):
    # ... existing fields ...
    reasoning: str = Column(Text, nullable=True)  # LLM reasoning
    execution_plan: dict = Column(JSON, nullable=True)  # Structured plan
    llm_metadata: dict = Column(JSON, nullable=True)  # Additional LLM info
```

#### 2.2 Database Migration
**File**: `alembic/versions/xxx_add_reasoning_fields.py`

```python
"""Add reasoning fields to queries table

Revision ID: xxx
Revises: previous_revision
Create Date: 2024-01-XX

"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('queries', sa.Column('reasoning', sa.Text(), nullable=True))
    op.add_column('queries', sa.Column('execution_plan', sa.JSON(), nullable=True))
    op.add_column('queries', sa.Column('llm_metadata', sa.JSON(), nullable=True))

def downgrade():
    op.drop_column('queries', 'reasoning')
    op.drop_column('queries', 'execution_plan')
    op.drop_column('queries', 'llm_metadata')
```

### Phase 3: Backend Service Updates

#### 3.1 Update Code Generation Service
**File**: `app/services/code_generation_service.py`

**Modify `generate_and_execute` method**:
```python
def generate_and_execute(self, query: str, data_path: str, session_context: Dict = None) -> CodeGenerationResult:
    try:
        # Extract data context
        data_context = self._extract_data_context(data_path)
        
        # Generate code with reasoning
        generated_response = self._generate_code(query, session_context, data_context)
        
        if not generated_response:
            return CodeGenerationResult(
                success=False,
                generated_code="",
                reasoning="Failed to generate response",
                execution_result=None,
                error="Failed to generate code",
                context_used={}
            )
        
        # Extract reasoning and code
        reasoning, generated_code = self._extract_reasoning_and_code(generated_response)
        
        # Execute code
        execution_result = self.executor.execute(generated_code, data_path, session_context)
        
        return CodeGenerationResult(
            success=execution_result.success,
            generated_code=generated_code,
            reasoning=reasoning,
            execution_result=execution_result,
            error=execution_result.error if not execution_result.success else None,
            context_used=session_context or {}
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return CodeGenerationResult(
            success=False,
            generated_code="",
            reasoning="",
            execution_result=None,
            error=f"Code generation failed: {str(e)}",
            context_used=session_context or {}
        )
```

#### 3.2 Update CodeGenerationResult Dataclass
**File**: `app/services/code_generation_service.py`

```python
@dataclass
class CodeGenerationResult:
    """Result of code generation and execution."""
    success: bool
    generated_code: str
    reasoning: str  # NEW: LLM reasoning
    execution_result: Optional[ExecutionResult]
    error: Optional[str]
    context_used: Dict[str, Any]
```

### Phase 4: API Updates

#### 4.1 Update Query Processing
**File**: `app/api/query.py`

**Modify `process_query_async`**:
```python
async def process_query_async(query_id: str, file_id: str, session_id: str, query_text: str):
    # ... existing code ...
    
    # Process query with agent
    result = agent_service.analyze_query(
        query=query_text,
        file_path=file_record.filename,
        session_id=session_id
    )
    
    # Update query record with reasoning
    query_record = db.query(QueryModel).filter(QueryModel.id == query_id).first()
    if query_record:
        query_record.status = "completed" if result.get("success") else "error"
        query_record.result = result
        query_record.reasoning = result.get("reasoning", "")  # NEW
        query_record.execution_plan = result.get("execution_plan", {})  # NEW
        query_record.execution_time = result.get("execution_time", execution_time)
```

#### 4.2 Add Debug Endpoint
**File**: `app/api/query.py`

```python
@router.get("/query/{query_id}/debug")
async def get_query_debug_info(query_id: str):
    """Get debugging information including LLM reasoning (admin only)."""
    try:
        db = next(get_db())
        query_record = db.query(QueryModel).filter(QueryModel.id == query_id).first()
        
        if not query_record:
            raise HTTPException(status_code=404, detail="Query not found")
        
        debug_info = {
            "query_id": query_id,
            "query_text": query_record.query_text,
            "status": query_record.status,
            "reasoning": query_record.reasoning,
            "execution_plan": query_record.execution_plan,
            "llm_metadata": query_record.llm_metadata,
            "execution_time": query_record.execution_time,
            "result": query_record.result,
            "created_at": query_record.created_at.isoformat()
        }
        
        db.close()
        return debug_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug info retrieval error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get debug information"
        )
```

### Phase 5: Agent Service Updates

#### 5.1 Update Agent Service Response
**File**: `app/services/agent_service.py`

**Modify `_format_response` method**:
```python
def _format_response(self, result: Any, session_id: str = None) -> Dict[str, Any]:
    """Format the final response."""
    response = {
        "success": result.success,
        "query": "Query processed",
        "file_path": self._current_file_path,
        "session_id": session_id,
        "generated_code": result.generated_code,
        "reasoning": result.reasoning,  # NEW: Include reasoning
        "execution_time": 0
    }
    
    # ... rest of existing code ...
    
    return response
```

## Logical Flow Example

### User Query: "What is the top borough with the most crime?"

#### 1. Query Understanding (LLM)
```
# REASONING:
# The user wants the single borough with the highest crime count.
# I need to:
# - Load the CSV data
# - Identify 'borough' as the categorical column for grouping
# - Identify 'value' as the numeric column for crime counts
# - Group by borough and sum the crime values
# - Sort in descending order
# - Return only the top 1 result (not a table)
```

#### 2. Planning (LLM)
```
# EXECUTION PLAN:
# Step 1: Load CSV data using pd.read_csv()
# Step 2: Group by 'borough' column
# Step 3: Sum the 'value' column (crime counts)
# Step 4: Sort values in descending order
# Step 5: Get the top 1 result
# Step 6: Return as single value, not table
```

#### 3. Code Generation (LLM)
```python
# CODE:
df = pd.read_csv(data_path)
borough_crime = df.groupby('borough')['value'].sum().sort_values(ascending=False)
top_borough = borough_crime.head(1)
result = create_result(
    data=top_borough.to_dict(),
    result_type="text",
    metadata={"query": "top_borough_crime", "analysis_type": "top_single"}
)
```

#### 4. Execution (Sandbox)
- Run the generated Python code
- Validate results
- Handle any errors

#### 5. Response (Agent Service)
- Format result for display
- Store reasoning in database for debugging
- Return clean response to user

## Benefits

### For Developers/Admins
- **Debugging**: See exactly what the LLM was thinking
- **Quality Control**: Review reasoning to improve prompts
- **Performance Analysis**: Track which queries work well vs. poorly
- **Prompt Engineering**: Iterate on prompts based on reasoning patterns

### For System Quality
- **Error Diagnosis**: Understand why queries fail
- **Consistency**: Ensure similar queries get similar reasoning
- **Transparency**: Full audit trail of LLM decisions
- **Improvement**: Data-driven prompt optimization

## Implementation Checklist

- [ ] **Phase 1**: Update LLM prompts to require reasoning
- [ ] **Phase 1**: Add reasoning extraction logic
- [ ] **Phase 2**: Update database schema with new fields
- [ ] **Phase 2**: Create and run database migration
- [ ] **Phase 3**: Update CodeGenerationResult dataclass
- [ ] **Phase 3**: Modify code generation service
- [ ] **Phase 4**: Update query processing to store reasoning
- [ ] **Phase 4**: Add debug endpoint for admin access
- [ ] **Phase 5**: Update agent service response format
- [ ] **Testing**: Verify reasoning capture works correctly
- [ ] **Documentation**: Update API docs for debug endpoint

## Security Considerations

- **Admin Only**: Debug endpoint should require admin authentication
- **No User Exposure**: Reasoning is never shown to end users
- **Data Privacy**: Ensure reasoning doesn't expose sensitive information
- **Logging**: Consider logging reasoning for audit purposes

## Future Enhancements

- **Reasoning Quality Metrics**: Score reasoning quality
- **Automated Prompt Improvement**: Use reasoning data to optimize prompts
- **Query Pattern Analysis**: Identify common reasoning patterns
- **Performance Optimization**: Use reasoning to cache similar queries 