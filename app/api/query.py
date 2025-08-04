from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
from datetime import datetime

from app.core.database import get_db
from app.models import Query as QueryModel, Session as SessionModel, File as FileModel
from app.services.langchain_agent import LangChainCSVAgent
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    file_id: str
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    query_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
async def process_query(
    background_tasks: BackgroundTasks,
    request: QueryRequest
):
    """
    Process a natural language query against a CSV file
    
    Args:
        request: Query request with file_id and query text
        
    Returns:
        Query processing result with structured data
    """
    try:
        # Validate file exists and is processed
        db = next(get_db())
        file_record = db.query(FileModel).filter(FileModel.id == request.file_id).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
            
        if file_record.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail="File is not ready for querying. Status: " + file_record.status
            )
        
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Try to find existing session, create new one if not found
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            session = SessionModel(
                id=session_id,
                file_id=request.file_id,
                conversation_history=[],
                active_tables={},
                analysis_context={},
                created_at=datetime.utcnow()
            )
            db.add(session)
        
        # Create query record
        query_id = str(uuid.uuid4())
        query_record = QueryModel(
            id=query_id,
            session_id=session_id,
            query_text=request.query,
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(query_record)
        db.commit()
        db.close()
        
        # Start background processing
        background_tasks.add_task(
            process_query_async,
            query_id=query_id,
            file_id=request.file_id,
            session_id=session_id,
            query_text=request.query
        )
        
        logger.info(f"Query processing started: {query_id} -> {request.query[:50]}...")
        
        return QueryResponse(
            query_id=query_id,
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )

async def process_query_async(
    query_id: str,
    file_id: str,
    session_id: str,
    query_text: str
):
    """
    Background task to process natural language query
    
    Args:
        query_id: Database query ID
        file_id: Database file ID
        session_id: Database session ID
        query_text: Natural language query
    """
    db = next(get_db())
    start_time = datetime.utcnow()
    
    try:
        # Get file and session data
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        
        if not file_record or not session:
            raise Exception("File or session not found")
        
        # Initialize LangChain agent service
        logger.info(f"Getting LangChain CSV agent for query: {query_text[:50]}...")
        agent_service = LangChainCSVAgent()
        logger.info("LangChain CSV agent retrieved successfully")
        
        # Process query with LangChain agent
        logger.info(f"Starting LangChain agent analysis for query: {query_text}")
        result = agent_service.analyze_query(
            query=query_text,
            file_path=file_record.filename,
            session_id=session_id
        )
        logger.info(f"LangChain agent analysis completed, result success: {result.get('success', False)}")
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update query record
        query_record = db.query(QueryModel).filter(QueryModel.id == query_id).first()
        if query_record:
            query_record.status = "completed" if result.get("success") else "error"
            query_record.result = result
            query_record.reasoning = result.get("reasoning", "")  # NEW: Store reasoning
            query_record.execution_plan = result.get("execution_plan", {})  # NEW: Store execution plan
            query_record.llm_metadata = result.get("llm_metadata", {})  # NEW: Store LLM metadata
            query_record.execution_time = result.get("execution_time", execution_time)
        
        # Note: Conversation history is now managed by the agent service
        # The session is updated automatically by the agent service
        
        # Update active tables if result contains new tables
        if result.get("type") == "table" and result.get("table_id"):
            session.active_tables[result["table_id"]] = {
                "name": result.get("table_name", f"Table_{result['table_id']}"),
                "data": result.get("data", {}),
                "created_at": datetime.utcnow().isoformat()
            }
        
        db.commit()
        logger.info(f"Query processing completed: {query_id} in {execution_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        
        # Update error status
        query_record = db.query(QueryModel).filter(QueryModel.id == query_id).first()
        if query_record:
            query_record.status = "error"
            query_record.result = {"error": str(e)}
        
        db.commit()
        
    finally:
        db.close()

@router.get("/query/{query_id}")
async def get_query_result(query_id: str):
    """
    Get the result of a processed query
    
    Args:
        query_id: Query ID to retrieve
        
    Returns:
        Query result and metadata
    """
    try:
        db = next(get_db())
        query_record = db.query(QueryModel).filter(QueryModel.id == query_id).first()
        
        if not query_record:
            raise HTTPException(status_code=404, detail="Query not found")
        
        response = {
            "query_id": query_id,
            "status": query_record.status,
            "query_text": query_record.query_text,
            "created_at": query_record.created_at.isoformat(),
            "execution_time": query_record.execution_time
        }
        
        if query_record.result:
            response["result"] = query_record.result
            
        if query_record.status == "error":
            response["error"] = "Query processing failed"
            
        db.close()
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query result retrieval error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get query result"
        )

@router.get("/query/{query_id}/debug")
async def get_query_debug_info(query_id: str):
    """
    Get debugging information including LLM reasoning (admin only).
    
    Args:
        query_id: Query ID to retrieve debug info for
        
    Returns:
        Query debug information including reasoning, execution plan, and metadata
    """
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

@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get session information and conversation history
    
    Args:
        session_id: Session ID to retrieve
        
    Returns:
        Session information and conversation history
    """
    try:
        db = next(get_db())
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get file information
        file_record = db.query(FileModel).filter(FileModel.id == session.file_id).first()
        
        response = {
            "session_id": session_id,
            "file_id": session.file_id,
            "file_name": file_record.original_filename if file_record else "Unknown",
            "conversation_history": session.conversation_history,
            "active_tables": session.active_tables,
            "analysis_context": session.analysis_context,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
        
        db.close()
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get session information"
        )

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its conversation history
    
    Args:
        session_id: Session ID to delete
    """
    try:
        db = next(get_db())
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete associated queries
        db.query(QueryModel).filter(QueryModel.session_id == session_id).delete()
        
        # Delete session
        db.delete(session)
        db.commit()
        db.close()
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session"
        )
