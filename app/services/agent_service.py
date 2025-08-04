"""
Agent Service for CSV Analysis.

This module provides the main agent service that handles natural language queries
and generates secure Python code for data analysis with multi-turn conversation support.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session as DBSession

from app.core.database import get_db
from app.models import Session as SessionModel
from app.services.code_generation_service import CodeGenerationService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CSVAnalysisAgent:
    """Main agent for CSV data analysis using code generation."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.code_generation_service = CodeGenerationService(self.llm_service)
        self._current_file_path = None
    
    def analyze_query(
        self, 
        query: str, 
        file_path: str, 
        session_id: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Analyze a natural language query and generate results.
        
        Args:
            query: Natural language query
            file_path: Path to CSV file
            session_id: Optional session ID for multi-turn conversations
            metadata: File metadata
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self._current_file_path = file_path
            
            # Load session context for multi-turn conversations
            session_context = self._load_session_context(session_id) if session_id else {}
            
            # Generate and execute code
            result = self.code_generation_service.generate_and_execute(
                query=query,
                data_path=file_path,
                session_context=session_context
            )
            
            # Update session with new context
            if session_id and result.success:
                self._update_session_context(session_id, query, result)
            
            # Format response
            response = self._format_response(result, session_id)
            
            logger.info(f"Analysis completed successfully for query: {query}")
            return response
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "result": f"Analysis failed: {str(e)}",
                "error": str(e),
                "execution_time": 0,
                "query": query,
                "file_path": file_path,
                "session_id": session_id
            }
    
    def _load_session_context(self, session_id: str) -> Dict[str, Any]:
        """Load session context for multi-turn conversations."""
        try:
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session:
                context = {
                    "conversation_history": session.conversation_history,
                    "active_tables": session.active_tables,
                    "analysis_context": session.analysis_context
                }
                logger.info(f"Loaded session context for {session_id}")
                return context
            else:
                logger.warning(f"Session {session_id} not found")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load session context: {e}")
            return {}
    
    def _update_session_context(self, session_id: str, query: str, result: Any):
        """Update session with new conversation context."""
        try:
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session:
                # Add to conversation history
                session.conversation_history.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Add AI response
                ai_response = self._format_ai_response(result)
                session.conversation_history.append({
                    "role": "assistant", 
                    "content": ai_response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Update analysis context
                if result.execution_result and result.execution_result.result:
                    execution_result = result.execution_result.result
                    if execution_result.get('type') == 'table':
                        # Store table reference
                        table_name = f"result_{len(session.active_tables) + 1}"
                        session.active_tables[table_name] = {
                            "data": execution_result.get('data', []),
                            "columns": execution_result.get('columns', []),
                            "metadata": execution_result.get('metadata', {}),
                            "created_at": datetime.utcnow().isoformat()
                        }
                
                session.updated_at = datetime.utcnow()
                db.commit()
                logger.info(f"Updated session context for {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
    
    def _format_ai_response(self, result: Any) -> str:
        """Format AI response for conversation history."""
        if not result.success:
            return f"Analysis failed: {result.error}"
        
        if result.execution_result and result.execution_result.result:
            execution_result = result.execution_result.result
            result_type = execution_result.get('type', 'text')
            
            if result_type == 'text':
                return execution_result.get('data', 'Analysis completed')
            elif result_type == 'table':
                data = execution_result.get('data', [])
                return f"Generated table with {len(data)} rows"
            elif result_type == 'plot':
                return "Generated visualization"
            else:
                return "Analysis completed"
        else:
            return "Analysis completed"
    
    def _format_response(self, result: Any, session_id: str = None) -> Dict[str, Any]:
        """Format the final response."""
        response = {
            "success": result.success,
            "query": "Query processed",  # Will be set by caller
            "file_path": self._current_file_path,
            "session_id": session_id,
            "generated_code": result.generated_code,
            "reasoning": result.reasoning,  # NEW: Include reasoning
            "execution_plan": self._extract_execution_plan(result),  # NEW: Include execution plan
            "llm_metadata": self._extract_llm_metadata(result),  # NEW: Include LLM metadata
            "execution_time": 0
        }
        
        if result.execution_result:
            response["execution_time"] = result.execution_result.execution_time
            response["memory_used"] = result.execution_result.memory_used
            response["stdout"] = result.execution_result.stdout
            response["stderr"] = result.execution_result.stderr
        
        if result.success and result.execution_result and result.execution_result.result:
            execution_result = result.execution_result.result
            # Preserve the full execution result structure for proper frontend handling
            response["result"] = execution_result.get('data', 'Analysis completed')
            response["result_type"] = execution_result.get('type', 'text')
            response["metadata"] = execution_result.get('metadata', {})
            # Also preserve the full execution result for advanced handling
            response["full_result"] = execution_result
        else:
            response["result"] = result.error or "Analysis failed"
            response["error"] = result.error
        
        return response
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the current file path being analyzed."""
        return self._current_file_path
    
    def _extract_execution_plan(self, result: Any) -> Dict[str, Any]:
        """Extract execution plan from the result."""
        execution_plan = {
            "analysis_type": "unknown",
            "operations": [],
            "data_sources": [],
            "output_format": "unknown"
        }
        
        if result.success and result.execution_result and result.execution_result.result:
            execution_result = result.execution_result.result
            result_type = execution_result.get('type', 'text')
            
            # Determine analysis type based on reasoning and result type
            reasoning = result.reasoning.lower() if result.reasoning else ""
            
            if "top" in reasoning and "group" in reasoning:
                execution_plan["analysis_type"] = "top_analysis"
                execution_plan["operations"] = ["group_by", "sum", "sort_descending", "limit"]
            elif "plot" in reasoning or "chart" in reasoning or result_type == "plot":
                execution_plan["analysis_type"] = "visualization"
                execution_plan["operations"] = ["group_by", "aggregate", "create_chart"]
            elif "average" in reasoning or "mean" in reasoning:
                execution_plan["analysis_type"] = "statistical"
                execution_plan["operations"] = ["calculate_mean"]
            elif "count" in reasoning:
                execution_plan["analysis_type"] = "counting"
                execution_plan["operations"] = ["count_records"]
            elif "unique" in reasoning or "distinct" in reasoning:
                execution_plan["analysis_type"] = "unique_values"
                execution_plan["operations"] = ["extract_unique_values"]
            else:
                execution_plan["analysis_type"] = "general_analysis"
                execution_plan["operations"] = ["load_data", "basic_analysis"]
            
            execution_plan["output_format"] = result_type
            execution_plan["data_sources"] = ["csv_file"]
        
        return execution_plan
    
    def _extract_llm_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract LLM metadata from the result."""
        metadata = {
            "model_used": "gpt-3.5-turbo",  # Updated model
            "reasoning_length": len(result.reasoning) if result.reasoning else 0,
            "code_length": len(result.generated_code) if result.generated_code else 0,
            "execution_success": result.success,
            "has_error": bool(result.error)
        }
        
        if result.execution_result:
            metadata["execution_time"] = result.execution_result.execution_time
            metadata["memory_used"] = result.execution_result.memory_used
        
        return metadata


# Global agent instance
_csv_agent = None


def get_csv_agent() -> CSVAnalysisAgent:
    """Get the global CSV analysis agent instance."""
    global _csv_agent
    if _csv_agent is None:
        _csv_agent = CSVAnalysisAgent()
    return _csv_agent 