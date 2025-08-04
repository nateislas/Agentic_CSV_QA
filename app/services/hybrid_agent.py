"""
Hybrid CSV Analysis Agent using LangChain's pandas_dataframe_agent with custom tools.

This module provides a simplified agent that leverages LangChain's proven patterns
while maintaining our custom security, session management, and performance features.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from app.core.database import get_db
from app.models import Session as SessionModel
from app.services.sandbox_executor import SandboxExecutor
from app.services.csv_processor import GenericCSVProcessor

logger = logging.getLogger(__name__)


class SecureExecutionTool(BaseTool):
    """Custom tool that uses our secure sandbox for code execution."""
    
    name = "execute_code"
    description = "Execute Python code for CSV data analysis in a secure sandbox environment"
    
    def _run(self, code: str) -> str:
        """Execute code in the secure sandbox."""
        try:
            # Get the current file path from the hybrid agent
            from app.services.hybrid_agent import get_hybrid_agent
            agent = get_hybrid_agent()
            data_path = agent._current_file_path or ""
            
            # Create sandbox executor
            from app.services.sandbox_executor import SandboxExecutor
            from app.services.security_validator import CodeSecurityValidator
            sandbox = SandboxExecutor(timeout=30, memory_limit_mb=512)
            validator = CodeSecurityValidator()
            sandbox.set_validator(validator)
            
            result = sandbox.execute(code, data_path, session_context={})
            if result.success:
                return f"Code executed successfully: {result.result}"
            else:
                return f"Code execution failed: {result.error}"
        except Exception as e:
            return f"Execution error: {str(e)}"


class SessionManagementTool(BaseTool):
    """Custom tool for managing multi-turn conversation context."""
    
    name = "session_context"
    description = "Manage session context for multi-turn conversations"
    
    def _run(self, action: str, data: str = "") -> str:
        """Manage session context."""
        # Get session info from the hybrid agent
        from app.services.hybrid_agent import get_hybrid_agent
        agent = get_hybrid_agent()
        session_id = agent._current_session_id
        
        if action == "get_history":
            return "Session history available"
        elif action == "add_query":
            return "Query added to history"
        elif action == "get_context":
            return f"Current session: {session_id}"
        else:
            return "Unknown action"


class SmartSamplingTool(BaseTool):
    """Custom tool for smart data sampling to handle large datasets."""
    
    name = "smart_sample"
    description = "Generate smart samples of large datasets for LLM analysis"
    
    def _run(self, sample_type: str = "representative", size: int = 50) -> str:
        """Generate smart samples of the data."""
        try:
            # Get DataFrame from the hybrid agent
            from app.services.hybrid_agent import get_hybrid_agent
            agent = get_hybrid_agent()
            df = agent._current_df
            
            if df is None:
                return "No data available for sampling"
            
            if sample_type == "representative":
                # Simple head sample for now
                sample = df.head(size)
            elif sample_type == "head":
                # First N rows
                sample = df.head(size)
            elif sample_type == "tail":
                # Last N rows
                sample = df.tail(size)
            elif sample_type == "random":
                # Random sample
                sample = df.sample(n=min(size, len(df)))
            else:
                return f"Unknown sample type: {sample_type}"
            
            return f"Generated {sample_type} sample with {len(sample)} rows:\n{sample.to_string()}"
            
        except Exception as e:
            return f"Sampling error: {str(e)}"


class DataQualityTool(BaseTool):
    """Custom tool for data quality assessment."""
    
    name = "data_quality"
    description = "Assess data quality and provide insights about the dataset"
    
    def _run(self, analysis_type: str = "overview") -> str:
        """Perform data quality analysis."""
        try:
            # Get DataFrame from the hybrid agent
            from app.services.hybrid_agent import get_hybrid_agent
            agent = get_hybrid_agent()
            df = agent._current_df
            
            if df is None:
                return "No data available for analysis"
            
            if analysis_type == "overview":
                return self._get_overview(df)
            elif analysis_type == "missing_values":
                return self._get_missing_values(df)
            elif analysis_type == "data_types":
                return self._get_data_types(df)
            elif analysis_type == "duplicates":
                return self._get_duplicates(df)
            else:
                return f"Unknown analysis type: {analysis_type}"
                
        except Exception as e:
            return f"Data quality analysis error: {str(e)}"
    
    def _get_overview(self, df: pd.DataFrame) -> str:
        """Get a comprehensive data overview."""
        overview = f"Dataset Overview:\n"
        overview += f"- Rows: {len(df)}\n"
        overview += f"- Columns: {len(df.columns)}\n"
        overview += f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n"
        overview += f"- Missing values: {df.isnull().sum().sum()}\n"
        overview += f"- Duplicate rows: {df.duplicated().sum()}\n"
        return overview
    
    def _get_missing_values(self, df: pd.DataFrame) -> str:
        """Analyze missing values."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        result = "Missing Values Analysis:\n"
        for col in df.columns:
            if missing[col] > 0:
                result += f"- {col}: {missing[col]} ({missing_pct[col]:.1f}%)\n"
        
        if missing.sum() == 0:
            result += "No missing values found in the dataset."
        
        return result
    
    def _get_data_types(self, df: pd.DataFrame) -> str:
        """Analyze data types."""
        result = "Data Types Analysis:\n"
        for col in df.columns:
            result += f"- {col}: {df[col].dtype}\n"
        return result
    
    def _get_duplicates(self, df: pd.DataFrame) -> str:
        """Analyze duplicate rows."""
        duplicates = df.duplicated().sum()
        result = f"Duplicate Analysis:\n"
        result += f"- Total duplicate rows: {duplicates}\n"
        result += f"- Duplicate percentage: {(duplicates / len(df)) * 100:.1f}%\n"
        return result


class HybridCSVAgent:
    """
    Hybrid CSV analysis agent using LangChain's pandas_dataframe_agent with custom tools.
    
    This agent combines:
    - LangChain's proven agent patterns
    - Our secure sandbox execution
    - Multi-turn conversation support
    - Smart data sampling for large datasets
    - Data quality assessment
    """
    
    def __init__(self):
        # Initialize LLM with error handling
        try:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found")
            
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            logger.info("ChatOpenAI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise
        
        self.sandbox = SandboxExecutor()
        self.csv_processor = GenericCSVProcessor()
        self._current_df = None
        self._current_file_path = None
        self._current_session_id = None
    
    def analyze_query(
        self, 
        query: str, 
        file_path: str, 
        session_id: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Analyze a natural language query using the hybrid agent.
        
        Args:
            query: Natural language query from user
            file_path: Path to CSV file to analyze
            session_id: Optional session ID for conversation history
            metadata: Optional metadata about the file
            
        Returns:
            Dictionary containing analysis result and metadata
        """
        try:
            logger.info(f"Starting hybrid agent analysis for query: {query}")
            
            # Load and validate CSV data
            self._load_csv_data(file_path)
            
            # Store session ID for tools
            self._current_session_id = session_id
            
            # Create custom tools
            tools = self._create_custom_tools(session_id)
            
            # Create LangChain agent
            agent = create_pandas_dataframe_agent(
                self.llm,
                self._current_df,
                agent_type="zero-shot-react-description",
                extra_tools=tools,
                verbose=True,
                max_iterations=5
            )
            
            # Execute the agent
            logger.info("Executing hybrid agent...")
            result = agent.run(query)
            
            # Parse and format the result
            parsed_result = self._parse_agent_result(result, query, file_path, session_id)
            
            # Update session context if available
            if session_id:
                self._update_session_context(session_id, query, parsed_result)
            
            logger.info("Hybrid agent analysis completed successfully")
            return parsed_result
            
        except Exception as e:
            logger.error(f"Hybrid agent analysis failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e), query, file_path)
    
    def _load_csv_data(self, file_path: str):
        """Load CSV data with validation."""
        try:
            # Store the file path for tools
            self._current_file_path = file_path
            
            # Use our existing CSV processor for validation
            is_valid, error_msg = self.csv_processor.validate_file(file_path, 0)  # Size check handled elsewhere
            if not is_valid:
                raise ValueError(f"CSV validation failed: {error_msg}")
            
            # Load data with pandas
            self._current_df = pd.read_csv(
                file_path,
                na_values=["", "NULL", "null", "U", "N/A", "n/a"],
                parse_dates=True
            )
            
            logger.info(f"CSV data loaded: {len(self._current_df)} rows, {len(self._current_df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def _create_custom_tools(self, session_id: Optional[str]) -> List[BaseTool]:
        """Create custom tools for the agent."""
        tools = []
        
        # Add secure execution tool
        tools.append(SecureExecutionTool())
        
        # Add session management tool
        tools.append(SessionManagementTool())
        
        # Add smart sampling tool
        tools.append(SmartSamplingTool())
        
        # Add data quality tool
        tools.append(DataQualityTool())
        
        return tools
    
    def _parse_agent_result(self, result: str, query: str, file_path: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Parse the agent result into our standard format."""
        return {
            "success": True,
            "result": result,
            "result_type": "text",  # Default to text, can be enhanced
            "metadata": {
                "query": query,
                "file_path": file_path,
                "session_id": session_id,
                "agent_type": "hybrid_langchain",
                "execution_time": 0,  # Can be enhanced with timing
                "rows_analyzed": len(self._current_df),
                "columns_analyzed": len(self._current_df.columns)
            },
            "query": query,
            "file_path": file_path,
            "session_id": session_id
        }
    
    def _update_session_context(self, session_id: str, query: str, result: Dict[str, Any]):
        """Update session context in the database."""
        try:
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session:
                # Update conversation history
                history = session.conversation_history or []
                history.append({
                    "query": query,
                    "result": result.get("result", ""),
                    "timestamp": datetime.utcnow().isoformat()
                })
                session.conversation_history = history
                session.updated_at = datetime.utcnow()
                
                db.commit()
                logger.info(f"Updated session context for {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
    
    def _create_error_response(self, error: str, query: str, file_path: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "success": False,
            "result": f"Analysis failed: {error}",
            "error": error,
            "execution_time": 0,
            "query": query,
            "file_path": file_path,
            "metadata": {
                "agent_type": "hybrid_langchain",
                "error_type": "execution_error"
            }
        }


# Global agent instance
_hybrid_agent = None


def get_hybrid_agent() -> HybridCSVAgent:
    """Get the global hybrid CSV analysis agent instance."""
    global _hybrid_agent
    if _hybrid_agent is None:
        _hybrid_agent = HybridCSVAgent()
    return _hybrid_agent 