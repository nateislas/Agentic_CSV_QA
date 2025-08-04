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
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

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
        
        if not session_id:
            return "No active session"
        
        try:
            if action == "get_history":
                # Get conversation history from database
                db = next(get_db())
                session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                
                if session and session.conversation_history:
                    history_summary = []
                    for entry in session.conversation_history[-5:]:  # Last 5 entries
                        if isinstance(entry, dict):
                            if "query" in entry:
                                history_summary.append(f"User: {entry['query'][:100]}...")
                            if "result" in entry:
                                history_summary.append(f"Assistant: {str(entry['result'])[:100]}...")
                    
                    return f"Recent conversation history:\n" + "\n".join(history_summary)
                else:
                    return "No conversation history available"
                    
            elif action == "get_context":
                # Get session context
                db = next(get_db())
                session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                
                if session:
                    context_info = []
                    if session.active_tables:
                        context_info.append(f"Active tables: {len(session.active_tables)}")
                    if session.analysis_context:
                        context_info.append(f"Analysis context: {session.analysis_context}")
                    
                    return f"Session {session_id} context:\n" + "\n".join(context_info) if context_info else f"Session {session_id} (no additional context)"
                else:
                    return f"Session {session_id} not found"
                    
            elif action == "add_query":
                return f"Query '{data}' added to session {session_id}"
            else:
                return f"Unknown action: {action}. Available actions: get_history, get_context, add_query"
                
        except Exception as e:
            return f"Session management error: {str(e)}"


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
        # Configure matplotlib to use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend to prevent window opening
        logger.info("Matplotlib backend set to Agg (non-interactive)")
        
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
        
        self.sandbox_executor = SandboxExecutor()
        self.csv_processor = GenericCSVProcessor()
        self._current_df = None
        self._current_file_path = None
        self._current_session_id = None
        self._conversation_memory = {}  # Store memory per session
    
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
            
            # Load conversation history if session exists
            memory = None
            if session_id:
                self._load_conversation_history(session_id)
                memory = self._get_or_create_memory(session_id)
                logger.info(f"Loaded conversation memory for session {session_id}")
            
            # Create custom tools
            tools = self._create_custom_tools(session_id)
            
            # Check if this is a data request (asking for specific data)
            is_data_request = self._is_data_request(query)
            
            if is_data_request:
                # For data requests, execute pandas code directly and return the data
                result_data = self._execute_data_request(query)
                parsed_result = self._parse_data_result(result_data, query, file_path, session_id)
            else:
                # For analysis requests, use the standard agent
                agent = create_pandas_dataframe_agent(
                    self.llm,
                    self._current_df,
                    agent_type="zero-shot-react-description",
                    extra_tools=tools,
                    verbose=True,
                    max_iterations=10,  # Increased from 5 to 10 for complex queries
                    memory=memory  # Add conversation memory
                )
                
                # Execute the agent with better error handling
                logger.info("Executing hybrid agent...")
                try:
                    result = agent.run(query)
                    
                    # Check if result is meaningful
                    if not result or result.strip() == "":
                        raise Exception("Agent returned empty result")
                    
                    # Parse and format the result
                    parsed_result = self._parse_agent_result(result, query, file_path, session_id)
                    
                except Exception as agent_error:
                    logger.error(f"Agent execution failed: {agent_error}")
                    
                    # Try a simpler fallback approach
                    try:
                        logger.info("Attempting fallback analysis...")
                        fallback_result = self._execute_fallback_analysis(query, file_path)
                        parsed_result = self._parse_agent_result(fallback_result, query, file_path, session_id)
                    except Exception as fallback_error:
                        logger.error(f"Fallback analysis also failed: {fallback_error}")
                        parsed_result = self._create_error_response(
                            f"Analysis failed: {str(agent_error)}. Please try a simpler query.", 
                            query, 
                            file_path
                        )
            
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
    
    def _get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a session."""
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self._conversation_memory[session_id]
    
    def _load_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Load conversation history from database into LangChain memory."""
        try:
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session and session.conversation_history:
                memory = self._get_or_create_memory(session_id)
                
                # Clear existing memory
                memory.clear()
                
                # Load conversation history into memory
                for entry in session.conversation_history:
                    if isinstance(entry, dict):
                        if "query" in entry and "result" in entry:
                            # Current format
                            memory.chat_memory.add_user_message(entry["query"])
                            memory.chat_memory.add_ai_message(str(entry["result"]))
                        elif "role" in entry and "content" in entry:
                            # Alternative format
                            if entry["role"] == "user":
                                memory.chat_memory.add_user_message(entry["content"])
                            elif entry["role"] == "assistant":
                                memory.chat_memory.add_ai_message(entry["content"])
                
                logger.info(f"Loaded {len(session.conversation_history)} conversation entries for session {session_id}")
                return memory.chat_memory.messages
            else:
                logger.info(f"No conversation history found for session {session_id}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            return []
    
    def _update_session_context(self, session_id: str, query: str, result: Dict[str, Any]):
        """Update session context in the database and LangChain memory."""
        try:
            # Update LangChain memory
            memory = self._get_or_create_memory(session_id)
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(str(result.get("result", "")))
            
            # Update database
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

    def _is_data_request(self, query: str) -> bool:
        """Detect if the query is asking for specific data (not analysis)."""
        query_lower = query.lower()
        
        # Keywords that indicate data requests
        data_keywords = [
            'get', 'show', 'display', 'return', 'find', 'filter', 'select',
            'all', 'rows', 'data', 'records', 'entries', 'subset'
        ]
        
        # Keywords that indicate analysis requests
        analysis_keywords = [
            'how many', 'count', 'average', 'mean', 'median', 'sum', 'total',
            'percentage', 'proportion', 'correlation', 'trend', 'pattern',
            'what is', 'analyze', 'describe', 'explain', 'why', 'when',
            'barplot', 'chart', 'graph', 'plot', 'visualize'
        ]
        
        # Keywords that indicate insight requests
        insight_keywords = [
            'types of', 'patterns in', 'trends', 'anomalies', 'interesting',
            'insights', 'findings', 'discover'
        ]
        
        # Check for data request keywords
        has_data_keywords = any(keyword in query_lower for keyword in data_keywords)
        
        # Check for analysis/insight keywords
        has_analysis_keywords = any(keyword in query_lower for keyword in analysis_keywords)
        has_insight_keywords = any(keyword in query_lower for keyword in insight_keywords)
        
        # If it has data keywords but NO analysis/insight keywords, it's a data request
        if has_data_keywords and not (has_analysis_keywords or has_insight_keywords):
            return True
            
        # If it has analysis/insight keywords, it's NOT a data request
        if has_analysis_keywords or has_insight_keywords:
            return False
            
        # Default: treat as analysis request if unclear
        return False
    
    def _execute_data_request(self, query: str) -> Dict[str, Any]:
        """Execute a data request and return the actual data."""
        try:
            # Use the LLM to generate pandas code for the data request
            prompt = f"""
You are working with a pandas dataframe called 'df'. The user wants to get specific data.

User request: {query}

Generate ONLY the pandas code needed to get the requested data. Return ONLY the code, no explanations.
The code should return the filtered/selected dataframe.

Example:
- If user asks "get all shoplifting events", return: df[df['major_category'] == 'Shoplifting']
- If user asks "show first 10 rows", return: df.head(10)
- If user asks "filter by borough", return: df[df['borough'] == 'specific_borough']

Code:
"""
            
            # Get the code from LLM
            response = self.llm.invoke(prompt)
            code = response.content.strip()
            
            # Execute the code safely
            local_vars = {'df': self._current_df.copy()}
            exec(code, {}, local_vars)
            
            # Get the result (last expression)
            result_df = None
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                    result_df = var_value
                    break
            
            # If no specific dataframe found, try to get the last expression result
            if result_df is None:
                # Try to extract the result from the executed code
                try:
                    # Execute the code and capture the result
                    result = eval(code, {}, local_vars)
                    if isinstance(result, pd.DataFrame):
                        result_df = result
                    else:
                        # If it's not a dataframe, try to get the last dataframe operation
                        result_df = local_vars.get('df', self._current_df)
                except:
                    result_df = local_vars.get('df', self._current_df)
            
            # Convert to list of dictionaries for JSON serialization
            if result_df is not None:
                # Handle NaN values and other non-JSON-compliant data
                def clean_value(val):
                    if pd.isna(val):
                        return None
                    elif isinstance(val, (np.integer, np.floating)):
                        return float(val)
                    elif isinstance(val, np.ndarray):
                        return val.tolist()
                    else:
                        return val
                
                # Convert dataframe to records with cleaned values
                data = []
                for _, row in result_df.iterrows():
                    clean_row = {}
                    for col, val in row.items():
                        clean_row[col] = clean_value(val)
                    data.append(clean_row)
                
                return {
                    'success': True,
                    'data': data,
                    'columns': list(result_df.columns),
                    'total_rows': len(result_df),
                    'code_used': code
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not extract data from the executed code'
                }
                
        except Exception as e:
            logger.error(f"Data request execution failed: {e}")
            return {
                'success': False,
                'error': f'Failed to execute data request: {str(e)}'
            }
    
    def _parse_data_result(self, result_data: Dict[str, Any], query: str, file_path: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Parse data result into our standard format."""
        if result_data.get('success'):
            return {
                "success": True,
                "result": result_data['data'],  # Actual data as list of dicts
                "result_type": "table",  # Mark as table data
                "metadata": {
                    "query": query,
                    "file_path": file_path,
                    "session_id": session_id,
                    "agent_type": "hybrid_langchain",
                    "execution_time": 0,
                    "rows_analyzed": result_data.get('total_rows', 0),
                    "columns_analyzed": len(result_data.get('columns', [])),
                    "code_used": result_data.get('code_used', ''),
                    "data_request": True
                },
                "query": query,
                "file_path": file_path,
                "session_id": session_id
            }
        else:
            return {
                "success": False,
                "result": f"Data request failed: {result_data.get('error', 'Unknown error')}",
                "result_type": "text",
                "metadata": {
                    "query": query,
                    "file_path": file_path,
                    "session_id": session_id,
                    "agent_type": "hybrid_langchain",
                    "error": result_data.get('error', 'Unknown error')
                },
                "query": query,
                "file_path": file_path,
                "session_id": session_id
            }

    def _execute_fallback_analysis(self, query: str, file_path: str) -> str:
        """Execute a simpler fallback analysis when the main agent fails."""
        try:
            # Use a simpler prompt for basic analysis
            prompt = f"""
You are analyzing a pandas dataframe. The user asked: "{query}"

Please provide ONLY executable Python code to answer the query. Do not include explanations or markdown formatting.

Available columns: {", ".join(self._current_df.columns.tolist())}

Generate simple pandas code to answer the query. Return ONLY the code, no explanations.
"""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            code_response = response.content.strip()
            
            # Extract only the Python code (remove markdown, explanations, etc.)
            import re
            
            # Look for code blocks
            code_blocks = re.findall(r'```python\n(.*?)\n```', code_response, re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
            else:
                # Look for lines that start with common pandas operations
                lines = code_response.split('\n')
                code_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('"') and not line.startswith("'"):
                        if any(keyword in line.lower() for keyword in ['df[', 'df.', 'import', 'print', 'count', 'value_counts', 'groupby', 'filter']):
                            code_lines.append(line)
                code = '\n'.join(code_lines)
            
            if not code:
                return f"Unable to generate executable code for: {query}"
            
            # Try to execute the code in the sandbox
            try:
                result = self.sandbox_executor.execute(code, file_path, {"df": self._current_df})
                if result.success:
                    return f"Analysis completed:\n\n{code}\n\nResult: {result.result}"
                else:
                    return f"Analysis failed:\n\n{code}\n\nError: {result.error}"
            except Exception as e:
                return f"Code execution failed:\n\n{code}\n\nError: {str(e)}"
                
        except Exception as e:
            return f"Fallback analysis failed: {str(e)}"


# Global agent instance
_hybrid_agent = None


def get_hybrid_agent() -> HybridCSVAgent:
    """Get the global hybrid CSV analysis agent instance."""
    global _hybrid_agent
    if _hybrid_agent is None:
        _hybrid_agent = HybridCSVAgent()
    return _hybrid_agent 