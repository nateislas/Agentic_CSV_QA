"""
LangChain Agent Service for CSV Analysis.

This module provides the main LangChain agent that handles natural language queries
and generates secure Python code for data analysis with intelligent error recovery.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session as DBSession

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder

from app.core.database import get_db
from app.models import Session as SessionModel
from app.services.code_execution_tool import code_execution_tool
from app.core.config import settings

logger = logging.getLogger(__name__)


class LangChainCSVAgent:
    """Main LangChain agent for CSV data analysis with intelligent error recovery."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and memory."""
        tools = [code_execution_tool]
        
        # Create conversation memory for multi-turn conversations
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the agent with error handling capabilities
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            memory=memory,
            agent_kwargs={
                "system_message": self._get_system_message()
            }
        )
    
    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return """You are an intelligent CSV data analysis agent. Your job is to help users analyze CSV data by generating and executing Python code.

IMPORTANT GUIDELINES:

1. **Code Generation**: Always generate Python code that uses the execute_code tool
2. **Data Analysis**: Use pandas, numpy, matplotlib, and seaborn for analysis
3. **Result Format**: Always use the create_result() function to return results
4. **Error Recovery**: If code execution fails, analyze the error and generate corrected code
5. **Security**: Code runs in a secure sandbox - no file system access, no network calls

AVAILABLE LIBRARIES:
- pandas (pd)
- numpy (np) 
- matplotlib.pyplot (plt)
- seaborn (sns)
- datetime, json, math, statistics, collections

RESULT FORMAT EXAMPLES:

For data analysis:
```python
df = pd.read_csv(data_path)
# Your analysis code here
result = create_result(
    data=your_data,
    result_type="table",  # or "text", "plot"
    metadata={"query": "your_query", "analysis_type": "your_analysis"}
)
```

For visualizations:
```python
df = pd.read_csv(data_path)
plt.figure(figsize=(12, 6))
# Your plotting code here
plt.tight_layout()
result = create_result(
    data="plot_generated",
    result_type="plot",
    metadata={"query": "your_query", "plot_type": "your_plot_type"}
)
```

ERROR RECOVERY:
When code execution fails:
1. Analyze the error message carefully
2. Identify the specific issue (missing column, data type error, etc.)
3. Generate corrected code that addresses the issue
4. Try again with the fixed code

Always be helpful and provide clear explanations of what you're doing."""
    
    def analyze_query(
        self, 
        query: str, 
        file_path: str, 
        session_id: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Analyze a natural language query using the LangChain agent.
        
        Args:
            query: Natural language query from user
            file_path: Path to CSV file to analyze
            session_id: Optional session ID for conversation history
            metadata: Optional metadata about the file
            
        Returns:
            Dictionary containing analysis result and metadata
        """
        try:
            logger.info(f"Starting LangChain agent analysis for query: {query}")
            
            # Load session context if available
            session_context = self._load_session_context(session_id) if session_id else None
            
            # Prepare the agent input
            agent_input = self._prepare_agent_input(query, file_path, session_context)
            
            # Execute the agent
            logger.info("Executing LangChain agent...")
            result = self.agent.run(agent_input)
            
            # Parse the result
            parsed_result = self._parse_agent_result(result, query, file_path)
            
            # Update session context if available
            if session_id:
                self._update_session_context(session_id, query, parsed_result)
            
            logger.info("LangChain agent analysis completed successfully")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LangChain agent analysis failed: {e}")
            return self._create_error_response(str(e), query, file_path)
    
    def _prepare_agent_input(self, query: str, file_path: str, session_context: Dict = None) -> str:
        """Prepare input for the LangChain agent."""
        # Build context information
        context_parts = [
            f"User Query: {query}",
            f"CSV File Path: {file_path}"
        ]
        
        # Add session context if available
        if session_context:
            if session_context.get("conversation_history"):
                context_parts.append("Previous conversation context available")
            if session_context.get("active_tables"):
                context_parts.append("Previous analysis results available")
        
        # Add file metadata if available
        if session_context and session_context.get("file_metadata"):
            metadata = session_context["file_metadata"]
            context_parts.append(f"File has {metadata.get('rows', 'unknown')} rows and {metadata.get('columns', 'unknown')} columns")
        
        context = "\n".join(context_parts)
        
        # Create the agent input
        agent_input = f"""
{context}

Please analyze this CSV data according to the user's query. Generate and execute the appropriate Python code using the execute_code tool.
"""
        return agent_input
    
    def _parse_agent_result(self, result: str, query: str, file_path: str) -> Dict[str, Any]:
        """Parse the agent result into a structured response."""
        # For now, return a simple success response
        # In a full implementation, you would parse the agent's output more carefully
        return {
            "success": True,
            "query": query,
            "file_path": file_path,
            "result": {
                "type": "text",
                "data": result,
                "metadata": {
                    "agent_type": "langchain",
                    "analysis_type": "csv_analysis"
                }
            },
            "execution_time": 0  # Would be calculated in real implementation
        }
    
    def _load_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session context from database."""
        try:
            db = next(get_db())
            session_record = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session_record:
                context = {
                    "conversation_history": session_record.conversation_history or [],
                    "active_tables": session_record.active_tables or {},
                    "file_metadata": session_record.file_metadata or {}
                }
                db.close()
                return context
            else:
                db.close()
                return None
                
        except Exception as e:
            logger.error(f"Failed to load session context: {e}")
            return None
    
    def _update_session_context(self, session_id: str, query: str, result: Dict[str, Any]):
        """Update session context in database."""
        try:
            db = next(get_db())
            session_record = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if session_record:
                # Update conversation history
                history = session_record.conversation_history or []
                history.append({
                    "query": query,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                })
                session_record.conversation_history = history
                
                # Update active tables if result contains table data
                if result.get("result", {}).get("type") == "table":
                    active_tables = session_record.active_tables or {}
                    active_tables[f"result_{len(history)}"] = result
                    session_record.active_tables = active_tables
                
                db.commit()
            
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
    
    def _create_error_response(self, error: str, query: str, file_path: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "success": False,
            "query": query,
            "file_path": file_path,
            "error": error,
            "result": {
                "type": "error",
                "data": error,
                "metadata": {
                    "agent_type": "langchain",
                    "error_type": "agent_error"
                }
            },
            "execution_time": 0
        } 