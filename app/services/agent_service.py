"""
LangChain Agent Service for CSV Analysis Platform.

This module provides the main agent that coordinates between natural language
queries and CSV operations using generic structural analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from .llm_service import LLMService
from .csv_processor import csv_processor
from app.core.config import settings

logger = logging.getLogger(__name__)
# Set logger level to DEBUG to see all messages
logger.setLevel(logging.DEBUG)


class CSVAnalysisAgent:
    """
    LangChain agent for CSV analysis using generic structural analysis.
    
    This agent coordinates between natural language queries and CSV operations
    without making domain-specific assumptions about the data content.
    """
    
    def __init__(self):
        """Initialize the CSV analysis agent."""
        self.llm_service = LLMService()
        self.tools: List[BaseTool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.current_file_path: Optional[str] = None  # Store current file path
        
        # Initialize tools
        self._initialize_tools()
        logger.info(f"Initialized {len(self.tools)} tools")
        
        # Log tool information for debugging
        for i, tool in enumerate(self.tools):
            logger.info(f"Tool {i+1}: {tool.name} - {tool.description[:100]}...")
        
        # Initialize agent
        self._initialize_agent()
        logger.info("Agent initialization completed")
    
    def _initialize_tools(self):
        """Initialize the tools available to the agent."""
        from .tools.data_exploration_tool import DataExplorationTool
        from .tools.aggregation_tool import AggregationTool
        from .tools.filtering_tool import FilteringTool
        from .tools.statistics_tool import StatisticsTool
        from .tools.visualization_tool import VisualizationTool
        
        # Create tool instances
        self.tools = [
            DataExplorationTool(),
            AggregationTool(),
            FilteringTool(),
            StatisticsTool(),
            VisualizationTool()
        ]
        
        logger.info(f"Initialized {len(self.tools)} tools for CSV analysis agent")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and prompt."""
        try:
            # Create LLM with explicit API key
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Create system prompt template
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = self._create_system_prompt()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Wrap in AgentExecutor for proper execution
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
            logger.info("Successfully initialized CSV analysis agent")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are a CSV data analysis assistant. Your role is to help users understand and analyze their CSV data based on structural characteristics only.

You have access to tools that can perform various operations on CSV data:
- data_exploration: Get information about columns, data types, and sample data
- aggregation: Perform group by, sum, average, count operations
- filtering: Filter data based on conditions
- statistics: Calculate descriptive statistics and correlations
- visualization: Create charts and summaries

IMPORTANT GUIDELINES:
1. Focus on structural analysis - don't make assumptions about what the data represents
2. Use the available tools to perform operations
3. ALWAYS use tools to get actual data - don't just describe what you could do
4. For showing sample data, use data_exploration with 'sample_data' operation
5. For understanding structure, use data_exploration with 'summary' operation
6. For column information, use data_exploration with 'column_info' operation
7. Provide clear, actionable analysis
8. Be concise but thorough

When a user asks a question:
1. If they ask to see rows/data, use data_exploration with 'sample_data' operation
2. If they ask about structure, use data_exploration with 'summary' operation
3. If they ask about columns, use data_exploration with 'column_info' operation
4. Present the results clearly with explanations
5. Suggest additional analyses that might be useful

Remember: You work with ANY CSV data without knowing what it represents. Focus on structure, not content. The tools will automatically use the correct file path."""
    
    def analyze_query(
        self, 
        query: str, 
        file_path: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a natural language query using the agent.
        
        Args:
            query: Natural language query from user
            file_path: Path to the CSV file
            session_id: Optional session ID for conversation history
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Set the current file path for tools to use
            self.set_current_file_path(file_path)
            logger.info(f"Set current file path in agent: {file_path}")
            logger.info(f"Agent current file path after setting: {self.get_current_file_path()}")
            
            # Get structural metadata for context
            metadata = self._get_file_metadata(file_path)
            if not metadata:
                return {
                    "success": False,
                    "error": "Failed to get file metadata"
                }
            
            # Prepare context for the agent
            logger.info(f"Preparing agent context with file_path: {file_path}")
            context = self._prepare_agent_context(metadata, query, file_path)
            
            # Execute agent
            start_time = datetime.now()
            
            # Add conversation history if available
            messages = []
            if session_id and self.conversation_history:
                # Add relevant conversation history
                recent_history = self.conversation_history[-5:]  # Last 5 exchanges
                for msg in recent_history:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        messages.append(SystemMessage(content=msg.get("content", "")))
            
            # Add current query
            messages.append(HumanMessage(content=context))
            
            # Execute agent
            logger.info(f"About to execute agent with context length: {len(context)}")
            logger.info(f"Context preview: {context[:200]}...")
            logger.info(f"Agent executor type: {type(self.agent_executor)}")
            
            # Add debugging for agent input
            agent_input = {
                "input": query,
                "chat_history": []
            }
            logger.info(f"Agent input: {agent_input}")
            
            try:
                logger.info(f"About to invoke agent with input: {agent_input}")
                response = self.agent_executor.invoke(agent_input)
                logger.info(f"Agent response type: {type(response)}")
                logger.info(f"Agent response: {response}")
            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                import traceback
                logger.error(f"Agent execution traceback: {traceback.format_exc()}")
                raise
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log the full response structure for debugging
            logger.info(f"Agent response type: {type(response)}")
            logger.info(f"Agent response: {response}")
            
            if isinstance(response, dict):
                logger.info(f"Response keys: {list(response.keys())}")
                for key, value in response.items():
                    logger.info(f"  {key}: {type(value)} = {value}")
            
            # Extract output from response
            output = ""
            if isinstance(response, dict):
                output = response.get("output", "")
            elif hasattr(response, "output"):
                output = response.output
            else:
                output = str(response)
            
            # Update conversation history
            if session_id:
                self.conversation_history.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": output,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "result": output,
                "execution_time": execution_time,
                "query": query,
                "file_path": file_path,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "file_path": file_path
            }
    
    def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get structural metadata for the CSV file."""
        try:
            result = csv_processor.process_csv_file(file_path)
            if result["success"]:
                return result["metadata"]
            else:
                logger.error(f"Failed to get metadata: {result.get('error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return None
    
    def _prepare_agent_context(self, metadata: Dict[str, Any], query: str, file_path: str) -> str:
        """Prepare context for the agent based on structural metadata."""
        try:
            llm_context = metadata.get("llm_context", {})
            schema = llm_context.get("schema", {})
            column_structure = llm_context.get("column_structure", {})
            operational_capabilities = llm_context.get("operational_capabilities", {})
            
            context_parts = [
                f"Dataset Information:",
                f"- Total rows: {schema.get('total_rows', 'Unknown')}",
                f"- Total columns: {schema.get('total_columns', 'Unknown')}",
                f"- Columns: {', '.join(schema.get('columns', []))}",
                "",
                "Column Structure:"
            ]
            
            # Add column information
            for col_name, col_info in column_structure.items():
                data_type = col_info.get("data_type", "Unknown")
                cardinality = col_info.get("cardinality", "Unknown")
                completeness = col_info.get("completeness", "Unknown")
                
                context_parts.append(f"- {col_name}: {data_type} (cardinality: {cardinality}, completeness: {completeness})")
            
            # Add operational capabilities
            if operational_capabilities:
                context_parts.extend([
                    "",
                    "Available Operations:",
                    f"- Grouping columns: {operational_capabilities.get('grouping_columns', [])}",
                    f"- Aggregation columns: {operational_capabilities.get('aggregation_columns', [])}",
                    f"- Filtering columns: {operational_capabilities.get('filtering_columns', [])}",
                    f"- Identifier columns: {operational_capabilities.get('identifier_columns', [])}"
                ])
            
            context_parts.extend([
                "",
                f"User Query: {query}",
                "",
                "Please analyze this query and use the appropriate tools to provide a response."
            ])
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing agent context: {e}")
            return f"User Query: {query}\n\nPlease analyze this query and use the appropriate tools to provide a response."
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        # In a real implementation, this would be stored in a database
        return self.conversation_history
    
    def set_current_file_path(self, file_path: str):
        """Set the current file path for tools to use."""
        self.current_file_path = file_path
        logger.info(f"Set current file path: {file_path}")
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the current file path."""
        return self.current_file_path
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        self.conversation_history = []


# Create global agent instance
csv_agent = None

def get_csv_agent():
    """Get the CSV analysis agent instance (lazy initialization)."""
    global csv_agent
    logger.info("Getting CSV agent instance...")
    if csv_agent is None:
        logger.info("Creating new CSV agent instance...")
        csv_agent = CSVAnalysisAgent()
        logger.info("CSV agent instance created successfully")
    else:
        logger.info("Using existing CSV agent instance")
    return csv_agent 