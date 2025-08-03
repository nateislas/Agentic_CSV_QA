"""
LangChain Agent Service for CSV Analysis Platform.

This module provides the main agent that coordinates between natural language
queries and CSV operations using generic structural analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

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
            
            # Create prompt template with memory
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
        return """You are an intelligent data analysis assistant designed to help professionals understand their data through multi-turn conversations. Your role is to provide clear, actionable insights while maintaining context across the conversation.

You have access to tools that can perform various operations on data:
- data_exploration: Get information about data structure and sample records
- aggregation: Create summaries and group data by categories
- filtering: Focus on specific subsets of data
- statistics: Calculate averages, totals, and patterns
- visualization: Create clear visual summaries

MULTI-TURN CONVERSATION GUIDELINES:
1. **Maintain Context**: Reference previous results and tables when appropriate
2. **Follow-up Questions**: Understand references like "now add...", "filter to show only...", "sort by..."
3. **Active Tables**: Use tables created in previous queries when referenced
4. **Progressive Analysis**: Build on previous results to create more complex analyses
5. **Conversation Flow**: Acknowledge previous context and build naturally on it

IMPORTANT GUIDELINES:
1. Focus on what the data tells us - provide insights, not just technical descriptions
2. Use the available tools to get actual results - don't just describe possibilities
3. Present findings in clear, professional language
4. Explain what the results mean in practical terms
5. Suggest follow-up questions that would be valuable
6. Use "we" and "you" to create a collaborative tone
7. Avoid technical terms like "dataframe", "aggregation", "pivot tables" - instead use "summary", "breakdown", "grouping"

CONTEXT AWARENESS:
- If the user references a previous result, acknowledge it and build on it
- When they say "now add..." or "also include...", understand they want to enhance the previous result
- If they mention "the table above" or "this result", refer to the most recent table/result
- For follow-up questions, provide context about what you're building on

When a user asks a question:
1. If they want to see sample data, use data_exploration with 'sample_data' operation
2. If they want to understand the data structure, use data_exploration with 'summary' operation
3. If they want details about specific fields, use data_exploration with 'column_info' operation
4. If they reference a previous result, acknowledge it and build on it
5. Present results with clear explanations of what they mean
6. Suggest additional questions that would provide valuable insights

TONE AND LANGUAGE:
- Use professional but accessible language
- Explain findings in practical terms
- Focus on insights and patterns, not technical processes
- Use phrases like "Here's what we found..." and "This tells us that..."
- Avoid jargon - say "summary" instead of "aggregation", "grouping" instead of "group by"
- Make recommendations based on what the data reveals
- For follow-ups, use phrases like "Building on that..." or "Now let's add..."

Remember: You work with ANY data without making assumptions about what it represents. Focus on revealing patterns and insights that help users understand their information better. The tools will automatically use the correct file. Maintain conversation context and build naturally on previous results."""
    
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
            
            # Load or create memory for this session
            memory = self._get_or_create_memory(session_id)
            
            # Prepare context for the agent
            logger.info(f"Preparing agent context with file_path: {file_path}")
            context = self._prepare_agent_context(metadata, query, file_path)
            
            # Execute agent with memory
            start_time = datetime.now()
            
            # Execute agent with memory
            logger.info(f"About to execute agent with memory for session: {session_id}")
            
            try:
                # Use the agent with memory
                response = self.agent_executor.invoke({
                    "input": query,
                    "chat_history": memory.chat_memory.messages
                })
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
            
            # Save conversation to memory
            if session_id:
                memory.chat_memory.add_user_message(query)
                memory.chat_memory.add_ai_message(output)
                self._save_memory_to_session(session_id, memory)
            
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
    
    def _get_or_create_memory(self, session_id: Optional[str] = None) -> ConversationBufferMemory:
        """Get or create memory for a session."""
        if not session_id:
            # Create new memory for new session
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Try to load existing memory from session
        memory = self._load_memory_from_session(session_id)
        if memory:
            return memory
        
        # Create new memory if none exists
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _load_memory_from_session(self, session_id: str) -> Optional[ConversationBufferMemory]:
        """Load memory from session database."""
        try:
            from app.core.database import get_db
            from app.models import Session as SessionModel
            
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # Create memory and populate with conversation history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Convert conversation history to LangChain messages
            for msg in session.conversation_history:
                role = msg.get('role')
                content = msg.get('content', '')
                
                if role == 'user':
                    memory.chat_memory.add_user_message(content)
                elif role == 'assistant':
                    memory.chat_memory.add_ai_message(content)
            
            db.close()
            logger.info(f"Loaded memory with {len(session.conversation_history)} messages")
            return memory
            
        except Exception as e:
            logger.error(f"Error loading memory from session: {e}")
            return None
    
    def _save_memory_to_session(self, session_id: str, memory: ConversationBufferMemory):
        """Save memory to session database."""
        try:
            from app.core.database import get_db
            from app.models import Session as SessionModel
            
            db = next(get_db())
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            
            if not session:
                logger.warning(f"Session not found for memory save: {session_id}")
                return
            
            # Convert memory messages to conversation history format
            conversation_history = []
            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    conversation_history.append({
                        "role": "user",
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    conversation_history.append({
                        "role": "assistant",
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Update session
            session.conversation_history = conversation_history
            session.updated_at = datetime.utcnow()
            
            db.commit()
            db.close()
            
            logger.info(f"Saved memory to session: {session_id} with {len(conversation_history)} messages")
            
        except Exception as e:
            logger.error(f"Error saving memory to session: {e}")
    
    def _prepare_agent_context(self, metadata: Dict[str, Any], query: str, file_path: str) -> str:
        """Prepare context for the agent based on structural metadata."""
        try:
            llm_context = metadata.get("llm_context", {})
            schema = llm_context.get("schema", {})
            column_structure = llm_context.get("column_structure", {})
            operational_capabilities = llm_context.get("operational_capabilities", {})
            
            context_parts = [
                f"Data Overview:",
                f"- Total records: {schema.get('total_rows', 'Unknown')}",
                f"- Number of fields: {schema.get('total_columns', 'Unknown')}",
                f"- Available fields: {', '.join(schema.get('columns', []))}",
                "",
                "Field Details:"
            ]
            
            # Add column information in professional language
            for col_name, col_info in column_structure.items():
                data_type = col_info.get("data_type", "Unknown")
                cardinality = col_info.get("cardinality", "Unknown")
                completeness = col_info.get("completeness", "Unknown")
                
                # Translate technical terms to professional language
                data_type_desc = {
                    "numeric": "numbers",
                    "text": "text",
                    "date": "dates",
                    "categorical": "categories"
                }.get(data_type, data_type)
                
                cardinality_desc = {
                    "high": "many unique values",
                    "medium": "moderate variety",
                    "low": "few unique values"
                }.get(cardinality, cardinality)
                
                completeness_desc = {
                    "high": "mostly complete",
                    "medium": "partially complete", 
                    "low": "many missing values"
                }.get(completeness, completeness)
                
                context_parts.append(f"- {col_name}: {data_type_desc} ({cardinality_desc}, {completeness_desc})")
            
            # Add operational capabilities in professional language
            if operational_capabilities:
                context_parts.extend([
                    "",
                    "Analysis Capabilities:",
                    f"- Can group by: {operational_capabilities.get('grouping_columns', [])}",
                    f"- Can calculate totals/averages for: {operational_capabilities.get('aggregation_columns', [])}",
                    f"- Can filter by: {operational_capabilities.get('filtering_columns', [])}",
                    f"- Unique identifiers: {operational_capabilities.get('identifier_columns', [])}"
                ])
            
            context_parts.extend([
                "",
                f"User Question: {query}",
                "",
                "Please analyze this question and use the appropriate tools to provide a clear, professional response."
            ])
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing agent context: {e}")
            return f"User Query: {query}\n\nPlease analyze this query and use the appropriate tools to provide a response."
    
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
    
    def set_current_file_path(self, file_path: str):
        """Set the current file path for tools to use."""
        self.current_file_path = file_path
        logger.info(f"Set current file path: {file_path}")
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the current file path."""
        return self.current_file_path


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