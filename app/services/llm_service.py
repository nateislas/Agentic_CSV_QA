"""
LLM service for the CSV Analysis Platform.

This module provides OpenAI integration and LLM utilities for
the LangChain agent system with generic structural analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

# Use absolute imports
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service for OpenAI integration and chat completion.
    
    Provides a clean interface for LangChain agents and handles
    token management, response formatting, and error handling.
    Focuses on structural analysis without domain assumptions.
    """
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize LangChain chat model
        self.chat_model = ChatOpenAI(
            model=self.model,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )
    
    def get_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Get chat completion from OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            
        Returns:
            Dictionary with completion response
        """
        try:
            with get_openai_callback() as cb:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )
                
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens
                    },
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "usage": {}
            }
    
    def analyze_csv_structure(
        self, 
        query: str, 
        structural_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze CSV data using LLM with structural context only.
        
        Args:
            query: Natural language query
            structural_context: CSV structural context and metadata
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Prepare system message with structural information only
            system_message = self._create_structural_system_message(structural_context)
            
            # Prepare messages
            messages = [{"role": "system", "content": system_message}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-5:])  # Last 5 messages
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Get completion
            response = self.get_chat_completion(messages)
            
            if response["success"]:
                return {
                    "success": True,
                    "analysis": response["content"],
                    "usage": response["usage"],
                    "query": query
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"CSV structural analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def _create_structural_system_message(self, structural_context: Dict[str, Any]) -> str:
        """
        Create system message with structural context for LLM.
        
        Args:
            structural_context: CSV structural context and metadata
            
        Returns:
            Formatted system message focused on structure only
        """
        try:
            schema = structural_context.get("schema", {})
            column_structure = structural_context.get("column_structure", {})
            operational_capabilities = structural_context.get("operational_capabilities", {})
            analysis_guidance = structural_context.get("analysis_guidance", [])
            
            # Build system message focused on structure
            system_parts = [
                "You are a CSV data analysis assistant. Your role is to help users understand and analyze their CSV data based on structural characteristics only.",
                "",
                f"Dataset Structure:",
                f"- Total rows: {schema.get('total_rows', 'Unknown')}",
                f"- Total columns: {schema.get('total_columns', 'Unknown')}",
                f"- Columns: {', '.join(schema.get('columns', []))}",
                "",
                "Column Structural Analysis:"
            ]
            
            # Add column structural information
            for col_name, col_info in column_structure.items():
                data_type = col_info.get("data_type", "Unknown")
                cardinality = col_info.get("cardinality", "Unknown")
                completeness = col_info.get("completeness", "Unknown")
                characteristics = col_info.get("characteristics", {})
                
                system_parts.append(f"- {col_name}:")
                system_parts.append(f"  Type: {data_type}")
                system_parts.append(f"  Cardinality: {cardinality}")
                system_parts.append(f"  Completeness: {completeness}")
                
                # Add relevant characteristics
                if data_type == "numeric" and characteristics:
                    if "mean" in characteristics:
                        system_parts.append(f"  Range: {characteristics.get('min', 'N/A')} to {characteristics.get('max', 'N/A')}")
                        system_parts.append(f"  Mean: {characteristics.get('mean', 'N/A')}")
                elif data_type == "categorical" and characteristics:
                    if "most_common_values" in characteristics:
                        system_parts.append(f"  Most common: {list(characteristics.get('most_common_values', {}).keys())[:3]}")
            
            # Add operational capabilities
            if operational_capabilities:
                system_parts.extend([
                    "",
                    "Structural Capabilities:",
                    f"- Grouping columns: {operational_capabilities.get('grouping_columns', [])}",
                    f"- Aggregation columns: {operational_capabilities.get('aggregation_columns', [])}",
                    f"- Filtering columns: {operational_capabilities.get('filtering_columns', [])}",
                    f"- Identifier columns: {operational_capabilities.get('identifier_columns', [])}",
                    f"- Join key candidates: {operational_capabilities.get('join_key_candidates', [])}"
                ])
            
            # Add analysis guidance
            if analysis_guidance:
                system_parts.extend([
                    "",
                    "Analysis Guidance:",
                    *[f"- {guidance}" for guidance in analysis_guidance]
                ])
            
            system_parts.extend([
                "",
                "Instructions:",
                "- Focus on structural analysis and capabilities",
                "- Suggest operations based on data types and cardinality",
                "- Explain what operations are possible given the structure",
                "- Provide clear, actionable analysis",
                "- Format results as tables when appropriate",
                "- Be concise but thorough",
                "- Do not make assumptions about what the data represents"
            ])
            
            return "\n".join(system_parts)
            
        except Exception as e:
            logger.error(f"Structural system message creation error: {e}")
            return "You are a CSV data analysis assistant. Help users understand and analyze their CSV data based on structural characteristics."
    
    def generate_structural_operations(self, natural_query: str, structural_context: Dict[str, Any]) -> str:
        """
        Generate structural operations from natural language.
        
        Args:
            natural_query: Natural language query
            structural_context: Structural context information
            
        Returns:
            Generated operation description
        """
        try:
            schema = structural_context.get("schema", {})
            operational_capabilities = structural_context.get("operational_capabilities", {})
            
            system_message = f"""
            You are a data operation generator. Convert natural language to structural operations.
            
            Available columns: {', '.join(schema.get('columns', []))}
            Grouping columns: {operational_capabilities.get('grouping_columns', [])}
            Aggregation columns: {operational_capabilities.get('aggregation_columns', [])}
            Filtering columns: {operational_capabilities.get('filtering_columns', [])}
            
            Generate operations that can be performed with pandas/polars based on structural capabilities.
            Return only the operation description, not code.
            Focus on what is structurally possible given the data types and cardinality.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": natural_query}
            ]
            
            response = self.get_chat_completion(messages, temperature=0.1)
            
            if response["success"]:
                return response["content"]
            else:
                return f"Error generating operations: {response.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Structural operations generation error: {e}")
            return f"Error: {str(e)}"
    
    def validate_structural_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and classify user query based on structural capabilities.
        
        Args:
            query: User's natural language query
            
        Returns:
            Query validation and classification
        """
        try:
            system_message = """
            Classify the user's query into one of these categories based on structural operations:
            - data_exploration: Questions about data structure, columns, data types
            - aggregation: Group by, sum, average, count operations
            - filtering: Where clauses, data filtering
            - structural_analysis: Questions about relationships, cardinality, completeness
            - operational_analysis: Questions about what operations are possible
            
            Return only the category name.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            response = self.get_chat_completion(messages, temperature=0.1)
            
            if response["success"]:
                category = response["content"].strip().lower()
                return {
                    "valid": True,
                    "category": category,
                    "query": query
                }
            else:
                return {
                    "valid": False,
                    "error": response.get("error", "Unknown error"),
                    "query": query
                }
                
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return {
                "valid": False,
                "error": str(e),
                "query": query
            }
    
    def format_structural_result(
        self, 
        analysis: str, 
        data_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format structural analysis result for frontend display.
        
        Args:
            analysis: LLM analysis text
            data_result: Optional data processing result
            
        Returns:
            Formatted result for API response
        """
        try:
            result = {
                "type": "structural_analysis",
                "summary": analysis,
                "timestamp": "2024-01-01T00:00:00Z"  # Would use real timestamp
            }
            
            if data_result:
                result.update({
                    "type": "table",
                    "data": data_result.get("data", {}),
                    "metadata": {
                        "execution_time": data_result.get("execution_time", 0),
                        "rows_processed": data_result.get("rows_processed", 0),
                        "query_type": data_result.get("query_type", "structural_analysis")
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Result formatting error: {e}")
            return {
                "type": "error",
                "error": str(e),
                "summary": "Error formatting result"
            }


# Create global LLM service instance
llm_service = LLMService()
