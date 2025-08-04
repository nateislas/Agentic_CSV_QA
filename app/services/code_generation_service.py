"""
Code Generation Service for Data Analysis.

This module provides LLM-based code generation for secure data analysis
with context awareness for multi-turn conversations.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.services.security_validator import CodeSecurityValidator
from app.services.sandbox_executor import SandboxExecutor, ExecutionResult
from app.services.csv_processor import GenericCSVProcessor

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """Result of code generation and execution."""
    success: bool
    generated_code: str
    execution_result: Optional[ExecutionResult]
    error: Optional[str]
    context_used: Dict[str, Any]


class CodeGenerationService:
    """Service for generating and executing secure Python code for data analysis."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.validator = CodeSecurityValidator()
        self.executor = SandboxExecutor(timeout=30, memory_limit_mb=512)
        self.executor.set_validator(self.validator)
        self.csv_processor = GenericCSVProcessor()

    def generate_and_execute(
        self, 
        query: str, 
        data_path: str, 
        session_context: Dict = None
    ) -> CodeGenerationResult:
        """
        Generate and execute code for data analysis.
        
        Args:
            query: Natural language query
            data_path: Path to CSV file
            session_context: Session context for multi-turn conversations
            
        Returns:
            CodeGenerationResult with generation and execution details
        """
        try:
            # Extract data context for better code generation
            data_context = self._extract_data_context(data_path)
            
            # Generate code with data context
            generated_code = self._generate_code(query, session_context, data_context)
            
            if not generated_code:
                return CodeGenerationResult(
                    success=False,
                    generated_code="",
                    execution_result=None,
                    error="Failed to generate code",
                    context_used={}
                )
            
            # Execute code
            execution_result = self.executor.execute(generated_code, data_path, session_context)
            
            return CodeGenerationResult(
                success=execution_result.success,
                generated_code=generated_code,
                execution_result=execution_result,
                error=execution_result.error if not execution_result.success else None,
                context_used=session_context or {}
            )
            
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return CodeGenerationResult(
                success=False,
                generated_code="",
                execution_result=None,
                error=f"Code generation failed: {str(e)}",
                context_used=session_context or {}
            )
    
    def _extract_data_context(self, data_path: str) -> Dict[str, Any]:
        """Extract data context using CSV processor."""
        try:
            # Use the CSV processor to get structural metadata
            metadata = self.csv_processor.extract_structural_metadata(data_path)
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract data context: {e}")
            return {}
    
    def _generate_code(self, query: str, session_context: Dict = None, data_context: Dict = None) -> str:
        """Generate Python code using LLM with data context."""
        
        # Use LLM-based generation with data context
        if self.llm_service:
            return self._generate_with_llm(query, session_context, data_context)
        
        # Fallback to simple code if no LLM service
        return self._generate_simple_code(query, session_context)
    
    def _generate_with_llm(self, query: str, session_context: Dict = None, data_context: Dict = None) -> str:
        """Generate code using LLM service with data context."""
        try:
            # Build prompt with context and data context
            prompt = self._build_llm_prompt(query, session_context, data_context)
            
            # Generate code using LLM
            generated_code = self.llm_service.generate(prompt)
            
            if generated_code:
                logger.info(f"Generated code for query: {query}")
                return generated_code
            else:
                logger.warning("LLM returned empty code, using fallback")
                return self._generate_simple_code(query, session_context)
                
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return self._generate_simple_code(query, session_context)
    
    def _generate_simple_code(self, query: str, session_context: Dict = None) -> str:
        """Generate simple fallback code."""
        return f"""
# Load the data
df = pd.read_csv(data_path)

# Basic analysis
result = create_result(
    data=f"Dataset loaded with {{len(df)}} rows and {{len(df.columns)}} columns. Query: {query}",
    result_type='text',
    metadata={{
        'query': '{query}',
        'rows': len(df),
        'columns': len(df.columns)
    }}
)
"""
    
    def _build_llm_prompt(self, query: str, session_context: Dict = None, data_context: Dict = None) -> str:
        """Build prompt for LLM code generation."""
        prompt_parts = [
            f"Generate Python code to analyze CSV data for this query: {query}",
            "",
            "Requirements:",
            "- Use pandas to load and analyze the data",
            "- Use the variable 'data_path' for the CSV file path",
            "- Use the function 'create_result(data, result_type, metadata)' to return results",
            "- Focus on the specific analysis requested",
            "- Handle the data safely and efficiently",
            "- Return results that can be serialized to JSON"
        ]
        
        # Add session context if available
        if session_context:
            context_info = self._format_context(session_context)
            if context_info:
                prompt_parts.extend([
                    "",
                    "Previous context:",
                    context_info
                ])
        
        # Add data context if available
        if data_context:
            data_info = self._format_data_context(data_context)
            if data_info:
                prompt_parts.extend([
                    "",
                    "Data context:",
                    data_info
                ])
        
        return "\n".join(prompt_parts)
    
    def _format_context(self, session_context: Dict) -> str:
        """Format session context for LLM prompt."""
        context_parts = []
        
        # Add conversation history
        if session_context.get("conversation_history"):
            history = session_context["conversation_history"]
            if history:
                context_parts.append("Recent conversation:")
                for msg in history[-3:]:  # Last 3 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    context_parts.append(f"- {role}: {content}")
        
        # Add active tables info
        if session_context.get("active_tables"):
            tables = session_context["active_tables"]
            if tables:
                context_parts.append("Active tables:")
                for table_name, table_info in tables.items():
                    desc = table_info.get("description", "No description")
                    context_parts.append(f"- {table_name}: {desc}")
        
        return "\n".join(context_parts)
    
    def _format_data_context(self, data_context: Dict) -> str:
        """Format data context for LLM prompt."""
        data_parts = []
        
        # Add schema information
        if data_context.get("schema"):
            schema = data_context["schema"]
            data_parts.append(f"Dataset: {schema.get('total_rows', 'Unknown')} rows, {schema.get('total_columns', 'Unknown')} columns")
            data_parts.append(f"Columns: {', '.join(schema.get('columns', []))}")
        
        # Add column structure information
        if data_context.get("column_structure"):
            data_parts.append("Column details:")
            for col_name, col_info in data_context["column_structure"].items():
                data_type = col_info.get("data_type", "Unknown")
                cardinality = col_info.get("cardinality", "Unknown")
                completeness = col_info.get("completeness", "Unknown")
                data_parts.append(f"- {col_name}: {data_type} ({cardinality} cardinality, {completeness} completeness)")
        
        # Add operational capabilities
        if data_context.get("operational_capabilities"):
            capabilities = data_context["operational_capabilities"]
            if capabilities.get("grouping_columns"):
                data_parts.append(f"Grouping columns: {capabilities['grouping_columns']}")
            if capabilities.get("aggregation_columns"):
                data_parts.append(f"Aggregation columns: {capabilities['aggregation_columns']}")
            if capabilities.get("filtering_columns"):
                data_parts.append(f"Filtering columns: {capabilities['filtering_columns']}")
        
        # Add analysis guidance
        if data_context.get("analysis_guidance"):
            data_parts.append("Analysis guidance:")
            for guidance in data_context["analysis_guidance"]:
                data_parts.append(f"- {guidance}")
        
        return "\n".join(data_parts)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Simple code extraction - look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # If no code blocks, return the whole response
        return response.strip() 