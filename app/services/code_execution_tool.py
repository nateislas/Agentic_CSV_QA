"""
Code Execution Tool for LangChain Agent.

This tool provides a single interface for executing Python code for CSV data analysis,
integrating with the existing sandbox executor for secure code execution.
"""

import logging
import re
from typing import Any, Dict, Optional
from langchain_community.tools import Tool
from app.services.sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)


def execute_code(code: str) -> str:
    """
    Execute Python code for CSV data analysis.
    
    Args:
        code: Python code to execute (the agent will pass this as a string)
        
    Returns:
        String result of code execution or error details
    """
    sandbox_executor = SandboxExecutor()
    
    try:
        # Clean the code by removing markdown formatting
        cleaned_code = _clean_code_input(code)
        
        # For now, we'll use a default data path
        # In a real implementation, this would come from the agent's context
        data_path = "test.csv"  # Default for testing
        
        logger.info(f"Executing code for data_path: {data_path}")
        logger.debug(f"Code to execute: {cleaned_code}")
        
        result = sandbox_executor.execute(
            code=cleaned_code,
            data_path=data_path,
            session_context=None
        )
        
        if result.success:
            logger.info("Code execution successful")
            return _format_success_result(result)
        else:
            logger.warning(f"Code execution failed: {result.error}")
            return _format_error_result(result)
            
    except Exception as e:
        logger.error(f"Unexpected error in code execution: {e}")
        return _format_unexpected_error(str(e))


def _clean_code_input(code: str) -> str:
    """Clean the code input by removing markdown formatting."""
    # Remove markdown code blocks
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*$', '', code)
    code = re.sub(r'^```\s*', '', code)
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    return code


def _format_success_result(result) -> str:
    """Format successful execution result."""
    try:
        # Handle the actual ExecutionResult structure
        if hasattr(result, 'execution_result') and result.execution_result:
            execution_result = result.execution_result.result
        else:
            # If no execution_result, try to get result directly
            execution_result = result.result if hasattr(result, 'result') else result
        
        # Extract the result data and metadata
        if isinstance(execution_result, dict):
            result_data = execution_result.get('data', 'No data returned')
            result_type = execution_result.get('type', 'text')
            metadata = execution_result.get('metadata', {})
        else:
            # If it's not a dict, treat it as text
            result_data = str(execution_result)
            result_type = 'text'
            metadata = {}
        
        # Format based on result type
        if result_type == 'plot':
            if metadata.get('plot_encoded') and isinstance(result_data, str) and result_data.startswith('data:image/png'):
                return f"SUCCESS: Plot generated. {metadata.get('figure_count', 1)} figure(s) created."
            else:
                return f"SUCCESS: Plot generated. {metadata.get('figure_count', 1)} figure(s) created."
        
        elif result_type == 'table':
            if isinstance(result_data, dict):
                return f"SUCCESS: Table generated with {len(result_data)} rows."
            elif isinstance(result_data, list):
                return f"SUCCESS: Table generated with {len(result_data)} rows."
            else:
                return f"SUCCESS: Table generated."
        
        elif result_type == 'text':
            return f"SUCCESS: {result_data}"
        
        else:
            return f"SUCCESS: Analysis completed. Result type: {result_type}"
            
    except Exception as e:
        logger.error(f"Error formatting success result: {e}")
        return f"SUCCESS: Code executed successfully. Result: {str(result)}"


def _format_error_result(result) -> str:
    """Format error result with detailed information for agent analysis."""
    try:
        error_msg = result.error if hasattr(result, 'error') else "Unknown error occurred"
        
        # Get stdout and stderr if available
        stdout = ""
        stderr = ""
        if hasattr(result, 'execution_result') and result.execution_result:
            stdout = result.execution_result.stdout if hasattr(result.execution_result, 'stdout') else ""
            stderr = result.execution_result.stderr if hasattr(result.execution_result, 'stderr') else ""
        
        error_details = f"""
ERROR: Code execution failed

Error Message: {error_msg}

Standard Output:
{stdout}

Standard Error:
{stderr}

Please analyze this error and generate corrected code.
"""
        return error_details.strip()
        
    except Exception as e:
        logger.error(f"Error formatting error result: {e}")
        return f"ERROR: Code execution failed. Error: {str(result)}"


def _format_unexpected_error(error_msg: str) -> str:
    """Format unexpected error."""
    return f"""
UNEXPECTED ERROR: {error_msg}

This is an unexpected error in the code execution system. 
Please try a different approach or contact support.
""".strip()


# Create the LangChain tool
code_execution_tool = Tool(
    name="execute_code",
    func=execute_code,
    description=(
        "Execute Python code for CSV data analysis. "
        "Use this tool when you need to: load/analyze CSV data, perform transformations, "
        "create visualizations, generate summaries, filter/aggregate data. "
        "The code will be executed in a secure sandbox environment with pandas, numpy, matplotlib, seaborn, etc."
    )
) 