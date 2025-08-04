"""
Code Execution Tool for LangChain Agent.

This tool provides a single interface for executing Python code for CSV data analysis,
integrating with the existing sandbox executor for secure code execution.
"""

import logging
from typing import Any, Dict, Optional
from langchain_community.tools import Tool
from app.services.sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)


def execute_code(code: str, data_path: str, session_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute Python code for CSV data analysis.
    
    Args:
        code: Python code to execute
        data_path: Path to the CSV file to analyze
        session_context: Optional session context for multi-turn conversations
        
    Returns:
        String result of code execution or error details
    """
    sandbox_executor = SandboxExecutor()
    
    try:
        logger.info(f"Executing code for data_path: {data_path}")
        logger.debug(f"Code to execute: {code}")
        
        result = sandbox_executor.execute(
            code=code,
            data_path=data_path,
            session_context=session_context
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


def _format_success_result(result) -> str:
    """Format successful execution result."""
    execution_result = result.execution_result.result
    
    # Extract the result data and metadata
    result_data = execution_result.get('data', 'No data returned')
    result_type = execution_result.get('type', 'text')
    metadata = execution_result.get('metadata', {})
    
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


def _format_error_result(result) -> str:
    """Format error result with detailed information for agent analysis."""
    error_msg = result.error or "Unknown error occurred"
    stdout = result.execution_result.stdout if result.execution_result else ""
    stderr = result.execution_result.stderr if result.execution_result else ""
    
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