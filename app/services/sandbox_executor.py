"""
Sandboxed Code Executor.

This module provides a secure execution environment for generated Python code
with resource limits, timeout protection, and isolation.
"""

import time
import signal
import logging
import traceback
import resource
import threading
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    memory_used: int
    stdout: str
    stderr: str


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


class SandboxExecutor:
    """Secure sandboxed executor for Python code."""
    
    def __init__(self, timeout: int = 30, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.validator = None  # Will be set by the calling service
        
    def set_validator(self, validator):
        """Set the security validator."""
        self.validator = validator
    
    def execute(self, code: str, data_path: str, session_context: Dict = None) -> ExecutionResult:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            data_path: Path to the CSV file
            session_context: Optional session context for multi-turn conversations
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Capture stdout/stderr
        stdout_capture = []
        stderr_capture = []
        
        try:
            # Validate code first
            if self.validator:
                validation_result = self.validator.validate_code(code)
                if not validation_result.is_valid:
                    return ExecutionResult(
                        success=False,
                        result=None,
                        error=f"Code validation failed: {'; '.join(validation_result.errors)}",
                        execution_time=time.time() - start_time,
                        memory_used=0,
                        stdout="",
                        stderr="; ".join(validation_result.errors)
                    )
            
            # Create execution environment
            globals_dict = self._create_execution_environment(data_path, session_context)
            locals_dict = {}
            
            # Import pandas for DataFrame handling
            import pandas as pd
            
            # Execute with timeout
            with self._timeout_context(self.timeout):
                # Temporarily disable memory limit to focus on code generation
                # with self._memory_limit_context(self.memory_limit_mb):
                # Capture stdout/stderr
                with self._capture_output(stdout_capture, stderr_capture):
                    exec(code, globals_dict, locals_dict)
            
            # Extract results
            result = self._extract_results(locals_dict)
            
            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return ExecutionResult(
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                memory_used=memory_used,
                stdout="\n".join(stdout_capture),
                stderr="\n".join(stderr_capture)
            )
            
        except TimeoutError:
            return ExecutionResult(
                success=False,
                result=None,
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=time.time() - start_time,
                memory_used=0,
                stdout="\n".join(stdout_capture),
                stderr="Execution timed out"
            )
            
        except MemoryError:
            return ExecutionResult(
                success=False,
                result=None,
                error=f"Memory limit exceeded ({self.memory_limit_mb}MB)",
                execution_time=time.time() - start_time,
                memory_used=0,
                stdout="\n".join(stdout_capture),
                stderr="Memory limit exceeded"
            )
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return ExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                memory_used=0,
                stdout="\n".join(stdout_capture),
                stderr=error_msg
            )
    
    def _create_execution_environment(self, data_path: str, session_context: Dict = None) -> Dict:
        """Create the execution environment with safe globals."""
        if self.validator:
            globals_dict = self.validator.get_safe_globals()
        else:
            # Fallback if no validator
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            globals_dict = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                '__builtins__': self._get_safe_builtins()
            }
        
        # Add data loading
        globals_dict['data_path'] = data_path
        globals_dict['df'] = None  # Will be loaded by the code
        
        # Add session context if available
        if session_context:
            globals_dict['session_context'] = session_context
        
        # Add helper functions
        globals_dict['create_result'] = self._create_result_helper
        
        return globals_dict
    
    def _extract_results(self, locals_dict: Dict) -> Dict:
        """Extract results from the execution environment."""
        import pandas as pd
        
        result = {
            'type': 'text',
            'data': None,
            'metadata': {}
        }
        
        # First, look for the result variable (highest priority)
        if 'result' in locals_dict:
            value = locals_dict['result']
            if isinstance(value, dict):
                result = value
            else:
                result['data'] = str(value)
            return result
        
        # Look for other common result variables
        for var_name in ['output', 'analysis_result', 'data']:
            if var_name in locals_dict:
                value = locals_dict[var_name]
                if isinstance(value, dict):
                    result = value
                else:
                    result['data'] = str(value)
                break
        
        # Look for DataFrame results (lowest priority)
        for var_name in ['df', 'result_df', 'analysis_df']:
            if var_name in locals_dict:
                df = locals_dict[var_name]
                if hasattr(df, 'to_dict'):  # pandas DataFrame
                    # Clean DataFrame for JSON serialization
                    df_clean = df.copy()
                    
                    # Replace NaN values with None for JSON compatibility
                    df_clean = df_clean.where(pd.notnull(df_clean), None)
                    
                    # Convert to records with proper handling of non-serializable values
                    try:
                        data_records = df_clean.to_dict('records')
                        # Additional cleaning for any remaining non-serializable values
                        for record in data_records:
                            for key, value in record.items():
                                if pd.isna(value) or (isinstance(value, float) and (value == float('inf') or value == float('-inf'))):
                                    record[key] = None
                    except Exception as e:
                        # Fallback: convert to string representation
                        data_records = [{"error": f"Data serialization failed: {str(e)}"}]
                    
                    result = {
                        'type': 'table',
                        'data': data_records,
                        'columns': df.columns.tolist(),
                        'shape': df.shape,
                        'metadata': {
                            'total_rows': len(df),
                            'total_columns': len(df.columns)
                        }
                    }
                break
        
        # Look for plot results
        if 'plt' in locals_dict and hasattr(locals_dict['plt'], 'gcf'):
            fig = locals_dict['plt'].gcf()
            if fig and len(fig.axes) > 0:
                result = {
                    'type': 'plot',
                    'data': 'plot_generated',
                    'metadata': {
                        'figure_count': len(locals_dict['plt'].get_fignums()),
                        'axes_count': len(fig.axes)
                    }
                }
        
        return result
    
    def _create_result_helper(self, data, result_type='text', metadata=None):
        """Helper function for creating standardized results."""
        return {
            'type': result_type,
            'data': data,
            'metadata': metadata or {}
        }
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """Context manager for timeout protection."""
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    @contextmanager
    def _memory_limit_context(self, memory_limit_mb: int):
        """Context manager for memory limit protection."""
        try:
            # Convert MB to bytes
            memory_limit_bytes = memory_limit_mb * 1024 * 1024
            
            # Get current limits
            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Set memory limit (use current hard limit if it's lower)
            new_limit = min(memory_limit_bytes, current_hard)
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, current_hard))
            
            yield
            
        except Exception as e:
            # Log the error but don't fail
            logger.warning(f"Memory limit setting failed: {e}")
            yield
        finally:
            try:
                # Reset memory limit
                resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
            except Exception as e:
                logger.warning(f"Failed to reset memory limit: {e}")
    
    @contextmanager
    def _capture_output(self, stdout_capture: list, stderr_capture: list):
        """Context manager for capturing stdout/stderr."""
        import sys
        from io import StringIO
        
        # Create string buffers
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        # Store original streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # Redirect streams
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            yield
            
        finally:
            # Restore streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Capture output
            stdout_capture.append(stdout_buffer.getvalue())
            stderr_capture.append(stderr_buffer.getvalue())
            
            # Close buffers
            stdout_buffer.close()
            stderr_buffer.close()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to resource module
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    
    def _get_safe_builtins(self) -> Dict:
        """Get safe builtins for execution."""
        safe_builtins = {
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'dir': dir,
            'print': print,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'chr': chr,
            'ord': ord,
            'hex': hex,
            'oct': oct,
            'bin': bin,
            'format': format,
            'repr': repr,
            'ascii': ascii,
            'hash': hash,
            'id': id,
            'callable': callable,
            'issubclass': issubclass,
            'super': super,
            'property': property,
            'staticmethod': staticmethod,
            'classmethod': classmethod,
            'object': object,
        }
        
        return safe_builtins 