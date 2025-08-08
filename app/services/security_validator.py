"""
Security Validator for Code Generation.

This module provides comprehensive security validation for generated Python code
to ensure safe execution in the sandboxed environment.
"""

import ast
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import math
import statistics
import collections
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    allowed_imports: Set[str]
    dangerous_patterns: List[str]


class CodeSecurityValidator:
    """Validates generated Python code for security compliance."""
    
    def __init__(self):
        # Allowed imports for data analysis
        self.allowed_imports = {
            'pandas': 'pd',
            'numpy': 'np',
            'matplotlib.pyplot': 'plt',
            'seaborn': 'sns',
            'datetime': 'datetime',
            'json': 'json',
            'math': 'math',
            'statistics': 'statistics',
            'collections': 'collections'
        }
        
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = [
            # File operations
            r'open\(',
            r'file\(',
            r'\.read\(\)',
            r'\.write\(',
            r'\.save\(',
            
            # System operations
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'import\s+shutil',
            r'import\s+glob',
            r'import\s+pathlib',
            
            # Network operations
            r'import\s+requests',
            r'import\s+urllib',
            r'import\s+socket',
            r'import\s+ftplib',
            r'import\s+http',
            
            # Code execution
            r'eval\(',
            r'exec\(',
            r'__import__\(',
            r'compile\(',
            
            # Pickle and serialization
            r'import\s+pickle',
            r'pickle\.',
            r'import\s+cpickle',
            
            # Other dangerous modules
            r'import\s+ctypes',
            r'import\s+multiprocessing',
            r'import\s+threading',
            r'import\s+tempfile',
            
            # Direct attribute access
            r'\.__dict__',
            r'\.__class__',
            r'\.__bases__',
            r'\.__subclasses__',
            
            # Shell commands
            r'os\.system\(',
            r'subprocess\.',
            r'\.shell\(',
            
            # File path manipulation (removed overly restrictive patterns)
            # r'\.join\(',
            # r'\.split\(',
            # r'\.replace\(',
            # r'\.strip\(',
            
            # Database operations
            r'import\s+sqlite3',
            r'import\s+psycopg2',
            r'import\s+mysql',
            
            # Other dangerous operations
            r'\.format\(',
            # r'f"',  # Allow f-strings for now
            r'\.encode\(',
            r'\.decode\(',
        ]
        
        # Compile regex patterns for efficiency
        self.dangerous_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
        # AST node types to block (Python 3.9 compatible)
        self.dangerous_ast_nodes = {
            ast.Call,
            ast.Import,
            ast.ImportFrom,
            ast.Attribute,
            ast.Subscript,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.If,
            ast.For,
            ast.While,
            ast.Try,
            ast.With,
            ast.FunctionDef,
            ast.ClassDef,
            ast.Return,
            ast.Assign,
            ast.AugAssign,
            ast.Expr,
            ast.Pass,
            ast.Break,
            ast.Continue,
            ast.Raise,
            ast.Assert,
            ast.Delete,
            ast.Global,
            ast.Nonlocal,
            ast.AsyncFunctionDef,
            ast.AsyncFor,
            ast.AsyncWith,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Lambda,
            ast.IfExp,
            ast.NamedExpr,
        }
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Comprehensive validation of generated Python code.
        
        Args:
            code: Python code string to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        allowed_imports_found = set()
        dangerous_patterns_found = []
        
        try:
            # 1. Syntax validation
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
                return ValidationResult(False, errors, warnings, allowed_imports_found, dangerous_patterns_found)
            
            # 2. AST analysis (simplified for now)
            try:
                tree = ast.parse(code)
                ast_errors = self._validate_ast(tree)
                errors.extend(ast_errors)
            except Exception as e:
                errors.append(f"AST parsing error: {e}")
            
            # 3. Import validation
            import_errors, import_warnings, found_imports = self._validate_imports(tree)
            errors.extend(import_errors)
            warnings.extend(import_warnings)
            allowed_imports_found.update(found_imports)
            
            # 4. Pattern scanning
            pattern_errors, found_patterns = self._scan_dangerous_patterns(code)
            errors.extend(pattern_errors)
            dangerous_patterns_found.extend(found_patterns)
            
            # 5. Additional security checks
            security_errors = self._additional_security_checks(code)
            errors.extend(security_errors)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                allowed_imports=allowed_imports_found,
                dangerous_patterns=dangerous_patterns_found
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            errors.append(f"Validation failed: {e}")
            return ValidationResult(False, errors, warnings, allowed_imports_found, dangerous_patterns_found)
    
    def _validate_ast(self, tree: ast.AST) -> List[str]:
        """Validate AST for dangerous constructs."""
        errors = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile', '__import__']:
                        errors.append(f"Dangerous function call: {func_name}")
                
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name in ['system', 'popen', 'call']:
                        errors.append(f"Dangerous method call: {attr_name}")
            
            # Check for dangerous imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys', 'shutil']:
                        errors.append(f"Dangerous import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in ['os', 'subprocess', 'sys', 'shutil']:
                    errors.append(f"Dangerous import from: {node.module}")
        
        return errors
    
    def _validate_imports(self, tree: ast.AST) -> Tuple[List[str], List[str], Set[str]]:
        """Validate imports and return allowed imports found."""
        errors = []
        warnings = []
        allowed_imports_found = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.allowed_imports:
                        allowed_imports_found.add(alias.name)
                    else:
                        errors.append(f"Import not allowed: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.allowed_imports:
                    allowed_imports_found.add(node.module)
                else:
                    errors.append(f"Import from not allowed: {node.module}")
        
        return errors, warnings, allowed_imports_found
    
    def _scan_dangerous_patterns(self, code: str) -> Tuple[List[str], List[str]]:
        """Scan code for dangerous patterns."""
        errors = []
        found_patterns = []
        
        for i, pattern in enumerate(self.dangerous_regex):
            matches = pattern.findall(code)
            if matches:
                pattern_name = self.dangerous_patterns[i]
                errors.append(f"Dangerous pattern found: {pattern_name}")
                found_patterns.append(pattern_name)
        
        return errors, found_patterns
    
    def _additional_security_checks(self, code: str) -> List[str]:
        """Additional security checks."""
        errors = []
        
        # Check for exec/eval usage
        if 'exec(' in code or 'eval(' in code:
            errors.append("exec() or eval() usage detected")
        
        # Check for file operations
        if 'open(' in code:
            errors.append("File operations detected")
        
        # Check for network operations
        if 'requests.' in code or 'urllib.' in code:
            errors.append("Network operations detected")
        
        return errors
    
    def get_safe_globals(self) -> Dict:
        """Get safe globals for execution environment."""
        
        return {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'datetime': datetime,
            'json': json,
            'math': math,
            'statistics': statistics,
            'collections': collections,
            '__builtins__': self._get_safe_builtins()
        }
    
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
            'Exception': Exception,
            'BaseException': BaseException,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'AttributeError': AttributeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'NameError': NameError,
            'UnboundLocalError': UnboundLocalError,
            'RuntimeError': RuntimeError,
            'MemoryError': MemoryError,
            'OverflowError': OverflowError,
            'ZeroDivisionError': ZeroDivisionError,
            'AssertionError': AssertionError,
            'NotImplementedError': NotImplementedError,
            'ArithmeticError': ArithmeticError,
            'BufferError': BufferError,
            'LookupError': LookupError,
            'OSError': OSError,
            'ReferenceError': ReferenceError,
            'SyntaxError': SyntaxError,
            'SystemError': SystemError,
            'TabError': TabError,
            'UnicodeError': UnicodeError,
            'UnicodeDecodeError': UnicodeDecodeError,
            'UnicodeEncodeError': UnicodeEncodeError,
            'UnicodeTranslateError': UnicodeTranslateError,
            'Warning': Warning,
            'UserWarning': UserWarning,
            'DeprecationWarning': DeprecationWarning,
            'PendingDeprecationWarning': PendingDeprecationWarning,
            'SyntaxWarning': SyntaxWarning,
            'RuntimeWarning': RuntimeWarning,
            'FutureWarning': FutureWarning,
            'ImportWarning': ImportWarning,
            'UnicodeWarning': UnicodeWarning,
            'BytesWarning': BytesWarning,
            'ResourceWarning': ResourceWarning,
        }
        
        return safe_builtins 