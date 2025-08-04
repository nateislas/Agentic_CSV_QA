"""
LLM service for the CSV Analysis Platform.

This module provides OpenAI integration for code generation.
"""

import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Use absolute imports
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service for OpenAI integration and code generation.
    
    Provides a clean interface for generating Python code for data analysis.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        
        # Check if we're in test mode
        if not settings.TEST_MODE and not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client only if not in test mode
        if not settings.TEST_MODE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate Python code for data analysis based on a prompt.
        
        Args:
            prompt: The prompt describing what code to generate
            temperature: Sampling temperature (0.0 to 2.0)
            
        Returns:
            Generated Python code as a string
        """
        try:
            # If in test mode or no API key, return a simple fallback
            if settings.TEST_MODE or not self.api_key:
                return self._generate_fallback_code(prompt)
            
            # Create a system message for code generation
            system_message = """
            You are a Python code generator for data analysis. Generate clean, efficient Python code that:
            1. Loads the data first: df = pd.read_csv(data_path)
            2. Uses pandas for data manipulation (pd is already available)
            3. Handles the data safely and efficiently
            4. Returns results using create_result() function
            5. Focuses on the specific analysis requested
            
            IMPORTANT: Do NOT use import statements. The following modules are already available:
            - pandas (pd)
            - numpy (np) 
            - matplotlib.pyplot (plt)
            - seaborn (sns)
            - datetime
            - json
            - math
            - statistics
            - collections
            
            Available functions:
            - create_result(data, result_type, metadata) - Use this to return results
            
            Example:
            ```python
            # Load the data
            df = pd.read_csv(data_path)
            
            # Perform analysis
            result = create_result(
                data="Analysis result",
                result_type="text",
                metadata={"rows": len(df), "columns": len(df.columns)}
            )
            ```
            """
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            generated_code = response.choices[0].message.content
            
            # Extract code from response if it contains code blocks
            if "```python" in generated_code:
                start = generated_code.find("```python") + 9
                end = generated_code.find("```", start)
                if end > start:
                    generated_code = generated_code[start:end].strip()
            elif "```" in generated_code:
                start = generated_code.find("```") + 3
                end = generated_code.find("```", start)
                if end > start:
                    generated_code = generated_code[start:end].strip()
            
            logger.info(f"Generated code for prompt: {prompt[:100]}...")
            return generated_code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return self._generate_fallback_code(prompt)
    
    def _generate_fallback_code(self, prompt: str) -> str:
        """Generate simple fallback code when LLM is not available."""
        return f"""
# Load the data
df = pd.read_csv(data_path)

# Basic analysis
result = create_result(
    data=f"Dataset loaded with {{len(df)}} rows and {{len(df.columns)}} columns. Query: {prompt}",
    result_type="text",
    metadata={{
        "query": "{prompt}",
        "rows": len(df),
        "columns": len(df.columns)
    }}
)
"""


# Create global LLM service instance
llm_service = LLMService()
