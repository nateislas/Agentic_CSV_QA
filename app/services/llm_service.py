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
            # Debug logging to understand what's happening
            logger.info(f"LLM Service - TEST_MODE: {settings.TEST_MODE}")
            logger.info(f"LLM Service - API Key present: {bool(self.api_key)}")
            logger.info(f"LLM Service - API Key length: {len(self.api_key) if self.api_key else 0}")
            
            # If in test mode or no API key, return a simple fallback
            if settings.TEST_MODE or not self.api_key:
                logger.warning(f"Using fallback code generation. TEST_MODE={settings.TEST_MODE}, API_KEY_PRESENT={bool(self.api_key)}")
                return self._generate_fallback_code(prompt)
            
            logger.info("Using actual LLM for code generation")
            
            # Create a system message for code generation
            system_message = """
            You are a data analysis agent. For each query, follow this EXACT structure:

            # REASONING:
            # Explain your understanding of the query and your plan
            # - What does the user want?
            # - What columns/data will you use?
            # - What operations will you perform?
            # - How will you format the result?

            # CODE:
            # Generate the Python code to execute your plan

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
            
            OUTPUT FORMAT EXAMPLES:
            
            For column names:
            ```
            # REASONING:
            # The user wants to see column names. I will:
            # - Load the CSV data
            # - Extract the column names
            # - Return them as a list

            # CODE:
            df = pd.read_csv(data_path)
            column_names = list(df.columns)
            result = create_result(
                data=column_names,
                result_type="text",
                metadata={"query": "column_names", "total_columns": len(column_names)}
            )
            ```
            
            For top analysis:
            ```
            # REASONING:
            # The user wants to see the top 5 boroughs by crime count. I will:
            # - Load the CSV data
            # - Group by borough column
            # - Sum the crime counts
            # - Sort in descending order
            # - Take top 5 results
            # - Return as a table

            # CODE:
            df = pd.read_csv(data_path)
            grouped_data = df.groupby('borough')['value'].sum().sort_values(ascending=False)
            top_5_boroughs = grouped_data.head(5)
            result = create_result(
                data=top_5_boroughs.to_dict(),
                result_type="table",
                metadata={"query": "top_5_boroughs", "total_boroughs": len(grouped_data)}
            )
            ```
            
            For visualization:
            ```
            # REASONING:
            # The user wants to create a bar chart of crime by borough. I will:
            # - Load the CSV data
            # - Group by borough and sum crime counts
            # - Create a bar chart
            # - Return the plot

            # CODE:
            df = pd.read_csv(data_path)
            grouped_data = df.groupby('borough')['value'].sum().sort_values(ascending=False)
            
            plt.figure(figsize=(14, 8))
            grouped_data.plot(kind='bar')
            plt.title('Crime Count by Borough')
            plt.xlabel('Borough')
            plt.ylabel('Total Crime Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            result = create_result(
                data="plot_generated",
                result_type="plot",
                metadata={"query": "bar_chart", "plot_type": "bar_chart"}
            )
            ```
            
            ALWAYS use this exact format with # REASONING: and # CODE: sections.
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
            
            generated_response = response.choices[0].message.content
            
            # Return the full response for reasoning extraction
            logger.info(f"Generated response for prompt: {prompt[:100]}...")
            return generated_response
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return self._generate_fallback_code(prompt)
    
    def _generate_fallback_code(self, prompt: str) -> str:
        """Generate simple fallback code when LLM is not available."""
        
        # Handle common queries more intelligently
        prompt_lower = prompt.lower()
        
        if "column" in prompt_lower and ("name" in prompt_lower or "names" in prompt_lower):
            return """
# REASONING:
# The user wants to see the column names of the dataset. I will:
# - Load the CSV data using pd.read_csv()
# - Extract the column names using df.columns
# - Return them as a list for easy reading

# CODE:
df = pd.read_csv(data_path)
column_names = list(df.columns)
result = create_result(
    data=column_names,
    result_type="text",
    metadata={
        "query": "column_names",
        "total_columns": len(column_names),
        "columns": column_names
    }
)
"""
        
        elif "plot" in prompt_lower or "chart" in prompt_lower or "graph" in prompt_lower or "barplot" in prompt_lower or "histogram" in prompt_lower:
            # Handle plotting requests
            return """
# REASONING:
# The user wants to create a visualization. I will:
# - Load the CSV data
# - Identify categorical and numeric columns for plotting
# - Create a bar chart using the first categorical column for grouping
# - Use the first numeric column for values
# - Return the plot as a visualization

# CODE:
df = pd.read_csv(data_path)

# Create a plot based on the request
plt.figure(figsize=(12, 6))

# Try to create a bar plot based on available columns
# Look for categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
numeric_cols = df.select_dtypes(include=['number']).columns

if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    # Use first categorical column for grouping, first numeric for values
    group_col = categorical_cols[0]
    value_col = numeric_cols[0]
    
    # Group by categorical column and sum the numeric values
    grouped_data = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    grouped_data.plot(kind='bar')
    plt.title(f'{value_col} by {group_col}')
    plt.xlabel(group_col)
    plt.ylabel(f'Total {value_col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Return plot result
    result = create_result(
        data="plot_generated",
        result_type="plot",
        metadata={
            "query": "barplot_request",
            "figure_count": len(plt.get_fignums()),
            "plot_type": "bar_chart",
            "group_column": group_col,
            "value_column": value_col
        }
    )
else:
    # Fallback if suitable columns not found
    result = create_result(
        data=f"Could not create plot. Available columns: {list(df.columns)}. Need at least one categorical and one numeric column.",
        result_type="text",
        metadata={
            "query": "plot_request",
            "available_columns": list(df.columns)
        }
    )
"""
        
        elif "filter" in prompt_lower or "where" in prompt_lower or "for" in prompt_lower:
            # Try to extract filtering information from the prompt
            return """
# REASONING:
# The user wants to filter the data. Since this is a fallback implementation,
# I will show basic dataset information and sample data to help understand the structure.
# For better filtering results, the actual LLM service should be used.

# CODE:
df = pd.read_csv(data_path)

# Basic filtering - this is a fallback implementation
# For better results, use the actual LLM service

# Show sample of data for context
sample_data = df.head(10).to_dict('records')

result = create_result(
    data=f"Fallback: Dataset has {{len(df)}} rows and {{len(df.columns)}} columns. Query: {prompt}",
    result_type="text",
    metadata={
        "query": "fallback_filter",
        "rows": len(df),
        "columns": list(df.columns),
        "sample_data": sample_data
    }
)
"""
        
        else:
            # Try to generate more intelligent fallback code based on the query
            return f"""
# REASONING:
# The user has asked: "{prompt}". I will analyze this query and provide a meaningful response.
# I'll check for common patterns like "top", "count", "average", "summary", "unique" and respond accordingly.
# If no specific pattern is found, I'll provide general dataset information.

# CODE:
df = pd.read_csv(data_path)

# Analyze the query and attempt to provide a meaningful response
query_lower = "{prompt.lower()}"

# Check for common analysis patterns
if "top" in query_lower:
    # Handle top N queries
    # Look for categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        # Use first categorical column for grouping, first numeric for values
        group_col = categorical_cols[0]
        value_col = numeric_cols[0]
        
        top_results = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(10)
        result = create_result(
            data=top_results.to_dict(),
            result_type="table",
            metadata={{
                "query": "{prompt}",
                "analysis_type": "top_analysis",
                "group_column": group_col,
                "value_column": value_col,
                "total_rows": len(df)
            }}
        )
    else:
        result = create_result(
            data=f"Could not perform top analysis. Need at least one categorical and one numeric column. Available columns: {{list(df.columns)}}",
            result_type="text",
            metadata={{"query": "{prompt}", "available_columns": list(df.columns)}}
        )

elif "count" in query_lower or "how many" in query_lower:
    # Handle count queries
    result = create_result(
        data=f"Total rows in dataset: {{len(df)}}",
        result_type="text",
        metadata={{"query": "{prompt}", "total_rows": len(df)}}
    )

elif "average" in query_lower or "mean" in query_lower:
    # Handle average queries
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        averages = df[numeric_cols].mean().to_dict()
        result = create_result(
            data=averages,
            result_type="table",
            metadata={{"query": "{prompt}", "analysis_type": "average"}}
        )
    else:
        result = create_result(
            data="No numeric columns found for averaging",
            result_type="text",
            metadata={{"query": "{prompt}", "available_columns": list(df.columns)}}
        )

elif "summary" in query_lower or "describe" in query_lower:
    # Handle summary queries
    summary_stats = df.describe().to_dict()
    result = create_result(
        data=summary_stats,
        result_type="table",
        metadata={{"query": "{prompt}", "analysis_type": "summary"}}
    )

elif "unique" in query_lower or "distinct" in query_lower:
    # Handle unique value queries
    # Look for categorical columns first
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        # Use first categorical column
        unique_values = df[categorical_cols[0]].unique().tolist()
        result = create_result(
            data=unique_values,
            result_type="table",
            metadata={{"query": "{prompt}", "analysis_type": "unique_values", "column": categorical_cols[0]}}
        )
    else:
        # Try first column
        unique_values = df.iloc[:, 0].unique().tolist()
        result = create_result(
            data=unique_values,
            result_type="table",
            metadata={{"query": "{prompt}", "analysis_type": "unique_values", "column": df.columns[0]}}
        )

else:
    # Generic analysis based on available columns
    if len(df) > 0:
        # Show sample data and basic info
        sample_data = df.head(5).to_dict('records')
        result = create_result(
            data={{
                "message": f"Dataset analysis for: {{prompt}}",
                "total_rows": len(df),
                "columns": list(df.columns),
                "sample_data": sample_data
            }},
            result_type="table",
            metadata={{"query": "{prompt}", "analysis_type": "general"}}
        )
    else:
        result = create_result(
            data="Empty dataset or no data found",
            result_type="text",
            metadata={{"query": "{prompt}"}}
        )
"""


# Create global LLM service instance
llm_service = LLMService()
