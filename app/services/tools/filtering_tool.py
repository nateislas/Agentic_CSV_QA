"""
Filtering Tool for CSV Analysis Agent.

This tool allows the agent to filter CSV data based on various conditions.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


class FilteringInput(BaseModel):
    """Input schema for filtering tool."""
    file_path: str = Field(..., description="Path to the CSV file")
    column: str = Field(..., description="Column to filter on")
    condition: str = Field(..., description="Filter condition: 'equals', 'not_equals', 'greater_than', 'less_than', 'contains', 'in_list'")
    value: str = Field(..., description="Value to filter by")
    limit: Optional[int] = Field(None, description="Limit number of results")


class FilteringTool(BaseTool):
    """Tool for filtering CSV data based on conditions."""
    
    name: str = "filtering"
    description: str = """
    Filter CSV data based on various conditions such as equals, not equals, greater than, less than, contains, in list.
    Use this tool to subset data for analysis.
    
    Conditions:
    - equals: Exact match
    - not_equals: Not equal to value
    - greater_than: Greater than value (for numeric)
    - less_than: Less than value (for numeric)
    - contains: Contains substring (for text)
    - in_list: Value is in list (comma-separated)
    """
    args_schema = FilteringInput
    
    def _run(self, file_path: str, column: str, condition: str, value: str, limit: Optional[int] = None) -> str:
        """Execute the filtering operation."""
        try:
            logger.info(f"Filtering: {condition} on {column} in {file_path}")
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Validate column exists
            if column not in df.columns:
                return f"Error: Column '{column}' not found in the dataset"
            
            # Apply filter
            filtered_df = self._apply_filter(df, column, condition, value)
            
            if filtered_df is None:
                return f"Error: Invalid filter condition '{condition}'"
            
            # Apply limit if specified
            if limit and limit > 0:
                filtered_df = filtered_df.head(limit)
            
            return f"Filtered results ({len(filtered_df)} rows):\n{filtered_df.to_string(index=False)}"
            
        except Exception as e:
            logger.error(f"Filtering error: {e}")
            return f"Error during filtering: {str(e)}"
    
    def _apply_filter(self, df: pd.DataFrame, column: str, condition: str, value: str) -> Optional[pd.DataFrame]:
        """Apply filter condition to dataframe."""
        try:
            if condition == "equals":
                return df[df[column] == value]
            elif condition == "not_equals":
                return df[df[column] != value]
            elif condition == "greater_than":
                try:
                    numeric_value = float(value)
                    return df[df[column] > numeric_value]
                except ValueError:
                    return None
            elif condition == "less_than":
                try:
                    numeric_value = float(value)
                    return df[df[column] < numeric_value]
                except ValueError:
                    return None
            elif condition == "contains":
                return df[df[column].astype(str).str.contains(value, na=False)]
            elif condition == "in_list":
                values = [v.strip() for v in value.split(",")]
                return df[df[column].isin(values)]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return None 