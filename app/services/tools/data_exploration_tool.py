"""
Data Exploration Tool for CSV Analysis Agent.

This tool allows the agent to explore CSV data structure and get information
about columns, data types, and sample data without making domain assumptions.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

from app.services.csv_processor import csv_processor

logger = logging.getLogger(__name__)


class DataExplorationInput(BaseModel):
    """Input schema for data exploration tool."""
    file_path: str = Field(..., description="Path to the CSV file")
    operation: str = Field(..., description="Type of exploration: 'column_info', 'sample_data', 'data_types', 'summary'")
    column_name: Optional[str] = Field(None, description="Specific column to explore (optional)")


class DataExplorationTool(BaseTool):
    """Tool for exploring CSV data structure and content."""
    
    name: str = "data_exploration"
    description: str = """
    Explore CSV data structure and get information about columns, data types, and sample data.
    Use this tool to understand the structure of the dataset before performing analysis.
    
    Operations:
    - column_info: Get detailed information about all columns
    - sample_data: Get sample data from the dataset
    - data_types: Get data type information for all columns
    - summary: Get a summary of the dataset structure
    """
    args_schema = DataExplorationInput
    
    def _run(self, file_path: str, operation: str, column_name: Optional[str] = None) -> str:
        """Execute the data exploration operation."""
        try:
            logger.info(f"Data exploration: {operation} on {file_path}")
            
            if operation == "column_info":
                return self._get_column_info(file_path, column_name)
            elif operation == "sample_data":
                return self._get_sample_data(file_path, column_name)
            elif operation == "data_types":
                return self._get_data_types(file_path)
            elif operation == "summary":
                return self._get_summary(file_path)
            else:
                return f"Unknown operation: {operation}. Available operations: column_info, sample_data, data_types, summary"
                
        except Exception as e:
            logger.error(f"Data exploration error: {e}")
            return f"Error during data exploration: {str(e)}"
    
    def _get_column_info(self, file_path: str, column_name: Optional[str] = None) -> str:
        """Get detailed information about columns."""
        try:
            # Get metadata
            result = csv_processor.process_csv_file(file_path)
            if not result["success"]:
                return f"Failed to get column info: {result.get('error')}"
            
            metadata = result["metadata"]
            column_analysis = metadata["column_analysis"]
            
            if column_name:
                if column_name not in column_analysis:
                    return f"Column '{column_name}' not found in the dataset."
                
                col_info = column_analysis[column_name]
                return self._format_column_info(column_name, col_info)
            else:
                # Return info for all columns
                info_parts = ["Column Information:"]
                for col_name, col_info in column_analysis.items():
                    info_parts.append(self._format_column_info(col_name, col_info))
                
                return "\n\n".join(info_parts)
                
        except Exception as e:
            return f"Error getting column info: {str(e)}"
    
    def _format_column_info(self, column_name: str, col_info: Dict[str, Any]) -> str:
        """Format column information for display."""
        data_type = col_info.get("data_type_category", "Unknown")
        cardinality = col_info.get("cardinality", {})
        completeness = col_info.get("completeness", {})
        characteristics = col_info.get("characteristics", {})
        
        info_parts = [
            f"Column: {column_name}",
            f"  Data Type: {data_type}",
            f"  Cardinality: {cardinality.get('cardinality_level', 'Unknown')} ({cardinality.get('unique_count', 0)} unique values)",
            f"  Completeness: {completeness.get('completeness_level', 'Unknown')} ({completeness.get('completeness_ratio', 0):.1%} complete)"
        ]
        
        # Add type-specific characteristics
        if data_type == "numeric" and characteristics:
            if "mean" in characteristics:
                info_parts.extend([
                    f"  Range: {characteristics.get('min', 'N/A')} to {characteristics.get('max', 'N/A')}",
                    f"  Mean: {characteristics.get('mean', 'N/A')}",
                    f"  Standard Deviation: {characteristics.get('std', 'N/A')}"
                ])
        elif data_type == "categorical" and characteristics:
            if "most_common_values" in characteristics:
                common_values = list(characteristics.get("most_common_values", {}).keys())[:3]
                info_parts.append(f"  Most Common Values: {', '.join(common_values)}")
        
        return "\n".join(info_parts)
    
    def _get_sample_data(self, file_path: str, column_name: Optional[str] = None) -> str:
        """Get sample data from the dataset."""
        try:
            # Read CSV with polars
            df = pd.read_csv(file_path, nrows=10)  # Get first 10 rows
            
            if column_name:
                if column_name not in df.columns:
                    return f"Column '{column_name}' not found in the dataset."
                
                sample_data = df[column_name].head(10).to_list()
                return f"Sample data for column '{column_name}':\n{', '.join(map(str, sample_data))}"
            else:
                # Return sample of all columns
                sample_df = df.head(5)  # First 5 rows
                return f"Sample data (first 5 rows):\n{sample_df.to_pandas().to_string()}"
                
        except Exception as e:
            return f"Error getting sample data: {str(e)}"
    
    def _get_data_types(self, file_path: str) -> str:
        """Get data type information for all columns."""
        try:
            # Get metadata
            result = csv_processor.process_csv_file(file_path)
            if not result["success"]:
                return f"Failed to get data types: {result.get('error')}"
            
            metadata = result["metadata"]
            column_analysis = metadata["column_analysis"]
            
            type_info = ["Data Types:"]
            for col_name, col_info in column_analysis.items():
                data_type = col_info.get("data_type_category", "Unknown")
                dtype = col_info.get("dtype", "Unknown")
                type_info.append(f"  {col_name}: {data_type} ({dtype})")
            
            return "\n".join(type_info)
            
        except Exception as e:
            return f"Error getting data types: {str(e)}"
    
    def _get_summary(self, file_path: str) -> str:
        """Get a summary of the dataset structure."""
        try:
            # Get metadata
            result = csv_processor.process_csv_file(file_path)
            if not result["success"]:
                return f"Failed to get summary: {result.get('error')}"
            
            metadata = result["metadata"]
            file_info = metadata["file_info"]
            column_analysis = metadata["column_analysis"]
            operational_capabilities = metadata["operational_capabilities"]
            
            # Count data types
            type_counts = {}
            for col_info in column_analysis.values():
                data_type = col_info.get("data_type_category", "Unknown")
                type_counts[data_type] = type_counts.get(data_type, 0) + 1
            
            summary_parts = [
                f"Dataset Summary:",
                f"  Total Rows: {file_info['total_rows']:,}",
                f"  Total Columns: {file_info['total_columns']}",
                f"  File Size: {file_info['file_size_bytes'] / (1024*1024):.1f} MB",
                "",
                f"Data Type Distribution:",
            ]
            
            for data_type, count in type_counts.items():
                summary_parts.append(f"  {data_type}: {count} columns")
            
            summary_parts.extend([
                "",
                f"Operational Capabilities:",
                f"  Grouping Columns: {len(operational_capabilities.get('grouping_columns', []))}",
                f"  Aggregation Columns: {len(operational_capabilities.get('aggregation_columns', []))}",
                f"  Identifier Columns: {len(operational_capabilities.get('identifier_columns', []))}",
                f"  Join Key Candidates: {len(operational_capabilities.get('join_key_candidates', []))}"
            ])
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error getting summary: {str(e)}" 