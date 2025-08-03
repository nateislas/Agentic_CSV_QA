"""
Data Exploration Tool for CSV Analysis Agent.

This tool allows the agent to explore CSV data structure and get information
about columns, data types, and sample data without making domain assumptions.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

from app.services.csv_processor import csv_processor

logger = logging.getLogger(__name__)


class DataExplorationInput(BaseModel):
    """Input schema for data exploration tool."""
    operation: str = Field(default="summary", description="Type of exploration: 'column_info', 'sample_data', 'data_types', 'summary'")
    column_name: Optional[str] = Field(None, description="Specific column to explore (optional)")
    num_rows: Optional[int] = Field(default=5, description="Number of rows to show for sample_data operation (default: 5)")


class DataExplorationTool(BaseTool):
    """Tool for exploring CSV data structure and content."""
    
    name: str = "data_exploration"
    description: str = """
    Explore CSV data structure and get information about columns, data types, and sample data.
    Use this tool to understand the structure of the dataset before performing analysis.
    
    Arguments:
    - operation: Type of exploration (default: 'summary')
      * 'column_info': Get detailed information about all columns
      * 'sample_data': Get sample data from the dataset  
      * 'data_types': Get data type information for all columns
      * 'summary': Get a summary of the dataset structure
    - column_name: Specific column to explore (optional)
    - num_rows: Number of rows to show for sample_data operation (default: 5)
    
    Note: This tool automatically uses the current uploaded file. No file path needed.
    
    Example: Use this tool when you need to understand the structure of the CSV file.
    """
    args_schema = DataExplorationInput
    
    def _run(self, operation: str = "summary", column_name: Optional[str] = None, num_rows: Optional[int] = 5) -> str:
        """Execute the data exploration operation."""
        try:
            import os
            
            # Always get file path from agent, ignore any file_path passed by the agent
            try:
                from app.services.agent_service import get_csv_agent
                agent = get_csv_agent()
                file_path = agent.get_current_file_path()
                if file_path is None:
                    return "Error: No file path available. Please upload a file first."
            except Exception as e:
                logger.error(f"Failed to get file path from agent: {e}")
                return "Error: Could not determine file path."
            
            logger.info(f"Data exploration tool called with: operation={operation}, column_name={column_name}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            logger.info(f"Absolute file path: {os.path.abspath(file_path)}")
            logger.info(f"Data exploration: {operation} on {file_path}")
            
            if operation == "column_info":
                return self._get_column_info(file_path, column_name)
            elif operation == "sample_data":
                logger.info(f"Calling sample_data operation with num_rows={num_rows}")
                result = self._get_sample_data(file_path, column_name, num_rows)
                logger.info(f"Sample data result length: {len(result)}")
                return result
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
    
    def _get_sample_data(self, file_path: str, column_name: Optional[str] = None, num_rows: int = 5) -> str:
        """Get sample data from the dataset."""
        try:
            # Read CSV with specified number of rows
            df = pd.read_csv(file_path, nrows=num_rows)
            
            if column_name:
                if column_name not in df.columns:
                    return f"Column '{column_name}' not found in the dataset."
                
                sample_data = df[column_name].tolist()
                return f"Sample data for column '{column_name}' (first {num_rows} rows):\n{', '.join(map(str, sample_data))}"
            else:
                # Return sample of all columns
                html_table = self._dataframe_to_html(df)
            logger.info(f"Generated HTML table length: {len(html_table)}")
            return f"Sample data (first {num_rows} rows):\n{html_table}"
                
        except Exception as e:
            return f"Error getting sample data: {str(e)}"
    
    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """Convert pandas DataFrame to styled HTML table."""
        try:
            # Create HTML table with modern styling
            html = df.to_html(
                index=False,
                classes=['data-table', 'table', 'table-striped', 'table-hover'],
                table_id='data-table',
                escape=False,
                border=0
            )
            
            # Add custom CSS styling
            css = """
            <style>
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .data-table thead {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .data-table th {
                padding: 12px 16px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border: none;
            }
            
            .data-table td {
                padding: 12px 16px;
                border-bottom: 1px solid #f0f0f0;
                vertical-align: top;
            }
            
            .data-table tbody tr:hover {
                background-color: #f8f9fa;
                transition: background-color 0.2s ease;
            }
            
            .data-table tbody tr:last-child td {
                border-bottom: none;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .data-table {
                    font-size: 12px;
                }
                .data-table th,
                .data-table td {
                    padding: 8px 12px;
                }
            }
            </style>
            """
            
            return css + html
            
        except Exception as e:
            return f"Error creating HTML table: {str(e)}"
    
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
            import os
            logger.info(f"_get_summary called with file_path: {file_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            logger.info(f"Absolute file path: {os.path.abspath(file_path)}")
            
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