"""
Visualization Tool for CSV Analysis Agent.

This tool allows the agent to create basic charts and summaries for CSV data.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


class VisualizationInput(BaseModel):
    """Input schema for visualization tool."""
    file_path: str = Field(..., description="Path to the CSV file")
    chart_type: str = Field(..., description="Type of chart: 'summary_table', 'value_counts', 'numeric_summary', 'correlation_heatmap'")
    columns: Optional[List[str]] = Field(None, description="Columns to visualize")
    limit: Optional[int] = Field(10, description="Limit number of results to show")


class VisualizationTool(BaseTool):
    """Tool for creating visualizations and summaries of CSV data."""
    
    name: str = "visualization"
    description: str = """
    Create visualizations and summaries of CSV data including summary tables, value counts, numeric summaries, and correlation heatmaps.
    Use this tool to present data in a clear, visual format.
    
    Chart Types:
    - summary_table: Create a summary table of the dataset
    - value_counts: Show value counts for categorical columns
    - numeric_summary: Create a summary of numeric columns
    - correlation_heatmap: Show correlation matrix as a table
    """
    args_schema = VisualizationInput
    
    def _run(self, file_path: str, chart_type: str, columns: Optional[List[str]] = None, limit: int = 10) -> str:
        """Execute the visualization operation."""
        try:
            logger.info(f"Visualization: {chart_type} on {file_path}")
            
            if chart_type == "summary_table":
                return self._create_summary_table(file_path)
            elif chart_type == "value_counts":
                return self._create_value_counts(file_path, columns, limit)
            elif chart_type == "numeric_summary":
                return self._create_numeric_summary(file_path, columns)
            elif chart_type == "correlation_heatmap":
                return self._create_correlation_heatmap(file_path, columns)
            else:
                return f"Unknown chart type: {chart_type}. Available types: summary_table, value_counts, numeric_summary, correlation_heatmap"
                
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return f"Error during visualization: {str(e)}"
    
    def _create_summary_table(self, file_path: str) -> str:
        """Create a summary table of the dataset."""
        try:
            # Get metadata from CSV processor
            from app.services.csv_processor import csv_processor
            result = csv_processor.process_csv_file(file_path)
            
            if not result["success"]:
                return f"Failed to get dataset summary: {result.get('error')}"
            
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
                "Dataset Summary Table",
                "=" * 50,
                f"File: {file_path}",
                f"Rows: {file_info['total_rows']:,}",
                f"Columns: {file_info['total_columns']}",
                f"Size: {file_info['file_size_bytes'] / (1024*1024):.1f} MB",
                "",
                "Data Type Distribution:"
            ]
            
            for data_type, count in type_counts.items():
                summary_parts.append(f"  {data_type}: {count} columns")
            
            summary_parts.extend([
                "",
                "Operational Capabilities:",
                f"  Grouping: {len(operational_capabilities.get('grouping_columns', []))} columns",
                f"  Aggregation: {len(operational_capabilities.get('aggregation_columns', []))} columns",
                f"  Identifiers: {len(operational_capabilities.get('identifier_columns', []))} columns",
                f"  Join Keys: {len(operational_capabilities.get('join_key_candidates', []))} columns"
            ])
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error creating summary table: {str(e)}"
    
    def _create_value_counts(self, file_path: str, columns: Optional[List[str]] = None, limit: int = 10) -> str:
        """Create value counts for categorical columns."""
        try:
            df = pd.read_csv(file_path)
            
            if columns:
                # Analyze specific columns
                categorical_cols = [col for col in columns if col in df.columns and df[col].dtype == 'object']
            else:
                # Analyze all categorical columns
                categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
            
            if not categorical_cols:
                return "Error: No categorical columns found for value counts"
            
            value_counts_parts = ["Value Counts by Column:"]
            
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(limit)
                value_counts_parts.append(f"\n{col} (top {limit} values):")
                
                # Convert to list of tuples for iteration
                value_count_list = list(value_counts.items())
                for value, count in value_count_list:
                    percentage = (count / len(df)) * 100
                    value_counts_parts.append(f"  {value}: {count} ({percentage:.1f}%)")
            
            return "\n".join(value_counts_parts)
            
        except Exception as e:
            return f"Error creating value counts: {str(e)}"
    
    def _create_numeric_summary(self, file_path: str, columns: Optional[List[str]] = None) -> str:
        """Create a summary of numeric columns."""
        try:
            df = pd.read_csv(file_path)
            
            if columns:
                # Analyze specific columns
                numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                # Analyze all numeric columns
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_cols:
                return "Error: No numeric columns found for numeric summary"
            
            summary_parts = ["Numeric Columns Summary:"]
            
            for col in numeric_cols:
                col_stats = df[col].describe()
                summary_parts.append(f"\n{col}:")
                summary_parts.append(f"  Count: {col_stats['count']}")
                summary_parts.append(f"  Mean: {col_stats['mean']:.2f}")
                summary_parts.append(f"  Std: {col_stats['std']:.2f}")
                summary_parts.append(f"  Min: {col_stats['min']}")
                summary_parts.append(f"  Max: {col_stats['max']}")
                summary_parts.append(f"  Median: {col_stats['median']:.2f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error creating numeric summary: {str(e)}"
    
    def _create_correlation_heatmap(self, file_path: str, columns: Optional[List[str]] = None) -> str:
        """Create a correlation matrix table."""
        try:
            df = pd.read_csv(file_path)
            
            # Get numeric columns
            if columns:
                numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_cols) < 2:
                return "Error: Need at least 2 numeric columns for correlation analysis"
            
            # Calculate correlations
            corr_df = df[numeric_cols].corr()
            
            # Format as a nice table
            heatmap_parts = ["Correlation Matrix:"]
            heatmap_parts.append("=" * (len(numeric_cols) * 12))
            
            # Header
            header = "Column".ljust(12)
            for col in numeric_cols:
                header += f"{col[:10]:>10} "
            heatmap_parts.append(header)
            heatmap_parts.append("-" * (len(numeric_cols) * 12))
            
            # Rows
            for i, col1 in enumerate(numeric_cols):
                row = f"{col1[:10]:<12}"
                for j, col2 in enumerate(numeric_cols):
                    corr_val = corr_df.iloc[i, j]
                    row += f"{corr_val:>10.3f} "
                heatmap_parts.append(row)
            
            heatmap_parts.append("=" * (len(numeric_cols) * 12))
            
            return "\n".join(heatmap_parts)
            
        except Exception as e:
            return f"Error creating correlation heatmap: {str(e)}" 