"""
Statistics Tool for CSV Analysis Agent.

This tool allows the agent to perform statistical analysis on CSV data
including descriptive statistics, correlations, and data quality metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticsInput(BaseModel):
    """Input schema for statistics tool."""
    file_path: str = Field(..., description="Path to the CSV file")
    operation: str = Field(default="descriptive", description="Type of statistics: 'descriptive', 'correlation', 'data_quality', 'distribution'")
    columns: Optional[List[str]] = Field(None, description="Specific columns to analyze")


class StatisticsTool(BaseTool):
    """Tool for performing statistical analysis on CSV data."""
    
    name: str = "statistics"
    description: str = """
    Analyze data patterns and relationships using statistical methods.
    Use this tool to understand trends, correlations, and data quality.
    
    Operations:
    - descriptive: Calculate averages, ranges, and typical values
    - correlation: Find relationships between different fields
    - data_quality: Check for missing data and inconsistencies
    - distribution: Analyze how values are spread across categories
    """
    args_schema = StatisticsInput
    
    def _run(self, file_path: str, operation: str = "descriptive", columns: Optional[List[str]] = None) -> str:
        """Execute the statistics operation."""
        try:
            logger.info(f"Statistics: {operation} on {file_path}")
            
            if operation == "descriptive":
                return self._descriptive_statistics(file_path, columns)
            elif operation == "correlation":
                return self._correlation_analysis(file_path, columns)
            elif operation == "data_quality":
                return self._data_quality_analysis(file_path)
            elif operation == "distribution":
                return self._distribution_analysis(file_path, columns)
            else:
                return f"Unknown operation: {operation}. Available operations: descriptive, correlation, data_quality, distribution"
                
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return f"Error during statistics: {str(e)}"
    
    def _descriptive_statistics(self, file_path: str, columns: Optional[List[str]] = None) -> str:
        """Calculate descriptive statistics."""
        try:
            df = pd.read_csv(file_path)
            
            if columns:
                # Analyze specific columns
                numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                if not numeric_cols:
                    return "Error: No numeric columns found for descriptive statistics"
                
                stats_parts = ["Descriptive Statistics:"]
                for col in numeric_cols:
                    col_stats = df[col].describe()
                    stats_parts.append(f"\n{col}:")
                    stats_parts.append(f"  Count: {col_stats['count']}")
                    stats_parts.append(f"  Mean: {col_stats['mean']:.2f}")
                    stats_parts.append(f"  Std: {col_stats['std']:.2f}")
                    stats_parts.append(f"  Min: {col_stats['min']}")
                    stats_parts.append(f"  Max: {col_stats['max']}")
                    stats_parts.append(f"  Median: {col_stats['median']:.2f}")
                
                return "\n".join(stats_parts)
            else:
                # Analyze all numeric columns
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if not numeric_cols:
                    return "Error: No numeric columns found for descriptive statistics"
                
                stats_parts = ["Descriptive Statistics for all numeric columns:"]
                for col in numeric_cols:
                    col_stats = df[col].describe()
                    stats_parts.append(f"\n{col}:")
                    stats_parts.append(f"  Count: {col_stats['count']}")
                    stats_parts.append(f"  Mean: {col_stats['mean']:.2f}")
                    stats_parts.append(f"  Std: {col_stats['std']:.2f}")
                    stats_parts.append(f"  Min: {col_stats['min']}")
                    stats_parts.append(f"  Max: {col_stats['max']}")
                    stats_parts.append(f"  Median: {col_stats['median']:.2f}")
                
                return "\n".join(stats_parts)
                
        except Exception as e:
            return f"Error in descriptive statistics: {str(e)}"
    
    def _correlation_analysis(self, file_path: str, columns: Optional[List[str]] = None) -> str:
        """Calculate correlations between numeric columns."""
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
            
            return f"Correlation Matrix:\n{corr_df.to_string()}"
            
        except Exception as e:
            return f"Error in correlation analysis: {str(e)}"
    
    def _data_quality_analysis(self, file_path: str) -> str:
        """Analyze data quality metrics."""
        try:
            # Get metadata from CSV processor
            from app.services.csv_processor import csv_processor
            result = csv_processor.process_csv_file(file_path)
            
            if not result["success"]:
                return f"Failed to get data quality metrics: {result.get('error')}"
            
            metadata = result["metadata"]
            quality_metrics = metadata["quality_metrics"]
            
            if "overall" not in quality_metrics:
                return "Data quality metrics not available"
            
            overall = quality_metrics["overall"]
            
            quality_parts = [
                "Data Quality Analysis:",
                f"  Total cells: {overall.get('total_cells', 'N/A'):,}",
                f"  Null cells: {overall.get('null_cells', 'N/A'):,} ({overall.get('null_percentage', 0):.1f}%)",
                f"  Duplicate rows: {overall.get('duplicate_rows', 'N/A')}",
                f"  Empty columns: {overall.get('empty_columns', 'N/A')}"
            ]
            
            # Add column-level quality if available
            if "column_quality" in quality_metrics:
                quality_parts.append("\nColumn Quality:")
                for col, quality in quality_metrics["column_quality"].items():
                    quality_parts.append(f"  {col}: {quality.get('null_percentage', 0):.1f}% null, {quality.get('unique_percentage', 0):.1f}% unique")
            
            return "\n".join(quality_parts)
            
        except Exception as e:
            return f"Error in data quality analysis: {str(e)}"
    
    def _distribution_analysis(self, file_path: str, columns: Optional[List[str]] = None) -> str:
        """Analyze value distributions for categorical columns."""
        try:
            df = pl.read_csv(file_path)
            
            if columns:
                # Analyze specific columns
                categorical_cols = [col for col in columns if col in df.columns and df[col].dtype == pl.Utf8]
            else:
                # Analyze all categorical columns
                categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
            
            if not categorical_cols:
                return "Error: No categorical columns found for distribution analysis"
            
            distribution_parts = ["Value Distribution Analysis:"]
            
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                distribution_parts.append(f"\n{col} (top 10 values):")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    distribution_parts.append(f"  {value}: {count} ({percentage:.1f}%)")
            
            return "\n".join(distribution_parts)
            
        except Exception as e:
            return f"Error in distribution analysis: {str(e)}" 