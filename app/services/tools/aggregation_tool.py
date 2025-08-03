"""
Aggregation Tool for CSV Analysis Agent.

This tool allows the agent to perform aggregation operations like group by,
sum, average, count, and other statistical aggregations on CSV data.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import polars as pl

logger = logging.getLogger(__name__)


class AggregationInput(BaseModel):
    """Input schema for aggregation tool."""
    file_path: str = Field(..., description="Path to the CSV file")
    operation: str = Field(..., description="Type of aggregation: 'group_by', 'sum', 'average', 'count', 'min', 'max', 'median'")
    group_by_columns: Optional[List[str]] = Field(None, description="Columns to group by (for group_by operations)")
    aggregate_columns: Optional[List[str]] = Field(None, description="Columns to aggregate")
    filter_condition: Optional[str] = Field(None, description="Optional filter condition")


class AggregationTool(BaseTool):
    """Tool for performing aggregation operations on CSV data."""
    
    name = "aggregation"
    description = """
    Perform aggregation operations on CSV data such as group by, sum, average, count, min, max, median.
    Use this tool to analyze data patterns and create summaries.
    
    Operations:
    - group_by: Group data by specified columns and aggregate
    - sum: Calculate sum of numeric columns
    - average: Calculate average of numeric columns
    - count: Count records or unique values
    - min: Find minimum values
    - max: Find maximum values
    - median: Calculate median values
    """
    args_schema = AggregationInput
    
    def _run(
        self, 
        file_path: str, 
        operation: str, 
        group_by_columns: Optional[List[str]] = None,
        aggregate_columns: Optional[List[str]] = None,
        filter_condition: Optional[str] = None
    ) -> str:
        """Execute the aggregation operation."""
        try:
            logger.info(f"Aggregation: {operation} on {file_path}")
            
            if operation == "group_by":
                return self._group_by(file_path, group_by_columns, aggregate_columns, filter_condition)
            elif operation == "sum":
                return self._sum_operation(file_path, aggregate_columns, filter_condition)
            elif operation == "average":
                return self._average_operation(file_path, aggregate_columns, filter_condition)
            elif operation == "count":
                return self._count_operation(file_path, aggregate_columns, filter_condition)
            elif operation == "min":
                return self._min_operation(file_path, aggregate_columns, filter_condition)
            elif operation == "max":
                return self._max_operation(file_path, aggregate_columns, filter_condition)
            elif operation == "median":
                return self._median_operation(file_path, aggregate_columns, filter_condition)
            else:
                return f"Unknown operation: {operation}. Available operations: group_by, sum, average, count, min, max, median"
                
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return f"Error during aggregation: {str(e)}"
    
    def _group_by(
        self, 
        file_path: str, 
        group_by_columns: Optional[List[str]] = None,
        aggregate_columns: Optional[List[str]] = None,
        filter_condition: Optional[str] = None
    ) -> str:
        """Perform group by operation."""
        try:
            # Read CSV
            df = pl.read_csv(file_path)
            
            # Apply filter if specified
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if not group_by_columns:
                return "Error: group_by_columns must be specified for group_by operation"
            
            # Validate columns exist
            missing_cols = [col for col in group_by_columns if col not in df.columns]
            if missing_cols:
                return f"Error: Columns not found: {missing_cols}"
            
            # Perform group by
            if aggregate_columns:
                # Group by with specific aggregations
                agg_exprs = []
                for col in aggregate_columns:
                    if col in df.columns:
                        agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                        agg_exprs.append(pl.col(col).mean().alias(f"{col}_avg"))
                        agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
                
                result = df.group_by(group_by_columns).agg(agg_exprs)
            else:
                # Simple group by with count
                result = df.group_by(group_by_columns).agg(pl.count().alias("count"))
            
            # Convert to pandas for better display
            result_pd = result.to_pandas()
            
            return f"Group by results:\n{result_pd.to_string(index=False)}"
            
        except Exception as e:
            return f"Error in group_by operation: {str(e)}"
    
    def _sum_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Calculate sum of numeric columns."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if aggregate_columns:
                # Sum specific columns
                numeric_cols = [col for col in aggregate_columns if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for sum operation"
                
                sums = {}
                for col in numeric_cols:
                    sums[col] = df[col].sum()
                
                result_parts = ["Sum of numeric columns:"]
                for col, sum_val in sums.items():
                    result_parts.append(f"  {col}: {sum_val}")
                
                return "\n".join(result_parts)
            else:
                # Sum all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for sum operation"
                
                sums = {}
                for col in numeric_cols:
                    sums[col] = df[col].sum()
                
                result_parts = ["Sum of all numeric columns:"]
                for col, sum_val in sums.items():
                    result_parts.append(f"  {col}: {sum_val}")
                
                return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error in sum operation: {str(e)}"
    
    def _average_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Calculate average of numeric columns."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if aggregate_columns:
                # Average specific columns
                numeric_cols = [col for col in aggregate_columns if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for average operation"
                
                averages = {}
                for col in numeric_cols:
                    averages[col] = df[col].mean()
                
                result_parts = ["Average of numeric columns:"]
                for col, avg_val in averages.items():
                    result_parts.append(f"  {col}: {avg_val:.2f}")
                
                return "\n".join(result_parts)
            else:
                # Average all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for average operation"
                
                averages = {}
                for col in numeric_cols:
                    averages[col] = df[col].mean()
                
                result_parts = ["Average of all numeric columns:"]
                for col, avg_val in averages.items():
                    result_parts.append(f"  {col}: {avg_val:.2f}")
                
                return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error in average operation: {str(e)}"
    
    def _count_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Count records or unique values."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            total_count = len(df)
            
            if aggregate_columns:
                # Count unique values in specific columns
                result_parts = [f"Total records: {total_count}"]
                for col in aggregate_columns:
                    if col in df.columns:
                        unique_count = df[col].n_unique()
                        result_parts.append(f"  {col}: {unique_count} unique values")
                
                return "\n".join(result_parts)
            else:
                return f"Total records: {total_count}"
                
        except Exception as e:
            return f"Error in count operation: {str(e)}"
    
    def _min_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Find minimum values."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if aggregate_columns:
                # Min of specific columns
                numeric_cols = [col for col in aggregate_columns if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for min operation"
                
                mins = {}
                for col in numeric_cols:
                    mins[col] = df[col].min()
                
                result_parts = ["Minimum values:"]
                for col, min_val in mins.items():
                    result_parts.append(f"  {col}: {min_val}")
                
                return "\n".join(result_parts)
            else:
                # Min of all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for min operation"
                
                mins = {}
                for col in numeric_cols:
                    mins[col] = df[col].min()
                
                result_parts = ["Minimum values of all numeric columns:"]
                for col, min_val in mins.items():
                    result_parts.append(f"  {col}: {min_val}")
                
                return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error in min operation: {str(e)}"
    
    def _max_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Find maximum values."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if aggregate_columns:
                # Max of specific columns
                numeric_cols = [col for col in aggregate_columns if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for max operation"
                
                maxs = {}
                for col in numeric_cols:
                    maxs[col] = df[col].max()
                
                result_parts = ["Maximum values:"]
                for col, max_val in maxs.items():
                    result_parts.append(f"  {col}: {max_val}")
                
                return "\n".join(result_parts)
            else:
                # Max of all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for max operation"
                
                maxs = {}
                for col in numeric_cols:
                    maxs[col] = df[col].max()
                
                result_parts = ["Maximum values of all numeric columns:"]
                for col, max_val in maxs.items():
                    result_parts.append(f"  {col}: {max_val}")
                
                return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error in max operation: {str(e)}"
    
    def _median_operation(self, file_path: str, aggregate_columns: Optional[List[str]] = None, filter_condition: Optional[str] = None) -> str:
        """Calculate median values."""
        try:
            df = pl.read_csv(file_path)
            
            if filter_condition:
                df = self._apply_filter(df, filter_condition)
            
            if aggregate_columns:
                # Median of specific columns
                numeric_cols = [col for col in aggregate_columns if col in df.columns and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for median operation"
                
                medians = {}
                for col in numeric_cols:
                    medians[col] = df[col].median()
                
                result_parts = ["Median values:"]
                for col, median_val in medians.items():
                    result_parts.append(f"  {col}: {median_val}")
                
                return "\n".join(result_parts)
            else:
                # Median of all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                if not numeric_cols:
                    return "Error: No numeric columns found for median operation"
                
                medians = {}
                for col in numeric_cols:
                    medians[col] = df[col].median()
                
                result_parts = ["Median values of all numeric columns:"]
                for col, median_val in medians.items():
                    result_parts.append(f"  {col}: {median_val}")
                
                return "\n".join(result_parts)
                
        except Exception as e:
            return f"Error in median operation: {str(e)}"
    
    def _apply_filter(self, df: pl.DataFrame, filter_condition: str) -> pl.DataFrame:
        """Apply a filter condition to the dataframe."""
        try:
            # Simple filter implementation - in a real system, this would be more sophisticated
            # For now, we'll just return the original dataframe
            # TODO: Implement proper filter parsing
            logger.warning(f"Filter condition not implemented: {filter_condition}")
            return df
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return df 