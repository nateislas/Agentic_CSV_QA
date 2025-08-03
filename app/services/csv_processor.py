"""
Generic CSV processing service for the CSV Analysis Platform.

This module handles CSV file validation, structural analysis, and metadata extraction
without making any domain-specific assumptions about the data content.
"""

import os
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)


class GenericCSVProcessor:
    """
    Generic CSV file processor with structural analysis only.
    
    Analyzes CSV files based purely on data structure characteristics:
    - Data types and cardinality
    - Completeness and distributions
    - Potential operational capabilities
    - No domain-specific assumptions
    """
    
    def __init__(self):
        self.supported_formats = {'.csv'}
        self.max_file_size = settings.MAX_FILE_SIZE
    
    def validate_file(self, file_path: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate CSV file format and size.
        
        Args:
            file_path: Path to the CSV file
            file_size: Size of the file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if file_size > self.max_file_size:
                return False, f"File size {file_size} exceeds maximum allowed size {self.max_file_size}"
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported file format: {file_ext}"
            
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Try to read first few lines to validate CSV format
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line or ',' not in first_line:
                    return False, "File does not appear to be a valid CSV"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"File validation failed: {str(e)}"
    
    def extract_structural_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive structural metadata from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing structural metadata
        """
        try:
            # Use polars for fast structural analysis
            df = pl.read_csv(file_path)
            
            # Basic file info
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # Column structural analysis
            column_analysis = self._analyze_column_structure(df)
            
            # Data quality assessment
            quality_metrics = self._assess_data_quality(df)
            
            # Operational capabilities
            operational_capabilities = self._identify_operational_capabilities(df, column_analysis)
            
            # Structural relationships
            structural_relationships = self._identify_structural_relationships(df, column_analysis)
            
            metadata = {
                "file_info": {
                    "total_rows": total_rows,
                    "total_columns": total_columns,
                    "file_size_bytes": os.path.getsize(file_path),
                    "processing_timestamp": datetime.utcnow().isoformat()
                },
                "column_analysis": column_analysis,
                "quality_metrics": quality_metrics,
                "operational_capabilities": operational_capabilities,
                "structural_relationships": structural_relationships,
                "analysis_guidance": self._generate_analysis_guidance(column_analysis, operational_capabilities),
                "llm_context": self._prepare_llm_context(df, column_analysis, operational_capabilities)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Structural metadata extraction error: {e}")
            raise ValueError(f"Failed to extract structural metadata: {str(e)}")
    
    def _analyze_column_structure(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze column structure characteristics without domain assumptions.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Dictionary with column structural analysis
        """
        column_analysis = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Data type classification
            dtype = str(col_data.dtype)
            data_type_category = self._classify_data_type(dtype)
            
            # Cardinality analysis
            unique_count = col_data.n_unique()
            total_count = len(col_data)
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Completeness analysis
            null_count = col_data.null_count()
            completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0
            
            # Structural characteristics
            characteristics = self._analyze_column_characteristics(col_data, data_type_category)
            
            column_analysis[col] = {
                "dtype": dtype,
                "data_type_category": data_type_category,
                "cardinality": {
                    "unique_count": unique_count,
                    "total_count": total_count,
                    "cardinality_ratio": cardinality_ratio,
                    "cardinality_level": self._classify_cardinality(cardinality_ratio)
                },
                "completeness": {
                    "null_count": null_count,
                    "non_null_count": total_count - null_count,
                    "completeness_ratio": completeness_ratio,
                    "completeness_level": self._classify_completeness(completeness_ratio)
                },
                "characteristics": characteristics
            }
        
        return column_analysis
    
    def _classify_data_type(self, dtype: str) -> str:
        """
        Classify data type into broad categories.
        
        Args:
            dtype: Polars data type string
            
        Returns:
            Data type category
        """
        dtype_lower = dtype.lower()
        
        if any(num_type in dtype_lower for num_type in ['int', 'float']):
            return "numeric"
        elif 'datetime' in dtype_lower or 'date' in dtype_lower:
            return "datetime"
        elif 'bool' in dtype_lower:
            return "boolean"
        else:
            return "categorical"
    
    def _classify_cardinality(self, ratio: float) -> str:
        """
        Classify cardinality level.
        
        Args:
            ratio: Unique count / total count ratio
            
        Returns:
            Cardinality level
        """
        if ratio == 1.0:
            return "unique"
        elif ratio > 0.8:
            return "high"
        elif ratio > 0.2:
            return "medium"
        else:
            return "low"
    
    def _classify_completeness(self, ratio: float) -> str:
        """
        Classify completeness level.
        
        Args:
            ratio: Non-null count / total count ratio
            
        Returns:
            Completeness level
        """
        if ratio == 1.0:
            return "complete"
        elif ratio > 0.9:
            return "high"
        elif ratio > 0.7:
            return "medium"
        else:
            return "low"
    
    def _analyze_column_characteristics(self, col_data: pl.Series, data_type_category: str) -> Dict[str, Any]:
        """
        Analyze column characteristics based on data type.
        
        Args:
            col_data: Column data
            data_type_category: Data type category
            
        Returns:
            Dictionary with characteristics
        """
        characteristics = {}
        
        if data_type_category == "numeric":
            # Numeric characteristics
            non_null_data = col_data.drop_nulls()
            if len(non_null_data) > 0:
                characteristics.update({
                    "min": float(non_null_data.min()),
                    "max": float(non_null_data.max()),
                    "mean": float(non_null_data.mean()),
                    "std": float(non_null_data.std()) if len(non_null_data) > 1 else 0,
                    "median": float(non_null_data.median()),
                    "range": float(non_null_data.max() - non_null_data.min()),
                    "zero_count": int((non_null_data == 0).sum()),
                    "negative_count": int((non_null_data < 0).sum()),
                    "positive_count": int((non_null_data > 0).sum())
                })
        
        elif data_type_category == "categorical":
            # Categorical characteristics
            non_null_data = col_data.drop_nulls()
            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts()
                characteristics.update({
                    "most_common_values": value_counts.head(5).to_dict(),
                    "least_common_values": value_counts.tail(5).to_dict(),
                    "avg_length": float(non_null_data.str.lengths().mean()) if hasattr(non_null_data, 'str') else None,
                    "max_length": int(non_null_data.str.lengths().max()) if hasattr(non_null_data, 'str') else None,
                    "min_length": int(non_null_data.str.lengths().min()) if hasattr(non_null_data, 'str') else None
                })
        
        elif data_type_category == "datetime":
            # Datetime characteristics
            non_null_data = col_data.drop_nulls()
            if len(non_null_data) > 0:
                characteristics.update({
                    "min_date": str(non_null_data.min()),
                    "max_date": str(non_null_data.max()),
                    "date_range_days": int((non_null_data.max() - non_null_data.min()).days) if hasattr(non_null_data.max() - non_null_data.min(), 'days') else None
                })
        
        return characteristics
    
    def _assess_data_quality(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality from structural perspective.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            total_cells = len(df) * len(df.columns)
            null_counts = df.null_count()
            null_cells = sum(null_counts.to_dict().values())
            
            # Column-level quality
            column_quality = {}
            for col in df.columns:
                col_data = df[col]
                null_count = col_data.null_count()
                unique_count = col_data.n_unique()
                
                column_quality[col] = {
                    "null_percentage": (null_count / len(col_data)) * 100,
                    "unique_percentage": (unique_count / len(col_data)) * 100,
                    "duplicate_percentage": ((len(col_data) - unique_count) / len(col_data)) * 100
                }
            
            quality_metrics = {
                "overall": {
                    "total_cells": total_cells,
                    "null_cells": null_cells,
                    "null_percentage": (null_cells / total_cells) * 100 if total_cells > 0 else 0,
                    "duplicate_rows": len(df) - len(df.unique()),
                    "empty_columns": sum(1 for col in df.columns if df[col].null_count() == len(df))
                },
                "column_quality": column_quality,
                "data_type_distribution": {
                    "numeric": sum(1 for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]),
                    "categorical": sum(1 for col in df.columns if df[col].dtype == pl.Utf8),
                    "datetime": sum(1 for col in df.columns if df[col].dtype in [pl.Datetime, pl.Date]),
                    "boolean": sum(1 for col in df.columns if df[col].dtype == pl.Boolean)
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Data quality assessment error: {e}")
            return {}
    
    def _identify_operational_capabilities(self, df: pl.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
        """
        Identify what operations are structurally possible.
        
        Args:
            df: Polars DataFrame
            column_analysis: Column analysis results
            
        Returns:
            Dictionary with operational capabilities
        """
        capabilities = {
            "grouping_columns": [],
            "aggregation_columns": [],
            "filtering_columns": [],
            "identifier_columns": [],
            "measure_columns": [],
            "join_key_candidates": []
        }
        
        for col, analysis in column_analysis.items():
            cardinality_level = analysis["cardinality"]["cardinality_level"]
            data_type_category = analysis["data_type_category"]
            completeness_level = analysis["completeness"]["completeness_level"]
            
            # Grouping columns (low cardinality)
            if cardinality_level in ["low", "medium"] and completeness_level in ["high", "complete"]:
                capabilities["grouping_columns"].append(col)
            
            # Aggregation columns (numeric)
            if data_type_category == "numeric" and completeness_level in ["high", "complete"]:
                capabilities["aggregation_columns"].append(col)
                capabilities["measure_columns"].append(col)
            
            # Filtering columns (any type with good completeness)
            if completeness_level in ["high", "complete"]:
                capabilities["filtering_columns"].append(col)
            
            # Identifier columns (high uniqueness)
            if cardinality_level == "unique":
                capabilities["identifier_columns"].append(col)
            
            # Join key candidates (medium cardinality)
            if cardinality_level == "medium" and completeness_level in ["high", "complete"]:
                capabilities["join_key_candidates"].append(col)
        
        return capabilities
    
    def _identify_structural_relationships(self, df: pl.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
        """
        Identify structural relationships between columns.
        
        Args:
            df: Polars DataFrame
            column_analysis: Column analysis results
            
        Returns:
            Dictionary with structural relationships
        """
        relationships = {
            "potential_primary_keys": [],
            "potential_foreign_keys": [],
            "correlated_columns": [],
            "complementary_columns": []
        }
        
        # Find potential primary keys (unique columns)
        for col, analysis in column_analysis.items():
            if analysis["cardinality"]["cardinality_level"] == "unique":
                relationships["potential_primary_keys"].append(col)
        
        # Find potential foreign keys (medium cardinality, categorical)
        for col, analysis in column_analysis.items():
            if (analysis["cardinality"]["cardinality_level"] == "medium" and 
                analysis["data_type_category"] == "categorical"):
                relationships["potential_foreign_keys"].append(col)
        
        # Note: Full correlation analysis would be computationally expensive
        # and is left for the LLM to suggest when needed
        
        return relationships
    
    def _prepare_llm_context(self, df: pl.DataFrame, column_analysis: Dict, operational_capabilities: Dict) -> Dict[str, Any]:
        """
        Prepare structural context for LLM analysis.
        
        Args:
            df: Polars DataFrame
            column_analysis: Column analysis results
            operational_capabilities: Operational capabilities
            
        Returns:
            Dictionary optimized for LLM context
        """
        try:
            # Create a compact structural representation for LLM
            llm_context = {
                "schema": {
                    "columns": list(df.columns),
                    "total_rows": len(df),
                    "total_columns": len(df.columns)
                },
                "column_structure": {
                    col: {
                        "data_type": info["data_type_category"],
                        "cardinality": info["cardinality"]["cardinality_level"],
                        "completeness": info["completeness"]["completeness_level"],
                        "characteristics": info["characteristics"]
                    }
                    for col, info in column_analysis.items()
                },
                "operational_capabilities": operational_capabilities,
                "analysis_guidance": self._generate_analysis_guidance(column_analysis, operational_capabilities)
            }
            
            return llm_context
            
        except Exception as e:
            logger.error(f"LLM context preparation error: {e}")
            return {}
    
    def _generate_analysis_guidance(self, column_analysis: Dict, operational_capabilities: Dict) -> List[str]:
        """
        Generate structural analysis guidance.
        
        Args:
            column_analysis: Column analysis results
            operational_capabilities: Operational capabilities
            
        Returns:
            List of analysis guidance points
        """
        guidance = []
        
        try:
            # Grouping capabilities
            if operational_capabilities["grouping_columns"]:
                guidance.append(f"Grouping possible on: {operational_capabilities['grouping_columns']}")
            
            # Aggregation capabilities
            if operational_capabilities["aggregation_columns"]:
                guidance.append(f"Aggregation possible on: {operational_capabilities['aggregation_columns']}")
            
            # Filtering capabilities
            if operational_capabilities["filtering_columns"]:
                guidance.append(f"Filtering possible on: {operational_capabilities['filtering_columns']}")
            
            # Data quality insights
            low_completeness_cols = [
                col for col, analysis in column_analysis.items()
                if analysis["completeness"]["completeness_level"] == "low"
            ]
            if low_completeness_cols:
                guidance.append(f"Low completeness columns: {low_completeness_cols}")
            
            # Structural insights
            if operational_capabilities["identifier_columns"]:
                guidance.append(f"Potential identifier columns: {operational_capabilities['identifier_columns']}")
            
            if operational_capabilities["join_key_candidates"]:
                guidance.append(f"Potential join key candidates: {operational_capabilities['join_key_candidates']}")
                
        except Exception as e:
            logger.error(f"Analysis guidance generation error: {e}")
        
        return guidance
    
    def process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete CSV processing pipeline with structural analysis only.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Validate file
            file_size = os.path.getsize(file_path)
            is_valid, error_msg = self.validate_file(file_path, file_size)
            
            if not is_valid:
                raise ValueError(error_msg)
            
            # Extract structural metadata
            metadata = self.extract_structural_metadata(file_path)
            
            # Create processing result
            result = {
                "success": True,
                "file_path": file_path,
                "metadata": metadata,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"CSV structural analysis completed for {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"CSV processing failed for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "processing_timestamp": datetime.utcnow().isoformat()
            }


# Create global processor instance
csv_processor = GenericCSVProcessor()
