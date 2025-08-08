from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import logging
import math
from typing import Optional, Dict, Any
import os
from datetime import datetime

from app.core.database import get_db
from app.models import File as FileModel, Job
from app.services.csv_processor import csv_processor
from app.core.config import settings
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a CSV file for analysis
    
    Args:
        file: CSV file to upload
        
    Returns:
        Job information with tracking ID
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV files are supported"
            )
        
        # Check file size (50MB limit)
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 50MB limit"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        
        # Save file to disk
        file_path = f"uploads/{file_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database records
        db = next(get_db())
        
        # Create file record
        file_record = FileModel(
            id=file_id,
            filename=file_path,
            original_filename=file.filename,
            file_size=len(content),
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(file_record)
        
        # Create job record
        job_record = Job(
            id=job_id,
            job_type="file_upload",
            status="queued",
            progress=0,
            created_at=datetime.utcnow()
        )
        db.add(job_record)
        
        db.commit()
        db.close()
        
        # Start background processing
        background_tasks.add_task(
            process_csv_file,
            file_id=file_id,
            job_id=job_id,
            file_path=file_path
        )
        
        logger.info(f"File upload started: {file_id} -> {file_path}")
        
        return {
            "job_id": job_id,
            "file_id": file_id,
            "status": "processing",
            "file_info": {
                "filename": file.filename,
                "size": len(content),
                "estimated_rows": "calculating...",
                "estimated_columns": "calculating..."
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload file"
        )

async def process_csv_file(file_id: str, job_id: str, file_path: str):
    """
    Background task to process uploaded CSV file
    
    Args:
        file_id: Database file ID
        job_id: Database job ID
        file_path: Path to uploaded file
    """
    logger.info(f"Starting background task for file: {file_id}, job: {job_id}")
    db = next(get_db())
    
    try:
        # Update job status
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "processing"
            job.progress = 10
            db.commit()
            logger.info(f"Updated job status to processing: {job_id}")
        
        # Process CSV file
        logger.info(f"Processing CSV file: {file_path}")
        try:
            result = csv_processor.process_csv_file(file_path)
            logger.info(f"CSV processing result: {result}")
            metadata = result["metadata"] if result["success"] else {}
            if not result["success"]:
                raise Exception(result.get("error", "CSV processing failed"))
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            raise Exception(f"Failed to process CSV file: {str(e)}")
        
        # Update file record with metadata
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if file_record:
            file_record.status = "completed"
            file_record.file_metadata = metadata
            file_record.processed_at = datetime.utcnow()
            logger.info(f"Updated file record to completed: {file_id}")
        
        # Update job status
        if job:
            job.status = "completed"
            job.progress = 100
            job.result = {"file_id": file_id, "metadata": metadata}
            job.completed_at = datetime.utcnow()
            logger.info(f"Updated job status to completed: {job_id}")
        
        db.commit()
        logger.info(f"File processing completed: {file_id}")
        
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        
        # Update error status
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if file_record:
            file_record.status = "error"
        
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
        
        db.commit()
        
    finally:
        db.close()
        logger.info(f"Background task finished for job: {job_id}")

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a background job
    
    Args:
        job_id: Job ID to check
        
    Returns:
        Job status and progress information
    """
    try:
        db = next(get_db())
        job = db.query(Job).filter(Job.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
        
        if job.result:
            response["result"] = job.result
            
        if job.error_message:
            response["error"] = job.error_message
            
        db.close()
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get job status"
        )

@router.get("/preview/{file_id}")
async def get_csv_preview(file_id: str, rows: int = 20):
    """
    Get a preview of the CSV data (first N rows)
    
    Args:
        file_id: File ID to get preview for
        rows: Number of rows to return (default 20, max 100)
        
    Returns:
        Preview data with metadata
    """
    try:
        # Limit rows to prevent abuse
        rows = min(rows, 100)
        
        db = next(get_db())
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_record.status != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_record.filename, nrows=rows)
            
                            # Clean data for JSON serialization
            def clean_value(val):
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float)):
                    # Handle infinite and very large values
                    if pd.isna(val) or math.isinf(val):
                        return None
                    if abs(val) > 1e15:  # Very large numbers
                        return str(val)
                return val
            
            # Convert to records with cleaned values
            preview_data = []
            for _, row in df.iterrows():
                cleaned_row = {}
                for col in df.columns:
                    cleaned_row[col] = clean_value(row[col])
                preview_data.append(cleaned_row)
            
            # Get basic metadata
            total_rows = len(pd.read_csv(file_record.filename))
            total_columns = len(df.columns)
            
            # Get data quality info
            quality_info = {}
            for col in df.columns:
                null_count = df[col].isnull().sum()
                quality_info[col] = {
                    "null_count": int(null_count),
                    "null_percentage": round((null_count / len(df)) * 100, 1),
                    "data_type": str(df[col].dtype)
                }
            
            response = {
                "file_id": file_id,
                "filename": file_record.original_filename,
                "preview_rows": len(preview_data),
                "total_rows": total_rows,
                "total_columns": total_columns,
                "columns": list(df.columns),
                "data": preview_data,
                "quality_info": quality_info,
                "file_metadata": file_record.file_metadata
            }
            
            db.close()
            return response
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read CSV file: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get CSV preview"
        )
