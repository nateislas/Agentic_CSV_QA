"""
Database models for the CSV Analysis Platform.

This module defines SQLAlchemy models for file management, session tracking,
query processing, and job monitoring.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .core.database import Base


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


class File(Base):
    """
    File model for tracking uploaded CSV files.
    
    Tracks file metadata, processing status, and provides relationship
    to sessions and queries that use this file.
    """
    __tablename__ = "files"
    
    # Primary key
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # File information
    filename = Column(String, nullable=False, index=True)
    original_filename = Column(String, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    
    # Processing status
    status = Column(String, default="processing", nullable=False, index=True)
    # Status values: processing, completed, error
    
    # File metadata (JSON)
    file_metadata = Column(JSON, nullable=True)
    # Contains: column_info, data_types, statistics, sample_data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("Session", back_populates="file", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<File(id='{self.id}', filename='{self.filename}', status='{self.status}')>"
    
    @property
    def is_processed(self) -> bool:
        """Check if file processing is complete."""
        return self.status == "completed"
    
    @property
    def has_error(self) -> bool:
        """Check if file processing failed."""
        return self.status == "error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "status": self.status,
            "metadata": self.file_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class Session(Base):
    """
    Session model for tracking conversation context.
    
    Maintains conversation history and active tables for multi-turn
    conversations with the CSV analysis agent.
    """
    __tablename__ = "sessions"
    
    # Primary key
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Associated file
    file_id = Column(String, ForeignKey("files.id"), nullable=False, index=True)
    
    # Conversation data
    conversation_history = Column(JSON, default=list, nullable=False)
    # List of message objects: {"role": "user|assistant", "content": "message", "timestamp": "..."}
    
    active_tables = Column(JSON, default=dict, nullable=False)
    # Dictionary of active table references: {"table_name": "result_id"}
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    file = relationship("File", back_populates="sessions")
    queries = relationship("Query", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Session(id='{self.id}', file_id='{self.file_id}')>"
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.conversation_history.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation messages."""
        return self.conversation_history[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "conversation_history": self.conversation_history,
            "active_tables": self.active_tables,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Query(Base):
    """
    Query model for tracking individual queries and their results.
    
    Stores query text, execution results, and performance metrics
    for analysis and debugging.
    """
    __tablename__ = "queries"
    
    # Primary key
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Associated session
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    
    # Query information
    query_text = Column(Text, nullable=False)
    result = Column(JSON, nullable=True)
    # Contains: type, data, metadata, summary
    
    # Performance metrics
    execution_time = Column(Float, nullable=True)
    status = Column(String, default="processing", nullable=False, index=True)
    # Status values: processing, completed, error
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="queries")
    
    def __repr__(self) -> str:
        return f"<Query(id='{self.id}', status='{self.status}')>"
    
    @property
    def is_completed(self) -> bool:
        """Check if query processing is complete."""
        return self.status == "completed"
    
    @property
    def has_error(self) -> bool:
        """Check if query processing failed."""
        return self.status == "error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "result": self.result,
            "execution_time": self.execution_time,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Job(Base):
    """
    Job model for tracking background processing tasks.
    
    Monitors file uploads, CSV processing, and query execution
    with progress tracking and error handling.
    """
    __tablename__ = "jobs"
    
    # Primary key
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Job information
    job_type = Column(String, nullable=False, index=True)
    # Job types: file_upload, csv_processing, query_processing
    
    status = Column(String, default="queued", nullable=False, index=True)
    # Status values: queued, processing, completed, failed
    
    # Progress tracking
    progress = Column(Integer, default=0, nullable=False)
    # Progress percentage: 0-100
    
    # Results and errors
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    def __repr__(self) -> str:
        return f"<Job(id='{self.id}', type='{self.job_type}', status='{self.status}')>"
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == "failed"
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == "processing"
    
    def update_progress(self, progress: int) -> None:
        """Update job progress."""
        self.progress = max(0, min(100, progress))
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as completed."""
        self.status = "completed"
        self.progress = 100
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str) -> None:
        """Mark job as failed."""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# Model utility functions
def get_model_by_id(model_class: type, session, model_id: str):
    """
    Get a model instance by ID.
    
    Args:
        model_class: The model class to query
        session: Database session
        model_id: The ID to search for
        
    Returns:
        Model instance or None if not found
    """
    return session.query(model_class).filter(model_class.id == model_id).first()


def create_model_instance(model_class: type, session, **kwargs):
    """
    Create and save a new model instance.
    
    Args:
        model_class: The model class to instantiate
        session: Database session
        **kwargs: Model attributes
        
    Returns:
        Created model instance
    """
    instance = model_class(**kwargs)
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return instance
