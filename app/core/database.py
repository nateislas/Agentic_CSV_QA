"""
Database connection and session management for the CSV Analysis Platform.

This module provides SQLAlchemy setup with sync operations, session management,
and database initialization utilities.
"""

from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import settings

# Create SQLAlchemy engine
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DEBUG  # Log SQL queries in debug mode
    )
    
    # Enable foreign key support for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL/MySQL configuration for production
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600,   # Recycle connections every hour
        echo=settings.DEBUG
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all database tables (use with caution)."""
    Base.metadata.drop_all(bind=engine)


def get_db_session() -> Session:
    """
    Get a database session for manual use.
    
    Returns:
        Session: SQLAlchemy database session
        
    Note:
        Remember to close the session when done:
        session = get_db_session()
        try:
            # Use session
            pass
        finally:
            session.close()
    """
    return SessionLocal()


# Database utility functions
def init_db() -> None:
    """Initialize database with tables and any required data."""
    create_tables()
    print("Database initialized successfully")


def check_db_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


def get_db_info() -> dict:
    """
    Get database information for debugging.
    
    Returns:
        dict: Database connection information
    """
    return {
        "database_url": settings.DATABASE_URL,
        "engine_name": engine.name,
        "pool_size": engine.pool.size(),
        "checked_in": engine.pool.checkedin(),
        "checked_out": engine.pool.checkedout(),
        "overflow": engine.pool.overflow(),
    }


# Database health check
def health_check() -> dict:
    """
    Perform database health check.
    
    Returns:
        dict: Health check results
    """
    try:
        connection_ok = check_db_connection()
        db_info = get_db_info()
        
        return {
            "status": "healthy" if connection_ok else "unhealthy",
            "database": db_info,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use real timestamp
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }
