#!/usr/bin/env python3
"""
Script to reset the database and recreate tables with the new schema.
"""

from app.core.database import drop_tables, create_tables
from app.models import Base

def reset_database():
    """Reset the database by dropping and recreating all tables."""
    print("🗑️  Dropping all database tables...")
    drop_tables()
    
    print("🏗️  Creating new database tables...")
    create_tables()
    
    print("✅ Database reset completed successfully!")

if __name__ == "__main__":
    reset_database() 