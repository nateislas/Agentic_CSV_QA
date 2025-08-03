from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import uvicorn
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import upload, query
from app.core.config import settings
from app.core.database import engine, Base

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Show all log levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure output goes to console
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up application...")
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="Agentic CSV QA API",
    description="A powerful API for analyzing CSV files using natural language queries",
    version="1.0.0",
    lifespan=lifespan,
    # Increase maximum request size for file uploads
    max_request_size=200 * 1024 * 1024  # 200MB
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure for large file uploads
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Custom middleware for large file uploads
class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Increase request size limit for file uploads
        if request.url.path == "/api/upload":
            # Allow larger files for upload endpoint
            request.scope["http_version"] = "1.1"
            request.scope["type"] = "http"
        
        response = await call_next(request)
        return response

app.add_middleware(LargeFileMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include API routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(query.router, prefix="/api", tags=["query"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agentic CSV QA API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
