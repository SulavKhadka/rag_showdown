import os
import json
import platform
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import uvicorn
import datetime
import uuid
import sys
import psutil
from pathlib import Path

from pipelines.configurable_rag import ConfigurableRAGRetriever
from pipelines.config import PipelineConfig, get_config_by_preset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory
LOGS_DIR = Path("logs")
QUERY_LOGS_DIR = LOGS_DIR / "queries"
LOGS_DIR.mkdir(exist_ok=True)
QUERY_LOGS_DIR.mkdir(exist_ok=True)

# Configure file handler for all logs
file_handler = logging.FileHandler(LOGS_DIR / "rag_app.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Default database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abstracts.db')

# Create pipeline cache
pipeline_cache = {}

# Define API models
class PipelineConfigModel(BaseModel):
    preset: str = Field(default="vector_only", description="Pipeline preset configuration name")
    use_vector: Optional[bool] = Field(default=True, description="Use vector search")
    use_bm25: Optional[bool] = Field(default=False, description="Use BM25 keyword search")
    use_reranker: Optional[bool] = Field(default=False, description="Use cross-encoder reranker")
    use_llm_reranker: Optional[bool] = Field(default=False, description="Use LLM-based relevance filtering")
    use_query_decomposition: Optional[bool] = Field(default=False, description="Use LLM-based query decomposition")
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve")
    min_similarity_pct: Optional[float] = Field(default=50.0, description="Vector search filter threshold")
    db_path: Optional[str] = Field(default=DEFAULT_DB_PATH, description="Path to SQLite database")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query text to process")
    config: PipelineConfigModel = Field(default_factory=PipelineConfigModel, description="Pipeline configuration")

class DocumentModel(BaseModel):
    id: int
    title: str
    abstract: str
    content: str
    published: Optional[str]
    authors: List[str]
    source: str
    similarity: Optional[float] = None
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[DocumentModel]

class PresetInfo(BaseModel):
    value: str
    name: str
    description: str

class DBStatus(BaseModel):
    exists: bool
    path: str

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline Explorer API",
    description="API for configurable RAG pipeline exploration",
    version="1.0.0"
)

def log_query_data(query_id: str, data: Dict[str, Any]) -> None:
    """
    Log query data to a file for later analysis.
    
    Args:
        query_id: Unique identifier for the query
        data: Dictionary containing query, config, and results data
    """
    try:
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{query_id}.json"
        filepath = QUERY_LOGS_DIR / filename
        
        # Add timestamp to data
        data["timestamp"] = timestamp
        data["query_id"] = query_id
        
        # Add system information
        data["system_info"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True)
        }
        
        # Write data to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Query data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving query data: {str(e)}")

def get_pipeline(config_dict: Dict[str, Any]) -> ConfigurableRAGRetriever:
    """
    Get or create a pipeline instance based on configuration.
    
    Args:
        config_dict: Dictionary with pipeline configuration
        
    Returns:
        A ConfigurableRAGRetriever instance
    """
    # Create a cache key from the config
    cache_key = json.dumps(config_dict, sort_keys=True)
    
    # Check if we have a cached instance
    if cache_key in pipeline_cache:
        logger.info("Using cached pipeline instance")
        return pipeline_cache[cache_key]
    
    # Handle custom preset or use a predefined one
    preset = config_dict.get('preset', 'vector_only')
    
    if preset == 'custom':
        # For custom preset, start with default config and apply customizations
        config = PipelineConfig()
        logger.info("Using custom configuration")
    else:
        # Use a predefined preset as the base
        config = get_config_by_preset(preset)
        logger.info(f"Using preset configuration: {preset}")
    
    # Override with custom settings
    logger.debug(f"Applying custom settings to configuration")
    if config_dict.get('use_vector') is not None:
        config.use_vector = config_dict['use_vector']
    if config_dict.get('use_bm25') is not None:
        config.use_bm25 = config_dict['use_bm25']
    if config_dict.get('use_reranker') is not None:
        config.use_reranker = config_dict['use_reranker']
    if config_dict.get('use_llm_reranker') is not None:
        config.use_llm_reranker = config_dict['use_llm_reranker']
    if config_dict.get('use_query_decomposition') is not None:
        config.use_query_decomposition = config_dict['use_query_decomposition']
    if config_dict.get('top_k') is not None:
        config.top_k = int(config_dict['top_k'])
    if config_dict.get('min_similarity_pct') is not None:
        config.min_similarity_pct = float(config_dict['min_similarity_pct'])
    
    # Create a new pipeline instance
    logger.info(f"Creating new pipeline with config: {config}")
    db_path = config_dict.get('db_path', DEFAULT_DB_PATH)
    
    try:
        pipeline = ConfigurableRAGRetriever(
            db_path=db_path, 
            config=config,
            logger=logger
        )
        
        # Cache the instance
        pipeline_cache[cache_key] = pipeline
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def index():
    """Serve the main index.html page"""
    return FileResponse("static/index.html")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    """
    Process a RAG query with specified configuration.
    """
    query_id = str(uuid.uuid4())
    logger.info(f"[QueryID: {query_id}] Processing new query: '{query_request.query[:50]}...'")
    
    try:
        query = query_request.query
        if not query:
            logger.warning(f"[QueryID: {query_id}] Empty query received")
            raise HTTPException(status_code=400, detail="No query provided")
        
        config = query_request.config.dict()
        logger.info(f"[QueryID: {query_id}] Using configuration preset: {config.get('preset')}")
        logger.debug(f"[QueryID: {query_id}] Configuration details: {config}")
        
        # Get or create pipeline with the specified configuration
        pipeline = get_pipeline(config)
        
        start_time = datetime.datetime.now()
        
        # Process the query - pass the query_id to the pipeline
        logger.info(f"[QueryID: {query_id}] Executing RAG pipeline")
        result = pipeline.rag_query(query, query_id=query_id)
        
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"[QueryID: {query_id}] Query processed in {processing_time:.2f} seconds")
        
        # Log details about retrieved documents
        doc_count = len(result.get("retrieved_documents", []))
        logger.info(f"[QueryID: {query_id}] Retrieved {doc_count} documents")
        
        # Get pipeline configuration for tracking model information
        pipeline_config = pipeline.cfg
        
        # Save query data for analysis
        query_data = {
            "query": query,
            "config": config,
            "models": {
                "embedding_model": pipeline_config.embedding_model,
                "reranker_model": pipeline_config.reranker_model if pipeline_config.use_reranker else None,
                "llm_model": pipeline_config.llm_model,
                "reranker_model_type": pipeline_config.reranker_model_type if pipeline_config.use_reranker else None
            },
            "device": pipeline_config.device,
            "result": {
                "answer": result.get("answer", ""),
                "retrieved_documents_count": doc_count,
                "retrieved_documents": result.get("retrieved_documents", [])
            },
            "processing_time": processing_time
        }
        log_query_data(query_id, query_data)
        
        return result
    
    except Exception as e:
        logger.exception(f"[QueryID: {query_id}] Error processing query")
        # Log the error for data collection purposes
        error_data = {
            "query": query_request.query,
            "config": query_request.config.dict(),
            "error": str(e),
            "error_type": type(e).__name__
        }
        log_query_data(f"{query_id}_error", error_data)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/presets", response_model=List[PresetInfo])
async def get_presets():
    """Get available presets"""
    logger.debug("Getting available presets")
    presets = [
        {"value": "vector_only", "name": "Vector Only", "description": "Simple vector retrieval only."},
        {"value": "vector_plus_rerank", "name": "Vector + Reranker", "description": "Vector retrieval with cross-encoder reranking."},
        {"value": "vector_plus_bm25", "name": "Vector + BM25", "description": "Combined vector and BM25 retrieval."},
        {"value": "vector_bm25_rerank", "name": "Vector + BM25 + Reranker", "description": "Combined vector and BM25 with cross-encoder reranking."},
        {"value": "vector_bm25_rerank_llm", "name": "Vector + BM25 + Reranker + LLM Filter", "description": "Combined retrieval with reranking and LLM filtering."},
        {"value": "full_hybrid", "name": "Full Hybrid", "description": "Full hybrid pipeline with all features enabled."}
    ]
    return presets

@app.get("/api/db_status", response_model=DBStatus)
async def db_status(path: str = DEFAULT_DB_PATH):
    """Check if the database file exists and return status"""
    logger.debug(f"Checking database status at: {path}")
    exists = os.path.exists(path)
    logger.info(f"Database at {path} exists: {exists}")
    return {
        "exists": exists,
        "path": path
    }

def main():
    """Run the FastAPI application with Uvicorn"""
    # Check if database exists
    if not os.path.exists(DEFAULT_DB_PATH):
        logger.warning(f"Database file {DEFAULT_DB_PATH} not found. Please create the database first.")
    
    # Run the FastAPI app with Uvicorn
    logger.info("Starting RAG Pipeline Explorer server with FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()