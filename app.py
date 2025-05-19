import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import uvicorn

from pipelines.configurable_rag import ConfigurableRAGRetriever
from pipelines.config import PipelineConfig, get_config_by_preset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    similarity: float
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
    
    # Override with custom settings
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
    try:
        query = query_request.query
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        config = query_request.config.dict()
        
        # Get or create pipeline with the specified configuration
        pipeline = get_pipeline(config)
        
        # Process the query
        result = pipeline.rag_query(query)
        
        return result
    
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/presets", response_model=List[PresetInfo])
async def get_presets():
    """Get available presets"""
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
    exists = os.path.exists(path)
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