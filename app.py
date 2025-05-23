import os
import json
import platform
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, Query, Depends, status
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import uvicorn
import datetime
import uuid
from datetime import timedelta

# Authentication imports
from auth import (
    init_auth_db, authenticate_user, create_user, create_access_token,
    get_current_active_user, get_user_from_request,
    User, UserCreate, UserLogin
)
import sys
import psutil
import sqlite3
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

def get_db_connection():
    """Get database connection with FTS5 support"""
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_fts5_table():
    """Create FTS5 virtual table if it doesn't exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if FTS5 table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='abstracts_fts'
        """)
        
        if not cursor.fetchone():
            # Create FTS5 virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE abstracts_fts USING fts5(
                    title, abstract, authors, content=abstracts
                )
            """)
            
            # Populate FTS5 table with existing data
            cursor.execute("""
                INSERT INTO abstracts_fts(rowid, title, abstract, authors)
                SELECT id, title, abstract, authors FROM abstracts
            """)
            
            conn.commit()
            logger.info("FTS5 table created and populated successfully")
        else:
            logger.info("FTS5 table already exists")
            
        conn.close()
    except Exception as e:
        logger.error(f"Error setting up FTS5 table: {str(e)}")

# Create pipeline cache
pipeline_cache = {}

# Define API models
class PipelineConfigModel(BaseModel):
    preset: str = Field(default="vector_only", description="Pipeline preset configuration name")
    use_vector: Optional[bool] = Field(default=True, description="Use vector search")
    use_bm25: Optional[bool] = Field(default=False, description="Use BM25 keyword search")
    use_colbert: Optional[bool] = Field(default=False, description="Use ColBERT retrieval with PLAID index")
    vector_retrieval_method: Optional[str] = Field(default="standard", description="Vector retrieval method: 'none', 'standard' (single-vector), or 'colbert' (multi-vector)")
    use_reranker: Optional[bool] = Field(default=False, description="Use cross-encoder reranker")
    use_llm_reranker: Optional[bool] = Field(default=False, description="Use LLM-based relevance filtering")
    use_query_decomposition: Optional[bool] = Field(default=False, description="Use LLM-based query decomposition")
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve", ge=1, le=20)
    min_similarity_pct: Optional[float] = Field(default=50.0, description="Vector search filter threshold", ge=0.0, le=100.0)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The query text to process")
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

class DatasetStats(BaseModel):
    total_documents: int
    date_range: Dict[str, Optional[str]]
    top_authors: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    avg_abstract_length: float

class DocumentListItem(BaseModel):
    id: int
    title: str
    authors: List[str]
    published: str
    source: str
    abstract_preview: str  # First 200 chars

class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int
    page: int
    limit: int
    total_pages: int

class DocumentDetail(BaseModel):
    id: int
    title: str
    abstract: str
    authors: List[str]
    published: str
    source: str
    link: str

class AuthorInfo(BaseModel):
    name: str
    count: int

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline Explorer API",
    description="API for configurable RAG pipeline exploration",
    version="1.0.0"
)

# Custom rate limiting key function
def get_rate_limit_key(request: Request):
    """Get rate limiting key - use username if authenticated, else IP"""
    username = get_user_from_request(request)
    if username:
        return f"user:{username}"
    return f"ip:{get_remote_address(request)}"

# Initialize Limiter with user-aware rate limiting
limiter = Limiter(key_func=get_rate_limit_key, default_limits=["30/minute"])  # Higher limit for authenticated users
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize authentication on startup
@app.on_event("startup")
async def startup_event():
    """Initialize authentication database on startup"""
    init_auth_db(DEFAULT_DB_PATH)

def log_query_data(query_id: str, data: Dict[str, Any]) -> None:
    """
    Log query data to a file for later analysis.
    
    Args:
        query_id: Unique identifier for the query
        data: Dictionary containing query, config, and results data
    """
    if os.environ.get("ENABLE_DETAILED_QUERY_LOGS") != "true":
        logger.info("Detailed query logging is disabled. Skipping detailed log file creation for query_id: %s", query_id)
        return

    try:
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{query_id}.json"
        filepath = QUERY_LOGS_DIR / filename
        
        # Add timestamp to data
        data["timestamp"] = timestamp
        data["query_id"] = query_id
        
        # Add system information
        # System info removed for security
        
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
    if config_dict.get('use_colbert') is not None:
        config.use_colbert = config_dict['use_colbert']
    if config_dict.get('vector_retrieval_method') is not None:
        config.vector_retrieval_method = config_dict['vector_retrieval_method']
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
    
    # For backward compatibility, ensure use_vector and use_colbert flags match the vector_retrieval_method
    if 'vector_retrieval_method' in config_dict:
        if config.vector_retrieval_method == "standard":
            config.use_vector = True
            config.use_colbert = False
        elif config.vector_retrieval_method == "colbert":
            config.use_vector = True
            config.use_colbert = True
        elif config.vector_retrieval_method == "none":
            config.use_vector = False
            config.use_colbert = False
    
    # Create a new pipeline instance
    logger.info(f"Creating new pipeline with config: {config}")
    
    try:
        pipeline = ConfigurableRAGRetriever(
            db_path=DEFAULT_DB_PATH, 
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
@limiter.limit("10/minute")  # Increased limit for authenticated users
async def process_query(request: Request, query_request: QueryRequest, current_user: User = Depends(get_current_active_user)):
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
        
        config = query_request.config.model_dump()
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
                "reranker_model_type": pipeline_config.reranker_model_type if pipeline_config.use_reranker else None,
                "colbert_model": pipeline_config.colbert_model_name if pipeline_config.use_colbert else None
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
            "config": query_request.config.model_dump(), # Use model_dump() for Pydantic v2
            "error": str(e),
            "error_type": type(e).__name__
        }
        log_query_data(f"{query_id}_error", error_data)
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your query.")

# Authentication endpoints
@app.post("/api/auth/register")
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        user = create_user(user_data, DEFAULT_DB_PATH)
        access_token = create_access_token(data={"sub": user.username})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"username": user.username, "email": user.email}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    """Login and get access token"""
    user = authenticate_user(user_data.username, user_data.password, DEFAULT_DB_PATH)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"username": user.username, "email": user.email}
    }

@app.get("/api/auth/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return {"username": current_user.username, "email": current_user.email}

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
        {"value": "colbert_only", "name": "ColBERT Only", "description": "ColBERT retrieval using PLAID index."},
        {"value": "colbert_plus_rerank", "name": "ColBERT + Reranker", "description": "ColBERT retrieval with cross-encoder reranking."},
        {"value": "full_hybrid", "name": "Full Hybrid", "description": "Full hybrid pipeline with all features enabled."}
    ]
    return presets

@app.get("/api/db_status", response_model=DBStatus)
async def db_status():
    """Check if the default database file exists and return status"""
    logger.debug(f"Checking database status")
    exists = os.path.exists(DEFAULT_DB_PATH)
    logger.info(f"Database exists: {exists}")
    return {
        "exists": exists
    }

@app.get("/api/dataset/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get basic statistics about the dataset"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total documents
        cursor.execute("SELECT COUNT(*) as total FROM abstracts")
        total_docs = cursor.fetchone()["total"]
        
        # Date range
        cursor.execute("""
            SELECT MIN(published) as min_date, MAX(published) as max_date 
            FROM abstracts WHERE published != ''
        """)
        date_range_result = cursor.fetchone()
        date_range = {
            "earliest": date_range_result["min_date"],
            "latest": date_range_result["max_date"]
        }
        
        # Top authors by publication count - fixed query
        try:
            cursor.execute("""
                SELECT value as author, COUNT(*) as count
                FROM abstracts, json_each(authors)
                WHERE json_valid(authors)
                GROUP BY value
                ORDER BY count DESC
                LIMIT 10
            """)
            top_authors_rows = cursor.fetchall()
            top_authors = [{"name": row["author"], "count": row["count"]} 
                          for row in top_authors_rows]
        except Exception as e:
            logger.error(f"Error getting top authors: {str(e)}")
            top_authors = []
        
        # Source distribution
        cursor.execute("""
            SELECT source_file, COUNT(*) as count
            FROM abstracts
            GROUP BY source_file
            ORDER BY count DESC
        """)
        sources = [{"source": os.path.basename(row["source_file"]), "count": row["count"]} 
                  for row in cursor.fetchall()]
        
        # Average abstract length
        cursor.execute("SELECT AVG(LENGTH(abstract)) as avg_length FROM abstracts")
        avg_length = cursor.fetchone()["avg_length"] or 0
        
        conn.close()
        
        return DatasetStats(
            total_documents=total_docs,
            date_range=date_range,
            top_authors=top_authors,
            sources=sources,
            avg_abstract_length=round(avg_length, 1)
        )
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving dataset statistics")

@app.get("/api/dataset/authors", response_model=List[AuthorInfo])
async def get_authors():
    """Get list of all authors for filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Return authors list for filtering - fixed query
        try:
            cursor.execute("""
                SELECT value as author, COUNT(*) as count
                FROM abstracts, json_each(authors)
                WHERE json_valid(authors)
                GROUP BY value
                ORDER BY count DESC, author
            """)
            
            authors = [AuthorInfo(name=row["author"], count=row["count"]) 
                      for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error querying authors: {str(e)}")
            authors = []
        
        conn.close()
        return authors
    except Exception as e:
        logger.error(f"Error getting authors: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving authors")

@app.get("/api/documents", response_model=DocumentListResponse)
async def get_documents(
    page: int = Query(1, ge=1, le=1000),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=500),
    author: Optional[str] = Query(None, max_length=200),
    year_start: Optional[int] = Query(None, ge=1900, le=2030),
    year_end: Optional[int] = Query(None, ge=1900, le=2030),
    sort: str = Query("date", pattern="^(date|title|relevance)$")
):
    """Get paginated list of documents with optional filtering and search"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if search:
            # Use FTS5 for search
            where_conditions.append("abstracts.id IN (SELECT rowid FROM abstracts_fts WHERE abstracts_fts MATCH ?)")
            params.append(search)
        
        if author:
            where_conditions.append("EXISTS (SELECT 1 FROM json_each(authors) WHERE value = ?)")
            params.append(author)
        
        if year_start:
            where_conditions.append("CAST(SUBSTR(published, 1, 4) AS INTEGER) >= ?")
            params.append(year_start)
        
        if year_end:
            where_conditions.append("CAST(SUBSTR(published, 1, 4) AS INTEGER) <= ?")
            params.append(year_end)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Build ORDER BY clause
        if sort == "date":
            order_clause = "ORDER BY published DESC"
        elif sort == "title":
            order_clause = "ORDER BY title ASC"
        elif sort == "relevance" and search:
            order_clause = "ORDER BY bm25(abstracts_fts) ASC"
        else:
            order_clause = "ORDER BY id DESC"
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM abstracts {where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()["total"]
        
        # Calculate pagination
        offset = (page - 1) * limit
        total_pages = (total + limit - 1) // limit
        
        # Get documents
        if search and sort == "relevance":
            # Use FTS5 with ranking for relevance sort
            query = f"""
                SELECT abstracts.id, title, authors, published, source_file,
                       SUBSTR(abstract, 1, 200) as abstract_preview
                FROM abstracts
                JOIN abstracts_fts ON abstracts.id = abstracts_fts.rowid
                {where_clause}
                {order_clause}
                LIMIT ? OFFSET ?
            """
        else:
            query = f"""
                SELECT id, title, authors, published, source_file,
                       SUBSTR(abstract, 1, 200) as abstract_preview
                FROM abstracts
                {where_clause}
                {order_clause}
                LIMIT ? OFFSET ?
            """
        
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            authors = json.loads(row["authors"]) if row["authors"] else []
            documents.append(DocumentListItem(
                id=row["id"],
                title=row["title"],
                authors=authors,
                published=row["published"],
                source=os.path.basename(row["source_file"]),
                abstract_preview=row["abstract_preview"] + "..." if len(row["abstract_preview"]) == 200 else row["abstract_preview"]
            ))
        
        conn.close()
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages
        )
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving documents")

@app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
async def get_document_detail(doc_id: int):
    """Get detailed information about a specific document"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, abstract, authors, published, source_file, link
            FROM abstracts
            WHERE id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        authors = json.loads(row["authors"]) if row["authors"] else []
        
        document = DocumentDetail(
            id=row["id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=authors,
            published=row["published"],
            source=os.path.basename(row["source_file"]),
            link=row["link"]
        )
        
        conn.close()
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document detail: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving document")

@app.get("/api/documents/{doc_id}/similar", response_model=List[DocumentListItem])
async def get_similar_documents(doc_id: int, limit: int = Query(5, ge=1, le=20)):
    """Get documents similar to the specified document using vector embeddings"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, check if the document exists and get its embedding
        cursor.execute("SELECT id, embedding FROM abstracts WHERE id = ?", (doc_id,))
        source_doc = cursor.fetchone()
        if not source_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not source_doc["embedding"]:
            # Fallback: return documents from same author or similar publication year
            cursor.execute("""
                SELECT id, title, authors, published, source_file,
                       SUBSTR(abstract, 1, 200) as abstract_preview
                FROM abstracts
                WHERE id != ? AND (
                    (authors IS NOT NULL AND authors != '' AND 
                     EXISTS (
                        SELECT 1 FROM abstracts a2 
                        WHERE a2.id = ? AND 
                        json_extract(abstracts.authors, '$[0]') = json_extract(a2.authors, '$[0]')
                     )) OR
                    (published IS NOT NULL AND published != '' AND 
                     ABS(CAST(SUBSTR(published, 1, 4) AS INTEGER) - 
                         (SELECT CAST(SUBSTR(published, 1, 4) AS INTEGER) FROM abstracts WHERE id = ?)) <= 2)
                )
                ORDER BY published DESC
                LIMIT ?
            """, (doc_id, doc_id, doc_id, limit))
        else:
            # Try vector similarity with fallback for runtime issues
            try:
                # Attempt sqlite-vec approach first
                cursor.execute("""
                    SELECT a.id, a.title, a.authors, a.published, a.source_file,
                           SUBSTR(a.abstract, 1, 200) as abstract_preview
                    FROM vss_abstracts v
                    JOIN abstracts a ON v.rowid = a.id
                    WHERE v.rowid != ? AND v.embedding MATCH (
                        SELECT embedding FROM vss_abstracts WHERE rowid = ?
                    )
                    LIMIT ?
                """, (doc_id, doc_id, limit))
            except Exception as vec_error:
                logger.warning(f"sqlite-vec not available, using fallback similarity: {vec_error}")
                # Fallback: simple text-based similarity using abstract length and author overlap
                cursor.execute("""
                    SELECT a1.id, a1.title, a1.authors, a1.published, a1.source_file,
                           SUBSTR(a1.abstract, 1, 200) as abstract_preview,
                           ABS(LENGTH(a1.abstract) - (SELECT LENGTH(abstract) FROM abstracts WHERE id = ?)) as len_diff
                    FROM abstracts a1
                    WHERE a1.id != ?
                    ORDER BY len_diff ASC, 
                             CASE WHEN a1.published IS NOT NULL AND a1.published != '' 
                                  THEN ABS(CAST(SUBSTR(a1.published, 1, 4) AS INTEGER) - 
                                          (SELECT CAST(SUBSTR(published, 1, 4) AS INTEGER) FROM abstracts WHERE id = ?))
                                  ELSE 9999 END ASC
                    LIMIT ?
                """, (doc_id, doc_id, doc_id, limit))
        
        rows = cursor.fetchall()
        
        similar_docs = []
        for row in rows:
            authors = json.loads(row["authors"]) if row["authors"] else []
            similar_docs.append(DocumentListItem(
                id=row["id"],
                title=row["title"],
                authors=authors,
                published=row["published"],
                source=os.path.basename(row["source_file"]) if row["source_file"] else "Unknown",
                abstract_preview=row["abstract_preview"] + "..." if len(row["abstract_preview"]) == 200 else row["abstract_preview"]
            ))
        
        conn.close()
        return similar_docs
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving similar documents")

def main():
    """Run the FastAPI application with Uvicorn"""
    # Check if database exists
    if not os.path.exists(DEFAULT_DB_PATH):
        logger.warning(f"Database file {DEFAULT_DB_PATH} not found. Please create the database first.")
    else:
        # Set up FTS5 table if database exists
        setup_fts5_table()
    
    # Run the FastAPI app with Uvicorn
    logger.info("Starting RAG Pipeline Explorer server with FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()