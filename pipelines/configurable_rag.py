import os
import sys
import json
import re
import logging
import hashlib
import sqlite3
import sqlite_vec
import requests
import threading
import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

# Get the absolute path to the project root directory and add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_processor.kb_retriever import KBRetrieverBase
from pipelines.config import PipelineConfig, get_config_by_preset

# Create logs directory
LOGS_DIR = Path("logs/queries")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Import conditional dependencies
try:
    import bm25s
    import rerankers
    from pylate import models, indexes, retrieve
except ImportError:
    pass  # Will be handled during actual usage

# Define query decomposition prompts
QUERY_DECOMPOSITION_PROMPT = """You are an AI system specialized in decomposing a user query into sub-queries for a vector search system. 
Your task is to deconstruct the user query into multiple search queries that will help retrieve relevant information.

Generate 3-5 search queries that explore different perspectives and aspects of the main query.
Format your output as a JSON array containing only the search queries.

Input: {input_text}
"""

BM25_QUERY_DECOMPOSITION_PROMPT = """You are an AI system specialized in decomposing a user query into sub-queries for a keyword-based search system (BM25).
Your task is to deconstruct the user query into multiple search queries that will help retrieve relevant information using specific keywords.

Generate 3-5 keyword-rich search queries that will retrieve documents containing key terms from the main query.
Format your output as a JSON array containing only the search queries.

Input: {input_text}
"""

CONTEXT_RELEVANCE_FILTERING_PROMPT = """You are an AI system specialized in evaluating the relevance of text passages to a given query.
Your task is to analyze both the user's query and a set of retrieved text passages, then identify which passages are truly relevant.

Return the indices (starting from 0) of the passages that are directly relevant to answering the query.
Format your output as a JSON array containing only the indices of relevant passages.

Query: {query}

Passages:
{passages}
"""


class ConfigurableRAGRetriever(KBRetrieverBase):
    """
    A unified, configurable RAG retriever that can reproduce all current RAG variants
    by toggling individual sub-modules on/off through configuration.
    
    This class implements a retrieval approach that can:
    1. Optionally decompose queries into multiple sub-queries 
    2. Retrieve documents using one of multiple methods:
       - Standard vector similarity with dense embeddings
       - BM25 keyword-based search
       - ColBERT late-interaction model using PLAID index
    3. Optionally rerank results using cross-encoder and/or LLM filtering
    4. Generate answers based on the retrieved context
    
    The specific behavior is controlled by the provided PipelineConfig.
    
    ColBERT retrieval uses the PyLate library to access a pre-built PLAID index,
    offering more contextual understanding through token-level interactions
    compared to traditional dense retrieval methods.
    """
    
    def __init__(
        self,
        db_path: str,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the configurable RAG retriever.
        
        Args:
            db_path: Path to the SQLite database with knowledge content
            config: Pipeline configuration that controls which components are enabled
            logger: Optional logger instance
        """
        # Initialize the base class with embedding model configuration
        super().__init__(db_path, config.embedding_model, logger=logger)
        
        # Store configuration
        self.cfg = config
        
        # Initialize locks for thread safety
        self._reranker_lock = threading.Lock()
        self._bm25_lock = threading.Lock()
        self._colbert_lock = threading.Lock()
        
        # Lazy-loaded components
        self._reranker = None
        self._bm25 = None
        self._colbert_model = None
        self._colbert_index = None
        self._colbert_retriever = None
        
        # Set up LLM API key
        self.llm_api_key = config.llm_api_key
        
        # Check important API components
        if (config.use_query_decomposition or config.use_llm_reranker) and not self.llm_api_key:
            self.logger.warning("LLM API key not provided, but LLM-based features are enabled. "
                               "Query decomposition and LLM reranking will not work.")
        
        # For backward compatibility, if vector_retrieval_method is not set, infer it from use_vector and use_colbert
        if not hasattr(config, 'vector_retrieval_method') or not config.vector_retrieval_method:
            if config.use_colbert:
                config.vector_retrieval_method = "colbert"
            elif config.use_vector:
                config.vector_retrieval_method = "standard"
            else:
                config.vector_retrieval_method = "none"
    
    def log_query_details(self, query_id: str, data: Dict[str, Any]) -> None:
        """
        Log detailed query information to a file for analysis.
        
        Args:
            query_id: Unique identifier for the query
            data: Dictionary with query details, pipeline steps, and results
        """
        try:
            # Create a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_details_{timestamp}_{query_id}.json"
            filepath = LOGS_DIR / filename
            
            # Add timestamp
            data["timestamp"] = timestamp
            data["query_id"] = query_id
            
            # Write data to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"[QueryID: {query_id}] Detailed query log saved to {filepath}")
        except Exception as e:
            self.logger.error(f"[QueryID: {query_id}] Error saving query details: {str(e)}")
    
    def get_db_connection(self):
        """
        Get a database connection with vector search extension loaded.
        
        Returns:
            A sqlite3 connection object
        """
        try:
            conn = sqlite3.connect(self.conn_string)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            self.logger.info("Make sure sqlite-vec is properly installed: pip install sqlite-vec")
            raise
    
    def _ensure_reranker(self) -> Any:
        """
        Lazily initialize the reranker when needed.
        
        Returns:
            The initialized reranker instance
        """
        # Only initialize if needed and not already initialized
        if self.cfg.use_reranker and self._reranker is None:
            with self._reranker_lock:
                # Check again inside lock to prevent race conditions
                if self._reranker is None:
                    try:
                        self.logger.info(f"Initializing reranker: {self.cfg.reranker_model}")
                        self._reranker = rerankers.Reranker(
                            self.cfg.reranker_model, 
                            model_type=self.cfg.reranker_model_type, 
                            device=self.cfg.device
                        )
                        self.logger.info("Reranker initialized successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize reranker: {str(e)}")
                        raise
        
        return self._reranker
    
    def _ensure_colbert(self) -> Tuple[Any, Any, Any]:
        """
        Lazily initialize the ColBERT model and PLAID index when needed.
        
        Returns:
            Tuple of (colbert_model, colbert_index, colbert_retriever)
        """
        # Only initialize if needed and not already initialized
        if self.cfg.use_colbert and self._colbert_model is None:
            with self._colbert_lock:
                # Check again inside lock to prevent race conditions
                if self._colbert_model is None:
                    try:
                        self.logger.info(f"Initializing ColBERT model: {self.cfg.colbert_model_name}")
                        
                        # Initialize ColBERT model
                        colbert_model = models.ColBERT(
                            model_name_or_path=self.cfg.colbert_model_name,
                            device=self.cfg.device
                        )
                        
                        # Initialize PLAID index
                        index_path = os.path.join(self.cfg.plaid_index_folder, self.cfg.plaid_index_name)
                        self.logger.info(f"Loading PLAID index from {index_path}")
                        
                        plaid_index = indexes.PLAID(
                            index_folder=self.cfg.plaid_index_folder,
                            index_name=self.cfg.plaid_index_name,
                            override=False  # Don't override existing index
                        )
                        
                        # Initialize retriever
                        colbert_retriever = retrieve.ColBERT(index=plaid_index)
                        
                        self._colbert_model = colbert_model
                        self._colbert_index = plaid_index
                        self._colbert_retriever = colbert_retriever
                        self.logger.info("ColBERT model and PLAID index initialized successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to initialize ColBERT: {str(e)}")
                        raise
        
        return self._colbert_model, self._colbert_index, self._colbert_retriever
    
    def _ensure_bm25(self) -> Tuple[Any, Dict[str, int], List[str]]:
        """
        Lazily initialize the BM25 retriever when needed.
        
        Returns:
            Tuple of (bm25_retriever, doc_texts_to_ids, doc_texts)
        """
        # Only initialize if needed and not already initialized
        if self.cfg.use_bm25 and self._bm25 is None:
            with self._bm25_lock:
                # Check again inside lock to prevent race conditions
                if self._bm25 is None:
                    try:
                        self.logger.info("Initializing BM25 retriever")
                        
                        # Fetch documents from knowledge base
                        self.logger.info("Fetching documents for BM25 index")
                        conn = self.get_db_connection()
                        cursor = conn.cursor()
                        
                        # Query abstracts from SQLite database
                        cursor.execute("SELECT id, title, abstract FROM abstracts")
                        doc_results = cursor.fetchall()
                        conn.close()
                        
                        # Store document text and create mapping from text hash to document ID
                        doc_texts_to_ids = {}
                        doc_texts = []
                        
                        for doc in doc_results:
                            doc_id, title, abstract = doc
                            content = abstract  # Use abstract as the primary text content
                            
                            # Store mapping from content hash to document ID
                            doc_texts_to_ids[hashlib.sha256(content.encode('utf-8')).hexdigest()] = doc_id
                            doc_texts.append(content)
                        
                        # Initialize BM25 retriever
                        self.logger.info(f"Creating BM25 index for {len(doc_texts)} documents")
                        bm25_retriever = bm25s.BM25(corpus=doc_texts)
                        bm25_retriever.index(bm25s.tokenize(doc_texts))
                        
                        self._bm25 = (bm25_retriever, doc_texts_to_ids, doc_texts)
                        self.logger.info("BM25 retriever initialized successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to initialize BM25 retriever: {str(e)}")
                        raise
        
        return self._bm25
    
    def _decompose_query(self, input_texts: List[str]) -> Dict[str, List[str]]:
        """
        Decompose input texts into multiple search queries using LLM.
        
        Args:
            input_texts: List of input texts to decompose
            
        Returns:
            Dictionary of decomposed queries for different retrieval methods
        """
        self.logger.info(f"Decomposing {len(input_texts)} input text(s) into queries")
        texts_str = '\n'.join(input_texts)
        
        decomposed_queries = {
            'vector_query_decomposition': [],
            'bm25_query_decomposition': []
        }
        
        # Skip if LLM API key not available
        if not self.llm_api_key:
            self.logger.warning("Skipping query decomposition due to missing LLM API key")
            return {
                'vector_query_decomposition': input_texts,
                'bm25_query_decomposition': input_texts
            }
        
        # Process both vector and BM25 query decomposition
        for decomposition_prompt in [QUERY_DECOMPOSITION_PROMPT, BM25_QUERY_DECOMPOSITION_PROMPT]:            
            decomposition_type = 'vector_query_decomposition' if decomposition_prompt == QUERY_DECOMPOSITION_PROMPT else 'bm25_query_decomposition'
            
            try:
                self.logger.info(f"Sending {decomposition_type} request to LLM")
                
                # Prepare the API request
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_api_key}"
                }
                
                payload = {
                    "model": self.cfg.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that decomposes queries for retrieval systems."},
                        {"role": "user", "content": decomposition_prompt.format(input_text=texts_str)}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                
                # Make the API request
                response = requests.post(
                    self.cfg.llm_api_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                response_text = result["choices"][0]["message"]["content"]
                
                # Extract JSON array from text
                queries_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if queries_match:
                    queries_json = queries_match.group(0)
                    extracted_queries = json.loads(queries_json)
                    decomposed_queries[decomposition_type] = extracted_queries
                    self.logger.info(f"Extracted {len(extracted_queries)} {decomposition_type} queries")
                else:
                    self.logger.warning(f"No JSON response found in LLM response for {decomposition_type}")
                    decomposed_queries[decomposition_type] = input_texts
                    
            except Exception as e:
                self.logger.error(f"Error during {decomposition_type}: {str(e)}")
                decomposed_queries[decomposition_type] = input_texts
        
        return decomposed_queries
    
    def _colbert_retrieve(self, queries: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve documents using ColBERT and the PLAID index.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        self.logger.info(f"Performing ColBERT retrieval for {len(queries)} queries")
        
        combined_results = {}
        
        try:
            # Ensure ColBERT is initialized
            colbert_model, colbert_index, colbert_retriever = self._ensure_colbert()
            
            # Generate ColBERT query embeddings
            self.logger.info(f"Encoding {len(queries)} queries using ColBERT model")
            queries_embeddings = colbert_model.encode(
                queries,
                batch_size=8,
                is_query=True,
                show_progress_bar=False
            )
            
            # Retrieve top-k documents for each query
            self.logger.info(f"Retrieving top-{self.cfg.top_k} documents for each query using PLAID index")
            retrieval_results = colbert_retriever.retrieve(
                queries_embeddings=queries_embeddings,
                k=self.cfg.top_k
            )
            
            # Process results
            for query_idx, results in enumerate(retrieval_results):
                query = queries[query_idx]
                self.logger.debug(f"ColBERT search for query {query_idx+1}/{len(queries)}: '{query[:30]}...'")
                
                # Map PLAID document IDs to database IDs and add to results
                for result in results:
                    doc_id = result["id"]
                    score = result["score"]
                    
                    # Convert PLAID doc ID to integer for database lookup
                    try:
                        # Assuming PLAID uses document IDs that can be converted to integers
                        db_doc_id = int(doc_id)
                        
                        # Normalize score to 0-1 range (ColBERT scores are typically >0)
                        # Higher is better, like cosine similarity
                        similarity = min(1.0, score / 20.0)  # Normalize to 0-1 range
                        
                        # Skip results that don't meet the similarity threshold
                        if similarity * 100 < self.cfg.min_similarity_pct:
                            continue
                            
                        # Add to results if not already present or with higher score
                        if db_doc_id not in combined_results or combined_results[db_doc_id]['similarity'] < similarity:
                            # We'll fill in actual content in the next step
                            combined_results[db_doc_id] = {
                                'source': 'colbert',
                                'similarity': similarity
                            }
                    except ValueError:
                        self.logger.warning(f"Could not convert ColBERT document ID '{doc_id}' to integer")
                
                self.logger.info(f"ColBERT search returned {len(results)} results for query {query_idx+1}/{len(queries)}")
            
            # Fetch document contents for all retrieved IDs
            if combined_results:
                self.logger.info(f"Fetching content for {len(combined_results)} documents")
                doc_ids = list(combined_results.keys())
                
                # Get database connection
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Create comma-separated string of IDs for query
                ids_str = ','.join([str(i) for i in doc_ids])
                
                # Fetch document content from database
                query = f"SELECT id, title, abstract, authors, published FROM abstracts WHERE id IN ({ids_str})"
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Add content to results
                for row in results:
                    doc_id, title, abstract, authors_json, published = row
                    
                    # Parse authors if available
                    try:
                        authors = json.loads(authors_json) if authors_json else []
                    except json.JSONDecodeError:
                        authors = []
                        
                    # Update the result with content
                    combined_results[doc_id].update({
                        'text': abstract,
                        'title': title,
                        'authors': authors,
                        'published': published
                    })
                
                conn.close()
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in ColBERT retrieval: {str(e)}")
            self.logger.debug(f"Full exception details:", exc_info=True)
            return {}
    
    def _vector_retrieve(self, queries: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve documents using vector similarity search or ColBERT based on configuration.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        # Determine which vector retrieval method to use
        vector_method = getattr(self.cfg, 'vector_retrieval_method', None)
        
        # If vector_retrieval_method is not set, use backward compatibility logic
        if vector_method is None:
            if self.cfg.use_colbert and self.cfg.use_vector:
                vector_method = "colbert"
            elif self.cfg.use_vector:
                vector_method = "standard"
            else:
                vector_method = "none"
        
        # Check if vector retrieval is disabled
        if vector_method == "none":
            self.logger.info("Vector retrieval is disabled in configuration")
            return {}
        
        # Use ColBERT for vector retrieval if specified
        if vector_method == "colbert":
            self.logger.info(f"Using ColBERT for vector retrieval with {len(queries)} queries")
            return self._colbert_retrieve(queries)
        
        # Regular vector retrieval using dense embeddings
        self.logger.info(f"Performing standard vector retrieval for {len(queries)} queries")
        
        combined_results = {}
        
        try:
            # Generate embeddings for all queries at once
            query_embeddings = [self.generate_query_embedding(q) for q in queries]
            
            # Process each query and its embedding
            for i, (query, query_embedding) in enumerate(zip(queries, query_embeddings)):
                self.logger.debug(f"Vector search for query {i+1}/{len(queries)}: '{query[:30]}...'")
                
                # Get vector search results
                results = self._vector_search_from_kb(query_embedding)
                combined_results.update(results)
                
                self.logger.info(f"Vector search returned {len(results)} results for query {i+1}/{len(queries)}")
                
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in vector retrieval: {str(e)}")
            return {}
    
    def _bm25_retrieve(self, queries: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve documents using BM25 keyword search.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        self.logger.info(f"Performing BM25 retrieval for {len(queries)} queries")
        
        # Skip if BM25 is not enabled
        if not self.cfg.use_bm25:
            self.logger.warning("BM25 retrieval called but not enabled in config")
            return {}
        
        # Ensure BM25 retriever is initialized
        bm25_retriever, doc_texts_to_ids, _ = self._ensure_bm25()
        
        combined_results = {}
        
        for i, query in enumerate(queries):
            try:
                self.logger.debug(f"BM25 search for query {i+1}/{len(queries)}: '{query[:30]}...'")
                
                # Get BM25 results
                bm25_texts, _ = bm25_retriever.retrieve(bm25s.tokenize(query), k=self.cfg.top_k)
                bm25_texts = bm25_texts.tolist()[0]
                
                # Map texts to document IDs using the hash table
                bm25_ids = []
                for text in bm25_texts:
                    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    if text_hash in doc_texts_to_ids:
                        bm25_ids.append(doc_texts_to_ids[text_hash])
                
                # Create results dictionary mapping IDs to content
                for doc_id, text in zip(bm25_ids, bm25_texts):
                    combined_results[doc_id] = {
                        'text': text, 
                        'source': 'bm25'
                    }
                
                self.logger.info(f"BM25 search returned {len(bm25_ids)} results for query {i+1}/{len(queries)}")
                
            except Exception as e:
                self.logger.error(f"Error in BM25 retrieval for query '{query}': {str(e)}")
        
        return combined_results
    
    def _classic_rerank(self, query: str, candidate_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank candidate texts using a cross-encoder reranker.
        
        Args:
            query: The original query
            candidate_texts: List of candidate texts to rerank
            
        Returns:
            List of ranked results with scores
        """
        self.logger.info(f"Reranking {len(candidate_texts)} candidate texts")
        
        # Skip if reranker is not enabled
        if not self.cfg.use_reranker:
            self.logger.warning("Classic reranking called but not enabled in config")
            return [{'text': text, 'score': 0.5} for text in candidate_texts]
        
        try:
            # Ensure reranker is initialized
            reranker = self._ensure_reranker()
            
            # Perform reranking
            reranked = reranker.rank(query, candidate_texts)
            self.logger.info(f"Successfully reranked {len(reranked.results)} results")
            
            # Convert to simplified format
            return [{'text': r.text, 'score': r.score} for r in reranked.results]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            # Return the original texts with neutral scores as fallback
            return [{'text': text, 'score': 0.5} for text in candidate_texts]
    
    def _llm_rerank(self, query: str, contexts: List[Dict[str, Any]]) -> List[int]:
        """
        Filter contexts by relevance to query using LLM.
        
        Args:
            query: The query string
            contexts: List of context documents
            
        Returns:
            List of indices of relevant contexts
        """
        self.logger.info(f"LLM relevance filtering for {len(contexts)} contexts")
        
        # Skip if LLM reranker is not enabled or no API key
        if not self.cfg.use_llm_reranker or not self.llm_api_key:
            self.logger.warning("LLM reranking skipped (not enabled or missing API key)")
            return list(range(len(contexts)))
        
        if not contexts:
            self.logger.warning("No contexts provided for LLM relevance filtering")
            return []
        
        try:
            # Extract text from context objects
            context_texts = []
            for c in contexts:
                context_texts.append(f"Title: {c['metadata']['title']}\n\nAbstract: {c['text']}\n\nPublished: {c['metadata']['publication_date']}\n\nAuthors: {c['metadata']['authors']}")
            
            # Format context for LLM prompt
            context_texts_formatted = '\n\n'.join([f"<document id={i}>\n{ctx}\n</document>" for i, ctx in enumerate(context_texts)])
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            # Prepare prompt for relevance filtering
            prompt = CONTEXT_RELEVANCE_FILTERING_PROMPT.format(
                query=query,
                passages=context_texts_formatted
            )
            
            payload = {
                "model": self.cfg.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that evaluates text relevance."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }
            
            # Make the API request
            response = requests.post(
                self.cfg.llm_api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Extract JSON array from text
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                indices_json = json_match.group(0)
                relevant_indices = json.loads(indices_json)
                self.logger.info(f"Successfully extracted {len(relevant_indices)} relevant context indices")
                
                # Validate indices
                validated_indices = [idx for idx in relevant_indices 
                                    if isinstance(idx, int) and 0 <= idx < len(contexts)]
                
                if not validated_indices:
                    self.logger.warning("No valid indices found, falling back to all contexts")
                    return list(range(len(contexts)))
                
                return validated_indices
            else:
                self.logger.warning("No JSON response found in LLM response, falling back to all contexts")
                return list(range(len(contexts)))
                
        except Exception as e:
            self.logger.error(f"Error during LLM relevance filtering: {str(e)}")
            return list(range(len(contexts)))  # Fall back to all contexts
    
    def _vector_search_from_kb(self, query_embedding: List[float]) -> Dict[int, Dict[str, Any]]:
        """
        Perform vector similarity search against the knowledge base.
        
        Args:
            query_embedding: Embedding vector for the query
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        self.logger.info(f"Performing vector search with top_k={self.cfg.top_k}")
        
        try:
            # Serialize embedding for SQLite-vec query
            query_blob = self.serialize_embedding(query_embedding)
            
            # Get database connection
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Perform vector search using SQLite-vec
            results = cursor.execute('''
            SELECT 
                a.id, a.title, a.abstract, a.authors, a.published, a.source_file, a.content,
                v.distance
            FROM 
                vss_abstracts v
            JOIN 
                abstracts a ON v.rowid = a.id
            WHERE 
                v.embedding MATCH ? AND k = ?
            ORDER BY 
                v.distance
            ''', (query_blob, self.cfg.top_k)).fetchall()
            
            # Process results
            returned_contexts = {}
            for row in results:
                doc_id, title, abstract, authors_json, published, source_file, content, distance = row
                similarity = 1 - distance
                
                # Skip results that don't meet the similarity threshold
                if similarity * 100 < self.cfg.min_similarity_pct:
                    continue
                    
                # Use abstract as the main text content
                text_content = abstract
                
                # Parse authors if available
                try:
                    authors = json.loads(authors_json) if authors_json else []
                except json.JSONDecodeError:
                    authors = []
                
                # Add to returned contexts
                returned_contexts[doc_id] = {
                    'text': text_content,
                    'title': title,
                    'authors': authors,
                    'published': published,
                    'source': 'vector',
                    'similarity': similarity
                }
            
            conn.close()
            self.logger.info(f"Vector search returned {len(returned_contexts)} results")
            return returned_contexts
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return {}
    
    def _combine_results(self, vector_results: Dict[int, Dict[str, Any]], 
                        bm25_results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Combine and deduplicate results from different retrieval methods.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Combined dictionary mapping document IDs to document data
        """
        self.logger.info(f"Combining {len(vector_results)} vector results and {len(bm25_results)} BM25 results")
        
        # Copy vector results as the base
        combined = vector_results.copy()
        
        # Add BM25 results, marking duplicates
        for key, value in bm25_results.items():
            if key not in combined:
                # New document
                combined[key] = value
            else:
                # Document found by both methods
                combined[key]['source'] = f"vector,{value['source']}"
                self.logger.debug(f"Document {key} found by both methods")
                
                # Merge any missing fields
                for field, field_value in value.items():
                    if field != 'source' and field not in combined[key]:
                        combined[key][field] = field_value
        
        self.logger.info(f"Combined into {len(combined)} unique results")
        return combined
    
    def _get_metadata_by_ids(self, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get metadata for document IDs from the database.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Dictionary mapping document IDs to metadata
        """
        if not doc_ids:
            return {}
            
        # Create comma-separated string of IDs for SQL query
        ids_str = ', '.join([str(i) for i in doc_ids])
        
        try:
            # Get database connection
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Fetch document metadata
            query = f"SELECT id, title, published, authors FROM abstracts WHERE id IN ({ids_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            # Map results to dictionary keyed by ID
            metadata_by_id = {}
            for row in results:
                doc_id, title, published, authors_json = row
                
                # Parse authors if available
                try:
                    authors = json.loads(authors_json) if authors_json else []
                except json.JSONDecodeError:
                    authors = []
                
                # Create metadata dictionary
                metadata_by_id[doc_id] = {
                    'publication_date': published,
                    'title': title,
                    'authors': authors
                }
            
            self.logger.info(f"Retrieved metadata for {len(metadata_by_id)}/{len(doc_ids)} documents")
            return metadata_by_id
            
        except Exception as e:
            self.logger.error(f"Error retrieving document metadata: {str(e)}")
            return {}
    
    def retrieve_documents(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the knowledge base matching the query.
        The retrieval pipeline is configured according to the PipelineConfig.
        
        Depending on configuration, this method can use:
        - Standard vector retrieval with dense embeddings
        - BM25 keyword search
        - ColBERT retrieval with PLAID index (when vector_retrieval_method="colbert")
        
        Args:
            query: Query text to match against documents
            limit: Maximum number of documents to return (overrides config.top_k)
            **kwargs: Additional retrieval parameters
                - override_config: Optional PipelineConfig to use for this request only
                - query_id: Optional query identifier for logging
                
        Returns:
            List of document dictionaries with retrieval scores and metadata
        """
        query_id = kwargs.get('query_id', 'unknown')
        self.logger.info(f"[QueryID: {query_id}] Retrieving documents for query: {query[:50]}...")
        
        # Use the specified limit or fall back to the default top_k
        actual_limit = limit if limit is not None else self.cfg.top_k
        
        # For storing detailed pipeline logs
        pipeline_log = {
            "config": {k: v for k, v in self.cfg.__dict__.items() if not k.startswith('_')},
            "query": query,
            "limit": actual_limit,
            "models": {
                "embedding_model": self.cfg.embedding_model,
                "reranker_model": self.cfg.reranker_model if self.cfg.use_reranker else None,
                "reranker_model_type": self.cfg.reranker_model_type if self.cfg.use_reranker else None,
                "colbert_model": self.cfg.colbert_model_name if hasattr(self.cfg, 'vector_retrieval_method') and self.cfg.vector_retrieval_method == "colbert" else None
            },
            "device": self.cfg.device,
            "steps": {}
        }
        
        # Check for an override configuration for this request only
        config = kwargs.get('override_config', self.cfg)
        
        # For backward compatibility, if vector_retrieval_method is not set, infer it from use_vector and use_colbert
        if not hasattr(config, 'vector_retrieval_method'):
            if config.use_colbert:
                config.vector_retrieval_method = "colbert"
            elif config.use_vector:
                config.vector_retrieval_method = "standard"
            else:
                config.vector_retrieval_method = "none"
        
        # Validate that at least one retrieval method is enabled
        if config.vector_retrieval_method == "none" and not config.use_bm25:
            self.logger.error(f"[QueryID: {query_id}] No retrieval methods enabled.")
            raise ValueError("At least one retrieval method (vector or BM25) must be enabled")
        
        try:
            # Step 1: Prepare queries (decompose if configured)
            self.logger.info(f"[QueryID: {query_id}] Step 1: Preparing queries (decomposition: {config.use_query_decomposition})")
            step_start = datetime.datetime.now()
            
            if config.use_query_decomposition:
                # Use LLM to decompose query into multiple queries for better retrieval
                input_queries = [query] if isinstance(query, str) else [query]
                queries = self._decompose_query(input_queries)
                
                pipeline_log["steps"]["query_decomposition"] = {
                    "enabled": True,
                    "decomposed_queries": queries,
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            else:
                # Use simple query without decomposition
                queries = {
                    'vector_query_decomposition': [query] if config.vector_retrieval_method != "none" else [],
                    'bm25_query_decomposition': [query] if config.use_bm25 else []
                }
                
                pipeline_log["steps"]["query_decomposition"] = {
                    "enabled": False,
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            
            # Step 2: Retrieve documents using enabled methods
            self.logger.info(f"[QueryID: {query_id}] Step 2: Retrieving documents using enabled methods")
            vector_results = {}
            bm25_results = {}
            
            # Vector retrieval (standard or ColBERT)
            if config.vector_retrieval_method != "none" and queries['vector_query_decomposition']:
                step_start = datetime.datetime.now()
                
                if config.vector_retrieval_method == "colbert":
                    self.logger.info(f"[QueryID: {query_id}] Performing ColBERT retrieval with {len(queries['vector_query_decomposition'])} queries")
                else:
                    self.logger.info(f"[QueryID: {query_id}] Performing standard vector retrieval with {len(queries['vector_query_decomposition'])} queries")
                
                vector_results = self._vector_retrieve(queries['vector_query_decomposition'])
                
                pipeline_log["steps"]["vector_retrieval"] = {
                    "enabled": True,
                    "method": config.vector_retrieval_method,
                    "colbert_model": self.cfg.colbert_model_name if config.vector_retrieval_method == "colbert" else None,
                    "query_count": len(queries['vector_query_decomposition']),
                    "result_count": len(vector_results),
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            
            # BM25 retrieval
            if config.use_bm25 and queries['bm25_query_decomposition']:
                step_start = datetime.datetime.now()
                self.logger.info(f"[QueryID: {query_id}] Performing BM25 retrieval with {len(queries['bm25_query_decomposition'])} queries")
                bm25_results = self._bm25_retrieve(queries['bm25_query_decomposition'])
                
                pipeline_log["steps"]["bm25_retrieval"] = {
                    "enabled": True,
                    "query_count": len(queries['bm25_query_decomposition']),
                    "result_count": len(bm25_results),
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            
            # Step 3: Combine results from different retrieval methods
            step_start = datetime.datetime.now()
            self.logger.info(f"[QueryID: {query_id}] Step 3: Combining results from different retrieval methods")
            combined_map = self._combine_results(vector_results, bm25_results)
            
            pipeline_log["steps"]["result_combination"] = {
                "vector_result_count": len(vector_results),
                "bm25_result_count": len(bm25_results),
                "combined_result_count": len(combined_map),
                "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
            }
            
            # Early return if no results
            if not combined_map:
                self.logger.warning(f"[QueryID: {query_id}] No results found by retrieval methods")
                return []
            
            # Get document metadata
            step_start = datetime.datetime.now()
            metadata_by_id = self._get_metadata_by_ids(list(combined_map.keys()))
            
            pipeline_log["steps"]["metadata_retrieval"] = {
                "doc_count": len(combined_map.keys()),
                "metadata_retrieved_count": len(metadata_by_id),
                "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
            }
            
            # Extract text for reranking
            doc_texts = [item['text'] for item in combined_map.values()]
            doc_ids = list(combined_map.keys())
            
            # Step 4: Rerank results if configured
            if config.use_reranker and doc_texts:
                step_start = datetime.datetime.now()
                self.logger.info(f"[QueryID: {query_id}] Step 4: Reranking {len(doc_texts)} results with cross-encoder")
                reranked = self._classic_rerank(query, doc_texts)
                
                # Create a list of ranked results with document IDs
                ranked_results = []
                for i, ranked_doc in enumerate(reranked):
                    if i < len(doc_ids):
                        doc_id = doc_ids[i]
                        doc_data = combined_map[doc_id]
                        
                        # Create ranked result - keep original similarity score, not reranker score
                        ranked_results.append({
                            'id': doc_id,
                            'text': ranked_doc['text'],
                            'score': doc_data.get('similarity', 0.5),  # Use original similarity score
                            'source': doc_data['source'],
                            'metadata': metadata_by_id.get(doc_id, {})
                        })
                
                pipeline_log["steps"]["reranking"] = {
                    "enabled": True,
                    "doc_count": len(doc_texts),
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            else:
                # Skip reranking, just format results
                step_start = datetime.datetime.now()
                self.logger.info(f"[QueryID: {query_id}] Step 4: Skipping reranking (not enabled)")
                ranked_results = []
                for doc_id, doc_data in combined_map.items():
                    result = {
                        'id': doc_id,
                        'text': doc_data['text'],
                        'source': doc_data['source'],
                        'metadata': metadata_by_id.get(doc_id, {})
                    }
                    
                    # Only include similarity score for vector search results, not for pure BM25
                    if doc_data['source'] != 'bm25':
                        result['score'] = doc_data.get('similarity', 0.5)
                        
                    ranked_results.append(result)
                
                pipeline_log["steps"]["reranking"] = {
                    "enabled": False,
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            
            # Step 5: Apply LLM relevance filtering if configured
            if config.use_llm_reranker and ranked_results:
                step_start = datetime.datetime.now()
                self.logger.info(f"[QueryID: {query_id}] Step 5: Applying LLM relevance filtering to {len(ranked_results)} results")
                relevant_indices = self._llm_rerank(query, ranked_results)
                filtered_results = [ranked_results[i] for i in relevant_indices]
                self.logger.info(f"[QueryID: {query_id}] LLM filtering kept {len(filtered_results)}/{len(ranked_results)} results")
                
                pipeline_log["steps"]["llm_filtering"] = {
                    "enabled": True,
                    "input_count": len(ranked_results),
                    "kept_count": len(filtered_results),
                    "kept_indices": relevant_indices,
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            else:
                step_start = datetime.datetime.now()
                self.logger.info(f"[QueryID: {query_id}] Step 5: Skipping LLM relevance filtering (not enabled)")
                filtered_results = ranked_results
                
                pipeline_log["steps"]["llm_filtering"] = {
                    "enabled": False,
                    "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
                }
            
            # Sort by score (higher is better)
            step_start = datetime.datetime.now()
            # Use a sorting function that handles missing scores
            def get_score(item):
                return item.get('score', 0) if 'score' in item else 0
            
            sorted_results = sorted(filtered_results, key=get_score, reverse=True)
            
            # Limit to requested number of results
            final_results = sorted_results[:actual_limit]
            
            pipeline_log["steps"]["final_processing"] = {
                "filtered_count": len(filtered_results),
                "final_result_count": len(final_results),
                "time_ms": (datetime.datetime.now() - step_start).total_seconds() * 1000
            }
            
            # Format results according to expected interface
            formatted_results = []
            for res in final_results:
                metadata = res['metadata']
                formatted_doc = {
                    'id': res['id'],
                    'title': metadata.get('title', 'Unknown'),
                    'abstract': res['text'],
                    'content': res['text'],
                    'published': metadata.get('publication_date'),
                    'authors': metadata.get('authors', []),
                    'source': res['source'],
                    'metadata': {
                        'source': res['source'],
                        'publication_date': metadata.get('publication_date'),
                        'title': metadata.get('title', 'Unknown')
                    }
                }
                
                # Only include similarity for non-BM25 results
                if 'score' in res and res['source'] != 'bm25':
                    formatted_doc['similarity'] = res['score']
                formatted_results.append(formatted_doc)
            
            self.logger.info(f"[QueryID: {query_id}] Retrieved {len(formatted_results)} documents")
            
            # Log detailed pipeline information
            if query_id != 'unknown':
                pipeline_log["final_results_count"] = len(formatted_results)
                self.log_query_details(f"{query_id}_retrieval", pipeline_log)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"[QueryID: {query_id}] Error retrieving documents: {str(e)}")
            self.logger.debug(f"[QueryID: {query_id}] Full exception details:", exc_info=True)
            return []
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]], query_id: str = 'unknown') -> str:
        """
        Generate an answer using an LLM with the retrieved context.
        
        Args:
            query: The user's query
            context_docs: The retrieved context documents
            query_id: Optional query identifier for logging
            
        Returns:
            The generated answer
        """
        self.logger.info(f"[QueryID: {query_id}] Generating answer from {len(context_docs)} retrieved contexts")
        
        generation_log = {
            "query": query,
            "context_doc_count": len(context_docs),
            "context_doc_ids": [doc.get("id") for doc in context_docs],
            "model": {
                "llm_model": self.cfg.llm_model,
                "llm_api_url": self.cfg.llm_api_url
            },
            "temperature": 0.3,
            "max_tokens": 1000,
            "device": self.cfg.device
        }
        
        if not self.llm_api_key:
            self.logger.error(f"[QueryID: {query_id}] No LLM API key provided")
            return "Error: LLM API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        try:
            # Format context from documents
            context_text = ""
            for i, doc in enumerate(context_docs):
                title = doc.get("title", "Untitled Document")
                abstract = doc.get("abstract", doc.get("content", ""))
                authors = doc.get("authors", [])
                if isinstance(authors, list):
                    authors_text = ", ".join(authors)
                else:
                    authors_text = str(authors)
                
                source = doc.get("source", "")
                
                # Different formatting based on source
                if source == "bm25":
                    context_text += f"Document {i+1} [Source: BM25]\n"
                else:
                    similarity = doc.get("similarity", 0.0)
                    context_text += f"Document {i+1} [Similarity: {similarity:.2f}]\n"
                    
                context_text += f"Title: {title}\n"
                context_text += f"Authors: {authors_text}\n"
                context_text += f"Abstract: {abstract}\n\n"
            
            # Save formatted context
            generation_log["formatted_context"] = context_text
            
            # Create prompt for the LLM
            prompt = f"""You are a research assistant helping to answer scientific questions.
            
            Use the provided scientific papers to answer the question below. 
            If you don't know the answer or if the context doesn't 
            contain relevant information to answer the question, say so.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""
            generation_log["prompt"] = prompt
            
            start_time = datetime.datetime.now()
            self.logger.info(f"[QueryID: {query_id}] Sending request to LLM API")
            
            # Call the LLM API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            payload = {
                "model": self.cfg.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.cfg.llm_api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            end_time = datetime.datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            generation_log["answer"] = answer
            generation_log["generation_time_seconds"] = generation_time
            generation_log["timestamp"] = datetime.datetime.now().isoformat()
            
            self.logger.info(f"[QueryID: {query_id}] Successfully generated answer in {generation_time:.2f} seconds")
            
            # Log generation details
            if query_id != 'unknown':
                self.log_query_details(f"{query_id}_generation", generation_log)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"[QueryID: {query_id}] Error generating answer: {str(e)}")
            self.logger.debug(f"[QueryID: {query_id}] Full exception details:", exc_info=True)
            
            # Log error details
            generation_log["error"] = str(e)
            generation_log["error_type"] = type(e).__name__
            
            if query_id != 'unknown':
                self.log_query_details(f"{query_id}_generation_error", generation_log)
            
            return f"Error generating answer: {str(e)}"
    
    def rag_query(self, query: str, query_id: str = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline by retrieving documents and generating an answer.
        
        The retrieval step can use standard vector search, BM25, ColBERT, or a combination,
        based on the configuration. ColBERT retrieval (when enabled) uses token-level
        interaction via the PyLate library and a PLAID index.
        
        Args:
            query: The user's query
            query_id: Optional query identifier for logging
            
        Returns:
            A dictionary containing the query, retrieved documents, and generated answer
        """
        # Generate query ID if not provided
        if query_id is None:
            query_id = hashlib.md5(f"{query}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:10]
        
        self.logger.info(f"[QueryID: {query_id}] Processing RAG query: {query}")
        
        start_time = datetime.datetime.now()
        
        # Log the overall query process including models and device info
        query_log = {
            "query": query,
            "query_id": query_id,
            "pipeline_config": {k: v for k, v in self.cfg.__dict__.items() if not k.startswith('_')},
            "start_time": start_time.isoformat(),
            "models": {
                "embedding_model": self.cfg.embedding_model,
                "reranker_model": self.cfg.reranker_model if self.cfg.use_reranker else None,
                "llm_model": self.cfg.llm_model,
                "reranker_model_type": self.cfg.reranker_model_type if self.cfg.use_reranker else None
            },
            "device": self.cfg.device,
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version
            }
        }
        
        # Step 1: Retrieve relevant documents
        retrieval_start = datetime.datetime.now()
        self.logger.info(f"[QueryID: {query_id}] Step 1: Document retrieval")
        retrieved_docs = self.retrieve_documents(query, limit=self.cfg.top_k, query_id=query_id)
        retrieval_time = (datetime.datetime.now() - retrieval_start).total_seconds()
        
        query_log["document_retrieval"] = {
            "time_seconds": retrieval_time,
            "document_count": len(retrieved_docs)
        }
        
        # Step 2: Generate an answer using the retrieved context
        generation_start = datetime.datetime.now()
        self.logger.info(f"[QueryID: {query_id}] Step 2: Answer generation")
        answer = self.generate(query, retrieved_docs, query_id=query_id)
        generation_time = (datetime.datetime.now() - generation_start).total_seconds()
        
        query_log["answer_generation"] = {
            "time_seconds": generation_time
        }
        
        # Calculate total processing time
        end_time = datetime.datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        query_log["end_time"] = end_time.isoformat()
        query_log["total_time_seconds"] = total_time
        
        # Return the complete result
        result = {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }
        
        # Log the combined query details
        query_log["answer"] = answer
        query_log["retrieved_documents_count"] = len(retrieved_docs)
        
        self.logger.info(f"[QueryID: {query_id}] RAG query completed in {total_time:.2f} seconds: {len(retrieved_docs)} docs retrieved")
        self.log_query_details(query_id, query_log)
        
        return result


def main():
    """
    Main function to run the configurable RAG retriever from the command line.
    """
    import argparse
    from pipelines.config import get_config_by_preset
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the configurable RAG retriever")
    parser.add_argument("--db_path", type=str, default="abstracts.db", help="Path to the SQLite database")
    parser.add_argument("--preset", type=str, default="vector_only", 
                      help="Pipeline preset (vector_only, vector_plus_rerank, vector_plus_bm25, vector_bm25_rerank, "
                           "vector_bm25_rerank_llm, colbert_only, colbert_plus_rerank, full_hybrid)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for models (cpu or cuda)")
    parser.add_argument("--query", type=str, help="Query to run (if not provided, interactive mode is used)")
    
    # Add flags for individual pipeline components (override preset)
    parser.add_argument("--use_vector", type=bool, default=None, help="Use vector search")
    parser.add_argument("--use_bm25", type=bool, default=None, help="Use BM25 search")
    parser.add_argument("--use_colbert", type=bool, default=None, help="Use ColBERT retrieval")
    parser.add_argument("--use_reranker", type=bool, default=None, help="Use cross-encoder reranker")
    parser.add_argument("--use_llm_reranker", type=bool, default=None, help="Use LLM reranker")
    parser.add_argument("--use_query_decomposition", type=bool, default=None, help="Use query decomposition")
    
    args = parser.parse_args()
    
    # Get config from preset
    config = get_config_by_preset(args.preset)
    
    # Override config with CLI arguments
    config.top_k = args.top_k
    config.device = args.device
    
    # Override individual pipeline components if specified
    for flag in ['use_vector', 'use_bm25', 'use_colbert', 'use_reranker', 'use_llm_reranker', 'use_query_decomposition']:
        value = getattr(args, flag)
        if value is not None:
            setattr(config, flag, value)
    
    # Initialize the RAG retriever
    retriever = ConfigurableRAGRetriever(
        db_path=args.db_path,
        config=config
    )
    
    # Single query mode
    if args.query:
        result = retriever.rag_query(args.query)
        print("\nAnswer:\n")
        print(result["answer"])
        return
    
    # Interactive query loop
    print(f"\nConfigurable RAG Retriever - Preset: {args.preset}\n")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == "exit":
            break
            
        # Process the query
        result = retriever.rag_query(query)
        
        # Display the answer
        print("\nAnswer:\n")
        print(result["answer"])
        
        # Display retrieved documents if requested
        show_docs = input("\nShow retrieved documents? (y/n): ")
        if show_docs.lower() == "y":
            print("\nRetrieved Documents:\n")
            for i, doc in enumerate(result["retrieved_documents"]):
                print(f"Document {i+1}:")
                print(f"Title: {doc.get('title', 'Unknown')}")
                text = doc.get('abstract', '')
                print(f"Content: {text[:200]}..." if len(text) > 200 else f"Content: {text}")
                print(f"Source: {doc.get('source', 'unknown')}")
                print(f"Similarity: {doc.get('similarity', 0):.4f}")
                print()


if __name__ == "__main__":
    main()