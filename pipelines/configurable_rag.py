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
from typing import List, Dict, Any, Union, Optional, Tuple

# Get the absolute path to the project root directory and add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_processor.kb_retriever import KBRetrieverBase
from pipelines.config import PipelineConfig, get_config_by_preset

# Import conditional dependencies
try:
    import bm25s
    import rerankers
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
    2. Retrieve documents using vector similarity, BM25, or both
    3. Optionally rerank results using cross-encoder and/or LLM filtering
    4. Generate answers based on the retrieved context
    
    The specific behavior is controlled by the provided PipelineConfig.
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
        
        # Lazy-loaded components
        self._reranker = None
        self._bm25 = None
        
        # Set up LLM API key
        self.llm_api_key = config.llm_api_key
        
        # Check important API components
        if (config.use_query_decomposition or config.use_llm_reranker) and not self.llm_api_key:
            self.logger.warning("LLM API key not provided, but LLM-based features are enabled. "
                               "Query decomposition and LLM reranking will not work.")
    
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
    
    def _vector_retrieve(self, queries: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve documents using vector similarity search.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        self.logger.info(f"Performing vector retrieval for {len(queries)} queries")
        
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
                if isinstance(c, dict) and 'text' in c:
                    context_texts.append(c['text'])
                else:
                    context_texts.append(str(c))
            
            # Format context for LLM prompt
            context_texts_formatted = '\n\n'.join([f"[{i}] {ctx[:500]}..." for i, ctx in enumerate(context_texts)])
            
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
                doc_id, title, abstract, authors_json, published, source_file, content, similarity = row
                
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
        
        Args:
            query: Query text to match against documents
            limit: Maximum number of documents to return (overrides config.top_k)
            **kwargs: Additional retrieval parameters
                - override_config: Optional PipelineConfig to use for this request only
                
        Returns:
            List of document dictionaries with retrieval scores and metadata
        """
        self.logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Use the specified limit or fall back to the default top_k
        actual_limit = limit if limit is not None else self.cfg.top_k
        
        # Check for an override configuration for this request only
        config = kwargs.get('override_config', self.cfg)
        
        # Validate that at least one retrieval method is enabled
        if not (config.use_vector or config.use_bm25):
            self.logger.error("No retrieval methods enabled. At least one of vector or BM25 must be enabled.")
            raise ValueError("At least one retrieval method (vector or BM25) must be enabled")
        
        try:
            # Step 1: Prepare queries (decompose if configured)
            if config.use_query_decomposition:
                # Use LLM to decompose query into multiple queries for better retrieval
                input_queries = [query] if isinstance(query, str) else [query]
                queries = self._decompose_query(input_queries)
            else:
                # Use simple query without decomposition
                queries = {
                    'vector_query_decomposition': [query] if config.use_vector else [],
                    'bm25_query_decomposition': [query] if config.use_bm25 else []
                }
            
            # Step 2: Retrieve documents using enabled methods
            vector_results = {}
            bm25_results = {}
            
            # Vector retrieval
            if config.use_vector and queries['vector_query_decomposition']:
                vector_results = self._vector_retrieve(queries['vector_query_decomposition'])
            
            # BM25 retrieval
            if config.use_bm25 and queries['bm25_query_decomposition']:
                bm25_results = self._bm25_retrieve(queries['bm25_query_decomposition'])
            
            # Step 3: Combine results from different retrieval methods
            combined_map = self._combine_results(vector_results, bm25_results)
            
            # Early return if no results
            if not combined_map:
                self.logger.warning("No results found by retrieval methods")
                return []
            
            # Get document metadata
            metadata_by_id = self._get_metadata_by_ids(list(combined_map.keys()))
            
            # Extract text for reranking
            doc_texts = [item['text'] for item in combined_map.values()]
            doc_ids = list(combined_map.keys())
            
            # Step 4: Rerank results if configured
            if config.use_reranker and doc_texts:
                self.logger.info("Reranking combined results with cross-encoder")
                reranked = self._classic_rerank(query, doc_texts)
                
                # Create a list of ranked results with document IDs
                ranked_results = []
                for i, ranked_doc in enumerate(reranked):
                    if i < len(doc_ids):
                        doc_id = doc_ids[i]
                        doc_data = combined_map[doc_id]
                        
                        # Create ranked result
                        ranked_results.append({
                            'id': doc_id,
                            'text': ranked_doc['text'],
                            'score': ranked_doc['score'],
                            'source': doc_data['source'],
                            'metadata': metadata_by_id.get(doc_id, {})
                        })
            else:
                # Skip reranking, just format results
                ranked_results = []
                for doc_id, doc_data in combined_map.items():
                    ranked_results.append({
                        'id': doc_id,
                        'text': doc_data['text'],
                        'score': doc_data.get('similarity', 0.5),  # Use similarity if available
                        'source': doc_data['source'],
                        'metadata': metadata_by_id.get(doc_id, {})
                    })
            
            # Step 5: Apply LLM relevance filtering if configured
            if config.use_llm_reranker and ranked_results:
                self.logger.info("Applying LLM relevance filtering")
                relevant_indices = self._llm_rerank(query, ranked_results)
                filtered_results = [ranked_results[i] for i in relevant_indices]
                self.logger.info(f"LLM filtering kept {len(filtered_results)}/{len(ranked_results)} results")
            else:
                filtered_results = ranked_results
            
            # Sort by score (higher is better)
            sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
            
            # Limit to requested number of results
            final_results = sorted_results[:actual_limit]
            
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
                    'similarity': res['score'],
                    'metadata': {
                        'source': res['source'],
                        'rerank_score': res['score'],
                        'publication_date': metadata.get('publication_date'),
                        'title': metadata.get('title', 'Unknown')
                    }
                }
                formatted_results.append(formatted_doc)
            
            self.logger.info(f"Retrieved {len(formatted_results)} documents")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            self.logger.debug("Full exception details:", exc_info=True)
            return []
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using an LLM with the retrieved context.
        
        Args:
            query: The user's query
            context_docs: The retrieved context documents
            
        Returns:
            The generated answer
        """
        self.logger.info("Generating answer from retrieved context")
        
        if not self.llm_api_key:
            self.logger.error("No LLM API key provided")
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
                
                similarity = doc.get("similarity", 0.0)
                
                context_text += f"Document {i+1} [Similarity: {similarity:.2f}]\n"
                context_text += f"Title: {title}\n"
                context_text += f"Authors: {authors_text}\n"
                context_text += f"Abstract: {abstract}\n\n"
            
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
            
            self.logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            self.logger.debug("Full exception details:", exc_info=True)
            return f"Error generating answer: {str(e)}"
    
    def rag_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline by retrieving documents and generating an answer.
        
        Args:
            query: The user's query
            
        Returns:
            A dictionary containing the query, retrieved documents, and generated answer
        """
        self.logger.info(f"Processing RAG query: {query}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, limit=self.cfg.top_k)
        
        # Step 2: Generate an answer using the retrieved context
        answer = self.generate(query, retrieved_docs)
        
        # Return the complete result
        result = {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }
        
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
                      help="Pipeline preset (vector_only, vector_plus_rerank, vector_plus_bm25, vector_bm25_rerank, vector_bm25_rerank_llm, full_hybrid)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for models (cpu or cuda)")
    parser.add_argument("--query", type=str, help="Query to run (if not provided, interactive mode is used)")
    
    # Add flags for individual pipeline components (override preset)
    parser.add_argument("--use_vector", type=bool, default=None, help="Use vector search")
    parser.add_argument("--use_bm25", type=bool, default=None, help="Use BM25 search")
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
    for flag in ['use_vector', 'use_bm25', 'use_reranker', 'use_llm_reranker', 'use_query_decomposition']:
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