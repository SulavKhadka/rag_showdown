import os
import sys
import json
import re
import logging
import hashlib
import sqlite3
import sqlite_vec
import requests
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
import bm25s
import rerankers

# Get the absolute path to the project root directory and add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base retriever class
from data_processor.kb_retriever import KBRetrieverBase

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

# Load environment variables
load_dotenv()


class HybridRAGRetriever(KBRetrieverBase):
    """Hybrid RAG retriever that combines vector search, BM25, and reranker for better results.
    
    This class implements a hybrid retrieval approach that:
    1. Decomposes queries into multiple sub-queries for both vector and BM25 search
    2. Retrieves documents using both vector similarity and BM25 keyword matching
    3. Combines and reranks results for improved relevance
    4. Integrates with an LLM to generate answers based on the retrieved context
    """
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "jinaai/jina-embeddings-v3",
        reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base",
        reranker_model_type: str = "cross-encoder",
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_seq_length: int = 4096,
        top_k: int = 5,
        min_similarity_pct: float = 50.0,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hybrid RAG retriever with both retrieval and LLM capabilities.
        
        Args:
            db_path: Path to the SQLite database with abstracts
            model_name: Name of the embedding model to use
            reranker_model: Model name for reranking results
            reranker_model_type: Type of reranker model
            llm_api_url: URL for the LLM API (defaults to OpenRouter)
            llm_api_key: API key for the LLM
            max_seq_length: Maximum sequence length for the embedding model
            top_k: Number of documents to retrieve
            min_similarity_pct: Minimum similarity percentage for retrieved documents
            device: Device to use for models ("cpu" or "cuda")
            logger: Optional logger instance
        """
        # Initialize the base class (retrieval capabilities)
        super().__init__(db_path, model_name, max_seq_length, logger)
        
        # Store configuration
        self.top_k = top_k
        self.min_similarity_pct = min_similarity_pct
        self.device = device
        
        # Set up LLM configuration
        self.llm_api_url = llm_api_url or "https://openrouter.ai/api/v1/chat/completions"
        self.llm_api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.llm_api_key:
            self.logger.warning("No LLM API key provided. Generation will not work.")

        # Initialize reranker
        try:
            self.logger.info("Initializing reranker")
            self.reranker = rerankers.Reranker(reranker_model, model_type=reranker_model_type, device=device)
            self.logger.info("Reranker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize reranker: {str(e)}")
            raise
            
        # Initialize BM25 retriever
        try:
            self.logger.info("Initializing BM25 retriever")
            self._initialize_bm25_retrievers()
            self.logger.info("BM25 retriever initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 retriever: {str(e)}")
            raise
        
    def get_db_connection(self):
        """
        Get a database connection using the connection string.
        
        Returns:
            A sqlite3 connection object with vector extension loaded
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
    
    def _initialize_bm25_retrievers(self):
        """
        Initialize BM25 retrievers with knowledge base content from SQLite database.
        """
        self.logger.info("Fetching documents from knowledge base")
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Query abstracts from SQLite database
        cursor.execute("SELECT id, title, abstract FROM abstracts")
        doc_results = cursor.fetchall()
        conn.close()
        
        self.doc_texts_to_ids = {}
        self.doc_texts = []
        
        for doc in doc_results:
            doc_id, title, abstract = doc
            # Use abstract as the primary text content for BM25
            content = abstract
            
            # Store mapping from content hash to document ID
            self.doc_texts_to_ids[hashlib.sha256(content.encode('utf-8')).hexdigest()] = doc_id
            self.doc_texts.append(content)

        self.logger.info(f"Creating BM25 index for {len(self.doc_texts)} documents")
        # Initialize BM25 retriever
        self.bm25_chunk_retriever = bm25s.BM25(corpus=self.doc_texts)
        self.bm25_chunk_retriever.index(bm25s.tokenize(self.doc_texts))
        self.logger.info("BM25 index created successfully")
    
    def decompose_queries(self, input_texts: List[str]) -> Dict[str, List[str]]:
        """
        Decompose input texts into RAG queries using LLM.
        
        Args:
            input_texts: List of text inputs
            
        Returns:
            Dictionary of RAG queries with 'vector_query_decomposition' and 'bm25_query_decomposition' keys
        """
        self.logger.info(f"Decomposing {len(input_texts)} input texts into queries")
        texts_str = '\n'.join(input_texts)
        
        decomposed_queries = {
            'vector_query_decomposition': [],
            'bm25_query_decomposition': []
        }

        for decomposition_prompt in [QUERY_DECOMPOSITION_PROMPT, BM25_QUERY_DECOMPOSITION_PROMPT]:            
            decomposition_type = 'vector_query_decomposition' if decomposition_prompt == QUERY_DECOMPOSITION_PROMPT else 'bm25_query_decomposition'
            
            try:
                self.logger.info(f"Sending {decomposition_type} request to LLM")
                
                # Prepare the API request to OpenRouter
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_api_key}"
                }
                
                payload = {
                    "model": "mistralai/mistral-medium-3",  # Use Mistral for query decomposition
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that decomposes queries for retrieval systems."},
                        {"role": "user", "content": decomposition_prompt.format(input_text=texts_str)}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                
                # Make the API request
                response = requests.post(
                    self.llm_api_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                response_text = result["choices"][0]["message"]["content"]
                
                # Extract JSON array from text using a more flexible approach
                import re
                queries_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if queries_match:
                    queries_json = queries_match.group(0)
                    extracted_queries = json.loads(queries_json)
                    decomposed_queries[decomposition_type] = extracted_queries
                    self.logger.info(f"Extracted {len(extracted_queries)} {decomposition_type} queries")
                else:
                    self.logger.warning("No JSON response found in LLM response")
                    decomposed_queries[decomposition_type] = []
            except Exception as e:
                self.logger.error(f"Error during query decomposition: {str(e)}")
                decomposed_queries[decomposition_type] = []
        
        return decomposed_queries
    
    def filter_context_by_relevance(self, input_texts: List[str], context: List[Dict[str, Any]]) -> List[int]:
        """
        Filter context by relevance to input texts using LLM.
        
        Args:
            input_texts: List of input texts
            context: List of context texts
            
        Returns:
            List of indices of relevant contexts
        """
        if not context or not input_texts:
            self.logger.warning("No context or input texts provided for relevance filtering")
            return list(range(len(context)))
        
        input_text = '\n\n'.join(input_texts)
        self.logger.info(f"Filtering {len(context)} contexts for relevance to {len(input_texts)} input texts")
        
        # Extract text from context objects
        context_texts = []
        for c in context:
            if isinstance(c, dict) and 'text' in c:
                context_texts.append(c['text'])
            elif isinstance(c, str):
                context_texts.append(c)
            else:
                self.logger.warning(f"Unexpected context format. Expected dict with 'text' key or string, got: {type(c)}")
                
        # Format context for LLM prompt
        context_texts_formatted = '\n\n'.join([f"[{i}] {ctx[:500]}..." for i, ctx in enumerate(context_texts)])

        try:
            # Send to LLM for relevance filtering
            self.logger.info("Sending relevance filtering request to LLM")
            
            # Prepare the API request to OpenRouter
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            # Prepare prompt for relevance filtering
            prompt = CONTEXT_RELEVANCE_FILTERING_PROMPT.format(
                query=input_text,
                passages=context_texts_formatted
            )
            
            payload = {
                "model": "mistralai/mistral-medium-3", 
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that evaluates text relevance."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }
            
            # Make the API request
            response = requests.post(
                self.llm_api_url,
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
                try:
                    relevant_indices = json.loads(indices_json)
                    self.logger.info(f"Successfully extracted {len(relevant_indices)} relevant context indices")
                    
                    # Validate indices
                    validated_indices = [idx for idx in relevant_indices if isinstance(idx, int) and 0 <= idx < len(context)]
                    self.logger.debug(f"Validated {len(validated_indices)}/{len(relevant_indices)} indices")
                    
                    if not validated_indices:
                        self.logger.warning("No valid indices found, falling back to all contexts")
                        return list(range(len(context)))
                    
                    return validated_indices
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON from LLM response: {str(e)}")
                    return list(range(len(context)))
            else:
                self.logger.warning("No JSON response found in LLM response, falling back to all contexts")
                return list(range(len(context)))
        except Exception as e:
            self.logger.error(f"Error during context filtering: {str(e)}")
            return list(range(len(context)))

    def retrieve_documents(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the knowledge base matching the query using hybrid retrieval.
        Implements the abstract method from KBRetrieverBase, combining vector search,
        BM25 retrieval and reranking.
        
        Args:
            query: Query text to match against documents
            limit: Maximum number of documents to return
            **kwargs: Additional retrieval parameters
                - decompose_query: Whether to decompose the query using LLM (default: False)
                - min_similarity_pct: Minimum similarity percentage (default: self.min_similarity_pct)
                - use_bm25: Whether to use BM25 retrieval (default: True)
                - use_vector: Whether to use vector search (default: True)
            
        Returns:
            List of document dictionaries with retrieval scores and metadata
        """
        self.logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Process parameters
        actual_limit = limit if limit is not None else self.top_k
        min_similarity = kwargs.get('min_similarity_pct', self.min_similarity_pct)
        decompose_query = kwargs.get('decompose_query', False)
        use_bm25 = kwargs.get('use_bm25', True)
        use_vector = kwargs.get('use_vector', True)
        
        # We'll build our results throughout the function
        
        try:
            # Step 1: Prepare queries
            if decompose_query:
                # Use LLM to decompose query into multiple queries for better retrieval
                query_list = [query] if isinstance(query, str) else query
                queries = self.decompose_queries(query_list)
            else:
                # Use simple query without decomposition
                queries = {
                    'vector_query_decomposition': [query] if use_vector else [],
                    'bm25_query_decomposition': [query] if use_bm25 else []
                }
            
            # Step 2: BM25 Retrieval
            bm25_retrieved_contexts = {}
            if use_bm25 and queries['bm25_query_decomposition']:
                for bm25_query in queries['bm25_query_decomposition']:
                    try:
                        # Get BM25 chunk results
                        self.logger.debug(f"Performing BM25 search for query: '{bm25_query}'")
                        bm25_chunk_texts, _ = self.bm25_chunk_retriever.retrieve(bm25s.tokenize(bm25_query), k=actual_limit)
                        bm25_chunk_texts = bm25_chunk_texts.tolist()[0]
                        
                        # Map texts to document IDs using the hash table
                        bm25_chunk_ids = []
                        for text in bm25_chunk_texts:
                            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                            if text_hash in self.doc_texts_to_ids:
                                bm25_chunk_ids.append(self.doc_texts_to_ids[text_hash])
                        
                        # Create results dictionary mapping IDs to content
                        for id, text in zip(bm25_chunk_ids, bm25_chunk_texts):
                            bm25_retrieved_contexts[id] = {'text': text, 'source': 'bm25'}
                        
                        self.logger.info(f"BM25 search returned {len(bm25_chunk_ids)} results for query: '{bm25_query[:30]}...'")
                    except Exception as e:
                        self.logger.error(f"Error in BM25 retrieval for query '{bm25_query}': {str(e)}")
            
            # Step 3: Vector Retrieval
            vector_retrieved_contexts = {}
            if use_vector and queries['vector_query_decomposition']:
                try:
                    # Generate embeddings for all vector queries
                    query_embeddings = [self.generate_query_embedding(q) for q in queries['vector_query_decomposition']]
                    
                    # Process each query and its embedding
                    for i, (vec_query, query_embedding) in enumerate(zip(queries['vector_query_decomposition'], query_embeddings)):
                        self.logger.debug(f"Performing vector search for query: '{vec_query}'")
                        
                        # Get vector search results
                        vector_results = self.vector_search_from_kb(query_embedding, top_k=actual_limit, min_similarity_pct=min_similarity)
                        vector_retrieved_contexts.update(vector_results)
                        
                        self.logger.info(f"Vector search returned {len(vector_results)} results for query {i+1}/{len(queries['vector_query_decomposition'])}")
                except Exception as e:
                    self.logger.error(f"Error in vector retrieval: {str(e)}")
            
            # Step 4: Combine and Rerank
            try:
                # Combine BM25 and vector results, deduplicating by document ID
                combined_map = bm25_retrieved_contexts.copy()
                for key, value in vector_retrieved_contexts.items():
                    if key not in combined_map:
                        combined_map[key] = value
                    else:
                        # Mark documents found by both methods
                        combined_map[key]['source'] += f",{value['source']}"
                        self.logger.debug(f"Document {key} found by both BM25 and vector search")
                
                # Extract texts for reranking
                combined_result_texts = [item['text'] for item in combined_map.values()]
                self.logger.info(f"Combined unique results: {len(combined_result_texts)}")
                
                if not combined_result_texts:
                    self.logger.warning("No results found by retrieval methods")
                    return []
                
                # Get publication dates and additional metadata
                metadata_by_id = self.get_publication_date_by_chunk_ids(list(combined_map.keys()))
                
                # Choose a query for reranker (original query)
                rerank_query = query
                
                # Rerank combined results
                self.logger.info("Reranking combined results")
                reranked_results = self.reranker.rank(rerank_query, combined_result_texts)
                
                # Process reranked results
                final_results = []
                for ranked_doc in reranked_results.results[:actual_limit]:
                    # Find document ID by text hash
                    doc_hash = hashlib.sha256(ranked_doc.text.encode('utf-8')).hexdigest()
                    doc_id = self.doc_texts_to_ids.get(doc_hash)
                    
                    if doc_id is not None:
                        # Get metadata and construct result
                        metadata = metadata_by_id.get(doc_id, {})
                        source = combined_map[doc_id]['source']
                        
                        # Format according to expected interface
                        document_result = {
                            'id': doc_id,
                            'title': metadata.get('title', 'Unknown'),
                            'text': ranked_doc.text,  # Main content
                            'abstract': ranked_doc.text,  # Use text as abstract
                            'published': metadata.get('publication_date'),
                            'authors': metadata.get('authors', []),
                            'source': source,
                            'similarity': 1.0 - min(1.0, max(0.0, ranked_doc.score)),  # Convert score to similarity
                            'metadata': {  # Additional metadata
                                'source': source,
                                'rerank_score': ranked_doc.score,
                                'publication_date': metadata.get('publication_date'),
                                'title': metadata.get('title', 'Unknown')
                            }
                        }
                        
                        final_results.append(document_result)
                
                self.logger.info(f"Retrieved {len(final_results)} documents")
                return final_results
                
            except Exception as e:
                self.logger.error(f"Error combining results: {str(e)}")
                return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def vector_search_from_kb(self, query_embedding: List[float], top_k: int = 5, min_similarity_pct: float = 50) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve context using embeddings based cosine similarity search from the SQLite knowledge base.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            min_similarity_pct: Minimum similarity percentage threshold
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        self.logger.info(f"Performing vector search with top_k={top_k}")
        
        # Convert min_similarity_pct to distance threshold (cosine distance = 1 - similarity)
        distance_threshold = 1.0 - (min_similarity_pct / 100.0)
        self.logger.debug(f"Distance threshold: {distance_threshold}")
        
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
            ''', (query_blob, top_k)).fetchall()
            
            # Process results
            returned_contexts = {}
            for row in results:
                doc_id, title, abstract, authors_json, published, source_file, content, similarity = row
                
                # Skip results that don't meet the similarity threshold
                if similarity < (min_similarity_pct / 100.0):
                    continue
                    
                # Use abstract as the main text content
                text_content = abstract
                
                # Add to returned contexts
                returned_contexts[doc_id] = {
                    'text': text_content,
                    'title': title, 
                    'source': 'vector',
                    'similarity': similarity
                }
            
            conn.close()
            self.logger.info(f"Vector search returned {len(returned_contexts)} results")
            return returned_contexts
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return {}
    
    def get_publication_date_by_chunk_ids(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get publication dates and metadata for document IDs.
        
        Args:
            ids: List of document IDs to retrieve metadata for
            
        Returns:
            Dictionary mapping document IDs to metadata dictionaries
        """
        if not ids:
            return {}

        # Create comma-separated string of IDs for SQL query
        ids_str = ', '.join([str(i) for i in ids])
        self.logger.info(f"Getting metadata for document IDs: {ids_str if len(ids_str) < 100 else ids_str[:100] + '...'}")

        try:
            # Get database connection
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Fetch publication dates and metadata
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

            all_found = len(metadata_by_id) == len(ids)
            self.logger.info(f"Found metadata for {len(metadata_by_id)}/{len(ids)} documents. All found: {all_found}")

            return metadata_by_id
        except Exception as e:
            self.logger.error(f"Error retrieving document metadata: {str(e)}")
            return {}  

    def retrieve_context(self, queries: Dict[str, List[str]], top_k: int = 5, min_similarity_pct: float = 50) -> List[Dict[str, Any]]:
        """
        Retrieve context using multiple methods and rerank results.
        
        Args:
            queries: Dictionary containing decomposed queries for different retrieval methods
            top_k: Number of top results to return after reranking
            min_similarity_pct: Minimum similarity percentage threshold
            
        Returns:
            List of relevant context passages
        """
        # Empty list to store final contexts
        contexts = []
        
        # Step 1: BM25 Retrieval
        self.logger.info("Starting BM25 retrieval phase")
        bm25_retrieved_contexts = {}
        for query in queries['bm25_query_decomposition']:
            try:
                # Get BM25 chunk results
                self.logger.debug(f"Performing BM25 search for query: '{query}'")
                bm25_chunk_texts, _ = self.bm25_chunk_retriever.retrieve(bm25s.tokenize(query), k=top_k)
                bm25_chunk_texts = bm25_chunk_texts.tolist()[0]
                
                # Map texts to document IDs using the hash table
                bm25_chunk_ids = []
                for text in bm25_chunk_texts:
                    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    if text_hash in self.doc_texts_to_ids:
                        bm25_chunk_ids.append(self.doc_texts_to_ids[text_hash])
                
                # Create results dictionary mapping IDs to content
                for id, text in zip(bm25_chunk_ids, bm25_chunk_texts):
                    bm25_retrieved_contexts[id] = {'text': text, 'source': 'bm25'}
                
                self.logger.info(f"BM25 search returned {len(bm25_chunk_ids)} results for query: '{query[:30]}...'")
            except Exception as e:
                self.logger.error(f"Error in BM25 retrieval for query '{query}': {str(e)}")

        # Step 2: Vector Retrieval
        self.logger.info("Starting vector retrieval phase")
        vector_retrieved_contexts = {}
        
        # Generate embeddings for all vector queries at once
        try:
            query_embeddings = [self.generate_query_embedding(query) for query in queries['vector_query_decomposition']]
            
            # Process each query and its embedding
            for i, (query, query_embedding) in enumerate(zip(queries['vector_query_decomposition'], query_embeddings)):
                self.logger.info(f"Performing vector search for query: '{query}'")
                
                # Get vector search results
                vector_results = self.vector_search_from_kb(query_embedding, top_k=top_k, min_similarity_pct=min_similarity_pct)
                vector_retrieved_contexts.update(vector_results)
                
                self.logger.info(f"Vector search returned {len(vector_results)} results for query {i+1}/{len(queries['vector_query_decomposition'])}")
        except Exception as e:
            self.logger.error(f"Error in vector retrieval: {str(e)}")

        # Step 3: Combine and Rerank Results
        try:
            # Combine BM25 and vector results, deduplicating by document ID
            combined_map = bm25_retrieved_contexts.copy()
            for key, value in vector_retrieved_contexts.items():
                if key not in combined_map:
                    combined_map[key] = value
                else:
                    # Mark documents found by both methods
                    combined_map[key]['source'] += f",{value['source']}"
                    self.logger.debug(f"Document {key} found by both BM25 and vector search")
            
            # Extract texts for reranking
            combined_result_texts = [item['text'] for item in combined_map.values()]
            self.logger.info(f"Combined unique results: {len(combined_result_texts)}")
            
            if not combined_result_texts:
                self.logger.warning("No results found by either retrieval method")
                return []
            
            # Get publication dates and additional metadata
            metadata_by_id = self.get_publication_date_by_chunk_ids(list(combined_map.keys()))
            
            # Choose a query for reranker (first vector query)
            rerank_query = queries['vector_query_decomposition'][0] if queries['vector_query_decomposition'] else ""
            
            # Rerank combined results
            self.logger.info("Reranking combined results")
            reranked_results = self.reranker.rank(rerank_query, combined_result_texts)
            
            # Process reranked results
            final_results = []
            for ranked_doc in reranked_results.results[:top_k]:
                # Find document ID by text hash
                doc_hash = hashlib.sha256(ranked_doc.text.encode('utf-8')).hexdigest()
                doc_id = self.doc_texts_to_ids.get(doc_hash)
                
                if doc_id is not None:
                    # Get metadata and construct result
                    metadata = metadata_by_id.get(doc_id, {})
                    metadata['source'] = combined_map[doc_id]['source']
                    metadata['rerank_score'] = ranked_doc.score
                    
                    final_results.append({
                        "id": doc_id,
                        "text": ranked_doc.text,
                        "title": metadata.get('title', 'Unknown'),
                        "metadata": metadata,
                        "similarity": metadata.get('similarity', 0),
                    })
            
            self.logger.info(f"Returning {len(final_results)} reranked results")
            contexts = final_results
            
        except Exception as e:
            self.logger.error(f"Error in result combination and reranking: {str(e)}")
        
        return contexts
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using an LLM with the retrieved context.
        
        Args:
            query: The user's query
            context_docs: The retrieved context documents
            
        Returns:
            The generated answer
        """
        self.logger.info(f"Generating answer for query: {query}")
        
        if not self.llm_api_key:
            self.logger.error("No LLM API key provided")
            return "Error: LLM API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        try:
            # Prepare context for prompt
            context_text = ""
            for i, doc in enumerate(context_docs):
                doc_text = doc.get('text', '')
                doc_title = doc.get('title', 'Unknown Document')
                doc_source = doc.get('metadata', {}).get('source', 'unknown')
                
                context_text += f"Document {i+1}: {doc_title}\n"
                context_text += f"Source: {doc_source}\n"
                context_text += f"Content: {doc_text}\n\n"
            
            # Create the prompt
            prompt = f"""Answer the following question based only on the provided context documents. If the context doesn't contain enough information to answer the question directly, say so, but try to provide the most helpful response possible based on the available information.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"""
            
            # Call LLM API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            payload = {
                "model": "mistralai/mistral-medium-3",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.llm_api_url,
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
        
        # Step 1: Process input to get decomposed queries and retrieve context
        result = self.process_input(query, top_k=self.top_k)
        
        # Step 2: Generate an answer using the retrieved context
        answer = self.generate(query, result["contexts"])
        
        # Return the complete result
        result["answer"] = answer
        
        return result
    
    def process_input(self, input_text: Union[str, List[str]], top_k: int = 3) -> Dict[str, Any]:
        """
        Process input text to retrieve relevant context.
        
        Args:
            input_text: Single string or list of strings to process
            top_k: Number of top context passages to return per query
            
        Returns:
            Dictionary with input texts and retrieved context
        """
        self.logger.info(f"Processing input of type: {type(input_text)}")
        
        # Convert single string to list for consistent processing
        if isinstance(input_text, str):
            input_text = [input_text]
        
        try:
            # Decompose queries
            rag_queries = self.decompose_queries(input_text)
            self.logger.info(f"Decomposed input into {len(rag_queries['vector_query_decomposition'])} vector queries and {len(rag_queries['bm25_query_decomposition'])} BM25 queries")
            
            # Retrieve context for each query
            contexts = self.retrieve_context(rag_queries, top_k=top_k, min_similarity_pct=self.min_similarity_pct)

            # Filter by relevance if we have enough contexts
            if contexts and len(contexts) > 1:
                self.logger.debug(f"Filtering {len(contexts)} total contexts by relevance")
                filtered_context_indexes = self.filter_context_by_relevance(input_text, contexts)
                filtered_contexts = [contexts[i] for i in filtered_context_indexes]
                self.logger.info(f"Filtered down to {len(filtered_contexts)}/{len(contexts)} relevant contexts")
            else:
                self.logger.warning("Not enough contexts for relevance filtering")
                filtered_contexts = contexts

            return {
                "query": input_text[0] if len(input_text) == 1 else input_text,
                "queries": rag_queries,
                "contexts": filtered_contexts
            }
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return {
                "query": input_text[0] if len(input_text) == 1 else input_text,
                "queries": {},
                "contexts": []
            }
    
    def close(self):
        """Close database connections and cleanup resources."""
        self.logger.info("Closing database connections")


def main():
    """
    Main function to run the hybrid RAG retriever from the command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the hybrid RAG retriever")
    parser.add_argument("--db_path", type=str, default="abstracts.db", help="Path to the SQLite database")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v3", help="Name of the embedding model")
    parser.add_argument("--reranker_model", type=str, default="Alibaba-NLP/gte-reranker-modernbert-base", help="Name of the reranker model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for models (cpu or cuda)")
    args = parser.parse_args()
    
    # Initialize the RAG retriever
    retriever = HybridRAGRetriever(
        db_path=args.db_path,
        model_name=args.model_name,
        reranker_model=args.reranker_model,
        top_k=args.top_k,
        device=args.device
    )
    
    # Interactive query loop
    print("\nHybrid RAG Retriever\n")
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
            for i, doc in enumerate(result["contexts"]):
                print(f"Document {i+1}:")
                print(f"Title: {doc.get('title', 'Unknown')}")
                text = doc.get('text', '')
                print(f"Content: {text[:200]}..." if len(text) > 200 else f"Content: {text}")
                print(f"Source: {doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"Score: {doc.get('metadata', {}).get('rerank_score', 0):.4f}")
                print()


if __name__ == "__main__":
    main()
