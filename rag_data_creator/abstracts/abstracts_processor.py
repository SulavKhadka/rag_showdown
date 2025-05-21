import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Tuple, Any, Optional

# Import PyLate for ColBERT indexing
from pylate import indexes, models

# Get the absolute path to the rag_data_creator directory
rag_data_creator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add to path if needed
if rag_data_creator_dir not in sys.path:
    sys.path.insert(0, rag_data_creator_dir)

# Use absolute import instead of relative import
from data_processor.kb_processor import KBProcessorBase, KBDocument, KBDocumentMetadata

class SQLiteKBProcessor(KBProcessorBase):
    """
    Implementation of KBProcessorBase using SQLite with sqlite-vec extension for scientific abstracts.
    
    This class processes scientific paper abstracts from JSON files and stores them in an SQLite database
    with vector embeddings for semantic search capabilities. It supports dual embedding approaches:
    
    1. Regular embeddings using SentenceTransformer models stored in SQLite
    2. ColBERT embeddings using PyLate and PLAID index for more advanced retrieval
    
    The ColBERT integration allows for more advanced token-level late interaction, which can provide
    more accurate retrieval, especially for longer documents or complex queries.
    """
    
    def __init__(self, 
                 db_path: str,
                 embedding_dim: int = 768,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 max_seq_length: int = 4096,
                 batch_size: int = 100,
                 logger: Optional[logging.Logger] = None,
                 use_colbert: bool = False,
                 colbert_model_name: str = "lightonai/GTE-ModernColBERT-v1",
                 plaid_index_folder: str = "plaid-index",
                 plaid_index_name: str = "abstracts_index"):
        """
        Initialize the SQLite knowledge base processor.
        
        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of the embedding vectors
            model_name: Name of the embedding model to use
            max_seq_length: Maximum sequence length for the embedding model
            batch_size: Number of records to process in a batch before DB insertion
            logger: Optional logger instance; if not provided, a new one will be created
        """
        self.embedding_dim = embedding_dim
        
        # ColBERT configuration
        self.use_colbert = use_colbert
        self.colbert_model_name = colbert_model_name
        self.plaid_index_folder = plaid_index_folder
        self.plaid_index_name = plaid_index_name
        self.colbert_model = None
        self.plaid_index = None
        
        super().__init__(db_path, model_name, max_seq_length, batch_size, logger)
        self._setup_database()
        
        # Initialize ColBERT if enabled
        if self.use_colbert:
            self._setup_colbert()
    
    def _setup_colbert(self):
        """
        Initialize the ColBERT model and PLAID index for efficient retrieval.
        """
        self.logger.info(f"Initializing ColBERT model: {self.colbert_model_name}")
        
        try:
            # Initialize the ColBERT model
            self.colbert_model = models.ColBERT(
                model_name_or_path=self.colbert_model_name,
            )
            
            # Create the index directory if it doesn't exist
            os.makedirs(self.plaid_index_folder, exist_ok=True)
            
            # Initialize the PLAID index
            self.plaid_index = indexes.PLAID(
                index_folder=self.plaid_index_folder,
                index_name=self.plaid_index_name,
                override=False,  # Don't override existing index
            )
            
            self.logger.info("ColBERT model and PLAID index initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing ColBERT: {str(e)}")
            self.use_colbert = False  # Disable ColBERT if initialization fails
    
    def _setup_database(self):
        """
        Create the necessary database tables for document storage and vector search.
        This includes both regular tables and the vec0 virtual tables for embeddings.
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Create abstracts table for storing scientific paper data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS abstracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT NOT NULL,  -- JSON array of author names
            link TEXT NOT NULL,
            published TEXT NOT NULL,
            source_file TEXT NOT NULL, -- Path to the source JSON file
            content TEXT NOT NULL,  -- Document content (same as abstract for this processor)
            embedding BLOB
        )
        ''')
        
        # Create vector search table for abstract embeddings
        cursor.execute(f'''
        CREATE VIRTUAL TABLE IF NOT EXISTS vss_abstracts USING vec0(
            embedding FLOAT[{self.embedding_dim}] distance_metric=cosine
        )
        ''')
        
        # Add indices for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_abstracts_source ON abstracts(source_file)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_abstracts_title ON abstracts(title)')
        
        conn.commit()
        conn.close()
        
        self.logger.debug("Database tables created successfully")
    
    def _parse_source(self, source_path: str) -> List[KBDocument]:
        """
        Parse a scientific abstracts JSON file into a list of KBDocument objects.
        
        Args:
            source_path: Path to the source file to parse
            
        Returns:
            List of KBDocument objects, each representing a scientific paper abstract
        """
        documents = []
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                abstracts_data = json.load(f)
            
            self.logger.info(f"Processing {len(abstracts_data)} abstracts from {source_path}")
            
            # Process each abstract in the dataset
            for i, paper in enumerate(abstracts_data):
                # Extract paper information
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                authors = paper.get('authors', [])
                link = paper.get('link', '')
                published = paper.get('published', '')
                
                if not title or not abstract:
                    self.logger.warning(f"Skipping paper with missing title or abstract at index {i}")
                    continue
                
                # Create a structured context that includes all paper information
                # Store as JSON string to maintain structure while fitting in the metadata model
                context_data = {
                    'title': title,
                    'authors': authors,
                    'link': link
                }
                
                # Create metadata object with all necessary information
                metadata = KBDocumentMetadata(
                    source=source_path,
                    context=json.dumps(context_data),  # Store structured data in context
                    publish_date=published,
                    confidence=1.0,  # Full confidence for direct data
                    parent_ids=[],   # No hierarchical relationships
                    children_ids=[]  # No hierarchical relationships
                )
                
                # Create document with metadata and content
                document = KBDocument(
                    id=i,  # Temporary ID, will be replaced upon insertion
                    metadata=metadata,
                    content=abstract  # Use the abstract text as the document content
                )
                
                documents.append(document)
        
        except Exception as e:
            self.logger.error(f"Error parsing {source_path}: {str(e)}")
        
        return documents
    
    def _prepare_embeddings(self, parsed_data: List[KBDocument]) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Prepare texts for embedding from parsed abstract KBDocument objects.
        
        Args:
            parsed_data: List of KBDocument objects to prepare for embedding
            
        Returns:
            A tuple containing:
            - A dictionary mapping field names to lists of texts to embed
            - Any additional metadata needed to map embeddings back to KBDocument objects
        """
        # Extract text fields to be embedded
        texts = {
            'abstract': []
        }
        
        # For ColBERT, we'll use the same format for now
        # We could customize this format if needed in the future
        if self.use_colbert:
            texts['colbert'] = []
        
        # For each paper, create a rich text representation for embedding
        for doc in parsed_data:
            # Extract title from metadata context
            try:
                context_data = json.loads(doc.metadata.context)
                title = context_data.get('title', '')
            except (json.JSONDecodeError, AttributeError):
                # Fallback for backward compatibility
                title = doc.metadata.context  # In the old format, context was the title
            
            # Combine title and content (abstract) for better semantic representation
            rich_text = f"Title: {title}\n\nAbstract: {doc.content}"
            texts['abstract'].append(rich_text)
            
            # Use the same format for ColBERT embeddings
            if self.use_colbert:
                texts['colbert'].append(rich_text)
        
        # No additional metadata needed
        metadata = {}
        
        return texts, metadata
    
    def generate_colbert_embeddings(self, texts: List[str], is_query: bool = False) -> List[Any]:
        """
        Generate ColBERT embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            is_query: Boolean indicating if the texts are queries or documents
            
        Returns:
            List of ColBERT embeddings (tensor sequences)
        """
        if not self.use_colbert or not texts:
            return []
        
        self.logger.debug(f"Generating ColBERT embeddings for {len(texts)} texts")
        try:
            embeddings = self.colbert_model.encode(
                texts,
                batch_size=min(32, len(texts)),
                is_query=is_query,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating ColBERT embeddings: {str(e)}")
            return []
    
    def _insert_batch(self, batch_data: List[KBDocument], embeddings: Dict[str, List[List[float]]]) -> List[int]:
        """
        Insert a batch of abstract KBDocument objects with embeddings into the database.
        Also adds documents to ColBERT PLAID index if enabled.
        
        Args:
            batch_data: List of KBDocument objects to insert
            embeddings: Dictionary mapping field names to lists of embeddings
            
        Returns:
            List of primary key IDs for the inserted records
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Get abstract embeddings
        abstract_embeddings = embeddings.get('abstract', [])
        record_ids = []
        
        # Make sure we have embeddings for all abstracts
        if len(batch_data) != len(abstract_embeddings):
            self.logger.error(f"Mismatch between batch_data length ({len(batch_data)}) and embeddings length ({len(abstract_embeddings)})")
            return []
        
        # Prepare ColBERT embeddings if enabled
        colbert_docs_embeddings = None
        if self.use_colbert and 'colbert' in embeddings:
            # Generate ColBERT embeddings for documents
            colbert_texts = embeddings.get('colbert', [])
            if colbert_texts:
                colbert_docs_embeddings = self.generate_colbert_embeddings(
                    colbert_texts, is_query=False
                )
        
        # Insert abstracts and their embeddings
        for i, doc in enumerate(batch_data):
            embedding = abstract_embeddings[i]
            
            # Serialize embedding to binary format for storage
            embedding_blob = self.serialize_embedding(embedding)
            
            # Extract paper information from metadata context
            try:
                context_data = json.loads(doc.metadata.context)
                title = context_data.get('title', '')
                authors = context_data.get('authors', [])
                link = context_data.get('link', '')
            except (json.JSONDecodeError, AttributeError):
                # Fallback for backward compatibility or if context is not properly formatted
                title = doc.metadata.context  # In the old format, context was the title
                authors = []
                link = ''
            
            # Get abstract from content
            abstract = doc.content
            
            # Insert into abstracts table
            cursor.execute('''
            INSERT INTO abstracts 
            (title, abstract, authors, link, published, source_file, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                title,
                abstract,
                json.dumps(authors),
                link,
                doc.metadata.publish_date,
                doc.metadata.source,
                doc.content,  # Add the content field from the KBDocument
                embedding_blob
            ))
            
            # Get the ID of the inserted abstract
            abstract_id = cursor.lastrowid
            record_ids.append(abstract_id)
            
            # Insert into vector search table
            cursor.execute('''
            INSERT INTO vss_abstracts (rowid, embedding)
            VALUES (?, ?)
            ''', (abstract_id, embedding_blob))
        
        conn.commit()
        conn.close()
        
        # Add documents to ColBERT PLAID index if enabled
        if self.use_colbert and self.plaid_index and colbert_docs_embeddings and len(colbert_docs_embeddings) > 0:
            try:
                self.logger.info(f"Adding {len(record_ids)} documents to ColBERT PLAID index")
                
                # Convert record IDs to strings for PLAID index
                documents_ids = [str(record_id) for record_id in record_ids]
                
                # Add documents to PLAID index
                self.plaid_index.add_documents(
                    documents_ids=documents_ids,
                    documents_embeddings=colbert_docs_embeddings,
                )
                
                self.logger.info("Successfully added documents to ColBERT PLAID index")
            except Exception as e:
                self.logger.error(f"Error adding documents to ColBERT PLAID index: {str(e)}")
        
        return record_ids
    
    def process_source(self, source_path: str) -> List[int]:
        """
        Process a single source file and insert its records into the database.
        Overrides the base class method to add support for ColBERT embeddings.
        
        Args:
            source_path: Path to the source file
            
        Returns:
            List of primary key IDs for the inserted records
        """
        self.logger.debug(f"Processing source: {source_path}")
        
        try:
            # Parse the source file into KBDocument objects
            kb_documents = self._parse_source(source_path)
            self.logger.debug(f"Parsed {len(kb_documents)} KBDocument objects from source")
            
            # Process in batches
            all_record_ids = []
            
            for i in range(0, len(kb_documents), self.batch_size):
                batch = kb_documents[i:i+self.batch_size]
                self.logger.debug(f"Processing batch {i//self.batch_size + 1}, size: {len(batch)}")
                
                # Prepare texts for embedding from KBDocument objects
                texts_to_embed, metadata = self._prepare_embeddings(batch)
                
                # Generate embeddings for each field
                embeddings = {}
                for field_name, texts in texts_to_embed.items():
                    # For regular embeddings (not colbert), use the regular embedding method
                    if field_name != 'colbert':
                        embeddings[field_name] = self.generate_embeddings(texts, text_type="passage")
                    # For colbert field, we'll let the _insert_batch method handle it directly
                    # as ColBERT embeddings are special tensor sequences, not just float lists
                    else:
                        # Store the texts only, will be embedded in _insert_batch
                        embeddings[field_name] = texts
                
                # Insert batch of KBDocument objects with embeddings
                record_ids = self._insert_batch(batch, embeddings)
                all_record_ids.extend(record_ids)
                
            return all_record_ids
            
        except Exception as e:
            self.logger.error(f"Error processing source {source_path}: {str(e)}")
            self.logger.debug("Full exception details:", exc_info=True)
            return []

    def process_abstracts_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single abstracts JSON file and insert its contents into the database.
        
        Args:
            file_path: Path to the JSON file containing abstracts
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Processing abstracts file: {file_path}")
        start_time = datetime.datetime.now()
        
        # Process the source file
        record_ids = self.process_source(file_path)
        
        # Compute statistics
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats = {
            'file_path': file_path,
            'abstracts_processed': len(record_ids),
            'processing_time_seconds': processing_time,
            'average_time_per_abstract': processing_time / max(1, len(record_ids)),
            'colbert_enabled': self.use_colbert
        }
        
        self.logger.info(f"Processed {len(record_ids)} abstracts in {processing_time:.2f} seconds")
        return stats
