import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Tuple, Any, Optional

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
    with vector embeddings for semantic search capabilities.
    """
    
    def __init__(self, 
                 db_path: str,
                 embedding_dim: int = 768,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 max_seq_length: int = 4096,
                 batch_size: int = 100,
                 logger: Optional[logging.Logger] = None):
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
        super().__init__(db_path, model_name, max_seq_length, batch_size, logger)
        self._setup_database()
    
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
            embedding FLOAT[{self.embedding_dim}]
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
        
        # No additional metadata needed
        metadata = {}
        
        return texts, metadata
    
    def _insert_batch(self, batch_data: List[KBDocument], embeddings: Dict[str, List[List[float]]]) -> List[int]:
        """
        Insert a batch of abstract KBDocument objects with embeddings into the database.
        
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
        
        return record_ids
    
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
            'average_time_per_abstract': processing_time / max(1, len(record_ids))
        }
        
        self.logger.info(f"Processed {len(record_ids)} abstracts in {processing_time:.2f} seconds")
        return stats
