import os
import logging
import sqlite3
import sqlite_vec
import struct
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


class KBDocumentMetadata(BaseModel):
    source: str
    context: str
    publish_date: str
    confidence: float
    parent_ids: list[int]
    children_ids: list[int]
    

class KBDocument(BaseModel):
    id: int
    metadata: KBDocumentMetadata
    content: str


class KBProcessorBase(ABC):
    """
    Abstract base class for knowledge base processors.
    
    This class provides a common structure and utilities for processing different types of datasets
    and storing them in a database with embeddings for RAG applications.
    
    All data is processed as KBDocument objects to provide a consistent interface.
    
    Concrete subclasses must implement the abstract methods to handle dataset-specific parsing,
    embedding preparation, and database insertion.
    """
    
    def __init__(self, 
                 conn_string: str,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 max_seq_length: int = 4096,
                 batch_size: int = 100,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the base knowledge base processor.
        
        Args:
            conn_string: Database connection string
            model_name: Name of the embedding model to use
            max_seq_length: Maximum sequence length for the embedding model
            batch_size: Number of records to process in a batch before DB insertion
            logger: Optional logger instance; if not provided, a new one will be created
        """
        # Set up logger
        self.logger = logger or self._setup_logger()
        self.logger.debug(f"Initializing {self.__class__.__name__}")
        
        # Store configuration
        self.conn_string = conn_string
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        # Initialize the embedding model
        self.model = self.initialize_embedding_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger instance."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def initialize_embedding_model(self) -> SentenceTransformer:
        """
        Initialize the sentence transformer model for embeddings.
        
        Returns:
            Initialized SentenceTransformer model
        """
        self.logger.debug(f"Initializing embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name, trust_remote_code=True)
        model.max_seq_length = self.max_seq_length
        self.logger.debug(f"Model initialized with max sequence length: {model.max_seq_length}")
        return model
    
    def generate_embeddings(self, texts: List[str], text_type: str = "passage") -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            text_type: Either "query" or "passage" to indicate the type of text being embedded
            
        Returns:
            List of embeddings as Python lists (not NumPy arrays)
        """
        assert text_type in ["query", "passage"], "Invalid text type"
        
        if not texts:
            return []
        
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            task = "retrieval.query" if text_type == "query" else "retrieval.passage",
            convert_to_tensor=False, 
            show_progress_bar=False)
        
        # Convert NumPy arrays to Python lists for SQLite compatibility
        embeddings_as_lists = [embedding.tolist() for embedding in embeddings]
        return embeddings_as_lists
    
    def get_db_connection(self):
        """
        Get a database connection using the connection string.
        
        Returns:
            A sqlite3 connection object with vector extension loaded
        """
        conn = sqlite3.connect(self.conn_string)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Load the sqlite-vec extension
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            self.logger.error(f"Failed to load sqlite-vec extension: {str(e)}")
            self.logger.info("Make sure sqlite-vec is properly installed: pip install sqlite-vec")
            raise
        
        return conn
    
    @abstractmethod
    def _parse_source(self, source_path: str) -> List[KBDocument]:
        """
        Parse a single source file into a list of KBDocument objects.
        
        Args:
            source_path: Path to the source file to parse
            
        Returns:
            List of KBDocument objects, each representing a record from the source
        """
        pass
    
    @abstractmethod
    def _prepare_embeddings(self, parsed_data: List[KBDocument]) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Prepare texts for embedding from parsed KBDocument objects.
        
        This method should extract all text fields that need embeddings from the KBDocument objects.
        
        Args:
            parsed_data: List of KBDocument objects to prepare for embedding
            
        Returns:
            A tuple containing:
            - A dictionary mapping field names to lists of texts to embed
              (e.g., {'title': [title1, title2, ...], 'content': [content1, content2, ...]})
            - Any additional metadata needed to map embeddings back to KBDocument objects
        """
        pass
    
    def serialize_embedding(self, embedding: List[float]) -> bytes:
        """
        Serialize a list of floats into a binary blob for SQLite storage.
        
        Args:
            embedding: List of float values representing an embedding
            
        Returns:
            Serialized binary data
        """
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    def deserialize_embedding(self, blob: bytes) -> List[float]:
        """
        Deserialize a binary blob back into a list of floats.
        
        Args:
            blob: Binary data previously serialized with serialize_embedding
            
        Returns:
            List of float values
        """
        count = len(blob) // 4  # 4 bytes per float
        return list(struct.unpack(f'{count}f', blob))
    
    @abstractmethod
    def _insert_batch(self, batch_data: List[KBDocument], embeddings: Dict[str, List[List[float]]]) -> List[int]:
        """
        Insert a batch of KBDocument objects with embeddings into the database.
        
        Args:
            batch_data: List of KBDocument objects to insert
            embeddings: Dictionary mapping field names to lists of embeddings
            
        Returns:
            List of primary key IDs for the inserted records
        """
        pass
    
    def process_source(self, source_path: str) -> List[int]:
        """
        Process a single source file and insert its records into the database.
        
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
                    embeddings[field_name] = self.generate_embeddings(texts, text_type="passage")
                
                # Insert batch of KBDocument objects with embeddings
                record_ids = self._insert_batch(batch, embeddings)
                all_record_ids.extend(record_ids)
                
            return all_record_ids
            
        except Exception as e:
            self.logger.error(f"Error processing source {source_path}: {str(e)}")
            self.logger.debug("Full exception details:", exc_info=True)
            return []
    
    def process_directory(self, directory_path: str, file_extension: str = '.json') -> Dict[str, List[int]]:
        """
        Process all files with the specified extension in a directory.
        
        Args:
            directory_path: Path to the directory containing files to process
            file_extension: File extension to filter for (default: '.json')
            
        Returns:
            Dictionary mapping file paths to lists of inserted record IDs
        """
        self.logger.debug(f"Processing directory: {directory_path}")
        results = {}
        
        # Get all matching files in the directory
        files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.endswith(file_extension)
        ]
        self.logger.debug(f"Found {len(files)} {file_extension} files in {directory_path}")

        # Process each file
        for file_path in tqdm(files, desc="Processing files"):
            record_ids = self.process_source(file_path)
            results[file_path] = record_ids
            self.logger.debug(f"Completed processing file {os.path.basename(file_path)} "
                             f"with {len(record_ids)} records")
        
        return results
    
    def run(self, directory_path: Optional[str] = None, file_path: Optional[str] = None, 
            file_extension: str = '.json') -> Dict[str, List[int]]:
        """
        Run the processor on either a directory or a single file.
        
        Args:
            directory_path: Path to directory containing files to process (if processing a directory)
            file_path: Path to a single file to process (if processing a single file)
            file_extension: File extension to filter for when processing a directory
            
        Returns:
            Dictionary mapping file paths to lists of inserted record IDs
        """
        if file_path:
            # Process a single file
            file_path = os.path.abspath(file_path)
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return {}
            
            record_ids = self.process_source(file_path)
            self.logger.debug(f"Completed processing file {os.path.basename(file_path)}, "
                             f"inserted {len(record_ids)} records")
            return {file_path: record_ids}
        
        elif directory_path:
            # Process the entire directory
            dir_path = os.path.abspath(directory_path)
            if not os.path.exists(dir_path):
                self.logger.error(f"Directory not found: {dir_path}")
                return {}
            
            results = self.process_directory(dir_path, file_extension)
            total_records = sum(len(ids) for ids in results.values())
            self.logger.debug(f"Completed processing {len(results)} files, "
                             f"inserted {total_records} records")
            return results
        
        else:
            self.logger.error("Neither directory_path nor file_path provided")
            return {} 