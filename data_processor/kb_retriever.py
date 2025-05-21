import logging
import struct
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class KBRetrieverBase(ABC):
    """
    Abstract base class for knowledge base retrievers.
    
    This class provides a common structure and utilities for retrieving documents
    from a knowledge base using vector search or other retrieval methods.
    
    Concrete subclasses must implement the abstract methods to handle dataset-specific
    retrieval operations.
    """
    
    def __init__(self, 
                 conn_string: str,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 max_seq_length: int = 4096,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the base knowledge base retriever.
        
        Args:
            conn_string: Database connection string
            model_name: Name of the embedding model to use
            max_seq_length: Maximum sequence length for the embedding model
            logger: Optional logger instance; if not provided, a new one will be created
        """
        # Set up logger
        self.logger = logger or self._setup_logger()
        self.logger.debug(f"Initializing {self.__class__.__name__}")
        
        # Store configuration
        self.conn_string = conn_string
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
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
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            query: The query text to embed
            
        Returns:
            List of float values representing the query embedding
        """
        self.logger.debug(f"Generating embedding for query: {query[:50]}...")
        embedding = self.model.encode(query, task="retrieval.query", convert_to_tensor=False)
        return embedding.tolist()
    
    def serialize_embedding(self, embedding: List[float]) -> bytes:
        """
        Serialize a list of floats into a binary blob for database storage.
        
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
    def get_db_connection(self):
        """
        Get a database connection using the connection string.
        
        Returns:
            A database connection object
        """
        pass
    
    @abstractmethod
    def retrieve_documents(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the knowledge base matching the query.
        
        Args:
            query: Query text to match against documents
            limit: Maximum number of documents to return
            **kwargs: Additional retrieval parameters specific to the implementation
            
        Returns:
            List of document dictionaries with retrieval scores and metadata
        """
        pass
