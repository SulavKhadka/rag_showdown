import os
import sys
import json
import logging
import requests
import sqlite3
import sqlite_vec
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Get the absolute path to the project root directory and add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base retriever class
from data_processor.kb_retriever import KBRetrieverBase

# Load environment variables
load_dotenv()

class CosineSimilarityRAGRetriever(KBRetrieverBase):
    """
    A RAG retriever that uses cosine similarity to find relevant documents
    and integrates with an LLM to generate answers based on the retrieved context.
    """
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "jinaai/jina-embeddings-v3",
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_seq_length: int = 4096,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RAG retriever with both retrieval and LLM capabilities.
        
        Args:
            db_path: Path to the SQLite database with abstracts
            model_name: Name of the embedding model to use
            llm_api_url: URL for the LLM API (defaults to OpenRouter)
            llm_api_key: API key for the LLM
            max_seq_length: Maximum sequence length for the embedding model
            top_k: Number of documents to retrieve
            logger: Optional logger instance
        """
        # Initialize the base class (retrieval capabilities)
        super().__init__(db_path, model_name, max_seq_length, logger)
        self.top_k = top_k
        
        # Set up LLM configuration
        self.llm_api_url = llm_api_url or "https://openrouter.ai/api/v1/chat/completions"
        self.llm_api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.llm_api_key:
            self.logger.warning("No LLM API key provided. Generation will not work.")
    
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
    
    def retrieve_documents(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve abstracts similar to the query text.
        
        Args:
            query: The text to find similar abstracts for
            limit: Maximum number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of abstracts with similarity scores
        """
        self.logger.info(f"Searching for documents similar to: {query[:50]}...")
        
        # Use the specified limit or fall back to the default top_k
        actual_limit = limit if limit is not None else self.top_k
        
        try:
            # Generate embedding for the query
            query_embedding = self.generate_query_embedding(query)
            
            # Serialize the embedding
            query_blob = self.serialize_embedding(query_embedding)
            
            # Query the database for similar abstracts
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            results = cursor.execute('''
            SELECT 
                a.id, a.title, a.abstract, a.authors, a.link, a.published, a.source_file, a.content,
                v.distance
            FROM 
                vss_abstracts v
            JOIN 
                abstracts a ON v.rowid = a.id
            WHERE 
                v.embedding MATCH ? AND k = ?
            ORDER BY 
                v.distance
            ''', (query_blob, actual_limit)).fetchall()
            
            # Process results
            documents = []
            for row in results:
                doc_id, title, abstract, authors_json, link, published, source_file, content, distance = row
                # Parse authors JSON
                try:
                    authors = json.loads(authors_json)
                except json.JSONDecodeError:
                    authors = []
                
                # Create the document result
                document_result = {
                    'id': doc_id,
                    'abstract': abstract,
                    'published': published,
                    'source_file': source_file,
                    'content': content,
                    'similarity': 1.0 - min(1.0, max(0.0, distance))  # Convert distance to similarity score (0-1)
                }
                
                # Try to extract additional metadata from the content field
                try:
                    # Get context data from the database
                    context_query = "SELECT context FROM abstracts WHERE id = ?"
                    context_result = cursor.execute(context_query, (doc_id,)).fetchone()
                    
                    if context_result and context_result[0]:
                        context_data = json.loads(context_result[0])
                        document_result['title'] = context_data.get('title', title)
                        document_result['authors'] = context_data.get('authors', authors)
                        document_result['link'] = context_data.get('link', link)
                    else:
                        # Fallback to database values
                        document_result['title'] = title
                        document_result['authors'] = authors
                        document_result['link'] = link
                except (json.JSONDecodeError, Exception):
                    # Fallback to database values if context parsing fails
                    document_result['title'] = title
                    document_result['authors'] = authors
                    document_result['link'] = link
                    
                documents.append(document_result)
            
            conn.close()
            self.logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
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
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, limit=self.top_k)
        
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
    Main function to run the RAG pipeline from the command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the cosine similarity RAG retriever")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v3", help="Name of the embedding model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    args = parser.parse_args()
    
    # Initialize the RAG retriever
    retriever = CosineSimilarityRAGRetriever(
        db_path=args.db_path,
        model_name=args.model_name,
        top_k=args.top_k
    )
    
    # Interactive query loop
    print("\nCosine Similarity RAG Retriever\n")
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
                print(f"Abstract: {doc.get('abstract', doc.get('content', ''))[:200]}...")
                print(f"Authors: {', '.join(doc.get('authors', [])) if isinstance(doc.get('authors', []), list) else str(doc.get('authors', []))}")
                print(f"Similarity: {doc.get('similarity', 0):.4f}")
                print()


if __name__ == "__main__":
    main()
