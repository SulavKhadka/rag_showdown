from pipelines.configurable_rag import ConfigurableRAGRetriever
from pipelines.config import get_config_by_preset


def main():
    """
    Main function to run the configurable RAG retriever from the command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the configurable RAG retriever")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database")
    parser.add_argument("--preset", type=str, default="vector_only", 
                      help="Pipeline preset (vector_only, vector_plus_rerank, vector_plus_bm25, vector_bm25_rerank, vector_bm25_rerank_llm, full_hybrid)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for models (cpu or cuda)")
    parser.add_argument("--query", type=str, help="Query to run (if not provided, interactive mode is used)")
    
    # Add flags for individual pipeline components (override preset)
    parser.add_argument("--use_vector", action="store_true", help="Use vector search")
    parser.add_argument("--use_bm25", action="store_true", help="Use BM25 search")
    parser.add_argument("--use_reranker", action="store_true", help="Use cross-encoder reranker")
    parser.add_argument("--use_llm_reranker", action="store_true", help="Use LLM reranker")
    parser.add_argument("--use_query_decomposition", action="store_true", help="Use query decomposition")
    
    args = parser.parse_args()
    
    # Get config from preset
    config = get_config_by_preset(args.preset)
    
    # Override config with CLI arguments
    config.top_k = args.top_k
    config.device = args.device
    
    # Override individual pipeline components if flags are set
    if args.use_vector:
        config.use_vector = True
    
    if args.use_bm25:
        config.use_bm25 = True
    
    if args.use_reranker:
        config.use_reranker = True
    
    if args.use_llm_reranker:
        config.use_llm_reranker = True
    
    if args.use_query_decomposition:
        config.use_query_decomposition = True
    
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