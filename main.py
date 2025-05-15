from pipelines.hybrd_rag import HybridRAGRetriever

def main():
    """
    Main function to run the hybrid RAG retriever from the command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the hybrid RAG retriever")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database")
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
