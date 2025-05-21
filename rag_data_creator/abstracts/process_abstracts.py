#!/usr/bin/env python3
"""
Script to demonstrate the usage of SQLiteKBProcessor for processing scientific abstracts.
This script will ingest abstracts from the dataset file into an SQLite database.
"""

import os
import logging
import argparse
from abstracts_processor import SQLiteKBProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('abstract_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Process scientific abstracts dataset.')
    parser.add_argument('--input', '-i', type=str, default='/home/bobby/Repos/rag_showdown/rag_data_creator/abstracts/abstracts_dataset_15857.json',
                        help='Path to the input JSON file with abstracts')
    parser.add_argument('--db', '-d', type=str, default='abstracts.db',
                        help='Path to the SQLite database file')
    parser.add_argument('--model', '-m', type=str, default='jinaai/jina-embeddings-v3',
                        help='Embedding model to use')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Embedding dimension')
    parser.add_argument('--batch', '-b', type=int, default=50,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Verify input file exists
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Create database directory if needed
    db_path = os.path.abspath(args.db)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize the processor
    logger.info(f"Initializing processor with model {args.model}")
    processor = SQLiteKBProcessor(
        db_path=db_path,
        embedding_dim=args.dim,
        model_name=args.model,
        batch_size=args.batch,
        logger=logger
    )
    
    # Process the abstracts file
    logger.info(f"Starting to process abstracts from {input_path}")
    stats = processor.process_abstracts_file(input_path)
    
    # Log summary
    logger.info(f"Processing completed: {stats['abstracts_processed']} abstracts in {stats['processing_time_seconds']:.2f} seconds")
    logger.info(f"Average time per abstract: {stats['average_time_per_abstract']:.4f} seconds")
    
    # Example query
    example_query = "quantum computing algorithms"
    logger.info(f"Running example query: '{example_query}'")
    results = processor.search_similar_documents(example_query, limit=3)
    
    # Display sample results
    logger.info(f"Found {len(results)} matching abstracts")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (similarity: {result['similarity']:.4f}):")
        logger.info(f"Title: {result['title']}")
        logger.info(f"Authors: {', '.join(result['authors'])}")
        logger.info(f"Link: {result['link']}")
        logger.info(f"Abstract: {result['abstract'][:150]}...")
        logger.info("---")
    
    return 0

if __name__ == "__main__":
    exit(main())
