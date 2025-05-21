# RAG Showdown

A project for testing and comparing different RAG (Retrieval-Augmented Generation) approaches with an interactive UI for exploration.

## Project Overview

RAG Showdown provides tools for evaluating different retrieval-augmented generation strategies. The project includes a configurable, unified RAG pipeline that can reproduce multiple retrieval and generation strategies by toggling components on/off. It features both a web-based UI for interactive exploration and a command-line interface for scripted evaluation.

The system is designed to work with a SQLite database containing abstracts, using vector similarity search, keyword search, and various reranking strategies to retrieve the most relevant documents for a given query.

## Features

### Configurable RAG Pipeline

- **Modular Design**: Components can be enabled/disabled through configuration
- **Retrieval Options**:
  - Vector similarity search (using [jina-embeddings-v3](https://jina.ai/embeddings/) by default)
  - BM25 keyword search
  - Combined vector and BM25
- **Enhancement Features**:
  - Query decomposition using LLM
  - Cross-encoder reranking
  - LLM-based relevance filtering
- **Preset Configurations**:
  - `vector_only`: Simple vector retrieval
  - `vector_plus_rerank`: Vector retrieval with reranking
  - `vector_plus_bm25`: Combined vector and BM25 retrieval
  - `vector_bm25_rerank`: Combined retrieval with reranking
  - `vector_bm25_rerank_llm`: Adds LLM-based filtering
  - `full_hybrid`: All features enabled

### Web-based UI

- **Interactive Interface**: Explore and compare different RAG configurations
- **Real-time Configuration**: Toggle features on/off and adjust parameters
- **Query History**: Save and revisit previous queries and results
- **Results Visualization**: View source documents with relevance scores
- **Markdown Rendering**: Nicely formatted answers from the LLM

### Command-line Interface

- Run queries with different configurations
- Interactive or single-query modes
- Override preset configurations with custom settings

### Agent Module

- A simple agent implementation with tool-calling capabilities
- Example tools like `get_weather` and `get_news` (mock implementations)
- Integration with OpenRouter API through OpenAI client library

## System Architecture

### Backend

- **FastAPI** server for the web interface
- **SQLite** database with **sqlite-vec** extension for vector search
- **Sentence Transformers** for embedding generation
- Integration with reranking libraries and LLM APIs
- Configurable pipeline components

### Frontend

- Modern HTML/CSS/JS interface
- Interactive configuration controls
- Real-time query processing
- Results visualization
- Responsive design

### Data Processing

- Processing and embedding of knowledge base content
- Support for different document types and fields
- Batch processing for efficient database population

### Knowledge Base Structure

- SQLite database with abstracts table and metadata
- Vector search indexes for efficient similarity search
- Support for document relationships and metadata

## Installation

### Requirements

- Python 3.9+
- SQLite with the sqlite-vec extension
- Dependencies listed in `pyproject.toml`

```bash
# Install in development mode
pip install -e .
```

### Database Preparation

Before running the system, you need to prepare a database with document abstracts. The repository includes an abstracts processor in the `rag_data_creator/abstracts` directory.

```bash
# Process abstracts JSON file into SQLite database
python -m rag_data_creator.abstracts.process_abstracts \
    --input rag_data_creator/abstracts/abstracts_dataset_15857.json \
    --db abstracts.db \
    --model jinaai/jina-embeddings-v3 \
    --dim 1024 \
    --batch 50
```

### API Key Setup

For LLM-based features like query decomposition, relevance filtering, and the agent module, you'll need to set up an OpenRouter API key:

1. Create a `secret_keys.py` file in the project root
2. Add your OpenRouter API key:
   ```python
   OPENROUTER_API_KEY = "your_api_key_here"
   ```

## Usage

### Web UI

1. Start the web server:
   ```bash
   python -m app
   ```

2. Open a browser and navigate to `http://localhost:5000`

3. Configure your RAG pipeline:
   - Select retrieval methods (Vector Search and/or BM25)
   - Enable/disable reranking options
   - Set parameters like Top K and similarity threshold

4. Enter a query and click Submit

5. View the generated answer and source documents

### Command-line Interface

```bash
# Basic usage with preset
python -m pipelines.configurable_rag \
  --db_path abstracts.db \
  --preset full_hybrid \
  --top_k 8 \
  --query "What are recent advances in protein folding?"

# Override preset with specific flags
python -m pipelines.configurable_rag \
  --db_path abstracts.db \
  --preset vector_only \
  --use_bm25 True \
  --use_reranker True
```

### Python API

```python
from pipelines.configurable_rag import ConfigurableRAGRetriever
from pipelines.config import PipelineConfig, get_config_by_preset

# Use a preset configuration
config = get_config_by_preset("vector_bm25_rerank")

# Or create a custom configuration
config = PipelineConfig(
    use_vector=True,
    use_bm25=True,
    use_reranker=True,
    use_query_decomposition=False,
    top_k=8,
    min_similarity_pct=60.0
)

# Initialize the retriever
retriever = ConfigurableRAGRetriever(
    db_path="abstracts.db",
    config=config
)

# Process a query
result = retriever.rag_query("What are the most effective treatments for Alzheimer's disease?")

# Access the answer and retrieved documents
answer = result["answer"]
documents = result["retrieved_documents"]
```

### Agent Module

```python
from agent import Agent, get_weather, get_news

# Create an agent instance
agent = Agent(
    name="my_agent",
    model_name="Qwen/Qwen3-4B",  # Or any other model
    system_prompt="You are a helpful assistant who has access to tools.",
    tools=[get_weather, get_news]
)

# Run the agent in interactive mode
agent.run()
```

## Configuration Options

### Pipeline Presets

The following presets are available through `get_config_by_preset()`:

| Preset | Description |
|--------|-------------|
| `vector_only` | Simple vector retrieval only |
| `vector_plus_rerank` | Vector retrieval with cross-encoder reranking |
| `vector_plus_bm25` | Combined vector and BM25 retrieval |
| `vector_bm25_rerank` | Combined vector and BM25 with cross-encoder reranking |
| `vector_bm25_rerank_llm` | Combined retrieval with reranking and LLM filtering |
| `full_hybrid` | All features enabled (including query decomposition) |

### Custom Configuration

You can create a custom configuration by setting individual parameters:

```python
from pipelines.config import PipelineConfig

config = PipelineConfig(
    # Retrieval options
    use_vector=True,
    use_bm25=True,
    
    # Enhancement features
    use_reranker=True,
    use_llm_reranker=False,
    use_query_decomposition=False,
    
    # Parameters
    top_k=5,
    min_similarity_pct=50.0,
    
    # Model configuration
    embedding_model="jinaai/jina-embeddings-v3",
    reranker_model="Alibaba-NLP/gte-reranker-modernbert-base",
    device="cpu",  # "cuda" for GPU
    
    # LLM configuration
    llm_model="mistralai/mistral-medium-3"
)
```

## Development

### Project Structure

```
rag_showdown/
├── agent/                  # Agent implementation
├── data_processor/         # Data processing utilities
├── pipelines/              # RAG pipeline implementations
│   ├── config.py           # Pipeline configuration
│   └── configurable_rag.py # Unified RAG pipeline with CLI functionality
├── rag_data_creator/       # Data preparation tools
├── static/                 # Web UI assets
├── app.py                  # FastAPI web application
├── secret_keys.py          # API keys
└── abstracts.db            # SQLite database with embedded vectors
```

## License and Credits

This project is licensed under the MIT License - see the LICENSE file for details.

Created for the purpose of evaluating and comparing different RAG approaches.