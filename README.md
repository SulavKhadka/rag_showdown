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

## Setup and Configuration

This section covers essential setup steps required after installation, primarily focusing on environment variables that control application behavior and integrations.

### API Key Configuration

For LLM-based features such as query decomposition, LLM-based relevance filtering, and the agent module, the application requires an API key from [OpenRouter.ai](https://openrouter.ai/).

This key should be set as an environment variable named `OPENROUTER_API_KEY`. **Do not hardcode your API key directly into the codebase.**

**Setting the Environment Variable:**

*   **Linux/macOS:**
    Open your terminal and run:
    ```bash
    export OPENROUTER_API_KEY='your_actual_openrouter_api_key'
    ```
    To make this change permanent, add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).

*   **Windows (Command Prompt):**
    ```cmd
    set OPENROUTER_API_KEY=your_actual_openrouter_api_key
    ```
    To set it permanently, you can use the System Properties:
    1. Search for "environment variables" in the Start Menu.
    2. Click on "Edit the system environment variables".
    3. Click the "Environment Variables..." button.
    4. In the "User variables" or "System variables" section, click "New..."
    5. Set Variable name to `OPENROUTER_API_KEY` and Variable value to your key.
    6. Click OK on all dialogs. You may need to restart your command prompt or IDE for the changes to take effect.

Ensure the `OPENROUTER_API_KEY` is correctly set in your environment before running the application to use features relying on the LLM.

### Logging Configuration

For user privacy and to reduce disk usage, detailed logging of query content, full LLM responses, intermediate pipeline steps, and system information to individual JSON files in the `logs/queries/` directory is **disabled by default**. These detailed logs can be useful for debugging and performance analysis.

Standard application logs (e.g., server start, basic query processing info, errors) are still written to the console and/or the main log file (`logs/rag_app.log`) as configured.

To enable detailed query logging, set the following environment variable to `true`:

```bash
export ENABLE_DETAILED_QUERY_LOGS=true
```

On Windows, you can use:
```cmd
set ENABLE_DETAILED_QUERY_LOGS=true
```
Or set it permanently via System Properties as described in the "API Key Configuration" section.

When enabled, detailed JSON logs for each query (or each step like retrieval/generation if logged separately) will be saved in the `logs/queries/` directory.

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

## Pipeline Configuration Options

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

## Security

### Dependency Vulnerability Scanning

Project dependencies can sometimes have known security vulnerabilities. It's crucial to regularly check your installed packages against vulnerability databases to ensure your application remains secure. This section provides instructions for using common Python tools to perform these checks.

It is recommended to run these checks periodically, especially before deploying the application or after updating dependencies. If vulnerabilities are found, you should look for updated versions of the affected packages. If a patched version is not immediately available, consider whether the vulnerability impacts your specific usage and explore potential mitigations or alternative packages.

#### Using pip-audit (Recommended)

`pip-audit` is a tool from the Python Packaging Authority (PyPA) that audits your project's dependencies for known vulnerabilities by checking against the Python Packaging Advisory Database (PyPI Advisory Database).

1.  **Install `pip-audit`**:
    If you don't have it installed, you can install it using pip:
    ```bash
    pip install pip-audit
    ```

2.  **Run the audit**:
    `pip-audit` automatically checks dependencies listed in `pyproject.toml` (if present and supported for dependency listing) or a `requirements.txt` file. Navigate to your project's root directory and run:
    ```bash
    pip-audit
    ```
    If your project uses a virtual environment, ensure it is activated so `pip-audit` checks the correct package versions.

#### Using safety

`safety` is another popular tool for checking Python dependencies against a known vulnerability database. It typically requires a `requirements.txt` file.

1.  **Generate `requirements.txt` (if needed)**:
    If your project uses `pyproject.toml` with a build system like Poetry or PDM, you might need to generate a `requirements.txt` file.

    *   For Poetry:
        ```bash
        poetry export -f requirements.txt --output requirements.txt --without-hashes
        ```
    *   For PDM:
        ```bash
        pdm export -f requirements.txt --output requirements.txt --without-hashes
        ```
    *   If you are managing dependencies directly in `pyproject.toml` without a specific build tool's lock file, `pip-audit` might be more straightforward.

2.  **Install `safety`**:
    ```bash
    pip install safety
    ```

3.  **Run the check**:
    Once you have a `requirements.txt` file, run:
    ```bash
    safety check -r requirements.txt
    ```
    `safety` also offers options to integrate with CI systems and use API keys for more up-to-date vulnerability data from commercial sources, though the basic check against the open database is free.

Regularly performing these checks helps maintain the security posture of your application.

### Implemented Security Measures

This application incorporates several security best practices:

*   **API Key Management**: The `OPENROUTER_API_KEY` is handled via environment variables, preventing it from being hardcoded in the source.
*   **Application-Level Rate Limiting**: The API (`/api/query` endpoint) implements rate limiting (e.g., 5 requests per minute) to protect against abuse, using `slowapi`.
*   **Input Validation**:
    *   API request bodies are validated using Pydantic models (e.g., `QueryRequest`, `PipelineConfigModel` in `app.py`).
    *   Specific fields like `top_k` and `min_similarity_pct` have range constraints (e.g., `top_k` between 1-20).
*   **XSS Protection (Frontend)**:
    *   User-generated content and document metadata displayed in the frontend UI (e.g., document titles, abstracts, authors) are primarily rendered using `textContent` to prevent HTML injection from data sources.
    *   Markdown content from the LLM is sanitized using `marked.js` with the `sanitize: true` option before being rendered as HTML.
*   **Conditional Detailed Logging**: Detailed query logs, which may contain sensitive information, are disabled by default and can only be enabled via the `ENABLE_DETAILED_QUERY_LOGS` environment variable.
*   **Generic Error Messages**: API endpoints are configured to return generic error messages for server-side exceptions, avoiding leakage of potentially sensitive internal details (e.g., stack traces or specific error strings) to the client. Detailed errors are logged internally.
*   **Secure Endpoint Design**: The `/api/db_status` endpoint was refactored to only check the status of the predefined `DEFAULT_DB_PATH`, preventing its use for arbitrary file path checks.

### Deployment Environment Security

While the application implements the security measures listed above, the security of the overall deployment environment is the user's responsibility. If deploying this application, consider the following:

*   **Nginx (or other reverse proxy) Configuration**:
    *   Implement appropriate rate limiting and connection limits.
    *   Configure security headers (e.g., `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`).
    *   Ensure SSL/TLS is properly configured for HTTPS.
    *   Regularly update Nginx to patch vulnerabilities.
*   **Cloudflare (or similar CDN/WAF service) Utilization**:
    *   Leverage Web Application Firewall (WAF) capabilities to filter malicious traffic.
    *   Utilize DDoS protection features.
    *   Consider using Bot Management features to block unwanted automated traffic.
*   **Operating System Security**: Ensure the underlying server is hardened and regularly patched.
*   **Database Security**: If using a more complex database setup than the default SQLite, ensure it is properly secured.

Security is an ongoing process. Regularly review your security configurations, stay updated on best practices, and monitor your application for suspicious activity.

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
└── abstracts.db            # SQLite database with embedded vectors
```

## License and Credits

This project is licensed under the MIT License - see the LICENSE file for details.

Created for the purpose of evaluating and comparing different RAG approaches.