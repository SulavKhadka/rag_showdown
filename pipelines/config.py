from dataclasses import dataclass
from typing import Optional
from secret_keys import OPENROUTER_API_KEY

@dataclass
class PipelineConfig:
    """
    Configuration for the RAG pipeline components.
    
    This class controls which features are enabled in the pipeline
    and their configuration parameters.
    """
    # Retrieval options
    use_vector: bool = True                # cosine similarity search (always True in current variants)
    use_bm25: bool = False                 # BM25 keyword search
    use_query_decomposition: bool = False  # LLM-based query decomposition
    
    # Reranking options
    use_reranker: bool = False             # Cross-encoder reranker
    use_llm_reranker: bool = False         # LLM-based relevance filtering
    
    # Knobs shared by several modules
    top_k: int = 5                         # Number of documents to retrieve
    min_similarity_pct: float = 50.0       # Vector search filter threshold
    
    # Model configuration
    embedding_model: str = "jinaai/jina-embeddings-v3"
    reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    reranker_model_type: str = "cross-encoder"
    device: str = "cpu"                    # "cuda" or "cpu"
    
    # LLM API configuration
    llm_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_api_key: Optional[str] = OPENROUTER_API_KEY
    llm_model: str = "mistralai/mistral-medium-3"


# Factory functions for common pipeline configurations
def vector_only() -> PipelineConfig:
    """Simple vector retrieval only."""
    return PipelineConfig()


def vector_plus_rerank() -> PipelineConfig:
    """Vector retrieval with cross-encoder reranking."""
    return PipelineConfig(use_reranker=True)


def vector_plus_bm25() -> PipelineConfig:
    """Combined vector and BM25 retrieval."""
    return PipelineConfig(use_bm25=True)


def vector_bm25_rerank() -> PipelineConfig:
    """Combined vector and BM25 with cross-encoder reranking."""
    return PipelineConfig(use_bm25=True, use_reranker=True)


def vector_bm25_rerank_llm() -> PipelineConfig:
    """Combined retrieval with reranking and LLM filtering."""
    return PipelineConfig(use_bm25=True, use_reranker=True, use_llm_reranker=True)


def full_hybrid() -> PipelineConfig:
    """Full hybrid pipeline with all features enabled."""
    return PipelineConfig(
        use_bm25=True,
        use_reranker=True,
        use_llm_reranker=True,
        use_query_decomposition=True
    )


# Factory function to get a configuration by name
def get_config_by_preset(preset: str) -> PipelineConfig:
    """
    Get a pipeline configuration by preset name.
    
    Args:
        preset: Name of the preset configuration
        
    Returns:
        The corresponding PipelineConfig
        
    Raises:
        ValueError: If preset name is unknown
    """
    presets = {
        "vector_only": vector_only,
        "vector_plus_rerank": vector_plus_rerank,
        "vector_plus_bm25": vector_plus_bm25,
        "vector_bm25_rerank": vector_bm25_rerank,
        "vector_bm25_rerank_llm": vector_bm25_rerank_llm,
        "full_hybrid": full_hybrid
    }
    
    if preset.lower() not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {', '.join(presets.keys())}")
    
    return presets[preset.lower()]()