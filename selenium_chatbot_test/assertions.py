"""
Semantic assertion utilities for non-deterministic AI responses.

This module provides the SemanticAssert class which uses sentence-transformers
to perform semantic similarity comparisons instead of exact string matching.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class _ModelLoader:
    """
    Lazy loader / Singleton for sentence-transformer models.
    
    This class ensures that heavy ML models are only loaded when first needed,
    not at module import time. It also handles GPU availability gracefully.
    """
    
    _instance: Optional["_ModelLoader"] = None
    _models: dict = {}
    
    def __new__(cls) -> "_ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance
    
    def get_model(self, model_name: str):
        """
        Get or load a sentence-transformer model.
        
        Args:
            model_name: The name of the model to load (e.g., "all-MiniLM-L6-v2").
        
        Returns:
            SentenceTransformer: The loaded model instance.
        """
        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)
        return self._models[model_name]
    
    def _load_model(self, model_name: str):
        """
        Load a sentence-transformer model with proper error handling.
        
        Handles:
        - Missing CUDA/GPU (falls back to CPU silently)
        - Model not cached (logs warning about download)
        """
        # Import here to avoid loading at module import time
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for semantic assertions. "
                "Install it with: pip install sentence-transformers"
            ) from e
        
        # Check if model is likely cached
        cache_dir = os.path.join(
            os.path.expanduser("~"), 
            ".cache", 
            "torch", 
            "sentence_transformers"
        )
        model_cache_path = os.path.join(cache_dir, model_name.replace("/", "_"))
        
        if not os.path.exists(model_cache_path):
            logger.warning(
                f"Model '{model_name}' not found in cache. "
                f"First run will download the model (~90MB for all-MiniLM-L6-v2). "
                f"This may take a moment..."
            )
        
        # Force CPU if CUDA is not available to avoid warnings/errors
        # sentence-transformers handles this, but we ensure silent fallback
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.debug("CUDA available, using GPU for embeddings")
            else:
                logger.debug("CUDA not available, using CPU for embeddings")
        except ImportError:
            logger.debug("PyTorch CUDA check failed, defaulting to CPU")
        
        try:
            model = SentenceTransformer(model_name, device=device)
            logger.info(f"Loaded model '{model_name}' on {device}")
            return model
        except Exception as e:
            # If GPU loading fails, try CPU as fallback
            if device != "cpu":
                logger.warning(
                    f"Failed to load model on {device}, falling back to CPU: {e}"
                )
                model = SentenceTransformer(model_name, device="cpu")
                logger.info(f"Loaded model '{model_name}' on CPU (fallback)")
                return model
            raise


class SemanticAssert:
    """
    Semantic similarity assertions for non-deterministic AI responses.
    
    Standard string assertions fail on AI-generated responses because the exact
    wording varies between runs. SemanticAssert uses sentence embeddings and
    cosine similarity to verify that responses are semantically equivalent,
    even when the exact words differ.
    
    Example:
        >>> from selenium_chatbot_test import SemanticAssert
        >>> 
        >>> asserter = SemanticAssert()
        >>> # These will pass despite different wording
        >>> asserter.assert_similarity(
        ...     "Hello! How can I help you today?",
        ...     "Hi there, what can I assist you with?",
        ...     min_score=0.7
        ... )
    
    Note:
        The sentence-transformer model is loaded lazily on first call to
        assert_similarity(), not at import time. The first call may be slower
        if the model needs to be downloaded.
    """
    
    def __init__(self) -> None:
        """Initialize SemanticAssert."""
        self._model_loader = _ModelLoader()
    
    def assert_similarity(
        self,
        actual: str,
        expected: str,
        min_score: float = 0.8,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Assert that two texts are semantically similar.
        
        Converts both texts to embeddings using a sentence-transformer model
        and calculates their cosine similarity. If the similarity score is
        below the minimum threshold, raises an AssertionError.
        
        Args:
            actual: The actual text (e.g., from the AI response).
            expected: The expected text to compare against.
            min_score: Minimum cosine similarity score (0.0 to 1.0).
                      Default is 0.8 (80% similar).
            model_name: The sentence-transformer model to use.
                       Default is "all-MiniLM-L6-v2" (fast and accurate).
        
        Raises:
            AssertionError: If the similarity score is below min_score.
            ImportError: If sentence-transformers is not installed.
        
        Example:
            >>> asserter = SemanticAssert()
            >>> asserter.assert_similarity(
            ...     actual="The weather is nice today",
            ...     expected="It's a beautiful day outside",
            ...     min_score=0.6
            ... )
        """
        if not isinstance(actual, str) or not isinstance(expected, str):
            raise TypeError(
                f"Both actual and expected must be strings, got "
                f"actual={type(actual).__name__}, expected={type(expected).__name__}"
            )
        
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                f"min_score must be between 0.0 and 1.0, got {min_score}"
            )
        
        model = self._model_loader.get_model(model_name)
        
        # Generate embeddings
        embeddings = model.encode([actual, expected], convert_to_numpy=True)
        actual_embedding = embeddings[0]
        expected_embedding = embeddings[1]
        
        # Calculate cosine similarity
        score = self._cosine_similarity(actual_embedding, expected_embedding)
        
        logger.debug(
            f"Semantic similarity score: {score:.4f} (min required: {min_score})"
        )
        
        if score < min_score:
            # Truncate texts for error message
            actual_preview = actual[:100] + "..." if len(actual) > 100 else actual
            expected_preview = expected[:100] + "..." if len(expected) > 100 else expected
            
            raise AssertionError(
                f"Semantic similarity assertion failed.\n"
                f"Score: {score:.4f} (required: >= {min_score})\n"
                f"Actual: {actual_preview!r}\n"
                f"Expected: {expected_preview!r}"
            )
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.
        
        Returns:
            float: Cosine similarity score between -1.0 and 1.0.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_similarity_score(
        self,
        text1: str,
        text2: str,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> float:
        """
        Get the semantic similarity score between two texts without asserting.
        
        Useful for debugging or when you want to examine the score before
        deciding on a threshold.
        
        Args:
            text1: First text to compare.
            text2: Second text to compare.
            model_name: The sentence-transformer model to use.
        
        Returns:
            float: Cosine similarity score between -1.0 and 1.0.
        
        Example:
            >>> asserter = SemanticAssert()
            >>> score = asserter.get_similarity_score(
            ...     "Hello, how are you?",
            ...     "Hi, how's it going?"
            ... )
            >>> print(f"Similarity: {score:.2%}")
        """
        model = self._model_loader.get_model(model_name)
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        return self._cosine_similarity(embeddings[0], embeddings[1])
