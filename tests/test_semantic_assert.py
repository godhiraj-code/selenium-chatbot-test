"""
Tests for SemanticAssert class.

Tests the semantic similarity assertion functionality.
Note: Some tests require the sentence-transformers model and may be slow on first run.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from selenium_chatbot_test.assertions import SemanticAssert, _ModelLoader


class TestModelLoader:
    """Tests for the _ModelLoader singleton."""
    
    def test_singleton_pattern(self):
        """Test that _ModelLoader returns the same instance."""
        loader1 = _ModelLoader()
        loader2 = _ModelLoader()
        
        assert loader1 is loader2
    
    def test_model_cached_after_load(self):
        """Test that models are cached after first load."""
        loader = _ModelLoader()
        
        # Clear any existing cache for this test
        loader._models = {}
        
        # Need to patch at the import location
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            
            # Load model twice
            model1 = loader.get_model("test-model")
            model2 = loader.get_model("test-model")
            
            # Should only instantiate once
            assert mock_st.call_count == 1
            assert model1 is model2


class TestSemanticAssertInit:
    """Tests for SemanticAssert initialization."""
    
    def test_create_instance(self):
        """Test that SemanticAssert can be instantiated."""
        asserter = SemanticAssert()
        
        assert asserter is not None
    
    def test_model_not_loaded_on_init(self):
        """Test that model is not loaded at initialization (lazy loading)."""
        with patch.object(_ModelLoader, 'get_model') as mock_get:
            asserter = SemanticAssert()
            
            # Model should not be loaded yet
            mock_get.assert_not_called()


class TestSemanticAssertValidation:
    """Tests for input validation in SemanticAssert."""
    
    def test_non_string_actual_raises_error(self):
        """Test that non-string actual raises TypeError."""
        asserter = SemanticAssert()
        
        with pytest.raises(TypeError, match="Both actual and expected must be strings"):
            asserter.assert_similarity(123, "expected text")
    
    def test_non_string_expected_raises_error(self):
        """Test that non-string expected raises TypeError."""
        asserter = SemanticAssert()
        
        with pytest.raises(TypeError, match="Both actual and expected must be strings"):
            asserter.assert_similarity("actual text", 456)
    
    def test_none_values_raise_error(self):
        """Test that None values raise TypeError."""
        asserter = SemanticAssert()
        
        with pytest.raises(TypeError):
            asserter.assert_similarity(None, "expected")
        
        with pytest.raises(TypeError):
            asserter.assert_similarity("actual", None)
    
    def test_min_score_below_zero_raises_error(self):
        """Test that min_score below 0 raises ValueError."""
        asserter = SemanticAssert()
        
        with pytest.raises(ValueError, match="min_score must be between 0.0 and 1.0"):
            asserter.assert_similarity("a", "b", min_score=-0.1)
    
    def test_min_score_above_one_raises_error(self):
        """Test that min_score above 1 raises ValueError."""
        asserter = SemanticAssert()
        
        with pytest.raises(ValueError, match="min_score must be between 0.0 and 1.0"):
            asserter.assert_similarity("a", "b", min_score=1.5)
    
    def test_valid_min_score_boundaries(self):
        """Test that boundary values for min_score are valid."""
        asserter = SemanticAssert()
        
        # Mock the model to avoid actual loading
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[1, 0], [1, 0]])
            mock_get.return_value = mock_model
            
            # These should not raise ValueError for min_score
            asserter.assert_similarity("a", "b", min_score=0.0)
            asserter.assert_similarity("a", "b", min_score=1.0)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        
        result = SemanticAssert._cosine_similarity(vec, vec)
        
        assert abs(result - 1.0) < 0.0001
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        result = SemanticAssert._cosine_similarity(vec1, vec2)
        
        assert abs(result) < 0.0001
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        
        result = SemanticAssert._cosine_similarity(vec1, vec2)
        
        assert abs(result - (-1.0)) < 0.0001
    
    def test_zero_vector_returns_zero(self):
        """Test that zero vector returns 0.0 similarity."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        result = SemanticAssert._cosine_similarity(vec1, vec2)
        
        assert result == 0.0


class TestSemanticAssertAssertion:
    """Tests for the assertion logic."""
    
    def test_high_similarity_passes(self):
        """Test that high similarity passes assertion."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            # Return identical embeddings (similarity = 1.0)
            mock_model.encode.return_value = np.array([[1, 0, 0], [1, 0, 0]])
            mock_get.return_value = mock_model
            
            # Should not raise
            asserter.assert_similarity("hello", "hi", min_score=0.9)
    
    def test_low_similarity_fails(self):
        """Test that low similarity fails assertion."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            # Return orthogonal embeddings (similarity = 0.0)
            mock_model.encode.return_value = np.array([[1, 0], [0, 1]])
            mock_get.return_value = mock_model
            
            with pytest.raises(AssertionError, match="Semantic similarity assertion failed"):
                asserter.assert_similarity("hello", "goodbye", min_score=0.5)
    
    def test_assertion_error_includes_score(self):
        """Test that AssertionError includes the actual score."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[1, 0], [0, 1]])
            mock_get.return_value = mock_model
            
            with pytest.raises(AssertionError) as exc_info:
                asserter.assert_similarity("a", "b", min_score=0.8)
            
            assert "Score:" in str(exc_info.value)
            assert "required:" in str(exc_info.value)
    
    def test_assertion_error_includes_text_preview(self):
        """Test that AssertionError includes text previews."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[1, 0], [0, 1]])
            mock_get.return_value = mock_model
            
            with pytest.raises(AssertionError) as exc_info:
                asserter.assert_similarity("actual text", "expected text", min_score=0.9)
            
            error_msg = str(exc_info.value)
            assert "Actual:" in error_msg
            assert "Expected:" in error_msg


class TestGetSimilarityScore:
    """Tests for get_similarity_score method."""
    
    def test_returns_float(self):
        """Test that get_similarity_score returns a float."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[1, 0], [1, 0]])
            mock_get.return_value = mock_model
            
            score = asserter.get_similarity_score("hello", "hi")
            
            assert isinstance(score, float)
    
    def test_score_in_valid_range(self):
        """Test that score is between -1 and 1."""
        asserter = SemanticAssert()
        
        with patch.object(asserter._model_loader, 'get_model') as mock_get:
            mock_model = MagicMock()
            # Random normalized vectors
            mock_model.encode.return_value = np.array([
                [0.5, 0.5, 0.5],
                [0.3, 0.7, 0.2]
            ])
            mock_get.return_value = mock_model
            
            score = asserter.get_similarity_score("text1", "text2")
            
            assert -1.0 <= score <= 1.0


# Integration tests (require actual model - mark as slow)
@pytest.mark.slow
class TestSemanticAssertIntegration:
    """Integration tests that use the actual sentence-transformers model.
    
    These tests are marked slow and may be skipped in CI with: pytest -m "not slow"
    """
    
    def test_similar_texts_high_score(self, similar_texts):
        """Test that semantically similar texts have high scores."""
        asserter = SemanticAssert()
        
        for text1, text2 in similar_texts:
            score = asserter.get_similarity_score(text1, text2)
            assert score > 0.5, f"Expected high similarity for: {text1!r} vs {text2!r}"
    
    def test_dissimilar_texts_low_score(self, dissimilar_texts):
        """Test that semantically dissimilar texts have low scores."""
        asserter = SemanticAssert()
        
        for text1, text2 in dissimilar_texts:
            score = asserter.get_similarity_score(text1, text2)
            assert score < 0.5, f"Expected low similarity for: {text1!r} vs {text2!r}"
    
    def test_identical_texts_perfect_score(self):
        """Test that identical texts have score of 1.0."""
        asserter = SemanticAssert()
        text = "This is a test sentence."
        
        score = asserter.get_similarity_score(text, text)
        
        assert abs(score - 1.0) < 0.01  # Allow small floating point error
