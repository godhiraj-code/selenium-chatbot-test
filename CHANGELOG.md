# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-25

### Added
- Comprehensive test suite with 54 unit tests
- GitHub Actions CI workflow with matrix testing (Python 3.9-3.12, Ubuntu/Windows)
- pytest and coverage configuration in `pyproject.toml`
- Test fixtures for mocking WebDriver and WebElement
- Pre-commit configuration (black, isort, mypy)

## [0.1.0] - 2025-12-25

### Added
- `StreamWaiter` class for MutationObserver-based stream detection
- `SemanticAssert` class for ML-powered semantic similarity assertions
- `LatencyMonitor` context manager for TTFT and total latency measurement
- `LatencyMetrics` dataclass for latency measurement results
- Lazy model loading for sentence-transformers
- GPU/CPU fallback for semantic embeddings
- Demo script (`demo_chatbot.py`) with local simulation
- Full type hints (PEP-561 compliant with `py.typed`)
- Comprehensive documentation and README

### Technical Details
- Uses `all-MiniLM-L6-v2` model for fast semantic similarity
- JavaScript MutationObserver for accurate stream detection
- Context manager pattern for automatic resource cleanup

[Unreleased]: https://github.com/godhiraj-code/selenium-chatbot-test/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/godhiraj-code/selenium-chatbot-test/releases/tag/v0.1.0
