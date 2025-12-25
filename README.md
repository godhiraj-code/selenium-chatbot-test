# Selenium Chatbot Test

[![CI](https://github.com/godhiraj-code/selenium-chatbot-test/actions/workflows/ci.yml/badge.svg)](https://github.com/godhiraj-code/selenium-chatbot-test/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/godhiraj-code/selenium-chatbot-test)

A Python library that extends Selenium WebDriver to reliably test **Generative AI interfaces** — Chatbots, Copilots, and Streaming UIs.

**Author**: [Dhiraj Das](https://www.dhirajdas.dev) | **Version**: 0.2.0

## The Problem

Standard Selenium fails on GenAI interfaces because:

1. **Streaming Responses**: Standard waits read partial text mid-stream
2. **Non-Deterministic Output**: Exact string assertions fail on AI-generated content

## The Solution

`selenium-chatbot-test` provides three powerful tools:

| Module | Purpose |
|--------|---------|
| `StreamWaiter` | Waits for streaming responses to complete using MutationObserver |
| `SemanticAssert` | Asserts semantic similarity instead of exact string matching |
| `LatencyMonitor` | Measures TTFT and total latency with millisecond precision |

## ✨ Key Benefits

- **Reliable Stream Detection** — No more flaky tests due to partial text capture
- **Non-Deterministic Friendly** — Test AI outputs without exact string matching
- **Performance Insights** — Built-in TTFT and latency metrics for every interaction
- **CI/CD Ready** — Silent CPU fallback, no GPU required
- **Zero Polling** — Pure JavaScript MutationObserver, no `time.sleep()` hacks

## 🚀 What Makes It Unique

| Feature | Traditional Selenium | selenium-chatbot-test |
|---------|---------------------|----------------------|
| Streaming Text | ❌ Reads partial text | ✅ Waits for complete response |
| AI Assertions | ❌ Exact match only | ✅ Semantic similarity (ML-powered) |
| Latency Metrics | ❌ Manual timing | ✅ Automatic TTFT tracking |
| Memory Safety | ❌ Potential leaks | ✅ Auto-cleanup observers |

## 🏆 Standing Out Features

1. **MutationObserver-Based Waiting** — Industry-first approach using browser-native APIs instead of polling
2. **Lazy Model Loading** — Heavy ML models load on first use, not import (fast test startup)
3. **Semantic Embeddings** — Uses `all-MiniLM-L6-v2` for blazing-fast similarity scoring
4. **Context Manager Pattern** — Clean, Pythonic API with automatic resource cleanup
5. **Full Type Hints** — PEP-561 compliant with `py.typed` marker

## ⚠️ Limitations

| Limitation | Details |
|------------|---------|
| **Model Download** | First run downloads ~90MB model (cached thereafter) |
| **Semantic Threshold** | Requires tuning `min_score` per use case |
| **Browser Support** | Tested on Chrome; other browsers may vary |
| **JavaScript Required** | Target pages must allow script injection |
| **Not for Unit Tests** | Designed for E2E/integration testing only |

## Installation

```bash
# Install from PyPI
pip install selenium-chatbot-test

# Or install from source
git clone https://github.com/godhiraj-code/selenium-chatbot-test.git
cd selenium-chatbot-test
pip install -e .
```

## Quick Start

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_chatbot_test import StreamWaiter, SemanticAssert, LatencyMonitor

driver = webdriver.Chrome()
waiter = StreamWaiter()
asserter = SemanticAssert()

# Navigate to chatbot
driver.get("https://your-chatbot-url.com")

# Send a message and wait for streaming response
with LatencyMonitor(driver, (By.ID, "response-box")) as monitor:
    driver.find_element(By.ID, "send-btn").click()
    
    # Wait for stream to complete (not partial text!)
    element = waiter.wait_for_stream_end(
        driver,
        (By.ID, "response-box"),
        silence_timeout=0.5,  # 500ms of silence = stream complete
        timeout=30.0
    )

# Get the complete response
response = element.text
print(f"Response: {response}")

# Assert semantic similarity (works with non-deterministic AI!)
asserter.assert_similarity(
    response,
    "Hello! How can I help you today?",
    min_score=0.7  # 70% semantic similarity required
)

# Check latency metrics
print(f"Time-To-First-Token: {monitor.metrics.ttft_ms:.1f}ms")
print(f"Total Latency: {monitor.metrics.total_ms:.1f}ms")

driver.quit()
```

## API Reference

### StreamWaiter

Waits for streaming content to complete using JavaScript MutationObserver.

```python
waiter = StreamWaiter()

element = waiter.wait_for_stream_end(
    driver,                    # Selenium WebDriver
    (By.ID, "response"),       # Element locator
    silence_timeout=0.5,       # Seconds of silence before "complete"
    timeout=30.0               # Maximum wait time
)
```

**How it works**: Injects a MutationObserver that resets a timer on each DOM mutation. Only resolves when no mutations occur for `silence_timeout` seconds.

### SemanticAssert

Performs semantic similarity assertions using sentence-transformers.

```python
asserter = SemanticAssert()

# Assert similarity (raises AssertionError if below threshold)
asserter.assert_similarity(
    actual="The weather is nice today",
    expected="It's a beautiful day",
    min_score=0.7,
    model_name="all-MiniLM-L6-v2"  # Fast and accurate
)

# Or just get the score
score = asserter.get_similarity_score(text1, text2)
print(f"Similarity: {score:.2%}")
```

**Features**:
- **Lazy Loading**: Model loads on first use, not import
- **GPU Fallback**: Automatically uses CPU if CUDA unavailable
- **Caching**: Model is singleton, loaded once per session

### LatencyMonitor

Context manager for measuring streaming response latency.

```python
with LatencyMonitor(driver, (By.ID, "chat-box")) as monitor:
    send_button.click()
    # ... wait for response ...

print(f"TTFT: {monitor.metrics.ttft_ms}ms")
print(f"Total: {monitor.metrics.total_ms}ms")
print(f"Mutations: {monitor.metrics.token_count}")
```

**Metrics**:
- `ttft_ms`: Time-To-First-Token (first mutation)
- `total_ms`: Total response time (last mutation)
- `token_count`: Number of mutations observed

## Running the Demo

```bash
# Run the demo (uses local streaming simulation)
python demo_chatbot.py

# Run in headless mode
python demo_chatbot.py --headless
```

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/godhiraj-code/selenium-chatbot-test.git
cd selenium-chatbot-test
pip install -e .[dev]

# Run tests
pytest tests/ -v -m "not slow"

# Run linting
black selenium_chatbot_test tests
isort selenium_chatbot_test tests
mypy selenium_chatbot_test --ignore-missing-imports
```

## Requirements

- Python ≥ 3.9
- `selenium` ≥ 4.0.0
- `sentence-transformers` ≥ 2.2.0
- `numpy` ≥ 1.21.0

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT License - see [LICENSE](LICENSE) for details.
