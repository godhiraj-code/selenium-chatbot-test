# Selenium Chatbot Test

[![CI](https://github.com/godhiraj-code/selenium-chatbot-test/actions/workflows/ci.yml/badge.svg)](https://github.com/godhiraj-code/selenium-chatbot-test/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Selenium helpers for testing chatbots, copilots, and other streaming interfaces: wait for DOM activity to settle, compare response similarity, and measure browser-observed latency.

**Author**: [Dhiraj Das](https://www.dhirajdas.dev) | **Version**: 0.2.0

## What it provides

A visible response element may still be changing, and generated answers may vary
between runs. This library adds three helpers to existing Selenium tests:

| Helper | Purpose |
|--------|---------|
| `StreamWaiter` | Wait for a configurable period without text or child-node mutations |
| `SemanticAssert` | Compare sentence embeddings with a cosine-similarity threshold |
| `LatencyMonitor` | Record time to the first and last observed DOM mutations |

The stream waiter uses the browser's `MutationObserver` rather than a polling
loop. Semantic models load on first use and are cached in the Python process;
they can run on CPU without a GPU. The latency monitor is a context manager that
disconnects its observer on exit. The package includes type hints and a `py.typed`
marker.

## Limitations

- **Quiet is not necessarily complete.** `StreamWaiter` treats a pause of
  `silence_timeout` seconds as completion, including an initially quiet element.
  A delayed first chunk or a longer pause between chunks can return early. Use
  application-specific start/completion checks where available and tune the
  silence window for the interface.
- **Similarity is not correctness.** Tune `min_score` for the use case and retain
  assertions for required facts and behavior. A high similarity score does not
  establish factual accuracy.
- **Semantic assertions need optional dependencies.** The first use may download
  the selected model (about 90 MB for `all-MiniLM-L6-v2`), which is cached for
  subsequent use.
- **Latency is measured at the DOM.** These timings are not server-side model
  timings, and mutation counts are not tokenizer token counts.
- **Browser integration is required.** The target element must exist before
  observation starts, and the page must allow script execution. The demo uses
  Chrome; validate other browsers in the target environment.

## Installation

```bash
# Install from PyPI
pip install selenium-chatbot-test

# Add ML-powered semantic assertions when needed
pip install 'selenium-chatbot-test[semantic]'

# Or install from source
git clone https://github.com/godhiraj-code/selenium-chatbot-test.git
cd selenium-chatbot-test
pip install -e .
```

## Quick Start

This template requires the `[semantic]` extra. Replace the URL and locators with
your application's values. The response container must already exist; add the
application's own response-start and completion checks when available.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_chatbot_test import StreamWaiter, SemanticAssert, LatencyMonitor

driver = webdriver.Chrome()
waiter = StreamWaiter()
asserter = SemanticAssert()  # Requires installation with the [semantic] extra

# Navigate to chatbot
driver.get("https://your-chatbot-url.com")

# Send a message and wait for streaming response
with LatencyMonitor(driver, (By.ID, "response-box")) as monitor:
    driver.find_element(By.ID, "send-btn").click()
    
    # Wait for the configured quiet period
    element = waiter.wait_for_stream_end(
        driver,
        (By.ID, "response-box"),
        silence_timeout=0.5,  # Treat 500 ms without mutations as settled
        timeout=30.0
    )

# Read the response after the quiet period
response = element.text
print(f"Response: {response}")

# Compare similarity to the expected response
asserter.assert_similarity(
    response,
    "Hello! How can I help you today?",
    min_score=0.7  # Minimum cosine-similarity score
)

# Check latency metrics
print(f"Time to first mutation: {monitor.metrics.ttft_ms} ms")
print(f"Time to last mutation: {monitor.metrics.total_ms} ms")

driver.quit()
```

## API Reference

### StreamWaiter

Waits for a quiet period in the target element using JavaScript `MutationObserver`.

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

Install the optional ML dependencies first:

```bash
pip install 'selenium-chatbot-test[semantic]'
```

```python
asserter = SemanticAssert()

# Assert similarity (raises AssertionError if below threshold)
asserter.assert_similarity(
    actual="The weather is nice today",
    expected="It's a beautiful day",
    min_score=0.7,
    model_name="all-MiniLM-L6-v2"
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
- `ttft_ms`: Milliseconds from monitor start to the first observed mutation.
- `total_ms`: Milliseconds from monitor start to the last observed mutation.
- `token_count`: Number of mutation records observed, not an exact token count.

Timing values remain `None` if no mutations are observed. Read metrics after the
context manager exits.

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
pip install -e ".[dev]"

# Run tests
pytest tests/ -v -m "not slow"

# Check formatting
black --check selenium_chatbot_test tests
isort --check-only selenium_chatbot_test tests
mypy selenium_chatbot_test --ignore-missing-imports
```

## Requirements

- Python ≥ 3.9
- `selenium` ≥ 4.0.0
- Optional for semantic assertions: `sentence-transformers` ≥ 2.2.0 and
  `numpy` ≥ 1.21.0, installed with `selenium-chatbot-test[semantic]`

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

Licensed under MIT, as declared in [pyproject.toml](pyproject.toml).
