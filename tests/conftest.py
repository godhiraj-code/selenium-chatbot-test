"""
Pytest configuration and fixtures for selenium-chatbot-test.

Provides mock WebDriver, WebElement, and other fixtures for unit testing
without requiring a real browser.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock

import pytest


class MockWebElement:
    """Mock Selenium WebElement for testing."""

    def __init__(self, text: str = "", tag_name: str = "div"):
        self.text = text
        self.tag_name = tag_name
        self._attributes: Dict[str, str] = {}

    def get_attribute(self, name: str) -> Optional[str]:
        return self._attributes.get(name)

    def find_element(self, by: str, value: str) -> "MockWebElement":
        return MockWebElement()

    def find_elements(self, by: str, value: str) -> list:
        return []


class MockWebDriver:
    """Mock Selenium WebDriver for testing."""

    def __init__(self):
        self._script_results: Dict[str, Any] = {}
        self._execute_script_calls: list = []
        self._current_url = "about:blank"

    def execute_script(self, script: str, *args) -> Any:
        """Mock execute_script that returns pre-configured results."""
        self._execute_script_calls.append((script, args))

        # Return element for observer scripts
        if "MutationObserver" in script and "Promise" in script:
            return MockWebElement(text="Mocked response text")

        # Return monitor key for latency monitor inject
        if "startTime: performance.now()" in script:
            return "__latencyMonitor_test_123"

        # Return metrics for latency monitor retrieve
        if "firstMutationTime" in script and "mutationCount" in script:
            return {
                "startTime": 1000.0,
                "firstMutationTime": 1050.0,
                "lastMutationTime": 1500.0,
                "mutationCount": 15,
            }

        return self._script_results.get("default")

    def set_script_result(self, key: str, result: Any) -> None:
        """Pre-configure a script result."""
        self._script_results[key] = result

    def get(self, url: str) -> None:
        self._current_url = url

    @property
    def current_url(self) -> str:
        return self._current_url

    def quit(self) -> None:
        pass


@pytest.fixture
def mock_driver() -> MockWebDriver:
    """Provide a mock WebDriver instance."""
    return MockWebDriver()


@pytest.fixture
def mock_element() -> MockWebElement:
    """Provide a mock WebElement instance."""
    return MockWebElement(text="Hello, how can I help you today?")


@pytest.fixture
def sample_locator():
    """Provide a sample locator tuple."""
    return ("id", "response-box")


@pytest.fixture
def css_locator():
    """Provide a CSS selector locator tuple."""
    return ("css selector", ".chat-response")


@pytest.fixture
def xpath_locator():
    """Provide an XPath locator tuple."""
    return ("xpath", "//div[@class='response']")


# Semantic assertion fixtures
@pytest.fixture
def similar_texts():
    """Provide semantically similar text pairs."""
    return [
        ("Hello, how can I help you today?", "Hi, how may I help you?"),
        ("The weather is beautiful today", "Today the weather is lovely"),
        ("I am a helpful assistant", "I'm an assistant here to help"),
    ]


@pytest.fixture
def dissimilar_texts():
    """Provide semantically dissimilar text pairs."""
    return [
        ("Hello, how are you?", "The capital of France is Paris"),
        ("I like pizza", "Quantum physics is complex"),
        ("Good morning!", "The stock market crashed"),
    ]
