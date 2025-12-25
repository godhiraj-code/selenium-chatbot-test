"""
Tests for StreamWaiter class.

Tests the MutationObserver-based stream waiting functionality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from selenium.common.exceptions import JavascriptException, TimeoutException

from selenium_chatbot_test.waiters import StreamWaiter


class TestStreamWaiterInit:
    """Tests for StreamWaiter initialization."""

    def test_create_instance(self):
        """Test that StreamWaiter can be instantiated."""
        waiter = StreamWaiter()
        assert waiter is not None

    def test_observer_script_exists(self):
        """Test that the observer script is defined."""
        waiter = StreamWaiter()
        assert hasattr(waiter, "_OBSERVER_SCRIPT")
        assert "MutationObserver" in waiter._OBSERVER_SCRIPT


class TestStreamWaiterValidation:
    """Tests for input validation in StreamWaiter."""

    def test_invalid_locator_type_raises_error(self, mock_driver):
        """Test that invalid locator format raises ValueError."""
        waiter = StreamWaiter()

        with pytest.raises(ValueError, match="Locator must be a tuple"):
            waiter.wait_for_stream_end(mock_driver, "not a tuple")

    def test_invalid_locator_length_raises_error(self, mock_driver):
        """Test that locator with wrong length raises ValueError."""
        waiter = StreamWaiter()

        with pytest.raises(ValueError, match="Locator must be a tuple"):
            waiter.wait_for_stream_end(mock_driver, ("id",))

    def test_valid_locator_formats(self, mock_driver, sample_locator):
        """Test that valid locator formats work."""
        waiter = StreamWaiter()

        # Should not raise
        result = waiter.wait_for_stream_end(
            mock_driver, sample_locator, silence_timeout=0.1, timeout=1.0
        )
        assert result is not None


class TestStreamWaiterExecution:
    """Tests for StreamWaiter execution logic."""

    def test_script_executed_with_correct_args(self, mock_driver):
        """Test that execute_script is called with correct arguments."""
        waiter = StreamWaiter()
        locator = ("id", "test-element")

        waiter.wait_for_stream_end(
            mock_driver, locator, silence_timeout=0.5, timeout=30.0
        )

        # Check script was called
        assert len(mock_driver._execute_script_calls) == 1
        script, args = mock_driver._execute_script_calls[0]

        # Verify arguments passed to script
        assert args[0] == "id"
        assert args[1] == "test-element"
        assert args[2] == 500  # silence_timeout in ms
        assert args[3] == 30000  # timeout in ms

    def test_different_locator_types(self, mock_driver):
        """Test various supported locator types."""
        waiter = StreamWaiter()

        locator_types = [
            ("id", "element-id"),
            ("css selector", ".class-name"),
            ("xpath", "//div[@id='test']"),
            ("class name", "my-class"),
            ("tag name", "div"),
            ("name", "input-name"),
        ]

        for locator in locator_types:
            result = waiter.wait_for_stream_end(
                mock_driver, locator, silence_timeout=0.1, timeout=1.0
            )
            assert result is not None

    def test_returns_web_element(self, mock_driver, sample_locator):
        """Test that a WebElement-like object is returned."""
        waiter = StreamWaiter()

        result = waiter.wait_for_stream_end(
            mock_driver, sample_locator, silence_timeout=0.1, timeout=1.0
        )

        # Should have text attribute like WebElement
        assert hasattr(result, "text")


class TestStreamWaiterErrorHandling:
    """Tests for error handling in StreamWaiter."""

    def test_timeout_exception_on_stream_timeout(self, mock_driver, sample_locator):
        """Test that TimeoutException is raised on stream timeout."""
        waiter = StreamWaiter()

        # Configure mock to raise JavascriptException with timeout message
        def raise_timeout(*args):
            raise JavascriptException("Timeout waiting for stream to complete")

        mock_driver.execute_script = raise_timeout

        with pytest.raises(TimeoutException, match="Stream did not complete"):
            waiter.wait_for_stream_end(mock_driver, sample_locator)

    def test_timeout_exception_on_element_not_found(self, mock_driver, sample_locator):
        """Test that TimeoutException is raised when element not found."""
        waiter = StreamWaiter()

        def raise_not_found(*args):
            raise JavascriptException("Element not found: id=missing")

        mock_driver.execute_script = raise_not_found

        with pytest.raises(TimeoutException, match="Element not found"):
            waiter.wait_for_stream_end(mock_driver, sample_locator)

    def test_javascript_exception_propagated(self, mock_driver, sample_locator):
        """Test that other JavascriptExceptions are propagated."""
        waiter = StreamWaiter()

        def raise_other_error(*args):
            raise JavascriptException("Some other JavaScript error")

        mock_driver.execute_script = raise_other_error

        with pytest.raises(JavascriptException, match="Some other JavaScript error"):
            waiter.wait_for_stream_end(mock_driver, sample_locator)


class TestStreamWaiterTimeoutConversion:
    """Tests for timeout value conversion."""

    def test_silence_timeout_converted_to_ms(self, mock_driver, sample_locator):
        """Test that silence_timeout is converted from seconds to milliseconds."""
        waiter = StreamWaiter()

        waiter.wait_for_stream_end(
            mock_driver,
            sample_locator,
            silence_timeout=1.5,  # 1.5 seconds
            timeout=10.0,
        )

        _, args = mock_driver._execute_script_calls[0]
        assert args[2] == 1500  # Should be 1500ms

    def test_total_timeout_converted_to_ms(self, mock_driver, sample_locator):
        """Test that total timeout is converted from seconds to milliseconds."""
        waiter = StreamWaiter()

        waiter.wait_for_stream_end(
            mock_driver, sample_locator, silence_timeout=0.5, timeout=45.0  # 45 seconds
        )

        _, args = mock_driver._execute_script_calls[0]
        assert args[3] == 45000  # Should be 45000ms
