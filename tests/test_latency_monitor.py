"""
Tests for LatencyMonitor class.

Tests the context manager for measuring TTFT and total latency.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from selenium.common.exceptions import JavascriptException

from selenium_chatbot_test.metrics import LatencyMonitor, LatencyMetrics


class TestLatencyMetrics:
    """Tests for LatencyMetrics dataclass."""
    
    def test_default_values(self):
        """Test that default values are None/0."""
        metrics = LatencyMetrics()
        
        assert metrics.ttft_ms is None
        assert metrics.total_ms is None
        assert metrics.token_count == 0
    
    def test_ttft_seconds_conversion(self):
        """Test ttft_seconds property converts correctly."""
        metrics = LatencyMetrics(ttft_ms=1500.0)
        
        assert metrics.ttft_seconds == 1.5
    
    def test_ttft_seconds_none_when_not_set(self):
        """Test ttft_seconds returns None when ttft_ms is None."""
        metrics = LatencyMetrics()
        
        assert metrics.ttft_seconds is None
    
    def test_total_seconds_conversion(self):
        """Test total_seconds property converts correctly."""
        metrics = LatencyMetrics(total_ms=3000.0)
        
        assert metrics.total_seconds == 3.0
    
    def test_total_seconds_none_when_not_set(self):
        """Test total_seconds returns None when total_ms is None."""
        metrics = LatencyMetrics()
        
        assert metrics.total_seconds is None
    
    def test_all_values_set(self):
        """Test metrics with all values populated."""
        metrics = LatencyMetrics(
            ttft_ms=100.0,
            total_ms=2000.0,
            token_count=50
        )
        
        assert metrics.ttft_ms == 100.0
        assert metrics.total_ms == 2000.0
        assert metrics.token_count == 50
        assert metrics.ttft_seconds == 0.1
        assert metrics.total_seconds == 2.0


class TestLatencyMonitorInit:
    """Tests for LatencyMonitor initialization."""
    
    def test_create_instance(self, mock_driver, sample_locator):
        """Test that LatencyMonitor can be instantiated."""
        monitor = LatencyMonitor(mock_driver, sample_locator)
        
        assert monitor is not None
        assert monitor.metrics is not None
    
    def test_initial_metrics_empty(self, mock_driver, sample_locator):
        """Test that initial metrics are empty."""
        monitor = LatencyMonitor(mock_driver, sample_locator)
        
        assert monitor.metrics.ttft_ms is None
        assert monitor.metrics.total_ms is None
        assert monitor.metrics.token_count == 0


class TestLatencyMonitorValidation:
    """Tests for input validation in LatencyMonitor."""
    
    def test_invalid_locator_type_raises_error(self, mock_driver):
        """Test that invalid locator format raises ValueError on enter."""
        monitor = LatencyMonitor(mock_driver, "not a tuple")
        
        with pytest.raises(ValueError, match="Locator must be a tuple"):
            monitor.__enter__()
    
    def test_invalid_locator_length_raises_error(self, mock_driver):
        """Test that locator with wrong length raises ValueError."""
        monitor = LatencyMonitor(mock_driver, ("id",))
        
        with pytest.raises(ValueError, match="Locator must be a tuple"):
            monitor.__enter__()


class TestLatencyMonitorContextManager:
    """Tests for context manager behavior."""
    
    def test_enter_returns_self(self, mock_driver, sample_locator):
        """Test that __enter__ returns the monitor instance."""
        monitor = LatencyMonitor(mock_driver, sample_locator)
        
        result = monitor.__enter__()
        
        assert result is monitor
        monitor.__exit__(None, None, None)
    
    def test_context_manager_syntax(self, mock_driver, sample_locator):
        """Test using with statement."""
        with LatencyMonitor(mock_driver, sample_locator) as monitor:
            assert monitor is not None
            assert isinstance(monitor, LatencyMonitor)
    
    def test_exit_returns_false(self, mock_driver, sample_locator):
        """Test that __exit__ returns False (doesn't suppress exceptions)."""
        monitor = LatencyMonitor(mock_driver, sample_locator)
        monitor.__enter__()
        
        result = monitor.__exit__(None, None, None)
        
        assert result is False
    
    def test_observer_injected_on_enter(self, mock_driver, sample_locator):
        """Test that observer is injected when entering context."""
        with LatencyMonitor(mock_driver, sample_locator):
            # Script should have been called
            assert len(mock_driver._execute_script_calls) >= 1
            
            # First call should be inject script
            script, _ = mock_driver._execute_script_calls[0]
            assert "startTime: performance.now()" in script


class TestLatencyMonitorMetrics:
    """Tests for metrics collection."""
    
    def test_metrics_populated_after_exit(self, mock_driver, sample_locator):
        """Test that metrics are populated after exiting context."""
        with LatencyMonitor(mock_driver, sample_locator) as monitor:
            pass  # Simulate some activity
        
        # Metrics should now be populated from mock
        assert monitor.metrics.ttft_ms == 50.0  # 1050 - 1000
        assert monitor.metrics.total_ms == 500.0  # 1500 - 1000
        assert monitor.metrics.token_count == 15
    
    def test_metrics_accessible_after_context(self, mock_driver, sample_locator):
        """Test that metrics remain accessible after context exits."""
        monitor = None
        
        with LatencyMonitor(mock_driver, sample_locator) as m:
            monitor = m
        
        # Should still be able to access metrics
        assert monitor.metrics is not None
        assert monitor.metrics.ttft_ms is not None


class TestLatencyMonitorErrorHandling:
    """Tests for error handling in LatencyMonitor."""
    
    def test_javascript_exception_on_enter(self, mock_driver, sample_locator):
        """Test that JavascriptException on enter is propagated."""
        def raise_error(*args):
            raise JavascriptException("Element not found")
        
        mock_driver.execute_script = raise_error
        monitor = LatencyMonitor(mock_driver, sample_locator)
        
        with pytest.raises(JavascriptException):
            monitor.__enter__()
    
    def test_cleanup_on_error(self, mock_driver, sample_locator):
        """Test that cleanup happens even on error during exit."""
        monitor = LatencyMonitor(mock_driver, sample_locator)
        monitor.__enter__()
        
        # Simulate error during metric retrieval
        original_execute = mock_driver.execute_script
        call_count = [0]
        
        def error_on_retrieve(*args):
            call_count[0] += 1
            if call_count[0] > 1:  # Error on retrieve call
                raise JavascriptException("Retrieval failed")
            return original_execute(*args)
        
        mock_driver.execute_script = error_on_retrieve
        
        # Should not raise, cleanup should happen
        monitor.__exit__(None, None, None)
    
    def test_exception_not_suppressed(self, mock_driver, sample_locator):
        """Test that exceptions in context are not suppressed."""
        with pytest.raises(ValueError, match="Test error"):
            with LatencyMonitor(mock_driver, sample_locator):
                raise ValueError("Test error")


class TestLatencyMonitorScriptExecution:
    """Tests for script execution details."""
    
    def test_inject_script_arguments(self, mock_driver):
        """Test that inject script receives correct arguments."""
        locator = ("css selector", ".response-box")
        
        with LatencyMonitor(mock_driver, locator):
            pass
        
        # Check inject script arguments
        _, args = mock_driver._execute_script_calls[0]
        assert args[0] == "css selector"
        assert args[1] == ".response-box"
    
    def test_retrieve_script_uses_monitor_key(self, mock_driver, sample_locator):
        """Test that retrieve script uses the monitor key."""
        with LatencyMonitor(mock_driver, sample_locator):
            pass
        
        # Should have at least 2 calls (inject and retrieve)
        assert len(mock_driver._execute_script_calls) >= 2
        
        # Retrieve call should include monitor key
        retrieve_script, args = mock_driver._execute_script_calls[1]
        assert "firstMutationTime" in retrieve_script
        assert "__latencyMonitor_test_123" in args
