"""
Latency monitoring utilities for Generative AI interfaces.

This module provides the LatencyMonitor context manager for measuring
Time-To-First-Token (TTFT) and total response latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Optional, Tuple, Type

from selenium.common.exceptions import JavascriptException
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """
    Container for latency measurements.

    Attributes:
        ttft_ms: Time-To-First-Token in milliseconds. Time from monitor start
                to the first mutation (content appearing).
        total_ms: Total response time in milliseconds. Time from monitor start
                 to the last mutation.
        token_count: Estimated number of mutations observed (rough proxy for tokens).
    """

    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    token_count: int = 0

    @property
    def ttft_seconds(self) -> Optional[float]:
        """Time-To-First-Token in seconds."""
        return self.ttft_ms / 1000.0 if self.ttft_ms is not None else None

    @property
    def total_seconds(self) -> Optional[float]:
        """Total response time in seconds."""
        return self.total_ms / 1000.0 if self.total_ms is not None else None


class LatencyMonitor:
    """
    Context manager for measuring streaming response latency.

    Injects a JavaScript MutationObserver that records performance.now()
    timestamps for the first and last mutations, enabling accurate
    Time-To-First-Token (TTFT) and total latency measurements.

    Example:
        >>> from selenium_chatbot_test import LatencyMonitor
        >>> from selenium.webdriver.common.by import By
        >>>
        >>> with LatencyMonitor(driver, (By.ID, "chat-box")) as monitor:
        ...     send_button.click()
        ...     # Wait for response to complete...
        ...     waiter.wait_for_stream_end(driver, (By.ID, "chat-box"))
        >>>
        >>> print(f"TTFT: {monitor.metrics.ttft_ms:.1f}ms")
        >>> print(f"Total: {monitor.metrics.total_ms:.1f}ms")

    Note:
        The observer is automatically disconnected in the __exit__ method
        to prevent memory leaks in Single Page Applications.
    """

    # JavaScript to inject the latency monitoring observer
    _INJECT_OBSERVER_SCRIPT = """
    const locatorType = arguments[0];
    const locatorValue = arguments[1];
    
    let element;
    if (locatorType === 'id') {
        element = document.getElementById(locatorValue);
    } else if (locatorType === 'css selector') {
        element = document.querySelector(locatorValue);
    } else if (locatorType === 'xpath') {
        element = document.evaluate(
            locatorValue, 
            document, 
            null, 
            XPathResult.FIRST_ORDERED_NODE_TYPE, 
            null
        ).singleNodeValue;
    } else if (locatorType === 'class name') {
        element = document.getElementsByClassName(locatorValue)[0];
    } else if (locatorType === 'tag name') {
        element = document.getElementsByTagName(locatorValue)[0];
    } else if (locatorType === 'name') {
        element = document.getElementsByName(locatorValue)[0];
    } else {
        throw new Error('Unsupported locator type: ' + locatorType);
    }
    
    if (!element) {
        throw new Error('Element not found: ' + locatorType + '=' + locatorValue);
    }
    
    // Create a unique key for this monitor instance
    const monitorKey = '__latencyMonitor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    // Initialize the monitor state
    window[monitorKey] = {
        startTime: performance.now(),
        firstMutationTime: null,
        lastMutationTime: null,
        mutationCount: 0,
        observer: null
    };
    
    const state = window[monitorKey];
    
    state.observer = new MutationObserver((mutations) => {
        const now = performance.now();
        
        if (mutations.length > 0) {
            state.mutationCount += mutations.length;
            
            if (state.firstMutationTime === null) {
                state.firstMutationTime = now;
            }
            state.lastMutationTime = now;
        }
    });
    
    state.observer.observe(element, {
        childList: true,
        characterData: true,
        subtree: true,
        characterDataOldValue: true
    });
    
    return monitorKey;
    """

    # JavaScript to retrieve metrics and disconnect observer
    _RETRIEVE_METRICS_SCRIPT = """
    const monitorKey = arguments[0];
    const state = window[monitorKey];
    
    if (!state) {
        return { error: 'Monitor not found: ' + monitorKey };
    }
    
    // Disconnect observer in finally block equivalent
    try {
        if (state.observer) {
            state.observer.disconnect();
        }
    } catch (e) {
        // Ignore disconnect errors
    }
    
    const result = {
        startTime: state.startTime,
        firstMutationTime: state.firstMutationTime,
        lastMutationTime: state.lastMutationTime,
        mutationCount: state.mutationCount
    };
    
    // Clean up
    try {
        delete window[monitorKey];
    } catch (e) {
        window[monitorKey] = undefined;
    }
    
    return result;
    """

    # JavaScript to force disconnect observer (for error cases)
    _DISCONNECT_SCRIPT = """
    const monitorKey = arguments[0];
    const state = window[monitorKey];
    
    if (state && state.observer) {
        try {
            state.observer.disconnect();
        } catch (e) {
            // Ignore
        }
    }
    
    try {
        delete window[monitorKey];
    } catch (e) {
        window[monitorKey] = undefined;
    }
    """

    def __init__(self, driver: WebDriver, locator: Tuple[str, str]) -> None:
        """
        Initialize the LatencyMonitor.

        Args:
            driver: Selenium WebDriver instance.
            locator: Tuple of (locator_type, locator_value), e.g.,
                    (By.ID, "chat-box") or (By.CSS_SELECTOR, ".response").

        Example:
            >>> from selenium.webdriver.common.by import By
            >>> monitor = LatencyMonitor(driver, (By.ID, "response-container"))
        """
        self._driver = driver
        self._locator = locator
        self._monitor_key: Optional[str] = None
        self._metrics = LatencyMetrics()

    @property
    def metrics(self) -> LatencyMetrics:
        """
        Get the collected latency metrics.

        Returns:
            LatencyMetrics: Object containing ttft_ms, total_ms, and token_count.

        Note:
            Metrics are only populated after exiting the context manager.
        """
        return self._metrics

    def __enter__(self) -> "LatencyMonitor":
        """
        Start monitoring latency.

        Injects the MutationObserver into the page and begins recording
        mutation timestamps.

        Returns:
            LatencyMonitor: Self for use in with statement.

        Raises:
            JavascriptException: If the observer cannot be injected.
        """
        if not isinstance(self._locator, (tuple, list)) or len(self._locator) != 2:
            raise ValueError(
                f"Locator must be a tuple of (type, value), got: {self._locator}"
            )

        locator_type, locator_value = self._locator

        logger.debug(f"Starting latency monitor for {locator_type}={locator_value}")

        try:
            self._monitor_key = self._driver.execute_script(
                self._INJECT_OBSERVER_SCRIPT, locator_type, locator_value
            )
            logger.debug(f"Latency monitor started with key: {self._monitor_key}")
            return self

        except JavascriptException as e:
            logger.error(f"Failed to start latency monitor: {e}")
            raise

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Stop monitoring and retrieve metrics.

        Disconnects the MutationObserver and calculates latency metrics
        from the recorded timestamps.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.

        Returns:
            bool: False to propagate any exceptions.

        Note:
            The observer is always disconnected, even if an exception occurred,
            to prevent memory leaks.
        """
        if self._monitor_key is None:
            return

        try:
            result = self._driver.execute_script(
                self._RETRIEVE_METRICS_SCRIPT, self._monitor_key
            )

            if isinstance(result, dict) and "error" not in result:
                start_time = result.get("startTime")
                first_mutation = result.get("firstMutationTime")
                last_mutation = result.get("lastMutationTime")
                mutation_count = result.get("mutationCount", 0)

                if first_mutation is not None and start_time is not None:
                    self._metrics.ttft_ms = first_mutation - start_time

                if last_mutation is not None and start_time is not None:
                    self._metrics.total_ms = last_mutation - start_time

                self._metrics.token_count = mutation_count

                logger.debug(
                    f"Latency metrics: TTFT={self._metrics.ttft_ms}ms, "
                    f"Total={self._metrics.total_ms}ms, "
                    f"Mutations={mutation_count}"
                )
            else:
                logger.warning(f"Failed to retrieve metrics: {result}")

        except JavascriptException as e:
            logger.error(f"Error retrieving latency metrics: {e}")
            # Attempt cleanup even if retrieval failed
            try:
                self._driver.execute_script(self._DISCONNECT_SCRIPT, self._monitor_key)
            except Exception:
                pass

        finally:
            self._monitor_key = None
