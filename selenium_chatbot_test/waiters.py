"""
Stream-aware waiting utilities for Generative AI interfaces.

This module provides the StreamWaiter class which uses JavaScript MutationObserver
to detect when a streaming response has completed.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, JavascriptException

logger = logging.getLogger(__name__)


class StreamWaiter:
    """
    Waits for a streaming response to complete using MutationObserver.
    
    Standard Selenium waits capture partial text during streaming responses.
    StreamWaiter injects a JavaScript MutationObserver that only resolves
    when the target element has been silent (no mutations) for a configurable
    timeout period, ensuring the complete response is captured.
    
    Example:
        >>> from selenium_chatbot_test import StreamWaiter
        >>> from selenium.webdriver.common.by import By
        >>> 
        >>> waiter = StreamWaiter()
        >>> element = waiter.wait_for_stream_end(
        ...     driver, 
        ...     (By.CSS_SELECTOR, ".chat-response"),
        ...     silence_timeout=0.5,
        ...     timeout=30.0
        ... )
        >>> print(element.text)  # Complete response text
    """
    
    # JavaScript code for MutationObserver-based stream detection
    _OBSERVER_SCRIPT = """
    return new Promise((resolve, reject) => {
        const locatorType = arguments[0];
        const locatorValue = arguments[1];
        const silenceTimeoutMs = arguments[2];
        const totalTimeoutMs = arguments[3];
        
        let element;
        try {
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
                reject(new Error('Unsupported locator type: ' + locatorType));
                return;
            }
        } catch (e) {
            reject(new Error('Error finding element: ' + e.message));
            return;
        }
        
        if (!element) {
            reject(new Error('Element not found with locator: ' + locatorType + '=' + locatorValue));
            return;
        }
        
        let silenceTimer = null;
        let observer = null;
        let totalTimer = null;
        let resolved = false;
        
        const cleanup = () => {
            if (silenceTimer) {
                clearTimeout(silenceTimer);
                silenceTimer = null;
            }
            if (totalTimer) {
                clearTimeout(totalTimer);
                totalTimer = null;
            }
            if (observer) {
                observer.disconnect();
                observer = null;
            }
        };
        
        const resolveWithElement = () => {
            if (resolved) return;
            resolved = true;
            cleanup();
            resolve(element);
        };
        
        const resetSilenceTimer = () => {
            if (silenceTimer) {
                clearTimeout(silenceTimer);
            }
            silenceTimer = setTimeout(resolveWithElement, silenceTimeoutMs);
        };
        
        // Set up total timeout
        totalTimer = setTimeout(() => {
            if (resolved) return;
            resolved = true;
            cleanup();
            reject(new Error('Timeout waiting for stream to complete after ' + totalTimeoutMs + 'ms'));
        }, totalTimeoutMs);
        
        // Set up mutation observer
        observer = new MutationObserver((mutations) => {
            if (mutations.length > 0) {
                resetSilenceTimer();
            }
        });
        
        observer.observe(element, {
            childList: true,
            characterData: true,
            subtree: true,
            characterDataOldValue: true
        });
        
        // Start initial silence timer
        resetSilenceTimer();
    });
    """

    def wait_for_stream_end(
        self,
        driver: WebDriver,
        locator: Tuple[str, str],
        silence_timeout: float = 0.5,
        timeout: float = 30.0
    ) -> WebElement:
        """
        Wait for a streaming response to complete.
        
        This method injects a JavaScript MutationObserver that monitors the target
        element for mutations. It only resolves when the element has been silent
        (no mutations) for the specified silence_timeout period.
        
        Args:
            driver: Selenium WebDriver instance.
            locator: Tuple of (locator_type, locator_value), e.g., 
                     (By.CSS_SELECTOR, ".response") or (By.ID, "chat-box").
            silence_timeout: Time in seconds without mutations before considering
                           the stream complete. Default is 0.5 seconds.
            timeout: Maximum time in seconds to wait for stream completion.
                    Default is 30.0 seconds.
        
        Returns:
            WebElement: The target element after streaming has completed.
        
        Raises:
            TimeoutException: If the stream doesn't complete within the timeout.
            JavascriptException: If there's an error executing the observer script.
            ValueError: If the locator format is invalid.
        
        Example:
            >>> from selenium.webdriver.common.by import By
            >>> waiter = StreamWaiter()
            >>> element = waiter.wait_for_stream_end(
            ...     driver,
            ...     (By.ID, "response-container"),
            ...     silence_timeout=0.5,
            ...     timeout=30.0
            ... )
        """
        if not isinstance(locator, (tuple, list)) or len(locator) != 2:
            raise ValueError(
                f"Locator must be a tuple of (type, value), got: {locator}"
            )
        
        locator_type, locator_value = locator
        silence_timeout_ms = int(silence_timeout * 1000)
        timeout_ms = int(timeout * 1000)
        
        logger.debug(
            f"Waiting for stream end on element {locator_type}={locator_value} "
            f"(silence={silence_timeout}s, timeout={timeout}s)"
        )
        
        try:
            result = driver.execute_script(
                self._OBSERVER_SCRIPT,
                locator_type,
                locator_value,
                silence_timeout_ms,
                timeout_ms
            )
            
            logger.debug("Stream completed successfully")
            return result
            
        except JavascriptException as e:
            error_msg = str(e)
            if "Timeout waiting for stream" in error_msg:
                raise TimeoutException(
                    f"Stream did not complete within {timeout}s timeout"
                ) from e
            elif "Element not found" in error_msg:
                raise TimeoutException(
                    f"Element not found: {locator_type}={locator_value}"
                ) from e
            else:
                raise
