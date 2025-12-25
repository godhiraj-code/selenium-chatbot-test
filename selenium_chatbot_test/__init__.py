"""
Selenium Chatbot Test - A library for testing Generative AI interfaces.

This library extends Selenium WebDriver to reliably test Chatbots, Copilots,
and Streaming UIs by providing stream-aware waiting, semantic assertions,
and latency monitoring.
"""

from __future__ import annotations

from selenium_chatbot_test.waiters import StreamWaiter
from selenium_chatbot_test.assertions import SemanticAssert
from selenium_chatbot_test.metrics import LatencyMonitor

__version__ = "0.2.0"
__all__ = ["StreamWaiter", "SemanticAssert", "LatencyMonitor"]
