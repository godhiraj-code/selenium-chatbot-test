#!/usr/bin/env python3
"""
Demo script for selenium-chatbot-test library.

This script demonstrates the library's capabilities using a public
text streaming demo. It shows how to:
1. Wait for streaming responses to complete
2. Assert semantic similarity of AI responses
3. Measure response latency (TTFT and total time)

Usage:
    python demo_chatbot.py

Requirements:
    - Chrome browser installed
    - Internet connection
    - selenium-chatbot-test package installed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from selenium_chatbot_test import StreamWaiter, SemanticAssert, LatencyMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Create a Chrome WebDriver instance.
    
    Args:
        headless: Whether to run in headless mode.
    
    Returns:
        Chrome WebDriver instance.
    """
    options = Options()
    
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    # Suppress logging
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    
    return driver


def demo_huggingface_chat(driver: webdriver.Chrome) -> bool:
    """
    Demo using HuggingFace Chat (if accessible).
    
    Note: HuggingFace Chat may require login or have rate limiting.
    This function attempts to use it but may fail gracefully.
    """
    logger.info("Attempting HuggingFace Chat demo...")
    
    try:
        driver.get("https://huggingface.co/chat/")
        time.sleep(3)  # Allow page to load
        
        # Check if we're blocked or need to login
        page_source = driver.page_source.lower()
        if "sign in" in page_source or "login" in page_source:
            logger.warning("HuggingFace Chat requires login, skipping...")
            return False
        
        # Look for chat input
        chat_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )
        
        if not chat_input:
            logger.warning("Could not find chat input on HuggingFace")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"HuggingFace Chat not accessible: {e}")
        return False


def demo_with_gradio_textbox(driver: webdriver.Chrome) -> None:
    """
    Demo using a simple Gradio text generation space.
    
    Uses a public Gradio space that streams text output.
    """
    # Use a simple, public Gradio space for text generation
    # This is a fallback demo that simulates the streaming behavior
    demo_url = "https://huggingface.co/spaces/gradio/hello_world"
    
    logger.info(f"Opening demo at: {demo_url}")
    driver.get(demo_url)
    
    # Wait for the Gradio interface to load
    time.sleep(5)
    
    try:
        # Look for the Gradio frame (embedded spaces use iframes)
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        if frames:
            driver.switch_to.frame(frames[0])
            logger.info("Switched to Gradio iframe")
        
        # Find the input textbox
        input_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
        )
        
        logger.info("Found input element, entering text...")
        input_box.clear()
        input_box.send_keys("Hello")
        
        # Find and click submit button
        submit_btn = driver.find_element(By.CSS_SELECTOR, "button.primary, button[type='submit']")
        
        # Set up latency monitoring on the output area
        output_locator = (By.CSS_SELECTOR, ".output-class, .output-text, .prose, [data-testid='output']")
        
        waiter = StreamWaiter()
        asserter = SemanticAssert()
        
        with LatencyMonitor(driver, output_locator) as monitor:
            submit_btn.click()
            logger.info("Clicked submit, waiting for response...")
            
            # Wait for streaming to complete
            try:
                element = waiter.wait_for_stream_end(
                    driver, 
                    output_locator,
                    silence_timeout=1.0,
                    timeout=30.0
                )
                response_text = element.text
            except TimeoutException:
                # Fallback: just get current text
                output_element = driver.find_element(*output_locator)
                response_text = output_element.text
        
        logger.info(f"Response received: {response_text[:100]}...")
        logger.info(f"TTFT: {monitor.metrics.ttft_ms:.1f}ms" if monitor.metrics.ttft_ms else "TTFT: N/A")
        logger.info(f"Total latency: {monitor.metrics.total_ms:.1f}ms" if monitor.metrics.total_ms else "Total: N/A")
        
        # Semantic assertion (relaxed for simple demo)
        try:
            asserter.assert_similarity(
                response_text,
                "Hello, this is a greeting response",
                min_score=0.3  # Very relaxed for demo
            )
            logger.info("✓ Semantic assertion passed!")
        except AssertionError as e:
            logger.warning(f"Semantic assertion failed (expected for simple demo): {e}")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


def demo_local_simulation(driver: webdriver.Chrome) -> None:
    """
    Demo using a locally simulated streaming response.
    
    This creates a simple HTML page that simulates streaming text,
    allowing demonstration of the library without external dependencies.
    """
    logger.info("Running local streaming simulation demo...")
    
    # Create a local HTML page with simulated streaming
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streaming Demo</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }
            #chat-container { max-width: 600px; margin: 0 auto; }
            #input-area { display: flex; gap: 10px; margin-bottom: 20px; }
            #user-input { flex: 1; padding: 10px; border-radius: 5px; border: none; }
            #send-btn { padding: 10px 20px; background: #4a90d9; color: white; border: none; border-radius: 5px; cursor: pointer; }
            #response-box { 
                background: #16213e; 
                padding: 20px; 
                border-radius: 10px; 
                min-height: 100px;
                white-space: pre-wrap;
            }
            .cursor { animation: blink 1s infinite; }
            @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>🤖 Streaming Chat Demo</h1>
            <div id="input-area">
                <input type="text" id="user-input" placeholder="Type a message...">
                <button id="send-btn">Send</button>
            </div>
            <div id="response-box"></div>
        </div>
        
        <script>
            const responses = {
                'hello': 'Hello! How can I assist you today? I am a helpful AI assistant ready to answer your questions.',
                'hi': 'Hi there! Great to meet you. How may I help you?',
                'default': 'Thank you for your message. I understand you said: "{input}". How can I help you further?'
            };
            
            function simulateStreaming(text, element) {
                element.textContent = '';
                let index = 0;
                
                function addChar() {
                    if (index < text.length) {
                        // Add 1-3 characters at a time to simulate token streaming
                        const chunkSize = Math.floor(Math.random() * 3) + 1;
                        element.textContent += text.slice(index, index + chunkSize);
                        index += chunkSize;
                        
                        // Variable delay to simulate realistic streaming
                        const delay = Math.random() * 50 + 20;
                        setTimeout(addChar, delay);
                    }
                }
                
                addChar();
            }
            
            document.getElementById('send-btn').addEventListener('click', function() {
                const input = document.getElementById('user-input').value.toLowerCase().trim();
                const responseBox = document.getElementById('response-box');
                
                let response = responses[input] || responses['default'].replace('{input}', input);
                simulateStreaming(response, responseBox);
            });
            
            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    document.getElementById('send-btn').click();
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Navigate to the data URL
    driver.get("data:text/html;charset=utf-8," + html_content.replace("#", "%23"))
    time.sleep(1)
    
    # Initialize our library components
    waiter = StreamWaiter()
    asserter = SemanticAssert()
    
    # Find elements
    input_box = driver.find_element(By.ID, "user-input")
    send_btn = driver.find_element(By.ID, "send-btn")
    response_locator = (By.ID, "response-box")
    
    # Enter message
    input_box.send_keys("Hello")
    
    logger.info("Sending 'Hello' message...")
    
    # Start monitoring and send message
    with LatencyMonitor(driver, response_locator) as monitor:
        send_btn.click()
        
        # Wait for streaming to complete
        response_element = waiter.wait_for_stream_end(
            driver,
            response_locator,
            silence_timeout=0.3,  # Short timeout for local demo
            timeout=10.0
        )
        
        response_text = response_element.text
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 DEMO RESULTS")
    print("=" * 60)
    print(f"\n📝 Response: {response_text}")
    print(f"\n⏱️  TTFT (Time-To-First-Token): {monitor.metrics.ttft_ms:.1f}ms" if monitor.metrics.ttft_ms else "\n⏱️  TTFT: N/A")
    print(f"⏱️  Total Latency: {monitor.metrics.total_ms:.1f}ms" if monitor.metrics.total_ms else "⏱️  Total: N/A")
    print(f"📈 Mutation Count: {monitor.metrics.token_count}")
    
    # Semantic assertion
    expected = "Hello, how can I help you today?"
    score = asserter.get_similarity_score(response_text, expected)
    print(f"\n🎯 Semantic Similarity Score: {score:.2%}")
    
    try:
        asserter.assert_similarity(
            response_text,
            expected,
            min_score=0.5
        )
        print("✅ Semantic assertion PASSED!")
    except AssertionError as e:
        print(f"❌ Semantic assertion failed: {e}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Demo script for selenium-chatbot-test library"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "huggingface", "auto"],
        default="local",
        help="Demo mode: 'local' for simulation, 'huggingface' for HF Chat, 'auto' to try both"
    )
    
    args = parser.parse_args()
    
    driver: Optional[webdriver.Chrome] = None
    
    try:
        logger.info("Starting selenium-chatbot-test demo...")
        logger.info("Creating Chrome WebDriver...")
        
        driver = create_driver(headless=args.headless)
        
        if args.mode == "local":
            demo_local_simulation(driver)
        elif args.mode == "huggingface":
            if demo_huggingface_chat(driver):
                demo_with_gradio_textbox(driver)
            else:
                logger.info("Falling back to local simulation...")
                demo_local_simulation(driver)
        else:  # auto
            if not demo_huggingface_chat(driver):
                demo_local_simulation(driver)
        
        logger.info("Demo completed successfully!")
        
    except WebDriverException as e:
        logger.error(f"WebDriver error: {e}")
        logger.error("Make sure Chrome is installed and chromedriver is in PATH")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
        
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Browser closed")
            except Exception:
                pass


if __name__ == "__main__":
    main()
