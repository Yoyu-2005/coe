"""
VECTA Web Dashboard Launcher.
Launches the FastAPI server and optionally opens the browser for the panel presentation.
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def open_browser(url: str, delay_s: float = 1.2):
    """Open user's default browser after server spins up."""
    time.sleep(delay_s)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 70)
    print(" VECTA // Intelligent ANN Auto-Tuning Kernel - Presentation Launcher")
    print(f" Dashboard URL: {url}")
    print("=" * 70)

    # Launch browser in a background daemon thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Start FastAPI server
    from vecta.api.app import app
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
