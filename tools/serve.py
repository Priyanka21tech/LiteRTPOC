"""
Small static file server for local browser testing.

This avoids introducing Python web dependencies while still letting the
project run from inside the requested virtual environment.
"""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os


HOST = "127.0.0.1"
PORT = 8000
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    server = ThreadingHTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    print(f"Serving {ROOT} at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
