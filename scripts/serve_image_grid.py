#!/usr/bin/env python3
"""Serve the local image-grid composer on localhost."""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "tools" / "image_grid_composer"


class NoCacheRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve app files without retaining stale JavaScript in the browser."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the browser-based image-grid composer."
    )
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP port.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = functools.partial(
        NoCacheRequestHandler,
        directory=str(APP_DIR),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"Image Grid Composer: {url}")
    print("Press Ctrl-C to stop the server.")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Image Grid Composer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
