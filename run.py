#!/usr/bin/env python3
"""Run the SinkVis server."""

import uvicorn


def main():
    """Start the SinkVis server."""
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

