"""Explicit setup utility for the LangGraph PostgreSQL checkpoint schema."""

from __future__ import annotations

import argparse

from app.config import Config
from app.graph.checkpointer import close_postgres_checkpointer, get_configured_checkpointer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup",), help="Initialize/upgrade package-managed checkpoint tables")
    parser.parse_args()
    if Config.CHECKPOINT_BACKEND != "postgres":
        parser.error("Set CHECKPOINT_BACKEND=postgres before running checkpoint setup.")
    try:
        get_configured_checkpointer()  # Opens the managed pool and calls saver.setup().
        print("LangGraph PostgreSQL checkpoint schema is initialized.")
    finally:
        close_postgres_checkpointer()


if __name__ == "__main__":
    main()
