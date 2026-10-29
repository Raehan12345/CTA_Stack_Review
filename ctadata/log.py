"""Opt-in load tracing: set CTA_DEBUG=1 to see what each loader read."""
import os

DEBUG = bool(os.environ.get("CTA_DEBUG"))


def log(msg: str) -> None:
    if DEBUG:
        print(f"[ctadata] {msg}")
