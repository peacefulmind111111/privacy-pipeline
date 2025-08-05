#!/usr/bin/env python
"""Convenience script to run all unit tests in the repository."""
import pytest

def main() -> int:
    """Execute the test suite via pytest."""
    return pytest.main(["-q"])

if __name__ == "__main__":
    raise SystemExit(main())
