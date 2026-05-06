#!/usr/bin/env python3
"""Compatibility entry point for the SAM 3.1 checkpoint downloader."""

import sys

from tools import download_model as _download_model

sys.modules[__name__] = _download_model

if __name__ == "__main__":
    raise SystemExit(_download_model.main())
