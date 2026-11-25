#!/bin/bash
# Double-click to open the Babble Pack Uploader (scripts/pack_uploader.py).
cd "$(dirname "$0")/.." && exec .venv/bin/python scripts/pack_uploader.py "$@"
