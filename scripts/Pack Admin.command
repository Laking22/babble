#!/bin/bash
# Double-click to open Babble Pack Admin (scripts/pack_admin.py).
cd "$(dirname "$0")/.." && exec .venv/bin/python scripts/pack_admin.py "$@"
