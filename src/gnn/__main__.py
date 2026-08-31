"""Entry point for ``python -m src.gnn``.

Delegates to :func:`src.gnn.main` so the same smoke helper is callable
both as a module (``python -m src.gnn``) and as a function. Print format
is stable so downstream tooling can grep on it.
"""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
