"""Make `kernelized_backends` and the repo's `src` package importable from any pytest invocation."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
