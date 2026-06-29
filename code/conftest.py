"""Pytest configuration: make the project modules importable.

The package directory is named ``code``, which clashes with Python's standard
library ``code`` module. To avoid that clash, tests import the project modules
by their bare names (``config``, ``environment``, ...) and this conftest puts the
directory containing them at the front of ``sys.path``.
"""

import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
