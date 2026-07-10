"""Reconciliation and reporting layer.

`report.data` assembles every number the dashboard and the notebook
display, `report.registry` is the canonical record of every model this
project has run, and `report.figures` renders the shared matplotlib
figures. `python -m fifa_fantasy.report` writes the aggregate JSON and
the SVGs the web pages inline.
"""

from .data import assemble, official_recommendation  # noqa: F401
from .registry import MODEL_REGISTRY, NEGATIVE_RESULTS  # noqa: F401
