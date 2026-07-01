"""Command handlers for the pidcast CLI verbs.

Each module owns the body of one verb (or verb group). ``cli.py`` wires these to
subparsers via ``set_defaults(func=...)`` and stays a thin parser/dispatch layer.
"""
