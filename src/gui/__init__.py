"""SETAI desktop control center (PySide6).

This package only depends on:
  - PySide6
  - src.gui_services (the safe service layer)

It must NEVER import from src.execution, src.training, src.backtest, or any
other backend module directly. The service layer is the only allowed seam.
"""
