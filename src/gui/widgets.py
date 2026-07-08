"""Small reusable widgets for the desktop control center."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui_services.state import RuntimeOverrides


# ── Shared color palette ────────────────────────────────────────────
# The whole GUI pulls from these constants so accent/value coloring
# stays consistent across tabs.
COLOR_BG            = "#0a0e14"
COLOR_PANEL_BG      = "#11161c"
COLOR_PANEL_BORDER  = "#1f2937"
COLOR_TEXT          = "#e2e8f0"
COLOR_MUTED         = "#94a3b8"
COLOR_ACCENT        = "#4bd0d8"   # cyan — accents / card border
COLOR_POSITIVE      = "#4ade80"   # green — gains, running, ok
COLOR_NEGATIVE      = "#f87171"   # red — loss, stopped, error
COLOR_WARN          = "#fbbf24"   # amber — transitional
COLOR_INFO          = "#60a5fa"   # blue — informational highlights


_BADGE_COLORS = {
    "stopped":  "#475569",
    "starting": COLOR_WARN,
    "running":  COLOR_POSITIVE,
    "stopping": COLOR_WARN,
    "error":    COLOR_NEGATIVE,
    "ok":       COLOR_POSITIVE,
    "warn":     COLOR_WARN,
    "off":      "#475569",
}


class StatusBadge(QLabel):
    def __init__(self, text: str = "stopped", parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.setMaximumHeight(30)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.setFont(font)
        self.set_state("stopped")

    def set_state(self, state: str, label: str | None = None) -> None:
        color = _BADGE_COLORS.get(state.lower(), "#475569")
        text = label if label is not None else state.upper()
        self.setText(text)
        # font-size MUST accompany font-weight in QSS — Qt's stylesheet
        # engine does not inherit font-size from setFont() above; without
        # an explicit size here Qt falls back to the platform default
        # font (which has pointSize() == -1 on Windows) and emits a
        # "QFont::setPointSize: Point size <= 0 (-1)" warning.
        self.setStyleSheet(
            f"background-color: {color}; color: #0a0e14; "
            f"border-radius: 4px; padding: 5px 12px; "
            f"letter-spacing: 1px; font-size: 10px; font-weight: 700;"
        )


class MetricCard(QFrame):
    """A titled metric card with an accent border and optional value color.

    The card uses a cyan left-edge accent by default; pass ``accent`` to
    override.  ``set_value(...)`` accepts an optional ``color`` kwarg for
    positive/negative tinting without changing any existing call sites.
    """

    def __init__(
        self,
        title: str,
        value: str = "—",
        *,
        accent: str = COLOR_ACCENT,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._accent = accent
        self.setStyleSheet(
            f"MetricCard {{ background-color: {COLOR_PANEL_BG}; "
            f"border: 1px solid {COLOR_PANEL_BORDER}; "
            f"border-left: 3px solid {accent}; "
            f"border-radius: 5px; padding: 6px; }}"
        )
        # Preferred (not Fixed) height lets the card grow vertically when
        # the value is a multi-line string (e.g. a wrapped symbol list).
        # Otherwise long values get clipped at the right edge because the
        # inner QLabel's word-wrap can't add a second line.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(2)

        self._title = QLabel(title.upper())
        self._title.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 10px; "
            f"letter-spacing: 1px; font-weight: 600; border: 0; background: transparent;"
        )
        self._value = QLabel(value)
        self._value_font_style = (
            "font-size: 18px; font-weight: 700; "
            "font-family: 'Cascadia Mono', Consolas, monospace; "
            "border: 0; background: transparent;"
        )
        self._value.setStyleSheet(f"color: {COLOR_TEXT}; {self._value_font_style}")
        self._value.setWordWrap(True)
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: str, *, color: str | None = None) -> None:
        self._value.setText(value)
        c = color or COLOR_TEXT
        self._value.setStyleSheet(f"color: {c}; {self._value_font_style}")


class RuntimeOverridesPanel(QGroupBox):
    """Runtime toggles + memory caps. Post-Great-Simplification surface.

    Packs 5 on/off checkboxes and 1 numeric spin-box into a QGroupBox:
      - Frozen brain (runtime.allow_online_model_updates inverted)
      - Memory enabled (memory.enabled)
      - Memory updates (runtime.allow_online_memory_updates)
      - Platt calibration (training.platt_calibration_enabled)
      - Risk heads / EV gate (model.separate_risk_heads)
      - Max memory items (memory.max_items)

    Dropped in Simplification (features no longer exist):
    fusion, topology growth, modern hopfield, multi-timescale, RAN merge,
    regime hysteresis, market priors, counterfactual, value-aware decay,
    ESOM regime labels, max dynamic units, ESOM edge lifetime, ESOM neighbor K.
    """

    _ITEM_RANGE = (100, 500_000)

    def __init__(self, title: str = "Runtime Toggles", parent=None) -> None:
        super().__init__(title, parent)
        self._build_ui()
        self._set_all_to_config_defaults()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 14, 12, 10)
        outer.setSpacing(8)

        # --- Runtime toggles ---
        self.chk_frozen_brain = QCheckBox("Frozen brain")
        self.chk_frozen_brain.setToolTip(
            "When ON, model weights are frozen at runtime (never updated).\n"
            "Default: ON (recommended for production paper trading)."
        )
        self.chk_memory = QCheckBox("Memory enabled")
        self.chk_memory.setToolTip(
            "Enable the passive memory scaffold (store encoded states + outcomes,\n"
            "expose kNN lookup in diagnostics). Does not influence trade decisions."
        )
        self.chk_memory_updates = QCheckBox("Memory updates")
        self.chk_memory_updates.setToolTip(
            "Allow the memory bank to accumulate new items at runtime.\n"
            "Disable to freeze the bank at startup."
        )
        self.chk_platt = QCheckBox("Platt calibration")
        self.chk_platt.setToolTip(
            "Fit Platt scaling on validation confidences after training so the\n"
            "EV gate sees a calibrated P(profitable) instead of raw sigmoid output."
        )
        self.chk_separate_risk_heads = QCheckBox("Risk heads (EV gate)")
        self.chk_separate_risk_heads.setToolTip(
            "Train dedicated MFE/MAE heads that feed the EV gate as\n"
            "expected_gain_bps / expected_loss_bps. Required for the EV gate\n"
            "to see real signal — with this off, the gate correctly refuses\n"
            "every trade (unknown edge → negative EV)."
        )

        toggle_grid = QGridLayout()
        toggle_grid.setHorizontalSpacing(28)
        toggle_grid.setVerticalSpacing(4)
        toggle_grid.addWidget(self.chk_frozen_brain, 0, 0)
        toggle_grid.addWidget(self.chk_memory, 0, 1)
        toggle_grid.addWidget(self.chk_memory_updates, 0, 2)
        toggle_grid.addWidget(self.chk_platt, 1, 0)
        toggle_grid.addWidget(self.chk_separate_risk_heads, 1, 1)
        outer.addLayout(toggle_grid)

        # Thin separator.
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {COLOR_PANEL_BORDER}; background: {COLOR_PANEL_BORDER};")
        line.setMaximumHeight(1)
        outer.addWidget(line)

        # --- Numeric caps: only max_items survives the simplification ---
        self.spin_max_items = self._make_spin(*self._ITEM_RANGE, 50_000)
        self.spin_max_items.setToolTip(
            "Maximum items in the passive memory bank (memory.max_items).\n"
            "Older items get evicted FIFO when the cap is reached."
        )

        cap_grid = QGridLayout()
        cap_grid.setHorizontalSpacing(12)
        cap_grid.setVerticalSpacing(4)
        cap_grid.addWidget(self._cap_label("Max memory items"), 0, 0)
        cap_grid.addWidget(self.spin_max_items, 0, 1)
        cap_grid.setColumnStretch(1, 1)
        outer.addLayout(cap_grid)

        # Reset-to-config defaults button.
        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        from PySide6.QtWidgets import QPushButton
        self.btn_reset = QPushButton("Reset to config defaults")
        self.btn_reset.setToolTip(
            "Restore all toggles and caps to the currently-selected YAML stack."
        )
        self.btn_reset.clicked.connect(self._on_reset)
        reset_row.addWidget(self.btn_reset)
        outer.addLayout(reset_row)

    def _cap_label(self, text: str) -> QLabel:
        lbl = QLabel(text + ":")
        lbl.setStyleSheet(f"color: {COLOR_MUTED}; border: 0; background: transparent;")
        return lbl

    @staticmethod
    def _make_spin(lo: int, hi: int, default: int) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        sp.setSingleStep(1)
        sp.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        sp.setMinimumWidth(100)
        return sp

    # ------------------------------------------------------------------
    # public API (consumed by RuntimeTab / BacktestTab)
    # ------------------------------------------------------------------
    def load_defaults(
        self,
        cfg_summary: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra = extra or {}
        self.chk_frozen_brain.setChecked(bool(cfg_summary.get("frozen_brain", True)))
        self.chk_memory.setChecked(bool(cfg_summary.get("memory_enabled", False)))
        self.chk_memory_updates.setChecked(
            bool(cfg_summary.get("memory_updates_allowed", False))
        )
        self.chk_platt.setChecked(
            bool(cfg_summary.get("platt_calibration_enabled", False))
        )
        self.chk_separate_risk_heads.setChecked(
            bool(cfg_summary.get("separate_risk_heads", False))
        )
        if "max_items" in extra:
            self.spin_max_items.setValue(self._clamp(extra["max_items"], self._ITEM_RANGE))
        self._last_defaults = (dict(cfg_summary), dict(extra))

    def to_overrides(self) -> RuntimeOverrides:
        return RuntimeOverrides(
            frozen_brain=self.chk_frozen_brain.isChecked(),
            memory_enabled=self.chk_memory.isChecked(),
            memory_updates_enabled=self.chk_memory_updates.isChecked(),
            platt_calibration_enabled=self.chk_platt.isChecked(),
            separate_risk_heads=self.chk_separate_risk_heads.isChecked(),
            max_memory_items=self.spin_max_items.value(),
        )

    def set_editable(self, editable: bool) -> None:
        for w in (
            self.chk_frozen_brain,
            self.chk_memory,
            self.chk_memory_updates,
            self.chk_platt,
            self.chk_separate_risk_heads,
            self.spin_max_items,
            self.btn_reset,
        ):
            w.setEnabled(editable)

    def set_memory_controls_relevant(self, relevant: bool) -> None:
        """Used by BacktestTab: grey out memory-related controls when the
        selected policy is 'baseline' (which ignores memory entirely)."""
        tooltip = "Baseline policy ignores memory-related settings." if not relevant else ""
        for w in (self.chk_memory, self.chk_memory_updates, self.spin_max_items):
            w.setEnabled(relevant)
            if tooltip:
                w.setToolTip(tooltip)

    # ------------------------------------------------------------------
    def _set_all_to_config_defaults(self) -> None:
        # Defaults matching configs/default.yaml (Phases 1–9).
        self.chk_frozen_brain.setChecked(True)
        self.chk_memory.setChecked(True)
        self.chk_memory_updates.setChecked(True)
        self.chk_platt.setChecked(True)
        self.chk_separate_risk_heads.setChecked(True)
        self.spin_max_items.setValue(50_000)
        self._last_defaults: tuple[dict[str, Any], dict[str, Any]] = ({}, {})

    def _on_reset(self) -> None:
        cfg_summary, extra = self._last_defaults
        if cfg_summary:
            self.load_defaults(cfg_summary, extra=extra)
        else:
            self._set_all_to_config_defaults()

    @staticmethod
    def _clamp(val: Any, rng: tuple[int, int]) -> int:
        try:
            v = int(val)
        except (TypeError, ValueError):
            return rng[0]
        return max(rng[0], min(rng[1], v))


class HeaderMetric(QWidget):
    """Compact inline metric for the persistent top header bar.

    Smaller than ``MetricCard`` — designed to pack several values into a
    single horizontal strip above the tab widget without wasting space.
    """

    def __init__(self, title: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(0)

        self._title = QLabel(title.upper())
        self._title.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 9px; "
            f"letter-spacing: 1px; font-weight: 600; "
            f"border: 0; background: transparent;"
        )
        self._value = QLabel(value)
        self._value_font_style = (
            "font-size: 15px; font-weight: 700; "
            "font-family: 'Cascadia Mono', Consolas, monospace; "
            "border: 0; background: transparent;"
        )
        self._value.setStyleSheet(f"color: {COLOR_TEXT}; {self._value_font_style}")
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: str, *, color: str | None = None) -> None:
        self._value.setText(value)
        c = color or COLOR_TEXT
        self._value.setStyleSheet(f"color: {c}; {self._value_font_style}")
