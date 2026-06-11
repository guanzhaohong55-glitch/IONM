from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

QT_API = ""

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    QT_API = "PySide6"
except (ImportError, ModuleNotFoundError):
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        QT_API = "PyQt6"
    except ModuleNotFoundError as exc:  # pragma: no cover - friendly startup message
        missing = exc.name or "Qt dependency"
        print(f"Missing dependency: {missing}")
        print("Install with: python -m pip install -r requirements.txt")
        raise

import pyqtgraph as pg


USER_ROLE = QtCore.Qt.ItemDataRole.UserRole
CHECKED = QtCore.Qt.CheckState.Checked
NO_EDIT_TRIGGERS = QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
SELECT_ROWS = QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
SINGLE_SELECTION = QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
TEXT_SELECTABLE = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
ITEM_IS_USER_CHECKABLE = QtCore.Qt.ItemFlag.ItemIsUserCheckable
HORIZONTAL = QtCore.Qt.Orientation.Horizontal
VERTICAL = QtCore.Qt.Orientation.Vertical
DOT_LINE = QtCore.Qt.PenStyle.DotLine
DASH_LINE = QtCore.Qt.PenStyle.DashLine


class FixedUnitAxis(pg.AxisItem):
    def __init__(self, *args: Any, suffix: str = "", precision: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.suffix = suffix
        self.precision = precision
        if hasattr(self, "enableAutoSIPrefix"):
            self.enableAutoSIPrefix(False)

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        labels: list[str] = []
        for value in values:
            raw = float(value) * float(scale)
            if self.precision is not None:
                text = f"{raw:.{self.precision}f}"
            elif abs(raw) >= 100:
                text = f"{raw:.0f}"
            elif abs(raw) >= 10:
                text = f"{raw:.1f}"
            elif spacing < 0.01:
                text = f"{raw:.3f}"
            elif spacing < 0.1:
                text = f"{raw:.2f}"
            else:
                text = f"{raw:.1f}"
            labels.append(f"{text}{self.suffix}")
        return labels


APP_TITLE = "IONM Waveform Console"
WINDOW_BG = "#071114"
PANEL_BG = "#0d1a1e"
PANEL_2 = "#102229"
TEXT = "#d8eef0"
MUTED = "#86a2a7"
GRID = "#27434a"
ACCENT = "#18d7d0"
ACCENT_2 = "#f2c14e"
ALERT = "#ff6b66"
GOOD = "#53d769"


def timestamp_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw > 1e14:
        return raw / 1_000_000.0
    if raw > 1e11:
        return raw / 1_000.0
    return raw


def exported_datetime(value: Any) -> datetime | None:
    seconds = timestamp_seconds(value)
    if seconds is None:
        return None
    # The exporter stores wall-clock timestamps as Unix microseconds. Displaying
    # the UTC interpretation matches the operative clock in the exported notes.
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def format_clock(value: Any) -> str:
    dt = exported_datetime(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "-"


def format_time(value: Any) -> str:
    dt = exported_datetime(value)
    return dt.strftime("%H:%M:%S") if dt else "-"


def format_elapsed(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def shorten(text: Any, limit: int = 54) -> str:
    value = "" if text is None else str(text).replace("\r", " ").replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "..."


def patient_name(case_json: dict[str, Any]) -> str:
    first = case_json.get("PatientFirstName") or ""
    last = case_json.get("PatientLastName") or ""
    return f"{first}{last}".strip() or "Unknown"


def path_label(path: Path) -> str:
    parts = [p for p in path.parts if p]
    if "阳性" in parts:
        return "阳性"
    if "正常" in parts:
        return "正常"
    return "未分类"


def event_color(event_type: str, deleted: bool = False) -> str:
    if deleted:
        return "#5d6f73"
    if event_type == "User Event":
        return ACCENT_2
    if event_type == "Impedance":
        return "#53a8ff"
    if "Connection" in event_type:
        return "#789198"
    return "#b38cff"


def trace_unit_for_mode(mode_type: str, traces: list["TraceRecord"]) -> tuple[str, float]:
    upper = (mode_type or "").upper()
    if upper in {"MEP", "EMG", "TEMG", "TOF"}:
        return "mV", 1_000.0
    peak = 0.0
    for trace in traces[:16]:
        if trace.values_v.size:
            peak = max(peak, float(np.nanmax(np.abs(trace.values_v))))
    if peak >= 1e-3:
        return "mV", 1_000.0
    return "μV", 1_000_000.0


@dataclass
class CursorRecord:
    label: str
    position_s: float | None
    voltage_v: float | None


@dataclass
class ChannelRecord:
    name: str
    gain: float | None = None
    low_cut: float | None = None
    high_cut: float | None = None
    amplifier: str = ""


@dataclass
class TraceRecord:
    timestamp: Any
    channel: ChannelRecord
    sweep_s: float
    scalar: float
    values_v: np.ndarray
    cursors: list[CursorRecord] = field(default_factory=list)

    @property
    def x_ms(self) -> np.ndarray:
        if self.values_v.size == 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, self.sweep_s * 1000.0, self.values_v.size, endpoint=False)


@dataclass
class TrialRecord:
    timestamp: Any
    trial_number: int
    stimulus: dict[str, Any]
    traces: list[TraceRecord]
    baseline: bool = False
    highlighted: bool = False
    requested_rep_rate: float | None = None
    actual_rep_rate: float | None = None
    ep_average_count: int | None = None
    ep_reject_count: int | None = None


@dataclass
class ModeRecord:
    name: str
    mode_type: str
    trials: list[TrialRecord]

    @property
    def channels(self) -> list[str]:
        seen: list[str] = []
        for trial in self.trials:
            for trace in trial.traces:
                if trace.channel.name not in seen:
                    seen.append(trace.channel.name)
        return seen

    @property
    def all_traces(self) -> list[TraceRecord]:
        return [trace for trial in self.trials for trace in trial.traces]


@dataclass
class EventRecord:
    timestamp: Any
    author: str
    event_type: str
    message: str
    deleted: bool = False


@dataclass
class CaseRecord:
    path: Path
    result: str
    patient: str
    patient_id: str
    gender: str
    start_date: Any
    procedure: str
    state: str
    fields: dict[str, str]
    hardware: str
    modes: list[ModeRecord]
    events: list[EventRecord]

    @property
    def start_seconds(self) -> float | None:
        return timestamp_seconds(self.start_date)


@dataclass
class CaseSummary:
    path: Path
    result: str
    patient: str
    patient_id: str
    start_date: Any
    procedure: str
    mode_count: int
    trial_count: int
    event_count: int
    file_size: int


def make_field_map(case_json: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in case_json.get("CaseFields") or []:
        name = str(item.get("Name") or "")
        if name:
            fields[name] = "" if item.get("Value") is None else str(item.get("Value"))
    return fields


def parse_channel(raw: dict[str, Any]) -> ChannelRecord:
    active = raw.get("ActiveBodySiteName")
    reference = raw.get("ReferenceBodySiteName")
    fallback = f"{active} - {reference}" if active and reference and active != reference else active
    return ChannelRecord(
        name=str(raw.get("Name") or fallback or "Channel"),
        gain=raw.get("Gain"),
        low_cut=raw.get("LowCut"),
        high_cut=raw.get("HighCut"),
        amplifier=f"{raw.get('AmplifierType') or ''} {raw.get('AmplifierNumber') or ''}".strip(),
    )


def parse_trace(raw: dict[str, Any]) -> TraceRecord:
    scalar = float(raw.get("TraceDataScalar") or 1.0)
    data = np.asarray(raw.get("TraceData") or [], dtype=np.float64) * scalar
    cursors: list[CursorRecord] = []
    for cursor in raw.get("Cursors") or []:
        cursors.append(
            CursorRecord(
                label=str(cursor.get("Label") or ""),
                position_s=cursor.get("Position"),
                voltage_v=cursor.get("Voltage"),
            )
        )
    return TraceRecord(
        timestamp=raw.get("Timestamp"),
        channel=parse_channel(raw.get("Channel") or {}),
        sweep_s=float(raw.get("Sweep") or 0.1),
        scalar=scalar,
        values_v=data,
        cursors=cursors,
    )


def parse_trial(raw: dict[str, Any]) -> TrialRecord:
    return TrialRecord(
        timestamp=raw.get("Timestamp"),
        trial_number=int(raw.get("TrialNumber") or 0),
        stimulus=raw.get("Stimulus") or {},
        traces=[parse_trace(trace) for trace in raw.get("Traces") or []],
        baseline=bool(raw.get("Baseline")),
        highlighted=bool(raw.get("Highlighted")),
        requested_rep_rate=raw.get("RequestedRepRate"),
        actual_rep_rate=raw.get("ActualRepRate"),
        ep_average_count=raw.get("EPAverageCount"),
        ep_reject_count=raw.get("EPRejectCount"),
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("not an IONM case export")
    return data


def summarize_case(path: Path) -> CaseSummary:
    data = load_json(path)
    cases = data.get("Cases")
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
        raise ValueError("missing Cases")
    case_json = cases[0]
    modes = case_json.get("Modes") or []
    trial_count = sum(len(mode.get("Trials") or []) for mode in modes)
    return CaseSummary(
        path=path,
        result=path_label(path),
        patient=patient_name(case_json),
        patient_id=str(case_json.get("PatientID") or ""),
        start_date=case_json.get("StartDate"),
        procedure=str(case_json.get("ProcedureName") or ""),
        mode_count=len(modes),
        trial_count=trial_count,
        event_count=len(case_json.get("Events") or []),
        file_size=path.stat().st_size,
    )


def load_case(path: Path) -> CaseRecord:
    data = load_json(path)
    cases = data.get("Cases")
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
        raise ValueError("missing Cases")
    case_json = cases[0]
    modes: list[ModeRecord] = []
    for mode_json in case_json.get("Modes") or []:
        trials = [parse_trial(trial) for trial in mode_json.get("Trials") or []]
        modes.append(
            ModeRecord(
                name=str(mode_json.get("Name") or "Mode"),
                mode_type=str(mode_json.get("Type") or ""),
                trials=trials,
            )
        )
    events = [
        EventRecord(
            timestamp=event.get("Timestamp"),
            author=str(event.get("Author") or ""),
            event_type=str(event.get("Type") or ""),
            message=str(event.get("Message") or ""),
            deleted=bool(event.get("Deleted")),
        )
        for event in case_json.get("Events") or []
    ]
    events.sort(key=lambda event: timestamp_seconds(event.timestamp) or 0)
    return CaseRecord(
        path=path,
        result=path_label(path),
        patient=patient_name(case_json),
        patient_id=str(case_json.get("PatientID") or ""),
        gender=str(case_json.get("PatientGender") or ""),
        start_date=case_json.get("StartDate"),
        procedure=str(case_json.get("ProcedureName") or ""),
        state=str(case_json.get("State") or ""),
        fields=make_field_map(case_json),
        hardware=str(case_json.get("HardwareType") or ""),
        modes=modes,
        events=events,
    )


def stimulus_summary(stimulus: dict[str, Any]) -> str:
    if not stimulus:
        return "-"
    parts: list[str] = []
    intensity = stimulus.get("Intensity")
    units = stimulus.get("IntensityUnits") or ""
    if intensity is not None:
        parts.append(f"{intensity:g}{units}")
    pulse_count = stimulus.get("PulseCount")
    if pulse_count is not None:
        parts.append(f"{pulse_count} pulses")
    width = stimulus.get("PulseWidth")
    if width is not None:
        parts.append(f"{float(width) * 1000:g} ms")
    outputs = stimulus.get("Outputs") or []
    if outputs:
        usages = [str(output.get("Usage") or "") for output in outputs if output.get("Usage")]
        if usages:
            parts.append("/".join(usages))
    sensed_current = stimulus.get("SensedCurrent")
    if sensed_current is not None:
        parts.append(f"sensed {float(sensed_current) * 1000:.1f} mA")
    sensed_voltage = stimulus.get("SensedVoltage")
    if sensed_voltage is not None:
        parts.append(f"{float(sensed_voltage):.1f} V")
    return " | ".join(parts) or "-"


class StatusPill(QtWidgets.QLabel):
    def __init__(self, text: str = "", color: str = ACCENT, parent: QtWidgets.QWidget | None = None):
        super().__init__(text, parent)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(
            f"""
            QLabel {{
                background: {color};
                color: #061114;
                border-radius: 9px;
                padding: 3px 10px;
                font-weight: 700;
            }}
            """
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.case: CaseRecord | None = None
        self.summaries: list[CaseSummary] = []
        self.filtered_summaries: list[CaseSummary] = []
        self.current_mode: ModeRecord | None = None
        self.current_trial_index = 0
        self.selected_event: EventRecord | None = None
        self._updating = False
        self._channel_updating = False
        self.wave_plots: list[pg.PlotItem] = []

        self.setWindowTitle(APP_TITLE)
        self.resize(1520, 920)
        self.setMinimumSize(1180, 720)
        self._build_ui()
        self._apply_theme()
        self.scan_cases()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self._build_top_bar(root_layout)

        splitter = QtWidgets.QSplitter(HORIZONTAL)
        splitter.setHandleWidth(7)
        root_layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([310, 880, 360])

    def _build_top_bar(self, root_layout: QtWidgets.QVBoxLayout) -> None:
        bar = QtWidgets.QFrame()
        bar.setObjectName("topBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("IONM Waveform Console")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        self.result_pill = StatusPill("No case", "#6b7f86")
        layout.addWidget(self.result_pill)

        self.case_title = QtWidgets.QLabel("Select a JSON case")
        self.case_title.setObjectName("caseTitle")
        layout.addWidget(self.case_title, 1)

        self.clock_label = QtWidgets.QLabel("")
        self.clock_label.setObjectName("muted")
        layout.addWidget(self.clock_label)

        root_layout.addWidget(bar)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Cases")
        label.setObjectName("sectionTitle")
        row.addWidget(label)
        row.addStretch()
        refresh = QtWidgets.QToolButton()
        refresh.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        refresh.setToolTip("Rescan JSON files")
        refresh.clicked.connect(self.scan_cases)
        row.addWidget(refresh)
        open_button = QtWidgets.QToolButton()
        open_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        open_button.setToolTip("Open a JSON file")
        open_button.clicked.connect(self.open_json_dialog)
        row.addWidget(open_button)
        layout.addLayout(row)

        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search patient, ID, date, path...")
        self.search_box.textChanged.connect(self.filter_cases)
        layout.addWidget(self.search_box)

        self.case_tree = QtWidgets.QTreeWidget()
        self.case_tree.setColumnCount(4)
        self.case_tree.setHeaderLabels(["Patient", "Result", "Trials", "Start"])
        self.case_tree.setRootIsDecorated(False)
        self.case_tree.setAlternatingRowColors(True)
        self.case_tree.itemSelectionChanged.connect(self.case_selection_changed)
        layout.addWidget(self.case_tree, 1)

        self.scan_label = QtWidgets.QLabel("")
        self.scan_label.setObjectName("muted")
        self.scan_label.setWordWrap(True)
        layout.addWidget(self.scan_label)
        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("centerPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        info = QtWidgets.QFrame()
        info.setObjectName("panel")
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(8)

        self.meta_line = QtWidgets.QLabel("Load a case to inspect waveform modes.")
        self.meta_line.setObjectName("metaLine")
        self.meta_line.setWordWrap(True)
        info_layout.addWidget(self.meta_line)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        controls.addWidget(QtWidgets.QLabel("Mode"))
        controls.addWidget(self.mode_combo, 2)

        prev_btn = QtWidgets.QToolButton()
        prev_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
        prev_btn.setToolTip("Previous trial")
        prev_btn.clicked.connect(lambda: self.step_trial(-1))
        controls.addWidget(prev_btn)

        self.trial_slider = QtWidgets.QSlider(HORIZONTAL)
        self.trial_slider.valueChanged.connect(self.trial_slider_changed)
        controls.addWidget(self.trial_slider, 4)

        next_btn = QtWidgets.QToolButton()
        next_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        next_btn.setToolTip("Next trial")
        next_btn.clicked.connect(lambda: self.step_trial(1))
        controls.addWidget(next_btn)

        self.trial_spin = QtWidgets.QSpinBox()
        self.trial_spin.setMinimum(1)
        self.trial_spin.valueChanged.connect(self.trial_spin_changed)
        controls.addWidget(self.trial_spin)

        self.baseline_check = QtWidgets.QCheckBox("Baseline")
        self.baseline_check.setChecked(True)
        self.baseline_check.setToolTip("Overlay the first baseline trial when available")
        self.baseline_check.stateChanged.connect(self.refresh_waveforms)
        controls.addWidget(self.baseline_check)

        self.cursor_check = QtWidgets.QCheckBox("Cursors")
        self.cursor_check.setChecked(True)
        self.cursor_check.stateChanged.connect(self.refresh_waveforms)
        controls.addWidget(self.cursor_check)

        info_layout.addLayout(controls)
        layout.addWidget(info)

        vertical = QtWidgets.QSplitter(VERTICAL)
        vertical.setHandleWidth(7)
        layout.addWidget(vertical, 1)

        self.wave_layout = pg.GraphicsLayoutWidget()
        self.wave_layout.setBackground(WINDOW_BG)
        vertical.addWidget(self.wave_layout)

        self.timeline_plot = pg.PlotWidget(axisItems={"bottom": FixedUnitAxis("bottom", suffix="h")})
        self.timeline_plot.setBackground(WINDOW_BG)
        self.timeline_plot.setMinimumHeight(160)
        vertical.addWidget(self.timeline_plot)
        vertical.setSizes([650, 170])

        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        stats_title = QtWidgets.QLabel("Monitor")
        stats_title.setObjectName("sectionTitle")
        layout.addWidget(stats_title)

        self.stats_grid = QtWidgets.QGridLayout()
        self.stats_grid.setHorizontalSpacing(10)
        self.stats_grid.setVerticalSpacing(6)
        layout.addLayout(self.stats_grid)
        self.stat_labels: dict[str, QtWidgets.QLabel] = {}
        for row, key in enumerate(["Trial", "Clock", "Elapsed", "Sweep", "Scale", "Stimulus", "Peak"]):
            key_label = QtWidgets.QLabel(key)
            key_label.setObjectName("muted")
            value = QtWidgets.QLabel("-")
            value.setWordWrap(True)
            value.setTextInteractionFlags(TEXT_SELECTABLE)
            self.stats_grid.addWidget(key_label, row, 0)
            self.stats_grid.addWidget(value, row, 1)
            self.stat_labels[key] = value

        layout.addSpacing(4)
        channel_title = QtWidgets.QLabel("Channels")
        channel_title.setObjectName("sectionTitle")
        layout.addWidget(channel_title)

        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMaximumHeight(185)
        self.channel_list.itemChanged.connect(self.channel_filter_changed)
        layout.addWidget(self.channel_list)

        event_controls = QtWidgets.QHBoxLayout()
        event_title = QtWidgets.QLabel("Events")
        event_title.setObjectName("sectionTitle")
        event_controls.addWidget(event_title)
        event_controls.addStretch()
        self.event_filter = QtWidgets.QComboBox()
        self.event_filter.addItems(["All", "User Event", "Impedance", "Connections"])
        self.event_filter.currentIndexChanged.connect(self.refresh_events)
        event_controls.addWidget(self.event_filter)
        layout.addLayout(event_controls)

        self.deleted_check = QtWidgets.QCheckBox("Deleted")
        self.deleted_check.stateChanged.connect(self.refresh_events)
        layout.addWidget(self.deleted_check)

        self.event_table = QtWidgets.QTableWidget(0, 4)
        self.event_table.setHorizontalHeaderLabels(["Time", "+Time", "Type", "Message"])
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.event_table.setSelectionBehavior(SELECT_ROWS)
        self.event_table.setSelectionMode(SINGLE_SELECTION)
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.itemSelectionChanged.connect(self.event_selection_changed)
        layout.addWidget(self.event_table, 1)

        nearby_title = QtWidgets.QLabel("Near Current Trial")
        nearby_title.setObjectName("sectionTitle")
        layout.addWidget(nearby_title)
        self.nearby_list = QtWidgets.QListWidget()
        self.nearby_list.setMaximumHeight(130)
        layout.addWidget(self.nearby_list)
        return panel

    def _apply_theme(self) -> None:
        pg.setConfigOptions(antialias=True, foreground=TEXT)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {WINDOW_BG};
                color: {TEXT};
                font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
                font-size: 12px;
            }}
            QFrame#panel, QFrame#topBar {{
                background: {PANEL_BG};
                border: 1px solid #183138;
                border-radius: 8px;
            }}
            QFrame#centerPanel {{
                background: transparent;
            }}
            QLabel#appTitle {{
                color: #ffffff;
                font-size: 18px;
                font-weight: 800;
            }}
            QLabel#caseTitle {{
                color: {TEXT};
                font-size: 15px;
                font-weight: 700;
            }}
            QLabel#sectionTitle {{
                color: #f4fbfb;
                font-weight: 800;
                font-size: 13px;
            }}
            QLabel#muted {{
                color: {MUTED};
            }}
            QLabel#metaLine {{
                color: {TEXT};
                font-size: 13px;
            }}
            QLineEdit, QComboBox, QSpinBox {{
                background: {PANEL_2};
                border: 1px solid #24434b;
                border-radius: 5px;
                color: {TEXT};
                padding: 5px 7px;
                selection-background-color: {ACCENT};
            }}
            QTreeWidget, QListWidget, QTableWidget {{
                background: #09171b;
                alternate-background-color: #0d1d22;
                border: 1px solid #1b343b;
                border-radius: 6px;
                color: {TEXT};
                gridline-color: #183138;
                selection-background-color: #174e55;
                selection-color: #ffffff;
            }}
            QHeaderView::section {{
                background: #13262d;
                color: #cfecef;
                border: 0;
                padding: 5px;
                font-weight: 700;
            }}
            QToolButton {{
                background: #13282f;
                border: 1px solid #284a53;
                border-radius: 5px;
                padding: 5px;
            }}
            QToolButton:hover {{
                background: #1a3840;
                border-color: {ACCENT};
            }}
            QCheckBox {{
                color: {TEXT};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
            }}
            QSlider::groove:horizontal {{
                height: 5px;
                background: #1c343c;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                border: 1px solid #9ff8f4;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSplitter::handle {{
                background: #0a1518;
            }}
            """
        )

    def scan_cases(self) -> None:
        self.summaries = []
        paths = [
            path
            for path in sorted(self.root.rglob("*.json"))
            if ".venv" not in path.parts and "site-packages" not in path.parts
        ]
        for path in paths:
            try:
                self.summaries.append(summarize_case(path))
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
        self.filter_cases()

    def filter_cases(self) -> None:
        needle = self.search_box.text().strip().lower()
        if needle:
            self.filtered_summaries = [
                summary
                for summary in self.summaries
                if needle
                in " ".join(
                    [
                        summary.patient,
                        summary.patient_id,
                        summary.result,
                        format_clock(summary.start_date),
                        str(summary.path),
                    ]
                ).lower()
            ]
        else:
            self.filtered_summaries = list(self.summaries)
        self.populate_case_tree()

    def populate_case_tree(self) -> None:
        self.case_tree.clear()
        for summary in self.filtered_summaries:
            item = QtWidgets.QTreeWidgetItem(
                [
                    summary.patient or summary.path.stem,
                    summary.result,
                    str(summary.trial_count),
                    format_clock(summary.start_date).split(" ")[0],
                ]
            )
            item.setData(0, USER_ROLE, summary.path)
            item.setToolTip(0, str(summary.path))
            if summary.result == "阳性":
                item.setForeground(1, QtGui.QBrush(QtGui.QColor(ALERT)))
            elif summary.result == "正常":
                item.setForeground(1, QtGui.QBrush(QtGui.QColor(GOOD)))
            self.case_tree.addTopLevelItem(item)
        self.case_tree.resizeColumnToContents(0)
        self.case_tree.resizeColumnToContents(1)
        self.scan_label.setText(f"Scanned {len(self.summaries)} JSON files, showing {len(self.filtered_summaries)}.")
        if self.filtered_summaries and self.case is None:
            self.case_tree.setCurrentItem(self.case_tree.topLevelItem(0))

    def open_json_dialog(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open IONM JSON",
            str(self.root),
            "JSON Files (*.json);;All Files (*)",
        )
        if file_name:
            self.load_case(Path(file_name))

    def case_selection_changed(self) -> None:
        items = self.case_tree.selectedItems()
        if not items:
            return
        path = items[0].data(0, USER_ROLE)
        if path:
            self.load_case(Path(path))

    def load_case(self, path: Path) -> None:
        try:
            self.case = load_case(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Could not load JSON:\n{path}\n\n{exc}")
            return
        self.current_trial_index = 0
        self.selected_event = None
        self.update_case_header()
        self.populate_modes()
        self.refresh_events()

    def update_case_header(self) -> None:
        if not self.case:
            return
        pill_color = ALERT if self.case.result == "阳性" else GOOD if self.case.result == "正常" else "#6b7f86"
        self.result_pill.setText(self.case.result)
        self.result_pill.set_color(pill_color)
        self.case_title.setText(
            f"{self.case.patient}  {self.case.patient_id}  |  {self.case.procedure or self.case.path.name}"
        )
        self.clock_label.setText(f"Export clock {format_clock(self.case.start_date)}")
        diagnosis = self.case.fields.get("Diagnosis", "")
        surgeon = self.case.fields.get("Surgeon", "")
        anesth = self.case.fields.get("Anesthesiologist", "")
        surgery = self.case.fields.get("Surgical Description", "")
        self.meta_line.setText(
            "  |  ".join(
                part
                for part in [
                    f"State: {self.case.state}",
                    f"Hardware: {self.case.hardware}",
                    f"Diagnosis: {diagnosis}" if diagnosis else "",
                    f"Surgery: {surgery}" if surgery else "",
                    f"Surgeon: {surgeon}" if surgeon else "",
                    f"Anesthesia: {anesth}" if anesth else "",
                ]
                if part
            )
        )

    def populate_modes(self) -> None:
        self._updating = True
        self.mode_combo.clear()
        if self.case:
            for index, mode in enumerate(self.case.modes):
                label = f"{mode.name}  [{mode.mode_type}]  {len(mode.trials)} trials"
                self.mode_combo.addItem(label, index)
        self._updating = False
        first_with_trials = 0
        if self.case:
            for index, mode in enumerate(self.case.modes):
                if mode.trials:
                    first_with_trials = index
                    break
        self.mode_combo.setCurrentIndex(first_with_trials)
        self.mode_changed()

    def selected_mode(self) -> ModeRecord | None:
        if not self.case:
            return None
        data = self.mode_combo.currentData()
        if data is None:
            return None
        index = int(data)
        return self.case.modes[index] if 0 <= index < len(self.case.modes) else None

    def mode_changed(self) -> None:
        if self._updating:
            return
        self.current_mode = self.selected_mode()
        self.current_trial_index = 0
        self.populate_channels()
        self.configure_trial_controls()
        self.refresh_all_views()

    def configure_trial_controls(self) -> None:
        mode = self.current_mode
        count = len(mode.trials) if mode else 0
        self._updating = True
        self.trial_slider.setEnabled(count > 0)
        self.trial_spin.setEnabled(count > 0)
        self.trial_slider.setRange(0, max(0, count - 1))
        self.trial_spin.setRange(1, max(1, count))
        self.trial_slider.setValue(0)
        self.trial_spin.setValue(1)
        self._updating = False

    def populate_channels(self) -> None:
        self._channel_updating = True
        self.channel_list.clear()
        mode = self.current_mode
        if mode:
            for channel in mode.channels:
                item = QtWidgets.QListWidgetItem(channel)
                item.setFlags(item.flags() | ITEM_IS_USER_CHECKABLE)
                item.setCheckState(CHECKED)
                self.channel_list.addItem(item)
        self._channel_updating = False

    def checked_channels(self) -> set[str]:
        channels: set[str] = set()
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.checkState() == CHECKED:
                channels.add(item.text())
        return channels

    def channel_filter_changed(self) -> None:
        if self._channel_updating:
            return
        self.refresh_waveforms()

    def trial_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self.set_trial_index(value)

    def trial_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self.set_trial_index(value - 1)

    def step_trial(self, delta: int) -> None:
        self.set_trial_index(self.current_trial_index + delta)

    def set_trial_index(self, index: int) -> None:
        mode = self.current_mode
        if not mode or not mode.trials:
            return
        index = max(0, min(index, len(mode.trials) - 1))
        self.current_trial_index = index
        self._updating = True
        self.trial_slider.setValue(index)
        self.trial_spin.setValue(index + 1)
        self._updating = False
        self.refresh_all_views()

    def current_trial(self) -> TrialRecord | None:
        mode = self.current_mode
        if not mode or not mode.trials:
            return None
        if 0 <= self.current_trial_index < len(mode.trials):
            return mode.trials[self.current_trial_index]
        return None

    def refresh_all_views(self) -> None:
        self.refresh_stats()
        self.refresh_waveforms()
        self.refresh_timeline()
        self.refresh_nearby_events()

    def reference_trial(self) -> TrialRecord | None:
        mode = self.current_mode
        if not mode or not mode.trials:
            return None
        for trial in mode.trials:
            if trial.baseline:
                return trial
        return mode.trials[0]

    def matching_baseline_trace(self, trace: TraceRecord) -> TraceRecord | None:
        ref = self.reference_trial()
        if not ref:
            return None
        for candidate in ref.traces:
            if candidate.channel.name == trace.channel.name:
                return candidate
        return None

    def refresh_waveforms(self) -> None:
        self.wave_layout.clear()
        self.wave_plots.clear()
        mode = self.current_mode
        trial = self.current_trial()
        if not mode or not trial:
            self.wave_layout.addLabel("No waveform trials in this mode.", color=TEXT)
            return
        selected_channels = self.checked_channels() or set(mode.channels)
        traces = [trace for trace in trial.traces if trace.channel.name in selected_channels]
        if not traces:
            self.wave_layout.addLabel("No channels selected.", color=TEXT)
            return
        unit, factor = trace_unit_for_mode(mode.mode_type, traces)
        palette = ["#18d7d0", "#53d769", "#f2c14e", "#ff8a65", "#a58bff", "#5cc8ff", "#ff6b9d", "#b6e354"]

        for row, trace in enumerate(traces):
            axis_items = {
                "bottom": FixedUnitAxis("bottom", suffix="ms"),
                "left": FixedUnitAxis("left"),
            }
            plot = self.wave_layout.addPlot(row=row, col=0, axisItems=axis_items)
            self.style_plot(plot)
            if self.wave_plots:
                plot.setXLink(self.wave_plots[0])
            self.wave_plots.append(plot)
            if row < len(traces) - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "Response window", color=TEXT)
            plot.setLabel("left", f"{trace.channel.name} ({unit})", color=TEXT)
            plot.getAxis("left").setWidth(96)

            x = trace.x_ms
            y = trace.values_v * factor
            color = palette[row % len(palette)]
            if self.baseline_check.isChecked():
                baseline = self.matching_baseline_trace(trace)
                if baseline and baseline is not trace and baseline.values_v.size:
                    plot.plot(
                        baseline.x_ms,
                        baseline.values_v * factor,
                        pen=pg.mkPen("#536a70", width=1.0),
                        name="baseline",
                    )
            plot.plot(x, y, pen=pg.mkPen(color, width=1.35), name=trace.channel.name)
            self.add_zero_line(plot)
            self.add_cursors(plot, trace, factor, unit)

            high_cut = f"HC {trace.channel.high_cut:g}Hz" if trace.channel.high_cut else ""
            low_cut = f"LC {trace.channel.low_cut:g}Hz" if trace.channel.low_cut else ""
            gain = f"Gain {trace.channel.gain:g}" if trace.channel.gain else ""
            title = "  ".join(part for part in [trace.channel.name, gain, low_cut, high_cut] if part)
            plot.setTitle(title, color="#cfecef", size="10pt")

            peak = float(np.nanmax(np.abs(y))) if y.size else 1.0
            if self.baseline_check.isChecked():
                baseline = self.matching_baseline_trace(trace)
                if baseline and baseline.values_v.size:
                    peak = max(peak, float(np.nanmax(np.abs(baseline.values_v * factor))))
            peak = peak if peak > 0 else 1.0
            plot.setYRange(-peak * 1.25, peak * 1.25, padding=0.02)
            if trace.sweep_s > 0:
                plot.setXRange(0, trace.sweep_s * 1000.0, padding=0)

        self.wave_layout.ci.layout.setSpacing(5)

    def style_plot(self, plot: pg.PlotItem) -> None:
        plot.showGrid(x=True, y=True, alpha=0.24)
        plot.setMouseEnabled(x=True, y=True)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen("#5d7f86"))
            axis.setTextPen(pg.mkPen("#bdd7da"))
            if hasattr(axis, "enableAutoSIPrefix"):
                axis.enableAutoSIPrefix(False)

    def add_zero_line(self, plot: pg.PlotItem) -> None:
        line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#31535b", width=1))
        plot.addItem(line)

    def add_cursors(self, plot: pg.PlotItem, trace: TraceRecord, factor: float, unit: str) -> None:
        if not self.cursor_check.isChecked():
            return
        for cursor in trace.cursors:
            if cursor.position_s is None:
                continue
            x = float(cursor.position_s) * 1000.0
            line = pg.InfiniteLine(
                pos=x,
                angle=90,
                movable=False,
                pen=pg.mkPen(ACCENT_2, width=1.0, style=DOT_LINE),
            )
            plot.addItem(line)
            label = cursor.label or "cursor"
            voltage = cursor.voltage_v * factor if cursor.voltage_v is not None else None
            if voltage is not None:
                text = f"{label} {x:.1f}ms\n{voltage:.2f}{unit}"
                y = voltage
            else:
                text = f"{label} {x:.1f}ms"
                y = 0
            item = pg.TextItem(text=text, color=ACCENT_2, anchor=(0, 1))
            item.setPos(x, y)
            plot.addItem(item)

    def refresh_stats(self) -> None:
        mode = self.current_mode
        trial = self.current_trial()
        if not self.case or not mode or not trial:
            for label in self.stat_labels.values():
                label.setText("-")
            return
        traces = trial.traces
        unit, factor = trace_unit_for_mode(mode.mode_type, traces)
        start = self.case.start_seconds
        trial_seconds = timestamp_seconds(trial.timestamp)
        elapsed = trial_seconds - start if start is not None and trial_seconds is not None else None
        sweep = traces[0].sweep_s if traces else 0.0
        samples = traces[0].values_v.size if traces else 0
        dt_ms = (sweep * 1000.0 / samples) if samples else 0.0

        peak_text = "-"
        if traces:
            best_channel = ""
            best_p2p = 0.0
            for trace in traces:
                if trace.values_v.size:
                    values = trace.values_v * factor
                    p2p = float(np.nanmax(values) - np.nanmin(values))
                    if p2p > best_p2p:
                        best_p2p = p2p
                        best_channel = trace.channel.name
            peak_text = f"{best_p2p:.2f}{unit} p-p  {best_channel}"

        self.stat_labels["Trial"].setText(f"{trial.trial_number or self.current_trial_index + 1} / {len(mode.trials)}")
        self.stat_labels["Clock"].setText(format_time(trial.timestamp))
        self.stat_labels["Elapsed"].setText(format_elapsed(elapsed))
        self.stat_labels["Sweep"].setText(f"{sweep * 1000:.1f} ms | {samples} samples | {dt_ms:.3f} ms/pt")
        self.stat_labels["Scale"].setText(f"TraceData x scalar -> V, display {unit}")
        self.stat_labels["Stimulus"].setText(stimulus_summary(trial.stimulus))
        self.stat_labels["Peak"].setText(peak_text)

    def visible_events(self) -> list[EventRecord]:
        if not self.case:
            return []
        selected = self.event_filter.currentText()
        show_deleted = self.deleted_check.isChecked()
        events: list[EventRecord] = []
        for event in self.case.events:
            if event.deleted and not show_deleted:
                continue
            if selected == "User Event" and event.event_type != "User Event":
                continue
            if selected == "Impedance" and event.event_type != "Impedance":
                continue
            if selected == "Connections" and "Connection" not in event.event_type:
                continue
            events.append(event)
        return events

    def refresh_events(self) -> None:
        events = self.visible_events()
        self.event_table.setRowCount(len(events))
        start = self.case.start_seconds if self.case else None
        for row, event in enumerate(events):
            event_seconds = timestamp_seconds(event.timestamp)
            elapsed = event_seconds - start if start is not None and event_seconds is not None else None
            values = [
                format_time(event.timestamp),
                format_elapsed(elapsed),
                event.event_type,
                shorten(event.message, 120),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(USER_ROLE, event)
                color = event_color(event.event_type, event.deleted)
                item.setForeground(QtGui.QBrush(QtGui.QColor(color if col in {2, 3} else TEXT)))
                if event.deleted:
                    font = item.font()
                    font.setStrikeOut(True)
                    item.setFont(font)
                self.event_table.setItem(row, col, item)
        self.event_table.resizeColumnsToContents()
        self.refresh_timeline()
        self.refresh_nearby_events()

    def event_selection_changed(self) -> None:
        items = self.event_table.selectedItems()
        if not items:
            return
        event = items[0].data(USER_ROLE)
        if not isinstance(event, EventRecord):
            return
        self.selected_event = event
        self.jump_to_nearest_trial(event)
        self.refresh_timeline()

    def jump_to_nearest_trial(self, event: EventRecord) -> None:
        mode = self.current_mode
        if not mode or not mode.trials:
            return
        event_seconds = timestamp_seconds(event.timestamp)
        if event_seconds is None:
            return
        best_index = min(
            range(len(mode.trials)),
            key=lambda index: abs((timestamp_seconds(mode.trials[index].timestamp) or 0) - event_seconds),
        )
        self.set_trial_index(best_index)

    def refresh_timeline(self) -> None:
        self.timeline_plot.clear()
        self.style_plot(self.timeline_plot.getPlotItem())
        self.timeline_plot.setLabel("bottom", "Surgical timeline", color=TEXT)
        self.timeline_plot.hideAxis("left")
        self.timeline_plot.setYRange(-1.0, 1.0)
        if not self.case:
            return
        start = self.case.start_seconds
        if start is None:
            return

        mode = self.current_mode
        if mode and mode.trials:
            xs = []
            ys = []
            for trial in mode.trials:
                seconds = timestamp_seconds(trial.timestamp)
                if seconds is None:
                    continue
                xs.append((seconds - start) / 3600.0)
                ys.append(0.0)
            if xs:
                scatter = pg.ScatterPlotItem(
                    xs,
                    ys,
                    size=7,
                    pen=pg.mkPen("#84f4ef", width=1),
                    brush=pg.mkBrush(ACCENT),
                )
                self.timeline_plot.addItem(scatter)

        events = self.visible_events()
        for event in events:
            seconds = timestamp_seconds(event.timestamp)
            if seconds is None:
                continue
            x = (seconds - start) / 3600.0
            color = event_color(event.event_type, event.deleted)
            line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(color, width=1, style=DASH_LINE))
            line.setToolTip(f"{format_time(event.timestamp)}  {event.event_type}\n{event.message}")
            self.timeline_plot.addItem(line)

        for label_index, event in enumerate(self.timeline_label_events(events)):
            seconds = timestamp_seconds(event.timestamp)
            if seconds is None:
                continue
            x = (seconds - start) / 3600.0
            color = event_color(event.event_type, event.deleted)
            text = pg.TextItem(shorten(event.message, 22), color=color, anchor=(0, 1), angle=90)
            text.setPos(x, 0.9 - (label_index % 3) * 0.24)
            self.timeline_plot.addItem(text)

        trial = self.current_trial()
        if trial:
            seconds = timestamp_seconds(trial.timestamp)
            if seconds is not None:
                x = (seconds - start) / 3600.0
                selected = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen("#ffffff", width=2.0))
                self.timeline_plot.addItem(selected)
                marker = pg.TextItem("Trial", color="#ffffff", anchor=(0, 0))
                marker.setPos(x, -0.75)
                self.timeline_plot.addItem(marker)

        if self.selected_event:
            seconds = timestamp_seconds(self.selected_event.timestamp)
            if seconds is not None:
                x = (seconds - start) / 3600.0
                selected_event = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(ALERT, width=2.0))
                self.timeline_plot.addItem(selected_event)

        self.timeline_plot.enableAutoRange(axis=pg.ViewBox.XAxis)

    def timeline_label_events(self, events: list[EventRecord]) -> list[EventRecord]:
        labels: list[EventRecord] = []

        if self.selected_event:
            labels.append(self.selected_event)

        trial = self.current_trial()
        trial_seconds = timestamp_seconds(trial.timestamp) if trial else None
        if trial_seconds is not None:
            nearby: list[tuple[float, EventRecord]] = []
            for event in events:
                if event.event_type != "User Event" or event.deleted:
                    continue
                event_seconds = timestamp_seconds(event.timestamp)
                if event_seconds is None:
                    continue
                delta = abs(event_seconds - trial_seconds)
                if delta <= 10 * 60:
                    nearby.append((delta, event))
            nearby.sort(key=lambda item: item[0])
            labels.extend(event for _, event in nearby[:7])

        deduped: list[EventRecord] = []
        seen: set[tuple[Any, str, str]] = set()
        for event in labels:
            key = (event.timestamp, event.event_type, event.message)
            if key not in seen:
                deduped.append(event)
                seen.add(key)
        return deduped[:8]

    def refresh_nearby_events(self) -> None:
        self.nearby_list.clear()
        if not self.case:
            return
        trial = self.current_trial()
        if not trial:
            return
        trial_seconds = timestamp_seconds(trial.timestamp)
        if trial_seconds is None:
            return
        candidates: list[tuple[float, EventRecord]] = []
        for event in self.case.events:
            if event.deleted and not self.deleted_check.isChecked():
                continue
            event_seconds = timestamp_seconds(event.timestamp)
            if event_seconds is None:
                continue
            delta = event_seconds - trial_seconds
            if abs(delta) <= 15 * 60:
                candidates.append((delta, event))
        candidates.sort(key=lambda item: abs(item[0]))
        for delta, event in candidates[:8]:
            sign = "+" if delta >= 0 else "-"
            item = QtWidgets.QListWidgetItem(f"{sign}{format_elapsed(abs(delta))}  {event.event_type}  {shorten(event.message, 46)}")
            item.setForeground(QtGui.QBrush(QtGui.QColor(event_color(event.event_type, event.deleted))))
            self.nearby_list.addItem(item)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = MainWindow(Path.cwd())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
