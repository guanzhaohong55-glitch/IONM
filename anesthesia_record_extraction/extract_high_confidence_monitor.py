from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

import cv2
import fitz
import numpy as np


PDF_NAME_GLOB = "*.pdf"
UPSCALE = 3.5


@dataclass(frozen=True)
class PageConfig:
    page_number: int
    color_xref: int
    graph_roi: tuple[int, int, int, int]
    start_time: str
    expected_points: int
    spo2_values: list[int]
    etco2_values: list[int]


@dataclass
class TracePoint:
    time: str
    page: int
    systolic: int | None
    diastolic: int | None
    heart_rate: int | None
    spo2: int | None
    etco2: int | None
    confidence: str
    propofol_concentration: str
    note: str


PAGE_CONFIGS: list[PageConfig] = []


def cluster_indices(indices: np.ndarray, gap: int = 3) -> list[list[int]]:
    if len(indices) == 0:
        return []
    groups = [[int(indices[0])]]
    for raw in indices[1:]:
        value = int(raw)
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def page_image_from_xref(doc: fitz.Document, xref: int, scale: float = UPSCALE) -> np.ndarray:
    pix = fitz.Pixmap(doc, xref)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def detect_minor_vertical_lines(graph: np.ndarray) -> list[float]:
    plot = graph[:, :2605]
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    mask = (gray < 235).astype(np.uint8) * 255
    connected = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)),
    )
    vertical = cv2.morphologyEx(
        connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)),
    )
    score = (vertical > 0).mean(axis=0)
    peaks: list[float] = []
    for group in cluster_indices(np.where(score > 0.75)[0], gap=5):
        peaks.append(sum(group) / len(group))
    return peaks


def select_cell_boundaries(minor_lines: list[float]) -> list[float]:
    if len(minor_lines) < 4:
        raise RuntimeError("Unable to detect enough vertical grid lines.")
    diffs = np.diff(minor_lines)
    minor_step = float(median(diffs))
    boundaries = [minor_lines[0]]
    last = minor_lines[0]
    for value in minor_lines[1:]:
        if value - last >= minor_step * 2.5:
            boundaries.append(value)
            last = value
    if minor_lines[-1] - boundaries[-1] >= minor_step * 2.5:
        boundaries.append(minor_lines[-1])
    return boundaries


def detect_bp_grid_rows(graph: np.ndarray) -> tuple[list[float], float]:
    plot = graph[:, :2605]
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    mask = (gray < 235).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1)),
    )
    score = (horizontal > 0).mean(axis=1)
    candidates = []
    for group in cluster_indices(np.where(score > 0.6)[0], gap=4):
        center = sum(group) / len(group)
        if 120 <= center <= 1120:
            candidates.append(center)
    if len(candidates) < 12:
        raise RuntimeError("Unable to detect enough horizontal BP/PR grid lines.")
    usable = candidates[:13]
    spacing = float(median(np.diff(usable)))
    return usable, spacing


def build_color_masks(graph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = graph[:, :, 0].astype(np.int16)
    g = graph[:, :, 1].astype(np.int16)
    r = graph[:, :, 2].astype(np.int16)

    red = (r > 150) & (r - g > 40) & (r - b > 40)
    green = (g > 80) & (g - r > 25) & (g - b > 10)
    return red, green


def cluster_axis(values: np.ndarray, gap: int = 6) -> list[tuple[int, int, float]]:
    if len(values) == 0:
        return []
    values = np.unique(values)
    groups = cluster_indices(values, gap=gap)
    return [(group[0], group[-1], sum(group) / len(group)) for group in groups]


def y_to_bp_value(y: float, top_grid_y: float, grid_step: float) -> int:
    value = 260 - ((y - top_grid_y) / grid_step) * 20
    return int(round(value))


def time_points(start_time: str, count: int) -> list[str]:
    base = datetime.strptime(start_time, "%H:%M")
    return [(base + timedelta(minutes=15 * idx)).strftime("%H:%M") for idx in range(count)]


def extract_trace_points(config: PageConfig, graph: np.ndarray) -> tuple[list[TracePoint], dict]:
    minor_lines = detect_minor_vertical_lines(graph)
    boundaries = select_cell_boundaries(minor_lines)
    boundaries = boundaries[: config.expected_points + 1]
    centers = [(left + right) / 2 for left, right in zip(boundaries[:-1], boundaries[1:])]

    bp_rows, bp_step = detect_bp_grid_rows(graph)
    bp_top = bp_rows[0]
    bp_bottom = bp_rows[-1]

    red_mask, green_mask = build_color_masks(graph[:, :2605])

    times = time_points(config.start_time, len(centers))
    points: list[TracePoint] = []
    debug_points = []

    for index, (time_label, x_center) in enumerate(zip(times, centers)):
        x0 = max(0, int(round(x_center - 5)))
        x1 = min(red_mask.shape[1], int(round(x_center + 6)))

        red_y = np.where(red_mask[:, x0:x1])[0]
        red_clusters = [
            cluster
            for cluster in cluster_axis(red_y)
            if bp_top + 10 <= cluster[2] <= bp_bottom - 10
        ]
        red_clusters.sort(key=lambda item: item[2])
        green_y = np.where(green_mask[:, x0:x1])[0]
        green_clusters = [
            cluster
            for cluster in cluster_axis(green_y)
            if bp_top + 200 <= cluster[2] <= bp_bottom - 80
        ]

        systolic = diastolic = heart_rate = None
        confidence = "omit"
        note = ""

        if len(red_clusters) >= 2 and len(green_clusters) >= 1:
            systolic = y_to_bp_value(red_clusters[0][2], bp_top, bp_step)
            diastolic = y_to_bp_value(red_clusters[1][2], bp_top, bp_step)
            heart_rate = y_to_bp_value(green_clusters[0][2], bp_top, bp_step)
            separation = systolic - diastolic if systolic is not None and diastolic is not None else None
            if (
                separation is not None
                and 25 <= separation <= 95
                and 35 <= diastolic <= 140
                and 60 <= systolic <= 260
                and 30 <= heart_rate <= 220
            ):
                confidence = "high"
            else:
                note = "red-trace separation out of range"
        else:
            note = "missing red or green trace cluster"

        if config.page_number == 1 and time_label < "16:20":
            confidence = "omit"
            note = "pre-arterial-line or non-invasive segment"

        if confidence == "high":
            propofol_note = "原图未标浓度"
        else:
            propofol_note = "原图未标浓度"

        points.append(
            TracePoint(
                time=time_label,
                page=config.page_number,
                systolic=systolic if confidence == "high" else None,
                diastolic=diastolic if confidence == "high" else None,
                heart_rate=heart_rate if confidence == "high" else None,
                spo2=config.spo2_values[index] if index < len(config.spo2_values) else None,
                etco2=config.etco2_values[index] if index < len(config.etco2_values) else None,
                confidence=confidence,
                propofol_concentration=propofol_note,
                note=note,
            )
        )

        debug_points.append(
            {
                "time": time_label,
                "x_center": round(float(x_center), 2),
                "red_clusters": [(a, b, round(float(c), 2)) for a, b, c in red_clusters],
                "green_clusters": [(a, b, round(float(c), 2)) for a, b, c in green_clusters],
                "confidence": confidence,
                "note": note,
            }
        )

    debug = {
        "page": config.page_number,
        "minor_vertical_lines": [round(float(v), 2) for v in minor_lines],
        "cell_boundaries": [round(float(v), 2) for v in boundaries],
        "cell_centers": [round(float(v), 2) for v in centers],
        "bp_grid_rows": [round(float(v), 2) for v in bp_rows],
        "bp_grid_step": round(float(bp_step), 4),
        "points": debug_points,
    }
    return points, debug


def write_csv(path: Path, rows: list[TracePoint]) -> None:
    fieldnames = [
        "time",
        "page",
        "systolic",
        "diastolic",
        "heart_rate",
        "spo2",
        "etco2",
        "confidence",
        "propofol_concentration",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, rows: list[TracePoint]) -> None:
    header = [
        "# 高置信版监测数据",
        "",
        "说明：",
        "- 该表只保留脚本判定为 `high` 的动脉压/心率点位。",
        "- `SpO2` 与 `EtCO2` 为原单监测表中的人工复核值。",
        "- 源图未标出丙泊酚“浓度”数值，只能确认存在微泵记录，因此该列统一标注为“原图未标浓度”。",
        "",
        "| 时间 | 动脉收缩压 | 动脉舒张压 | 心率 | SpO2 | EtCO2 | 丙泊酚浓度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines = header[:]
    for row in rows:
        if row.confidence != "high":
            continue
        lines.append(
            f"| {row.time} | {row.systolic} | {row.diastolic} | {row.heart_rate} | "
            f"{row.spo2} | {row.etco2} | {row.propofol_concentration} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scheme(path: Path) -> None:
    content = """# PDF 监测图像识别方案

## 目标
对麻醉记录单中的监测曲线区做“高置信”反推，输出动脉收缩压、动脉舒张压、心率，并保留原单中人工可直接复核的 `SpO2` / `EtCO2`。

## 关键前提
- 当前 PDF 每页只是扫描位图，没有可提取文字、表格结构或矢量曲线。
- 因此“绝对精确”受限于扫描分辨率与描记线条质量，只能做高置信反推。
- 原图未标出丙泊酚浓度数值，只能确认有无丙泊酚微泵记录，不能反推出真实浓度。

## 处理流程
1. 从 PDF 中直接提取彩色页面图像，并放大到 3.5 倍，便于做亚像素级网格定位。
2. 用固定版式 ROI 裁出监测曲线区。
3. 对曲线区做黑灰网格检测：
   - 纵向：先阈值分离灰色网格，再做纵向闭运算与开运算，抓出 5 分钟一格的竖向网格线。
   - 横向：用宽水平核抓出 BP/PR 主水平网格线。
4. 用每 3 根 5 分钟网格线合并为 1 个 15 分钟监测单元，单元中心作为时间点。
5. 用检测出的主水平网格线建立 `y -> 数值` 的线性映射：
   - 顶部主线记为 260
   - 向下每一大格减 20
6. 用颜色分离曲线：
   - 红色：动脉压
   - 绿色：心率
7. 在每个 15 分钟时间点的窄窗口中提取曲线纵坐标：
   - 红色上簇 -> 动脉收缩压
   - 红色下簇 -> 动脉舒张压
   - 绿色簇 -> 心率
8. 只保留满足以下条件的点位：
   - 同时检测到 2 个红色簇和 1 个绿色簇
   - 红色两簇差值落在合理生理范围内
   - 页 1 的 16:20 之前默认视为置动脉穿刺前/非有创段，不纳入高置信输出

## 输出策略
- `high_confidence_monitoring.csv`：完整逐点结果，含 `high/omit` 置信标签。
- `高置信版监测数据.md`：只保留高置信点，便于直接阅读。
- `high_confidence_debug.json`：保留检测到的网格坐标、时间中心与各列曲线簇，便于后续人工复核或继续优化。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    if not PAGE_CONFIGS:
        raise RuntimeError(
            "PAGE_CONFIGS is empty. Add local PageConfig entries for your anesthesia-record template "
            "before running this single-case high-confidence extractor."
        )

    workspace = Path(__file__).resolve().parent
    pdf_candidates = list(workspace.glob(PDF_NAME_GLOB))
    if not pdf_candidates:
        raise FileNotFoundError("No PDF file found in workspace.")
    pdf_path = pdf_candidates[0]
    doc = fitz.open(pdf_path)

    all_rows: list[TracePoint] = []
    debug_payload = {"pdf": pdf_path.name, "pages": []}

    for config in PAGE_CONFIGS:
        page_image = page_image_from_xref(doc, config.color_xref)
        x0, y0, x1, y1 = config.graph_roi
        graph = page_image[y0:y1, x0:x1]
        rows, debug = extract_trace_points(config, graph)
        all_rows.extend(rows)
        debug_payload["pages"].append(debug)

    write_csv(workspace / "high_confidence_monitoring.csv", all_rows)
    write_markdown(workspace / "高置信版监测数据.md", all_rows)
    write_scheme(workspace / "PDF监测图像识别方案.md")
    (workspace / "high_confidence_debug.json").write_text(
        json.dumps(debug_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
