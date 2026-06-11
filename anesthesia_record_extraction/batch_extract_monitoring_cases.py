from __future__ import annotations

import csv
import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import fitz
import numpy as np
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional runtime dependency
    RapidOCR = None

from extract_high_confidence_monitor import (
    PAGE_CONFIGS,
    cluster_axis,
    detect_bp_grid_rows,
    detect_minor_vertical_lines,
    page_image_from_xref,
    select_cell_boundaries,
    y_to_bp_value,
)


ROOT = Path(__file__).resolve().parent
TARGET_WIDTH = 3987
GRAPH_ROI = (850, 2500, 3987, 4700)
SOURCE_PAGE_REF_SIZE = (3986, 5761)
TIME_BOX_REF_PAGE_SIZE = (4556, 6584)
TIME_BOX_ANES = (3460, 3910, 3940, 4085)
TIME_BOX_SURG = (3460, 4260, 3940, 4445)
TOP_TIMELINE_BOX = (350, 850, 2400, 1200)
REF_ANES_TIME = "15:53~21:10"
REF_SURG_TIME = "16:30~20:40"
TABLE_TOP = 1253
TABLE_ROW_HEIGHT = 88
NORMAL_GROUP = "正常"
POSITIVE_GROUP = "阳性"
OUTPUT_DIR_NAME = "批量监测CSV"
SUMMARY_FILE_NAME = "处理汇总.csv"

# The anesthesia time field is a fixed-layout slot box on the source form.
TIME_SLOT_Y_RANGE = (0, 70)
TIME_DIGIT_SLOTS = (
    (21, 39),
    (40, 60),
    (93, 112),
    (113, 132),
    (161, 183),
    (184, 205),
    (221, 244),
    (245, 269),
)
REFERENCE_GRAPH_SIZE = (3136, 2200)
REFERENCE_BP_ROWS = (151.5, 231.5, 311.5, 388.5, 469.0, 546.5, 623.5, 704.0, 780.5, 857.5, 938.5, 1015.5, 1095.5)
REFERENCE_MINOR_VERTICALS = (
    54.5, 110.0, 163.0, 215.0, 268.0, 322.0, 376.0, 429.0, 483.0, 536.0,
    590.0, 643.0, 695.0, 748.0, 801.0, 852.0, 905.0, 958.0, 1013.0, 1065.0,
    1118.0, 1171.0, 1224.0, 1278.0, 1332.0, 1387.0, 1440.0, 1493.0, 1546.0, 1600.0,
    1654.0, 1708.0, 1760.0, 1811.0, 1864.0, 1916.0, 1972.0, 2024.0, 2077.0, 2130.0,
    2183.0, 2238.0, 2291.0, 2343.0, 2395.5, 2452.0, 2504.0, 2557.0, 2603.5,
)
GRAPH_TIME_BOX_ANES = (2600, 1320, 3136, 1510)
GRAPH_TIME_BOX_SURG = (2600, 1500, 3136, 1690)
TEMP_AXIS_VALUES = list(range(40, 14, -2))
BP_AXIS_VALUES = list(range(260, 0, -20))

_OCR_ENGINE = None


@dataclass
class CaseSource:
    case_name: str
    case_dir: Path | None
    source_path: Path | None
    source_kind: str


@dataclass
class MinuteRow:
    time: str
    page: int
    bp_source: str
    systolic: int | None
    diastolic: int | None
    heart_rate: int | None
    spo2: int | None
    etco2: int | None
    confidence: str
    note: str


def save_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode png: {path}")
    buf.tofile(str(path))


def render_pdf_pages(pdf_path: Path) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    doc = fitz.open(pdf_path)
    for page_index in range(doc.page_count):
        best_image = None
        best_score = -1
        for img in doc.get_page_images(page_index, full=True):
            xref = img[0]
            image = page_image_from_xref(doc, xref)
            score = 0
            try:
                score = len(detect_minor_vertical_lines(graph_crop(image)))
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best_image = image
        if best_image is not None:
            pages.append(best_image)
    return pages


def render_pdf_pages_combined(pdf_path: Path) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    doc = fitz.open(pdf_path)
    for page_index in range(doc.page_count):
        layer_images: list[np.ndarray] = []
        for img in doc.get_page_images(page_index, full=True):
            xref = img[0]
            layer_images.append(page_image_from_xref(doc, xref))
        if not layer_images:
            continue

        layer_images.sort(key=lambda image: int(np.any(image > 5, axis=2).sum()), reverse=True)
        composed = layer_images[0].copy()
        for layer in layer_images[1:]:
            non_black = np.any(layer > 5, axis=2, keepdims=True)
            composed = np.where(non_black, layer, composed)
        pages.append(composed)
    return pages


def load_raster_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode image: {path}")
    scale = TARGET_WIDTH / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (TARGET_WIDTH, height), interpolation=cv2.INTER_CUBIC)


def find_case_sources(
    root: Path | None = None,
    group_names: tuple[str, ...] | None = None,
) -> list[CaseSource]:
    search_root = root or ROOT
    groups = group_names or (NORMAL_GROUP, POSITIVE_GROUP)
    cases: list[CaseSource] = []
    for group in groups:
        group_dir = search_root / group
        if not group_dir.exists():
            continue
        for case_dir in sorted([path for path in group_dir.iterdir() if path.is_dir()]):
            pdfs = sorted(case_dir.glob("*.pdf"))
            rasters = sorted([*case_dir.glob("*.jpg"), *case_dir.glob("*.jpeg"), *case_dir.glob("*.png")])
            if pdfs:
                cases.append(
                    CaseSource(
                        case_name=case_dir.name,
                        case_dir=case_dir,
                        source_path=pdfs[0],
                        source_kind="pdf",
                    )
                )
            elif rasters:
                cases.append(
                    CaseSource(
                        case_name=case_dir.name,
                        case_dir=case_dir,
                        source_path=rasters[0],
                        source_kind="image",
                    )
                )
            else:
                cases.append(
                    CaseSource(
                        case_name=case_dir.name,
                        case_dir=case_dir,
                        source_path=None,
                        source_kind="missing",
                    )
                )
    return cases


def extract_source_pages(case: CaseSource) -> list[np.ndarray]:
    if case.source_path is None:
        return []
    if case.source_kind == "pdf":
        return render_pdf_pages(case.source_path)
    if case.source_kind == "image":
        return [load_raster_image(case.source_path)]
    return []


def extract_aux_pages(case: CaseSource) -> list[np.ndarray]:
    if case.source_path is None:
        return []
    if case.source_kind == "pdf":
        return render_pdf_pages_combined(case.source_path)
    if case.source_kind == "image":
        return [load_raster_image(case.source_path)]
    return []


def graph_crop(page_image: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = GRAPH_ROI
    y1 = min(y1, page_image.shape[0])
    x1 = min(x1, page_image.shape[1])
    return page_image[y0:y1, x0:x1]


def build_alignment_preview_canvas(page_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx0, gy0, gx1, gy1 = GRAPH_ROI
    gy1 = min(gy1, page_image.shape[0])
    gx1 = min(gx1, page_image.shape[1])

    preview_x0 = 0
    preview_x1 = min(page_image.shape[1], 3850)
    preview_y0 = max(0, gy0 - 60)
    preview_y1 = min(page_image.shape[0], gy0 + 1700)
    preview = page_image[preview_y0:preview_y1, preview_x0:preview_x1].copy()

    graph_px0 = max(0, gx0 - preview_x0)
    graph_py0 = max(0, gy0 - preview_y0)
    graph_px1 = min(preview.shape[1], gx1 - preview_x0)
    graph_py1 = min(preview.shape[0], gy1 - preview_y0)
    graph_view = preview[graph_py0:graph_py1, graph_px0:graph_px1]
    return preview, graph_view


def is_chart_page(page_image: np.ndarray) -> bool:
    graph = graph_crop(page_image)
    try:
        minor_verticals = detect_minor_vertical_lines(graph)
        bp_rows, _ = detect_bp_grid_rows(graph)
    except Exception:
        return False
    return len(minor_verticals) >= 8 and len(bp_rows) >= 12


def grayscale_mask(image: np.ndarray, threshold: int = 190) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return inv


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def component_boxes(mask: np.ndarray, min_area: int = 60, min_height: int = 20) -> list[tuple[int, int, int, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area or h < min_height or w < 4:
            continue
        boxes.append((x, y, x + w, y + h))
    boxes.sort(key=lambda item: item[0])
    return boxes


def normalize_component(mask: np.ndarray, box: tuple[int, int, int, int], width: int = 24, height: int = 40) -> np.ndarray:
    x0, y0, x1, y1 = box
    glyph = mask[y0:y1, x0:x1]
    if glyph.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    scale = min((width - 4) / max(glyph.shape[1], 1), (height - 4) / max(glyph.shape[0], 1))
    new_w = max(1, int(round(glyph.shape[1] * scale)))
    new_h = max(1, int(round(glyph.shape[0] * scale)))
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.uint8)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def scale_box(
    box: tuple[int, int, int, int],
    reference_size: tuple[int, int],
    target_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    ref_w, ref_h = reference_size
    target_h, target_w = target_shape[:2]
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * target_w / ref_w)),
        int(round(y0 * target_h / ref_h)),
        int(round(x1 * target_w / ref_w)),
        int(round(y1 * target_h / ref_h)),
    )


def get_ocr_engine():
    global _OCR_ENGINE
    if RapidOCR is None:
        return None
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def normalize_mask_region(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return normalize_component(mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def synthetic_digit_template(digit: str, width: int = 24, height: int = 40) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(canvas, digit, (2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
    _, binary = cv2.threshold(canvas, 10, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_time_crop(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, crop.shape[1] // 16), 1))
    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    cleaned = cv2.subtract(inv, horizontal)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def time_crop(page_image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = scale_box(box, TIME_BOX_REF_PAGE_SIZE, page_image.shape)
    return page_image[y0:y1, x0:x1]


def positive_runs(values: np.ndarray, threshold: int = 1) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for idx, value in enumerate(values):
        if value > threshold and not in_run:
            start = idx
            in_run = True
        elif value <= threshold and in_run:
            runs.append((start, idx - 1))
            in_run = False
    if in_run:
        runs.append((start, len(values) - 1))
    return runs


def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def time_digit_windows(mask: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    col_sums = (mask > 0).sum(axis=0)
    runs = positive_runs(col_sums, threshold=1)
    runs = [run for run in runs if run[1] - run[0] + 1 >= max(8, mask.shape[1] // 30)]
    groups = merge_close_runs(runs, max_gap=max(16, mask.shape[1] // 18))
    groups = [group for group in groups if group[1] - group[0] + 1 >= max(30, mask.shape[1] // 12)]
    if len(groups) < 2:
        return None
    groups = groups[:2]

    y0, y1 = TIME_SLOT_Y_RANGE
    y1 = min(mask.shape[0], y1)
    windows: list[tuple[int, int, int, int]] = []
    for group_start, group_end in groups:
        width = group_end - group_start + 1
        for idx in range(4):
            x0 = group_start + int(round(idx * width / 4))
            x1 = group_start + int(round((idx + 1) * width / 4))
            x0 = max(0, x0 - 2)
            x1 = min(mask.shape[1], x1 + 2)
            windows.append((x0, y0, x1, y1))
    return windows


def scaled_time_digit_slot(
    slot: tuple[int, int],
    crop_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    crop_h, crop_w = crop_shape[:2]
    ref_w = TIME_BOX_ANES[2] - TIME_BOX_ANES[0]
    ref_h = TIME_BOX_ANES[3] - TIME_BOX_ANES[1]
    sx0, sx1 = slot
    sy0, sy1 = TIME_SLOT_Y_RANGE
    return (
        int(round(sx0 * crop_w / ref_w)),
        int(round(sy0 * crop_h / ref_h)),
        int(round(sx1 * crop_w / ref_w)),
        int(round(sy1 * crop_h / ref_h)),
    )


def table_text_mask(crop: np.ndarray, threshold: int = 180) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    trimmed_height = max(1, int(round(mask.shape[0] * 0.65)))
    return mask[:trimmed_height]


def add_cell_digit_templates(
    graph: np.ndarray,
    boundaries: list[float],
    values: list[int],
    band: tuple[int, int],
    templates: dict[str, list[np.ndarray]],
    slot_count: int,
) -> None:
    for idx, value in enumerate(values):
        left = int(round(boundaries[idx])) + 10
        right = int(round(boundaries[idx + 1])) - 10
        crop = graph[band[0] : band[1], left:right]
        mask = table_text_mask(crop)
        _, w = mask.shape
        text = str(value)
        start_slot = slot_count - len(text)
        for pos, digit in enumerate(text):
            slot_idx = start_slot + pos
            sx0 = int(round(slot_idx * w / slot_count))
            sx1 = int(round((slot_idx + 1) * w / slot_count))
            glyph = normalize_mask_region(mask[:, sx0:sx1])
            if glyph is not None:
                templates[digit].append(glyph)


def add_reference_time_digit_templates(page_image: np.ndarray, templates: dict[str, list[np.ndarray]]) -> None:
    crop = preprocess_time_crop(time_crop(page_image, TIME_BOX_ANES))
    windows = time_digit_windows(crop)
    if windows is None:
        return
    reference_digits = [REF_ANES_TIME[i] for i in (0, 1, 3, 4, 6, 7, 9, 10)]
    for digit, (x0, y0, x1, y1) in zip(reference_digits, windows):
        glyph = normalize_mask_region(crop[y0:y1, x0:x1])
        if glyph is not None:
            templates[digit].append(glyph)


def build_templates(reference_root: Path | None = None) -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {ch: [] for ch in "0123456789"}
    # Keep the public version data-free: do not derive templates from private reference cases.
    # Sites can add local labeled templates later, but synthetic glyphs keep the pipeline runnable.
    for digit in "0123456789":
        templates[digit].append(synthetic_digit_template(digit))
    return templates


def table_band(row_index: int) -> tuple[int, int]:
    top = TABLE_TOP + row_index * TABLE_ROW_HEIGHT + 10
    bottom = TABLE_TOP + (row_index + 1) * TABLE_ROW_HEIGHT - 10
    return top, bottom


def add_numeric_templates(
    graph: np.ndarray,
    boundaries: list[float],
    values: list[int],
    band: tuple[int, int],
    templates: dict[str, list[np.ndarray]],
) -> None:
    for idx, value in enumerate(values):
        left = int(round(boundaries[idx])) + 10
        right = int(round(boundaries[idx + 1])) - 10
        crop = graph[band[0] : band[1], left:right]
        add_labeled_crop(crop, str(value), templates)


def add_fixed_value_templates(
    graph: np.ndarray,
    boundaries: list[float],
    value: str,
    band: tuple[int, int],
    templates: dict[str, list[np.ndarray]],
) -> None:
    left = int(round(boundaries[0])) + 10
    right = int(round(boundaries[1])) - 10
    crop = graph[band[0] : band[1], left:right]
    add_labeled_crop(crop, value, templates)


def add_time_templates(
    page_image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    templates: dict[str, list[np.ndarray]],
) -> None:
    x0, y0, x1, y1 = box
    crop = page_image[y0:y1, x0:x1]
    add_labeled_crop(crop, label, templates)


def add_labeled_crop(crop: np.ndarray, label: str, templates: dict[str, list[np.ndarray]]) -> None:
    mask = clean_mask(grayscale_mask(crop))
    boxes = component_boxes(mask, min_area=40, min_height=18)
    target = [ch for ch in label if ch in templates]
    if len(boxes) < len(target):
        return
    if len(boxes) > len(target):
        boxes = boxes[: len(target)]
    if len(boxes) != len(target):
        return
    for box, char in zip(boxes, target):
        templates[char].append(normalize_component(mask, box))


def classify_component(component: np.ndarray, templates: dict[str, list[np.ndarray]]) -> str:
    best_char = "?"
    best_score = -1.0
    comp = component.astype(np.float32) / 255.0
    for char, glyphs in templates.items():
        for glyph in glyphs:
            glyph_arr = glyph.astype(np.float32) / 255.0
            score = float(np.sum(comp * glyph_arr) / (np.linalg.norm(comp) * np.linalg.norm(glyph_arr) + 1e-6))
            if score > best_score:
                best_score = score
                best_char = char
    return best_char


def classify_component_with_score(component: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    best_char = "?"
    best_score = -1.0
    comp = component.astype(np.float32) / 255.0
    for char, glyphs in templates.items():
        for glyph in glyphs:
            glyph_arr = glyph.astype(np.float32) / 255.0
            score = float(np.sum(comp * glyph_arr) / (np.linalg.norm(comp) * np.linalg.norm(glyph_arr) + 1e-6))
            if score > best_score:
                best_score = score
                best_char = char
    return best_char, best_score


def ocr_characters(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> str:
    mask = clean_mask(grayscale_mask(crop))
    boxes = component_boxes(mask, min_area=40, min_height=18)
    chars: list[str] = []
    for box in boxes:
        component = normalize_component(mask, box)
        chars.append(classify_component(component, templates))
    return "".join(chars)


def scaled_reference_boundaries(graph_width: int) -> list[float]:
    return [boundary * graph_width / REFERENCE_GRAPH_SIZE[0] for boundary in select_cell_boundaries(list(REFERENCE_MINOR_VERTICALS))]


def ocr_spo2_table_value(crop: np.ndarray) -> int | None:
    mask = table_text_mask(crop)
    slot_counts: list[int] = []
    for slot in range(3):
        sx0 = int(round(slot * mask.shape[1] / 3))
        sx1 = int(round((slot + 1) * mask.shape[1] / 3))
        slot_counts.append(int(np.sum(mask[:, sx0:sx1] > 200)))

    if slot_counts[0] > 80:
        return 100
    if slot_counts[2] > 25:
        return 99
    if slot_counts[2] < 10 and slot_counts[1] > 150:
        return 97
    return None


def ocr_two_digit_table_value(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> int | None:
    mask = table_text_mask(crop)
    digits: list[str] = []
    for slot in range(2):
        sx0 = int(round(slot * mask.shape[1] / 2))
        sx1 = int(round((slot + 1) * mask.shape[1] / 2))
        region = mask[:, sx0:sx1]
        glyph = normalize_mask_region(region)
        if glyph is None:
            return None
        digit, score = classify_component_with_score(glyph, templates)
        if score < 0.55:
            return None
        digits.append(digit)
    if not all(ch.isdigit() for ch in digits):
        return None
    return int("".join(digits))


def parse_time_range(text: str) -> tuple[str, str] | None:
    match = re.search(r"(\d{2}:\d{2}).*?(\d{2}:\d{2})", text)
    if match:
        return match.group(1), match.group(2)
    compact = re.sub(r"[^0-9:]", "", text)
    match = re.search(r"(\d{2}:\d{2})(\d{2}:\d{2})", compact)
    if match:
        return match.group(1), match.group(2)
    return None


def ocr_time_box(page_image: np.ndarray, box: tuple[int, int, int, int], templates: dict[str, list[np.ndarray]]) -> tuple[str, str] | None:
    graph_time_box = None
    if box == TIME_BOX_ANES:
        graph_time_box = GRAPH_TIME_BOX_ANES
    elif box == TIME_BOX_SURG:
        graph_time_box = GRAPH_TIME_BOX_SURG

    ocr_engine = get_ocr_engine()
    if graph_time_box is not None and ocr_engine is not None:
        graph = graph_crop(page_image)
        x0, y0, x1, y1 = scale_box(graph_time_box, REFERENCE_GRAPH_SIZE, graph.shape)
        result, _ = ocr_engine(graph[y0:y1, x0:x1])
        if result:
            text = "".join(item[1] for item in result)
            parsed = parse_time_range(text)
            if parsed is not None:
                return parsed

    crop = preprocess_time_crop(time_crop(page_image, box))
    windows = time_digit_windows(crop)
    if windows is None:
        return None
    digits: list[str] = []
    for x0, y0, x1, y1 in windows:
        glyph = normalize_mask_region(crop[y0:y1, x0:x1])
        if glyph is None:
            return None
        digits.append(classify_component(glyph, templates))

    start = f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}"
    end = f"{digits[4]}{digits[5]}:{digits[6]}{digits[7]}"
    try:
        start_dt = datetime.strptime(start, "%H:%M")
        end_dt = datetime.strptime(end, "%H:%M")
    except ValueError:
        return None
    if end_dt <= start_dt:
        return None
    return start, end


def ocr_top_timeline_start(page_image: np.ndarray) -> str | None:
    ocr_engine = get_ocr_engine()
    if ocr_engine is None:
        return None
    x0, y0, x1, y1 = scale_box(TOP_TIMELINE_BOX, SOURCE_PAGE_REF_SIZE, page_image.shape)
    crop = page_image[y0:y1, x0:x1]
    result, _ = ocr_engine(crop)
    if not result:
        return None
    text = " ".join(item[1] for item in result)
    matches = re.findall(r"\d{2}:\d{2}", text)
    if not matches:
        return None
    try:
        first_dt = datetime.strptime(matches[0], "%H:%M")
    except ValueError:
        return None
    if len(matches) >= 2:
        try:
            second_dt = datetime.strptime(matches[1], "%H:%M")
        except ValueError:
            return None
        if second_dt <= first_dt or int((second_dt - first_dt).total_seconds() // 60) not in {30, 60}:
            return None
    return matches[0]


def floor_to_5min(time_text: str) -> str:
    dt = datetime.strptime(time_text, "%H:%M")
    floored = dt - timedelta(minutes=dt.minute % 5)
    return floored.strftime("%H:%M")


def add_minutes(time_text: str, minutes: int) -> str:
    dt = datetime.strptime(time_text, "%H:%M")
    dt += timedelta(minutes=minutes)
    return dt.strftime("%H:%M")


def hsv_masks(graph_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(graph_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red = (((h <= 12) | (h >= 170)) & (s >= 70) & (v >= 70)).astype(np.uint8)
    blue = ((h >= 95) & (h <= 130) & (s >= 60) & (v >= 60)).astype(np.uint8)
    green = ((h >= 40) & (h <= 100) & (s >= 35) & (v >= 35)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    red = cv2.dilate(red, kernel, iterations=1).astype(bool)
    blue = cv2.dilate(blue, kernel, iterations=1).astype(bool)
    green = cv2.dilate(green, kernel, iterations=1).astype(bool)
    return red, blue, green


def cluster_candidates(mask: np.ndarray, x_center: float, y_min: int, y_max: int, window: int = 6) -> list[dict]:
    x0 = max(0, int(round(x_center - window)))
    x1 = min(mask.shape[1], int(round(x_center + window + 1)))
    ys = np.where(mask[y_min:y_max, x0:x1])[0]
    candidates = []
    for start, end, center in cluster_axis(ys + y_min, gap=6):
        count = int(np.sum(mask[start : end + 1, x0:x1]))
        candidates.append({"start": start, "end": end, "center": float(center), "count": count})
    return candidates


def choose_candidate(candidates: list[dict], prev_y: float | None, expected_y: float, max_jump: float = 120) -> dict | None:
    if not candidates:
        return None
    ranked = []
    for item in candidates:
        target = prev_y if prev_y is not None else expected_y
        delta = abs(item["center"] - target)
        penalty = 0 if delta <= max_jump else (delta - max_jump) * 2
        score = penalty + delta - min(item["count"], 40) * 0.05
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0])
    best = ranked[0][1]
    if prev_y is not None and abs(best["center"] - prev_y) > max_jump * 1.6:
        return None
    return best


def choose_green_candidate(
    candidates: list[dict],
    prev_y: float | None,
    expected_y: float,
    avoid_upper_y: float | None = None,
    avoid_y: float | None = None,
    avoid_upper_radius: float = 18.0,
    avoid_radius: float = 14.0,
    max_jump: float = 80.0,
) -> dict | None:
    if not candidates:
        return None
    target = prev_y if prev_y is not None else expected_y
    ranked: list[tuple[float, dict]] = []
    for item in candidates:
        center = float(item["center"])
        delta = abs(center - target)
        penalty = 0.0 if delta <= max_jump else (delta - max_jump) * 2.5
        score = penalty + delta - min(float(item["count"]), 40.0) * 0.05
        if avoid_upper_y is not None:
            upper_gap = abs(center - avoid_upper_y)
            if upper_gap < avoid_upper_radius:
                score += (avoid_upper_radius - upper_gap) * 3.0
        if avoid_y is not None:
            gap = abs(center - avoid_y)
            if gap < avoid_radius:
                score += (avoid_radius - gap) * 2.2
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0])

    best = ranked[0][1]
    best_center = float(best["center"])
    if prev_y is not None and abs(best_center - prev_y) > max_jump * 1.4:
        return None

    # Guard against snapping onto the diastolic cluster when green trace is weak.
    if avoid_upper_y is not None and abs(best_center - avoid_upper_y) < 8.0:
        if len(ranked) > 1:
            alt = ranked[1][1]
            alt_center = float(alt["center"])
            if abs(alt_center - avoid_upper_y) >= 8.0 and abs(alt_center - target) <= abs(best_center - target) + 12.0:
                return alt
        if prev_y is None or abs(best_center - prev_y) > 10.0:
            return None

    if avoid_y is not None and abs(best_center - avoid_y) < 6.0:
        if len(ranked) > 1:
            alt = ranked[1][1]
            alt_center = float(alt["center"])
            if abs(alt_center - avoid_y) >= 6.0 and abs(alt_center - target) <= abs(best_center - target) + 10.0:
                return alt
        if prev_y is None or abs(best_center - prev_y) > 10.0:
            return None
    return best


def y_to_temp_value(y: float, top_grid_y: float, grid_step: float) -> float:
    value = 40.0 - ((y - top_grid_y) / grid_step) * 2.0
    return round(value, 1)


def draw_value_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    x_offset: int = 6,
    y_offset: int = -6,
    scale: float = 0.32,
) -> None:
    tx = max(0, min(canvas.shape[1] - 90, x + x_offset))
    ty = max(12, min(canvas.shape[0] - 4, y + y_offset))
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_completed_y_separators(
    canvas: np.ndarray,
    minor_verticals: list[float],
    bp_rows: list[float],
) -> None:
    if len(minor_verticals) < 2 or len(bp_rows) < 2:
        return
    x0 = int(round(minor_verticals[0]))
    x1 = int(round(minor_verticals[-1]))
    major_count = min(12, len(bp_rows) - 1)

    for i in range(major_count + 1):
        y = int(round(bp_rows[i]))
        cv2.line(canvas, (x0, y), (x1, y), (0, 165, 255), 1, cv2.LINE_AA)
        if i == major_count:
            continue
        step = (bp_rows[i + 1] - bp_rows[i]) / 5.0
        for k in range(1, 5):
            yy = int(round(bp_rows[i] + k * step))
            cv2.line(canvas, (x0, yy), (x1, yy), (135, 190, 190), 1, cv2.LINE_AA)


def draw_axis_labels_in_place(
    canvas: np.ndarray,
    minor_verticals: list[float],
    bp_rows: list[float],
    page_start_time: str,
) -> None:
    if len(bp_rows) < 2:
        return
    draw_completed_y_separators(canvas, minor_verticals, bp_rows)
    major_count = min(13, len(bp_rows), len(TEMP_AXIS_VALUES), len(BP_AXIS_VALUES))
    x_temp = 4
    x_bp = 46
    for i in range(major_count):
        y = int(round(bp_rows[i]))
        temp_label = str(TEMP_AXIS_VALUES[i])
        bp_label = str(BP_AXIS_VALUES[i])
        draw_value_label(canvas, temp_label, x_temp, y, (0, 140, 255), x_offset=0, y_offset=4, scale=0.30)
        draw_value_label(canvas, bp_label, x_bp, y, (0, 140, 255), x_offset=0, y_offset=4, scale=0.30)
        if i >= major_count - 1:
            continue

        step = (bp_rows[i + 1] - bp_rows[i]) / 5.0
        for k in range(1, 5):
            yy = int(round(bp_rows[i] + k * step))
            temp_value = TEMP_AXIS_VALUES[i] - 0.4 * k
            bp_value = BP_AXIS_VALUES[i] - 4 * k
            temp_minor = f"{temp_value:.1f}".rstrip("0").rstrip(".")
            bp_minor = str(int(round(bp_value)))
            draw_value_label(canvas, temp_minor, x_temp, yy, (130, 190, 190), x_offset=0, y_offset=3, scale=0.24)
            draw_value_label(canvas, bp_minor, x_bp, yy, (130, 190, 190), x_offset=0, y_offset=3, scale=0.24)

    axis_y = int(round(bp_rows[min(12, len(bp_rows) - 1)] + 20))
    for idx, x_center in enumerate(minor_verticals):
        x = int(round(x_center))
        label = add_minutes(page_start_time, idx * 5)
        if idx % 6 == 0:
            draw_value_label(canvas, label, x, axis_y, (0, 120, 255), x_offset=-18, y_offset=0, scale=0.34)
        else:
            draw_value_label(canvas, label, x, axis_y, (0, 170, 170), x_offset=-12, y_offset=0, scale=0.26)


def build_page_gas_lookup(
    page_image: np.ndarray,
    page_start_time: str,
    templates: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, int | None]]:
    graph = graph_crop(page_image)
    try:
        boundaries = select_cell_boundaries(detect_minor_vertical_lines(graph))
    except Exception:
        boundaries = scaled_reference_boundaries(graph.shape[1])
    spo2_band = table_band(1)
    etco2_band = table_band(2)

    lookup: dict[str, dict[str, int | None]] = {}
    cell_count = min(len(boundaries) - 1, 16)
    for idx in range(cell_count):
        left = int(round(boundaries[idx])) + 10
        right = int(round(boundaries[idx + 1])) - 10
        spo2_crop = graph[spo2_band[0] : spo2_band[1], left:right]
        etco2_crop = graph[etco2_band[0] : etco2_band[1], left:right]
        spo2_value = ocr_spo2_table_value(spo2_crop)
        etco2_value = ocr_two_digit_table_value(etco2_crop, templates)
        time_label = add_minutes(page_start_time, idx * 15)
        lookup[time_label] = {
            "spo2": spo2_value,
            "etco2": etco2_value,
        }
    return lookup


def extract_page_rows(
    page_image: np.ndarray,
    page_number: int,
    page_start_time: str,
    gas_lookup: dict[str, dict[str, int | None]],
    prev_green: float | None = None,
    debug_canvas: np.ndarray | None = None,
) -> tuple[list[MinuteRow], float | None]:
    graph = graph_crop(page_image)
    try:
        minor_verticals = detect_minor_vertical_lines(graph)
    except Exception:
        minor_verticals = [x * graph.shape[1] / REFERENCE_GRAPH_SIZE[0] for x in REFERENCE_MINOR_VERTICALS]
    try:
        bp_rows, bp_step = detect_bp_grid_rows(graph)
    except Exception:
        bp_rows = [y * graph.shape[0] / REFERENCE_GRAPH_SIZE[1] for y in REFERENCE_BP_ROWS]
        bp_step = float(np.median(np.diff(bp_rows)))
    bp_top = bp_rows[0]
    red_mask, blue_mask, green_mask = hsv_masks(graph[:, :2605])
    if debug_canvas is not None:
        draw_axis_labels_in_place(debug_canvas, minor_verticals, bp_rows, page_start_time)

    green_prev = prev_green
    temp_prev = None
    red_upper_prev = None
    red_lower_prev = None
    blue_upper_prev = None
    blue_lower_prev = None
    bp_source = "blue"
    points: list[MinuteRow] = []

    for idx, x_center in enumerate(minor_verticals):
        time_label = add_minutes(page_start_time, idx * 5)
        temp_pick = None
        temperature = None
        green_pick = None

        red_upper = choose_candidate(
            cluster_candidates(red_mask, x_center, int(bp_rows[5] - 15), int(bp_rows[8] + 10)),
            red_upper_prev,
            expected_y=float(bp_rows[6]),
        )
        red_lower = choose_candidate(
            cluster_candidates(red_mask, x_center, int(bp_rows[8] - 10), int(bp_rows[11] + 20)),
            red_lower_prev,
            expected_y=float(bp_rows[10]),
        )
        blue_upper = choose_candidate(
            cluster_candidates(blue_mask, x_center, int(bp_rows[6] - 25), int(bp_rows[9] + 10)),
            blue_upper_prev,
            expected_y=float(bp_rows[8]),
            max_jump=90,
        )
        blue_lower = choose_candidate(
            cluster_candidates(blue_mask, x_center, int(bp_rows[8] - 10), int(bp_rows[11] + 25)),
            blue_lower_prev,
            expected_y=float(bp_rows[10]),
            max_jump=90,
        )

        if blue_upper is not None:
            blue_upper_prev = blue_upper["center"]
        if blue_lower is not None:
            blue_lower_prev = blue_lower["center"]
        if red_upper is not None:
            red_upper_prev = red_upper["center"]
        if red_lower is not None:
            red_lower_prev = red_lower["center"]

        source_used = ""
        systolic = diastolic = heart_rate = None
        confidence = "omit"
        note = ""

        red_ok = red_upper is not None and red_lower is not None
        blue_ok = blue_upper is not None and blue_lower is not None

        if page_number == 1:
            if idx <= 4 and blue_ok:
                bp_source = "blue"
            elif idx >= 5 and red_ok:
                bp_source = "red"
            elif bp_source == "blue" and red_ok and not blue_ok:
                bp_source = "red"
        else:
            bp_source = "red"

        selected_upper = selected_lower = None
        if bp_source == "red" and red_ok:
            selected_upper, selected_lower = red_upper, red_lower
            source_used = "动脉"
        elif bp_source == "blue" and blue_ok:
            selected_upper, selected_lower = blue_upper, blue_lower
            source_used = "无创"
        elif red_ok:
            selected_upper, selected_lower = red_upper, red_lower
            source_used = "动脉"
            bp_source = "red"
        elif blue_ok:
            selected_upper, selected_lower = blue_upper, blue_lower
            source_used = "无创"
            bp_source = "blue"

        temp_candidates = cluster_candidates(
            red_mask,
            x_center,
            int(bp_rows[1] - 10),
            int(bp_rows[5] - 20),
            window=8,
        )
        if selected_upper is not None:
            temp_candidates = [
                candidate
                for candidate in temp_candidates
                if float(candidate["center"]) < float(selected_upper["center"]) - 35.0
            ]
        temp_pick = choose_candidate(
            temp_candidates,
            temp_prev,
            expected_y=float(bp_rows[2]),
            max_jump=55,
        )
        if temp_pick is not None:
            temp_prev = float(temp_pick["center"])
            temperature = y_to_temp_value(temp_pick["center"], bp_top, bp_step)

        green_candidates = cluster_candidates(
            green_mask,
            x_center,
            int(bp_rows[7] - 40),
            int(bp_rows[12] + 20),
        )
        if selected_upper is not None and selected_lower is not None:
            systolic_hint = y_to_bp_value(selected_upper["center"], bp_top, bp_step)
            diastolic_hint = y_to_bp_value(selected_lower["center"], bp_top, bp_step)
            band_filtered = [
                candidate
                for candidate in green_candidates
                if diastolic_hint + 3 <= y_to_bp_value(candidate["center"], bp_top, bp_step) <= systolic_hint - 4
            ]
            if band_filtered:
                green_candidates = band_filtered
        avoid_y = float(selected_lower["center"]) if selected_lower is not None else None
        green_pick = choose_green_candidate(
            green_candidates,
            green_prev,
            expected_y=float(bp_rows[9]),
            avoid_upper_y=float(selected_upper["center"]) if selected_upper is not None else None,
            avoid_y=avoid_y,
        )
        if green_pick is not None:
            green_prev = green_pick["center"]

        if selected_upper is not None and selected_lower is not None and green_pick is not None:
            systolic = y_to_bp_value(selected_upper["center"], bp_top, bp_step)
            diastolic = y_to_bp_value(selected_lower["center"], bp_top, bp_step)
            heart_rate = y_to_bp_value(green_pick["center"], bp_top, bp_step)
            pulse_pressure = systolic - diastolic
            if 15 <= pulse_pressure <= 100 and 20 <= heart_rate <= 220:
                confidence = "high"
            else:
                note = "out-of-range after y-axis mapping"
        else:
            note = "missing trace cluster"

        gas_values = gas_lookup.get(time_label, {})
        points.append(
            MinuteRow(
                time=time_label,
                page=page_number,
                bp_source=source_used,
                systolic=systolic if confidence == "high" else None,
                diastolic=diastolic if confidence == "high" else None,
                heart_rate=heart_rate if confidence == "high" else None,
                spo2=gas_values.get("spo2"),
                etco2=gas_values.get("etco2"),
                confidence=confidence,
                note=note,
            )
        )

        if debug_canvas is not None:
            x = int(round(x_center))
            cv2.line(debug_canvas, (x, 0), (x, debug_canvas.shape[0] - 1), (220, 220, 0), 1)
            if temp_pick is not None:
                temp_y = int(round(temp_pick["center"]))
                cv2.circle(debug_canvas, (x, temp_y), 5, (0, 0, 255), -1)
                if temperature is not None:
                    draw_value_label(debug_canvas, f"T {temperature:.1f}", x, temp_y, (0, 0, 255), x_offset=8, y_offset=-8)
            if green_pick is not None:
                green_y = int(round(green_pick["center"]))
                cv2.circle(debug_canvas, (x, green_y), 5, (0, 180, 0), -1)
                if heart_rate is not None:
                    draw_value_label(debug_canvas, f"HR {heart_rate}", x, green_y, (0, 180, 0), x_offset=8, y_offset=13)
            if selected_upper is not None:
                upper_color = (0, 0, 255) if source_used == "动脉" else (255, 0, 0)
                upper_y = int(round(selected_upper["center"]))
                cv2.circle(debug_canvas, (x, upper_y), 5, upper_color, -1)
                if systolic is not None:
                    draw_value_label(debug_canvas, f"SYS {systolic}", x, upper_y, upper_color, x_offset=8, y_offset=-8)
            if selected_lower is not None:
                lower_color = (0, 80, 255) if source_used == "动脉" else (255, 120, 0)
                lower_y = int(round(selected_lower["center"]))
                cv2.circle(debug_canvas, (x, lower_y), 5, lower_color, -1)
                if diastolic is not None:
                    draw_value_label(debug_canvas, f"DIA {diastolic}", x, lower_y, lower_color, x_offset=8, y_offset=13)

    return points, green_prev


def time_to_dt(time_text: str) -> datetime:
    return datetime.strptime(time_text, "%H:%M")


def ceil_to_5min(time_text: str) -> str:
    dt = time_to_dt(time_text)
    extra = (5 - dt.minute % 5) % 5
    if extra:
        dt += timedelta(minutes=extra)
    return dt.strftime("%H:%M")


def json_fallback_start_time(case: CaseSource) -> str | None:
    if case.case_dir is None:
        return None
    json_files = sorted(case.case_dir.glob("*.json"))
    if not json_files:
        return None
    try:
        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        start_ts = payload["Cases"][0]["StartDate"]
    except Exception:
        return None
    start_dt = datetime.fromtimestamp(start_ts / 1_000_000, tz=timezone.utc) + timedelta(minutes=10)
    return floor_to_5min(start_dt.strftime("%H:%M"))


def trim_rows(rows: list[MinuteRow], end_time: str | None) -> list[MinuteRow]:
    if end_time is None:
        return rows
    end_dt = time_to_dt(ceil_to_5min(end_time))
    return [row for row in rows if time_to_dt(row.time) <= end_dt]


def should_ignore_trimmed_end_time(
    all_rows: list[MinuteRow],
    trimmed_rows: list[MinuteRow],
    page_count: int,
) -> bool:
    if page_count <= 1 or not all_rows:
        return False
    if not trimmed_rows:
        return True
    return trimmed_rows[-1].page < all_rows[-1].page


def _open_csv_for_write(path: Path):
    try:
        return path.open("w", newline="", encoding="utf-8-sig"), path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}__locked_copy.csv")
        return fallback.open("w", newline="", encoding="utf-8-sig"), fallback


def write_case_csv(path: Path, rows: list[MinuteRow]) -> Path:
    f, used_path = _open_csv_for_write(path)
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "page",
                "bp_source",
                "systolic",
                "diastolic",
                "heart_rate",
                "spo2",
                "etco2",
                "confidence",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return used_path


def write_status_csv(path: Path, case: CaseSource, note: str) -> Path:
    f, used_path = _open_csv_for_write(path)
    with f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "source_kind", "status", "note"])
        writer.writeheader()
        writer.writerow(
            {
                "case_name": case.case_name,
                "source_kind": case.source_kind,
                "status": "unavailable",
                "note": note,
            }
        )
    return used_path


def process_case(case: CaseSource, templates: dict[str, list[np.ndarray]], output_dir: Path) -> tuple[str, str]:
    output_path = output_dir / f"{case.case_name}.csv"
    if case.source_path is None:
        write_status_csv(output_path, case, "no pdf/jpg/png source found in case folder")
        return case.case_name, "missing_source"

    chart_source_pages = extract_source_pages(case)
    aux_source_pages = extract_aux_pages(case)
    chart_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, chart_page in enumerate(chart_source_pages):
        aux_page = aux_source_pages[idx] if idx < len(aux_source_pages) else chart_page
        if is_chart_page(chart_page):
            chart_pairs.append((chart_page, aux_page))
    if not chart_pairs and case.source_kind == "image" and chart_source_pages:
        chart_pairs = [(chart_source_pages[0], aux_source_pages[0] if aux_source_pages else chart_source_pages[0])]
    chart_pages = [pair[0] for pair in chart_pairs]
    if not chart_pages:
        write_status_csv(output_path, case, "no monitoring chart page detected")
        return case.case_name, "no_chart_page"

    first_aux_page = chart_pairs[0][1]
    anes_times = ocr_time_box(first_aux_page, TIME_BOX_ANES, templates)
    status = "ok"
    if anes_times is None:
        fallback_start = json_fallback_start_time(case)
        if fallback_start is None:
            write_status_csv(output_path, case, "failed to OCR anesthesia time range and no json fallback found")
            return case.case_name, "time_ocr_failed"
        start_time = fallback_start
        end_time = None
        status = "ok_json_time_fallback"
    else:
        start_time, end_time = anes_times

    timeline_start = ocr_top_timeline_start(chart_pairs[0][0])
    if timeline_start is not None:
        page_start_time = timeline_start
    else:
        page_start_time = floor_to_5min(start_time)
    all_rows: list[MinuteRow] = []
    green_prev = None

    for page_idx, (chart_page, aux_page) in enumerate(chart_pairs, start=1):
        gas_lookup = build_page_gas_lookup(aux_page, page_start_time, templates)
        debug_preview, debug_canvas = build_alignment_preview_canvas(chart_page)
        try:
            page_rows, green_prev = extract_page_rows(
                chart_page,
                page_number=page_idx,
                page_start_time=page_start_time,
                gas_lookup=gas_lookup,
                prev_green=green_prev,
                debug_canvas=debug_canvas,
            )
        except Exception as exc:
            write_status_csv(output_path, case, f"monitor trace extraction failed: {exc}")
            return case.case_name, "trace_extract_failed"
        save_png(output_dir / f"{case.case_name}_overlay_page{page_idx}.png", debug_preview)
        all_rows.extend(page_rows)
        if page_rows:
            page_start_time = add_minutes(page_rows[-1].time, 5)

    final_rows = all_rows
    if end_time is not None:
        trimmed_rows = trim_rows(all_rows, end_time)
        if should_ignore_trimmed_end_time(all_rows, trimmed_rows, len(chart_pairs)):
            final_rows = all_rows
        else:
            final_rows = trimmed_rows
    high_rows = sum(1 for row in final_rows if row.confidence == "high")
    if high_rows < 5:
        write_status_csv(output_path, case, "monitor trace extracted too few high-confidence points")
        return case.case_name, "trace_extract_failed"
    write_case_csv(output_path, final_rows)
    return case.case_name, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract blood pressure, heart rate, SpO2, and EtCO2 from anesthesia record PDFs/images."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT,
        help="Directory containing case groups such as 正常/ and 阳性/. Defaults to this script directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <workspace>/{OUTPUT_DIR_NAME}.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[NORMAL_GROUP, POSITIVE_GROUP],
        help="Case group directory names to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else workspace / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    templates = build_templates(workspace)
    cases = find_case_sources(workspace, tuple(args.groups))
    summary_rows: list[tuple[str, str]] = []
    for case in cases:
        summary_rows.append(process_case(case, templates, output_dir))

    summary_path = output_dir / SUMMARY_FILE_NAME
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "status"])
        writer.writeheader()
        for case_name, status in summary_rows:
            writer.writerow({"case_name": case_name, "status": status})


if __name__ == "__main__":
    main()
