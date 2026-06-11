from __future__ import annotations

import csv
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from batch_extract_monitoring_cases import (
    CaseSource,
    REFERENCE_GRAPH_SIZE,
    REFERENCE_MINOR_VERTICALS,
    TIME_BOX_ANES,
    add_minutes,
    build_templates,
    detect_minor_vertical_lines,
    extract_aux_pages,
    extract_source_pages,
    find_case_sources,
    floor_to_5min,
    graph_crop,
    is_chart_page,
    json_fallback_start_time,
    normalize_mask_region,
    ocr_time_box,
    ocr_top_timeline_start,
    save_png,
    select_cell_boundaries,
    table_band,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_NAME = "批量气体识别CSV"
SUMMARY_FILE_NAME = "处理汇总.csv"
REFERENCE_PDF_GLOB = "*.pdf"


@dataclass
class GasRow:
    time: str
    page: int
    cell_index: int
    spo2: int | None
    spo2_confidence: str
    spo2_method: str
    spo2_score: float
    etco2: int | None
    etco2_confidence: str
    etco2_method: str
    etco2_score: float
    note: str


@dataclass
class MetricResult:
    value: int | None
    confidence: str
    method: str
    score: float
    note: str = ""


def time_to_dt(time_text: str) -> datetime:
    return datetime.strptime(time_text, "%H:%M")


def prepare_gas_cell_mask(crop: np.ndarray, keep_ratio: float = 0.68) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    cut_h = max(1, int(round(mask.shape[0] * keep_ratio)))
    mask = mask[:cut_h, :]

    # Strip table borders before matching digits.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, mask.shape[1] // 2), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, mask.shape[0] - 4)))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
    mask = cv2.subtract(mask, horizontal)
    mask = cv2.subtract(mask, vertical)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return mask


def match_mask_against_library(
    mask: np.ndarray,
    library: dict[str, list[np.ndarray]],
) -> tuple[str | None, float]:
    normalized = normalize_mask_region(mask)
    if normalized is None:
        return None, 0.0

    comp = normalized.astype(np.float32) / 255.0
    best_text: str | None = None
    best_score = -1.0

    for text, glyphs in library.items():
        for glyph in glyphs:
            glyph_arr = glyph.astype(np.float32) / 255.0
            score = float(
                np.sum(comp * glyph_arr)
                / (np.linalg.norm(comp) * np.linalg.norm(glyph_arr) + 1e-6)
            )
            if score > best_score:
                best_score = score
                best_text = text
    return best_text, max(0.0, best_score)


def synthetic_value_template(value: int, width: int = 40, height: int = 40) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(canvas, str(value), (2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    _, binary = cv2.threshold(canvas, 10, 255, cv2.THRESH_BINARY)
    normalized = normalize_mask_region(binary)
    return normalized if normalized is not None else binary


def add_synthetic_value_libraries(libraries: dict[str, dict[str, list[np.ndarray]]]) -> None:
    for value in range(90, 101):
        libraries["spo2"][str(value)].append(synthetic_value_template(value))
    for value in range(15, 61):
        libraries["etco2"][str(value)].append(synthetic_value_template(value))


def build_reference_value_libraries(reference_root: Path | None = None) -> dict[str, dict[str, list[np.ndarray]]]:
    libraries: dict[str, dict[str, list[np.ndarray]]] = {
        "spo2": defaultdict(list),
        "etco2": defaultdict(list),
    }
    add_synthetic_value_libraries(libraries)
    return libraries


def build_page_minor_verticals(page_image: np.ndarray) -> list[float]:
    graph = graph_crop(page_image)
    try:
        minor_verticals = detect_minor_vertical_lines(graph)
    except Exception:
        minor_verticals = []

    if len(minor_verticals) < 12:
        minor_verticals = [
            x * graph.shape[1] / REFERENCE_GRAPH_SIZE[0]
            for x in REFERENCE_MINOR_VERTICALS
        ]
    return minor_verticals


def scale_positions(positions: list[float], source_width: int, target_width: int) -> list[float]:
    if source_width <= 0 or target_width <= 0 or source_width == target_width:
        return positions
    scale = target_width / source_width
    return [value * scale for value in positions]


def build_gas_boundaries_from_minor_verticals(
    minor_verticals: list[float],
    graph_width: int,
) -> list[float]:
    if len(minor_verticals) < 12:
        fallback_minor_verticals = [
            x * graph_width / REFERENCE_GRAPH_SIZE[0]
            for x in REFERENCE_MINOR_VERTICALS
        ]
        minor_verticals = fallback_minor_verticals
    boundaries = select_cell_boundaries(minor_verticals)
    return boundaries[:17]


def build_gas_boundaries_from_axis(
    axis_page_image: np.ndarray,
    target_graph_width: int,
) -> list[float]:
    axis_graph = graph_crop(axis_page_image)
    minor_verticals = build_page_minor_verticals(axis_page_image)
    scaled_minor_verticals = scale_positions(
        minor_verticals,
        axis_graph.shape[1],
        target_graph_width,
    )
    return build_gas_boundaries_from_minor_verticals(scaled_minor_verticals, target_graph_width)


def read_slot_digits(
    crop: np.ndarray,
    templates: dict[str, list[np.ndarray]],
    slot_count: int,
) -> tuple[str, float, list[tuple[str, float]]]:
    from batch_extract_monitoring_cases import classify_component_with_score, normalize_mask_region, table_text_mask

    mask = table_text_mask(crop)
    chars: list[str] = []
    slot_details: list[tuple[str, float]] = []

    for slot in range(slot_count):
        sx0 = int(round(slot * mask.shape[1] / slot_count))
        sx1 = int(round((slot + 1) * mask.shape[1] / slot_count))
        glyph = normalize_mask_region(mask[:, sx0:sx1])
        if glyph is None:
            slot_details.append(("", 0.0))
            continue
        ch, score = classify_component_with_score(glyph, templates)
        chars.append(ch)
        slot_details.append((ch, score))

    numeric_scores = [score for ch, score in slot_details if ch.isdigit()]
    avg_score = float(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0.0
    return "".join(chars), avg_score, slot_details


def parse_slot_integer(raw_text: str, low: int, high: int) -> int | None:
    if not raw_text.isdigit():
        return None
    value = int(raw_text)
    if low <= value <= high:
        return value
    return None


def confidence_from_score(score: float, agree: bool = False) -> str:
    bonus = 0.05 if agree else 0.0
    score += bonus
    if score >= 0.95:
        return "high"
    if score >= 0.85:
        return "medium"
    if score >= 0.72:
        return "low"
    return "omit"


def recognize_spo2(
    crop: np.ndarray,
    actual_library: dict[str, list[np.ndarray]],
    digit_templates: dict[str, list[np.ndarray]],
) -> MetricResult:
    mask = prepare_gas_cell_mask(crop)
    ink = int(np.count_nonzero(mask))
    if ink < 70:
        return MetricResult(None, "omit", "blank", 0.0, "cell too sparse")

    actual_text, actual_score = match_mask_against_library(mask, actual_library)
    actual_value = parse_slot_integer(actual_text or "", 90, 100)

    slot_text, slot_score, slot_details = read_slot_digits(crop, digit_templates, slot_count=3)
    slot_value = parse_slot_integer(slot_text, 90, 100)

    if actual_value is not None and slot_value is not None and actual_value == slot_value:
        score = max(actual_score, slot_score)
        return MetricResult(
            actual_value,
            confidence_from_score(score, agree=True),
            "actual+slot",
            round(score, 4),
        )

    if actual_value is not None and actual_score >= 0.88:
        return MetricResult(
            actual_value,
            confidence_from_score(actual_score),
            "actual",
            round(actual_score, 4),
        )

    if slot_value is not None and slot_score >= 0.92:
        return MetricResult(
            slot_value,
            confidence_from_score(slot_score),
            "slot",
            round(slot_score, 4),
        )

    # SpO2 digits on these sheets are usually 97/99/100; prefer the reference-cell
    # matcher when slot OCR is ambiguous or drops the last zero.
    if actual_value is not None and actual_score >= 0.78:
        note = f"slot={slot_text or '-'}"
        if slot_details:
            note += "," + ",".join(f"{ch or '_'}:{score:.2f}" for ch, score in slot_details)
        return MetricResult(
            actual_value,
            confidence_from_score(actual_score),
            "actual",
            round(actual_score, 4),
            note,
        )

    return MetricResult(None, "omit", "unreadable", max(actual_score, slot_score), f"slot={slot_text or '-'}")


def recognize_etco2(
    crop: np.ndarray,
    actual_library: dict[str, list[np.ndarray]],
    digit_templates: dict[str, list[np.ndarray]],
) -> MetricResult:
    mask = prepare_gas_cell_mask(crop)
    ink = int(np.count_nonzero(mask))
    if ink < 60:
        return MetricResult(None, "omit", "blank", 0.0, "cell too sparse")

    actual_text, actual_score = match_mask_against_library(mask, actual_library)
    actual_value = parse_slot_integer(actual_text or "", 15, 60)

    slot_text, slot_score, _ = read_slot_digits(crop, digit_templates, slot_count=2)
    slot_value = parse_slot_integer(slot_text, 15, 60)

    if actual_value is not None and slot_value is not None and actual_value == slot_value:
        score = max(actual_score, slot_score)
        return MetricResult(
            actual_value,
            confidence_from_score(score, agree=True),
            "actual+slot",
            round(score, 4),
        )

    if slot_value is not None and slot_score >= 0.90 and (actual_value is None or abs(slot_value - actual_value) >= 2):
        return MetricResult(
            slot_value,
            confidence_from_score(slot_score),
            "slot",
            round(slot_score, 4),
        )

    if actual_value is not None and actual_score >= 0.80:
        return MetricResult(
            actual_value,
            confidence_from_score(actual_score),
            "actual",
            round(actual_score, 4),
        )

    if slot_value is not None and slot_score >= 0.72:
        return MetricResult(
            slot_value,
            confidence_from_score(slot_score),
            "slot",
            round(slot_score, 4),
        )

    return MetricResult(None, "omit", "unreadable", max(actual_score, slot_score), f"actual={actual_text},slot={slot_text}")


def maybe_smooth_metric(
    rows: list[GasRow],
    metric_name: str,
    max_jump: int,
    valid_low: int,
    valid_high: int,
) -> None:
    for idx, row in enumerate(rows):
        value = getattr(row, metric_name)
        confidence = getattr(row, f"{metric_name}_confidence")
        if value is None or confidence == "high":
            continue

        neighbor_values: list[int] = []
        for offset in (-2, -1, 1, 2):
            j = idx + offset
            if not (0 <= j < len(rows)):
                continue
            neighbor = getattr(rows[j], metric_name)
            if neighbor is not None:
                neighbor_values.append(neighbor)

        if len(neighbor_values) < 2:
            continue

        target = int(round(sum(neighbor_values) / len(neighbor_values)))
        if not (valid_low <= target <= valid_high):
            continue
        if abs(value - target) <= max_jump:
            continue

        old_value = value
        setattr(row, metric_name, target)
        setattr(row, f"{metric_name}_confidence", "low")
        setattr(row, f"{metric_name}_method", f"{getattr(row, f'{metric_name}_method')}+smoothed")
        setattr(row, f"{metric_name}_score", round(float(getattr(row, f"{metric_name}_score")), 4))

        if row.note:
            row.note += "; "
        row.note += f"{metric_name} smoothed {old_value}->{target}"


def trim_gas_rows(rows: list[GasRow], end_time: str | None) -> list[GasRow]:
    if end_time is None:
        return rows
    end_dt = time_to_dt(end_time)
    return [row for row in rows if time_to_dt(row.time) <= end_dt]


def should_accept_page_timeline_start(
    expected_start_time: str,
    candidate_start_time: str | None,
    page_index: int,
) -> bool:
    if candidate_start_time is None:
        return False
    if page_index == 1:
        return True

    expected_dt = time_to_dt(expected_start_time)
    candidate_dt = time_to_dt(candidate_start_time)
    delta_minutes = int((candidate_dt - expected_dt).total_seconds() // 60)
    return 0 <= delta_minutes <= 60


def should_ignore_trimmed_end_time(
    all_rows: list[GasRow],
    trimmed_rows: list[GasRow],
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


def write_case_csv(path: Path, rows: list[GasRow]) -> Path:
    f, used_path = _open_csv_for_write(path)
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "page",
                "cell_index",
                "spo2",
                "spo2_confidence",
                "spo2_method",
                "spo2_score",
                "etco2",
                "etco2_confidence",
                "etco2_method",
                "etco2_score",
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


def annotate_page_debug(
    page_image: np.ndarray,
    rows: list[GasRow],
    boundaries: list[float],
    output_path: Path,
) -> None:
    graph = graph_crop(page_image)
    if len(boundaries) < 2:
        return

    y0 = table_band(1)[0] - 32
    y1 = table_band(2)[1] + 32
    y0 = max(0, y0)
    y1 = min(graph.shape[0], y1)
    debug = graph[y0:y1, : min(graph.shape[1], 2605)].copy()

    for idx, row in enumerate(rows):
        if idx + 1 >= len(boundaries):
            break
        left = int(round(boundaries[idx]))
        right = int(round(boundaries[idx + 1]))
        center_x = int(round((left + right) / 2))
        cv2.line(debug, (left, 0), (left, debug.shape[0] - 1), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(
            debug,
            row.time,
            (max(0, center_x - 22), 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
        if row.spo2 is not None:
            cv2.putText(
                debug,
                f"SpO2 {row.spo2}",
                (max(0, center_x - 26), 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (40, 200, 40),
                1,
                cv2.LINE_AA,
            )
        if row.etco2 is not None:
            cv2.putText(
                debug,
                f"Et {row.etco2}",
                (max(0, center_x - 22), 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 160, 255),
                1,
                cv2.LINE_AA,
            )

    save_png(output_path, debug)


def process_case(
    case: CaseSource,
    digit_templates: dict[str, list[np.ndarray]],
    value_libraries: dict[str, dict[str, list[np.ndarray]]],
    output_dir: Path,
) -> tuple[str, str, int, int, int]:
    output_path = output_dir / f"{case.case_name}.csv"
    if case.source_path is None:
        write_status_csv(output_path, case, "no pdf/jpg/png source found in case folder")
        return case.case_name, "missing_source", 0, 0, 0

    chart_source_pages = extract_source_pages(case)
    aux_source_pages = extract_aux_pages(case)
    chart_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, chart_page in enumerate(chart_source_pages):
        aux_page = aux_source_pages[idx] if idx < len(aux_source_pages) else chart_page
        if is_chart_page(chart_page):
            chart_pairs.append((chart_page, aux_page))
    if not chart_pairs and case.source_kind == "image" and chart_source_pages:
        chart_pairs = [(chart_source_pages[0], aux_source_pages[0] if aux_source_pages else chart_source_pages[0])]
    if not chart_pairs:
        write_status_csv(output_path, case, "no monitoring chart page detected")
        return case.case_name, "no_chart_page", 0, 0, 0

    first_aux_page = chart_pairs[0][1]
    anes_times = ocr_time_box(first_aux_page, TIME_BOX_ANES, digit_templates)
    status = "ok"
    if anes_times is None:
        fallback_start = json_fallback_start_time(case)
        if fallback_start is None:
            write_status_csv(output_path, case, "failed to OCR anesthesia time range and no json fallback found")
            return case.case_name, "time_ocr_failed", 0, 0, 0
        start_time = fallback_start
        end_time = None
        status = "ok_json_time_fallback"
    else:
        start_time, end_time = anes_times

    timeline_start = ocr_top_timeline_start(chart_pairs[0][0])
    page_start_time = timeline_start if timeline_start is not None else floor_to_5min(start_time)

    all_rows: list[GasRow] = []

    for page_idx, (chart_page, aux_page) in enumerate(chart_pairs, start=1):
        page_timeline_start = ocr_top_timeline_start(chart_page)
        if should_accept_page_timeline_start(page_start_time, page_timeline_start, page_idx):
            page_start_time = page_timeline_start

        graph = graph_crop(aux_page)
        boundaries = build_gas_boundaries_from_axis(chart_page, graph.shape[1])
        page_rows: list[GasRow] = []
        cell_count = min(len(boundaries) - 1, 16)

        for cell_idx in range(cell_count):
            left = int(round(boundaries[cell_idx])) + 10
            right = int(round(boundaries[cell_idx + 1])) - 10
            spo2_crop = graph[table_band(1)[0] : table_band(1)[1], left:right]
            etco2_crop = graph[table_band(2)[0] : table_band(2)[1], left:right]

            spo2_result = recognize_spo2(spo2_crop, value_libraries["spo2"], digit_templates)
            etco2_result = recognize_etco2(etco2_crop, value_libraries["etco2"], digit_templates)
            note_parts = [part for part in (spo2_result.note, etco2_result.note) if part]

            page_rows.append(
                GasRow(
                    time=add_minutes(page_start_time, cell_idx * 15),
                    page=page_idx,
                    cell_index=cell_idx + 1,
                    spo2=spo2_result.value,
                    spo2_confidence=spo2_result.confidence,
                    spo2_method=spo2_result.method,
                    spo2_score=spo2_result.score,
                    etco2=etco2_result.value,
                    etco2_confidence=etco2_result.confidence,
                    etco2_method=etco2_result.method,
                    etco2_score=etco2_result.score,
                    note="; ".join(note_parts),
                )
            )

        maybe_smooth_metric(page_rows, "spo2", max_jump=2, valid_low=90, valid_high=100)
        maybe_smooth_metric(page_rows, "etco2", max_jump=8, valid_low=15, valid_high=60)

        annotate_page_debug(
            aux_page,
            page_rows,
            boundaries=boundaries,
            output_path=output_dir / f"{case.case_name}_gas_page{page_idx}.png",
        )

        all_rows.extend(page_rows)
        if page_rows:
            page_start_time = add_minutes(page_rows[-1].time, 15)

    final_rows = all_rows
    if end_time is not None:
        trimmed_rows = trim_gas_rows(all_rows, end_time)
        if should_ignore_trimmed_end_time(all_rows, trimmed_rows, len(chart_pairs)):
            final_rows = all_rows
        else:
            final_rows = trimmed_rows
    if not final_rows:
        write_status_csv(output_path, case, "no gas values extracted")
        return case.case_name, "gas_extract_failed", 0, 0, 0

    spo2_filled = sum(1 for row in final_rows if row.spo2 is not None)
    etco2_filled = sum(1 for row in final_rows if row.etco2 is not None)
    write_case_csv(output_path, final_rows)

    if spo2_filled == 0 and etco2_filled == 0:
        status = f"{status}_no_gas_values"

    return case.case_name, status, len(final_rows), spo2_filled, etco2_filled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract SpO2 and EtCO2 table values from anesthesia record PDFs/images."
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
        default=["正常", "阳性"],
        help="Case group directory names to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else workspace / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    digit_templates = build_templates(workspace)
    value_libraries = build_reference_value_libraries(workspace)
    cases = find_case_sources(workspace, tuple(args.groups))

    summary_rows: list[dict[str, object]] = []
    for case in cases:
        case_name, status, row_count, spo2_filled, etco2_filled = process_case(
            case,
            digit_templates=digit_templates,
            value_libraries=value_libraries,
            output_dir=output_dir,
        )
        summary_rows.append(
            {
                "case_name": case_name,
                "status": status,
                "row_count": row_count,
                "spo2_filled": spo2_filled,
                "etco2_filled": etco2_filled,
            }
        )

    f, _ = _open_csv_for_write(output_dir / SUMMARY_FILE_NAME)
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_name", "status", "row_count", "spo2_filled", "etco2_filled"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
