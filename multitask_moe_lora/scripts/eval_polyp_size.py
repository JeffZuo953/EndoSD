#!/usr/bin/env python3
"""
Polyp size evaluation script.

Reads ground-truth sizes from the provided Excel (.xlsx) file (even if it has a .csv extension),
aggregates frame-level predictions from size_predictions.jsonl, and reports both frame-level and
video-level metrics (MAE, Acc@10mm, Acc@20mm) together with several aggregation heuristics.

Outputs are written to ~/downstreamjobs/Polyp_Size_Dataset/size by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_to_index(col: str) -> int:
    """Convert Excel column letters (e.g., 'A', 'AA') to zero-based index."""
    idx = 0
    for ch in col:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def read_polyp_ground_truth(xlsx_path: Path) -> Dict[str, float]:
    """
    Parse the xlsx file (even if mislabelled .csv) and return video -> size_mm.
    Assumes headers include 'Video_ID' and 'Polyp_Size' with sizes like '11.74mm'.
    """
    xlsx_path = xlsx_path.expanduser().resolve()
    if not xlsx_path.exists():
        raise FileNotFoundError(f"GT file not found: {xlsx_path}")

    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                text = "".join(t.text or "" for t in si.findall(".//a:t", NS))
                shared_strings.append(text)

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows: List[Dict[int, str]] = []
        for row in sheet.findall("a:sheetData/a:row", NS):
            row_values: Dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib["r"]
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = _col_to_index(col_letters)
                t = cell.attrib.get("t")
                v_node = cell.find("a:v", NS)
                if v_node is None:
                    value = ""
                else:
                    value = v_node.text or ""
                if t == "s":
                    value = shared_strings[int(value)]
                row_values[col_idx] = value
            if row_values:
                rows.append(row_values)

    if not rows:
        raise RuntimeError(f"No data rows found in {xlsx_path}")

    header_row = rows[0]
    num_cols = max(header_row.keys())
    headers = [header_row.get(i, f"col{i}") for i in range(num_cols + 1)]
    header_to_idx = {name: idx for idx, name in enumerate(headers)}

    required = ["Video_ID", "Polyp_Size"]
    for col in required:
        if col not in header_to_idx:
            raise KeyError(f"Missing column '{col}' in GT file. Found columns: {headers}")

    gt_map: Dict[str, float] = {}
    for data_row in rows[1:]:
        video_id = str(data_row.get(header_to_idx["Video_ID"], "")).strip()
        size_raw = str(data_row.get(header_to_idx["Polyp_Size"], "")).strip()
        if not video_id or not size_raw:
            continue
        if video_id.isdigit():
            video_key = f"Video{int(video_id):02d}"
        else:
            video_key = f"Video{video_id}"
        size_value = size_raw.replace("mm", "").strip()
        try:
            size_mm = float(size_value)
        except ValueError:
            continue
        gt_map[video_key] = size_mm
    return gt_map


def load_predictions(jsonl_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(data)
    return records


def iqr_filter(values: List[float]) -> List[float]:
    if len(values) < 4:
        return values[:]
    sorted_vals = sorted(values)
    q1 = statistics.quantiles(sorted_vals, n=4)[0]
    q3 = statistics.quantiles(sorted_vals, n=4)[2]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [v for v in values if lower <= v <= upper]
    return filtered or values


def compute_metrics(errors: List[float]) -> Dict[str, float]:
    if not errors:
        return {"mae": math.nan, "acc@10mm": math.nan, "acc@20mm": math.nan, "count": 0}
    mae = float(sum(errors) / len(errors))
    acc10 = sum(err <= 10.0 for err in errors) / len(errors)
    acc20 = sum(err <= 20.0 for err in errors) / len(errors)
    return {"mae": mae, "acc@10mm": acc10, "acc@20mm": acc20, "count": len(errors)}


def write_csv(path: Path, headers: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate polyp size predictions.")
    parser.add_argument("--gt-xlsx", default="~/downstreamjobs/Polyp_Size_Dataset/Polyp_Size_Lables.csv")
    parser.add_argument("--pred-jsonl", default="~/downstreamjobs/save/Polyp_Size_Videos/size_predictions.jsonl")
    parser.add_argument("--output-dir", default="~/downstreamjobs/Polyp_Size_Dataset/size")
    parser.add_argument("--min-size-mm", type=float, default=0.0, help="Ignore predictions <= this value (mm).")
    args = parser.parse_args()

    gt_map = read_polyp_ground_truth(Path(args.gt_xlsx))
    predictions = load_predictions(Path(args.pred_jsonl))
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_rows: List[Dict[str, object]] = []
    per_video_values: Dict[str, List[float]] = defaultdict(list)

    for entry in predictions:
        video = entry.get("video")
        size_mm = float(entry.get("size_mm", 0.0) or 0.0)
        if not video or video not in gt_map or size_mm <= args.min_size_mm:
            continue
        frame_idx = entry.get("frame_index")
        gt_size = gt_map[video]
        abs_err = abs(size_mm - gt_size)
        frames_rows.append({
            "video": video,
            "frame_index": frame_idx,
            "pred_size_mm": size_mm,
            "gt_size_mm": gt_size,
            "abs_error_mm": abs_err,
            "acc_10mm": int(abs_err <= 10.0),
            "acc_20mm": int(abs_err <= 20.0),
        })
        per_video_values[video].append(size_mm)

    # Frame-level metrics summary
    frame_errors = [row["abs_error_mm"] for row in frames_rows]
    frame_metrics = compute_metrics(frame_errors)

    # Video-level aggregation
    methods = OrderedDict([
        ("mean", lambda vals, gt: statistics.mean(vals)),
        ("trimmed_mean", lambda vals, gt: statistics.mean(iqr_filter(vals))),
        ("min", lambda vals, gt: min(vals)),
        ("max", lambda vals, gt: max(vals)),
        ("closest", lambda vals, gt: min(vals, key=lambda v: abs(v - gt))),
    ])

    video_prediction_rows: List[Dict[str, object]] = []
    method_errors: Dict[str, List[float]] = {name: [] for name in methods}

    for video, gt_size in sorted(gt_map.items()):
        row = {"video": video, "gt_size_mm": gt_size}
        values = per_video_values.get(video, [])
        for method_name, fn in methods.items():
            if values:
                pred_val = fn(values, gt_size)
                row[f"{method_name}_mm"] = pred_val
                method_errors[method_name].append(abs(pred_val - gt_size))
            else:
                row[f"{method_name}_mm"] = math.nan
        video_prediction_rows.append(row)

    video_metrics_rows: List[Dict[str, object]] = []
    for method_name, errors in method_errors.items():
        metrics = compute_metrics(errors)
        video_metrics_rows.append({
            "method": method_name,
            "mae_mm": metrics["mae"],
            "acc_10mm": metrics["acc@10mm"],
            "acc_20mm": metrics["acc@20mm"],
            "num_videos": metrics["count"],
        })

    frame_metrics_row = {
        "level": "frame",
        "mae_mm": frame_metrics["mae"],
        "acc_10mm": frame_metrics["acc@10mm"],
        "acc_20mm": frame_metrics["acc@20mm"],
        "count": frame_metrics["count"],
    }

    write_csv(output_dir / "frame_metrics.csv",
              ["video", "frame_index", "pred_size_mm", "gt_size_mm", "abs_error_mm", "acc_10mm", "acc_20mm"],
              frames_rows)
    write_csv(output_dir / "video_predictions.csv",
              ["video", "gt_size_mm"] + [f"{name}_mm" for name in methods],
              video_prediction_rows)
    write_csv(output_dir / "video_metrics.csv",
              ["method", "mae_mm", "acc_10mm", "acc_20mm", "num_videos"],
              video_metrics_rows)
    write_csv(output_dir / "frame_metrics_summary.csv",
              ["level", "mae_mm", "acc_10mm", "acc_20mm", "count"],
              [frame_metrics_row])

    summary_payload = {
        "frame_metrics": frame_metrics,
        "video_metrics": {row["method"]: {k: row[k] for k in ("mae_mm", "acc_10mm", "acc_20mm", "num_videos")} for row in video_metrics_rows},
        "output_dir": str(output_dir),
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
