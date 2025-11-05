#!/usr/bin/env python3
"""
Utility to generate file lists for the SCard/SCARED dataset.

The script can be pointed either to a single SCARED data bundle
(`<data_root>/left_rectified`, `depthmap_rectified`, `poses`) or to a higher-
level directory such as `/.../train` that contains nested `dataset_*/keyframe_*`
folders. Each discovered bundle gets its own file list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCard file list.")
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the SCARED keyframe directory containing left_rectified/depthmap_rectified/poses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("filelist.txt"),
        help="Output txt file path. Defaults to ./filelist.txt relative to the current working directory.",
    )
    parser.add_argument(
        "--image-ext",
        default=".png",
        help="Extension used for RGB images (default: .png).",
    )
    parser.add_argument(
        "--depth-ext",
        default=".png",
        help="Extension used for depth maps (default: .png).",
    )
    parser.add_argument(
        "--pose-ext",
        default=".json",
        help="Extension used for pose files (default: .json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any frame is missing an asset instead of skipping it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print status information while processing.",
    )
    return parser.parse_args(argv)


def discover_data_roots(base_root: Path) -> List[Path]:
    """
    Return a list of directories that contain the expected SCARED layout.

    A "bundle" is a directory that has left_rectified/depthmap_rectified/poses.
    If the base root already matches this layout it is returned directly.
    Otherwise, the function searches for `dataset_*/keyframe_*` sub-folders
    (optionally with an additional `data/` suffix) that match the layout.
    """
    bundle_dirs: List[Path] = []

    def is_bundle(path: Path) -> bool:
        return all((path / name).is_dir() for name in ("left_rectified", "depthmap_rectified", "poses"))

    if is_bundle(base_root):
        return [base_root]

    for dataset_dir in sorted(base_root.glob("dataset_*")):
        if not dataset_dir.is_dir():
            continue
        for keyframe_dir in sorted(dataset_dir.glob("keyframe_*")):
            if not keyframe_dir.is_dir():
                continue
            candidate_dirs = [keyframe_dir, keyframe_dir / "data"]
            for candidate in candidate_dirs:
                if is_bundle(candidate):
                    bundle_dirs.append(candidate)
                    break

    if not bundle_dirs:
        raise FileNotFoundError(
            f"No SCARED bundles found under {base_root}. "
            "Ensure the directory contains the expected structure."
        )

    return bundle_dirs


def _normalize_ext(ext: str) -> str:
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext.lower()


def _candidate_extensions(preferred_ext: str) -> List[str]:
    base = _normalize_ext(preferred_ext)
    exts = {base, base.upper()}
    return sorted(exts)


def _find_asset(
    directory: Path,
    frame_id: str,
    base_ext: str,
    suffixes: Sequence[str],
    prefer_left: bool = False,
) -> Optional[Path]:
    """
    Locate the first matching asset for the given frame_id.

    Tries explicit suffix/extension combinations first, then falls back to
    globbing with case-insensitive extension checks. When prefer_left=True,
    entries containing "right" in the filename are skipped if possible.
    """
    extensions = _candidate_extensions(base_ext)

    for suffix in suffixes:
        for ext in extensions:
            candidate = directory / f"{frame_id}{suffix}{ext}"
            if candidate.exists():
                return candidate

    matches = []
    for match in directory.glob(f"{frame_id}*"):
        if not match.is_file():
            continue
        if match.suffix.lower() != _normalize_ext(base_ext):
            continue
        matches.append(match)

    if prefer_left:
        matches_left = [m for m in matches if "right" not in m.stem.lower()]
        if matches_left:
            matches = matches_left

    return matches[0] if matches else None


def gather_frame_ids(
    data_root: Path,
    image_ext: str,
    depth_ext: str,
    pose_ext: str,
    strict: bool = False,
    verbose: bool = False,
) -> List[str]:
    image_dir = data_root / "left_rectified"
    depth_dir = data_root / "depthmap_rectified"
    pose_dir = data_root / "poses"

    for directory in (image_dir, depth_dir, pose_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Expected directory missing: {directory}")

    frame_ids: List[str] = []
    missing_assets: List[str] = []

    normalized_pose_ext = _normalize_ext(pose_ext)
    pose_files = [
        p for p in pose_dir.iterdir()
        if p.is_file() and p.suffix.lower() == normalized_pose_ext
    ]
    if not pose_files:
        pose_files = [p for p in pose_dir.iterdir() if p.is_file()]
        normalized_pose_ext = None

    pose_files = sorted(pose_files)

    for pose_path in pose_files:
        if normalized_pose_ext and pose_path.suffix.lower() != normalized_pose_ext:
            continue
        frame_id = pose_path.stem
        raw_frame_id = frame_id
        if frame_id.startswith("frame_data"):
            stripped = frame_id.replace("frame_data", "", 1).lstrip("_")
            if stripped.isdigit():
                frame_id = stripped
        image_path = _find_asset(
            directory=image_dir,
            frame_id=frame_id,
            base_ext=image_ext,
            suffixes=("", "_left", "_left_rectified", "_rectified_left"),
            prefer_left=True,
        )
        depth_path = _find_asset(
            directory=depth_dir,
            frame_id=frame_id,
            base_ext=depth_ext,
            suffixes=("", "_depth", "_depthmap_rectified", "_rectified", "_depthmap"),
        )

        missing = []
        if image_path is None:
            missing.append("image")
        if depth_path is None:
            missing.append("depth")

        if missing:
            message = f"[skip] {frame_id}: missing {', '.join(missing)}"
            missing_assets.append(message)
            if verbose:
                print(message)
            if strict:
                raise FileNotFoundError(message)
            continue

        frame_ids.append(raw_frame_id)
        if verbose:
            print(f"[keep] {frame_id}")

    if not frame_ids:
        warning = "No valid frames were found; check the dataset structure."
        if strict:
            raise RuntimeError(warning)
        if verbose:
            print(warning)

    if verbose and missing_assets:
        print("\nIgnored frames:")
        for line in missing_assets:
            print(f"  {line}")

    return frame_ids


def save_file_list(frame_ids: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for frame_id in frame_ids:
            f.write(frame_id + "\n")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    base_root = args.data_root.resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {base_root}")

    bundle_roots = discover_data_roots(base_root)

    total_entries = 0
    aggregated_entries: List[tuple[str, str]] = []
    for bundle_root in bundle_roots:
        frame_ids = gather_frame_ids(
            data_root=bundle_root,
            image_ext=args.image_ext,
            depth_ext=args.depth_ext,
            pose_ext=args.pose_ext,
            strict=args.strict,
            verbose=args.verbose,
        )

        if not frame_ids:
            if args.verbose:
                print(f"[warn] {bundle_root}: no valid frames found, skipping.")
            continue

        total_entries += len(frame_ids)
        aggregated_entries.extend((bundle_root.as_posix(), frame_id) for frame_id in frame_ids)

        if args.verbose:
            print(f"[bundle] {bundle_root}: {len(frame_ids)} entries")

    if not aggregated_entries:
        raise RuntimeError("No valid frames were found in any bundle; nothing to write.")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for bundle_root, frame_id in aggregated_entries:
            f.write(f"{bundle_root} {frame_id}\n")

    if args.verbose:
        print(f"\nWrote {total_entries} entries from {len(bundle_roots)} bundle(s) to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
