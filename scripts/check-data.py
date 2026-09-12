#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_BASE_ROOT = Path(
    os.environ.get("BASE_DATASET_ROOT", "/path/to/datasets/qiaomuyu_search_dual_v1")
)
DEFAULT_REPO_PREFIX = "oopslink/qiaomuyu_search_dual_v1"


def parse_batches(values: list[str]) -> list[int]:
    batches: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if match is None:
                raise ValueError(f"invalid batch selector: {part!r}")
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start < 1 or end < 1:
                raise ValueError("batch numbers must be positive")
            if end < start:
                raise ValueError(f"descending batch range is not supported: {part!r}")
            batches.update(range(start, end + 1))
    if not batches:
        raise ValueError("at least one batch is required")
    return sorted(batches)


def read_parquet_group(root: Path, pattern: str, label: str, errors: list[str]) -> pa.Table | None:
    paths = sorted(root.glob(pattern))
    if not paths:
        errors.append(f"missing {label} Parquet files")
        return None

    tables: list[pa.Table] = []
    for path in paths:
        try:
            tables.append(pq.read_table(path))
        except Exception as exc:
            errors.append(f"unreadable {path.relative_to(root)}: {exc}")
    if len(tables) != len(paths):
        return None
    try:
        return pa.concat_tables(tables, promote_options="default")
    except Exception as exc:
        errors.append(f"cannot combine {label} Parquet files: {exc}")
        return None


def integral_values(table: pa.Table, column: str, errors: list[str]) -> list[int] | None:
    if column not in table.column_names:
        errors.append(f"missing column {column!r}")
        return None
    raw_values = table[column].to_pylist()
    values: list[int] = []
    for value in raw_values:
        if value is None or int(value) != value:
            errors.append(f"column {column!r} contains a non-integral value: {value!r}")
            return None
        values.append(int(value))
    return values


def check_batch(
    batch: int,
    expected_episodes: int,
    base_root: Path,
    repo_prefix: str,
) -> tuple[bool, str]:
    suffix = f"{batch:03d}"
    root = Path(f"{base_root}_batch_{suffix}")
    repo_id = f"{repo_prefix}_batch_{suffix}"
    errors: list[str] = []

    if not root.is_dir():
        return False, f"dataset directory does not exist: {root}"

    info_path = root / "meta/info.json"
    try:
        info = json.loads(info_path.read_text())
    except Exception as exc:
        return False, f"cannot read meta/info.json: {exc}"

    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    if total_episodes != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, found {total_episodes}")
    if not isinstance(total_frames, int) or total_frames <= 0:
        errors.append(f"invalid total_frames: {total_frames!r}")

    for required in (root / "meta/tasks.parquet", root / "meta/stats.json"):
        if not required.is_file() or required.stat().st_size == 0:
            errors.append(f"missing or empty {required.relative_to(root)}")

    meta_table = read_parquet_group(root, "meta/episodes/**/*.parquet", "episode metadata", errors)
    data_table = read_parquet_group(root, "data/**/*.parquet", "frame data", errors)

    meta_episode_ids: list[int] | None = None
    if meta_table is not None:
        if meta_table.num_rows != total_episodes:
            errors.append(
                f"episode metadata has {meta_table.num_rows} rows, expected {total_episodes}"
            )
        meta_episode_ids = integral_values(meta_table, "episode_index", errors)
        if meta_episode_ids is not None and meta_episode_ids != list(range(total_episodes)):
            errors.append(f"episode metadata indices are not contiguous: {meta_episode_ids}")

    if data_table is not None:
        if data_table.num_rows != total_frames:
            errors.append(f"frame data has {data_table.num_rows} rows, expected {total_frames}")
        frame_indices = integral_values(data_table, "index", errors)
        if frame_indices is not None and frame_indices != list(range(total_frames)):
            errors.append("global frame indices are not contiguous")
        data_episode_ids = integral_values(data_table, "episode_index", errors)
        if data_episode_ids is not None:
            counts = Counter(data_episode_ids)
            if sorted(counts) != list(range(total_episodes)):
                errors.append(f"frame data episode indices are not contiguous: {sorted(counts)}")
            if any(count <= 0 for count in counts.values()):
                errors.append("one or more episodes contain no frames")

    video_keys = [
        key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
    ]
    if not video_keys:
        errors.append("dataset has no video features")
    elif meta_table is not None:
        video_template = info.get("video_path")
        if not video_template:
            errors.append("meta/info.json has no video_path template")
        else:
            for video_key in video_keys:
                chunk_column = f"videos/{video_key}/chunk_index"
                file_column = f"videos/{video_key}/file_index"
                from_column = f"videos/{video_key}/from_timestamp"
                to_column = f"videos/{video_key}/to_timestamp"
                for column in (chunk_column, file_column, from_column, to_column):
                    if column not in meta_table.column_names:
                        errors.append(f"missing video metadata column {column!r}")
                if chunk_column not in meta_table.column_names or file_column not in meta_table.column_names:
                    continue
                for row in range(meta_table.num_rows):
                    chunk_index = meta_table[chunk_column][row].as_py()
                    file_index = meta_table[file_column][row].as_py()
                    if chunk_index is None or file_index is None:
                        errors.append(f"episode {row} has no {video_key} video reference")
                        continue
                    path = root / video_template.format(
                        video_key=video_key,
                        chunk_index=int(chunk_index),
                        file_index=int(file_index),
                    )
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(f"missing or empty video: {path.relative_to(root)}")

    png_count = sum(1 for _ in (root / "images").rglob("*.png")) if (root / "images").exists() else 0
    if png_count:
        errors.append(f"found {png_count} unencoded PNG frames")

    if not errors and meta_table is not None:
        try:
            dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
            if dataset.num_episodes != total_episodes or dataset.num_frames != total_frames:
                errors.append(
                    f"LeRobot loaded {dataset.num_episodes} episodes/{dataset.num_frames} frames"
                )
            else:
                starts = integral_values(meta_table, "dataset_from_index", errors)
                ends = integral_values(meta_table, "dataset_to_index", errors)
                if starts is not None and ends is not None:
                    for episode, (start, end) in enumerate(zip(starts, ends, strict=True)):
                        if end <= start:
                            errors.append(f"episode {episode} has invalid frame bounds {start}:{end}")
                            continue
                        sample = dataset[(start + end - 1) // 2]
                        if int(sample["episode_index"]) != episode:
                            errors.append(f"decoded sample for episode {episode} has wrong episode index")
        except Exception as exc:
            errors.append(f"LeRobot load/decode failed: {exc}")

    if errors:
        return False, "; ".join(errors)
    return True, f"{total_episodes} episodes, {total_frames} frames, {len(video_keys)} video streams"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check one or more independent LeRobot data batches for completeness."
    )
    parser.add_argument("batches", nargs="+", help="Batch selectors, e.g. 1 3 5-8 or 1,3,5-8")
    parser.add_argument("--expected", type=int, default=10, help="Expected episodes per batch")
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--repo-prefix", default=DEFAULT_REPO_PREFIX)
    args = parser.parse_args()

    if args.expected < 1:
        parser.error("--expected must be a positive integer")
    try:
        batches = parse_batches(args.batches)
    except ValueError as exc:
        parser.error(str(exc))

    failed = 0
    for batch in batches:
        ok, message = check_batch(batch, args.expected, args.base_root, args.repo_prefix)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] batch {batch:03d}: {message}")
        failed += not ok

    print(f"Checked {len(batches)} batch(es): {len(batches) - failed} OK, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
