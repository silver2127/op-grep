"""Generate synthetic op-grep supervision data from the local X4 mod workspace.

The script inspects a handful of high-signal files (jobs, AIScripts, docs)
and emits JSONL records compatible with ``datasets/example_supervised.jsonl``.

Usage
-----
    python generate_x4_synthetic.py \
        --output x4_synthetic.jsonl \
        [--max-jobs 80] [--max-labels 40]

The resulting dataset lives alongside the existing examples so the training
pipeline can be pointed at the larger corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def _relative(path: Path) -> str:
    """Return a workspace-relative POSIX path string."""
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def collect_xenon_jobs(max_jobs: int) -> List[Dict]:
    jobs_path = WORKSPACE_ROOT / "XenonInvasionMod" / "document" / "libraries" / "jobs.xml"
    if not jobs_path.exists():
        return []

    try:
        root = ET.parse(jobs_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse {jobs_path}: {exc}") from exc

    records: List[Dict] = []
    seen: set[str] = set()

    for job in root.iter("job"):
        job_id = job.attrib.get("id")
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        comment = job.attrib.get("comment")
        query_text = (
            f"Where is the Xenon job `{job_id}` defined in the invasion overhaul?"
            if not comment
            else f"Which file defines the job `{job_id}` ({comment})?"
        )
        records.append(
            {
                "query_id": f"JOB_{job_id}",
                "query": query_text,
                "ground_truth": [
                    {
                        "tool": "read",
                        "path": _relative(jobs_path),
                    }
                ],
            }
        )
        if 0 < max_jobs <= len(records):
            break
    return records


def collect_satellite_labels(max_labels: int) -> List[Dict]:
    script_path = WORKSPACE_ROOT / "FS_SAT_FleetMimic" / "mod" / "aiscripts" / "satelliteservice.xml"
    if not script_path.exists():
        return []

    labels: List[str] = []
    for event, elem in ET.iterparse(script_path, events=("start",)):
        if event == "start" and elem.tag == "label":
            name = elem.attrib.get("name")
            if name and name not in labels:
                labels.append(name)
        if 0 < max_labels < len(labels):
            break

    records: List[Dict] = []
    for name in labels[:max_labels or None]:
        records.append(
            {
                "query_id": f"AISCRIPT_LABEL_{name}",
                "query": f"Where is the `{name}` label implemented in the satellite service script?",
                "ground_truth": [
                    {
                        "tool": "grep",
                        "path": _relative(script_path),
                    }
                ],
            }
        )
    return records


def collect_satellite_globals() -> List[Dict]:
    script_path = WORKSPACE_ROOT / "FS_SAT_FleetMimic" / "mod" / "aiscripts" / "satelliteservice.xml"
    if not script_path.exists():
        return []

    target_tokens = {
        "global.$fs_satellite_sectors": "Which script initialises the shared satellite sector registry?",
        "global.$fs_satellite_offset_counter": "Where do we maintain the satellite offset counter for fleet deployment?",
        "$fs_satellite_price": "Where is the satellite price overridden for the fleet mimic order?",
    }

    records: List[Dict] = []
    for symbol, question in target_tokens.items():
        records.append(
            {
                "query_id": f"AISCRIPT_TOKEN_{symbol.replace('$', '').replace('.', '_')}",
                "query": question,
                "ground_truth": [
                    {
                        "tool": "read",
                        "path": _relative(script_path),
                    }
                ],
            }
        )
    return records


def collect_wiki_entries(max_entries: int) -> List[Dict]:
    wiki_path = WORKSPACE_ROOT / "XenonInvasionMod" / "wiki.md"
    if not wiki_path.exists():
        return []

    questions = [
        (
            "Where is the escalation controller documented?",
            "Which page explains the dynamic escalation controller for Xenon fleets?",
        ),
        (
            "Gravidar Obscuring Position Helper",
            "Where can I find the guidance on using `find_closest_gravidar_obscuring_position`?",
        ),
    ]

    records: List[Dict] = []
    for key, question in questions[: max_entries or None]:
        records.append(
            {
                "query_id": f"DOC_{key.replace(' ', '_').upper()}",
                "query": question,
                "ground_truth": [
                    {
                        "tool": "read",
                        "path": _relative(wiki_path),
                    }
                ],
            }
        )
    return records


def collect_satellite_variants() -> List[Dict]:
    base = WORKSPACE_ROOT / "FS_SAT_FleetMimic"
    if not base.exists():
        return []

    return [
        {
            "query_id": "GLOB_SATELLITE_VARIANTS",
            "query": "Where are the document and mod copies of `satelliteservice.xml` stored?",
            "ground_truth": [
                {
                    "tool": "glob",
                    "path": "FS_SAT_FleetMimic/**/satelliteservice.xml",
                }
            ],
        }
    ]


def build_dataset(max_jobs: int, max_labels: int) -> List[Dict]:
    records: List[Dict] = []
    records.extend(collect_xenon_jobs(max_jobs))
    records.extend(collect_satellite_labels(max_labels))
    records.extend(collect_satellite_globals())
    records.extend(collect_wiki_entries(max_entries=10))
    records.extend(collect_satellite_variants())

    # deterministic ordering is fine, but a light shuffle helps evaluation splits
    random.shuffle(records)
    return records


def write_jsonl(records: Iterable[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic op-grep data from X4 mod files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("x4_synthetic.jsonl"),
        help="Output JSONL path (default: datasets/x4_synthetic.jsonl)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=80,
        help="Maximum number of job-based queries to emit (default: 80)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=40,
        help="Maximum number of AIScript label queries to emit (default: 40)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_dataset(max_jobs=args.max_jobs, max_labels=args.max_labels)
    if not records:
        raise SystemExit("No records generated; verify the workspace layout")
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
