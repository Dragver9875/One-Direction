from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    description: str
    group: str
    accepts_config: bool = False


STAGES: list[Stage] = [
    Stage("prepare_trajectories", "01_prepare_trajectories.py", "Parse points.csv, project coordinates, derive yaw and speed", "preprocess"),
    Stage("prepare_gt_routes", "02_prepare_gt_routes.py", "Parse and project ground_truth.csv route geometries", "preprocess"),
    Stage("build_osm_graph", "03_build_osm_graph.py", "Build directed OSM road graph", "graph"),
    Stage("build_line_graph", "04_build_line_graph.py", "Build edge-centric line graph", "graph"),
    Stage("generate_candidates", "05_generate_candidates.py", "Generate candidate road segments per GPS point", "candidates"),
    Stage("build_training_tensors", "06_build_training_tensors.py", "Build train/val/test tensor datasets", "tensors"),
    Stage("train", "07_train_gnn_hmm.py", "Train GNN-HMM model", "train", False),
    Stage("decode", "08_decode_gnn_hmm.py", "Decode trajectories using trained GNN-HMM", "decode", False),
    Stage("evaluate", "09_evaluate.py", "Evaluate predictions against GT", "evaluate", False),
    Stage("visualize", "10_visualize_errors.py", "Generate visualizations and error plots", "visualize", False),
]


PIPELINE_ALIASES: dict[str, list[str]] = {
    "all": [stage.key for stage in STAGES],
    "preprocess": ["prepare_trajectories", "prepare_gt_routes"],
    "graph": ["build_osm_graph", "build_line_graph"],
    "candidates": ["generate_candidates"],
    "tensors": ["build_training_tensors"],
    "train": ["train"],
    "infer": ["decode"],
    "eval": ["evaluate"],
    "visualize": ["visualize"],
    "data": [
        "prepare_trajectories",
        "prepare_gt_routes",
        "build_osm_graph",
        "build_line_graph",
        "generate_candidates",
        "build_training_tensors",
    ],
    "model": ["train", "decode", "evaluate"],
    "post": ["evaluate", "visualize"],
    "smoke": [
        "prepare_trajectories",
        "prepare_gt_routes",
        "build_osm_graph",
        "build_line_graph",
        "generate_candidates",
    ],
}


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "scripts").exists():
            return candidate

    raise RuntimeError("Could not find repository root. Run from inside the One-Direction repo.")


def stage_by_key() -> dict[str, Stage]:
    return {stage.key: stage for stage in STAGES}


def resolve_stage_keys(selection: list[str]) -> list[str]:
    if not selection:
        selection = ["all"]

    known = stage_by_key()
    resolved: list[str] = []

    for item in selection:
        if item in PIPELINE_ALIASES:
            for key in PIPELINE_ALIASES[item]:
                if key not in resolved:
                    resolved.append(key)
        elif item in known:
            if item not in resolved:
                resolved.append(item)
        else:
            valid = sorted(set(PIPELINE_ALIASES) | set(known))
            raise ValueError(f"Unknown stage or group: {item}. Valid options: {valid}")

    return resolved


def slice_stages(keys: list[str], from_stage: str | None, to_stage: str | None) -> list[str]:
    ordered = [stage.key for stage in STAGES]
    selected_ordered = [key for key in ordered if key in keys]

    if from_stage is not None:
        if from_stage not in ordered:
            raise ValueError(f"Unknown from-stage: {from_stage}")
        start_idx = ordered.index(from_stage)
        selected_ordered = [key for key in selected_ordered if ordered.index(key) >= start_idx]

    if to_stage is not None:
        if to_stage not in ordered:
            raise ValueError(f"Unknown to-stage: {to_stage}")
        end_idx = ordered.index(to_stage)
        selected_ordered = [key for key in selected_ordered if ordered.index(key) <= end_idx]

    return selected_ordered


def make_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    root_text = str(repo_root)

    if existing:
        env["PYTHONPATH"] = root_text + os.pathsep + existing
    else:
        env["PYTHONPATH"] = root_text

    return env


def run_command(command: list[str], repo_root: Path, log_path: Path, dry_run: bool = False) -> int:
    print("")
    print("Command:")
    print(" ".join(command))
    print(f"Working directory: {repo_root}")
    print(f"Log: {log_path}")

    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(repo_root)

    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        log_file.write(f"Command: {' '.join(command)}\n")
        log_file.write(f"Working directory: {repo_root}\n")
        log_file.write(f"Started: {datetime.now().isoformat()}\n\n")

        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("Failed to capture subprocess output.")

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()
        log_file.write(f"\nFinished: {datetime.now().isoformat()}\n")
        log_file.write(f"Return code: {return_code}\n")

    return return_code


def build_stage_command(
    python_executable: str,
    repo_root: Path,
    stage: Stage,
    config: str | None,
    passthrough_args: list[str],
) -> list[str]:
    script_path = repo_root / "scripts" / stage.script

    if not script_path.exists():
        raise FileNotFoundError(f"Stage script not found: {script_path}")

    command = [python_executable, str(script_path)]

    if stage.accepts_config and config:
        command.extend(["--config", config])

    if passthrough_args:
        command.extend(passthrough_args)

    return command


def verify_required_inputs(repo_root: Path, require_osm: bool = False) -> None:
    required = [
        repo_root / "data" / "raw" / "trajectories" / "points.csv",
        repo_root / "data" / "raw" / "trajectories" / "ground_truth.csv",
    ]

    if require_osm:
        required.append(repo_root / "data" / "raw" / "osm" / "oberfranken-latest.osm.pbf")

    missing = [path for path in required if not path.exists()]

    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")


def print_plan(stage_keys: list[str]) -> None:
    known = stage_by_key()
    print("")
    print("Execution plan:")
    for idx, key in enumerate(stage_keys, start=1):
        stage = known[key]
        print(f"  {idx:02d}. {stage.key} - {stage.description}")


def print_stage_list() -> None:
    print("Available stages:")
    for stage in STAGES:
        print(f"  {stage.key:<24} {stage.group:<12} {stage.description}")

    print("")
    print("Available groups:")
    for name, keys in sorted(PIPELINE_ALIASES.items()):
        print(f"  {name:<12} {', '.join(keys)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified One-Direction pipeline runner.")

    parser.add_argument("stages", nargs="*", help="Stage keys or groups to run. Default: all.")
    parser.add_argument("--config", default="configs/local.yaml", help="Config path passed to train/decode/evaluate/visualize stages.")
    parser.add_argument("--from-stage", default=None, help="Start from this stage key after resolving selection.")
    parser.add_argument("--to-stage", default=None, help="Stop at this stage key after resolving selection.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later stages even if one stage fails.")
    parser.add_argument("--skip-input-check", action="store_true", help="Skip checking for points.csv, ground_truth.csv, and OSM PBF.")
    parser.add_argument("--list", action="store_true", help="List available stages and groups.")
    parser.add_argument("--log-dir", default="outputs/run_logs", help="Directory where stage logs are written.")
    parser.add_argument("--stage-args", nargs=argparse.REMAINDER, default=[], help="Extra arguments appended to every selected stage after --stage-args.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()

    if args.list:
        print_stage_list()
        return 0

    selected = resolve_stage_keys(args.stages)
    selected = slice_stages(selected, args.from_stage, args.to_stage)

    if not selected:
        raise RuntimeError("No stages selected.")

    if not args.skip_input_check:
        needs_osm = any(
            key in selected
            for key in [
                "build_osm_graph",
                "build_line_graph",
                "generate_candidates",
                "build_training_tensors",
                "train",
                "decode",
                "evaluate",
                "visualize",
            ]
        )
        verify_required_inputs(repo_root, require_osm=needs_osm)

    print(f"Repository root: {repo_root}")
    print_plan(selected)

    known = stage_by_key()
    log_dir = repo_root / args.log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures: list[tuple[str, int]] = []

    for key in selected:
        stage = known[key]
        print("")
        print("=" * 100)
        print(f"Running stage: {stage.key}")
        print("=" * 100)

        command = build_stage_command(
            python_executable=args.python,
            repo_root=repo_root,
            stage=stage,
            config=args.config,
            passthrough_args=args.stage_args,
        )

        log_path = log_dir / f"{timestamp}_{stage.key}.log"

        try:
            return_code = run_command(
                command=command,
                repo_root=repo_root,
                log_path=log_path,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"Stage {stage.key} failed before execution: {exc}")
            failures.append((stage.key, -1))
            if not args.continue_on_error:
                return 1
            continue

        if return_code != 0:
            print(f"Stage {stage.key} failed with return code {return_code}.")
            failures.append((stage.key, return_code))
            if not args.continue_on_error:
                return return_code

    print("")
    print("=" * 100)

    if failures:
        print("Pipeline finished with failures:")
        for key, code in failures:
            print(f"  {key}: {code}")
        return 1

    print("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
