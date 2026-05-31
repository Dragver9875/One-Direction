from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    description: str
    group: str
    default_args: list[str] = field(default_factory=list)


STAGES: list[Stage] = [
    Stage(
        key="prepare_trajectories",
        script="01_prepare_trajectories.py",
        description="Parse points.csv, project coordinates, derive yaw and speed",
        group="01_preprocess",
    ),
    Stage(
        key="prepare_gt_routes",
        script="02_prepare_gt_routes.py",
        description="Parse and project ground_truth.csv route geometries",
        group="02_preprocess",
    ),
    Stage(
        key="build_osm_graph",
        script="03_build_osm_graph.py",
        description="Build directed OSM road graph",
        group="03_graph",
    ),
    Stage(
        key="build_line_graph",
        script="04_build_line_graph.py",
        description="Build edge-centric line graph and transition table",
        group="04_graph",
    ),
    Stage(
        key="generate_candidates",
        script="05_generate_candidates.py",
        description="Generate candidate road segments per trajectory point",
        group="05_candidates",
    ),
    Stage(
        key="build_training_tensors",
        script="06_build_training_tensors.py",
        description="Build train/val/test tensor datasets",
        group="06_tensors",
        default_args=["--transition-mask-mode", "all"],
    ),
    Stage(
        key="train_gnn_hmm",
        script="07_train_gnn_hmm.py",
        description="Train GNN-HMM model",
        group="07_train",
        default_args=[
            "--output", "outputs/checkpoints",
            "--epochs", "100",
            "--batch-size", "2",
            "--lr", "0.001",
            "--emission-weight", "1.0",
            "--transition-weight", "2.0",
            "--label-smoothing", "0.02",
            "--margin-weight", "0.1",
            "--margin", "1.0",
            "--device", "auto",
        ],
    ),
    Stage(
        key="decode_gnn_hmm",
        script="08_decode_gnn_hmm.py",
        description="Decode trajectories using the best GNN-HMM checkpoint",
        group="08_decode",
        default_args=[
            "--checkpoint", "outputs/checkpoints/gnn_hmm_best.pt",
            "--output", "outputs/matches/gnn_hmm_matches.parquet",
            "--illegal-transition-mode", "soft",
            "--illegal-penalty", "5.0",
            "--device", "auto",
        ],
    ),
    Stage(
        key="evaluate_gnn_hmm",
        script="09_evaluate.py",
        description="Evaluate GNN-HMM predictions",
        group="09_evaluate",
    ),
    Stage(
        key="visualize_errors",
        script="10_visualize_errors.py",
        description="Generate static error/path visualizations",
        group="10_visualize",
    ),
    Stage(
        key="visualize_osm_overlay",
        script="11_visualize_osm_overlay.py",
        description="Generate interactive OSM overlay for decoded trajectories",
        group="11_visualize",
    ),
    Stage(
        key="debug_gnn_hmm_data",
        script="12_debug_gnn_hmm_data.py",
        description="Validate tensors, candidate labels, transition masks, and graph metadata",
        group="12_debug",
    ),
    Stage(
        key="run_gnn_hmm_experiments",
        script="13_run_gnn_hmm_experiments.py",
        description="Run transition-weight experiment sweep for GNN-HMM",
        group="13_experiments",
    ),
    Stage(
        key="evaluate_mm2_style",
        script="14_evaluate_mm2_style.py",
        description="Compute MM2-style weighted metrics for GNN-HMM output",
        group="14_metrics",
        default_args=[
            "--workflow", "gnn_hmm",
            "--matches", "outputs/matches/gnn_hmm_matches.parquet",
            "--output", "outputs/metrics/gnn_hmm_mm2_style_metrics_test.json",
            "--trajectory-output", "outputs/metrics/gnn_hmm_mm2_style_trajectory_test.csv",
            "--require-gt-candidate",
        ],
    ),
    Stage(
        key="compare_mm2_style",
        script="15_compare_mm2_style.py",
        description="Compare MM2-style metrics across available HMM/GNN-HMM/D-SAC/PPO outputs",
        group="15_metrics",
    ),
    Stage(
        key="evaluate_exact_mm2_route_overlap",
        script="16_evaluate_exact_mm2_route_overlap.py",
        description="Compare exact route-overlap style metrics across available HMM/GNN-HMM/D-SAC outputs",
        group="16_metrics",
        default_args=[
            "compare",
            "--workflows", "hmm", "gnn_hmm", "dsac",
            "--allow-point-fallback",
        ],
    ),
]


PIPELINE_ALIASES: dict[str, list[str]] = {
    "all": [stage.key for stage in STAGES],
    "full": [stage.key for stage in STAGES],
    "default": [stage.key for stage in STAGES],
    "preprocess": ["prepare_trajectories", "prepare_gt_routes"],
    "graph": ["build_osm_graph", "build_line_graph"],
    "candidates": ["generate_candidates"],
    "tensors": ["build_training_tensors"],
    "train": ["train_gnn_hmm"],
    "decode": ["decode_gnn_hmm"],
    "infer": ["decode_gnn_hmm"],
    "evaluate": ["evaluate_gnn_hmm"],
    "eval": ["evaluate_gnn_hmm"],
    "visualize": ["visualize_errors", "visualize_osm_overlay"],
    "debug": ["debug_gnn_hmm_data"],
    "experiments": ["run_gnn_hmm_experiments"],
    "mm2": ["evaluate_mm2_style", "compare_mm2_style", "evaluate_exact_mm2_route_overlap"],
    "metrics": ["evaluate_gnn_hmm", "evaluate_mm2_style", "compare_mm2_style", "evaluate_exact_mm2_route_overlap"],
    "post": [
        "evaluate_gnn_hmm",
        "visualize_errors",
        "visualize_osm_overlay",
        "debug_gnn_hmm_data",
        "evaluate_mm2_style",
        "compare_mm2_style",
        "evaluate_exact_mm2_route_overlap",
    ],
    "data": [
        "prepare_trajectories",
        "prepare_gt_routes",
        "build_osm_graph",
        "build_line_graph",
        "generate_candidates",
        "build_training_tensors",
    ],
    "model": ["train_gnn_hmm", "decode_gnn_hmm", "evaluate_gnn_hmm"],
    "gpu_e2e": [
        "prepare_trajectories",
        "prepare_gt_routes",
        "build_osm_graph",
        "build_line_graph",
        "generate_candidates",
        "build_training_tensors",
        "train_gnn_hmm",
        "decode_gnn_hmm",
        "evaluate_gnn_hmm",
        "visualize_errors",
        "visualize_osm_overlay",
        "debug_gnn_hmm_data",
        "run_gnn_hmm_experiments",
        "evaluate_mm2_style",
        "compare_mm2_style",
        "evaluate_exact_mm2_route_overlap",
    ],
    "smoke": [
        "prepare_trajectories",
        "prepare_gt_routes",
        "build_osm_graph",
        "build_line_graph",
        "generate_candidates",
        "build_training_tensors",
        "debug_gnn_hmm_data",
    ],
}


LEGACY_STAGE_ALIASES: dict[str, str] = {
    "train": "train_gnn_hmm",
    "decode": "decode_gnn_hmm",
    "evaluate": "evaluate_gnn_hmm",
    "visualize": "visualize_errors",
    "osm_overlay": "visualize_osm_overlay",
    "debug_data": "debug_gnn_hmm_data",
}


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "scripts").exists():
            numbered_scripts = [candidate / "scripts" / f"{i:02d}_" for i in range(1, 17)]
            if any(path.parent.exists() for path in numbered_scripts):
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
        item = LEGACY_STAGE_ALIASES.get(item, item)

        if item in PIPELINE_ALIASES:
            for key in PIPELINE_ALIASES[item]:
                if key not in resolved:
                    resolved.append(key)
        elif item in known:
            if item not in resolved:
                resolved.append(item)
        else:
            valid = sorted(set(PIPELINE_ALIASES) | set(known) | set(LEGACY_STAGE_ALIASES))
            raise ValueError(f"Unknown stage or group: {item}. Valid options: {valid}")

    return resolved


def slice_stages(keys: list[str], from_stage: str | None, to_stage: str | None) -> list[str]:
    ordered = [stage.key for stage in STAGES]
    selected_ordered = [key for key in ordered if key in keys]

    if from_stage is not None:
        from_stage = LEGACY_STAGE_ALIASES.get(from_stage, from_stage)
        if from_stage not in ordered:
            raise ValueError(f"Unknown from-stage: {from_stage}")
        start_idx = ordered.index(from_stage)
        selected_ordered = [key for key in selected_ordered if ordered.index(key) >= start_idx]

    if to_stage is not None:
        to_stage = LEGACY_STAGE_ALIASES.get(to_stage, to_stage)
        if to_stage not in ordered:
            raise ValueError(f"Unknown to-stage: {to_stage}")
        end_idx = ordered.index(to_stage)
        selected_ordered = [key for key in selected_ordered if ordered.index(key) <= end_idx]

    return selected_ordered


def make_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    root_text = str(repo_root)
    env["PYTHONPATH"] = root_text + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        gdal_data = Path(conda_prefix) / "Library" / "share" / "gdal"
        proj_lib = Path(conda_prefix) / "Library" / "share" / "proj"

        if gdal_data.exists() and "GDAL_DATA" not in env:
            env["GDAL_DATA"] = str(gdal_data)

        if proj_lib.exists() and "PROJ_LIB" not in env:
            env["PROJ_LIB"] = str(proj_lib)

    return env


def run_command(
    command: list[str],
    repo_root: Path,
    log_path: Path,
    dry_run: bool = False,
) -> int:
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
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        log_file.write(f"\nFinished: {datetime.now().isoformat()}\n")
        log_file.write(f"Return code: {return_code}\n")

    return return_code


def merge_stage_args(stage: Stage, extra_args: list[str], no_defaults: bool) -> list[str]:
    if no_defaults:
        return list(extra_args)
    return list(stage.default_args) + list(extra_args)


def build_stage_command(
    python_executable: str,
    repo_root: Path,
    stage: Stage,
    extra_args: list[str],
    no_stage_defaults: bool,
) -> list[str]:
    script_path = repo_root / "scripts" / stage.script

    if not script_path.exists():
        raise FileNotFoundError(f"Stage script not found: {script_path}")

    return [
        python_executable,
        "-u",
        str(script_path),
        *merge_stage_args(stage, extra_args, no_stage_defaults),
    ]


def verify_stage_scripts(repo_root: Path, selected: list[str]) -> None:
    known = stage_by_key()
    missing = []

    for key in selected:
        path = repo_root / "scripts" / known[key].script
        if not path.exists():
            missing.append(path)

    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing selected pipeline scripts:\n{joined}")


def verify_required_inputs(repo_root: Path, selected: list[str]) -> None:
    verify_stage_scripts(repo_root, selected)

    required = [
        repo_root / "data" / "raw" / "trajectories" / "points.csv",
        repo_root / "data" / "raw" / "trajectories" / "ground_truth.csv",
    ]

    needs_osm = any(
        key in selected
        for key in [
            "build_osm_graph",
            "build_line_graph",
            "generate_candidates",
            "build_training_tensors",
            "train_gnn_hmm",
            "decode_gnn_hmm",
            "evaluate_gnn_hmm",
            "visualize_errors",
            "visualize_osm_overlay",
            "debug_gnn_hmm_data",
            "run_gnn_hmm_experiments",
            "evaluate_mm2_style",
            "compare_mm2_style",
            "evaluate_exact_mm2_route_overlap",
        ]
    )

    if needs_osm:
        required.append(repo_root / "data" / "raw" / "osm" / "oberfranken-latest.osm.pbf")

    missing = [path for path in required if not path.exists()]

    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")


def print_plan(stage_keys: list[str], no_stage_defaults: bool) -> None:
    known = stage_by_key()
    print("")
    print("Execution plan:")
    for idx, key in enumerate(stage_keys, start=1):
        stage = known[key]
        default_text = "" if no_stage_defaults or not stage.default_args else f" | defaults: {' '.join(stage.default_args)}"
        print(f"  {idx:02d}. {stage.key} [{stage.script}] - {stage.description}{default_text}")


def print_stage_list() -> None:
    print("Available stages:")
    for idx, stage in enumerate(STAGES, start=1):
        defaults = " ".join(stage.default_args) if stage.default_args else "-"
        print(f"  {idx:02d}. {stage.key:<34} {stage.group:<18} {stage.script}")
        print(f"      {stage.description}")
        print(f"      defaults: {defaults}")

    print("")
    print("Available groups:")
    for name, keys in sorted(PIPELINE_ALIASES.items()):
        print(f"  {name:<14} {', '.join(keys)}")

    print("")
    print("Legacy stage aliases:")
    for old, new in sorted(LEGACY_STAGE_ALIASES.items()):
        print(f"  {old:<14} -> {new}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified One-Direction pipeline runner for scripts 01 through 16.")

    parser.add_argument("stages", nargs="*", help="Stage keys or groups to run. Default: all scripts from 01 to 16.")
    parser.add_argument("--from-stage", default=None, help="Start from this stage key after resolving selection.")
    parser.add_argument("--to-stage", default=None, help="Stop at this stage key after resolving selection.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later stages even if one stage fails.")
    parser.add_argument("--skip-input-check", action="store_true", help="Skip raw input and script existence checks.")
    parser.add_argument("--list", action="store_true", help="List available stages and groups.")
    parser.add_argument("--log-dir", default="outputs/run_logs", help="Directory where stage logs are written.")
    parser.add_argument("--no-stage-defaults", action="store_true", help="Run scripts without stage-specific default arguments.")
    parser.add_argument(
        "--stage-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended to every selected stage after --stage-args.",
    )

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
        verify_required_inputs(repo_root, selected)

    print(f"Repository root: {repo_root}")
    print_plan(selected, no_stage_defaults=args.no_stage_defaults)

    known = stage_by_key()
    log_dir = repo_root / args.log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures: list[tuple[str, int]] = []

    for key in selected:
        stage = known[key]
        print("")
        print("=" * 100)
        print(f"Running stage: {stage.key} ({stage.script})")
        print("=" * 100)

        command = build_stage_command(
            python_executable=args.python,
            repo_root=repo_root,
            stage=stage,
            extra_args=args.stage_args,
            no_stage_defaults=args.no_stage_defaults,
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
