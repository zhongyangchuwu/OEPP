"""Run the Q3.2 cluster-pair experiment across one process per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from oepp.data.cluster_pair import PAIR_CONDITIONS, PAIR_PLAN_SCHEMA
from oepp.training.support import canonical_json_hash, utc_now, write_json

RUN_SCHEMA = "oepp-q32-cluster-pair-run-v1"
MODEL_TEMPLATES = {
    "mlp": Path("configs/training/mlp.yaml"),
    "transformer": Path("configs/training/transformer.yaml"),
}


@dataclass(frozen=True)
class RunTask:
    model: str
    fold_index: int
    condition: str
    split_id: str
    config_path: Path
    run_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.model}-f{self.fold_index:02d}-{self.condition}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be an object: {path}")
    return value


def build_tasks(
    *,
    plan: dict[str, Any],
    output_root: Path,
    models: tuple[str, ...],
    epochs: int,
) -> tuple[RunTask, ...]:
    if plan.get("schema") != PAIR_PLAN_SCHEMA:
        raise ValueError("unsupported cluster-pair plan schema")
    folds = plan.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("cluster-pair plan requires non-empty folds")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    unknown_models = sorted(set(models).difference(MODEL_TEMPLATES))
    if unknown_models:
        raise ValueError(f"unsupported models: {unknown_models}")

    tasks: list[RunTask] = []
    for model in models:
        template_path = MODEL_TEMPLATES[model]
        template = _load_yaml(template_path)
        for fold in folds:
            fold_index = int(fold["fold_index"])
            splits = fold.get("splits")
            if not isinstance(splits, dict):
                raise ValueError(f"fold {fold_index} lacks split definitions")
            for condition in PAIR_CONDITIONS:
                split_id = str(splits[condition]["split_id"])
                config = json.loads(json.dumps(template))
                config["data"]["split_id"] = split_id
                config["training"]["epochs"] = epochs
                config["experiment_id"] = (
                    f"{plan['protocol_id']}:{model}:fold-{fold_index:02d}:{condition}"
                )
                config_path = (
                    output_root / "configs" / model / f"fold-{fold_index:02d}-{condition}.yaml"
                )
                config_path.parent.mkdir(parents=True, exist_ok=True)
                rendered = yaml.safe_dump(config, sort_keys=True)
                if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
                    raise ValueError(f"existing generated config differs: {config_path}")
                config_path.write_text(rendered, encoding="utf-8")
                tasks.append(
                    RunTask(
                        model=model,
                        fold_index=fold_index,
                        condition=condition,
                        split_id=split_id,
                        config_path=config_path,
                        run_dir=(
                            output_root
                            / "runs"
                            / model
                            / f"fold-{fold_index:02d}"
                            / condition
                        ),
                    )
                )
    return tuple(tasks)


def _run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}: {' '.join(command)}")


def _execute_task(
    task: RunTask,
    *,
    device: str,
    data_root: Path,
    videoclip_root: Path,
    eval_batch_size: int,
) -> dict[str, Any]:
    task.run_dir.parent.mkdir(parents=True, exist_ok=True)
    status_path = task.run_dir / "cluster_pair_status.json"
    selection_path = task.run_dir / "selection.json"
    best_path = task.run_dir / "best.pt"
    export_dir = task.run_dir / "exports"
    export_artifacts = (
        export_dir / "run_metadata.json",
        export_dir / "base_metrics_per_window_step.csv",
        export_dir / "novel_metrics_per_window_step.csv",
    )
    metrics_path = task.run_dir / "planning_metrics.json"
    required = (selection_path, best_path, *export_artifacts, metrics_path)
    if all(path.is_file() for path in required) and status_path.is_file():
        previous = _load_json(status_path)
        if previous.get("status") == "passed":
            return {**previous, "device": device, "status": "already_complete"}

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = device
    env["OEPP_VIDEOCLIP_ROOT"] = str(videoclip_root)
    log_path = task.run_dir.with_suffix(".log")
    started_at = utc_now()
    try:
        if metrics_path.exists() and not all(path.is_file() for path in required):
            raise FileExistsError(
                f"metrics exist without a complete audited task: {task.run_dir}"
            )
        if not selection_path.is_file():
            train_command = [
                sys.executable,
                "-m",
                "oepp.cli",
                "train",
                "--config",
                str(task.config_path),
                "--data-root",
                str(data_root),
                "--run-dir",
                str(task.run_dir),
                "--eval-batch-size",
                str(eval_batch_size),
                "--device",
                "cuda:0",
            ]
            if task.run_dir.is_dir():
                if (task.run_dir / "last.pt").is_file() and (task.run_dir / "best.pt").is_file():
                    train_command.append("--resume")
                else:
                    raise FileExistsError(
                        f"incomplete run directory cannot be resumed safely: {task.run_dir}"
                    )
            _run_command(train_command, env=env, log_path=log_path)

        if export_dir.exists() and not all(path.is_file() for path in export_artifacts):
            raise FileExistsError(f"partial export directory requires audit: {export_dir}")
        if not export_dir.is_dir():
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "oepp.cli",
                    "export",
                    "--checkpoint",
                    str(task.run_dir / "best.pt"),
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(export_dir),
                    "--batch-size",
                    str(eval_batch_size),
                    "--trust-checkpoint",
                    "--device",
                    "cuda:0",
                ],
                env=env,
                log_path=log_path,
            )
        if not metrics_path.is_file():
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "oepp.cli",
                    "summarize-planning-metrics",
                    "--base-csv",
                    str(export_dir / "base_metrics_per_window_step.csv"),
                    "--novel-csv",
                    str(export_dir / "novel_metrics_per_window_step.csv"),
                    "--output",
                    str(metrics_path),
                ],
                env=env,
                log_path=log_path,
            )
        result = {
            "schema": RUN_SCHEMA,
            "task_id": task.task_id,
            "model": task.model,
            "fold_index": task.fold_index,
            "condition": task.condition,
            "split_id": task.split_id,
            "device": device,
            "started_at": started_at,
            "completed_at": utc_now(),
            "status": "passed",
            "config": str(task.config_path),
            "config_hash": canonical_json_hash(_load_yaml(task.config_path)),
            "run_dir": str(task.run_dir),
            "metrics": str(metrics_path),
            "log": str(log_path),
        }
    except BaseException as error:
        result = {
            "schema": RUN_SCHEMA,
            "task_id": task.task_id,
            "model": task.model,
            "fold_index": task.fold_index,
            "condition": task.condition,
            "split_id": task.split_id,
            "device": device,
            "started_at": started_at,
            "completed_at": utc_now(),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "run_dir": str(task.run_dir),
            "log": str(log_path),
        }
    task.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, result)
    return result


def run_plan(
    *,
    plan_path: Path,
    output_root: Path,
    data_root: Path,
    videoclip_root: Path,
    models: tuple[str, ...],
    devices: tuple[str, ...],
    epochs: int,
    eval_batch_size: int,
) -> dict[str, Any]:
    if not devices:
        raise ValueError("at least one GPU device is required")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    plan = _load_json(plan_path)
    tasks = build_tasks(plan=plan, output_root=output_root, models=models, epochs=epochs)
    buckets = [list(tasks[index:: len(devices)]) for index in range(len(devices))]

    def worker(device: str, assigned: list[RunTask]) -> list[dict[str, Any]]:
        return [
            _execute_task(
                task,
                device=device,
                data_root=data_root,
                videoclip_root=videoclip_root,
                eval_batch_size=eval_batch_size,
            )
            for task in assigned
        ]

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, bucket)
            for device, bucket in zip(devices, buckets)
        ]
        results = [result for future in futures for result in future.result()]
    report = {
        "schema": RUN_SCHEMA,
        "protocol_id": plan["protocol_id"],
        "plan": str(plan_path),
        "plan_sha256": __import__("hashlib").sha256(plan_path.read_bytes()).hexdigest(),
        "created_at": utc_now(),
        "models": list(models),
        "devices": list(devices),
        "epochs": epochs,
        "tasks": sorted(results, key=lambda result: result["task_id"]),
    }
    report["passed"] = all(result["status"] in {"passed", "already_complete"} for result in results)
    write_json(output_root / "run_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Q3.2 cluster-pair training tasks.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--videoclip-root", type=Path, default=Path("features/OEPP_videoclip"))
    parser.add_argument("--models", nargs="+", default=["mlp", "transformer"])
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    arguments = parser.parse_args()
    devices = tuple(device.strip() for device in arguments.devices.split(",") if device.strip())
    report = run_plan(
        plan_path=arguments.plan,
        output_root=arguments.output_root,
        data_root=arguments.data_root.resolve(),
        videoclip_root=arguments.videoclip_root.resolve(),
        models=tuple(arguments.models),
        devices=devices,
        epochs=arguments.epochs,
        eval_batch_size=arguments.eval_batch_size,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "tasks": len(report["tasks"]),
                "failed": sum(result["status"] == "failed" for result in report["tasks"]),
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
