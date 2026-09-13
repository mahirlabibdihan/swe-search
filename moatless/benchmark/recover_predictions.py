"""Recover SWE-bench predictions by scanning every trajectory in a named run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from moatless.benchmark.export_predictions import patch_from_trajectory


def find_run_dir(run_id: str) -> Path:
    """Find a run under MOATLESS_DIR or the conventional local directories."""
    roots = []
    if moatless_dir := os.getenv("MOATLESS_DIR"):
        roots.append(Path(moatless_dir))
    roots.extend((Path("evaluations"), Path("evals")))

    for root in roots:
        run_dir = root / run_id
        if run_dir.is_dir():
            return run_dir.resolve()

    searched = "\n".join(f"  {root / run_id}" for root in roots)
    raise FileNotFoundError(f"Run '{run_id}' was not found. Searched:\n{searched}")


def configured_model(run_dir: Path) -> str:
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        return "unknown-model"

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return (
        evaluation.get("settings", {})
        .get("model", {})
        .get("model", "unknown-model")
    )


def load_predictions(output_path: Path) -> dict[str, dict]:
    """Load existing predictions, keyed by instance ID."""
    if not output_path.exists():
        return {}

    data = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {output_path}")

    predictions = {}
    for prediction in data:
        if not isinstance(prediction, dict) or not prediction.get("instance_id"):
            raise ValueError(f"Invalid prediction entry in {output_path}")
        predictions[prediction["instance_id"]] = prediction
    return predictions


def recover_predictions(run_id: str, overwrite: bool = False) -> Path:
    run_dir = find_run_dir(run_id)
    model_name = f"{run_id}__{configured_model(run_dir).replace('/', '__')}"
    trajectories = sorted(run_dir.glob("*/trajectory.json"))

    output_path = run_dir / "predictions.json"
    predictions_by_id = load_predictions(output_path)
    existing_ids = set(predictions_by_id)

    skipped_existing = 0
    recovered = 0
    no_patch = 0
    errors = 0

    for trajectory_path in tqdm(
        trajectories, desc="Recovering patches", unit="trajectory"
    ):
        instance_id = trajectory_path.parent.name
        if instance_id in existing_ids and not overwrite:
            skipped_existing += 1
            continue

        try:
            patch = patch_from_trajectory(trajectory_path)
        except Exception as exc:
            tqdm.write(f"Error reading {instance_id}: {exc}")
            errors += 1
            continue

        if not patch:
            no_patch += 1
            continue

        predictions_by_id[instance_id] = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }
        recovered += 1

    predictions = [predictions_by_id[key] for key in sorted(predictions_by_id)]
    output_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Trajectories: {len(trajectories)}")
    print(f"Predictions:  {len(predictions)}")
    print(f"Recovered:    {recovered}")
    print(f"Existing:     {skipped_existing}")
    print(f"No patch:     {no_patch}")
    print(f"Errors:       {errors}")
    print(f"Saved to:     {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover predictions from every instance trajectory in a run, "
            "without relying on evaluation.json's instance list."
        )
    )
    parser.add_argument("run_id", help="Evaluation run ID")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace predictions that already exist",
    )
    args = parser.parse_args()

    try:
        recover_predictions(args.run_id, overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
