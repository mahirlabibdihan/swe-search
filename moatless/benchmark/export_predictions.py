"""Export a Moatless evaluation as SWE-bench harness predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from moatless.search_tree import SearchTree

logger = logging.getLogger(__name__)


def patch_from_trajectory(trajectory_path: Path) -> str | None:
    """Return the patch from the trajectory selected by its discriminator."""
    if not trajectory_path.exists():
        return None

    tree = SearchTree.from_file(str(trajectory_path))
    best_node = tree.get_best_trajectory()
    if not best_node or not best_node.file_context:
        return None
    return best_node.file_context.generate_git_patch()


def export_predictions(
    evaluation_dir: Path,
    output_path: Path,
    model_name: str | None = None,
    include_empty: bool = False,
) -> tuple[int, int]:
    evaluation_path = evaluation_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_path}")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_name = evaluation.get("evaluation_name", evaluation_dir.name)
    configured_model = (
        evaluation.get("settings", {}).get("model", {}).get("model", "unknown-model")
    )
    prediction_model = model_name or (
        f"{evaluation_name}__{configured_model.replace('/', '__')}"
    )

    predictions = []
    skipped = 0
    seen_ids: set[str] = set()

    for instance in evaluation.get("instances", []):
        instance_id = instance.get("instance_id")
        if not instance_id:
            logger.warning("Skipping evaluation entry without instance_id")
            skipped += 1
            continue
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id in evaluation: {instance_id}")
        seen_ids.add(instance_id)

        patch = instance.get("submission")
        if patch is None:
            patch = patch_from_trajectory(
                evaluation_dir / instance_id / "trajectory.json"
            )

        if not patch and not include_empty:
            logger.warning(
                "Skipping %s because no generated patch was found", instance_id
            )
            skipped += 1
            continue

        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": prediction_model,
                "model_patch": patch or "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(predictions), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Moatless evaluation to SWE-bench predictions JSON"
    )
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: <evaluation_dir>/predictions.json)",
    )
    parser.add_argument(
        "--model-name",
        help="Override model_name_or_path in exported predictions",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include instances with no patch (the harness skips empty patches)",
    )
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    output_path = (args.output or evaluation_dir / "predictions.json").resolve()
    exported, skipped = export_predictions(
        evaluation_dir=evaluation_dir,
        output_path=output_path,
        model_name=args.model_name,
        include_empty=args.include_empty,
    )
    print(f"Exported {exported} predictions to {output_path}")
    if skipped:
        print(f"Skipped {skipped} instances without usable patches")


if __name__ == "__main__":
    main()
