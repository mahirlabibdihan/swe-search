"""Export a Moatless evaluation as SWE-bench harness predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from moatless.search_tree import SearchTree

logger = logging.getLogger(__name__)


def truncate_tree(tree: SearchTree, max_iterations: int) -> SearchTree:
    """Prune a freshly loaded tree and rebuild prefix-only backpropagation.

    max_iterations counts the root, just like SearchTree.is_finished().
    Node IDs encode creation order; traversal order does not.
    This reconstructs an ordinary append-only run, not historical edits/retries.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1 (the root)")
    nodes = tree.root.get_all_nodes()
    ordered = sorted(nodes, key=lambda node: node.node_id)
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("Trajectory has duplicate node IDs")
    if ordered[0] is not tree.root or any(
        node.parent is not None and node.parent.node_id >= node.node_id
        for node in nodes
    ):
        raise ValueError("Trajectory node IDs do not follow creation order")
    if max_iterations > tree.max_iterations:
        raise ValueError(
            f"Requested max_iterations={max_iterations} exceeds saved run limit "
            f"{tree.max_iterations}; cannot reconstruct additional search"
        )
    if len(nodes) < max_iterations:
        logger.warning(
            "Saved tree contains only %s nodes for requested cutoff %s; "
            "using all available nodes (run may have stopped early)",
            len(nodes), max_iterations,
        )
    retained = ordered[:max_iterations]
    retained_ids = {node.node_id for node in retained}
    for node in retained:
        node.children = [c for c in node.children if c.node_id in retained_ids]
        node.visits = 0
        node.value = 0.0
    # Ancestors' final values include later descendants. Recompute them so
    # MeanAwardDiscriminator cannot use information beyond this cutoff.
    for node in retained[1:]:
        tree._backpropagate(node)
    tree.max_iterations = max_iterations
    return tree


def patch_from_trajectory(
    trajectory_path: Path, max_iterations: int | None = None,
) -> str | None:
    """Return the patch from the trajectory selected by its discriminator."""
    if not trajectory_path.exists():
        return None

    tree = SearchTree.from_file(str(trajectory_path))
    if max_iterations is not None:
        tree = truncate_tree(tree, max_iterations)
    best_node = tree.get_best_trajectory()
    if not best_node or not best_node.file_context:
        return None
    return best_node.file_context.generate_git_patch()


def export_predictions(
    evaluation_dir: Path,
    output_path: Path,
    model_name: str | None = None,
    include_empty: bool = False,
    max_iterations: int | None = None,
    show_progress: bool = True,
) -> tuple[int, int]:
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
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
    if max_iterations is not None and model_name is None:
        prediction_model += f"__iterations_{max_iterations}"

    predictions = []
    skipped = 0
    seen_ids: set[str] = set()

    description = (
        f"Exporting max_iterations={max_iterations}"
        if max_iterations is not None else "Exporting final patches"
    )
    for instance in tqdm(
        evaluation.get("instances", []),
        desc=description,
        unit="instance",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        instance_id = instance.get("instance_id")
        if not instance_id:
            logger.warning("Skipping evaluation entry without instance_id")
            skipped += 1
            continue
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id in evaluation: {instance_id}")
        seen_ids.add(instance_id)

        # A saved submission belongs to the final run, never an earlier cutoff.
        patch = instance.get("submission") if max_iterations is None else None
        if patch is None:
            trajectory_path = evaluation_dir / instance_id / "trajectory.json"
            if max_iterations is not None and not trajectory_path.exists():
                raise FileNotFoundError(
                    f"Cannot reconstruct cutoff without {trajectory_path}"
                )
            patch = patch_from_trajectory(
                trajectory_path, max_iterations=max_iterations,
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
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar",
    )
    parser.add_argument(
        "--max-iterations", type=int, nargs="+",
        help="One or more root-inclusive cutoffs, e.g. 11 21 31 41 51",
    )
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
    if args.max_iterations and any(n < 1 for n in args.max_iterations):
        parser.error("--max-iterations values must be at least 1")
    if args.output and args.max_iterations and len(args.max_iterations) > 1:
        parser.error("--output requires a single cutoff")

    evaluation_dir = args.evaluation_dir.resolve()
    for cutoff in dict.fromkeys(args.max_iterations or [None]):
        filename = (
            f"predictions.iterations_{cutoff}.json"
            if cutoff is not None else "predictions.json"
        )
        output_path = (args.output or evaluation_dir / filename).resolve()
        exported, skipped = export_predictions(
            evaluation_dir=evaluation_dir,
            output_path=output_path,
            model_name=args.model_name,
            include_empty=args.include_empty,
            max_iterations=cutoff,
            show_progress=not args.no_progress,
        )
        print(f"Exported {exported} predictions to {output_path}")
        if skipped:
            print(f"Skipped {skipped} instances without usable patches")


if __name__ == "__main__":
    main()
