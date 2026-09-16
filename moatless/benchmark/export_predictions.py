"""Export a Moatless evaluation as SWE-bench harness predictions."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    if len(nodes) <= max_iterations:
        # Nothing to prune. Preserve the saved selector statistics, including
        # any repeated updates from a resumed run.
        tree.max_iterations = max_iterations
        return tree

    # Check whether one backpropagation per saved node explains the final
    # statistics. Retries/deletions can destroy history that a final snapshot
    # cannot recover. A matching result is a consistency check, not proof that
    # no file state was ever overwritten.
    expected = {node.node_id: [0, 0.0] for node in nodes}
    for node in ordered[1:]:
        if node.reward is None:
            continue
        ancestor = node
        while ancestor is not None:
            expected[ancestor.node_id][0] += 1
            expected[ancestor.node_id][1] += node.reward.value
            ancestor = ancestor.parent
    inconsistent = any(
        node.visits != expected[node.node_id][0]
        or not math.isclose(node.value or 0.0, expected[node.node_id][1], abs_tol=1e-9)
        for node in nodes
    )
    missing_ids = [node.node_id for node in ordered] != list(range(len(nodes)))
    if inconsistent or missing_ids:
        logger.warning(
            "Historical cutoff may be approximate: saved node IDs or reward "
            "statistics do not match a single append-only run. Retries or "
            "removed nodes cannot be reconstructed from the final snapshot."
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
    if not evaluation_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {evaluation_dir}")
    prediction_model = model_name or evaluation_dir.name
    if max_iterations is not None and model_name is None:
        prediction_model += f"__iterations_{max_iterations}"

    predictions = []
    skipped = 0
    instance_dirs = sorted(
        path for path in evaluation_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not instance_dirs:
        raise ValueError(f"No instance folders found in {evaluation_dir}")
    tqdm.write(
        f"Found {len(instance_dirs)} instance folders in {evaluation_dir}"
    )

    description = (
        f"Exporting max_iterations={max_iterations}"
        if max_iterations is not None else "Exporting final patches"
    )
    for instance_dir in tqdm(
        instance_dirs,
        desc=description,
        unit="instance",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        instance_id = instance_dir.name
        trajectory_path = instance_dir / "trajectory.json"
        patch = None
        if not trajectory_path.exists():
            logger.warning(
                "Missing trajectory for %s at cutoff %s: %s",
                instance_id, max_iterations, trajectory_path,
            )
        else:
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
