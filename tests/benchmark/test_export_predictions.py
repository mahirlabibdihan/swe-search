import json

import pytest

from moatless.benchmark import export_predictions as exporter
from moatless.actions.finish import FinishArgs
from moatless.discriminator import MeanAwardDiscriminator
from moatless.node import ActionStep, Node
from moatless.search_tree import SearchTree
from moatless.value_function.model import Reward


def make_tree():
    root = Node(node_id=0)
    first = Node(node_id=1, reward=Reward.model_construct(value=10))
    second = Node(node_id=2, reward=Reward.model_construct(value=20))
    later = Node(node_id=3, reward=Reward.model_construct(value=100))
    root.add_child(first)
    root.add_child(second)
    first.add_child(later)  # DFS order is 0, 1, 3, 2, unlike creation order.
    tree = SearchTree.model_construct(
        root=root, max_iterations=4, discriminator=MeanAwardDiscriminator(),
    )
    for node in [first, second, later]:
        tree._backpropagate(node)
    return tree


def test_cutoff_removes_future_reward_and_restores_leaf_selection():
    tree = make_tree()
    exporter.truncate_tree(tree, 3)
    assert [n.node_id for n in tree.root.get_all_nodes()] == [0, 1, 2]
    assert tree.root.visits == 2
    assert tree.root.value == 30
    assert tree.root.children[0].visits == 1
    assert tree.root.children[0].value == 10
    assert tree.get_best_trajectory().node_id == 2


def test_root_only_and_full_budget():
    tree = exporter.truncate_tree(make_tree(), 1)
    assert tree.root.children == []
    assert tree.root.visits == 0
    assert tree.get_best_trajectory() is tree.root
    tree = make_tree()
    before = [(n.node_id, n.value, n.visits) for n in tree.root.get_all_nodes()]
    exporter.truncate_tree(tree, 4)
    assert [(n.node_id, n.value, n.visits) for n in tree.root.get_all_nodes()] == before


def test_finished_candidate_is_preferred_to_higher_reward_leaf():
    tree = make_tree()
    tree.root.children[0].action_steps = [ActionStep(
        action=FinishArgs(thoughts="done", finish_reason="done"),
    )]
    exporter.truncate_tree(tree, 3)
    assert tree.get_best_trajectory().node_id == 1


@pytest.mark.parametrize("cutoff", [0, -1, 5])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError):
        exporter.truncate_tree(make_tree(), cutoff)


def test_early_stop_warns(caplog):
    tree = make_tree()
    tree.max_iterations = 51
    exporter.truncate_tree(tree, 11)
    assert len(tree.root.get_all_nodes()) == 4
    assert "only 4 nodes" in caplog.text


def test_export_ignores_final_submission_at_cutoff(tmp_path, monkeypatch):
    (tmp_path / "evaluation.json").write_text(json.dumps({
        "evaluation_name": "test", "instances": [
            {"instance_id": "task", "submission": "future patch"},
        ],
    }))
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "trajectory.json").write_text("{}")
    calls = []

    def patch(path, max_iterations=None):
        calls.append(max_iterations)
        return "prefix patch"

    monkeypatch.setattr(exporter, "patch_from_trajectory", patch)
    output = tmp_path / "predictions.json"
    assert exporter.export_predictions(tmp_path, output, max_iterations=11) == (1, 0)
    assert calls == [11]
    assert json.loads(output.read_text())[0]["model_patch"] == "prefix patch"
    exporter.export_predictions(tmp_path, output)
    assert json.loads(output.read_text())[0]["model_patch"] == "future patch"


def test_missing_trajectory_is_not_silently_an_empty_patch(tmp_path):
    (tmp_path / "evaluation.json").write_text(json.dumps({
        "instances": [{"instance_id": "missing", "submission": "final"}],
    }))
    with pytest.raises(FileNotFoundError):
        exporter.export_predictions(
            tmp_path, tmp_path / "out.json", include_empty=True, max_iterations=11,
        )
