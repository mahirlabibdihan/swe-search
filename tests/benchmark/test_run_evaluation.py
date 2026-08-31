import pytest

from moatless.benchmark.run_evaluation import load_dataset_split, slice_instance_ids


def test_load_verified_dataset_split():
    dataset = load_dataset_split("verified")

    assert dataset is not None
    assert dataset.name == "verified"
    assert len(dataset.instance_ids) == 500


@pytest.mark.parametrize(
    ("slice_spec", "expected"),
    [
        (None, ["a", "b", "c", "d", "e"]),
        ("0:2", ["a", "b"]),
        ("2:", ["c", "d", "e"]),
        ("::2", ["a", "c", "e"]),
        ("1:5:2", ["b", "d"]),
        ("2", ["a", "b"]),
    ],
)
def test_slice_instance_ids(slice_spec, expected):
    assert slice_instance_ids(["a", "b", "c", "d", "e"], slice_spec) == expected


@pytest.mark.parametrize("slice_spec", ["a:2", "1:2:3:4", "::0"])
def test_slice_instance_ids_rejects_invalid_slice(slice_spec):
    with pytest.raises(ValueError):
        slice_instance_ids(["a", "b", "c"], slice_spec)
