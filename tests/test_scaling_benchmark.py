import pytest

from business_entity_resolution.scaling_benchmark import run_scaling_benchmark


def test_scaling_benchmark_rejects_empty_sizes():
    with pytest.raises(ValueError):
        run_scaling_benchmark(targets_per_source=())


def test_scaling_benchmark_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        run_scaling_benchmark(targets_per_source=(100, 0))
