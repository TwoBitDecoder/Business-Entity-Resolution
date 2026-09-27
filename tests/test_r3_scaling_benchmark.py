import pytest

from business_entity_resolution.r3_scaling_benchmark import DEFAULT_TOTAL_TARGETS


def test_r3_scaling_ladder_is_monotonic_and_ends_at_one_million():
    assert DEFAULT_TOTAL_TARGETS == (125_000, 250_000, 500_000, 1_000_000)
    assert list(DEFAULT_TOTAL_TARGETS) == sorted(DEFAULT_TOTAL_TARGETS)
    assert DEFAULT_TOTAL_TARGETS[-1] == 1_000_000


def test_r3_target_sizes_split_evenly():
    assert all(total % 2 == 0 for total in DEFAULT_TOTAL_TARGETS)
