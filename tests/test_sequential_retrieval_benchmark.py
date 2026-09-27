import pytest

from business_entity_resolution.sequential_retrieval_benchmark import (
    run_sequential_benchmark,
)


def test_invalid_sequential_chunk_size_rejected():
    with pytest.raises(ValueError):
        run_sequential_benchmark(chunk_size=0)
