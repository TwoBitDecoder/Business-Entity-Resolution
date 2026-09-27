import pytest

from business_entity_resolution.production_benchmark import run_production_benchmark


def test_invalid_chunk_size_rejected():
    with pytest.raises(ValueError):
        run_production_benchmark(chunk_size=0)
