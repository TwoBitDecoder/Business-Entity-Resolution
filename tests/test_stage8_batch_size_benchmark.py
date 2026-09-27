from business_entity_resolution.stage8_batch_size_benchmark import BATCH_SIZES, QUERY_COUNT, run

def test_batch_benchmark_contract():
    assert BATCH_SIZES == (1000, 5000, 10000)
    assert QUERY_COUNT == 10000
    assert callable(run)
