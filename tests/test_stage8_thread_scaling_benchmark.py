from business_entity_resolution.stage8_thread_scaling_benchmark import QUERY_COUNT, run

def test_thread_scaling_benchmark_contract():
    assert QUERY_COUNT == 5000
    assert callable(run)
