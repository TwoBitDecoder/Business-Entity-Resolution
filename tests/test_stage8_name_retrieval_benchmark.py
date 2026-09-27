from business_entity_resolution.stage8_name_retrieval_benchmark import K, run

def test_stage8_benchmark_contract():
    assert K == 20
    assert callable(run)
