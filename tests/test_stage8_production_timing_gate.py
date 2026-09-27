from business_entity_resolution.stage8_production_timing_gate import QUERY_COUNT, TARGET_LIMIT_PER_SOURCE, run

def test_production_timing_gate_contract():
    assert QUERY_COUNT == 5000
    assert TARGET_LIMIT_PER_SOURCE == 250000
    assert callable(run)
