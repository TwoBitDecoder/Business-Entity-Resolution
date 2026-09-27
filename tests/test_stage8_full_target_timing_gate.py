from business_entity_resolution.stage8_full_target_timing_gate import QUERY_COUNT, run

def test_full_target_timing_gate_contract():
    assert QUERY_COUNT == 1000
    assert callable(run)
