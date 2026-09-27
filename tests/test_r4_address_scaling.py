import json

import pytest

import business_entity_resolution.r4_address_scaling as r4


def test_default_address_scaling_ladder():
    assert r4.DEFAULT_TOTAL_TARGETS == (250_000, 500_000, 1_000_000)


def test_rejects_odd_target_count():
    with pytest.raises(ValueError, match="positive and even"):
        r4.run_scaling_ladder(total_targets=(3,))


def test_stops_before_larger_run_when_rss_gate_is_hit(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        payload = {
            "target_rows": 250000,
            "process_max_rss_mb": 10000.0,
            "candidate_rows": 10000,
            "max_candidates_per_query": 10,
        }
        class Result:
            stdout = json.dumps(payload) + "\nSaved: ignored.json\n"
        return Result()

    monkeypatch.setattr(r4.subprocess, "run", fake_run)
    result = r4.run_scaling_ladder(total_targets=(250000, 500000))
    assert result["stopped_early"] is True
    assert len(calls) == 1
    assert len(result["runs"]) == 1


def test_stops_when_requested_target_pool_is_incomplete(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        payload = {
            "target_rows": 249999,
            "process_max_rss_mb": 500.0,
            "candidate_rows": 10000,
            "max_candidates_per_query": 10,
        }
        class Result:
            stdout = json.dumps(payload) + "\nSaved: ignored.json\n"
        return Result()

    monkeypatch.setattr(r4.subprocess, "run", fake_run)
    result = r4.run_scaling_ladder(total_targets=(250000, 500000))
    assert result["stopped_early"] is True
    assert result["runs"][0]["target_count_complete"] is False
    assert len(calls) == 1
