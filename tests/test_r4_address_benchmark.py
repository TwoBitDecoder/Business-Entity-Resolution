import polars as pl

from business_entity_resolution.r4_address_benchmark import _max_rss_mb


def test_r4_rss_measurement_is_positive():
    assert _max_rss_mb() > 0


def test_empty_address_count_logic():
    frame = pl.DataFrame({"address_norm": ["", "10 market road", ""]})
    assert frame.filter(pl.col("address_norm") != "").height == 1
