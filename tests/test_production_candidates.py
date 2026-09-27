from pathlib import Path
import polars as pl
import pytest
from business_entity_resolution.production_candidates import generate_country, NAME_K, ADDRESS_K, FINAL_K


def _write(root: Path, source: str, country: str, rows):
    p=root/source/f"country={country}"/"records.parquet"; p.parent.mkdir(parents=True,exist_ok=True)
    pl.DataFrame(rows).write_parquet(p)


def test_production_generator_chunks_and_bounds(tmp_path):
    root=tmp_path/"pre"; out=tmp_path/"out"; c="X"
    _write(root,"source1",c,{"entity_id":["q1","q2"],"name_compact":["alphaco","betaco"],"address_norm":["1 main","2 road"]})
    _write(root,"source2",c,{"entity_id":["a","b"],"name_compact":["alphaco","other"],"address_norm":["1 main","9 lane"]})
    _write(root,"source3",c,{"entity_id":["c","d"],"name_compact":["betaco","none"],"address_norm":["2 road","8 street"]})
    s=generate_country(root,out,c,chunk_size=1)
    assert s["chunks"]==2
    assert s["max_candidates_per_query"] <= FINAL_K
    parts=sorted((out/f"country={c}").glob("part-*.parquet"))
    assert len(parts)==2
    got=pl.concat([pl.read_parquet(p) for p in parts])
    assert got.select(["s1_id","target_id"]).n_unique()==got.height
    assert {"q1","q2"} <= set(got["s1_id"])


def test_production_config_is_frozen_r56():
    assert (NAME_K,ADDRESS_K,FINAL_K)==(20,20,40)


def test_production_generator_rejects_bad_chunk(tmp_path):
    with pytest.raises(ValueError):
        generate_country(tmp_path,tmp_path/"out","X",chunk_size=0)
