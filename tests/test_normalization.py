from business_entity_resolution.normalization import compact_text, normalize_text, numeric_tokens


def test_normalize_text():
    assert normalize_text("  A.B.C.  Pvt Ltd ") == "a b c pvt ltd"


def test_compact_text():
    assert compact_text("M.G. Road") == "mgroad"


def test_numeric_tokens():
    assert numeric_tokens("Plot 14, PIN 700001") == ("14", "700001")
