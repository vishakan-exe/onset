import pytest

from onset.learn import labels as L


@pytest.mark.parametrize("raw,want", [("Happiness", "happiness"), (" happy ", "happiness"), ("Sad", "sadness"), ("others", "others"),
                                      ("other", "others"), (None, None), ("", None), ("nan", None), ("N/A", None), ("Surprised", "surprise")])
def test_normalise(raw, want):
    assert L.normalise(raw) == want


def test_megc3_composite_scheme():
    m = lambda x: L.map_label(x, "megc3")                                          # noqa: E731
    assert m("happiness") == "positive" and m("surprise") == "surprise"
    for neg in ("disgust", "fear", "sadness", "anger", "contempt", "repression"):
        assert m(neg) == "negative"
    assert m("others") is None and m("tense") is None and m(None) is None         # outside the scheme: excluded, never relabelled


def test_casme2_scheme_folds_rare_labels_into_others():
    m = lambda x: L.map_label(x, "casme2_5")                                       # noqa: E731
    assert [m(x) for x in ("happiness", "disgust", "surprise", "repression", "others")] == ["happiness", "disgust", "surprise", "repression", "others"]
    assert m("fear") == "others" and m("tense") == "others"


def test_basic7_scheme_is_strict():
    assert L.map_label("anger", "basic7") == "anger"
    assert L.map_label("repression", "basic7") is None and L.map_label("others", "basic7") is None
    with pytest.raises(KeyError):
        L.map_label("anger", "nope")


@pytest.mark.parametrize("cell,want", [("R12 + R15", (12, 15)), ("4+7", (4, 7)), ("L14", (14,)), ("AU9, AU10", (9, 10)), ("R14A", (14,)),
                                       ("12 or 14", (12,)), ("1+2+5B", (1, 2, 5)), ("AU 12", (12,)), ("6+12", (6, 12)), ("", ()), (None, ()),
                                       ("nan", ()), ("L6+R12", (6, 12)), ("17(k)", ()), ("4 and 7", (4, 7)), ("99", ())])
def test_parse_aus(cell, want):
    assert L.parse_aus(cell) == want


def test_parse_aus_keeps_k_marks_when_asked():
    assert L.parse_aus("17(k)+4", drop_k=False) == (4, 17)
    assert L.parse_aus("17(k)+4") == (4,)
