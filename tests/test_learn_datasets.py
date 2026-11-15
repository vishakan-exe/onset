import os

import cv2
import numpy as np
import pandas as pd
import pytest

from onset.learn import datasets as D


def make_folder(path, numbers):
    os.makedirs(path, exist_ok=True)
    for n in numbers:
        cv2.imwrite(os.path.join(path, f"img{n}.jpg"), np.full((8, 8, 3), n % 250, np.uint8))


@pytest.fixture
def casme_like(tmp_path):
    root = tmp_path / "CASME2"
    make_folder(root / "sub01" / "EP01_01f", range(1, 41))
    make_folder(root / "sub01" / "EP01_02f", range(1, 41))
    make_folder(root / "sub02" / "EP02_01f", range(101, 141))                      # numbered from 101: sheet numbers match the file names
    make_folder(root / "sub03" / "EP03_01f", range(501, 541))                      # sheet counts from 1 inside the folder: position fallback
    rows = [
        # Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame, Action Units, Estimated Emotion
        [1, "EP01_01f", 5, 20, 35, "R12+R15", "Happiness"],
        [1, "EP01_02f", 10, 25, 30, "4+7", "Disgust"],
        [2, "EP02_01f", 110, 120, 130, "AU1, AU2", "surprise"],
        [3, "EP03_01f", 3, 10, 25, "14", "others"],
        [1, "EP01_99f", 5, 20, 35, "12", "happiness"],                              # folder does not exist
        [1, "EP01_01f", 30, 20, 10, "12", "happiness"],                             # offset before onset
        [2, "EP02_01f", 300, 310, 320, "12", "happiness"],                          # frames not in the folder
        [2, "EP02_01f", "x", 5, 9, "12", "happiness"],                              # unreadable onset
        [3, "EP03_01f", 3, 99, 25, "", None],                                       # apex outside [onset, offset]; no label
    ]
    df = pd.DataFrame(rows, columns=["Subject", "Filename", "OnsetFrame", "ApexFrame", "OffsetFrame", "Action Units", "Estimated Emotion"])
    df.to_csv(root / "CASME2-coding-test.csv", index=False)
    return root


def test_casme2_adapter_resolves_clips_and_reports_problems(casme_like):
    man = D.casme2(str(casme_like))
    assert man.n_rows == 9 and len(man.records) == 5
    assert len(man.problems) == 4
    why = " | ".join(p["problem"] for p in man.problems)
    assert "no images" in why and "bad onset/offset" in why and "not in" in why
    r = {x.clip_id: x for x in man.records}
    a = r["EP01_01f"]
    assert (a.subject, a.dataset, a.emotion, a.aus, a.fps) == ("01", "casme2", "happiness", (12, 15), 200.0)
    assert a.uid == "casme2/01/EP01_01f" and a.group == "casme2:01" and a.extra["numbering"] == "file"
    assert r["EP02_01f"].extra["numbering"] == "file" and r["EP02_01f"].extra["n_frames"] == 21
    assert r["EP03_01f"].extra["numbering"] == "relative"
    last = man.records[-1]
    assert last.emotion is None and last.apex is None                                # apex outside the interval becomes 'measure it'
    assert len(man.annotation_sha256) == 64


def test_manifest_summary_and_scheme_selection(casme_like):
    man = D.casme2(str(casme_like))
    s = man.summary()
    assert s["subjects"] == 3 and s["emotions"]["(none)"] == 1
    keep, y, excluded = D.with_scheme(man.records, "megc3")
    assert [r.emotion for r in keep] == ["happiness", "disgust", "surprise"] and y.tolist() == [0, 1, 2]
    assert len(excluded) == 2                                                       # 'others' and the unlabelled clip are not relabelled
    keep5, y5, ex5 = D.with_scheme(man.records, "casme2_5")
    assert len(keep5) == 4 and len(ex5) == 1


def test_limit_and_missing_sheet(casme_like, tmp_path):
    assert len(D.casme2(str(casme_like), limit=2).records) == 2
    with pytest.raises(FileNotFoundError):
        D.casme2(str(tmp_path / "empty_root_does_not_exist"))
    os.makedirs(tmp_path / "e")
    with pytest.raises(FileNotFoundError, match="pass --annotations"):
        D.casme2(str(tmp_path / "e"))


def test_missing_column_error_lists_what_the_sheet_has(tmp_path):
    pd.DataFrame({"Subject": [1], "Clip": ["a"], "Start": [1]}).to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(ValueError) as e:
        D.generic(str(tmp_path), str(tmp_path / "s.csv"))
    assert "Subject" in str(e.value) and "Start" in str(e.value)


def test_samm_style_sheet_with_text_above_the_header(tmp_path):
    root = tmp_path / "SAMM"
    make_folder(root / "006" / "006_1_2", range(1, 31))
    pre = [["SAMM micro-movement coding"], ["Version 2"], [], ["notes"]]
    head = ["Subject", "Filename", "Onset Frame", "Apex Frame", "Offset Frame", "Duration", "Action Units", "Estimated Emotion"]
    rows = [[6, "006_1_2", 4, 12, 22, 18, "R12", "Happiness"]]
    pd.DataFrame(pre + [head] + rows).to_excel(root / "SAMM_Micro_FACS_Codes_v2.xlsx", header=False, index=False)
    man = D.samm(str(root))
    assert len(man.records) == 1 and man.records[0].subject == "006" and man.records[0].aus == (12,)


def test_generic_adapter_with_folder_template(tmp_path):
    root = tmp_path / "mine"
    make_folder(root / "alice" / "c1", range(1, 21))
    pd.DataFrame({"subject": ["alice"], "clip": ["c1"], "onset": [2], "offset": [15], "emotion": ["Surprise"], "aus": ["1+2"]}).to_csv(tmp_path / "m.csv", index=False)
    man = D.generic(str(root), str(tmp_path / "m.csv"), "{root}/{subject}/{clip}", dataset="mine", fps=100.0)
    r = man.records[0]
    assert (r.dataset, r.subject, r.emotion, r.aus, r.fps) == ("mine", "alice", "surprise", (1, 2), 100.0)
    assert D.load("generic", str(root), str(tmp_path / "m.csv")).records[0].uid == "custom/alice/c1"
    with pytest.raises(KeyError):
        D.load("nope", str(root))


def test_resolve_frames_numbering_modes(tmp_path):
    make_folder(tmp_path / "a", range(10, 31))
    paths, a_idx, mode, problem = D.resolve_frames(str(tmp_path / "a"), 12, 20, 25)
    assert mode == "file" and len(paths) == 14 and a_idx == 8 and problem is None
    paths, a_idx, mode, problem = D.resolve_frames(str(tmp_path / "a"), 2, 6, 9)       # numbers not in file names: positions
    assert mode == "relative" and len(paths) == 8 and a_idx == 4
    paths, _, _, problem = D.resolve_frames(str(tmp_path / "a"), 100, None, 200)
    assert paths == [] and "not in" in problem
    assert D.resolve_frames(str(tmp_path / "missing"), 1, None, 2)[3].startswith("no images")


def test_frames_ordered_numerically_not_alphabetically(tmp_path):
    make_folder(tmp_path / "o", [1, 2, 3, 10, 11, 100])
    nums = [n for n, _ in D.list_frames(str(tmp_path / "o"))]
    assert nums == [1, 2, 3, 10, 11, 100]
