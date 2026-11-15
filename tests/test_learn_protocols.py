import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from onset.learn import protocols as P


def rec(uid, dataset, subject, folder=None, **kw):
    return SimpleNamespace(uid=uid, dataset=dataset, subject=subject, group=f"{dataset}:{subject}", folder=folder or f"/nowhere/{uid}",
                           onset=kw.get("onset", 1), offset=kw.get("offset", 9), extra=kw.get("extra", {}), clip_id=uid)


def records(n_sub=8, per=4, dataset="d"):
    return [rec(f"{dataset}/{s}/{c}", dataset, f"{s:02d}") for s in range(n_sub) for c in range(per)]


def test_loso_each_subject_held_out_once():
    rs = records()
    folds = P.loso([r.group for r in rs])
    assert len(folds) == 8
    P.audit(rs, folds)                                                            # no raise
    assert sorted(i for f in folds for i in f.test) == list(range(len(rs)))
    for f in folds:
        assert not set(f.train) & set(f.test)


def test_composite_keeps_equal_numbers_in_different_datasets_apart():
    rs = records(3, 2, "casme2") + records(3, 2, "samm")                           # both have a subject '00', '01', '02'
    folds = P.composite([r.group for r in rs])
    assert len(folds) == 6                                                        # not 3: casme2:01 and samm:01 are different people
    P.audit(rs, folds)


def test_lodo_needs_two_datasets_and_holds_one_out():
    rs = records(3, 2, "a") + records(3, 2, "b") + records(2, 2, "c")
    folds = P.lodo([r.dataset for r in rs])
    assert [f.meta["held_out"] for f in folds] == ["a", "b", "c"]
    P.audit(rs, folds)
    with pytest.raises(ValueError):
        P.lodo(["a"] * 5)


@pytest.mark.parametrize("n_sub,k", [(10, 4), (26, 5), (5, 5), (3, 5)])
def test_kfold_is_subject_disjoint_and_balanced(n_sub, k, rng):
    sizes = rng.integers(2, 12, n_sub)
    groups = np.concatenate([[f"d:{i}"] * s for i, s in enumerate(sizes)])
    y = rng.choice(3, len(groups), p=[0.5, 0.3, 0.2])
    folds = P.kfold(groups, y, k=k, seed=3)
    assert len(folds) == min(k, n_sub)
    assert all(len(f.test) > 0 for f in folds)                                    # no empty fold
    for f in folds:
        assert not set(groups[f.train]) & set(groups[f.test])
    assert sorted(i for f in folds for i in f.test) == list(range(len(groups)))
    if n_sub >= 10:
        sizes_t = [len(f.test) for f in folds]
        assert max(sizes_t) - min(sizes_t) <= max(sizes) + 1
    assert [f.test.tolist() for f in P.kfold(groups, y, k=k, seed=3)] == [f.test.tolist() for f in folds]       # deterministic


def test_inner_folds_stay_inside_training_subjects():
    rs = records(10, 3)
    groups = np.array([r.group for r in rs])
    y = np.arange(len(rs)) % 3
    outer = P.loso(groups)[0]
    inner = P.inner_folds(outer.train, groups, y, k=4)
    held_out_group = groups[outer.test[0]]
    for f in inner:
        assert set(f.train) | set(f.test) == set(outer.train.tolist())            # never touches the outer test subject
        assert not set(groups[f.train]) & set(groups[f.test])
        assert held_out_group not in set(groups[f.train]) | set(groups[f.test])
    with pytest.raises(ValueError):
        P.inner_folds(np.arange(3), np.array(["s", "s", "s"]), np.array([0, 1, 0]))


def test_audit_refuses_leaks():
    rs = records(6, 2)
    groups = [r.group for r in rs]
    good = P.loso(groups)
    leaky = [P.Fold("leaky", np.arange(len(rs)), np.array([0, 1]))]                # clips 0 and 1 on both sides
    with pytest.raises(P.LeakageError):
        P.audit(rs, leaky)
    # same subject on both sides but different clips
    bad = [P.Fold("bad", np.array([1, 2, 3, 4, 5]), np.array([0]))]
    with pytest.raises(P.LeakageError, match="groups on both sides"):
        P.audit(rs, bad, require_full_coverage=False)
    # a clip never tested
    with pytest.raises(P.LeakageError, match="never in a test fold"):
        P.audit(rs, good[:-1])
    # a clip tested twice
    with pytest.raises(P.LeakageError, match="more than once"):
        P.audit(rs, good + [good[0]])
    out = P.audit(rs, leaky, strict=False)
    assert not out["ok"] and out["errors"]


def test_audit_catches_shared_folders_and_duplicate_ids():
    rs = records(4, 2)
    rs[3].folder = rs[0].folder                                                   # one frame folder filed under two records
    with pytest.raises(P.LeakageError, match="frame folders"):
        P.audit(rs, P.loso([r.group for r in rs]))
    rs = records(4, 2)
    rs[1].uid = rs[0].uid
    with pytest.raises(P.LeakageError, match="duplicate clip ids"):
        P.audit(rs, P.loso([r.group for r in rs]))


def test_audit_warns_about_lookalike_subjects_and_copied_first_frames(tmp_path):
    rs = []
    img = np.zeros((8, 8, 3), np.uint8)
    for s in ("01", "sub1", "02"):
        for c in range(2):
            d = tmp_path / f"{s}_{c}"
            d.mkdir()
            cv2.imwrite(str(d / "img1.png"), img + (hash((s, c)) % 200))
            rs.append(rec(f"d/{s}/{c}", "d", s, folder=str(d)))
    out = P.audit(rs, P.loso([r.group for r in rs]), strict=False)
    assert any("look like one person" in w for w in out["warnings"])
    # a copied clip: identical first frame under two different subjects
    d2 = tmp_path / "copy"
    d2.mkdir()
    (d2 / "img1.png").write_bytes((tmp_path / "02_0" / "img1.png").read_bytes())
    rs.append(rec("d/03/0", "d", "03", folder=str(d2)))
    out = P.audit(rs, P.loso([r.group for r in rs]), strict=False)
    assert any("byte-identical first frame" in w for w in out["warnings"])


def test_check_train_classes_reports_absent_class():
    y = np.array([0, 0, 1, 1, 2, 2])
    f = P.Fold("f", np.array([0, 1, 2, 3]), np.array([4, 5]))
    assert "classes [2] absent" in P.check_train_classes(y, f, 3)[0]
    assert P.check_train_classes(y, P.Fold("g", np.arange(6), np.array([0])), 3) == []
