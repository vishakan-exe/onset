"""Magnification (a visual aid only), the motion-heat overlay, plate geometry, and the action-unit atlas."""
import cv2
import numpy as np
import pytest

from onset.analysis import atlas as AT
from onset.analysis import magnify as MG
from onset.analysis import templates as TPL
from onset.vision import geometry as G
from helpers import fake_geometry


def _texture(n=120, seed=1):
    img = (np.random.default_rng(seed).random((n, n)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 3)


# ------------------------------------------------------------------------------------------------ magnification
def test_noise_level_flow_is_shrunk_to_nothing_so_jitter_is_never_magnified():
    u = np.random.default_rng(0).normal(0, 0.05, (120, 120, 2)).astype(np.float32)
    shrunk, tau = MG.shrink_flow(u)
    assert tau >= 0.05 and np.linalg.norm(shrunk, axis=-1).max() == 0.0


def test_a_clear_movement_survives_shrinkage_almost_intact():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 0.05, (120, 120, 2)).astype(np.float32)
    yy, xx = np.mgrid[0:120, 0:120]
    u[..., 1] += -3.0 * np.exp(-((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 12 ** 2)).astype(np.float32)
    shrunk, tau = MG.shrink_flow(u)
    peak_before, peak_after = np.abs(u[..., 1]).max(), np.abs(shrunk[..., 1]).max()
    assert 0.9 * peak_before < peak_after <= peak_before                 # shrinkage never grows a vector
    assert np.linalg.norm(shrunk[0, 0]) == 0.0                           # far from the movement: nothing


def test_the_oval_sets_the_noise_level_from_the_face_only():
    u = np.zeros((100, 100, 2), np.float32)
    u[:, 50:, 0] = 2.0                                                    # the right half moves, the left half is still
    oval = np.zeros((100, 100), bool)
    oval[:, :50] = True
    _, tau_still = MG.shrink_flow(u, oval)
    _, tau_all = MG.shrink_flow(u)
    assert tau_still <= tau_all


def test_zero_motion_gives_back_the_reference_exactly():
    img = _texture()
    assert (MG.magnify_frame(img, np.zeros((120, 120, 2), np.float32), 5.0) == img).all()


def test_magnification_moves_the_picture_by_one_plus_alpha_times_the_displacement():
    img = _texture()
    u = np.zeros((120, 120, 2), np.float32)
    u[..., 0] = 1.0                                                       # everything moved one pixel to the right
    out = MG.magnify_frame(img, u, 2.0)                                   # so (1 + 2) x 1 = three pixels
    assert (out[:, 10:-10].astype(int) == np.roll(img, 3, axis=1)[:, 10:-10].astype(int)).all()


def test_magnification_keeps_the_reference_texture_and_noise():
    """The reference is warped, not re-synthesised: its grey-level statistics are those of the reference."""
    img = _texture()
    u = np.zeros((120, 120, 2), np.float32)
    u[..., 1] = 0.7
    out = MG.magnify_frame(img, u, 4.0)
    assert abs(float(out.mean()) - float(img.mean())) < 3 and abs(float(out.std()) - float(img.std())) < 3


def test_motion_heat_is_zero_below_the_noise_and_bounded():
    u = np.zeros((60, 60, 2), np.float32)
    u[20:40, 20:40, 0] = 2.0
    h = MG.motion_heat(u, None, tau=0.5, vmax=2.0)
    assert h.min() == 0.0 and h.max() == pytest.approx(1.0) and h[0, 0] == 0.0
    oval = np.zeros((60, 60), bool)
    oval[:, :30] = True
    h2 = MG.motion_heat(u, oval, tau=0.5, vmax=2.0)
    assert h2[30, 50] < 0.05 < h2[30, 22]                                 # outside the face oval the overlay fades out
    assert 0.0 <= h2.min() and h2.max() <= 1.0


def test_plate_box_stays_inside_the_canvas_and_is_three_eye_distances_wide():
    x0, y0, x1, y1 = MG.plate_box(384, 140, 96)
    assert 0 <= x0 < x1 <= 384 and 0 <= y0 < y1 <= 384 and (x1 - x0) == 288
    assert MG.plate_box(384, 30, 96)[1] == 0                              # a face near the top is clamped, never negative


def test_crop_resize_gives_the_requested_size_up_and_down():
    img = _texture(200)
    assert MG.crop_resize(img, (10, 10, 190, 190), 64).shape == (64, 64)
    assert MG.crop_resize(img, (10, 10, 40, 40), 64).shape == (64, 64)


# ------------------------------------------------------------------------------------------------ the atlas
@pytest.fixture(scope="module")
def atlas():
    c, centre, iod = fake_geometry()
    canon = dict(rois=[dict(name=n) for n in G.ROI_NAMES],
                 centroids=[dict(roi=n, x=float(c[i, 0]), y=float(c[i, 1])) for i, n in enumerate(G.ROI_NAMES)],
                 plate_size=176, groups=None)
    return AT.build_atlas(canon)


def test_atlas_covers_every_unit_with_anatomy_and_vectors(atlas):
    assert [e["au"] for e in atlas["aus"]] == list(TPL.AUS)
    for e in atlas["aus"]:
        assert e["muscle"] and e["sees"] and e["looks"] and e["name"]
        assert e["vectors"] and all({"roi", "dx", "dy"} <= set(v) for v in e["vectors"])
        assert len(e["confusable"]) == 3 and e["au"] not in {c["au"] for c in e["confusable"]}
        sims = [c["similarity"] for c in e["confusable"]]
        assert sims == sorted(sims, reverse=True)


def test_identifiability_is_a_symmetric_similarity_with_a_unit_diagonal(atlas):
    M = np.array(atlas["identifiability"]["matrix"])
    assert M.shape == (12, 12) and np.allclose(M, M.T) and np.allclose(np.diag(M), 1.0)
    assert np.abs(M).max() <= 1.0 + 1e-9
    assert atlas["identifiability"]["order"] == list(TPL.AUS)


def test_the_pairs_anatomy_says_are_hard_to_tell_apart_are_the_ones_found(atlas):
    """Computed, not asserted: the lid-tightener / cheek-raiser pair and the nose-wrinkler / lip-raiser pair are the classic confusions."""
    M = np.array(atlas["identifiability"]["matrix"])
    idx = {a: i for i, a in enumerate(TPL.AUS)}
    off = M - np.eye(12) * 10
    assert M[idx[6], idx[7]] > 0.5 and M[idx[9], idx[10]] > 0.5
    assert M[idx[6], idx[7]] >= np.sort(off.ravel())[-3]                  # among the three most similar pairs (each pair counts twice)
    assert abs(M[idx[5], idx[17]]) < 0.2 and abs(M[idx[1], idx[14]]) < 0.2   # unrelated units are not confused


def test_atlas_prototypes_use_only_known_units_and_carry_the_geometry(atlas):
    for p in atlas["prototypes"]:
        assert set(p["aus"]) <= set(TPL.AUS) and p["label"] and p["note"]
    assert len(atlas["canon"]["centroids"]) == 27 and atlas["canon"]["plate_size"] == 176
    assert "FACS" in atlas["source"] or "templates" in atlas["source"].lower()
