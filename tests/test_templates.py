import numpy as np
import pytest

from onset.analysis import templates as TPL
from onset.learn.features import mirror
from onset.vision import geometry as G
from helpers import fake_geometry

R = len(G.ROI_NAMES)


def idx(name):
    return G.ROI_NAMES.index(name)


def test_shapes_and_unit_peak():
    T = TPL.au_templates()
    assert T.shape == (len(TPL.AUS), R, 2) and TPL.AUS == [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]
    for t in list(T) + [TPL.blink_template(), TPL.speech_template()]:
        assert np.linalg.norm(t, axis=1).max() == pytest.approx(1.0)       # "an amplitude of a means the dominant region moves by a pixels"
    S = TPL.scan_templates()
    assert S.shape == (len(TPL.AUS) + 2, R, 2) and len(TPL.SCAN_KINDS) == len(TPL.AUS) + 2
    assert [TPL.scan_label(i) for i in (0, 8, 12, 13)] == ["AU1", "AU12", "blink", "speech"]


def test_templates_are_left_right_mirror_images():
    T = TPL.au_templates()
    assert np.allclose(mirror(T), T, atol=1e-9)


@pytest.mark.parametrize("au,roi,axis,sign", [
    (12, "corner_R", 0, +1), (12, "corner_L", 0, -1), (12, "corner_R", 1, -1),        # lip corners go out and up (image y points down)
    (4, "brow_in_R", 1, +1), (4, "brow_in_R", 0, -1),                                  # brows go down and together
    (1, "brow_in_L", 1, -1), (2, "brow_out_R", 1, -1), (5, "lid_up_L", 1, -1),         # raised inner brow, outer brow, upper lid
    (6, "cheek_R", 1, -1), (15, "corner_L", 1, +1), (17, "chin", 1, -1),
    (9, "wing_R", 1, -1), (10, "lip_up", 1, -1), (14, "corner_R", 0, -1),              # dimpler pulls the corner inward
])
def test_template_directions_follow_the_anatomy(au, roi, axis, sign):
    t = TPL.au_templates()[TPL.AUS.index(au)]
    assert np.sign(t[idx(roi), axis]) == sign, (au, roi, t[idx(roi)])


def test_blink_and_speech_templates_live_where_they_should():
    b, s = TPL.blink_template(), TPL.speech_template()
    assert np.linalg.norm(b[idx("lid_up_L")]) == pytest.approx(1.0)
    assert np.linalg.norm(b[idx("chin")]) == 0
    assert np.linalg.norm(s[idx("lip_lo")]) == pytest.approx(1.0) and np.linalg.norm(s[idx("lid_up_L")]) == 0


def test_nuisance_columns_span_the_expected_motions():
    c, centre, _ = fake_geometry()
    N = TPL.nuisance_columns(c, centre)
    assert N.shape == (7, R, 2)
    assert np.allclose(N[0, :, 0], 1) and np.allclose(N[0, :, 1], 0)                   # x translation: 1 px everywhere
    px = c[:, 0] - centre[0]
    assert np.allclose(N[2, :, 0], 0.01 * px)                                          # isotropic scale
    rank = np.linalg.matrix_rank(N.reshape(7, -1))
    assert rank == 7                                                                   # six affine directions + gaze are all distinct
    for a in TPL.au_templates():                                                       # no AU template is just a head movement
        flat = a.reshape(-1)
        Q, _ = np.linalg.qr(N.reshape(7, -1).T)
        resid = flat - Q @ (Q.T @ flat)
        assert np.linalg.norm(resid) > 0.5 * np.linalg.norm(flat)


def test_identifiability_matrix():
    I = TPL.identifiability()
    assert I.shape == (12, 12) and np.allclose(np.diag(I), 1) and np.allclose(I, I.T)
    j = {a: TPL.AUS.index(a) for a in TPL.AUS}
    assert I[j[1], j[2]] > I[j[1], j[12]]                                              # the two brow raisers look alike; brow vs mouth do not
    assert I[j[9], j[10]] > I[j[9], j[17]]
    c, centre, _ = fake_geometry()
    Ip = TPL.identifiability(np.full((R, 2), 0.1), None, TPL.nuisance_columns(c, centre))
    assert np.all(np.abs(Ip) <= 1 + 1e-9)


def test_roi_weights_and_unit_conversion():
    w = TPL.au_roi_weights()
    assert w.shape == (12, R) and np.allclose(w.sum(1), 1)
    assert w[TPL.AUS.index(12)].argmax() in (idx("corner_L"), idx("corner_R"))
    assert TPL.amp_to_pct_iod(1.0, 64.0) == pytest.approx(100 / 64)


def test_roi_centroids_from_masks():
    rois = {n: np.zeros((20, 20), bool) for n in G.ROI_NAMES}
    for i, n in enumerate(G.ROI_NAMES):
        rois[n][i % 20, (3 * i) % 20] = True
    c = TPL.roi_centroids(rois)
    assert c.shape == (R, 2) and c[3].tolist() == [9.0, 3.0]
    rois["chin"][:] = False
    assert np.isnan(TPL.roi_centroids(rois)[idx("chin")]).all()
