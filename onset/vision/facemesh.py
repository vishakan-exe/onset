"""Face detection + 468-point landmarks, dependency-free (OpenCV DNN only).

The two networks are the MediaPipe BlazeFace short-range detector and the MediaPipe
Face Mesh (468 point) landmark model, both Apache-2.0 licensed (see NOTICE.md).  They are
stored as ``.tflite`` files and executed with ``cv2.dnn.readNetFromTFLite`` so the project
needs neither the ``mediapipe`` wheel nor TensorFlow.

Conventions
-----------
* Pixel coordinates, origin top-left, y pointing down.
* ``L`` / ``R`` always mean the *image* side (the viewer's left/right), never the
  subject's anatomical side.  MediaPipe's own names are the opposite.
* The tracker never smooths: landmark jitter is real measurement noise and is dealt with
  downstream (registration averages many landmarks, flow removes residual rigid motion).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DETECTOR_FILE = os.path.join(_MODEL_DIR, "face_detection_short_range.tflite")
LANDMARK_FILE = os.path.join(_MODEL_DIR, "face_landmark.tflite")

N_LANDMARKS = 468


# --------------------------------------------------------------------------------------
# BlazeFace anchors / decoding
# --------------------------------------------------------------------------------------
def _ssd_anchors() -> np.ndarray:
    """Anchor centres of the short-range BlazeFace model (896 x 2, normalised 0..1).

    Configuration copied from the model's reference graph: 4 layers, strides 8/16/16/16,
    one aspect ratio, plus one interpolated-scale anchor per cell and layer.
    """
    strides = [8, 16, 16, 16]
    input_size = 128
    anchors = []
    layer_id = 0
    while layer_id < len(strides):
        # consecutive layers that share a stride are merged into one grid
        last = layer_id
        n_per_cell = 0
        while last < len(strides) and strides[last] == strides[layer_id]:
            n_per_cell += 2  # aspect 1.0 anchor + interpolated-scale anchor
            last += 1
        fm = int(math.ceil(input_size / strides[layer_id]))
        for y in range(fm):
            for x in range(fm):
                for _ in range(n_per_cell):
                    anchors.append(((x + 0.5) / fm, (y + 0.5) / fm))
        layer_id = last
    return np.asarray(anchors, dtype=np.float32)


_ANCHORS = _ssd_anchors()
assert _ANCHORS.shape == (896, 2)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


@dataclass
class Detection:
    box: np.ndarray        # x1,y1,x2,y2 in source pixels
    keypoints: np.ndarray  # 6x2: R-eye, L-eye (subject sides), nose, mouth, R-ear, L-ear
    score: float


class FaceDetector:
    """BlazeFace short range (best for faces filling >~20 % of the frame, i.e. webcam use)."""

    def __init__(self, score_threshold: float = 0.5, nms_threshold: float = 0.3):
        if not os.path.exists(DETECTOR_FILE):
            raise FileNotFoundError(DETECTOR_FILE)
        self.net = cv2.dnn.readNetFromTFLite(DETECTOR_FILE)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        # letterbox to a square so the aspect ratio of the face is preserved
        side = max(h, w)
        pad_x, pad_y = (side - w) / 2.0, (side - h) / 2.0
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[int(pad_y):int(pad_y) + h, int(pad_x):int(pad_x) + w] = bgr
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 127.5, (128, 128), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        # OpenCV's DNN engine only exposes one graph output per call; request each by name
        scores_raw = self.net.forward("classificators")
        regs = self.net.forward("regressors")
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores_raw.reshape(-1).astype(np.float64), -30, 30)))
        regs = regs.reshape(896, 16)
        keep = scores >= self.score_threshold
        if not keep.any():
            return []
        idx = np.nonzero(keep)[0]
        a = _ANCHORS[idx]
        r = regs[idx]
        xc = r[:, 0] / 128.0 + a[:, 0]
        yc = r[:, 1] / 128.0 + a[:, 1]
        bw = r[:, 2] / 128.0
        bh = r[:, 3] / 128.0
        boxes = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)
        kps = np.stack([r[:, 4 + 2 * k] / 128.0 + a[:, 0] for k in range(6)], 1), \
            np.stack([r[:, 5 + 2 * k] / 128.0 + a[:, 1] for k in range(6)], 1)
        kp = np.stack([kps[0], kps[1]], 2)  # n x 6 x 2
        sc = scores[idx]
        order = np.argsort(-sc)
        out: list[Detection] = []
        remaining = order
        while remaining.size:
            best = remaining[0]
            ov = _iou(boxes[best], boxes[remaining])
            group = remaining[ov > self.nms_threshold]
            wts = sc[group][:, None]
            box = (boxes[group] * wts).sum(0) / wts.sum()
            kpt = (kp[group] * wts[:, :, None]).sum(0) / wts.sum()
            out.append(Detection(box=box, keypoints=kpt, score=float(sc[best])))
            remaining = remaining[ov <= self.nms_threshold]
        # back to source pixels
        res = []
        for d in out:
            bx = d.box * side
            bx[[0, 2]] -= pad_x
            bx[[1, 3]] -= pad_y
            kp_px = d.keypoints * side
            kp_px[:, 0] -= pad_x
            kp_px[:, 1] -= pad_y
            res.append(Detection(bx.astype(np.float32), kp_px.astype(np.float32), d.score))
        return res


# --------------------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------------------
@dataclass
class Landmarks:
    xy: np.ndarray         # 468x2 source pixels
    z: np.ndarray          # 468 relative depth, same scale as pixels (negative = towards camera)
    score: float           # face-presence probability


def _roi_matrix(center: np.ndarray, size: float, angle: float, out_size: int = 192) -> np.ndarray:
    """2x3 matrix mapping source pixels -> crop pixels for a square ROI rotated by ``angle``.

    ``angle`` is the angle of the eye line (atan2(dy, dx), y down); the crop is rotated by
    -angle so the eyes end up horizontal.
    """
    s = out_size / float(size)
    c, si = math.cos(-angle), math.sin(-angle)
    # source -> centred -> rotate -> scale -> crop centre
    a = np.array([[c * s, -si * s], [si * s, c * s]], dtype=np.float64)
    t = np.array([out_size / 2.0, out_size / 2.0]) - a @ np.asarray(center, dtype=np.float64)
    return np.hstack([a, t[:, None]])


class FaceTracker:
    """Detect once, then track with ROIs derived from the previous landmarks."""

    def __init__(self, presence_threshold: float = 0.5):
        if not os.path.exists(LANDMARK_FILE):
            raise FileNotFoundError(LANDMARK_FILE)
        self.detector = FaceDetector()
        self.net = cv2.dnn.readNetFromTFLite(LANDMARK_FILE)
        self.presence_threshold = presence_threshold
        self._roi: Optional[tuple[np.ndarray, float, float]] = None  # centre, size, angle

    def reset(self) -> None:
        self._roi = None

    # -- ROI helpers -----------------------------------------------------------------
    @staticmethod
    def _roi_from_detection(det: Detection) -> tuple[np.ndarray, float, float]:
        r_eye, l_eye = det.keypoints[0], det.keypoints[1]
        angle = math.atan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])
        cx, cy = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
        size = max(det.box[2] - det.box[0], det.box[3] - det.box[1]) * 1.5
        return np.array([cx, cy]), float(size), angle

    @staticmethod
    def _roi_from_landmarks(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
        # eye outer corners: 33 (image-left) and 263 (image-right)
        a, b = xy[33], xy[263]
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        pts = (xy - xy.mean(0)) @ rot.T
        mn, mx = pts.min(0), pts.max(0)
        centre_rot = (mn + mx) / 2
        centre = xy.mean(0) + rot.T @ centre_rot
        size = max(mx[0] - mn[0], mx[1] - mn[1]) * 1.5
        return centre, float(size), angle

    # -- inference -------------------------------------------------------------------
    def _run(self, bgr: np.ndarray, roi: tuple[np.ndarray, float, float]) -> Optional[Landmarks]:
        centre, size, angle = roi
        if size < 24:
            return None
        m = _roi_matrix(centre, size, angle)
        crop = cv2.warpAffine(bgr, m, (192, 192), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (192, 192), (0, 0, 0), swapRB=True)
        self.net.setInput(blob)
        lm_raw = self.net.forward("conv2d_21")
        flag_raw = self.net.forward("conv2d_31")
        lm = lm_raw.reshape(N_LANDMARKS, 3).astype(np.float64)
        score = float(1.0 / (1.0 + math.exp(-float(flag_raw.reshape(-1)[0]))))
        inv = cv2.invertAffineTransform(m)
        pts = np.hstack([lm[:, :2], np.ones((N_LANDMARKS, 1))]) @ inv.T
        scale = size / 192.0
        return Landmarks(pts.astype(np.float32), (lm[:, 2] * scale).astype(np.float32), score)

    def process(self, bgr: np.ndarray) -> Optional[Landmarks]:
        res = None
        if self._roi is not None:
            res = self._run(bgr, self._roi)
            if res is not None and res.score < self.presence_threshold:
                res = None
        if res is None:
            dets = self.detector.detect(bgr)
            if not dets:
                self._roi = None
                return None
            best = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) * d.score)
            res = self._run(bgr, self._roi_from_detection(best))
            if res is None or res.score < self.presence_threshold:
                self._roi = None
                return None
        self._roi = self._roi_from_landmarks(res.xy)
        return res
