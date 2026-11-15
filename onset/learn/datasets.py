"""Readers for the public micro-expression datasets: annotation sheet + frame folders -> ``ClipRecord``\\ s.

The datasets are licensed and not redistributed; point an adapter at *your* copy.  Each adapter

* finds the annotation table (``.xlsx`` or ``.csv``) and the right header row, matching columns by name (case and
  spacing ignored, a few aliases accepted) rather than by position, and stops with a message that lists the columns it
  found if an essential one is missing;
* resolves every row to a frame folder and **reports** the rows it could not use (missing folder, no frames in the
  annotated range, impossible onset/apex/offset) instead of dropping them silently;
* never edits the data and never guesses a label: an empty or unrecognised emotion stays ``None`` and the clip is simply
  not part of the schemes that need a label.

Folder layouts follow the releases as distributed (they differ between datasets and between versions; the layouts
below are the commonly used ones, and ``generic`` accepts anything with a sheet of ``subject, clip, onset, apex,
offset, emotion, aus`` columns and a folder template).  Frame numbers in the sheets are matched against the numbers
in the file names; if they do not match (some releases number frames from 1 inside each folder) the adapter falls back
to position and says so in ``record.extra['numbering']``.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import labels as L

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class ClipRecord:
    dataset: str
    subject: str
    clip_id: str
    folder: str
    onset: int
    apex: Optional[int]
    offset: int
    emotion: Optional[str]
    aus: tuple = ()
    fps: float = 0.0                     # nominal frame rate of the release (0 = unknown; features use it for seconds)
    extra: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.dataset}/{self.subject}/{self.clip_id}"

    @property
    def group(self) -> str:
        """Subject key that is unique across datasets (composite protocols must not merge two people)."""
        return f"{self.dataset}:{self.subject}"


@dataclass
class Manifest:
    dataset: str
    records: list
    problems: list = field(default_factory=list)
    annotation_sha256: str = ""
    n_rows: int = 0

    def summary(self) -> dict:
        subs = sorted({r.subject for r in self.records})
        emo: dict = {}
        for r in self.records:
            emo[r.emotion or "(none)"] = emo.get(r.emotion or "(none)", 0) + 1
        return dict(dataset=self.dataset, rows=self.n_rows, clips=len(self.records), subjects=len(subs),
                    problems=len(self.problems), emotions=emo, annotation_sha256=self.annotation_sha256)


# --------------------------------------------------------------------------------------------------------------
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_table(path: str, must_have: tuple = (), scan_rows: int = 40) -> pd.DataFrame:
    """Read an ``.xlsx`` / ``.csv`` sheet.  Some sheets (SAMM) have free text above the table: the header row is the first
    row, within ``scan_rows``, that contains every name in ``must_have`` (matched loosely)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"annotation file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            raw = pd.read_excel(path, header=None, dtype=object)
        except ImportError as e:
            raise ImportError("reading .xlsx needs openpyxl (pip install openpyxl), or export the sheet to .csv") from e
    else:
        raw = pd.read_csv(path, header=None, dtype=object, keep_default_na=False, na_values=[""])
    need = [_norm(m) for m in must_have]
    header = 0
    for i in range(min(scan_rows, len(raw))):
        cells = {_norm(c) for c in raw.iloc[i].tolist() if c is not None and str(c) != "nan"}
        if all(any(n == c or n in c for c in cells) for n in need):
            header = i
            break
    else:
        if need:
            seen = [str(c) for c in raw.iloc[0].tolist()]
            raise ValueError(f"no header row containing {list(must_have)} in the first {scan_rows} rows of {path}; "
                             f"first row reads {seen}")
    cols = [str(c).strip() if c is not None and str(c) != "nan" else f"unnamed_{j}" for j, c in enumerate(raw.iloc[header].tolist())]
    df = raw.iloc[header + 1:].reset_index(drop=True)
    df.columns = cols
    return df


def pick(df: pd.DataFrame, *aliases: str, required: bool = True, what: str = "") -> Optional[str]:
    """The column of ``df`` whose normalised name matches one of ``aliases`` (exact first, then substring)."""
    cmap = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in cmap:
            return cmap[_norm(a)]
    for a in aliases:
        for k, c in cmap.items():
            if _norm(a) and _norm(a) in k:
                return c
    if required:
        raise ValueError(f"column for {what or aliases[0]} not found; looked for {list(aliases)}, sheet has {list(df.columns)}")
    return None


def _int(v) -> Optional[int]:
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def _last_int(name: str) -> Optional[int]:
    m = re.findall(r"\d+", os.path.splitext(os.path.basename(name))[0])
    return int(m[-1]) if m else None


def list_frames(folder: str) -> list[tuple[Optional[int], str]]:
    """Image files of a clip folder as ``(number in the file name, path)``, ordered by number then name."""
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(IMG_EXT)]
    except (FileNotFoundError, NotADirectoryError):
        return []
    items = [(_last_int(n), os.path.join(folder, n)) for n in names]
    items.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0, t[1]))
    return items


def resolve_frames(folder: str, onset: int, apex: Optional[int], offset: int):
    """Frame paths from ``onset`` to ``offset`` and the index of the apex inside them.

    Returns ``(paths, apex_index, numbering, problem)``; ``problem`` is ``None`` or a string, in which case ``paths`` is empty."""
    fr = list_frames(folder)
    if not fr:
        return [], None, "", f"no images in {folder}"
    nums = {n: p for n, p in fr if n is not None}
    if nums and onset in nums and offset in nums:
        sel = [(n, p) for n, p in fr if n is not None and onset <= n <= offset]
        a = None
        if apex is not None and onset <= apex <= offset:
            a = next((i for i, (n, _) in enumerate(sel) if n == apex), None)
        return [p for _, p in sel], a, "file", None
    if 1 <= onset <= offset <= len(fr):
        sel = fr[onset - 1: offset]
        a = (apex - onset) if apex is not None and onset <= apex <= offset else None
        return [p for _, p in sel], a, "relative", None
    return [], None, "", f"frames {onset}-{offset} are not in {folder} ({len(fr)} images, numbers {fr[0][0]}..{fr[-1][0]})"


# --------------------------------------------------------------------------------------------------------------
def _build(dataset: str, df: pd.DataFrame, cols: dict, folder_for: Callable[[dict], str], fps: float, root: str,
           annotations: str, subject_fmt: Callable[[object], str], limit: Optional[int] = None) -> Manifest:
    man = Manifest(dataset, [], [], file_sha256(annotations), len(df))
    seen = set()
    for i, row in df.iterrows():
        if limit and len(man.records) >= limit:
            break
        on, off = _int(row[cols["onset"]]), _int(row[cols["offset"]])
        ap = _int(row[cols["apex"]]) if cols.get("apex") else None
        subj = subject_fmt(row[cols["subject"]])
        clip = str(row[cols["clip"]]).strip() if cols.get("clip") else f"row{i}"
        if on is None or off is None or off < on:
            man.problems.append(dict(row=int(i), clip=clip, problem=f"bad onset/offset ({row[cols['onset']]!r}, {row[cols['offset']]!r})"))
            continue
        info = dict(row=row, subject=subj, clip=clip, emotion=L.normalise(row[cols["emotion"]]) if cols.get("emotion") else None)
        folder = folder_for(info)
        paths, a_idx, numbering, problem = resolve_frames(folder, on, ap, off)
        if problem:
            man.problems.append(dict(row=int(i), clip=clip, problem=problem))
            continue
        if ap is not None and a_idx is None:
            ap = None                                    # annotated apex outside [onset, offset]: use the measured one
        uid = (subj, clip)
        if uid in seen:
            clip = f"{clip}#{i}"
        seen.add(uid)
        man.records.append(ClipRecord(
            dataset=dataset, subject=subj, clip_id=clip, folder=folder, onset=on, apex=ap, offset=off, emotion=info["emotion"],
            aus=L.parse_aus(row[cols["aus"]]) if cols.get("aus") else (), fps=fps,
            extra=dict(numbering=numbering, n_frames=len(paths), apex_index=a_idx)))
    return man


def _sub_fmt(width: int = 0, prefix: str = ""):
    def f(v):
        s = str(v).strip()
        n = _int(v)
        if n is not None and re.fullmatch(r"\d+(\.0+)?", s):
            s = str(n).zfill(width) if width else str(n)
        return s if not prefix or s.startswith(prefix) else prefix + s
    return f


def casme2(root: str, annotations: Optional[str] = None, limit: Optional[int] = None) -> Manifest:
    """CASME II: ``<root>/sub<NN>/<Filename>/img<N>.jpg``; sheet ``CASME2-coding-updated.xlsx`` (200 fps)."""
    annotations = annotations or _find(root, ("CASME2-coding", "casme2_coding", "casme2"))
    df = read_table(annotations, ("Subject", "Filename", "OnsetFrame", "OffsetFrame"))
    cols = dict(subject=pick(df, "Subject"), clip=pick(df, "Filename"), onset=pick(df, "OnsetFrame", "Onset"),
                apex=pick(df, "ApexFrame", "Apex", required=False), offset=pick(df, "OffsetFrame", "Offset"),
                emotion=pick(df, "Estimated Emotion", "Emotion", required=False), aus=pick(df, "Action Units", "AU", required=False))
    sub = _sub_fmt(2)
    return _build("casme2", df, cols, lambda d: os.path.join(root, "sub" + d["subject"], d["clip"]), 200.0, root, annotations,
                  lambda v: sub(v).replace("sub", ""), limit)


def samm(root: str, annotations: Optional[str] = None, limit: Optional[int] = None) -> Manifest:
    """SAMM: ``<root>/<SSS>/<Filename>/<SSS>_<NNNNN>.jpg``; sheet ``SAMM_Micro_FACS_Codes_v2.xlsx`` (200 fps)."""
    annotations = annotations or _find(root, ("SAMM_Micro_FACS", "samm"))
    df = read_table(annotations, ("Subject", "Filename", "Onset Frame", "Offset Frame"))
    cols = dict(subject=pick(df, "Subject"), clip=pick(df, "Filename"), onset=pick(df, "Onset Frame", "Onset"),
                apex=pick(df, "Apex Frame", "Apex", required=False), offset=pick(df, "Offset Frame", "Offset"),
                emotion=pick(df, "Estimated Emotion", "Emotion", required=False), aus=pick(df, "Action Units", "AU", required=False))
    sub = _sub_fmt(3)
    return _build("samm", df, cols, lambda d: os.path.join(root, d["subject"], d["clip"]), 200.0, root, annotations, sub, limit)


def mmew(root: str, annotations: Optional[str] = None, limit: Optional[int] = None) -> Manifest:
    """MMEW: ``<root>/<emotion>/<Filename>/<N>.jpg``; sheet ``MMEW_Micro_Exp.xlsx`` (90 fps)."""
    annotations = annotations or _find(root, ("MMEW_Micro", "mmew"))
    df = read_table(annotations, ("Subject", "Filename", "OnsetFrame", "OffsetFrame"))
    cols = dict(subject=pick(df, "Subject"), clip=pick(df, "Filename"), onset=pick(df, "OnsetFrame", "Onset"),
                apex=pick(df, "ApexFrame", "Apex", required=False), offset=pick(df, "OffsetFrame", "Offset"),
                emotion=pick(df, "Estimated Emotion", "Emotion"), aus=pick(df, "Action Units", "AU", required=False))
    sub = _sub_fmt(0)
    return _build("mmew", df, cols, lambda d: os.path.join(root, d["row"][cols["emotion"]].strip(), d["clip"]), 90.0, root, annotations, sub, limit)


def fourd(root: str, annotations: Optional[str] = None, limit: Optional[int] = None) -> Manifest:
    """4DME (the 2-D video modality): ``<root>/<sub_ID>/<fold_name>/``; sheet ``4DME_Labelling_Micro_release v1.xlsx`` (60 fps)."""
    annotations = annotations or _find(root, ("4DME_Labelling", "4dme"))
    df = read_table(annotations, ("sub_ID", "fold_name", "Onset", "Offset"))
    cols = dict(subject=pick(df, "sub_ID", "subject"), clip=pick(df, "fold_name"), onset=pick(df, "Onset"),
                apex=pick(df, "Apex", required=False), offset=pick(df, "Offset"),
                emotion=pick(df, "Emotion", required=False), aus=pick(df, "AUs", "AU", required=False))
    return _build("fourd", df, cols, lambda d: os.path.join(root, d["subject"], d["clip"]), 60.0, root, annotations, _sub_fmt(0), limit)


def casme3a(root: str, annotations: Optional[str] = None, limit: Optional[int] = None) -> Manifest:
    """CAS(ME)^3 part A segments: ``<root>/<subject>/<Filename>/color/<N>.jpg``; sheet ``cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx`` (30 fps)."""
    annotations = annotations or _find(root, ("cas(me)3_part_A", "casme3", "part_A"))
    df = read_table(annotations, ("Subject", "Filename", "Onset", "Offset"))
    cols = dict(subject=pick(df, "Subject"), clip=pick(df, "Filename"), onset=pick(df, "Onset"),
                apex=pick(df, "Apex", required=False), offset=pick(df, "Offset"),
                emotion=pick(df, "Objective class", "Emotion", "Estimated Emotion", required=False),
                aus=pick(df, "Action Units", "AU", "AUs", required=False))
    return _build("casme3a", df, cols, lambda d: os.path.join(root, d["subject"], d["clip"], "color"), 30.0, root, annotations, _sub_fmt(0), limit)


def generic(root: str, annotations: str, folder_template: str = "{root}/{subject}/{clip}", dataset: str = "custom", fps: float = 0.0,
            limit: Optional[int] = None) -> Manifest:
    """Any sheet with ``subject, clip, onset, offset`` (+ optional ``apex, emotion, aus``) columns; ``folder_template`` may use
    ``{root}``, ``{subject}``, ``{clip}`` and ``{emotion}``."""
    df = read_table(annotations, ("subject", "clip", "onset", "offset"))
    cols = dict(subject=pick(df, "subject"), clip=pick(df, "clip", "filename"), onset=pick(df, "onset"),
                apex=pick(df, "apex", required=False), offset=pick(df, "offset"),
                emotion=pick(df, "emotion", required=False), aus=pick(df, "aus", "au", "action units", required=False))
    return _build(dataset, df, cols,
                  lambda d: folder_template.format(root=root, subject=d["subject"], clip=d["clip"], emotion=d["emotion"] or ""), fps, root,
                  annotations, _sub_fmt(0), limit)


ADAPTERS = dict(casme2=casme2, samm=samm, mmew=mmew, fourd=fourd, casme3a=casme3a)


def _find(root: str, hints: tuple) -> str:
    """The first sheet under ``root`` whose name contains one of ``hints``."""
    cands = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".xlsx", ".csv", ".xls")) and not f.startswith("~$"):
                cands.append(os.path.join(dp, f))
        if dp.count(os.sep) - root.count(os.sep) >= 2:
            continue
    for h in hints:
        for c in sorted(cands):
            if h.lower() in os.path.basename(c).lower():
                return c
    raise FileNotFoundError(f"no annotation sheet matching {list(hints)} under {root}; pass --annotations")


def load(dataset: str, root: str, annotations: Optional[str] = None, limit: Optional[int] = None, **kw) -> Manifest:
    if dataset == "generic":
        return generic(root, annotations, limit=limit, **kw)
    if dataset not in ADAPTERS:
        raise KeyError(f"unknown dataset '{dataset}' (choose from {', '.join(list(ADAPTERS) + ['generic'])})")
    return ADAPTERS[dataset](root, annotations, limit)


def with_scheme(records: list, scheme: str) -> tuple[list, np.ndarray, list]:
    """Records that belong to ``scheme`` and their integer labels; the rest are returned as ``excluded`` (never relabelled)."""
    classes = list(L.SCHEMES[scheme])
    keep, y, excluded = [], [], []
    for r in records:
        lab = L.map_label(r.emotion, scheme)
        if lab is None:
            excluded.append(r)
            continue
        keep.append(r)
        y.append(classes.index(lab))
    return keep, np.asarray(y, int), excluded
