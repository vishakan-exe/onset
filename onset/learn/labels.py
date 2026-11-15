"""Label vocabularies and the maps between dataset labels and the schemes ONSET reports.

The maps are *choices*, written down here so they can be argued with: datasets disagree on what 'others', 'tense' or
'repression' mean, and the composite (MEGC-style) three-class scheme merges several emotions into 'negative'.  A clip
whose dataset label has no counterpart in a scheme is **excluded** from that scheme (``map_label`` returns ``None``);
nothing is silently reassigned.
"""
from __future__ import annotations

import re
from typing import Optional

MEGC3 = ("positive", "negative", "surprise")
CASME2_5 = ("happiness", "disgust", "surprise", "repression", "others")
BASIC7 = ("happiness", "surprise", "fear", "anger", "disgust", "sadness", "contempt")
SCHEMES = {"megc3": MEGC3, "casme2_5": CASME2_5, "basic7": BASIC7}

_SYN = {
    "happy": "happiness", "joy": "happiness",
    "sad": "sadness", "angry": "anger", "contemptuous": "contempt", "disgusted": "disgust", "fearful": "fear",
    "surprised": "surprise", "other": "others", "tense": "tense", "repress": "repression", "repressed": "repression",
    "negative": "negative",
}
_NEG = {"disgust", "fear", "sadness", "anger", "contempt", "repression", "negative"}


def normalise(label) -> Optional[str]:
    """Lower-case, trimmed, synonyms folded; ``None`` for empty / NaN-like values."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s or s in ("nan", "none", "-", "n/a", "unknown"):
        return None
    return _SYN.get(s, s)


def map_label(label, scheme: str) -> Optional[str]:
    """Dataset label -> label of ``scheme``, or ``None`` when the clip does not belong to the scheme."""
    s = normalise(label)
    if s is None:
        return None
    if scheme == "megc3":
        if s in ("happiness", "positive"):
            return "positive"
        if s == "surprise":
            return "surprise"
        if s in _NEG:
            return "negative"
        return None                                  # others / tense / unknown are not part of the composite scheme
    if scheme == "casme2_5":
        if s in ("happiness", "disgust", "surprise", "repression"):
            return s
        if s in ("others", "tense", "fear", "sadness", "anger", "contempt"):
            return "others"
        return None
    if scheme == "basic7":
        return s if s in BASIC7 else None
    raise KeyError(f"unknown label scheme '{scheme}' (choose from {', '.join(SCHEMES)})")


_AU_TOKEN = re.compile(r"(?:AU)?\s*[LR]?\s*(\d{1,2})\s*[A-E]?(?![\d])", re.I)


def parse_aus(text, drop_k: bool = True) -> tuple[int, ...]:
    """Action-unit numbers in a coding-sheet cell: ``'R12 + R15'``, ``'4+7'``, ``'L14'``, ``'AU9, AU10'``, ``'R14A'``.

    Side (L/R) and intensity (A-E) markers are dropped.  ``(k)`` marks (4DME) flag AUs the coders were not sure of
    and are dropped unless ``drop_k`` is False.  A cell with alternatives (``'12 or 14'``) yields the first
    alternative only: the coders themselves did not decide."""
    if text is None:
        return ()
    s = str(text).strip()
    if not s or s.lower() in ("nan", "none"):
        return ()
    s = re.split(r"\bor\b", s, flags=re.I)[0]
    out = set()
    for tok in re.split(r"[+,;/&]|\band\b", s, flags=re.I):
        if drop_k and "(k)" in tok.lower():
            continue
        m = _AU_TOKEN.search(tok.replace("(k)", ""))
        if m:
            n = int(m.group(1))
            if 1 <= n <= 46:
                out.add(n)
    return tuple(sorted(out))
