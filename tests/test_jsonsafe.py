import json
import math

import numpy as np

from onset.jsonsafe import clean


def test_clean_makes_strict_json():
    o = dict(a=np.float32(1.5), b=np.int64(3), c=np.array([1.0, np.nan, np.inf]), d={1, 2}, e=(np.bool_(True), None), f=float("-inf"))
    s = json.dumps(clean(o), allow_nan=False)
    back = json.loads(s)
    assert back["a"] == 1.5 and back["b"] == 3 and back["c"] == [1.0, None, None]
    assert sorted(back["d"]) == [1, 2] and back["e"] == [True, None] and back["f"] is None
    assert math.isfinite(back["a"])
