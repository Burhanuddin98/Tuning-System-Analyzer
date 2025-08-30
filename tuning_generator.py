
from __future__ import annotations
from fractions import Fraction
import math
from typing import List, Dict, Optional

def ratio_to_cents(ratio: float) -> float:
    return 1200.0 * math.log2(ratio)

def cents_to_ratio(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)

def normalize_to_octave(ratio: float) -> float:
    if ratio <= 0:
        raise ValueError("Ratio must be positive")
    while ratio < 1.0:
        ratio *= 2.0
    while ratio >= 2.0:
        ratio /= 2.0
    return ratio

def canonicalize_scale(ratios: List[float], names: Optional[List[str]] = None) -> Dict:
    ratios = [normalize_to_octave(r) for r in ratios]
    cents = [(ratio_to_cents(r) % 1200.0) for r in ratios]
    order = sorted(range(len(ratios)), key=lambda i: cents[i])
    data = []
    for idx, i in enumerate(order):
        nm = names[i] if names and i < len(names) and names[i] else f"deg{idx}"
        data.append({"name": nm, "ratio": ratios[i], "cents": cents[i]})
    return {"notes": data, "period_cents": 1200.0}

def equal_temperament(n: int = 12, names: Optional[List[str]] = None) -> Dict:
    ratios = [cents_to_ratio(k * 1200.0 / n) for k in range(n)]
    if names is None and n == 12:
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return {"system": f"{n}-EDO", **canonicalize_scale(ratios, names)}

def pythagorean(n: int = 12, generator_ratio: float = 3/2, names: Optional[List[str]] = None) -> Dict:
    gen = generator_ratio
    ratios = [1.0]
    k = 1
    while len(ratios) < max(n, 24):
        ratios.append(normalize_to_octave(gen ** k))
        ratios.append(normalize_to_octave(gen ** (-k)))
        k += 1
        if k > 64:
            break
    cents = sorted(set(round((1200.0 * math.log2(r)) % 1200.0, 6) for r in ratios))
    ratios_sorted = [2.0 ** (c/1200.0) for c in cents][:n]
    return {"system": f"Pythagorean_{n}", **canonicalize_scale(ratios_sorted, names)}

def syntonic_comma_cents() -> float:
    return 1200.0 * math.log2(81/80)

def meantone(n: int = 12, comma_fraction: float = 1/4, names: Optional[List[str]] = None) -> Dict:
    pure_fifth = 1200.0 * math.log2(3/2)
    sc = syntonic_comma_cents()
    fifth_cents = pure_fifth - comma_fraction * sc
    ratios = [1.0]
    for k in range(1, max(n, 24)):
        ratios.extend([2.0 ** ((( k * fifth_cents) % 1200.0)/1200.0),
                       2.0 ** (((-k * fifth_cents) % 1200.0)/1200.0)])
    cents = sorted(set(round((1200.0 * math.log2(r)) % 1200.0, 6) for r in ratios))
    ratios_sorted = [2.0 ** (c/1200.0) for c in cents][:n]
    tag = f"Meantone_{n}_{comma_fraction}comma"
    return {"system": tag, **canonicalize_scale(ratios_sorted, names)}

def ji_major_5limit() -> Dict:
    names = ["1","2","3","4","5","6","7"]
    ratios = [1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]
    return {"system": "JI_5limit_major_7", **canonicalize_scale(ratios, names)}

def ji_minor_5limit() -> Dict:
    names = ["1","2","b3","4","5","b6","b7"]
    ratios = [1/1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5]
    return {"system": "JI_5limit_natural_minor_7", **canonicalize_scale(ratios, names)}

def ji_chromatic_12() -> Dict:
    names = ["1","b2","2","b3","3","4","#4","5","b6","6","b7","7"]
    ratios = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8]
    return {"system": "JI_5limit_chromatic_12", **canonicalize_scale(ratios, names)}

def pack_defaults() -> Dict[str, Dict]:
    systems = [
        equal_temperament(12),
        equal_temperament(19),
        pythagorean(12),
        meantone(12, comma_fraction=1/4),
        ji_major_5limit(),
        ji_minor_5limit(),
        ji_chromatic_12(),
    ]
    return {s["system"]: s for s in systems}
