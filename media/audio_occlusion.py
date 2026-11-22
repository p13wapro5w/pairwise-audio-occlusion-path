"""Pairwise 3D audio occlusion primitives.

The helpers here are intentionally geometry-agnostic: callers provide ray
samples, material hits, distances and gains, and these functions turn them
into an occlusion result.  They are the pieces needed to avoid the common
failure modes of a single-ray, all-or-nothing implementation:

* :func:`multi_path` aggregates several rays so a source that slips past a
  door frame is only partially occluded.
* :func:`distance_band` splits distance into near/mid/far bands.
* :func:`crossfade_state` smooths state changes so walking around a corner
  does not produce an audible step.
* :func:`all_types` gives every sound category (including ambience and
  music) an occlusion rule.
* :func:`per_listener` produces one result per listener.
* :func:`query_budget` caps how many occlusion queries run per frame.
* :func:`reverb_link` couples the reverb send/decay to occlusion.
* :func:`symmetric_calc` makes source/listener swaps agree.
* :func:`exclude_2d` keeps 2D UI sounds out of the occlusion path.
* :func:`material_band` distinguishes glass from concrete.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

__all__ = [
    "DistanceBand",
    "ListenerOcclusion",
    "Material",
    "MaterialBand",
    "MATERIALS",
    "MAX_LOWPASS_HZ",
    "MIN_LOWPASS_HZ",
    "ReverbLink",
    "SoundRule",
    "SoundType",
    "Vec3",
    "all_types",
    "attenuation_mix",
    "crossfade_state",
    "distance_band",
    "exclude_2d",
    "lowpass_cutoff",
    "material_band",
    "multi_path",
    "occlusion_ratio",
    "per_listener",
    "portal_path",
    "query_budget",
    "ray_budget",
    "reverb_link",
    "symmetric_calc",
]

#: Lowest cutoff a fully occluded source may reach.
MIN_LOWPASS_HZ = 400.0
#: Cutoff of an unoccluded source.
MAX_LOWPASS_HZ = 20_000.0

_NEAR_FRACTION = 0.35
_FAR_FRACTION = 0.75
_NEAR_GAIN = 1.0
_MID_GAIN = 0.72
_FAR_GAIN = 0.45


def _clamp(value, low: float = 0.0, high: float = 1.0) -> float:
    """Coerce *value* to a float in ``[low, high]``, tolerating junk."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return low
    if math.isnan(result):
        return low
    return min(high, max(low, result))


def _as_spatial(value) -> bool:
    """Interpret a spatial/3D flag, accepting explicit 2D strings."""
    if isinstance(value, str):
        return value.strip().lower() not in {"2d", "false", "no", "off", "ui"}
    return bool(value)


# ---------------------------------------------------------------------------
# Basic value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vec3:
    """Minimal 3D point used by callers that prefer a common type."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: "Vec3") -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        t = _clamp(t)
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )


@dataclass(frozen=True)
class Material:
    """Acoustic profile of a wall material.

    ``transmission`` is the amplitude that passes through one metre of the
    material; it is raised to the wall thickness by :func:`material_band`.
    """

    name: str
    transmission: float
    lowpass_hz: float


MATERIALS: dict[str, Material] = {
    "concrete": Material("concrete", transmission=0.05, lowpass_hz=500.0),
    "brick": Material("brick", transmission=0.08, lowpass_hz=800.0),
    "drywall": Material("drywall", transmission=0.28, lowpass_hz=2_500.0),
    "wood": Material("wood", transmission=0.20, lowpass_hz=2_000.0),
    "metal": Material("metal", transmission=0.10, lowpass_hz=1_200.0),
    "glass": Material("glass", transmission=0.65, lowpass_hz=6_000.0),
}
_MATERIAL_ORDER = tuple(MATERIALS.values())
_DEFAULT_MATERIAL = MATERIALS["concrete"]


def _resolve_material(material) -> Material:
    """Accept a :class:`Material`, a name, or an index into the registry."""
    if isinstance(material, Material):
        return material
    if isinstance(material, str):
        return MATERIALS.get(material.strip().lower(), _DEFAULT_MATERIAL)
    if isinstance(material, numbers.Integral) and not isinstance(material, bool):
        return _MATERIAL_ORDER[int(material) % len(_MATERIAL_ORDER)]
    return _DEFAULT_MATERIAL


class SoundType(str, Enum):
    """Sound categories that can reach the occlusion stage."""

    FOOTSTEP = "footstep"
    GUNSHOT = "gunshot"
    VOICE = "voice"
    AMBIENCE = "ambience"
    MUSIC = "music"
    UI = "ui"
    OTHER = "other"

    @classmethod
    def coerce(cls, value) -> "SoundType":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.OTHER
        if isinstance(value, numbers.Integral):
            members = tuple(cls)
            return members[int(value) % len(members)]
        text = str(value).strip().lower()
        for member in cls:
            if text in (member.value, member.name.lower()):
                return member
        return cls.OTHER


@dataclass(frozen=True)
class SoundRule:
    """How one sound category participates in occlusion."""

    sound_type: SoundType
    occluded: bool
    reverb_linked: bool
    lowpass: bool

    def __bool__(self) -> bool:
        return self.occluded


@dataclass(frozen=True)
class MaterialBand:
    """Result of :func:`material_band`; float value is transmission."""

    material: Material
    thickness: float
    transmission: float
    attenuation_db: float
    lowpass_hz: float

    def __float__(self) -> float:
        return self.transmission


@dataclass(frozen=True)
class ReverbLink:
    """Reverb parameters after applying occlusion."""

    wet_send: float
    decay_scale: float
    dry_gain: float

    def __float__(self) -> float:
        return self.wet_send


# ---------------------------------------------------------------------------
# Multi-path occlusion
# ---------------------------------------------------------------------------


def _path_occlusion(sample) -> float:
    """Normalise one ray sample to a blocked amount in ``[0, 1]``."""
    if isinstance(sample, bool):
        return 1.0 if sample else 0.0
    for attribute in ("occlusion", "blocked"):
        if hasattr(sample, attribute):
            return _clamp(getattr(sample, attribute))
    if hasattr(sample, "transmission"):
        return 1.0 - _clamp(sample.transmission)
    return _clamp(sample)


def multi_path(paths, total=None, *, weights: Iterable[float] | None = None) -> float:
    """Aggregate several occlusion rays into one blocked ratio.

    ``paths`` can be:

    * ``(blocked, total)`` ray counts when *total* is supplied;
    * a single blocked amount in ``[0, 1]``;
    * an iterable of per-ray samples.  A sample may be a bool, a blocked
      amount, or any object exposing ``occlusion``, ``blocked`` or
      ``transmission``.

    Averaging instead of testing one centre ray is what lets a sound source
    visible through part of a doorway stay partially audible.  ``weights``
    optionally biases the average (for example, towards the direct ray).
    """
    if total is not None:
        denominator = float(total)
        if denominator <= 0.0:
            return 0.0
        return _clamp(float(paths) / denominator)

    if isinstance(paths, numbers.Real) and not isinstance(paths, bool):
        return _clamp(paths)

    samples = [_path_occlusion(sample) for sample in paths]
    if not samples:
        return 0.0

    if weights is None:
        return _clamp(sum(samples) / len(samples))

    weights = [max(0.0, float(weight)) for weight in weights]
    if len(weights) != len(samples) or sum(weights) <= 0.0:
        return _clamp(sum(samples) / len(samples))
    total_weight = sum(weights)
    return _clamp(
        sum(sample * weight for sample, weight in zip(samples, weights)) / total_weight
    )


def occlusion_ratio(blocked_paths, total_paths) -> float:
    """Normalise blocked ray counts to a ratio, clamped to ``[0, 1]``."""
    total = float(total_paths)
    if total <= 0.0:
        return 0.0
    return _clamp(float(blocked_paths) / total)


def portal_path(direct_occlusion, portal_occlusion, *, portal_weight: float = 0.6) -> float:
    """Occlusion of a pair connected through a door/window.

    The clearer of the direct transmission and the portal transmission wins,
    so a wall that blocks the direct ray no longer fully mutes a source that
    is audible through the doorway.
    """
    direct = _clamp(direct_occlusion)
    portal = _clamp(portal_occlusion)
    weight = _clamp(portal_weight)
    transmission = max(1.0 - direct, (1.0 - portal) * weight)
    return _clamp(1.0 - transmission)


# ---------------------------------------------------------------------------
# Distance, filtering and state transitions
# ---------------------------------------------------------------------------


class DistanceBand(float):
    """A named distance band; the float value is its linear gain."""

    def __new__(cls, name: str, normalized: float, gain: float):
        instance = super().__new__(cls, _clamp(gain))
        instance.name = name
        instance.normalized = _clamp(normalized, 0.0, math.inf)
        return instance

    @property
    def gain(self) -> float:
        return float(self)


def distance_band(
    distance: float,
    max_distance: float,
    *,
    near_fraction: float = _NEAR_FRACTION,
    far_fraction: float = _FAR_FRACTION,
) -> DistanceBand:
    """Classify a distance and return its band plus linear gain.

    Distances beyond ``max_distance`` saturate at the far band.  The band
    gains are step values on purpose: callers use them for the coarse
    near/mid/far attenuation layer, while the exact distance attenuation is
    handled separately.
    """
    max_distance = float(max_distance)
    distance = max(0.0, float(distance))
    if max_distance <= 0.0:
        normalized = 0.0 if distance <= 0.0 else 1.0
    else:
        normalized = _clamp(distance / max_distance, 0.0, math.inf)

    near_fraction = _clamp(near_fraction)
    far_fraction = max(near_fraction, _clamp(far_fraction))
    if normalized < near_fraction:
        return DistanceBand("near", normalized, _NEAR_GAIN)
    if normalized < far_fraction:
        return DistanceBand("mid", normalized, _MID_GAIN)
    return DistanceBand("far", normalized, _FAR_GAIN)


def crossfade_state(
    current: float,
    target: float,
    alpha: float | None = None,
    *,
    dt: float | None = None,
    tau: float = 0.12,
) -> float:
    """Move an occlusion/1-pole state towards *target* without a step.

    With an explicit *alpha* this is a linear blend.  With *dt* it uses the
    frame-rate independent coefficient ``1 - exp(-dt / tau)``.  Called with
    only two arguments it takes a half-way step, which is the safe default
    when the caller has no frame time available.
    """
    current = _clamp(current)
    target = _clamp(target)

    if alpha is None:
        if dt is None:
            alpha = 0.5
        else:
            tau = max(float(tau), 1e-6)
            alpha = 1.0 - math.exp(-max(0.0, float(dt)) / tau)

    alpha = _clamp(alpha)
    return current + (target - current) * alpha


def lowpass_cutoff(
    base_hz: float,
    occlusion: float,
    *,
    min_hz: float = MIN_LOWPASS_HZ,
) -> float:
    """Lower the cutoff of a one-pole lowpass according to occlusion.

    The interpolation is logarithmic in frequency, which gives a natural
    timbre change between "one wall" and "three walls" instead of a single
    fixed cutoff.
    """
    min_hz = max(1.0, float(min_hz))
    base = max(float(base_hz), min_hz)
    base = min(base, MAX_LOWPASS_HZ)
    amount = _clamp(occlusion)
    return max(min_hz, base * (min_hz / base) ** amount)


def attenuation_mix(
    direct_gain: float,
    occluded_gain: float,
    *,
    occlusion: float = 0.5,
) -> float:
    """Blend unoccluded and occluded gains by the occlusion amount."""
    direct = max(0.0, float(direct_gain))
    occluded = max(0.0, float(occluded_gain))
    amount = _clamp(occlusion)
    return _clamp(direct * (1.0 - amount) + occluded * amount, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Sound types, listeners and query scheduling
# ---------------------------------------------------------------------------


def _rule_for(sound_type, spatial) -> SoundRule:
    sound_type = SoundType.coerce(sound_type)
    participates = _as_spatial(spatial) and sound_type is not SoundType.UI
    return SoundRule(
        sound_type=sound_type,
        occluded=participates,
        reverb_linked=participates,
        lowpass=participates,
    )


def all_types(sound_type=None, spatial: bool = True):
    """Return occlusion rules covering every sound category.

    With no argument, a tuple with one :class:`SoundRule` per
    :class:`SoundType` is returned.  Passing a type (name, enum or index)
    returns that single rule.  Ambience and music are spatial sources and
    therefore participate; UI is intrinsically 2D and is excluded.
    """
    if sound_type is None:
        return tuple(_rule_for(member, spatial) for member in SoundType)
    if isinstance(sound_type, (str, numbers.Integral, SoundType)):
        return _rule_for(sound_type, spatial)
    return tuple(_rule_for(member, spatial) for member in sound_type)


def exclude_2d(sound_type, is_3d: bool = True) -> bool:
    """Return ``True`` when a sound must bypass occlusion.

    UI sounds are always 2D.  Any other sound bypasses occlusion when the
    caller marks it as non-spatial, so menu clicks are never muffled by
    geometry.  A bare boolean may be passed as the first argument to test
    just the spatial flag.
    """
    if isinstance(sound_type, bool):
        return not sound_type
    sound_type = SoundType.coerce(sound_type)
    return not (_as_spatial(is_3d) and sound_type is not SoundType.UI)


class ListenerOcclusion(float):
    """Occlusion ratio for one listener; the float value is the ratio."""

    def __new__(cls, listener, ratio: float):
        instance = super().__new__(cls, _clamp(ratio))
        instance.listener = listener
        return instance

    @property
    def ratio(self) -> float:
        return float(self)


def per_listener(listeners=1, occlusion=0.0):
    """Build an independent occlusion value for every listener.

    ``listeners`` may be a count, an iterable of listener ids, or objects.
    ``occlusion`` may be one value applied to all listeners, a sequence
    aligned with the listeners, or a mapping keyed by listener.  The result
    is a tuple of :class:`ListenerOcclusion` values, one per listener, so a
    split-screen second player never reuses the first listener's result.
    """
    if isinstance(listeners, numbers.Integral) and not isinstance(listeners, bool):
        listener_ids = tuple(range(max(0, int(listeners))))
    else:
        listener_ids = tuple(listeners)

    if isinstance(occlusion, Mapping):
        values = [
            occlusion.get(listener, occlusion.get(str(listener), 0.0))
            for listener in listener_ids
        ]
    elif isinstance(occlusion, (list, tuple)):
        if occlusion:
            values = [occlusion[index % len(occlusion)] for index in range(len(listener_ids))]
        else:
            values = [0.0] * len(listener_ids)
    elif callable(occlusion):
        values = [occlusion(listener) for listener in listener_ids]
    else:
        values = [
            getattr(listener, "occlusion", occlusion)
            for listener in listener_ids
        ]

    results = tuple(
        ListenerOcclusion(listener, value)
        for listener, value in zip(listener_ids, values)
    )
    return results


def query_budget(
    sources,
    budget: int,
    *,
    frame: int = 0,
    priorities: Mapping | Iterable[float] | None = None,
):
    """Choose at most *budget* source queries for this frame.

    ``sources`` is a count or an iterable of source ids.  Without explicit
    priorities the selection is spread evenly over the source list and
    rotated by *frame*, which lets every source be refreshed over a few
    frames instead of running the full set every frame.  When priorities
    are supplied, the highest-priority sources occupy the budget.
    """
    if isinstance(sources, numbers.Integral) and not isinstance(sources, bool):
        items = list(range(max(0, int(sources))))
    else:
        items = list(sources)

    count = len(items)
    budget = max(0, min(int(budget), count))
    if budget == 0:
        return ()
    if budget >= count:
        return tuple(items)

    if priorities is not None:
        values = priorities if isinstance(priorities, Mapping) else list(priorities)

        def priority_of(index: int) -> float:
            item = items[index]
            if isinstance(values, Mapping):
                return float(values.get(item, 0.0))
            return float(values[index]) if index < len(values) else 0.0

        ranked = sorted(range(count), key=lambda index: (-priority_of(index), index))
        return tuple(items[index] for index in sorted(ranked[:budget]))

    step = count / budget
    start = (int(frame) % count) / count
    chosen = sorted({int((start + k / budget) * count) % count for k in range(budget)})
    cursor = 0
    while len(chosen) < budget and cursor < count:
        if cursor not in chosen:
            chosen.append(cursor)
        cursor += 1
    chosen.sort()
    return tuple(items[index] for index in chosen[:budget])


def ray_budget(
    rays: int,
    sources=1,
    *,
    priorities: Mapping | Iterable[float] | None = None,
    min_rays: int = 1,
    max_rays: int = 16,
):
    """Split a ray budget across sources.

    Returns a tuple with the number of rays to cast for each source.  The
    split is even by default; any remainder (or a budget smaller than the
    source count) goes to the highest-priority sources when priorities are
    provided.  Per-source allocation is capped by *max_rays*.
    """
    if isinstance(sources, numbers.Integral) and not isinstance(sources, bool):
        source_ids = tuple(range(max(0, int(sources))))
    else:
        source_ids = tuple(sources)
    count = len(source_ids)
    if count == 0:
        return ()

    max_rays = max(1, int(max_rays))
    min_rays = max(0, int(min_rays))
    total = max(0, min(int(rays), count * max_rays))

    base, remainder = divmod(total, count)
    counts = [base] * count

    if priorities is not None and remainder:
        if isinstance(priorities, Mapping):
            def priority_of(index: int) -> float:
                return float(priorities.get(source_ids[index], 0.0))
        else:
            values = list(priorities)

            def priority_of(index: int) -> float:
                return float(values[index]) if index < len(values) else 0.0

        order = sorted(range(count), key=lambda index: (-priority_of(index), index))
        for index in order[:remainder]:
            counts[index] += 1
    else:
        for index in range(remainder):
            counts[index] += 1

    if total >= min_rays * count:
        counts = [max(value, min_rays) for value in counts]
    counts = [min(value, max_rays) for value in counts]
    return tuple(counts)


# ---------------------------------------------------------------------------
# Reverb, symmetry and materials
# ---------------------------------------------------------------------------


def reverb_link(
    occlusion: float,
    reverb_send: float,
    *,
    max_reduction: float = 0.75,
    decay_floor: float = 0.35,
) -> ReverbLink:
    """Couple reverb to occlusion.

    A fully occluded source keeps only ``1 - max_reduction`` of its send and
    its decay is shortened towards ``decay_floor`` of the open-space value.
    This removes the open-field tail that otherwise leaks through walls.
    """
    amount = _clamp(occlusion)
    send = max(0.0, float(reverb_send))
    reduction = _clamp(max_reduction)
    floor = _clamp(decay_floor)
    return ReverbLink(
        wet_send=send * (1.0 - reduction * amount),
        decay_scale=1.0 - (1.0 - floor) * amount,
        dry_gain=1.0 - amount,
    )


def symmetric_calc(direction_a, direction_b=None) -> float:
    """Return one occlusion value for both directions of a pair.

    Occlusion must not depend on which endpoint is called the source.  When
    both directional estimates are supplied, they are clamped and averaged;
    the same value is then used for ``source -> listener`` and
    ``listener -> source``.
    """
    first = _clamp(direction_a)
    if direction_b is None:
        return first
    return _clamp((first + _clamp(direction_b)) * 0.5)


def material_band(
    material,
    thickness: float = 1.0,
    *,
    min_lowpass: float = MIN_LOWPASS_HZ,
) -> MaterialBand:
    """Apply a material and wall thickness to transmission and filtering.

    Glass and concrete no longer share an attenuation: each material has its
    own per-metre transmission and characteristic cutoff, and thickness is
    applied exponentially.
    """
    profile = _resolve_material(material)
    thickness = max(0.0, float(thickness))

    if thickness <= 0.0:
        transmission = 1.0
        attenuation_db = 0.0
        lowpass = min(profile.lowpass_hz, MAX_LOWPASS_HZ)
    else:
        transmission = _clamp(profile.transmission ** thickness)
        attenuation_db = -20.0 * math.log10(max(transmission, 1e-12))
        lowpass = max(
            float(min_lowpass),
            min(profile.lowpass_hz / math.sqrt(thickness), MAX_LOWPASS_HZ),
        )

    return MaterialBand(
        material=profile,
        thickness=thickness,
        transmission=transmission,
        attenuation_db=attenuation_db,
        lowpass_hz=lowpass,
    )
