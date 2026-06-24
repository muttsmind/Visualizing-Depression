from __future__ import annotations

"""
Pairwise image bank for the CES-D visual-choice setup.

Each image has one hidden 4D clinical vector:
    depletion, fog, disconnection, burden

Each image also has one render-only style coordinate:
    color_wash

color_wash is intentionally not part of LATENT_DIMS. It controls how much
faint watercolor/pastel tint the renderer applies, while the model-facing
measurement space remains 4D.

The participant-facing task is pairwise: choose left vs right.
The image set contains:
    64 total images
        16 single-dimension anchors
        24 two-dimension blends
        12 ambiguous mixed images
        12 foil/control images
"""

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


LATENT_DIMS = ("depletion", "fog", "disconnection", "burden")
MODULES = ("energy", "clarity", "connection", "burden")
DIM_TO_MODULE = {
    "depletion": "energy",
    "fog": "clarity",
    "disconnection": "connection",
    "burden": "burden",
}

# Motifs give images different local compositions while keeping the same
# 4D latent-vector structure
MOTIFS = (
    "winged_mantle",
    "tidal_pour",
    "folded_column",
    "mist_window",
    "splinter_cloud",
    "central_cavern",
    "paired_lobes",
    "island_gap",
    "eclipse_twins",
    "overhead_mass",
    "sunken_body",
    "root_plume",
)

# Participant-facing prompts are neutral. They are meant to reduce obvious
# face-validity rather than literally measure the "subconscious"
PAIRWISE_PROMPTS = [
    {"prompt_id": "P01", "text": "Which image feels closer to your past week?"},
    {"prompt_id": "P02", "text": "Which image feels more familiar right now?"},
    {"prompt_id": "P03", "text": "Which image would you rather move away from?"},
    {"prompt_id": "P04", "text": "Which image feels more like the background of your day?"},
    {"prompt_id": "P05", "text": "Which image feels more emotionally loud?"},
    {"prompt_id": "P06", "text": "Which image feels more like your current inner weather?"},
    {"prompt_id": "P07", "text": "Which image feels harder to leave?"},
    {"prompt_id": "P08", "text": "Which image feels more difficult to describe?"},
    {"prompt_id": "P09", "text": "Which image pulls your attention more?"},
    {"prompt_id": "P10", "text": "Which image feels more like the last few mornings?"},
    {"prompt_id": "P11", "text": "Which image feels more like the end of the day?"},
    {"prompt_id": "P12", "text": "Which image feels more like a place you have been in recently?"},
]


@dataclass(frozen=True)
class ImageSpec:
    image_id: str
    family: str
    module: str
    motif: str
    note: str
    seed: int
    seed_vector: dict[str, float]
    color_wash: float


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def stable_seed(*parts: object) -> int:
    """Deterministic seed for image rendering and pair generation."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def vec(**kwargs: float) -> dict[str, float]:
    """Create a 4D vector with omitted dims defaulting to a mild baseline."""
    out = {d: 0.12 for d in LATENT_DIMS}
    for k, v in kwargs.items():
        out[k] = clip01(v)
    return out


def dominant_dim(v: dict[str, float]) -> str:
    return max(LATENT_DIMS, key=lambda d: float(v[d]))


def color_wash_from_vec(v: dict[str, float], family: str) -> float:
    """Derive a conservative render-only pastel strength from the 4D vector.

    This is not a fifth clinical/model dimension. It is metadata for the
    renderer, kept low so color behaves like watercolor staining rather than
    a strong participant-facing cue. Foils are slightly dampened so the test
    cannot rely on a cheap "more colorful => more severe" shortcut.
    """
    values = np.array([float(v[d]) for d in LATENT_DIMS], dtype=float)
    severity = float(values.max())
    spread = float(values.std())

    # The renderer expects color_wash to be a visible 0...1 render coordinate
    wash = 0.20 + 0.58 * severity + 0.08 * spread
    if family == "foil":
        wash *= 0.78
    elif family == "ambiguous":
        wash *= 0.90

    return float(np.clip(wash, 0.16, 0.82))


def add(
    rows: list[ImageSpec],
    family: str,
    note: str,
    seed_vector: dict[str, float],
    motif_index: int,
    color_wash: float | None = None,
) -> None:
    image_id = f"IMG_{len(rows) + 1:03d}"
    dim = dominant_dim(seed_vector)
    module = DIM_TO_MODULE[dim]
    motif = MOTIFS[motif_index % len(MOTIFS)]
    wash = color_wash_from_vec(seed_vector, family) if color_wash is None else clip01(color_wash)
    rows.append(
        ImageSpec(
            image_id=image_id,
            family=family,
            module=module,
            motif=motif,
            note=note,
            seed=stable_seed(image_id, motif, family),
            seed_vector=seed_vector,
            color_wash=wash,
        )
    )


def build_image_bank() -> dict[str, dict[str, Any]]:
    rows: list[ImageSpec] = []
    motif_i = 0

    # ------------------------------------------------------------------
    # 1) Single-dimension anchors (16)
    # ------------------------------------------------------------------
    anchor_levels = [0.22, 0.38, 0.60, 0.82]
    spill = {
        "depletion": {"fog": 0.10, "disconnection": 0.12, "burden": 0.18},
        "fog": {"depletion": 0.16, "disconnection": 0.12, "burden": 0.14},
        "disconnection": {"depletion": 0.12, "fog": 0.12, "burden": 0.16},
        "burden": {"depletion": 0.16, "fog": 0.14, "disconnection": 0.12},
    }
    for dim in LATENT_DIMS:
        for lvl in anchor_levels:
            v = vec(**spill[dim], **{dim: lvl})
            add(rows, "anchor", f"single-dim anchor for {dim}", v, motif_i)
            motif_i += 1

    # ------------------------------------------------------------------
    # 2) Two-dimension blends (24)
    # ------------------------------------------------------------------
    # These are the main non-obvious stimuli. No image is simply "good" or "bad"
    blend_patterns = [
        (0.72, 0.46),
        (0.60, 0.60),
        (0.80, 0.28),
        (0.54, 0.74),
    ]
    dim_pairs = [
        ("depletion", "fog"),
        ("depletion", "disconnection"),
        ("depletion", "burden"),
        ("fog", "disconnection"),
        ("fog", "burden"),
        ("disconnection", "burden"),
    ]
    for d1, d2 in dim_pairs:
        for a, b in blend_patterns:
            v = vec(**{d1: a, d2: b})
            # Small nonzero values on the remaining dims keep the bank realistic.
            for d in LATENT_DIMS:
                if d not in {d1, d2}:
                    v[d] = 0.12 + 0.05 * ((motif_i + len(d)) % 3)
            add(rows, "blend", f"blend of {d1} and {d2}", v, motif_i)
            motif_i += 1

    # ------------------------------------------------------------------
    # 3) Ambiguous mixed images (12)
    # ------------------------------------------------------------------
    ambiguous = [
        ("dark but organized", vec(depletion=0.28, fog=0.18, disconnection=0.22, burden=0.72)),
        ("foggy but calm", vec(depletion=0.22, fog=0.78, disconnection=0.18, burden=0.20)),
        ("fragmented but light", vec(depletion=0.18, fog=0.34, disconnection=0.76, burden=0.16)),
        ("heavy but symmetrical", vec(depletion=0.34, fog=0.16, disconnection=0.18, burden=0.78)),
        ("slowed but spacious", vec(depletion=0.72, fog=0.24, disconnection=0.48, burden=0.18)),
        ("diffuse but connected", vec(depletion=0.18, fog=0.74, disconnection=0.18, burden=0.28)),
        ("distant but clean", vec(depletion=0.18, fog=0.20, disconnection=0.78, burden=0.22)),
        ("dense but not burdened", vec(depletion=0.46, fog=0.22, disconnection=0.26, burden=0.26)),
        ("energetic chaos", vec(depletion=0.20, fog=0.52, disconnection=0.44, burden=0.22)),
        ("quiet pressure", vec(depletion=0.40, fog=0.24, disconnection=0.22, burden=0.64)),
        ("faded separation", vec(depletion=0.26, fog=0.60, disconnection=0.62, burden=0.20)),
        ("sunken but clear", vec(depletion=0.68, fog=0.18, disconnection=0.22, burden=0.46)),
    ]
    for note, v in ambiguous:
        add(rows, "ambiguous", note, v, motif_i)
        motif_i += 1

    # ------------------------------------------------------------------
    # 4) Foils / controls (12)
    # ------------------------------------------------------------------
    # These help prevent cheap shortcuts such as "darker => more depressed"
    foils = [
        ("dark foil", vec(depletion=0.24, fog=0.24, disconnection=0.18, burden=0.34)),
        ("pale fog foil", vec(depletion=0.18, fog=0.62, disconnection=0.14, burden=0.16)),
        ("wide distance foil", vec(depletion=0.12, fog=0.18, disconnection=0.60, burden=0.20)),
        ("compressed foil", vec(depletion=0.28, fog=0.16, disconnection=0.14, burden=0.58)),
        ("spacious low-burden foil", vec(depletion=0.20, fog=0.24, disconnection=0.44, burden=0.14)),
        ("dense low-fog foil", vec(depletion=0.42, fog=0.12, disconnection=0.18, burden=0.30)),
        ("soft low-burden foil", vec(depletion=0.20, fog=0.52, disconnection=0.16, burden=0.18)),
        ("split low-fog foil", vec(depletion=0.18, fog=0.18, disconnection=0.64, burden=0.14)),
        ("weighted but coherent foil", vec(depletion=0.34, fog=0.12, disconnection=0.16, burden=0.66)),
        ("blurred middle foil", vec(depletion=0.32, fog=0.48, disconnection=0.24, burden=0.22)),
        ("restless foil", vec(depletion=0.54, fog=0.36, disconnection=0.22, burden=0.22)),
        ("mixed neutral foil", vec(depletion=0.28, fog=0.30, disconnection=0.28, burden=0.28)),
    ]
    for note, v in foils:
        add(rows, "foil", note, v, motif_i)
        motif_i += 1

    if len(rows) != 64:
        raise AssertionError(f"Expected 64 images, got {len(rows)}")

    return {
        r.image_id: {
            "image_id": r.image_id,
            "family": r.family,
            "module": r.module,
            "motif": r.motif,
            "note": r.note,
            "seed": r.seed,
            "seed_vector": r.seed_vector,
            "color_wash": r.color_wash,
        }
        for r in rows
    }


IMAGE_BANK = build_image_bank()


def pair_distance(v1: dict[str, float], v2: dict[str, float]) -> float:
    a = np.array([v1[d] for d in LATENT_DIMS], dtype=float)
    b = np.array([v2[d] for d in LATENT_DIMS], dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def build_trial_bank(
    n_trials: int = 40,
    seed: int = 7,
    min_dist: float = 0.30,
    max_dist: float = 1.00,
) -> pd.DataFrame:
    """
    Build one fixed pairwise study design.

    Design goals:
        - prompt texts are reused cyclically
        - image pairs have moderate 4D separation
        - family pairing is mildly balanced so one family does not dominate
    """
    rng = np.random.default_rng(seed)
    ids = list(IMAGE_BANK.keys())
    families = {img_id: IMAGE_BANK[img_id]["family"] for img_id in ids}

    # Precompute allowed pairs
    allowed: list[tuple[str, str, float]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            dist = pair_distance(IMAGE_BANK[a]["seed_vector"], IMAGE_BANK[b]["seed_vector"])
            if min_dist <= dist <= max_dist:
                allowed.append((a, b, dist))

    if len(allowed) < n_trials:
        raise ValueError("Not enough candidate pairs for the requested design.")

    # Mild balancing by family pairing to avoid too many near-identical stimuli
    rng.shuffle(allowed)
    selected: list[tuple[str, str, float]] = []
    used = set()
    family_counts: dict[tuple[str, str], int] = {}

    for a, b, dist in allowed:
        key = tuple(sorted((families[a], families[b])))
        if (a, b) in used:
            continue
        if family_counts.get(key, 0) >= max(2, n_trials // 8):
            continue
        selected.append((a, b, dist))
        used.add((a, b))
        family_counts[key] = family_counts.get(key, 0) + 1
        if len(selected) >= n_trials:
            break

    if len(selected) < n_trials:
        # Fallback: just top up with remaining allowed pairs
        for a, b, dist in allowed:
            if (a, b) not in used:
                selected.append((a, b, dist))
                used.add((a, b))
                if len(selected) >= n_trials:
                    break

    rows: list[dict[str, Any]] = []
    for trial_index, (a, b, dist) in enumerate(selected, start=1):
        prompt = PAIRWISE_PROMPTS[(trial_index - 1) % len(PAIRWISE_PROMPTS)]
        if rng.random() < 0.5:
            left, right = a, b
        else:
            left, right = b, a
        rows.append({
            "trial_id": f"T{trial_index:03d}",
            "trial_index": trial_index,
            "prompt_id": prompt["prompt_id"],
            "prompt_text": prompt["text"],
            "left_image_id": left,
            "right_image_id": right,
            "pair_distance": dist,
        })

    return pd.DataFrame(rows)
