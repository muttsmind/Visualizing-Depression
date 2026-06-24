from __future__ import annotations

"""
Rorschach-style renderer for the pairwise CES-D visual-choice image bank.

This module contains only the image-composition grammar used by the pairwise
pipeline. Each image is rendered from these inputs:
    vec        = 4D vector [depletion, fog, disconnection, burden]
    module     = dominant category/style context
    prompt_id  = image id used to look up the motif recipe
    seed       = deterministic image seed
    color_wash = optional render-only pastel/watercolor strength
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


LATENT_DIMS = ("depletion", "fog", "disconnection", "burden")

# For each prompt category, which latent dim should dominate the interpretation
MODULE_TO_DIM = {
    "energy": "depletion",
    "clarity": "fog",
    "connection": "disconnection",
    "burden": "burden",
}

# Motif recipes create local composition differences between images while
# preserving the same underlying 4D visual grammar
PROMPT_MOTIFS = {
    "E1": "winged_mantle",
    "E2": "tidal_pour",
    "E3": "folded_column",
    "C1": "mist_window",
    "C2": "splinter_cloud",
    "C3": "central_cavern",
    "S1": "paired_lobes",
    "S2": "island_gap",
    "S3": "eclipse_twins",
    "B1": "overhead_mass",
    "B2": "sunken_body",
    "B3": "root_plume",
}


# Local composition controls (These are not literal icons!!!)

# wide: horizontal spread
# vertical: tall vs flat
# void: more/less empty center
# tendril: drips/streaks
# grain: noisy/soft texture

MOTIF_STYLE = {
    "winged_mantle":  {"wide": 0.18, "vertical": -0.04, "void": -0.03, "tendril": 0.01, "grain": 0.02},
    "tidal_pour":     {"wide": 0.05, "vertical": 0.18, "void": -0.02, "tendril": 0.12, "grain": 0.02},
    "folded_column":  {"wide": -0.08, "vertical": 0.22, "void": -0.04, "tendril": 0.06, "grain": 0.01},
    "mist_window":    {"wide": 0.00, "vertical": 0.02, "void": 0.04, "tendril": 0.01, "grain": 0.09},
    "splinter_cloud": {"wide": 0.08, "vertical": 0.00, "void": 0.08, "tendril": 0.03, "grain": 0.13},
    "central_cavern": {"wide": 0.02, "vertical": 0.00, "void": 0.16, "tendril": 0.00, "grain": 0.06},
    "paired_lobes":   {"wide": 0.12, "vertical": -0.02, "void": 0.08, "tendril": 0.00, "grain": 0.03},
    "island_gap":     {"wide": 0.18, "vertical": 0.00, "void": 0.20, "tendril": 0.02, "grain": 0.04},
    "eclipse_twins":  {"wide": 0.14, "vertical": 0.00, "void": 0.18, "tendril": 0.00, "grain": 0.03},
    "overhead_mass":  {"wide": 0.04, "vertical": 0.14, "void": -0.05, "tendril": 0.05, "grain": 0.05},
    "sunken_body":    {"wide": 0.08, "vertical": 0.22, "void": -0.08, "tendril": 0.09, "grain": 0.03},
    "root_plume":     {"wide": 0.03, "vertical": 0.24, "void": -0.03, "tendril": 0.16, "grain": 0.05},
}


# Stronger module-level signatures
# These survive averaging across prompts better than motif-only variation
MODULE_STYLE = {
    "energy": {
        # Depletion: lower, pooled, slumped, less fragmented
        "wide": -0.10,
        "vertical": 0.12,
        "y_shift": 0.10,
        "x_gap": -0.04,
        "void": -0.10,
        "soft": -0.02,
        "edge": -0.04,
        "islands": -0.08,
        "gravity": 0.18,
        "tendril": 0.04,
        "mass": 0.02,
        "darkness": -0.02,
    },
    "clarity": {
        # Fog: soft, diffuse, partially erased, less solid
        "wide": 0.04,
        "vertical": -0.10,
        "y_shift": -0.02,
        "x_gap": 0.02,
        "void": 0.08,
        "soft": 0.28,
        "edge": 0.14,
        "islands": 0.05,
        "gravity": -0.08,
        "tendril": -0.06,
        "mass": -0.08,
        "darkness": -0.12,
    },
    "connection": {
        # Disconnection: paired lobes, wider gap, more detached fragments
        "wide": 0.20,
        "vertical": -0.02,
        "y_shift": 0.00,
        "x_gap": 0.22,
        "void": 0.28,
        "soft": -0.04,
        "edge": 0.16,
        "islands": 0.18,
        "gravity": -0.02,
        "tendril": -0.05,
        "mass": -0.02,
        "darkness": 0.02,
    },
    "burden": {
        # Burden: compressed, darker, pressured, downward streaking
        "wide": 0.02,
        "vertical": 0.14,
        "y_shift": 0.05,
        "x_gap": -0.06,
        "void": -0.16,
        "soft": -0.08,
        "edge": 0.10,
        "islands": -0.05,
        "gravity": 0.30,
        "tendril": 0.22,
        "mass": 0.12,
        "darkness": 0.18,
    },
}

ZERO_MODULE_STYLE = {
    "wide": 0.0,
    "vertical": 0.0,
    "y_shift": 0.0,
    "x_gap": 0.0,
    "void": 0.0,
    "soft": 0.0,
    "edge": 0.0,
    "islands": 0.0,
    "gravity": 0.0,
    "tendril": 0.0,
    "mass": 0.0,
    "darkness": 0.0,
}


# Render-only pastel watercolor palette. These colors are intentionally muted.
# Hue is blended from the existing 4D vector, while color_wash controls opacity.
# This keeps the clinical/model space 4D and treats color as visual style
PASTEL_TINT_BY_DIM = {
    # Still pastel/watercolor, but no longer so gray that it reads as monochrome
    "depletion": np.array([118, 150, 205]),      # muted blue
    "fog": np.array([118, 184, 162]),            # sage/teal
    "disconnection": np.array([174, 130, 205]),  # lavender
    "burden": np.array([210, 124, 112]),         # muted rose
}

MODULE_PASTEL_BIAS = {
    "energy": np.array([118, 150, 205]),
    "clarity": np.array([118, 184, 162]),
    "connection": np.array([174, 130, 205]),
    "burden": np.array([210, 124, 112]),
}


HEURISTIC_CHANNELS = {
    "depletion": {
        "meaning": "low energy, slowing, difficulty initiating",
        "visual_channels": [
            "lower center of gravity",
            "less upward reach",
            "slumped oval mass",
            "muted contrast at low values, dense pooling at high values",
        ],
    },
    "fog": {
        "meaning": "mental haze, loss of clarity, difficulty holding shape",
        "visual_channels": [
            "soft edges",
            "blurred washes around ink",
            "higher texture noise",
            "partly erased / occluded internal structure",
        ],
    },
    "disconnection": {
        "meaning": "distance, isolation, fragmentation, unreachable relation",
        "visual_channels": [
            "central negative space",
            "paired but separated lobes",
            "small detached islands",
            "reduced internal integration while preserving bilateral fold",
        ],
    },
    "burden": {
        "meaning": "weight, pressure, emotional load",
        "visual_channels": [
            "darker upper or lower mass",
            "vertical compression",
            "downward streaks",
            "lower openness / less white space at high values",
        ],
    },
}


@dataclass(frozen=True)
class RorschachGrammar:
    intensity: float
    mass: float
    darkness: float
    symmetry: float
    integration: float
    boundary_softness: float
    edge_complexity: float
    central_void: float
    satellite_islands: float
    gravity: float
    verticality: float
    tendrils: float
    paper_grain: float
    fold_axis: float
    motif: str
    module: str
    dominant_dim: str
    depletion: float
    fog: float
    disconnection: float
    burden: float


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))



def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return int(np.uint32(sum((i + 1) * ord(ch) for i, ch in enumerate(text))))


def vec_from_dict(vec: Mapping[str, float]) -> np.ndarray:
    return np.clip(np.array([float(vec[d]) for d in LATENT_DIMS], dtype=float), 0.0, 1.0)


def derive_rorschach_grammar(
    vec: np.ndarray,
    module: str,
    prompt_id: str,
) -> RorschachGrammar:
    """Map the 4D latent vector into Rorschach-like visual channels."""

    depletion, fog, disconnection, burden = [float(x) for x in vec.tolist()]


    module_dim = MODULE_TO_DIM.get(module, "burden")
    module_strength = float(vec[LATENT_DIMS.index(module_dim)])
    # mean_state = float(np.mean(vec))
    dominant_dim = LATENT_DIMS[int(np.argmax(vec))]

    motif = PROMPT_MOTIFS.get(prompt_id, "winged_mantle")
    style = MOTIF_STYLE[motif]
    mstyle = MODULE_STYLE.get(module, ZERO_MODULE_STYLE)

    # Overall visual intensity is driven by the prompt/module-relevant latent
    # dimension, with a small contribution from the full 4D state
    state_max = float(np.max(vec))
    off_state = float((np.sum(vec) - module_strength) / 3.0)

    intensity = clip01(
        0.60 * module_strength
        + 0.25 * state_max
        + 0.15 * off_state
    )

    badness = clip01(
        0.45 * intensity
        + 0.55 * state_max
    )

    # Keep folded-inkblot symmetry, but allow disconnection to loosen integration
    symmetry = clip01(
        0.96
        - 0.50 * disconnection
        - 0.08 * fog
        - 0.04 * max(0.0, mstyle["x_gap"])
    )

    # Module signature is stronger than motif signature because aggregate profiles
    # average across prompts
    mass = clip01(
        0.28
        + 0.28 * intensity
        + 0.20 * burden
        + 0.10 * depletion
        - 0.06 * fog
        + mstyle["mass"]
    )

    darkness = clip01(
        0.38
        + 0.42 * badness
        + 0.18 * burden
        + mstyle["darkness"]
    )

    integration = clip01(
        0.88
        - 0.36 * fog
        - 0.32 * disconnection
        + 0.05 * burden
        - 0.08 * max(0.0, mstyle["x_gap"])
    )

    boundary_softness = clip01(
        0.10
        + 0.85 * fog
        + 0.15 * depletion
        - 0.22 * burden
        + style["grain"]
        + 0.75 * mstyle["soft"]
    )

    edge_complexity = clip01(
        0.16
        + 0.28 * fog
        + 0.30 * disconnection
        + 0.18 * badness
        + style["grain"]
        + mstyle["edge"]
    )

    central_void = clip01(
        0.08
        + 0.62 * disconnection
        + 0.20 * fog
        - 0.08 * burden
        + style["void"]
        + 0.65 * mstyle["void"]
    )

    satellite_islands = clip01(
        0.04
        + 0.28 * disconnection
        + 0.18 * fog
        + mstyle["islands"]
    )

    gravity = clip01(
        0.08
        + 0.46 * burden
        + 0.28 * depletion
        + 0.10 * intensity
        + mstyle["gravity"]
    )

    verticality = clip01(
        0.20
        + 0.36 * burden
        + 0.18 * depletion
        + style["vertical"]
        + mstyle["vertical"]
    )

    tendrils = clip01(
        0.02
        + 0.26 * burden
        + 0.16 * depletion
        + 0.10 * fog
        + style["tendril"]
        + mstyle["tendril"]
    )

    paper_grain = clip01(
        0.03
        + 0.20 * fog
        + 0.05 * badness
        + style["grain"]
    )

    fold_axis = clip01(0.22 + 0.36 * symmetry - 0.08 * darkness)

    return RorschachGrammar(
        intensity=intensity,
        mass=mass,
        darkness=darkness,
        symmetry=symmetry,
        integration=integration,
        boundary_softness=boundary_softness,
        edge_complexity=edge_complexity,
        central_void=central_void,
        satellite_islands=satellite_islands,
        gravity=gravity,
        verticality=verticality,
        tendrils=tendrils,
        paper_grain=paper_grain,
        fold_axis=fold_axis,
        motif=motif,
        module=module,
        dominant_dim=dominant_dim,
        depletion=depletion,
        fog=fog,
        disconnection=disconnection,
        burden=burden,
    )


def pastel_rgb_from_grammar(g: RorschachGrammar) -> np.ndarray:
    """Blend a muted pastel tint from the existing 4D latent vector.

    The hue follows the 4D state, but opacity is controlled separately by
    color_wash in compose_rorschach(). This prevents color from replacing the
    core black-ink morphology.
    """
    weights = np.array([g.depletion, g.fog, g.disconnection, g.burden], dtype=float)
    if float(weights.sum()) < 1e-8:
        weights = np.ones(len(LATENT_DIMS), dtype=float) / len(LATENT_DIMS)
    else:
        weights = weights / float(weights.sum())

    palette = np.array([PASTEL_TINT_BY_DIM[d] for d in LATENT_DIMS], dtype=float)
    rgb = weights @ palette

    module_bias = MODULE_PASTEL_BIAS.get(g.module)
    if module_bias is not None:
        rgb = 0.78 * rgb + 0.22 * module_bias.astype(float)

    return np.clip(rgb, 0, 255).astype(int)


def paper(size: int, g: RorschachGrammar, rng: np.random.Generator) -> Image.Image:
    base = np.array([246, 244, 240], dtype=float)

    yy, xx = np.mgrid[0:size, 0:size]
    radial = np.sqrt(((xx - size / 2) / size) ** 2 + ((yy - size / 2) / size) ** 2)
    radial = 1.0 - np.clip(radial * 1.8, 0.0, 1.0)

    noise = rng.normal(0, 1, (size, size))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-12)

    grain = 0.030 + 0.075 * g.paper_grain

    arr = (
        base[None, None, :]
        + radial[..., None] * 5.0
        + (noise[..., None] - 0.5) * 34.0 * grain
    )

    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def organic_polygon(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rng: np.random.Generator,
    points: int = 36,
    roughness: float = 0.12,
    angle: float = 0.0,
) -> list[tuple[float, float]]:
    pts = []
    phase = rng.uniform(0, 2 * np.pi)

    for i in range(points):
        th = 2 * np.pi * i / points
        low = 0.08 * np.sin(2 * th + phase) + 0.06 * np.sin(3 * th - phase)
        rad = 1.0 + low + rng.normal(0, roughness)

        x = rx * rad * np.cos(th)
        y = ry * rad * np.sin(th)

        xr = x * np.cos(angle) - y * np.sin(angle)
        yr = x * np.sin(angle) + y * np.cos(angle)

        pts.append((cx + xr, cy + yr))

    return pts


def apply_module_signature(
    mask: Image.Image,
    g: RorschachGrammar,
    rng: np.random.Generator,
    cx_base: float,
    cy_base: float,
    sx: float,
    sy: float,
) -> None:
    """Add stronger module-specific morphology after the shared blot grammar."""

    size = mask.size[1]
    half_w = mask.size[0]

    if g.module == "energy":
        # Low, pooled, slumped base. This makes depletion less like burden
        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        rx = sx * (0.70 + 0.40 * g.depletion)
        ry = sy * (0.28 + 0.30 * g.depletion)
        cy = cy_base + sy * (0.42 + 0.22 * g.gravity)

        td.ellipse(
            (cx_base - rx, cy - ry, cx_base + rx * 0.85, cy + ry),
            fill=int(35 + 85 * g.depletion),
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=2.0 + 4.0 * g.boundary_softness)
        )

        mask.paste(ImageChops.lighter(mask, temp))

    elif g.module == "clarity":
        # Diffuse fog field plus partial erasure
        for _ in range(3 + int(5 * g.fog)):
            temp = Image.new("L", mask.size, 0)
            td = ImageDraw.Draw(temp)

            cx = cx_base + rng.normal(0, sx * 0.55)
            cy = cy_base + rng.normal(0, sy * 0.55)
            rx = sx * rng.uniform(0.55, 1.20)
            ry = sy * rng.uniform(0.35, 1.05)

            td.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry),
                fill=int(12 + 38 * g.fog),
            )

            temp = temp.filter(ImageFilter.GaussianBlur(radius=8 + 12 * g.fog))
            mask.paste(ImageChops.lighter(mask, temp))

        for _ in range(1 + int(4 * g.fog)):
            temp = Image.new("L", mask.size, 0)
            td = ImageDraw.Draw(temp)

            cx = cx_base + rng.normal(0, sx * 0.45)
            cy = cy_base + rng.normal(0, sy * 0.45)
            rx = sx * rng.uniform(0.16, 0.38)
            ry = sy * rng.uniform(0.12, 0.34)

            td.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry),
                fill=int(25 + 55 * g.fog),
            )

            temp = temp.filter(ImageFilter.GaussianBlur(radius=5 + 7 * g.fog))
            mask.paste(ImageChops.subtract(mask, temp))

    elif g.module == "connection":
        # Central separation while keeping bilateral symmetry
        gap = Image.new("L", mask.size, 0)
        gd = ImageDraw.Draw(gap)

        gap_w = size * (0.030 + 0.10 * g.disconnection + 0.045 * g.central_void)

        pts = organic_polygon(
            cx=half_w - gap_w * 0.15,
            cy=size * 0.50,
            rx=gap_w * rng.uniform(1.1, 1.6),
            ry=size * rng.uniform(0.22, 0.36),
            rng=rng,
            points=28,
            roughness=0.10 + 0.12 * g.fog,
        )

        gd.polygon(
            pts,
            fill=int(55 + 105 * g.disconnection),
        )

        gap = gap.filter(
            ImageFilter.GaussianBlur(radius=1.0 + 2.0 * g.boundary_softness)
        )

        mask.paste(ImageChops.subtract(mask, gap))

        # Lateral fragments reinforce distance rather than fog
        for _ in range(1 + int(5 * g.disconnection)):
            temp = Image.new("L", mask.size, 0)
            td = ImageDraw.Draw(temp)

            cx = cx_base - sx * rng.uniform(0.65, 1.35)
            cy = cy_base + rng.normal(0, sy * 0.95)
            rx = sx * rng.uniform(0.045, 0.14)
            ry = sy * rng.uniform(0.045, 0.16)

            td.polygon(
                organic_polygon(
                    cx=cx,
                    cy=cy,
                    rx=rx,
                    ry=ry,
                    rng=rng,
                    points=12,
                    roughness=0.14 + 0.18 * g.edge_complexity,
                ),
                fill=int(45 + 100 * g.darkness),
            )

            temp = temp.filter(
                ImageFilter.GaussianBlur(radius=0.3 + 0.8 * g.boundary_softness)
            )

            mask.paste(ImageChops.lighter(mask, temp))

    elif g.module == "burden":
        # Upper pressure cap: burden should feel compressed, not merely depleted
        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        rx = sx * (0.90 + 0.35 * g.burden)
        ry = sy * (0.20 + 0.22 * g.burden)
        cy = cy_base - sy * (0.48 - 0.10 * g.gravity)

        td.ellipse(
            (cx_base - rx, cy - ry, cx_base + rx * 0.85, cy + ry),
            fill=int(40 + 110 * g.burden),
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=2.0 + 3.0 * g.boundary_softness)
        )

        mask.paste(ImageChops.lighter(mask, temp))

        # Longer downward ink behavior
        draw = ImageDraw.Draw(mask)

        for _ in range(2 + int(6 * g.burden)):
            x = cx_base + rng.normal(0, sx * 0.42)
            y = cy_base + sy * rng.uniform(0.05, 0.55)
            length = size * rng.uniform(0.05, 0.18) * (0.75 + g.gravity)
            width = max(1, int(size * rng.uniform(0.003, 0.009)))
            alpha = int(35 + 100 * g.darkness)

            draw.line(
                (x, y, x + rng.normal(0, size * 0.010), y + length),
                fill=alpha,
                width=width,
            )


def draw_lobe(mask: Image.Image, g: RorschachGrammar, rng: np.random.Generator) -> None:
    """Draw the left half of the blot into an L-mode mask."""

    size = mask.size[1]
    half_w = mask.size[0]

    pad = size * 0.10   # 0.08–0.14
    usable = size - 2 * pad

    style = MOTIF_STYLE[g.motif]
    mstyle = MODULE_STYLE.get(g.module, ZERO_MODULE_STYLE)

    draw = ImageDraw.Draw(mask)

    cx_base = pad + (half_w - pad) * (
        0.78
        - 0.24 * g.central_void
        - 0.20 * mstyle["x_gap"]
        + 0.04 * style["wide"]
    )

    cy_base = pad + usable * (
        0.48
        + 0.16 * g.gravity
        - 0.04 * (1.0 - g.verticality)
        + mstyle["y_shift"]
    )

    sx = usable * max(
        0.08,
        0.16
        + 0.28 * g.mass
        + 0.12 * style["wide"]
        + 0.12 * mstyle["wide"]
        + 0.05 * g.disconnection,
    )

    sy = usable * max(
        0.08,
        0.14
        + 0.26 * g.mass
        + 0.12 * g.verticality
        + 0.10 * mstyle["vertical"],
    )

    rough = 0.04 + 0.24 * g.edge_complexity

    # Main masses
    n_main = int(np.clip(round(2 + 8 * g.mass + 5 * (1.0 - g.integration)), 3, 14))

    for i in range(n_main):
        t = 0.0 if i == 0 else rng.normal(0, 0.45)

        cx = cx_base + rng.normal(0, sx * 0.30) - abs(t) * sx * 0.10
        cy = cy_base + rng.normal(0, sy * 0.42) + g.gravity * sy * 0.25

        rx = sx * rng.uniform(0.34, 0.70) * (1.05 if i == 0 else 0.85)
        ry = sy * rng.uniform(0.30, 0.72) * (1.0 + 0.20 * g.verticality)

        angle = rng.uniform(-0.42, 0.42) + style["vertical"] * rng.uniform(-0.35, 0.35)

        alpha = int(95 + 145 * g.darkness - 7 * i)

        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        td.polygon(
            organic_polygon(
                cx=cx,
                cy=cy,
                rx=rx,
                ry=ry,
                rng=rng,
                roughness=rough,
                angle=angle,
            ),
            fill=max(20, alpha),
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=0.5 + 1.5 * g.boundary_softness)
        )

        mask.paste(ImageChops.lighter(mask, temp))

    # Washes / translucent ink
    n_wash = int(np.clip(round(1 + 8 * g.boundary_softness + 3 * g.mass), 1, 12))

    for _ in range(n_wash):
        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        cx = cx_base + rng.normal(0, sx * 0.42)
        cy = cy_base + rng.normal(0, sy * 0.60)

        rx = sx * rng.uniform(0.34, 0.90)
        ry = sy * rng.uniform(0.28, 0.90)

        alpha = int(18 + 62 * g.boundary_softness + 28 * g.mass)

        td.ellipse(
            (cx - rx, cy - ry, cx + rx, cy + ry),
            fill=int(np.clip(alpha, 0, 120)),
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=4 + int(12 * g.boundary_softness))
        )

        mask.paste(ImageChops.lighter(mask, temp))

    # Internal voids
    n_void = int(np.clip(round(0.5 + 8 * g.central_void), 1, 10))

    for _ in range(n_void):
        vx = cx_base - sx * rng.uniform(0.00, 0.38) - size * 0.05 * g.central_void
        vy = cy_base + rng.normal(0, sy * 0.38)

        vrx = sx * rng.uniform(0.12, 0.33) * (1.0 + 0.50 * g.central_void)
        vry = sy * rng.uniform(0.10, 0.30) * (1.0 + 0.25 * g.fog)

        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        td.polygon(
            organic_polygon(
                cx=vx,
                cy=vy,
                rx=vrx,
                ry=vry,
                rng=rng,
                points=24,
                roughness=0.05 + 0.12 * g.fog,
            ),
            fill=190,
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=1.0 + 3.5 * g.boundary_softness)
        )

        mask.paste(ImageChops.subtract(mask, temp))

    # Detached islands
    n_sat = int(np.clip(round(0.5 + 11 * g.satellite_islands), 0, 12))

    for _ in range(n_sat):
        temp = Image.new("L", mask.size, 0)
        td = ImageDraw.Draw(temp)

        direction = rng.choice([-1, 1])

        cx = cx_base + direction * rng.uniform(sx * 0.20, sx * 0.95)
        cy = cy_base + rng.normal(0, sy * 0.95)

        rx = sx * rng.uniform(0.045, 0.16)
        ry = sy * rng.uniform(0.040, 0.16)

        td.polygon(
            organic_polygon(
                cx=cx,
                cy=cy,
                rx=rx,
                ry=ry,
                rng=rng,
                points=14,
                roughness=0.12 + 0.18 * g.edge_complexity,
            ),
            fill=int(55 + 90 * g.darkness),
        )

        temp = temp.filter(
            ImageFilter.GaussianBlur(radius=0.5 + g.boundary_softness)
        )

        mask.paste(ImageChops.lighter(mask, temp))

    # Downward streaks / tendrils
    n_t = int(np.clip(round(11 * g.tendrils), 0, 12))

    for _ in range(n_t):
        x = cx_base + rng.normal(0, sx * 0.36)
        y = cy_base + sy * rng.uniform(0.10, 0.62)

        length = size * rng.uniform(0.035, 0.13) * (0.6 + g.gravity)
        width = max(1, int(size * rng.uniform(0.003, 0.010)))
        alpha = int(40 + 95 * g.darkness)

        draw.line(
            (x, y, x + rng.normal(0, size * 0.012), y + length),
            fill=alpha,
            width=width,
        )

        if rng.random() < 0.65:
            r = width * rng.uniform(1.2, 2.4)
            draw.ellipse(
                (x - r, y + length - r, x + r, y + length + r),
                fill=max(0, alpha - 15),
            )

    # Strong module-level morphology
    apply_module_signature(mask, g, rng, cx_base, cy_base, sx, sy)


def make_symmetric_mask(size: int, g: RorschachGrammar, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)

    half_w = size // 2
    left = Image.new("L", (half_w, size), 0)

    draw_lobe(left, g, rng)

    right = ImageOps.mirror(left)

    # Restrained perturbation => folded-inkblot feeling
    asym = 1.0 - g.symmetry

    if asym > 0.01:
        shifted = ImageChops.offset(
            right,
            int(rng.normal(0, asym * size * 0.035)),
            int(rng.normal(0, asym * size * 0.030)),
        )

        independent = Image.new("L", (half_w, size), 0)
        draw_lobe(independent, g, np.random.default_rng(seed + 1009))
        independent = ImageOps.mirror(independent)

        right = Image.blend(shifted, independent, min(0.22, 0.55 * asym))

    mask = Image.new("L", (size, size), 0)
    mask.paste(left, (0, 0))
    mask.paste(right, (half_w, 0))

    # Fold seam
    seam = Image.new("L", (size, size), 0)
    sd = ImageDraw.Draw(seam)

    seam_alpha = int(24 + 48 * g.fold_axis)

    sd.line(
        (half_w, int(size * 0.08), half_w, int(size * 0.92)),
        fill=seam_alpha,
        width=max(1, size // 220),
    )

    seam = seam.filter(ImageFilter.GaussianBlur(radius=0.7))
    mask = ImageChops.subtract(mask, seam)

    # Absorption and final smoothing
    if g.boundary_softness > 0.05:
        soft = mask.filter(
            ImageFilter.GaussianBlur(radius=0.8 + 3.2 * g.boundary_softness)
        )

        mask = Image.blend(mask, soft, 0.20 + 0.40 * g.boundary_softness)

    return mask


def compose_rorschach(
    size: int,
    module: str,
    prompt_id: str,
    vec: np.ndarray,
    seed: int,
    color_wash: float = 0.0,
) -> Image.Image:
    """Render one pairwise-bank image from its 4D vector and motif context.
    color_wash is a render-only control in [0, 1].
    """
    g = derive_rorschach_grammar(vec, module, prompt_id)
    color_wash = clip01(float(color_wash))

    rng = np.random.default_rng(seed + 47)

    bg = paper(size, g, rng).convert("RGBA")
    mask = make_symmetric_mask(size, g, seed).filter(ImageFilter.GaussianBlur(radius=0.15))

    # subtle global shrink to preserve margins
    mask = mask.resize(
        (int(size * 0.92), int(size * 0.92)),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (size, size), 0)
    offset = (size - mask.size[0]) // 2
    canvas.paste(mask, (offset, offset))
    mask = canvas


    ink_rgb = np.array([22, 23, 25], dtype=int)

    tint_map = {
        "energy": np.array([34, 38, 48]),
        "clarity": np.array([36, 39, 42]),
        "connection": np.array([39, 35, 45]),
        "burden": np.array([42, 32, 34]),
    }

    tint = tint_map.get(module, np.array([34, 34, 34]))

    # Module-level opacity makes aggregate profiles more visually separable
    wash_strength = {
        "energy": 0.16,
        "clarity": 0.34,
        "connection": 0.18,
        "burden": 0.23,
    }.get(module, 0.20)

    ink_strength = {
        "energy": 0.96,
        "clarity": 0.74,
        "connection": 0.95,
        "burden": 1.16,
    }.get(module, 1.00)

    wash_alpha = mask.filter(
        ImageFilter.GaussianBlur(radius=3 + int(11 * g.boundary_softness))
    )

    wash_alpha = wash_alpha.point(
        lambda p: int(p * (wash_strength + 0.24 * g.boundary_softness))
    )

    visible_color = clip01(color_wash)

    # Broad watercolor stain: very soft, attached to the blot, visible on paper
    paper_stain_alpha = mask.filter(
        ImageFilter.GaussianBlur(radius=14 + int(22 * g.boundary_softness))
    )
    paper_stain_alpha = paper_stain_alpha.point(
        lambda p: int(np.clip(p * (0.08 + 0.24 * visible_color) * visible_color, 0, 76))
    )

    # Local pastel stain: stronger than the paper wash, still softer than ink
    pastel_alpha = mask.filter(
        ImageFilter.GaussianBlur(radius=6 + int(14 * g.boundary_softness))
    )
    pastel_alpha = pastel_alpha.point(
        lambda p: int(np.clip(p * (0.16 + 0.52 * visible_color) * visible_color, 0, 150))
    )

    # Let the grayscale module wash stay, but reduce it slightly when color is on
    # so it does not re-monochromize the image
    wash_alpha = wash_alpha.point(
        lambda p: int(np.clip(p * (1.0 - 0.55 * visible_color), 0, 255))
    )

    ink_alpha = mask.point(
        lambda p: int(np.clip(p * ink_strength * (0.54 + 0.48 * g.darkness), 0, 255))
    )

    pastel_rgb = pastel_rgb_from_grammar(g)

    # Slightly tint the ink itself. Otherwise the
    # final black layer can hide all watercolor underneath
    dark_pastel_rgb = pastel_rgb.astype(float) * 0.42
    ink_color_mix = 0.0 if visible_color <= 0 else (0.22 + 0.62 * visible_color)
    ink_rgb_visible = np.clip(
        (1.0 - ink_color_mix) * ink_rgb
        + ink_color_mix * dark_pastel_rgb,
        0,
        255,
    ).astype(int)

    # Final transparent glaze sits over the ink so the image cannot collapse
    # back to monochrome after the dark layer is composited
    glaze_alpha = mask.filter(ImageFilter.GaussianBlur(radius=1.2)).point(
        lambda p: int(np.clip(p * (0.10 + 0.22 * visible_color) * visible_color, 0, 72))
    )

    paper_stain = Image.new("RGBA", (size, size), (*pastel_rgb.tolist(), 0))
    paper_stain.putalpha(paper_stain_alpha)

    pastel = Image.new("RGBA", (size, size), (*pastel_rgb.tolist(), 0))
    pastel.putalpha(pastel_alpha)

    wash = Image.new("RGBA", (size, size), (*tint.tolist(), 0))
    wash.putalpha(wash_alpha)

    ink = Image.new("RGBA", (size, size), (*ink_rgb_visible.tolist(), 0))
    ink.putalpha(ink_alpha)

    glaze = Image.new("RGBA", (size, size), (*pastel_rgb.tolist(), 0))
    glaze.putalpha(glaze_alpha)

    out = Image.alpha_composite(bg, paper_stain)
    out = Image.alpha_composite(out, wash)
    out = Image.alpha_composite(out, pastel)
    out = Image.alpha_composite(out, ink)
    out = Image.alpha_composite(out, glaze)

    rgb = out.convert("RGB")

    # d = ImageDraw.Draw(rgb)
    # edge = (86, 84, 82)

    # d.rounded_rectangle(
    #     (6, 6, size - 6, size - 6),
    #     radius=int(size * 0.055),
    #     outline=edge,
    #     width=max(1, size // 180),
    # )

    return rgb


__all__ = [
    "LATENT_DIMS",
    "MODULE_TO_DIM",
    "PROMPT_MOTIFS",
    "MOTIF_STYLE",
    "MODULE_STYLE",
    "HEURISTIC_CHANNELS",
    "RorschachGrammar",
    "clip01",
    "stable_seed",
    "vec_from_dict",
    "derive_rorschach_grammar",
    "pastel_rgb_from_grammar",
    "compose_rorschach",
]
