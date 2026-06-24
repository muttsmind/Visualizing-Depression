from __future__ import annotations

"""
Render all images in the pairwise image bank.

Each image is produced from:
    - its seed or learned 4D vector
    - module
    - motif
    - deterministic seed
    - optional render-only color_wash metadata
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Any
import pandas as pd

import numpy as np
from PIL import Image, ImageDraw


LATENT_DIMS = ("depletion", "fog", "disconnection", "burden")


def load_py(path: str, module_name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_pairwise_bank(bank_py: str) -> dict[str, dict[str, Any]]:
    module = load_py(bank_py, "pairwise_bank_render_source")
    bank = getattr(module, "IMAGE_BANK", None)
    if bank is None:
        raise ValueError("Bank must define IMAGE_BANK.")
    return bank


def apply_learned_vectors(
    image_bank: dict[str, dict[str, Any]],
    learned_csv: str | None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Attach learned vectors when available; otherwise render seed vectors."""
    out = {k: dict(v) for k, v in image_bank.items()}
    if not learned_csv:
        return out, "seed_vector"

    learned_path = Path(learned_csv)
    if not learned_path.exists():
        print(f"No learned vector CSV found at {learned_path}; rendering seed vectors.")
        return out, "seed_vector"

    learned = pd.read_csv(learned_path)
    required = ["image_id", *[f"learned_{d}" for d in LATENT_DIMS]]
    missing = [c for c in required if c not in learned.columns]
    if missing:
        raise ValueError(f"learned csv missing columns: {missing}")

    lookup = learned.set_index("image_id")
    for image_id, spec in out.items():
        if image_id not in lookup.index:
            continue
        row = lookup.loc[image_id]
        spec["render_vector"] = {d: float(row[f"learned_{d}"]) for d in LATENT_DIMS}

    return out, "render_vector"


def draw_label(img: Image.Image, image_id: str, family: str) -> Image.Image:
    """Small bottom label for inspection sheets."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    h = out.size[1]
    draw.rounded_rectangle((10, h - 38, out.size[0] - 10, h - 10), radius=10, fill=(246, 244, 240))
    draw.text((18, h - 32), f"{image_id} | {family}", fill=(60, 60, 60))
    return out


def save_sheet(items: list[tuple[str, Image.Image]], outpath: Path, cols: int = 4) -> None:
    if not items:
        return
    cell_w, cell_h = items[0][1].size
    label_h = 0
    rows = int(np.ceil(len(items) / cols))
    pad = 14
    sheet = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * pad, rows * (cell_h + label_h) + (rows + 1) * pad),
        (250, 248, 244),
    )
    for idx, (_, img) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        sheet.paste(img, (x, y))
    sheet.save(outpath)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_py", default="_pairwise_bank.py")
    parser.add_argument("--renderer_py", default="_render_images_rorschach_heuristics.py")
    # "image_bank_mlp/learned_image_vectors.csv" | None
    parser.add_argument("--learned_csv", default="image_bank_mlp/learned_image_vectors.csv")
    # "rendered_mlp" | "rendered_default"
    parser.add_argument("--outdir", default="rendered_mlp")
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument(
        "--color_scale",
        type=float,
        default=1.0,
        help="Multiplier for render-only pastel color_wash metadata. Use 0 for monochrome.",
    )
    parser.add_argument("--no_color", action="store_true", help="Disable pastel watercolor overlays.")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    pairwise_bank = load_pairwise_bank(args.bank_py)
    pairwise_bank, vector_key = apply_learned_vectors(pairwise_bank, args.learned_csv)
    renderer = load_py(args.renderer_py, "rorschach_renderer")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preview_items: list[tuple[str, Image.Image]] = []
    family_items: dict[str, list[tuple[str, Image.Image]]] = {}
    used_color_washes: list[float] = []

    for image_id in sorted(pairwise_bank.keys()):
        spec = pairwise_bank[image_id]
        module = str(spec["module"])
        motif = str(spec["motif"])
        family = str(spec["family"])
        seed = int(spec.get("seed", 0))
        vec = np.array([float(spec[vector_key][d]) for d in LATENT_DIMS], dtype=float)
        color_wash = 0.0 if args.no_color else float(spec.get("color_wash", 0.0)) * float(args.color_scale)
        color_wash = float(np.clip(color_wash, 0.0, 1.0))
        used_color_washes.append(color_wash)

        # Register the image-specific motif so the renderer can use the image id
        # as prompt_id without relying on prompt-option ladders.
        renderer.PROMPT_MOTIFS[image_id] = motif
        img = renderer.compose_rorschach(
            size=int(args.size),
            module=module,
            prompt_id=image_id,
            vec=vec,
            seed=seed,
            color_wash=color_wash,
        )

        if args.label:
            img = draw_label(img, image_id, family)

        fam_dir = outdir / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        img.save(fam_dir / f"{image_id}.png")

        preview_items.append((image_id, img))
        family_items.setdefault(family, []).append((image_id, img))

    save_sheet(preview_items, outdir / "all_images_sheet.jpg", cols=4)
    for family, items in family_items.items():
        save_sheet(items, outdir / f"sheet_{family}.jpg", cols=4)

    if used_color_washes:
        print(
            "Color wash used: "
            f"min={min(used_color_washes):.3f}, "
            f"mean={float(np.mean(used_color_washes)):.3f}, "
            f"max={max(used_color_washes):.3f}."
        )
    print(f"Rendered {len(pairwise_bank)} images to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
