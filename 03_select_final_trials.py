from __future__ import annotations

"""
Select a reduced final visual test from a trained choice-calibrated 4D model.

Selection logic
---------------
1. Load the final trained model.
2. Compute model predictions on the provided pilot data.
3. For each trial, shuffle that trial's chosen/rejected response across
   participants and measure how much performance drops.
4. Select the most useful trials while enforcing rough family coverage:
       anchor, blend, ambiguous, foil

This is an item-reduction helper. For a formal performance estimate =>
do CV or validate the reduced test on new data.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


LATENT_DIMS = ("depletion", "fog", "disconnection", "burden")
FAMILY_GROUPS = ("anchor", "blend", "ambiguous", "foil")
CESD_MAX = 60.0


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


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true.astype(float) - y_pred.astype(float)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "pearson_r": pearsonr_np(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
    }


def load_trained_model(
    train_module,
    model_path: str,
    expert_tensor: torch.Tensor,
    device: torch.device,
):
    """Recreate the trained model from checkpoint metadata."""
    ckpt = torch.load(model_path, map_location=device)
    cfg = train_module.ModelConfig(**ckpt.get("model_config", {}))
    model = train_module.ChoiceCalibrated4DModel(expert_tensor, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    target_transform = ckpt.get("target_transform", {"kind": "norm60"})
    model.target_transform = train_module.TargetTransform(**target_transform)
    model.eval()
    return model


def predict_cesd(model, train_module, chosen_ids: np.ndarray, rejected_ids: np.ndarray, device: torch.device) -> np.ndarray:
    """Return predictions on the original 0..60 CES-D scale."""
    pred, _, _, _, _ = train_module.predict_profiles_and_choice_metrics(
        model, chosen_ids, rejected_ids, device
    )
    return np.clip(pred, 0.0, CESD_MAX)


def pair_group(left_family: str, right_family: str) -> str:
    """
    Assign one exclusive display label to a pair.

    Important: this is not used to enforce quotas. A trial can contain multiple
    families, e.g. anchor+blend. Quotas are enforced with has_anchor, has_blend,
    has_ambiguous, and has_foil columns created in attach_trial_metadata().
    """
    fams = {str(left_family), str(right_family)}
    if fams == {"anchor"}:
        return "anchor"
    if "foil" in fams:
        return "foil"
    if "ambiguous" in fams:
        return "ambiguous"
    if "blend" in fams:
        return "blend"
    return "anchor"


def pair_families_label(left_family: str, right_family: str) -> str:
    fams = sorted({str(left_family), str(right_family)})
    return "+".join(fams)


def attach_trial_metadata(trial_table: pd.DataFrame, image_table: pd.DataFrame) -> pd.DataFrame:
    img = image_table[["image_id", "family", "module"]].copy()
    left = img.rename(columns={
        "image_id": "left_image_id",
        "family": "left_family",
        "module": "left_module",
    })
    right = img.rename(columns={
        "image_id": "right_image_id",
        "family": "right_family",
        "module": "right_module",
    })
    out = trial_table.copy()
    out = out.merge(left, on="left_image_id", how="left")
    out = out.merge(right, on="right_image_id", how="left")
    out["pair_group"] = [
        pair_group(lf, rf) for lf, rf in zip(out["left_family"], out["right_family"])
    ]
    out["pair_families"] = [
        pair_families_label(lf, rf) for lf, rf in zip(out["left_family"], out["right_family"])
    ]
    for family in FAMILY_GROUPS:
        out[f"has_{family}"] = (
            (out["left_family"].astype(str) == family)
            | (out["right_family"].astype(str) == family)
        )
    out["pair_modules"] = out["left_module"].astype(str) + "+" + out["right_module"].astype(str)
    return out


def permutation_importance(
    model,
    train_module,
    data,
    device: torch.device,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Trial importance by shuffling one trial across participants.

    Shuffling keeps the trial's chosen/rejected structure but breaks the link
    between that trial and the participant. If performance drops, the trial was
    useful to the trained model.
    """
    rng = np.random.default_rng(seed)
    y = data.y_cesd.astype(float)
    baseline_pred = predict_cesd(model, train_module, data.chosen_ids, data.rejected_ids, device)
    baseline = regression_metrics(y, baseline_pred)

    rows: list[dict[str, Any]] = []
    n_trials = data.chosen_ids.shape[1]
    n = data.chosen_ids.shape[0]

    for j in range(n_trials):
        perm_r: list[float] = []
        perm_rmse: list[float] = []

        for _ in range(repeats):
            perm = rng.permutation(n)
            c = data.chosen_ids.copy()
            r = data.rejected_ids.copy()
            c[:, j] = data.chosen_ids[perm, j]
            r[:, j] = data.rejected_ids[perm, j]

            pred = predict_cesd(model, train_module, c, r, device)
            m = regression_metrics(y, pred)
            perm_r.append(m["pearson_r"])
            perm_rmse.append(m["rmse"])

        trial_id = str(data.trial_table.iloc[j]["trial_id"])
        mean_perm_r = float(np.mean(perm_r))
        mean_perm_rmse = float(np.mean(perm_rmse))
        rows.append({
            "trial_id": trial_id,
            "trial_index": int(data.trial_table.iloc[j]["trial_index"]),
            "baseline_r": baseline["pearson_r"],
            "mean_permuted_r": mean_perm_r,
            "importance_r_drop": float(baseline["pearson_r"] - mean_perm_r),
            "baseline_rmse": baseline["rmse"],
            "mean_permuted_rmse": mean_perm_rmse,
            "importance_rmse_increase": float(mean_perm_rmse - baseline["rmse"]),
        })

    return pd.DataFrame(rows), baseline


def default_quotas(n_final: int) -> dict[str, int]:
    """Default 20-trial plan: 4 anchor, 8 blend, 6 ambiguous, 2 foil."""
    if n_final == 20:
        return {"anchor": 4, "blend": 8, "ambiguous": 6, "foil": 2}

    props = {"anchor": 0.20, "blend": 0.40, "ambiguous": 0.30, "foil": 0.10}
    quotas = {k: int(np.floor(v * n_final)) for k, v in props.items()}
    while sum(quotas.values()) < n_final:
        # Fill in priority order: blends/ambiguous usually carry most signal.
        for k in ["blend", "ambiguous", "anchor", "foil"]:
            quotas[k] += 1
            if sum(quotas.values()) >= n_final:
                break
    return quotas


def parse_quotas(text: str, n_final: int) -> dict[str, int]:
    if not text:
        return default_quotas(n_final)
    out = {"anchor": 0, "blend": 0, "ambiguous": 0, "foil": 0}
    for part in text.split(","):
        k, v = part.split(":")
        out[k.strip()] = int(v)
    if sum(out.values()) != n_final:
        raise ValueError(f"Quotas sum to {sum(out.values())}, but n_final={n_final}.")
    return out


def select_trials(
    scored_trials: pd.DataFrame,
    n_final: int,
    quotas: dict[str, int],
    max_image_reuse: int,
) -> pd.DataFrame:
    """
    Greedy constrained selection.

    Quotas are enforced by family presence, not by the exclusive pair_group label.
    Example: an anchor+blend trial is eligible for the anchor quota. This avoids
    silently dropping anchors when the trial bank has few/no anchor-anchor pairs.
    """
    candidates = scored_trials.sort_values(
        ["importance_r_drop", "importance_rmse_increase"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    selected_ids: set[str] = set()
    selected_group_counts = {g: 0 for g in FAMILY_GROUPS}
    image_use: dict[str, int] = {}

    def can_take(row: pd.Series, strict_reuse: bool = True) -> bool:
        if str(row["trial_id"]) in selected_ids:
            return False
        if not strict_reuse:
            return True
        left = str(row["left_image_id"])
        right = str(row["right_image_id"])
        return image_use.get(left, 0) < max_image_reuse and image_use.get(right, 0) < max_image_reuse

    def take(row: pd.Series, selection_group: str) -> None:
        row = row.copy()
        row["selection_group"] = selection_group
        selected_rows.append(row)
        selected_ids.add(str(row["trial_id"]))
        if selection_group in selected_group_counts:
            selected_group_counts[selection_group] += 1
        for col in ["left_image_id", "right_image_id"]:
            img = str(row[col])
            image_use[img] = image_use.get(img, 0) + 1

    def eligible_for(row: pd.Series, group: str) -> bool:
        col = f"has_{group}"
        return bool(row[col]) if col in row.index else str(row.get("pair_group", "")) == group

    def best_open_group(row: pd.Series) -> str:
        for group in FAMILY_GROUPS:
            if selected_group_counts.get(group, 0) < quotas.get(group, 0) and eligible_for(row, group):
                return group
        return "extra"

    # Quota pass: first try respecting image reuse.
    for group in FAMILY_GROUPS:
        quota = quotas.get(group, 0)
        group_rows = candidates[candidates.get(f"has_{group}", candidates["pair_group"].eq(group)).astype(bool)]
        for _, row in group_rows.iterrows():
            if selected_group_counts[group] >= quota:
                break
            if can_take(row, strict_reuse=True):
                take(row, group)

    # Quota fallback: if reuse blocked a requested family, relax reuse before
    # filling extra slots. This makes requested coverage more explicit.
    for group in FAMILY_GROUPS:
        quota = quotas.get(group, 0)
        if selected_group_counts[group] >= quota:
            continue
        group_rows = candidates[candidates.get(f"has_{group}", candidates["pair_group"].eq(group)).astype(bool)]
        for _, row in group_rows.iterrows():
            if selected_group_counts[group] >= quota or len(selected_rows) >= n_final:
                break
            if can_take(row, strict_reuse=False):
                take(row, group)

    # Fill remaining slots by importance.
    for _, row in candidates.iterrows():
        if len(selected_rows) >= n_final:
            break
        if can_take(row, strict_reuse=True):
            take(row, best_open_group(row))

    # Final fallback: if image reuse still blocked too much, relax it.
    for _, row in candidates.iterrows():
        if len(selected_rows) >= n_final:
            break
        if can_take(row, strict_reuse=False):
            take(row, best_open_group(row))

    selected = pd.DataFrame(selected_rows)
    if len(selected) < n_final:
        raise ValueError(f"Only selected {len(selected)} trials; requested {n_final}.")

    order = {g: i for i, g in enumerate((*FAMILY_GROUPS, "extra"))}
    selected["_group_order"] = selected["selection_group"].map(order).fillna(99).astype(int)
    selected = selected.sort_values(["_group_order", "importance_r_drop"], ascending=[True, False]).copy()
    selected = selected.drop(columns=["_group_order"])
    selected["final_trial_index"] = np.arange(1, len(selected) + 1)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_py", default="01_train_choice_calibrated_4d.py")
    parser.add_argument("--bank_py", default="_pairwise_bank.py")
    parser.add_argument("--participants_csv", default="data/cesd.csv")
    parser.add_argument("--responses_csv", default="data/visual_test_results.csv")
    parser.add_argument("--model_path", default="image_bank_mlp/final_choice_calibrated_4d_model.pt")
    parser.add_argument("--outdir", default="selected_final_trials")
    parser.add_argument("--n_final", type=int, default=20)
    parser.add_argument("--quotas", default="", help="Example: anchor:4,blend:8,ambiguous:6,foil:2")
    parser.add_argument("--max_image_reuse", type=int, default=2)
    parser.add_argument("--permutation_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device in {"cpu", "cuda"}:
        device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_module = load_py(args.train_py, "pairwise_train_module_for_selection")
    bank = train_module.load_bank(args.bank_py)
    data = train_module.prepare_data(
        participants_csv=args.participants_csv,
        responses_csv=args.responses_csv,
        bank=bank,
        cesd_col="cesd_total",
    )
    expert_tensor = train_module.expert_vectors_tensor(data.image_table, device=device)
    model = load_trained_model(
        train_module,
        args.model_path,
        expert_tensor,
        device=device,
    )

    importance, baseline = permutation_importance(
        model=model,
        train_module=train_module,
        data=data,
        device=device,
        repeats=args.permutation_repeats,
        seed=args.seed,
    )

    trial_meta = attach_trial_metadata(data.trial_table, data.image_table)
    scored = trial_meta.merge(importance, on=["trial_id", "trial_index"], how="left")
    scored = scored.sort_values("importance_r_drop", ascending=False).reset_index(drop=True)
    scored.to_csv(outdir / "trial_importance.csv", index=False)

    quotas = parse_quotas(args.quotas, args.n_final)
    selected = select_trials(
        scored_trials=scored,
        n_final=args.n_final,
        quotas=quotas,
        max_image_reuse=args.max_image_reuse,
    )

    selected_trial_ids = selected["trial_id"].astype(str).tolist()
    selected_trial_bank = data.trial_table[data.trial_table["trial_id"].astype(str).isin(selected_trial_ids)].copy()
    selected_trial_bank = selected_trial_bank.merge(
        selected[[
            "trial_id",
            "final_trial_index",
            "selection_group",
            "pair_group",
            "pair_families",
            "importance_r_drop",
            "importance_rmse_increase",
        ]],
        on="trial_id",
        how="left",
    )
    selected_trial_bank = selected_trial_bank.sort_values("final_trial_index")
    selected_trial_bank.to_csv(outdir / f"selected_{args.n_final}_trial_bank.csv", index=False)

    responses = pd.read_csv(args.responses_csv)
    responses_20 = responses[responses["trial_id"].astype(str).isin(selected_trial_ids)].copy()
    responses_20 = responses_20.merge(
        selected[["trial_id", "final_trial_index"]],
        on="trial_id",
        how="left",
    )
    responses_20 = responses_20.sort_values(["participant_id", "final_trial_index"])
    responses_20.to_csv(outdir / f"visual_test_results_{args.n_final}.csv", index=False)

    selected_counts_by_selection_group = selected["selection_group"].value_counts().to_dict()
    selected_counts_by_pair_group = selected["pair_group"].value_counts().to_dict()
    selected_counts_by_family_presence = {
        group: int(selected.get(f"has_{group}", pd.Series(False, index=selected.index)).sum())
        for group in FAMILY_GROUPS
    }
    available_trials_by_family_presence = {
        group: int(scored.get(f"has_{group}", pd.Series(False, index=scored.index)).sum())
        for group in FAMILY_GROUPS
    }
    unmet_quotas = {
        group: max(0, int(quotas.get(group, 0)) - int(selected_counts_by_selection_group.get(group, 0)))
        for group in FAMILY_GROUPS
    }

    summary = {
        "baseline_full_model_metrics_on_input_data": baseline,
        "n_final": int(args.n_final),
        "quotas_requested": quotas,
        "selected_counts_by_selection_group": selected_counts_by_selection_group,
        "selected_counts_by_pair_group": selected_counts_by_pair_group,
        "selected_counts_by_family_presence": selected_counts_by_family_presence,
        "available_trials_by_family_presence": available_trials_by_family_presence,
        "unmet_quotas": unmet_quotas,
        "mean_selected_importance_r_drop": float(selected["importance_r_drop"].mean()),
        "max_selected_importance_r_drop": float(selected["importance_r_drop"].max()),
        "files": {
            "trial_importance": str(outdir / "trial_importance.csv"),
            "selected_trial_bank": str(outdir / f"selected_{args.n_final}_trial_bank.csv"),
            "filtered_responses": str(outdir / f"visual_test_results_{args.n_final}.csv"),
        },
        "next_step": (
            "..."
        ),
    }
    with open(outdir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved selected trials to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
