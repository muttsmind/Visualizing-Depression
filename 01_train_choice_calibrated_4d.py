from __future__ import annotations

"""
Choice-calibrated 4D visual-profile model for pairwise visual tests.
    Job 1: infer a participant-level 4D visual profile from pairwise choices.
    Job 2: use that inferred profile to predict CES-D.
    Job 3: calibrate image-level 4D vectors so they better explain choices and CES-D,
           while staying close to the expert vectors.

Core statistical idea
---------------------
For participant i and image j:

    theta_i = participant's inferred 4D visual profile
    v_j     = image j's learned/calibrated 4D vector

For a pairwise trial with chosen image c and rejected image r, the model learns:

    P(chosen beats rejected | theta_i, v_c, v_r)

using a pairwise choice decoder. The default decoder says that an image is more
likely to be chosen when its calibrated 4D vector is closer to the participant's
4D profile.

The training objective is:

    total_loss =
        CES-D prediction loss
      + choice_loss_weight * pairwise choice reconstruction loss
      + image_prior_weight * distance(learned image vectors, expert image vectors)
      + nuisance regularization

Why nuisance terms exist
------------------------
People may choose images for reasons unrelated to the clinical/visual 4D space:
visual attractiveness, motif preference, style preference, symmetry, darkness, etc.
If the model has no nuisance channel, it may incorrectly force those effects into
"depletion/fog/disconnection/burden". This script therefore includes:

    image_bias_j          = general image attractiveness/salience
    image_nuisance_j      = learned non-clinical image embedding
    participant_nuisance_i = participant taste/style vector inferred from choices

The choice decoder uses both clinical 4D match and nuisance preference, but the
main exported participant profile remains the interpretable 4D theta_i.

Main outputs
------------
participants_scored_oof.csv
    Cross-validated participant predictions/profiles.

participant_profiles_final.csv
    Final full-data participant predictions/profiles.

learned_image_vectors.csv
    Expert vectors, learned/calibrated vectors, deltas, image bias, nuisance norm.

metrics_choice_calibrated_4d.json
    Regression metrics, choice-reconstruction metrics, config, and diagnostics.

final_choice_calibrated_4d_model.pt
    PyTorch checkpoint with model state and metadata.

Expected inputs
---------------
participants_csv:
    participant_id, cesd_total
or:
    participant_id, cesd_1 ... cesd_20

responses_csv:
    participant_id, trial_id, left_image_id, right_image_id, chosen_side
optional:
    trial_index, prompt_id, prompt_text

bank_py:
    IMAGE_BANK dict where each spec has seed_vector with keys:
    depletion, fog, disconnection, burden
"""

import argparse
import importlib.util
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


LATENT_DIMS = ("depletion", "fog", "disconnection", "burden")
REVERSE_CESD_ITEMS = {4, 8, 12, 16}
CESD_MAX = 60.0


@dataclass
class ModelConfig:
    # ---------------------------------------------------------------------
    # Encoder architecture.
    # Each trial is represented as chosen_4 + rejected_4 + diff_4 = 12D.
    # The same trial encoder is applied to every trial, then trial embeddings
    # are pooled into one participant representation.
    # ---------------------------------------------------------------------
    trial_input_dim: int = 12
    trial_hidden_dim: int = 64
    trial_embed_dim: int = 32
    profile_hidden_dim: int = 32
    profile_layers: int = 1
    pooling: str = "mean_std_max"      # mean | mean_std | mean_std_max
    dropout: float = 0.05
    activation: str = "relu"           # relu | gelu | tanh

    # ---------------------------------------------------------------------
    # Interpretable and nuisance latent spaces.
    # profile_dim should stay 4 !!!
    # nuisance_dim captures taste/style artifacts that should not be forced
    # into depletion/fog/disconnection/burden.
    # ---------------------------------------------------------------------
    profile_dim: int = 4
    nuisance_dim: int = 8
    cesd_uses_nuisance: bool = False    # default: CES-D mainly uses 4D profile
    use_raw_profile_for_cesd: bool = False

    # ---------------------------------------------------------------------
    # Image-vector calibration.
    # learned_vector = clamp(expert_vector + max_image_shift * tanh(delta_raw), 0, 1)
    # image_prior_weight controls how much learned vectors are pulled back
    # toward expert vectors.
    # ---------------------------------------------------------------------
    max_image_shift: float = 0.20
    image_prior_weight: float = 0.02
    freeze_image_vectors: bool = False

    # ---------------------------------------------------------------------
    # Choice decoder.
    # squared_distance is the most interpretable default:
    #   utility(image) = -||theta - image_vector||^2
    # dot_product is more flexible but less directly interpretable.
    # ---------------------------------------------------------------------
    choice_metric: str = "squared_distance"  # squared_distance | dot_product
    init_choice_scale: float = 3.0
    learn_choice_scale: bool = True
    nuisance_logit_weight: float = 1.0
    image_bias_logit_weight: float = 1.0

    # ---------------------------------------------------------------------
    # Loss weights.
    # cesd_loss is always included.
    # choice_loss_weight controls how strongly the 4D profile must explain
    # observed pairwise choices.
    # ---------------------------------------------------------------------
    choice_loss_weight: float = 0.75
    image_bias_l2: float = 1e-4
    image_nuisance_l2: float = 1e-4
    participant_nuisance_l2: float = 1e-4

    # ---------------------------------------------------------------------
    # Training.
    # CES-D is usually z-scored within each training fold.
    # ---------------------------------------------------------------------
    epochs: int = 600
    batch_size: int = 32
    lr: float = 1e-4
    image_lr_multiplier: float = 5.0
    weight_decay: float = 5e-6
    patience: int = 60
    grad_clip: float = 3.0
    loss: str = "mse"                  # mse | huber
    huber_delta: float = 1.0
    target_transform: str = "zscore"   # zscore | norm60
    early_stop_metric: str = "combined" # combined | cesd

    # ---------------------------------------------------------------------
    # Cross-validation.
    # Stratification is useful because CES-D is skewed/floor-heavy.
    # ---------------------------------------------------------------------
    n_splits: int = 5
    n_repeats: int = 3
    stratify_folds: bool = True

    # Optional tail-aware CES-D loss. Usually keep off 
    use_extreme_weighted_loss: bool = False
    extreme_loss_alpha: float = 2.0
    extreme_loss_center: float = 0.50
    normalize_extreme_weights: bool = True


@dataclass
class Bank:
    image_bank: dict[str, dict[str, Any]]
    image_ids: list[str]


@dataclass
class PreparedData:
    participant_ids: np.ndarray
    chosen_ids: np.ndarray
    rejected_ids: np.ndarray
    y_cesd: np.ndarray
    image_table: pd.DataFrame
    trial_table: pd.DataFrame


@dataclass
class TargetTransform:
    kind: str
    mean: float = 0.0
    sd: float = 1.0


# =============================================================================
# Basic utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "pearson_r": pearsonr_np(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "pred_mean": float(y_pred.mean()),
        "pred_sd": float(y_pred.std()),
        "true_mean": float(y_true.mean()),
        "true_sd": float(y_true.std()),
    }


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


def load_bank(bank_py: str) -> Bank:
    module = load_py(bank_py, "choice_calibrated_bank_source")
    bank = getattr(module, "IMAGE_BANK", None)
    if bank is None:
        raise ValueError("bank_py must define IMAGE_BANK.")
    return Bank(image_bank=bank, image_ids=sorted(str(k) for k in bank.keys()))


# =============================================================================
# Data loading and preparation
# =============================================================================


def build_image_table(bank: Bank) -> pd.DataFrame:
    """Create an image table with expert 4D vectors and metadata."""
    rows: list[dict[str, Any]] = []
    for image_id in bank.image_ids:
        spec = bank.image_bank[image_id]
        if "seed_vector" not in spec:
            raise ValueError(f"Image {image_id!r} is missing seed_vector.")
        row = {
            "image_id": image_id,
            "family": spec.get("family", ""),
            "module": spec.get("module", ""),
            "motif": spec.get("motif", ""),
            "note": spec.get("note", ""),
            "seed": int(spec.get("seed", 0)),
        }
        for d in LATENT_DIMS:
            if d not in spec["seed_vector"]:
                raise ValueError(f"Image {image_id!r} seed_vector is missing {d!r}.")
            row[f"expert_{d}"] = float(spec["seed_vector"][d])
        rows.append(row)
    return pd.DataFrame(rows)


def score_cesd_items(df: pd.DataFrame, cesd_col: str) -> pd.DataFrame:
    """Use cesd_total if present; otherwise score cesd_1...cesd_20."""
    if cesd_col in df.columns:
        return df

    item_cols = []
    for i in range(1, 21):
        candidates = [f"cesd_{i}", f"CESD_{i}", f"cesd{i}", f"CESD{i}"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(f"Missing {cesd_col}. To auto-score CES-D, provide item column cesd_{i}.")
        item_cols.append(found)

    scored = df.copy()
    total = np.zeros(len(scored), dtype=float)
    for item_id, col in enumerate(item_cols, start=1):
        raw = scored[col].astype(float)
        bad = sorted(set(raw.dropna().unique()) - {0.0, 1.0, 2.0, 3.0})
        if bad:
            raise ValueError(f"{col} must contain 0,1,2,3 responses. Bad values: {bad}")
        total += 3.0 - raw if item_id in REVERSE_CESD_ITEMS else raw

    scored[cesd_col] = total.astype(float)
    return scored


def load_trial_table(responses: pd.DataFrame) -> pd.DataFrame:
    required = {"trial_id", "left_image_id", "right_image_id"}
    if not required.issubset(responses.columns):
        raise ValueError("responses_csv must include trial_id,left_image_id,right_image_id.")

    trial_cols = ["trial_id", "left_image_id", "right_image_id"]
    for c in ["trial_index", "prompt_id", "prompt_text", "pair_distance"]:
        if c in responses.columns:
            trial_cols.append(c)

    trial_table = responses[trial_cols].drop_duplicates().copy()

    if "trial_index" not in trial_table.columns:
        trial_table["trial_index"] = np.arange(1, len(trial_table) + 1)
    else:
        conflicts = trial_table.groupby("trial_id")["trial_index"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts) > 0:
            raise ValueError(
                "responses_csv has conflicting trial_index values for trial_id(s): "
                f"{conflicts.index.tolist()[:10]}"
            )
        trial_table = trial_table.sort_values("trial_index").drop_duplicates(subset=["trial_id"], keep="first")

    return trial_table.sort_values("trial_index").reset_index(drop=True)


def normalize_responses(raw: pd.DataFrame, trial_table: pd.DataFrame) -> pd.DataFrame:
    """Convert left/right choice format to chosen_image_id and rejected_image_id."""
    out = raw.copy()
    if "chosen_side" in out.columns:
        out["chosen_side"] = out["chosen_side"].astype(str).str.strip().str.lower()
        bad = sorted(set(out["chosen_side"].dropna()) - {"left", "right", "l", "r"})
        if bad:
            raise ValueError(f"Invalid chosen_side values: {bad}")
    elif "chosen_image_id" not in out.columns:
        raise ValueError("responses_csv must contain chosen_side or chosen_image_id.")

    base_cols = ["trial_id", "left_image_id", "right_image_id"]
    if not set(base_cols).issubset(out.columns):
        out = out.merge(trial_table[base_cols], on="trial_id", how="left")

    if "chosen_image_id" not in out.columns:
        out["chosen_image_id"] = np.where(
            out["chosen_side"].isin(["left", "l"]), out["left_image_id"], out["right_image_id"]
        )

    out["rejected_image_id"] = np.where(
        out["chosen_image_id"] == out["left_image_id"], out["right_image_id"], out["left_image_id"]
    )
    return out


def prepare_data(participants_csv: str, responses_csv: str, bank: Bank, cesd_col: str) -> PreparedData:
    image_table = build_image_table(bank)
    image_lookup = {str(r.image_id): int(i) for i, r in image_table.reset_index(drop=True).iterrows()}

    participants = pd.read_csv(participants_csv)
    if "participant_id" not in participants.columns:
        raise ValueError("participants_csv must contain participant_id.")
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = score_cesd_items(participants, cesd_col)

    raw = pd.read_csv(responses_csv)
    if not {"participant_id", "trial_id"}.issubset(raw.columns):
        raise ValueError("responses_csv must contain participant_id and trial_id.")
    raw["participant_id"] = raw["participant_id"].astype(str)

    trial_table = load_trial_table(raw)
    responses = normalize_responses(raw, trial_table)
    responses = responses.drop(columns=["trial_index"], errors="ignore")
    responses = responses.merge(trial_table[["trial_id", "trial_index"]], on="trial_id", how="left")
    responses = responses.dropna(subset=["trial_index", "chosen_image_id", "rejected_image_id"])

    expected_trials = trial_table["trial_id"].tolist()
    n_trials = len(expected_trials)

    # Keep only participants with all expected trials. This keeps the tensor
    # representation simple and ensures each row has the same trial order.
    counts = responses.groupby("participant_id")["trial_id"].nunique()
    keep_ids = counts[counts == n_trials].index.astype(str)

    data = participants[["participant_id", cesd_col]].merge(
        responses[responses["participant_id"].isin(keep_ids)], on="participant_id", how="inner"
    )
    data = data[data["trial_id"].isin(expected_trials)].copy()
    data = data.sort_values(["participant_id", "trial_index"])

    if len(data) == 0:
        raise ValueError("No complete participants left after merging CES-D and pairwise responses.")

    participant_ids = data["participant_id"].drop_duplicates().to_numpy()
    y_lookup = data[["participant_id", cesd_col]].drop_duplicates().set_index("participant_id")[cesd_col]
    y_cesd = y_lookup.loc[participant_ids].to_numpy(dtype=np.float32)

    chosen_ids = np.zeros((len(participant_ids), n_trials), dtype=np.int64)
    rejected_ids = np.zeros((len(participant_ids), n_trials), dtype=np.int64)

    for i, pid in enumerate(participant_ids):
        g = data[data["participant_id"] == pid].sort_values("trial_index")
        if g["trial_id"].tolist() != expected_trials:
            raise ValueError(f"Participant {pid} does not match the fixed trial order.")
        bad = sorted((set(g["chosen_image_id"].astype(str)) | set(g["rejected_image_id"].astype(str))) - set(image_lookup))
        if bad:
            raise ValueError(f"Unknown image ids in responses: {bad[:10]}")
        chosen_ids[i, :] = [image_lookup[str(x)] for x in g["chosen_image_id"].tolist()]
        rejected_ids[i, :] = [image_lookup[str(x)] for x in g["rejected_image_id"].tolist()]

    return PreparedData(
        participant_ids=participant_ids,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
        y_cesd=y_cesd,
        image_table=image_table.reset_index(drop=True),
        trial_table=trial_table,
    )


# =============================================================================
# Baseline diagnostics
# =============================================================================


def ridge_fit_predict_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha: float = 3.0,
) -> np.ndarray:
    """Fit ridge on one training fold and predict one validation fold.

    Feature scaling and target centering are fitted on the training fold only.
    This keeps the baseline leakage-free and directly comparable to the neural
    model's out-of-fold eval.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    x_train = X[train_idx]
    y_train = y[train_idx]
    x_val = X[val_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_sd = x_train.std(axis=0, keepdims=True)
    x_sd[x_sd < 1e-8] = 1.0

    y_mean = float(y_train.mean())
    y_center = y_train - y_mean
    x_train_z = (x_train - x_mean) / x_sd
    x_val_z = (x_val - x_mean) / x_sd

    xtx = x_train_z.T @ x_train_z
    beta = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), x_train_z.T @ y_center)
    return x_val_z @ beta + y_mean


def ridge_cv_predict(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5, alpha: float = 3.0) -> np.ndarray:
    """Single-pass ridge CV baseline kept for quick diagnostics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_splits = max(2, min(n_splits, n))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_splits)
    pred = np.zeros(n, dtype=float)

    for val_idx in folds:
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
        pred[val_idx] = ridge_fit_predict_fold(X, y, train_idx, val_idx, alpha=alpha)
    return pred


def ridge_repeated_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 3,
    alpha: float = 3.0,
    stratify: bool = True,
) -> np.ndarray:
    """Repeated out-of-fold ridge predictions using the same fold logic as the NN.

    Each repeat creates a fresh fold split using make_folds(...). Predictions are
    accumulated for every participant and averaged across repeats, matching the
    NN model's repeated-CV reporting structure. Alpha is fixed => no
    nested hyperparameter tuning.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_repeats = max(1, int(n_repeats))

    pred_sum = np.zeros(n, dtype=float)
    pred_count = np.zeros(n, dtype=float)
    all_indices = np.arange(n)

    for rep in range(n_repeats):
        folds = make_folds(y, n_splits, seed + 1000 * rep, stratify)
        for val_idx in folds:
            train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
            pred_sum[val_idx] += ridge_fit_predict_fold(X, y, train_idx, val_idx, alpha=alpha)
            pred_count[val_idx] += 1.0

    return pred_sum / np.maximum(pred_count, 1.0)


def choice_summary_features(data: PreparedData) -> tuple[np.ndarray, list[str]]:
    """Simple baseline features: mean chosen vector, mean rejected vector, mean diff."""
    expert = data.image_table[[f"expert_{d}" for d in LATENT_DIMS]].to_numpy(dtype=float)
    chosen = expert[data.chosen_ids]      # n x trials x 4
    rejected = expert[data.rejected_ids]  # n x trials x 4
    diff = chosen - rejected

    parts = [chosen.mean(axis=1), rejected.mean(axis=1), diff.mean(axis=1), chosen.std(axis=1)]
    names: list[str] = []
    for prefix in ["mean_chosen", "mean_rejected", "mean_diff", "std_chosen"]:
        for d in LATENT_DIMS:
            names.append(f"{prefix}_{d}")
    return np.concatenate(parts, axis=1), names


# =============================================================================
# Model components
# =============================================================================


def activation_layer(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float, activation: str) -> nn.Sequential:
    """Small helper for profile/nuisance heads."""
    layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
    cur = in_dim
    for _ in range(max(1, n_layers)):
        layers.extend([
            nn.Linear(cur, hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        ])
        cur = hidden_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class ChoiceCalibrated4DModel(nn.Module):
    """
    Pairwise-choice measurement model with calibrated image vectors.

    High-level flow
    ---------------
    1. Learned image vectors start at expert vectors and can shift slightly.
    2. Participant choices are encoded into a pooled representation.
    3. The pooled representation is mapped to:
         - profile4: interpretable depletion/fog/disconnection/burden profile
         - nuisance: taste/style vector
    4. profile4 predicts CES-D.
    5. profile4 + nuisance reconstruct observed pairwise choices.

    The choice decoder makes the 4D profile do two jobs at once:
    predicting CES-D and explaining pairwise choices. This is the main guardrail
    against learning a CES-D predictor that no longer corresponds to the visual
    bank.
    """

    def __init__(self, expert_image_vectors: torch.Tensor, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("expert_image_vectors", expert_image_vectors)
        self.n_images = int(expert_image_vectors.shape[0])

        if config.profile_dim != 4:
            raise ValueError("This project expects profile_dim=4 for depletion/fog/disconnection/burden.")
        if config.pooling not in {"mean", "mean_std", "mean_std_max"}:
            raise ValueError(f"Unknown pooling: {config.pooling}")
        if config.choice_metric not in {"squared_distance", "dot_product"}:
            raise ValueError(f"Unknown choice_metric: {config.choice_metric}")

        # Clinical image-vector calibration parameters.
        # These are the parameters that produce final improved/calibrated 4D vectors.
        self.image_delta_raw = nn.Parameter(torch.zeros(self.n_images, 4))
        if config.freeze_image_vectors:
            self.image_delta_raw.requires_grad_(False)

        # Nuisance choice terms. These absorb general image salience and taste/style
        # effects so that the clinical 4D vectors do not have to explain everything.
        self.image_bias = nn.Parameter(torch.zeros(self.n_images))
        if config.nuisance_dim > 0:
            self.image_nuisance = nn.Parameter(torch.zeros(self.n_images, config.nuisance_dim))
        else:
            self.register_parameter("image_nuisance", None)

        # A positive scale for choice logits. Larger scale means more deterministic choices.
        init_scale = max(float(config.init_choice_scale), 1e-4)
        self.log_choice_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        if not config.learn_choice_scale:
            self.log_choice_scale.requires_grad_(False)

        self.trial_encoder = nn.Sequential(
            nn.LayerNorm(config.trial_input_dim),
            nn.Linear(config.trial_input_dim, config.trial_hidden_dim),
            activation_layer(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.trial_hidden_dim, config.trial_embed_dim),
            activation_layer(config.activation),
        )

        n_pool = {"mean": 1, "mean_std": 2, "mean_std_max": 3}[config.pooling]
        pooled_dim = config.trial_embed_dim * n_pool

        self.profile_head = make_mlp(
            in_dim=pooled_dim,
            hidden_dim=config.profile_hidden_dim,
            out_dim=4,
            n_layers=config.profile_layers,
            dropout=config.dropout,
            activation=config.activation,
        )

        if config.nuisance_dim > 0:
            self.nuisance_head = make_mlp(
                in_dim=pooled_dim,
                hidden_dim=config.profile_hidden_dim,
                out_dim=config.nuisance_dim,
                n_layers=config.profile_layers,
                dropout=config.dropout,
                activation=config.activation,
            )
        else:
            self.nuisance_head = None

        cesd_in_dim = 4
        if config.cesd_uses_nuisance:
            cesd_in_dim += config.nuisance_dim
        self.cesd_head = nn.Linear(cesd_in_dim, 1)

    def image_vectors(self) -> torch.Tensor:
        """Calibrated 4D vectors, initialized from and regularized toward expert vectors."""
        delta = self.config.max_image_shift * torch.tanh(self.image_delta_raw)
        return torch.clamp(self.expert_image_vectors + delta, 0.0, 1.0)

    def choice_scale(self) -> torch.Tensor:
        # Clamp prevents numerical explosions if scale is learned aggressively.
        return torch.exp(self.log_choice_scale).clamp(0.05, 50.0)

    def trial_features(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> torch.Tensor:
        """Build chosen/rejected/difference features for each trial."""
        vectors = self.image_vectors()
        chosen = vectors[chosen_ids]
        rejected = vectors[rejected_ids]
        diff = chosen - rejected
        return torch.cat([chosen, rejected, diff], dim=-1)  # batch x trials x 12

    def pooled_trial_embedding(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> torch.Tensor:
        x = self.trial_features(chosen_ids, rejected_ids)
        h = self.trial_encoder(x)  # batch x trials x embed_dim
        parts = [h.mean(dim=1)]
        if self.config.pooling in {"mean_std", "mean_std_max"}:
            parts.append(h.std(dim=1, unbiased=False))
        if self.config.pooling == "mean_std_max":
            parts.append(h.max(dim=1).values)
        return torch.cat(parts, dim=-1)

    def participant_state(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer participant state from all choices.

        Returns
        -------
        profile4:
            Bounded 0..1 interpretable profile.
        raw_profile4:
            Unbounded logits before sigmoid; useful for diagnostics.
        nuisance:
            Participant taste/style vector.
        """
        pooled = self.pooled_trial_embedding(chosen_ids, rejected_ids)
        raw_profile4 = self.profile_head(pooled)
        profile4 = torch.sigmoid(raw_profile4)
        if self.nuisance_head is None:
            nuisance = raw_profile4.new_zeros((raw_profile4.shape[0], 0))
        else:
            nuisance = self.nuisance_head(pooled)
        return profile4, raw_profile4, nuisance

    def clinical_choice_match(self, profile4: torch.Tensor, image_vecs: torch.Tensor) -> torch.Tensor:
        """Clinical utility of images under the inferred 4D profile."""
        theta = profile4[:, None, :]  # batch x 1 x 4
        if self.config.choice_metric == "squared_distance":
            # Higher is better: closer image vectors get higher utility.
            return -torch.sum((theta - image_vecs) ** 2, dim=-1)
        # More flexible but less metric-like.
        return torch.sum(theta * image_vecs, dim=-1)

    def choice_logits_from_state(
        self,
        profile4: torch.Tensor,
        nuisance: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict whether the observed chosen image should beat the rejected image.

        Because the tensors are already ordered as chosen/rejected, the target is
        always 1.0. A positive logit means the model thinks chosen > rejected.
        """
        vectors = self.image_vectors()
        chosen_vecs = vectors[chosen_ids]
        rejected_vecs = vectors[rejected_ids]

        chosen_match = self.clinical_choice_match(profile4, chosen_vecs)
        rejected_match = self.clinical_choice_match(profile4, rejected_vecs)
        clinical_logits = self.choice_scale() * (chosen_match - rejected_match)

        # General image bias: some images may be more often chosen independent
        # of clinical meaning.
        bias_logits = self.config.image_bias_logit_weight * (
            self.image_bias[chosen_ids] - self.image_bias[rejected_ids]
        )

        # Participant-specific nuisance/taste interaction with image nuisance embeddings.
        if self.image_nuisance is not None and nuisance.shape[1] > 0:
            chosen_nuisance = self.image_nuisance[chosen_ids]
            rejected_nuisance = self.image_nuisance[rejected_ids]
            nuisance_logits = self.config.nuisance_logit_weight * torch.sum(
                nuisance[:, None, :] * (chosen_nuisance - rejected_nuisance),
                dim=-1,
            )
        else:
            nuisance_logits = clinical_logits.new_zeros(clinical_logits.shape)

        return clinical_logits + bias_logits + nuisance_logits

    def cesd_prediction_from_state(self, profile4: torch.Tensor, raw_profile4: torch.Tensor, nuisance: torch.Tensor) -> torch.Tensor:
        if self.config.use_raw_profile_for_cesd:
            cesd_features = raw_profile4
        else:
            cesd_features = profile4
        if self.config.cesd_uses_nuisance and nuisance.shape[1] > 0:
            cesd_features = torch.cat([cesd_features, nuisance], dim=-1)
        return self.cesd_head(cesd_features).squeeze(-1)

    def forward(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        profile4, raw_profile4, nuisance = self.participant_state(chosen_ids, rejected_ids)
        pred = self.cesd_prediction_from_state(profile4, raw_profile4, nuisance)
        return pred, profile4, raw_profile4, nuisance


# =============================================================================
# Target transforms and losses
# =============================================================================


def expert_vectors_tensor(image_table: pd.DataFrame, device: torch.device) -> torch.Tensor:
    arr = image_table[[f"expert_{d}" for d in LATENT_DIMS]].to_numpy(dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def make_target_transform(y: np.ndarray, train_idx: np.ndarray, kind: str) -> TargetTransform:
    if kind == "norm60":
        return TargetTransform(kind="norm60")
    if kind == "zscore":
        mean = float(np.mean(y[train_idx]))
        sd = float(np.std(y[train_idx]))
        return TargetTransform(kind="zscore", mean=mean, sd=max(sd, 1e-6))
    raise ValueError(f"Unknown target_transform: {kind}")


def transform_y(y: np.ndarray, tfm: TargetTransform) -> np.ndarray:
    if tfm.kind == "norm60":
        return np.clip(y / CESD_MAX, 0.0, 1.0).astype(np.float32)
    return ((y - tfm.mean) / tfm.sd).astype(np.float32)


def inverse_y(y_model: np.ndarray, tfm: TargetTransform) -> np.ndarray:
    if tfm.kind == "norm60":
        return y_model * CESD_MAX
    return y_model * tfm.sd + tfm.mean


def cesd_prediction_loss(pred: torch.Tensor, target: torch.Tensor, y_norm_for_weights: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    if cfg.loss == "mse":
        per_item = (pred - target) ** 2
    elif cfg.loss == "huber":
        per_item = F.huber_loss(pred, target, delta=cfg.huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    if cfg.use_extreme_weighted_loss:
        weights = 1.0 + cfg.extreme_loss_alpha * torch.abs(y_norm_for_weights - cfg.extreme_loss_center)
        if cfg.normalize_extreme_weights:
            weights = weights / weights.mean().clamp_min(1e-12)
        per_item = per_item * weights
    return per_item.mean()


def regularization_loss(model: ChoiceCalibrated4DModel, nuisance_batch: torch.Tensor | None = None) -> torch.Tensor:
    cfg = model.config
    reg = model.expert_image_vectors.new_tensor(0.0)

    # Expert-vector prior: this is what keeps calibrated image vectors interpretable.
    if cfg.image_prior_weight > 0:
        reg = reg + cfg.image_prior_weight * F.mse_loss(model.image_vectors(), model.expert_image_vectors)

    # Keep general image attractiveness and nuisance embeddings modest unless
    # they truly improve choice reconstruction.
    if cfg.image_bias_l2 > 0:
        reg = reg + cfg.image_bias_l2 * torch.mean(model.image_bias ** 2)
    if model.image_nuisance is not None and cfg.image_nuisance_l2 > 0:
        reg = reg + cfg.image_nuisance_l2 * torch.mean(model.image_nuisance ** 2)
    if nuisance_batch is not None and nuisance_batch.numel() > 0 and cfg.participant_nuisance_l2 > 0:
        reg = reg + cfg.participant_nuisance_l2 * torch.mean(nuisance_batch ** 2)
    return reg


def combined_batch_loss(
    model: ChoiceCalibrated4DModel,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    y_target: torch.Tensor,
    y_norm: torch.Tensor,
    cfg: ModelConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred, profile4, raw_profile4, nuisance = model(chosen, rejected)

    # Supervised signal: profile-derived prediction should match CES-D.
    cesd_loss = cesd_prediction_loss(pred, y_target, y_norm, cfg)

    # Measurement signal: the same inferred profile should make chosen images
    # beat rejected images in the pairwise task. Because inputs are ordered as
    # chosen/rejected, every target label == 1.
    choice_logits = model.choice_logits_from_state(profile4, nuisance, chosen, rejected)
    choice_targets = torch.ones_like(choice_logits)
    choice_loss = F.binary_cross_entropy_with_logits(choice_logits, choice_targets)

    reg = regularization_loss(model, nuisance)
    total = cesd_loss + cfg.choice_loss_weight * choice_loss + reg

    with torch.no_grad():
        choice_acc = (choice_logits > 0).float().mean().item()
        stats = {
            "total_loss": float(total.detach().cpu()),
            "cesd_loss": float(cesd_loss.detach().cpu()),
            "choice_loss": float(choice_loss.detach().cpu()),
            "choice_acc": float(choice_acc),
            "reg_loss": float(reg.detach().cpu()),
            "choice_scale": float(model.choice_scale().detach().cpu()),
        }
    return total, stats


def make_optimizer(model: ChoiceCalibrated4DModel, cfg: ModelConfig) -> torch.optim.Optimizer:
    """Use a separate LR multiplier for calibrated image-vector parameters."""
    image_params = []
    if model.image_delta_raw.requires_grad:
        image_params.append(model.image_delta_raw)
    image_ids = {id(p) for p in image_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in image_ids]

    groups = [{"params": other_params, "lr": cfg.lr}]
    if image_params:
        groups.append({"params": image_params, "lr": cfg.lr * cfg.image_lr_multiplier})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


# =============================================================================
# Cross-validation helpers
# =============================================================================


def make_folds(y: np.ndarray, n_splits: int, seed: int, stratify: bool) -> list[np.ndarray]:
    """Create folds; optionally stratify by CES-D quantile bins."""
    n = len(y)
    n_splits = min(max(2, n_splits), n)
    rng = np.random.default_rng(seed)

    if not stratify:
        indices = rng.permutation(n)
        return [fold.astype(int) for fold in np.array_split(indices, n_splits)]

    # Quantile bins reduce the chance that a fold gets too many low or high CES-D participants.
    try:
        bins = pd.qcut(y, q=min(5, n_splits), labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(n, dtype=int)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for b in np.unique(bins):
        idx = np.where(np.asarray(bins) == b)[0]
        idx = rng.permutation(idx)
        for k, part in enumerate(np.array_split(idx, n_splits)):
            folds[k].extend(part.tolist())

    return [np.array(rng.permutation(fold), dtype=int) for fold in folds]


def fit_model(
    chosen_ids: np.ndarray,
    rejected_ids: np.ndarray,
    y_cesd: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    expert_image_vectors: torch.Tensor,
    device: torch.device,
    cfg: ModelConfig,
    max_epochs: int | None = None,
) -> ChoiceCalibrated4DModel:
    # Fit target scaling on the training fold only to avoid leakage. The model
    # predicts transformed CES-D during optimization; outputs are converted back
    # to the 0..60 CES-D scale
    target_tfm = make_target_transform(y_cesd, train_idx, cfg.target_transform)
    y_target_np = transform_y(y_cesd, target_tfm)
    y_norm_np = np.clip(y_cesd / CESD_MAX, 0.0, 1.0).astype(np.float32)

    model = ChoiceCalibrated4DModel(expert_image_vectors, cfg).to(device)
    optimizer = make_optimizer(model, cfg)

    x_chosen = torch.tensor(chosen_ids, dtype=torch.long, device=device)
    x_rejected = torch.tensor(rejected_ids, dtype=torch.long, device=device)
    y_target = torch.tensor(y_target_np, dtype=torch.float32, device=device)
    y_norm = torch.tensor(y_norm_np, dtype=torch.float32, device=device)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device) if val_idx is not None else None

    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    best_epoch = -1
    final_epoch = -1
    patience_left = cfg.patience
    early_stopped = False
    epochs_to_run = int(max_epochs if max_epochs is not None else cfg.epochs)

    for epoch in range(epochs_to_run):
        final_epoch = epoch
        model.train()
        perm = train_t[torch.randperm(len(train_t), device=device)]
        for start in range(0, len(perm), cfg.batch_size):
            batch = perm[start:start + cfg.batch_size]
            loss, _stats = combined_batch_loss(
                model,
                x_chosen[batch],
                x_rejected[batch],
                y_target[batch],
                y_norm[batch],
                cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if val_t is None:
            continue

        model.eval()
        with torch.no_grad():
            _total, stats = combined_batch_loss(
                model,
                x_chosen[val_t],
                x_rejected[val_t],
                y_target[val_t],
                y_norm[val_t],
                cfg,
            )
            val_metric = stats["cesd_loss"] if cfg.early_stop_metric == "cesd" else stats["total_loss"]

        if val_metric < best_val - 1e-6:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_stopped = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.target_transform = target_tfm
    model.training_summary = {
        "best_epoch": int(best_epoch),
        "final_epoch": int(final_epoch),
        "early_stopped": bool(early_stopped),
        "patience": int(cfg.patience),
        "target_transform": asdict(target_tfm),
        "early_stop_metric": cfg.early_stop_metric,
    }
    return model


def predict_profiles_and_choice_metrics(
    model: ChoiceCalibrated4DModel,
    chosen_ids: np.ndarray,
    rejected_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    tfm: TargetTransform = model.target_transform
    x_chosen = torch.tensor(chosen_ids, dtype=torch.long, device=device)
    x_rejected = torch.tensor(rejected_ids, dtype=torch.long, device=device)

    pred_parts: list[np.ndarray] = []
    profile_parts: list[np.ndarray] = []
    raw_profile_parts: list[np.ndarray] = []
    nuisance_parts: list[np.ndarray] = []

    choice_loss_sum = 0.0
    choice_correct = 0.0
    choice_count = 0
    logit_sum = 0.0
    logit_sq_sum = 0.0

    with torch.no_grad():
        for start in range(0, len(x_chosen), batch_size):
            c = x_chosen[start:start + batch_size]
            r = x_rejected[start:start + batch_size]
            pred_model, profile4, raw_profile4, nuisance = model(c, r)
            pred_cesd = inverse_y(pred_model.detach().cpu().numpy(), tfm)
            pred_parts.append(pred_cesd)
            profile_parts.append(profile4.detach().cpu().numpy())
            raw_profile_parts.append(raw_profile4.detach().cpu().numpy())
            nuisance_parts.append(nuisance.detach().cpu().numpy())

            logits = model.choice_logits_from_state(profile4, nuisance, c, r)
            targets = torch.ones_like(logits)
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
            choice_loss_sum += float(bce)
            choice_correct += float((logits > 0).float().sum().item())
            choice_count += int(logits.numel())
            logit_sum += float(logits.sum().item())
            logit_sq_sum += float((logits ** 2).sum().item())

    logits_mean = logit_sum / max(choice_count, 1)
    logits_var = max(logit_sq_sum / max(choice_count, 1) - logits_mean ** 2, 0.0)
    choice_metrics = {
        "choice_bce": choice_loss_sum / max(choice_count, 1),
        "choice_accuracy": choice_correct / max(choice_count, 1),
        "choice_logit_mean": logits_mean,
        "choice_logit_sd": float(math.sqrt(logits_var)),
        "choice_count": int(choice_count),
        "choice_scale": float(model.choice_scale().detach().cpu()),
    }

    nuisance_arr = np.vstack(nuisance_parts) if nuisance_parts and nuisance_parts[0].shape[1] > 0 else np.zeros((len(x_chosen), 0))
    return (
        np.clip(np.concatenate(pred_parts), 0.0, CESD_MAX),
        np.vstack(profile_parts),
        np.vstack(raw_profile_parts),
        nuisance_arr,
        choice_metrics,
    )


def cross_validate(data: PreparedData, expert_image_vectors: torch.Tensor, seed: int, device: torch.device, cfg: ModelConfig):
    n = len(data.y_cesd)
    pred_sum = np.zeros(n, dtype=float)
    profile_sum = np.zeros((n, 4), dtype=float)
    raw_profile_sum = np.zeros((n, 4), dtype=float)
    nuisance_sum = np.zeros((n, cfg.nuisance_dim), dtype=float) if cfg.nuisance_dim > 0 else np.zeros((n, 0), dtype=float)
    pred_count = np.zeros(n, dtype=float)
    best_epochs: list[int] = []

    choice_metric_accumulator = {
        "choice_loss_sum": 0.0,
        "choice_correct_sum": 0.0,
        "choice_count": 0,
        "logit_sum": 0.0,
        "logit_sq_sum": 0.0,
    }

    for rep in range(cfg.n_repeats):
        folds = make_folds(data.y_cesd, cfg.n_splits, seed + 1000 * rep, cfg.stratify_folds)
        all_indices = np.arange(n)
        for fold_id, val_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
            model = fit_model(
                chosen_ids=data.chosen_ids,
                rejected_ids=data.rejected_ids,
                y_cesd=data.y_cesd,
                train_idx=np.sort(train_idx),
                val_idx=np.sort(val_idx),
                expert_image_vectors=expert_image_vectors,
                device=device,
                cfg=cfg,
            )
            best_epochs.append(int(model.training_summary["best_epoch"]))

            pred_cesd, profile4, raw_profile4, nuisance, choice_metrics = predict_profiles_and_choice_metrics(
                model, data.chosen_ids[val_idx], data.rejected_ids[val_idx], device
            )
            pred_sum[val_idx] += pred_cesd
            profile_sum[val_idx] += profile4
            raw_profile_sum[val_idx] += raw_profile4
            if cfg.nuisance_dim > 0:
                nuisance_sum[val_idx] += nuisance
            pred_count[val_idx] += 1

            # Reconstruct global OOF choice metrics from per-fold means.
            count = choice_metrics["choice_count"]
            choice_metric_accumulator["choice_loss_sum"] += choice_metrics["choice_bce"] * count
            choice_metric_accumulator["choice_correct_sum"] += choice_metrics["choice_accuracy"] * count
            choice_metric_accumulator["choice_count"] += count
            choice_metric_accumulator["logit_sum"] += choice_metrics["choice_logit_mean"] * count
            choice_metric_accumulator["logit_sq_sum"] += (choice_metrics["choice_logit_sd"] ** 2 + choice_metrics["choice_logit_mean"] ** 2) * count

            print(
                f"CV rep {rep + 1}/{cfg.n_repeats}, fold {fold_id}/{len(folds)} done | "
                f"best_epoch={model.training_summary['best_epoch']} | "
                f"val_choice_acc={choice_metrics['choice_accuracy']:.3f}",
                flush=True,
            )

    pred_oof = np.clip(pred_sum / np.maximum(pred_count, 1.0), 0.0, CESD_MAX)
    profile_oof = profile_sum / np.maximum(pred_count[:, None], 1.0)
    raw_profile_oof = raw_profile_sum / np.maximum(pred_count[:, None], 1.0)
    nuisance_oof = nuisance_sum / np.maximum(pred_count[:, None], 1.0) if cfg.nuisance_dim > 0 else nuisance_sum

    c = max(int(choice_metric_accumulator["choice_count"]), 1)
    logit_mean = choice_metric_accumulator["logit_sum"] / c
    logit_var = max(choice_metric_accumulator["logit_sq_sum"] / c - logit_mean ** 2, 0.0)
    choice_oof_metrics = {
        "choice_bce": choice_metric_accumulator["choice_loss_sum"] / c,
        "choice_accuracy": choice_metric_accumulator["choice_correct_sum"] / c,
        "choice_logit_mean": logit_mean,
        "choice_logit_sd": float(math.sqrt(logit_var)),
        "choice_count": c,
    }
    return pred_oof, profile_oof, raw_profile_oof, nuisance_oof, choice_oof_metrics, best_epochs


# =============================================================================
# Output helpers
# =============================================================================


def profiles_dataframe(
    participant_ids: np.ndarray,
    y_cesd: np.ndarray,
    pred_cesd: np.ndarray,
    profile4: np.ndarray,
    raw_profile4: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "participant_id": participant_ids,
        "cesd_total": y_cesd,
        "predicted_cesd": pred_cesd,
        "residual": y_cesd - pred_cesd,
    })
    for i, dim in enumerate(LATENT_DIMS):
        out[f"profile_{dim}"] = profile4[:, i]
    out["profile_mean"] = profile4.mean(axis=1)
    out["profile_max"] = profile4.max(axis=1)
    out["profile_dominant_dim"] = [LATENT_DIMS[int(i)] for i in np.argmax(profile4, axis=1)]

    if raw_profile4 is not None:
        for i, dim in enumerate(LATENT_DIMS):
            out[f"profile_raw_{dim}"] = raw_profile4[:, i]

    if nuisance is not None and nuisance.shape[1] > 0:
        for j in range(nuisance.shape[1]):
            out[f"nuisance_{j + 1}"] = nuisance[:, j]
        out["nuisance_l2"] = np.sqrt(np.sum(nuisance ** 2, axis=1))
    return out


def export_learned_image_vectors(model: ChoiceCalibrated4DModel, image_table: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    """Export the calibrated image vectors: this is the main image-level result."""
    model.eval()
    with torch.no_grad():
        learned = model.image_vectors().detach().cpu().numpy()
        image_bias = model.image_bias.detach().cpu().numpy()
        if model.image_nuisance is not None:
            image_nuisance = model.image_nuisance.detach().cpu().numpy()
            image_nuisance_l2 = np.sqrt(np.sum(image_nuisance ** 2, axis=1))
        else:
            image_nuisance_l2 = np.zeros(len(image_table), dtype=float)

    out = image_table.copy()
    for i, dim in enumerate(LATENT_DIMS):
        out[f"learned_{dim}"] = learned[:, i]
        out[f"delta_{dim}"] = out[f"learned_{dim}"] - out[f"expert_{dim}"]
    delta_cols = [f"delta_{d}" for d in LATENT_DIMS]
    out["visual_shift_l2"] = np.sqrt(np.sum(out[delta_cols].to_numpy(dtype=float) ** 2, axis=1))
    out["image_bias"] = image_bias
    out["image_nuisance_l2"] = image_nuisance_l2
    out.to_csv(outpath, index=False)
    return out


def calibrated_bank_snippet(learned_images: pd.DataFrame, outpath: Path) -> None:
    """
    Write a simple CSV-like Python snippet with learned 4D vectors.

    This is intentionally not a full replacement for your bank file because your
    bank may contain motifs, seeds, paths, and rendering metadata. Use the CSV or
    this snippet to copy calibrated values back into your bank if desired.
    """
    lines = [
        "# Learned/calibrated 4D vectors exported by 01_train_choice_calibrated_4d.py",
        "CALIBRATED_IMAGE_VECTORS = {",
    ]
    for r in learned_images.itertuples(index=False):
        vec = {d: float(getattr(r, f"learned_{d}")) for d in LATENT_DIMS}
        lines.append(f"    {r.image_id!r}: {vec!r},")
    lines.append("}")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bank_py", default="_pairwise_bank.py")
    parser.add_argument("--participants_csv", default="data/cesd.csv")
    parser.add_argument("--responses_csv", default="data/visual_test_results.csv")
    parser.add_argument("--outdir", default="image_bank_mlp")
    parser.add_argument("--cesd_col", default="cesd_total")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    # Encoder architecture controls
    parser.add_argument("--trial_hidden_dim", type=int, default=32)
    parser.add_argument("--trial_embed_dim", type=int, default=32)
    parser.add_argument("--profile_hidden_dim", type=int, default=16)
    parser.add_argument("--profile_layers", type=int, default=1)
    parser.add_argument("--pooling", default="mean_std", choices=("mean", "mean_std", "mean_std_max"))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--activation", default="relu", choices=("relu", "gelu", "tanh"))

    # Latent design controls
    parser.add_argument("--nuisance_dim", type=int, default=4)
    parser.add_argument("--cesd_uses_nuisance", type=int, default=0)
    parser.add_argument("--use_raw_profile_for_cesd", type=int, default=1)

    # Image-vector calibration controls
    parser.add_argument("--max_image_shift", type=float, default=0.7)
    parser.add_argument("--image_prior_weight", type=float, default=0.035)
    parser.add_argument("--freeze_image_vectors", type=int, default=0)
    parser.add_argument("--image_lr_multiplier", type=float, default=5)  # 3

    # Choice decoder controls
    parser.add_argument("--choice_metric", default="squared_distance", choices=("squared_distance", "dot_product"))
    parser.add_argument("--init_choice_scale", type=float, default=3.0)
    parser.add_argument("--learn_choice_scale", type=int, default=1)
    parser.add_argument("--nuisance_logit_weight", type=float, default=1.0)
    parser.add_argument("--image_bias_logit_weight", type=float, default=0.0)

    # Loss controls
    parser.add_argument("--choice_loss_weight", type=float, default=0.5)  # 0.35
    parser.add_argument("--image_bias_l2", type=float, default=1e-4)
    parser.add_argument("--image_nuisance_l2", type=float, default=1e-4)
    parser.add_argument("--participant_nuisance_l2", type=float, default=1e-4)

    # Training controls
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--loss", default="mse", choices=("mse", "huber"))
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--target_transform", default="zscore", choices=("zscore", "norm60"))
    parser.add_argument("--early_stop_metric", default="cesd", choices=("combined", "cesd"))

    # CV controls
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--stratify_folds", type=int, default=1)

    # Tail-weighted CES-D loss controls
    parser.add_argument("--use_extreme_weighted_loss", type=int, default=1)
    parser.add_argument("--extreme_loss_alpha", type=float, default=2.0)
    parser.add_argument("--extreme_loss_center", type=float, default=0.50)
    parser.add_argument("--normalize_extreme_weights", type=int, default=1)


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        trial_hidden_dim=args.trial_hidden_dim,
        trial_embed_dim=args.trial_embed_dim,
        profile_hidden_dim=args.profile_hidden_dim,
        profile_layers=args.profile_layers,
        pooling=args.pooling,
        dropout=args.dropout,
        activation=args.activation,
        nuisance_dim=args.nuisance_dim,
        cesd_uses_nuisance=bool(args.cesd_uses_nuisance),
        use_raw_profile_for_cesd=bool(args.use_raw_profile_for_cesd),
        max_image_shift=args.max_image_shift,
        image_prior_weight=args.image_prior_weight,
        freeze_image_vectors=bool(args.freeze_image_vectors),
        choice_metric=args.choice_metric,
        init_choice_scale=args.init_choice_scale,
        learn_choice_scale=bool(args.learn_choice_scale),
        nuisance_logit_weight=args.nuisance_logit_weight,
        image_bias_logit_weight=args.image_bias_logit_weight,
        choice_loss_weight=args.choice_loss_weight,
        image_bias_l2=args.image_bias_l2,
        image_nuisance_l2=args.image_nuisance_l2,
        participant_nuisance_l2=args.participant_nuisance_l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_lr_multiplier=args.image_lr_multiplier,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        loss=args.loss,
        huber_delta=args.huber_delta,
        target_transform=args.target_transform,
        early_stop_metric=args.early_stop_metric,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        stratify_folds=bool(args.stratify_folds),
        use_extreme_weighted_loss=bool(args.use_extreme_weighted_loss),
        extreme_loss_alpha=args.extreme_loss_alpha,
        extreme_loss_center=args.extreme_loss_center,
        normalize_extreme_weights=bool(args.normalize_extreme_weights),
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device in {"cpu", "cuda"}:
        device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(args.bank_py)
    data = prepare_data(args.participants_csv, args.responses_csv, bank, args.cesd_col)
    expert_tensor = expert_vectors_tensor(data.image_table, device=device)

    print(f"Device: {device}")
    print(f"Participants: {len(data.y_cesd)} | Trials: {len(data.trial_table)} | Images: {len(data.image_table)}")
    print(
        "Model: choice-calibrated 4D | "
        f"trial 12->{cfg.trial_hidden_dim}->{cfg.trial_embed_dim}; "
        f"pooling={cfg.pooling}; nuisance_dim={cfg.nuisance_dim}"
    )
    print(
        f"Loss: CES-D + {cfg.choice_loss_weight}*choice + image_prior={cfg.image_prior_weight} | "
        f"max_shift={cfg.max_image_shift} | choice_metric={cfg.choice_metric}"
    )

    # Simple ridge baseline from observable choice summaries
    ridge_alpha = 3.0
    X_summary, summary_names = choice_summary_features(data)
    summary_pred = ridge_repeated_cv_predict(
        X_summary,
        data.y_cesd,
        seed=args.seed,
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        alpha=ridge_alpha,
        stratify=cfg.stratify_folds,
    )

    pred_oof, profile_oof, raw_profile_oof, nuisance_oof, choice_oof_metrics, cv_best_epochs = cross_validate(
        data, expert_tensor, args.seed, device, cfg
    )
    scored_oof = profiles_dataframe(
        data.participant_ids, data.y_cesd, pred_oof, profile_oof, raw_profile_oof, nuisance_oof
    )
    scored_oof.to_csv(outdir / "participants_scored_oof.csv", index=False)

    # Train final model for the median useful CV duration. +1 because epoch index is zero-based.
    valid_best = [e + 1 for e in cv_best_epochs if e >= 0]
    final_train_epochs = max(1, int(round(float(np.median(valid_best))))) if valid_best else cfg.epochs
    print(f"Final full-data training epochs (median CV best epoch + 1): {final_train_epochs}")

    all_idx = np.arange(len(data.y_cesd))
    final_model = fit_model(
        chosen_ids=data.chosen_ids,
        rejected_ids=data.rejected_ids,
        y_cesd=data.y_cesd,
        train_idx=all_idx,
        val_idx=None,
        expert_image_vectors=expert_tensor,
        device=device,
        cfg=cfg,
        max_epochs=final_train_epochs,
    )

    pred_final, profile_final, raw_profile_final, nuisance_final, choice_final_metrics = predict_profiles_and_choice_metrics(
        final_model, data.chosen_ids, data.rejected_ids, device
    )
    profiles_final = profiles_dataframe(
        data.participant_ids, data.y_cesd, pred_final, profile_final, raw_profile_final, nuisance_final
    )
    profiles_final.to_csv(outdir / "participant_profiles_final.csv", index=False)

    learned_images = export_learned_image_vectors(final_model, data.image_table, outdir / "learned_image_vectors.csv")
    calibrated_bank_snippet(learned_images, outdir / "calibrated_image_vectors_snippet.py")
    data.trial_table.to_csv(outdir / "trial_bank_used.csv", index=False)

    mean_baseline = np.full_like(data.y_cesd.astype(float), data.y_cesd.astype(float).mean())
    report = {
        "mean_baseline": regression_metrics(data.y_cesd, mean_baseline),
        "choice_summary_ridge_baseline": regression_metrics(data.y_cesd, summary_pred),
        "cross_validated": regression_metrics(data.y_cesd, pred_oof),
        "train_final_model": regression_metrics(data.y_cesd, pred_final),
        "cross_validated_choice_reconstruction": choice_oof_metrics,
        "final_model_choice_reconstruction": choice_final_metrics,
        "training": final_model.training_summary,
        "data": {
            "n_participants": int(len(data.y_cesd)),
            "n_trials": int(len(data.trial_table)),
            "n_images": int(len(data.image_table)),
            "trial_input_dim": 12,
            "profile_dim": 4,
            "nuisance_dim": int(cfg.nuisance_dim),
            "choice_summary_feature_names": summary_names,
            "choice_summary_ridge_cv": {
                "n_splits": int(cfg.n_splits),
                "n_repeats": int(cfg.n_repeats),
                "stratify_folds": bool(cfg.stratify_folds),
                "alpha": float(ridge_alpha),
                "alpha_tuned": False,
            },
        },
        "model_config": asdict(cfg),
        "image_calibration": {
            "mean_image_shift_l2": float(learned_images["visual_shift_l2"].mean()),
            "max_image_shift_l2": float(learned_images["visual_shift_l2"].max()),
            "mean_abs_image_bias": float(np.mean(np.abs(learned_images["image_bias"].to_numpy(dtype=float)))),
            "mean_image_nuisance_l2": float(learned_images["image_nuisance_l2"].mean()),
            "choice_scale": float(final_model.choice_scale().detach().cpu()),
        },
        "mean_oof_profile": {dim: float(scored_oof[f"profile_{dim}"].mean()) for dim in LATENT_DIMS},
    }

    with open(outdir / "metrics_choice_calibrated_4d.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "image_table": data.image_table.to_dict(orient="records"),
            "trial_table": data.trial_table.to_dict(orient="records"),
            "latent_dims": LATENT_DIMS,
            "model_config": asdict(cfg),
            "target_transform": asdict(final_model.target_transform),
            "model_type": "choice_calibrated_4d",
            "profile_definition": "sigmoid(profile_head(pooled_choices)); dimensions follow LATENT_DIMS",
            "choice_decoder": {
                "metric": cfg.choice_metric,
                "uses_clinical_profile": True,
                "uses_image_bias": True,
                "uses_participant_nuisance": cfg.nuisance_dim > 0,
            },
        },
        outdir / "final_choice_calibrated_4d_model.pt",
    )

    print(json.dumps(report, indent=2))
    print(f"Saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
