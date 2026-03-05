#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor ML LATAM — Plug & Play (v3)
====================================
v3 = v2 + "paper-grade" upgrades:
1) Hyperparameter tuning con CV temporal (rolling por años) para el modelo "best" por tarea.
2) Calibración de probabilidades (Isotonic/Platt) para evento y hazard (CalibratedClassifierCV).
3) Conformal prediction:
   - Regresión: split conformal (intervalos con cobertura nominal).
   - Clasificación: split conformal sets sobre probabilidades calibradas.
4) Ablation / robustness estructural:
   - Modelo "BASE" vs "EXTENDED" (por bloques de features).
5) Reportes de estabilidad:
   - Performance por año y por país en test.
6) Validación externa (opcional):
   - Si existe outcome de tabaco (tobacco_score / tobacco_law), corre el mismo pipeline nivel t+1.

Ejecución:
    python predictor_ml_pipeline_v3.py --base_path "/Users/.../predictor_ml_latam"

Salida: base_path/05_outputs/

Dependencias:
- pandas, numpy, scikit-learn, matplotlib, joblib, openpyxl
- shap (opcional; si falla usa permutation importance)
"""
from __future__ import annotations
import argparse, json, os, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, brier_score_loss
)
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
import joblib
import matplotlib.pyplot as plt


LATAM_ISO3_DEFAULT = [
    "ARG","BLZ","BOL","BRA","CHL","COL","CRI","ECU","SLV","GTM","GUY",
    "HND","MEX","NIC","PAN","PRY","PER","SUR","URY","VEN"
]


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def log(msg: str) -> None:
    print(msg, flush=True)

def warn(msg: str) -> None:
    warnings.warn(msg)
    print(f"WARNING: {msg}", flush=True)

def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def plot_and_save(fig_path: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def safe_read_excel(path: str, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)

def standardize_panel(df: pd.DataFrame,
                      iso_candidates: List[str] = None,
                      year_candidates: List[str] = None) -> pd.DataFrame:
    if iso_candidates is None:
        iso_candidates = ["iso3c","iso3","iso_code","code","ref_area","recipient_isocode"]
    if year_candidates is None:
        year_candidates = ["year","time_period","yr","Year"]

    cols = {c.lower(): c for c in df.columns}
    iso_col = next((cols[c.lower()] for c in iso_candidates if c.lower() in cols), None)
    year_col = next((cols[c.lower()] for c in year_candidates if c.lower() in cols), None)
    if iso_col is None:
        raise ValueError("No ISO column found.")
    if year_col is None:
        raise ValueError("No year column found.")

    out = df.copy()
    out["iso3c"] = out[iso_col].astype(str).str.upper().str.strip()
    out["year"] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    out = out.drop(columns=[c for c in [iso_col, year_col] if c not in ["iso3c","year"]], errors="ignore")
    out = out.dropna(subset=["iso3c","year"])
    out["year"] = out["year"].astype(int)
    out = out.drop_duplicates(subset=["iso3c","year"])
    return out

def filter_latam(df: pd.DataFrame, latam_iso3: List[str]) -> pd.DataFrame:
    return df[df["iso3c"].isin(latam_iso3)].copy()

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def year_split_idx(df: pd.DataFrame, train_end: int, val_end: int, test_end: int) -> Dict[str, np.ndarray]:
    y = df["year"].values
    return {
        "train": np.where(y <= train_end)[0],
        "val": np.where((y > train_end) & (y <= val_end))[0],
        "test": np.where((y > val_end) & (y <= test_end))[0],
    }
   
def pick_threshold_max_f1(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Pick probability threshold that maximizes F1 on a given set."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    # Guardrails
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "f1": None, "precision": None, "recall": None}

    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for t in thresholds:
        y_hat = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(prec), "recall": float(rec)}

    return best


def eval_probs_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Evaluate probabilistic classifier with a chosen threshold (plus rank/calibration metrics)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)

    y_hat = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)

    out = {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    # ranking metrics only meaningful if both classes exist
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out

# -----------------------------
# Temporal CV (rolling by year)
# -----------------------------
def make_year_folds(df: pd.DataFrame, min_train_years: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Rolling expanding window folds:
      fold i: train years <= years[i-1], val year = years[i]
    """
    years = sorted(df["year"].unique())
    folds = []
    for i in range(min_train_years, len(years)-1):
        train_end = years[i-1]
        val_year = years[i]
        tr = np.where(df["year"].values <= train_end)[0]
        va = np.where(df["year"].values == val_year)[0]
        if len(tr) < 50 or len(va) < 10:
            continue
        folds.append((tr, va))
    return folds

def folds_to_predefined_split(folds: List[Tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    test_fold = np.full(n, -1, dtype=int)
    for fid, (_, va) in enumerate(folds):
        test_fold[va] = fid
    return test_fold


# -----------------------------
# Preprocessing
# -----------------------------
def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")


# -----------------------------
# Paths & Loading
# -----------------------------
@dataclass
class DataPaths:
    base_path: str
    clean_dir: str
    raw_dir: str
    out_dir: str
    fig_dir: str
    model_dir: str
    tables_dir: str

def make_paths(base_path: str) -> DataPaths:
    clean_dir = os.path.join(base_path, "02_clean_data")
    raw_dir   = os.path.join(base_path, "01_raw_data")
    out_dir   = os.path.join(base_path, "05_outputs")
    fig_dir   = os.path.join(out_dir, "figures")
    model_dir = os.path.join(out_dir, "models")
    tables_dir= os.path.join(out_dir, "tables")
    for p in [out_dir, fig_dir, model_dir, tables_dir]:
        ensure_dir(p)
    return DataPaths(base_path, clean_dir, raw_dir, out_dir, fig_dir, model_dir, tables_dir)

def load_optional_panel(paths: DataPaths, name: str, candidates: List[str], reader: str = "csv") -> Optional[pd.DataFrame]:
    search_paths = []
    for c in candidates:
        search_paths.append(os.path.join(paths.clean_dir, c))
        search_paths.append(os.path.join(paths.raw_dir, c))
    p = first_existing(search_paths)
    if p is None:
        warn(f"[SKIP] No encontré {name}. Candidatos: {candidates}")
        return None
    log(f"[LOAD] {name}: {p}")
    try:
        df = safe_read_csv(p) if reader == "csv" else safe_read_excel(p)
        df = standardize_panel(df)
        return df
    except Exception as e:
        warn(f"[SKIP] Error cargando {name}: {e}")
        return None

def detect_outcome_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    cols = [c for c in df.columns if c not in ["iso3c","year"]]
    if not cols:
        raise ValueError("No outcome column found.")
    return cols[0]

def build_master_panel(paths: DataPaths, latam_iso3: List[str]) -> pd.DataFrame:
    # ---------------------------------------------------------
    # Prefer prebuilt panel (R output) if available
    # ---------------------------------------------------------
    panel_pruned_path = os.path.join(paths.base_path, "03_panel", "panel_master_ml_ready_pruned.csv")
    panel_ml_path     = os.path.join(paths.base_path, "03_panel", "panel_master_ml_ready.csv")
    panel_clean_path  = os.path.join(paths.base_path, "03_panel", "panel_master_clean.csv")

    for p in [panel_pruned_path, panel_ml_path, panel_clean_path]:
        if os.path.exists(p):
            log(f"[LOAD] Using prebuilt panel: {p}")
            df = pd.read_csv(p, low_memory=False)
            df = standardize_panel(df)
            df = filter_latam(df, latam_iso3)

            # If SPAR exists but targets not, derive them
            if "spar_cap_law" in df.columns:
                df = coerce_numeric(df, ["spar_cap_law"])
                df = df.sort_values(["iso3c", "year"]).reset_index(drop=True)

                if "spar_lag1" not in df.columns:
                    df["spar_lag1"] = df.groupby("iso3c")["spar_cap_law"].shift(1)
                if "spar_lag2" not in df.columns:
                    df["spar_lag2"] = df.groupby("iso3c")["spar_cap_law"].shift(2)
                if "spar_delta" not in df.columns:
                    df["spar_delta"] = df["spar_cap_law"] - df["spar_lag1"]
                if "spar_tplus1" not in df.columns:
                    df["spar_tplus1"] = df.groupby("iso3c")["spar_cap_law"].shift(-1)
                if "delta_tplus1" not in df.columns:
                    df["delta_tplus1"] = df["spar_tplus1"] - df["spar_cap_law"]

            return df

    # ---------------------------------------------------------
    # Fallback: build panel from individual clean datasets
    # ---------------------------------------------------------
    df_spar = load_optional_panel(paths, "SPAR", ["spar_clean_latam.csv", "spar_clean_latam_panel.csv"])
    if df_spar is None:
        raise FileNotFoundError("Necesito SPAR para correr el pipeline.")
    df_spar = filter_latam(df_spar, latam_iso3)
    outcol = detect_outcome_col(df_spar, ["spar_cap_law","y","spar_score","spar_law"])
    df_spar = df_spar.rename(columns={outcol: "spar_cap_law"})
    df_spar = coerce_numeric(df_spar, ["spar_cap_law"])

    blocks = [
        ("Political panel", ["political_panel_clean_latam.csv","political_panel_latam_2010_2023.csv","political_panel_latam_2010_2020.csv"]),
        ("V-Dem subset", ["vdem_subset_clean_latam.csv","vdem_model1_subset_latam.csv"]),
        ("WGI", ["wgi_clean_latam.csv","vgi_capa2_latam_2000_2024.csv"]),
        ("GDP pc", ["gdp_pc_clean_latam.csv","gdp_pc_wb_latam_2000_2024.csv"]),
        ("Health exp", ["health_exp_clean_latam.csv","health_gdp_wb_latam_2000_2024.csv"]),
        ("UHC", ["uhc_clean_latam.csv","uhc_latam_2010_2023.csv"]),
        ("COVID", ["covid_deaths_last12m_clean_latam.csv"]),
        ("Populism GPD", ["populism_gpd_term_expanded_clean_latam.csv"]),
        ("Votes4Populists", ["votes4populists_clean_latam.csv"]),
        ("Fiscal decentralization", ["fiscal_decentralization_clean_latam.csv"]),
        ("IHME DAH total", ["ihme_dah_total_clean_latam.csv"]),
        ("WHO participation", ["who_participation_clean_latam.csv"]),
        ("WHO exclusions", ["who_exclusions_clean_latam.csv"]),
    ]

    panel = df_spar.copy()
    for name, cands in blocks:
        d = load_optional_panel(paths, name, cands)
        if d is None:
            continue
        d = filter_latam(d, latam_iso3)
        panel = panel.merge(d, on=["iso3c","year"], how="left")

    panel = panel.sort_values(["iso3c","year"]).reset_index(drop=True)

    if "gdp_pc" in panel.columns and "log_gdp_pc" not in panel.columns:
        panel["log_gdp_pc"] = np.log(pd.to_numeric(panel["gdp_pc"], errors="coerce"))
    if "excluded_wha" in panel.columns:
        panel["excluded_wha"] = panel["excluded_wha"].fillna(0).astype(int)

    panel["spar_lag1"] = panel.groupby("iso3c")["spar_cap_law"].shift(1)
    panel["spar_lag2"] = panel.groupby("iso3c")["spar_cap_law"].shift(2)
    panel["spar_delta"] = panel["spar_cap_law"] - panel["spar_lag1"]
    panel["spar_tplus1"] = panel.groupby("iso3c")["spar_cap_law"].shift(-1)
    panel["delta_tplus1"] = panel["spar_tplus1"] - panel["spar_cap_law"]
    return panel


# -----------------------------
# Data quality report
# -----------------------------
def data_quality_report(panel: pd.DataFrame, out_dir: str) -> None:
    report_path = os.path.join(out_dir, "data_quality_report.md")
    miss = panel.isna().mean().sort_values(ascending=False)
    lines = [
        "# Data quality report\n",
        f"- Rows: {len(panel)}",
        f"- Countries: {panel['iso3c'].nunique()}",
        f"- Years: {int(panel['year'].min())}–{int(panel['year'].max())}\n",
        "## Missingness (top 30)\n",
        miss.head(30).to_string(),
        "\n## Duplicates check\n",
        f"- Duplicates iso3c-year: {panel.duplicated(['iso3c','year']).sum()}\n"
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Feature blocks (BASE vs EXTENDED)
# -----------------------------
def infer_feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = set(df.columns)
    core = [c for c in ["spar_lag1","spar_lag2","spar_delta"] if c in cols]

    political = [c for c in df.columns if any(k in c.lower() for k in ["gov_", "honeymoon", "election", "seat", "years_since", "coalition", "sologov", "pressystem"])]
    vdem = [c for c in df.columns if c.lower().startswith("v2") or "vdem" in c.lower() or "polar" in c.lower()]
    wgi  = [c for c in df.columns if c.lower().endswith("_est") or c.lower().startswith(("cc_","ge_","rq_","rl_","va_","pv_"))]
    econ = [c for c in df.columns if any(k in c.lower() for k in ["gdp", "log_gdp", "health_exp", "uhc"])]
    shock = [c for c in df.columns if "covid" in c.lower()]
    intl = [c for c in df.columns if any(k in c.lower() for k in ["dah", "who_part", "excluded_wha", "participation"])]
    pop = [c for c in df.columns if "popul" in c.lower()]
    fiscaldec = [c for c in df.columns if "decentral" in c.lower() or "fiscal" in c.lower()]

    base = list(dict.fromkeys(core + political + vdem + wgi + econ))
    extended = list(dict.fromkeys(base + shock + intl + pop + fiscaldec))

    for rm in ["iso3c","year","spar_cap_law","spar_tplus1","delta_tplus1","event_next","y_event"]:
        if rm in base: base.remove(rm)
        if rm in extended: extended.remove(rm)

    return {"base": base, "extended": extended}


# -----------------------------
# Conformal
# -----------------------------
def split_conformal_interval(y_cal: np.ndarray, yhat_cal: np.ndarray, alpha: float = 0.10) -> float:
    resid = np.abs(y_cal - yhat_cal)
    return float(np.quantile(resid, 1 - alpha, method="higher"))

def conformal_classification_tau(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10) -> float:
    p_true = np.where(y_cal == 1, p_cal, 1 - p_cal)
    s = 1 - p_true
    return float(np.quantile(s, 1 - alpha, method="higher"))

def apply_conformal_sets(p: np.ndarray, tau: float) -> List[str]:
    sets = []
    for pi in p:
        include1 = (1 - pi) <= tau
        include0 = pi <= tau
        if include0 and include1:
            sets.append("{0,1}")
        elif include1:
            sets.append("{1}")
        elif include0:
            sets.append("{0}")
        else:
            sets.append("{}")
    return sets


# -----------------------------
# Tuning
# -----------------------------
def tune_with_predefined_split(estimator, param_distributions: dict, X, y, folds, n_iter, seed, scoring):
    from sklearn.model_selection import PredefinedSplit
    test_fold = folds_to_predefined_split(folds, len(X))
    ps = PredefinedSplit(test_fold)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=ps,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search


# -----------------------------
# Metrics helpers
# -----------------------------
def eval_probs(y_true: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1-1e-6)
    y_hat = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier": float(brier_score_loss(y_true, p))
    }

def perf_by_group_reg(df, y_true_col, y_pred_col, group_col):
    out=[]
    for g, sub in df.groupby(group_col):
        out.append({
            "group": g, "n": len(sub),
            "rmse": rmse(sub[y_true_col], sub[y_pred_col]),
            "mae": float(mean_absolute_error(sub[y_true_col], sub[y_pred_col]))
        })
    return pd.DataFrame(out).sort_values("n", ascending=False)

def perf_by_group_clf(df, y_true_col, p_col, group_col):
    out=[]
    for g, sub in df.groupby(group_col):
        y = sub[y_true_col].values
        p = np.clip(sub[p_col].values, 1e-6, 1-1e-6)
        if len(np.unique(y)) < 2:
            out.append({"group": g, "n": len(sub), "roc_auc": None, "pr_auc": None, "brier": float(brier_score_loss(y,p))})
        else:
            out.append({
                "group": g, "n": len(sub),
                "roc_auc": float(roc_auc_score(y,p)),
                "pr_auc": float(average_precision_score(y,p)),
                "brier": float(brier_score_loss(y,p))
            })
    return pd.DataFrame(out).sort_values("n", ascending=False)


# -----------------------------
# Explainability (permutation on raw cols)
# -----------------------------
def permutation_explain(model, X_test, y_test, out_dir, fig_dir, task):
    scoring = "neg_root_mean_squared_error" if task == "reg" else "average_precision"
    try:
        pi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1, scoring=scoring)
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False)
        save_df(imp, os.path.join(out_dir, "perm_importance_rawcols.csv"))
        plt.figure()
        top = imp.head(25).iloc[::-1]
        plt.barh(top["feature"], top["importance_mean"])
        plt.title("Permutation importance (raw cols) — top 25")
        plt.xlabel("Mean importance")
        plot_and_save(os.path.join(fig_dir, "perm_importance_rawcols.png"))
    except Exception as e:
        warn(f"Permutation importance falló: {e}")


# -----------------------------
# Task 1: Level (tuned + conformal)
# -----------------------------
def run_level(panel, paths, args, features, tag):
    task_dir = os.path.join(paths.tables_dir, f"level_{tag}")
    fig_dir = os.path.join(paths.fig_dir, f"level_{tag}")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["spar_tplus1"]).copy()
    feats = [c for c in features if c in d.columns]
    if len(feats) < 5:
        warn(f"Level({tag}): muy pocas features. Usando todas menos targets.")
        feats = [c for c in d.columns if c not in ["iso3c","year","spar_cap_law","spar_tplus1","delta_tplus1"]]

    split = year_split_idx(d, args.train_end_year, args.val_end_year, args.test_end_year)
    tr, va, te = split["train"], split["val"], split["test"]

    X = d[feats].copy()
    y = d["spar_tplus1"].astype(float).values

    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    pre = make_preprocessor(num_cols, cat_cols)

    pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(random_state=args.seed))])

    # tune on train+val with rolling year folds
    tv_idx = np.concatenate([tr, va])
    d_tv = d.iloc[tv_idx]
    folds = make_year_folds(d_tv, min_train_years=args.min_train_years)

    param_dist = {
        "model__max_depth": [3,4,5,6,7],
        "model__learning_rate": np.linspace(0.03, 0.12, 10),
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }
    search = tune_with_predefined_split(pipe, param_dist, X.iloc[tv_idx], y[tv_idx], folds, args.tune_iter, args.seed, "neg_root_mean_squared_error")
    best = search.best_estimator_

    # fit train+val final, test
    best.fit(X.iloc[tv_idx], y[tv_idx])
    pred = best.predict(X.iloc[te])

    metrics = {
        "test_rmse": rmse(y[te], pred) if len(te) else None,
        "test_mae": float(mean_absolute_error(y[te], pred)) if len(te) else None,
        "test_r2": float(r2_score(y[te], pred)) if len(te) else None,
        "best_params": search.best_params_,
        "features_used": len(feats),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    }
    save_json(metrics, os.path.join(task_dir, "metrics.json"))

    model_path = os.path.join(paths.model_dir, f"level_{tag}_tuned.joblib")
    joblib.dump(best, model_path)

    pred_df = d.iloc[te][["iso3c","year","spar_cap_law","spar_tplus1"]].copy()
    pred_df["pred_spar_tplus1"] = pred

    # conformal interval using val as calibration, fit model on train
    best.fit(X.iloc[tr], y[tr])
    yhat_val = best.predict(X.iloc[va])
    qhat = split_conformal_interval(y[va], yhat_val, alpha=args.alpha)
    yhat_test = best.predict(X.iloc[te])
    pred_df["pi_lo"] = yhat_test - qhat
    pred_df["pi_hi"] = yhat_test + qhat
    pred_df["qhat"] = qhat
    save_df(pred_df, os.path.join(task_dir, "predictions_test_with_conformal_pi.csv"))

    # stability
    save_df(perf_by_group_reg(pred_df, "spar_tplus1", "pred_spar_tplus1", "year"), os.path.join(task_dir, "stability_by_year.csv"))
    save_df(perf_by_group_reg(pred_df, "spar_tplus1", "pred_spar_tplus1", "iso3c"), os.path.join(task_dir, "stability_by_country.csv"))

    # plots
    plt.figure()
    plt.scatter(pred_df["spar_tplus1"], pred_df["pred_spar_tplus1"], s=10)
    plt.xlabel("Actual spar_t+1"); plt.ylabel("Predicted spar_t+1")
    plt.title(f"Level tuned — {tag}")
    plot_and_save(os.path.join(fig_dir, "pred_vs_actual.png"))

    g = pred_df.groupby("year", as_index=False).agg(actual=("spar_tplus1","mean"), pred=("pred_spar_tplus1","mean"), lo=("pi_lo","mean"), hi=("pi_hi","mean"))
    plt.figure()
    plt.plot(g["year"], g["actual"], label="Actual")
    plt.plot(g["year"], g["pred"], label="Pred")
    plt.fill_between(g["year"], g["lo"], g["hi"], alpha=0.2, label=f"Conformal {int((1-args.alpha)*100)}%")
    plt.legend(); plt.xlabel("Year"); plt.ylabel("SPAR t+1")
    plt.title(f"Yearly mean + PI — {tag}")
    plot_and_save(os.path.join(fig_dir, "yearly_mean_pi.png"))

    permutation_explain(best, X.iloc[te], y[te], os.path.join(task_dir, "explain"), fig_dir, "reg")

    return {"model_path": model_path, "metrics": metrics}


# -----------------------------
# Task 2: Event (tuned + calibrated + conformal sets)
# -----------------------------
def run_event(panel, paths, args, features, tag):
    task_dir = os.path.join(paths.tables_dir, f"event_{tag}")
    fig_dir = os.path.join(paths.fig_dir, f"event_{tag}")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["delta_tplus1"]).copy()
    d["y_event"] = (d["delta_tplus1"] >= args.event_threshold).astype(int)

    feats = [c for c in features if c in d.columns]
    if len(feats) < 5:
        warn(f"Event({tag}): muy pocas features. Usando todas menos targets.")
        feats = [c for c in d.columns if c not in ["iso3c","year","spar_cap_law","spar_tplus1","delta_tplus1","y_event"]]

    split = year_split_idx(d, args.train_end_year, args.val_end_year, args.test_end_year)
    tr, va, te = split["train"], split["val"], split["test"]

    X = d[feats].copy()
    y = d["y_event"].values

    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    pre = make_preprocessor(num_cols, cat_cols)

    pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=args.seed))])

    tv_idx = np.concatenate([tr, va])
    d_tv = d.iloc[tv_idx]
    folds = make_year_folds(d_tv, min_train_years=args.min_train_years)

    param_dist = {
        "model__max_depth": [3,4,5,6],
        "model__learning_rate": np.linspace(0.03, 0.12, 10),
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }
    search = tune_with_predefined_split(pipe, param_dist, X.iloc[tv_idx], y[tv_idx], folds, args.tune_iter, args.seed, "average_precision")
    best = search.best_estimator_

    # fit train, calibrate on val
    best.fit(X.iloc[tr], y[tr])
    calib = CalibratedClassifierCV(best, method=args.calibration, cv="prefit")
    calib.fit(X.iloc[va], y[va])

    p_test = calib.predict_proba(X.iloc[te])[:,1]
    p_val  = calib.predict_proba(X.iloc[va])[:,1]

    # ---- choose threshold on validation ----
    thr_info = pick_threshold_max_f1(y[va], p_val) if len(va) else {"threshold": 0.5, "f1": None, "precision": None, "recall": None}
    best_thr = float(thr_info["threshold"])

    metrics = {
        "val_default05": eval_probs(y[va], p_val) if len(va) else None,
        "test_default05": eval_probs(y[te], p_test) if len(te) else None,

        "val": eval_probs_with_threshold(y[va], p_val, best_thr) if len(va) else None,
        "test": eval_probs_with_threshold(y[te], p_test, best_thr) if len(te) else None,
       
        "threshold_selection": thr_info,
        "best_params": search.best_params_,
        "features_used": len(feats),
        "calibration": args.calibration,
    }
    save_json(metrics, os.path.join(task_dir, "metrics.json"))

    model_path = os.path.join(paths.model_dir, f"event_{tag}_tuned_calibrated.joblib")
    joblib.dump({"model": best, "calibrator": calib}, model_path)

    pred_df = d.iloc[te][["iso3c","year","y_event","delta_tplus1"]].copy()
    pred_df["pred_prob"] = p_test
    pred_df["pred_label"] = (pred_df["pred_prob"] >= best_thr).astype(int)
    pred_df["threshold_used"] = (best_thr)

    tau = conformal_classification_tau(p_val, y[va], alpha=args.alpha)
    pred_df["conformal_set"] = apply_conformal_sets(p_test, tau)
    pred_df["tau"] = tau
    save_df(pred_df, os.path.join(task_dir, "predictions_test_with_conformal_sets.csv"))

    save_df(perf_by_group_clf(pred_df, "y_event", "pred_prob", "year"), os.path.join(task_dir, "stability_by_year.csv"))
    save_df(perf_by_group_clf(pred_df, "y_event", "pred_prob", "iso3c"), os.path.join(task_dir, "stability_by_country.csv"))

    if len(np.unique(y[te])) > 1:
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y[te], p_test)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR curve — {tag}")
        plot_and_save(os.path.join(fig_dir, "pr_curve.png"))

        frac_pos, mean_pred = calibration_curve(y[te], p_test, n_bins=10)
        plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
        plt.xlabel("Mean predicted"); plt.ylabel("Fraction positives")
        plt.title(f"Calibration — {tag}")
        plot_and_save(os.path.join(fig_dir, "calibration.png"))

    permutation_explain(calib, X.iloc[te], y[te], os.path.join(task_dir, "explain"), fig_dir, "clf")

    return {"model_path": model_path, "metrics": metrics}


# -----------------------------
# Task 3: Hazard (tuned + calibrated + conformal sets)
# -----------------------------
def run_hazard(panel, paths, args, features, tag):
    task_dir = os.path.join(paths.tables_dir, f"hazard_{tag}")
    fig_dir = os.path.join(paths.fig_dir, f"hazard_{tag}")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["spar_cap_law","spar_tplus1"]).copy()
    d = d[d["spar_cap_law"] < args.level_threshold].copy()
    d["event_next"] = (d["spar_tplus1"] >= args.level_threshold).astype(int)
    if len(d) < 80:
        warn(f"Hazard({tag}): dataset pequeño ({len(d)}). Saltando.")
        return {"skipped": True, "n": int(len(d))}

    feats = [c for c in features if c in d.columns]
    if len(feats) < 5:
        warn(f"Hazard({tag}): muy pocas features. Usando todas menos targets.")
        feats = [c for c in d.columns if c not in ["iso3c","year","spar_cap_law","spar_tplus1","delta_tplus1","event_next"]]

    split = year_split_idx(d, args.train_end_year, args.val_end_year, args.test_end_year)
    tr, va, te = split["train"], split["val"], split["test"]

    X = d[feats].copy()
    y = d["event_next"].values

    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    pre = make_preprocessor(num_cols, cat_cols)

    pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=args.seed))])

    tv_idx = np.concatenate([tr, va])
    d_tv = d.iloc[tv_idx]
    folds = make_year_folds(d_tv, min_train_years=args.min_train_years)

    param_dist = {
        "model__max_depth": [3,4,5,6],
        "model__learning_rate": np.linspace(0.03, 0.12, 10),
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }
    search = tune_with_predefined_split(pipe, param_dist, X.iloc[tv_idx], y[tv_idx], folds, args.tune_iter, args.seed, "average_precision")
    best = search.best_estimator_

    best.fit(X.iloc[tr], y[tr])
    calib = CalibratedClassifierCV(best, method=args.calibration, cv="prefit")
    calib.fit(X.iloc[va], y[va])

    p_test = calib.predict_proba(X.iloc[te])[:,1]
    p_val  = calib.predict_proba(X.iloc[va])[:,1]

    # ---- choose threshold on validation (headline-friendly) ----
    thr_info = pick_threshold_max_f1(y[va], p_val) if len(va) else {"threshold": 0.5, "f1": None, "precision": None, "recall": None}
    best_thr = thr_info["threshold"]

    metrics = {
        "val_default05": eval_probs(y[va], p_val) if len(va) else None,
        "test_default05": eval_probs(y[te], p_test) if len(te) else None,

        # operational metrics with tuned threshold
        "val": eval_probs_with_threshold(y[va], p_val, best_thr) if len(va) else None,
        "test": eval_probs_with_threshold(y[te], p_test, best_thr) if len(te) else None,

        "threshold_selection": thr_info,
        "best_params": search.best_params_,
        "features_used": len(feats),
        "calibration": args.calibration,
    }
    save_json(metrics, os.path.join(task_dir, "metrics.json"))

    model_path = os.path.join(paths.model_dir, f"hazard_{tag}_tuned_calibrated.joblib")
    joblib.dump({"model": best, "calibrator": calib}, model_path)

    pred_df = d.iloc[te][["iso3c","year","event_next","spar_cap_law","spar_tplus1"]].copy()
    pred_df["pred_hazard"] = p_test

    tau = conformal_classification_tau(p_val, y[va], alpha=args.alpha)
    pred_df["conformal_set"] = apply_conformal_sets(p_test, tau)
    pred_df["tau"] = tau
    save_df(pred_df, os.path.join(task_dir, "predictions_test_with_conformal_sets.csv"))

    save_df(perf_by_group_clf(pred_df, "event_next", "pred_hazard", "year"), os.path.join(task_dir, "stability_by_year.csv"))
    save_df(perf_by_group_clf(pred_df, "event_next", "pred_hazard", "iso3c"), os.path.join(task_dir, "stability_by_country.csv"))

    if len(np.unique(y[te])) > 1:
        frac_pos, mean_pred = calibration_curve(y[te], p_test, n_bins=10)
        plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
        plt.xlabel("Mean predicted hazard"); plt.ylabel("Fraction events")
        plt.title(f"Calibration — {tag}")
        plot_and_save(os.path.join(fig_dir, "calibration.png"))

    permutation_explain(calib, X.iloc[te], y[te], os.path.join(task_dir, "explain"), fig_dir, "clf")
    return {"model_path": model_path, "metrics": metrics}


# -----------------------------
# External validation optional: Tobacco
# -----------------------------
def run_tobacco_optional(paths: DataPaths, latam_iso3: List[str], args) -> Optional[dict]:
    df = load_optional_panel(paths, "Tobacco", [
        "tobacco_law_enforce_clean_latam.csv",
        "tobacco_law_enforce_index_latam_CORE_FULL.csv",
        "tobacco_clean_latam.csv"
    ])
    if df is None:
        return None
    df = filter_latam(df, latam_iso3)
    outcol = detect_outcome_col(df, ["tobacco_score","tobacco_law","y"])
    df = df.rename(columns={outcol:"tobacco"})
    df = coerce_numeric(df, ["tobacco"])
    df = df.sort_values(["iso3c","year"]).reset_index(drop=True)
    df["tobacco_tplus1"] = df.groupby("iso3c")["tobacco"].shift(-1)
    df = df.dropna(subset=["tobacco_tplus1"])
    # baseline predictor: use lag1/lag2 as minimal
    df["tob_lag1"] = df.groupby("iso3c")["tobacco"].shift(1)
    df["tob_lag2"] = df.groupby("iso3c")["tobacco"].shift(2)

    split = year_split_idx(df, args.train_end_year, args.val_end_year, args.test_end_year)
    tr, va, te = split["train"], split["val"], split["test"]

    feats = [c for c in ["tob_lag1","tob_lag2"] if c in df.columns]
    X = df[feats].copy()
    y = df["tobacco_tplus1"].astype(float).values

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("model", HistGradientBoostingRegressor(random_state=args.seed))])
    pipe.fit(X.iloc[np.concatenate([tr,va])], y[np.concatenate([tr,va])])
    pred = pipe.predict(X.iloc[te]) if len(te) else np.array([])

    out = {
        "test_rmse": rmse(y[te], pred) if len(te) else None,
        "test_mae": float(mean_absolute_error(y[te], pred)) if len(te) else None,
        "n_test": int(len(te))
    }
    save_json(out, os.path.join(paths.out_dir, "external_validation_tobacco.json"))
    return out


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, required=True)
    p.add_argument("--train_end_year", type=int, default=2018)
    p.add_argument("--val_end_year", type=int, default=2019)
    p.add_argument("--test_end_year", type=int, default=2023)
    p.add_argument("--event_threshold", type=float, default=2.0)
    p.add_argument("--level_threshold", type=float, default=80.0)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--calibration", type=str, default="isotonic", choices=["isotonic","sigmoid"])
    p.add_argument("--tune_iter", type=int, default=40)
    p.add_argument("--min_train_years", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latam_iso3", type=str, default=",".join(LATAM_ISO3_DEFAULT))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    latam_iso3 = [x.strip().upper() for x in args.latam_iso3.split(",") if x.strip()]
    paths = make_paths(args.base_path)

    panel = build_master_panel(paths, latam_iso3)
    save_df(panel, os.path.join(paths.tables_dir, "panel_master_snapshot.csv"))
    data_quality_report(panel, paths.out_dir)

    blocks = infer_feature_blocks(panel)

    log("[TASK] Level BASE");     level_base = run_level(panel, paths, args, blocks["base"], "base")
    log("[TASK] Level EXTENDED"); level_ext  = run_level(panel, paths, args, blocks["extended"], "extended")

    log("[TASK] Event BASE");     event_base = run_event(panel, paths, args, blocks["base"], "base")
    log("[TASK] Event EXTENDED"); event_ext  = run_event(panel, paths, args, blocks["extended"], "extended")

    # Hazard (optional): if it fails (e.g., empty temporal CV folds), skip without crashing
    try:
        log("[TASK] Hazard BASE")
        haz_base = run_hazard(panel, paths, args, blocks["base"], "base")
    except Exception as e:
        warn(f"Hazard BASE skipped due to error: {e}")
        haz_base = {"skipped": True, "error": str(e)}

    try:
        log("[TASK] Hazard EXTENDED")
        haz_ext = run_hazard(panel, paths, args, blocks["extended"], "extended")
    except Exception as e:
        warn(f"Hazard EXTENDED skipped due to error: {e}")
        haz_ext = {"skipped": True, "error": str(e)}

    ext_val = run_tobacco_optional(paths, latam_iso3, args)

    summary = {
        "config": vars(args),
        "level_base": level_base.get("metrics", {}),
        "level_extended": level_ext.get("metrics", {}),
        "event_base": event_base.get("metrics", {}),
        "event_extended": event_ext.get("metrics", {}),
        "hazard_base": haz_base.get("metrics", {}),
        "hazard_extended": haz_ext.get("metrics", {}),
        "external_validation_tobacco": ext_val
    }
    save_json(summary, os.path.join(paths.out_dir, "summary_v3.json"))
    log("[DONE] v3 listo — outputs en 05_outputs/")
