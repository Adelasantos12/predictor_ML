#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor ML LATAM — Plug & Play
================================
Objetivo:
- (1) Nivel esperado de internalización legal (SPAR law capacity) a t+1 (regresión)
- (2) Probabilidad de avance anual (evento: salto >= umbral) (clasificación)
- (3) Rapidez: tiempo esperado hasta alcanzar umbral de internalización (discrete-time survival)

El script:
- Lee múltiples archivos (si existen) desde 02_clean_data (preferido) o 01_raw_data (fallback)
- Construye panel maestro país-año (iso3c, year)
- Hace split temporal (train/val/test) y varias pruebas de robustez
- Entrena varios modelos
- Calcula SHAP si está instalado (si no, permutation importance)
- Exporta tablas, modelos y gráficas a 05_outputs/

Ejemplo:
    python predictor_ml_pipeline.py --base_path "/Users/adelasantos/Documents/predictor_ml_latam"

"""
from __future__ import annotations

import argparse
import json
import os
import warnings
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
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    brier_score_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)

import joblib
import matplotlib.pyplot as plt


LATAM_ISO3_DEFAULT = [
    "ARG","BLZ","BOL","BRA","CHL","COL","CRI","ECU","SLV","GTM","GUY",
    "HND","MEX","NIC","PAN","PRY","PER","SUR","URY","VEN"
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def safe_read_excel(path: str, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    warnings.warn(msg)
    print(f"WARNING: {msg}", flush=True)


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def standardize_panel(df: pd.DataFrame,
                      iso_candidates: List[str] = None,
                      year_candidates: List[str] = None) -> pd.DataFrame:
    """Ensure df has iso3c + year columns."""
    if iso_candidates is None:
        iso_candidates = ["iso3c","iso3","iso_code","code","ref_area","recipient_isocode"]
    if year_candidates is None:
        year_candidates = ["year","time_period","yr","Year"]

    cols = {c.lower(): c for c in df.columns}

    iso_col = None
    for c in iso_candidates:
        if c.lower() in cols:
            iso_col = cols[c.lower()]
            break
    if iso_col is None:
        raise ValueError("No encontré columna ISO (iso3c/code/ref_area/recipient_isocode).")

    year_col = None
    for c in year_candidates:
        if c.lower() in cols:
            year_col = cols[c.lower()]
            break
    if year_col is None:
        raise ValueError("No encontré columna year (year/time_period/etc.).")

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


def year_split(df: pd.DataFrame, train_end: int, val_end: int, test_end: int) -> Dict[str, np.ndarray]:
    years = df["year"].values
    train_idx = np.where(years <= train_end)[0]
    val_idx   = np.where((years > train_end) & (years <= val_end))[0]
    test_idx  = np.where((years > val_end) & (years <= test_end))[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def plot_and_save(fig_path: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


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


def load_optional_panel(paths: DataPaths, name: str, candidates: List[str],
                        reader: str = "csv") -> Optional[pd.DataFrame]:
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
        if reader == "csv":
            df = safe_read_csv(p)
        elif reader == "excel":
            df = safe_read_excel(p)
        else:
            raise ValueError("reader must be csv or excel")
        df = standardize_panel(df)
        return df
    except Exception as e:
        warn(f"[SKIP] Error cargando {name} ({p}): {e}")
        return None


def detect_spar_outcome_col(df_spar: pd.DataFrame) -> str:
    for c in ["spar_cap_law","spar_score","spar_law","y","spar_capability_law"]:
        if c in df_spar.columns:
            return c
    numeric_cols = [c for c in df_spar.columns if c not in ["iso3c","year"]]
    if not numeric_cols:
        raise ValueError("SPAR: no encontré columna outcome.")
    return numeric_cols[0]


def build_master_panel(paths: DataPaths, latam_iso3: List[str]) -> pd.DataFrame:
    df_spar = load_optional_panel(paths, "SPAR", ["spar_clean_latam.csv", "spar_clean_latam_CORE.csv", "spar_clean_latam_panel.csv"])
    if df_spar is None:
        raise FileNotFoundError("Necesito SPAR para correr el pipeline.")
    df_spar = filter_latam(df_spar, latam_iso3)
    outcome_col = detect_spar_outcome_col(df_spar)
    df_spar = df_spar.rename(columns={outcome_col: "spar_cap_law"})
    df_spar = coerce_numeric(df_spar, ["spar_cap_law"])

    df_pol   = load_optional_panel(paths, "Political panel", ["political_panel_latam_2010_2023.csv","political_panel_latam_2010_2020.csv"])
    df_vdem  = load_optional_panel(paths, "V-Dem subset", ["vdem_model1_subset_latam.csv","vdem_subset_clean_latam.csv"])
    df_wgi   = load_optional_panel(paths, "WGI", ["wgi_clean_latam.csv","vgi_capa2_latam_2000_2024.csv"])
    df_gdp   = load_optional_panel(paths, "GDP pc", ["gdp_pc_wb_latam_2000_2024.csv","gdp_pc_clean_latam.csv"])
    df_health= load_optional_panel(paths, "Health exp", ["health_gdp_wb_latam_2000_2024.csv","health_exp_clean_latam.csv"])
    df_uhc   = load_optional_panel(paths, "UHC", ["uhc_latam_2010_2023.csv","uhc_clean_latam.csv"])
    df_covid = load_optional_panel(paths, "COVID deaths last12m", ["covid_deaths_last12m_clean_latam.csv"])
    df_gpd   = load_optional_panel(paths, "Populism (GPD expanded)", ["populism_gpd_term_expanded_clean_latam.csv"])
    df_vfp   = load_optional_panel(paths, "Votes4Populists", ["votes4populists_clean_latam.csv"])
    df_fd    = load_optional_panel(paths, "Fiscal decentralization", ["fiscal_decentralization_clean_latam.csv"])
    df_dah   = load_optional_panel(paths, "IHME DAH total", ["ihme_dah_total_clean_latam.csv"])
    df_wha   = load_optional_panel(paths, "WHO participation", ["who_participation_clean_latam.csv"])
    df_excl  = load_optional_panel(paths, "WHO exclusions", ["who_exclusions_clean_latam.csv"])

    panel = df_spar.copy()
    for d in [df_pol, df_vdem, df_wgi, df_gdp, df_health, df_uhc, df_covid, df_gpd, df_vfp, df_fd, df_dah, df_wha, df_excl]:
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


def select_feature_columns(df: pd.DataFrame,
                           drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in ["entity","country","country_name","recipient_country"] if c in X.columns], errors="ignore")
    categorical = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, numeric_cols),
                              ("cat", cat_pipe, categorical_cols)],
                             remainder="drop")


def data_quality_report(panel: pd.DataFrame, out_dirs: DataPaths) -> None:
    report_path = os.path.join(out_dirs.out_dir, "data_quality_report.md")
    miss = panel.isna().mean().sort_values(ascending=False)
    lines = []
    lines.append("# Data quality report\n")
    lines.append(f"- Rows: {len(panel)}\n")
    lines.append(f"- Countries: {panel['iso3c'].nunique()}\n")
    lines.append(f"- Years: {int(panel['year'].min())}–{int(panel['year'].max())}\n")
    lines.append("\n## Missingness (top 30)\n")
    lines.append(miss.head(30).to_string())
    lines.append("\n\n## Duplicates check\n")
    dup = panel.duplicated(subset=["iso3c","year"]).sum()
    lines.append(f"- Duplicates iso3c-year: {dup}\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def fit_and_eval_regression(panel: pd.DataFrame, out_dirs: DataPaths, args: argparse.Namespace) -> dict:
    task_dir = os.path.join(out_dirs.tables_dir, "level_regression")
    fig_dir  = os.path.join(out_dirs.fig_dir, "level_regression")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["spar_tplus1"]).copy()

    drop_cols = ["spar_tplus1","delta_tplus1","spar_delta","spar_cap_law"]
    numeric_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(numeric_cols, cat_cols)

    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    y = d["spar_tplus1"].astype(float).values
    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)

    X_train, y_train = X.iloc[split["train"]], y[split["train"]]
    X_val,   y_val   = X.iloc[split["val"]],   y[split["val"]]
    X_test,  y_test  = X.iloc[split["test"]],  y[split["test"]]

    models = {
        "ridge": Ridge(alpha=1.0, random_state=0),
        "gbr": GradientBoostingRegressor(random_state=0),
        "rf": RandomForestRegressor(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "etr": ExtraTreesRegressor(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "hgb": HistGradientBoostingRegressor(random_state=0, max_depth=6, learning_rate=0.06),
    }

    results = {}
    pred_rows = []

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        pred_val = pipe.predict(X_val) if len(X_val) else np.array([])
        pred_test = pipe.predict(X_test) if len(X_test) else np.array([])

        metrics = {
            "val_mae": float(mean_absolute_error(y_val, pred_val)) if len(X_val) else None,
            "val_rmse": float(rmse(y_val, pred_val)) if len(X_val) else None,
            "val_r2": float(r2_score(y_val, pred_val)) if len(X_val) else None,
            "test_mae": float(mean_absolute_error(y_test, pred_test)) if len(X_test) else None,
            "test_rmse": float(rmse(y_test, pred_test)) if len(X_test) else None,
            "test_r2": float(r2_score(y_test, pred_test)) if len(X_test) else None,
        }
        results[name] = metrics
        joblib.dump(pipe, os.path.join(out_dirs.model_dir, f"level_{name}.joblib"))

        if len(X_test):
            tmp = d.iloc[split["test"]][["iso3c","year","spar_cap_law","spar_tplus1"]].copy()
            tmp["model"] = name
            tmp["pred_spar_tplus1"] = pred_test
            pred_rows.append(tmp)

            plt.figure()
            plt.scatter(y_test, pred_test, s=10)
            plt.xlabel("Actual spar_t+1")
            plt.ylabel("Predicted spar_t+1")
            plt.title(f"Level regression — {name}")
            plot_and_save(os.path.join(fig_dir, f"pred_vs_actual_{name}.png"))

    if pred_rows:
        save_df(pd.concat(pred_rows, ignore_index=True), os.path.join(task_dir, "predictions_test_level.csv"))

    # best by val_rmse
    best = min([k for k in results.keys() if results[k]["val_rmse"] is not None],
               key=lambda k: results[k]["val_rmse"],
               default=None)

    save_json({"models": results, "best": best}, os.path.join(task_dir, "metrics_level_models.json"))

    # Explain best model
    if best and len(X_test):
        pipe = joblib.load(os.path.join(out_dirs.model_dir, f"level_{best}.joblib"))
        X_explain = X_test.sample(min(args.shap_max_rows, len(X_test)), random_state=0)
        explain_dir = os.path.join(task_dir, "explain")
        ensure_dir(explain_dir)

        if not args.no_shap:
            try:
                import shap  # type: ignore
                Xt = pipe.named_steps["pre"].transform(X_explain)
                feature_names = pipe.named_steps["pre"].get_feature_names_out()
                model = pipe.named_steps["model"]
                explainer = shap.Explainer(model, Xt, feature_names=feature_names)
                sv = explainer(Xt)
                mean_abs = np.abs(sv.values).mean(axis=0)
                imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                save_df(imp, os.path.join(explain_dir, f"shap_importance_{best}.csv"))

                plt.figure()
                top = imp.head(25).iloc[::-1]
                plt.barh(top["feature"], top["mean_abs_shap"])
                plt.title(f"SHAP importance (top 25) — {best}")
                plt.xlabel("Mean |SHAP|")
                plot_and_save(os.path.join(fig_dir, f"shap_bar_{best}.png"))
            except Exception as e:
                warn(f"SHAP falló ({e}). Usando permutation importance.")
                pi = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1)
                feature_names = pipe.named_steps["pre"].get_feature_names_out()
                imp = pd.DataFrame({"feature": feature_names, "importance_mean": pi.importances_mean}).sort_values("importance_mean", ascending=False)
                save_df(imp, os.path.join(explain_dir, f"perm_importance_{best}.csv"))
        else:
            pi = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1)
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            imp = pd.DataFrame({"feature": feature_names, "importance_mean": pi.importances_mean}).sort_values("importance_mean", ascending=False)
            save_df(imp, os.path.join(explain_dir, f"perm_importance_{best}.csv"))

    return {"best": best, "results": results}


def fit_and_eval_event(panel: pd.DataFrame, out_dirs: DataPaths, args: argparse.Namespace) -> dict:
    task_dir = os.path.join(out_dirs.tables_dir, "event_classification")
    fig_dir  = os.path.join(out_dirs.fig_dir, "event_classification")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["delta_tplus1"]).copy()
    d["y_event"] = (d["delta_tplus1"] >= args.event_threshold).astype(int)

    drop_cols = ["y_event","spar_tplus1","delta_tplus1","spar_delta","spar_cap_law"]
    numeric_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(numeric_cols, cat_cols)

    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    y = d["y_event"].values
    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)

    X_train, y_train = X.iloc[split["train"]], y[split["train"]]
    X_val, y_val = X.iloc[split["val"]], y[split["val"]]
    X_test, y_test = X.iloc[split["test"]], y[split["test"]]

    models = {
        "logit": LogisticRegression(max_iter=4000, n_jobs=-1),
        "gbc": GradientBoostingClassifier(random_state=0),
        "rf": RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "etc": ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "hgb": HistGradientBoostingClassifier(random_state=0, max_depth=6, learning_rate=0.06),
    }

    def eval_probs(y_true, p):
        if len(y_true) == 0:
            return {}
        p = np.clip(p, 1e-6, 1-1e-6)
        yhat = (p >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        return {
            "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
            "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "brier": float(brier_score_loss(y_true, p))
        }

    results = {}
    preds = []
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:,1] if len(X_val) else np.array([])
        p_test = pipe.predict_proba(X_test)[:,1] if len(X_test) else np.array([])
        results[name] = {"val": eval_probs(y_val, p_val), "test": eval_probs(y_test, p_test)}
        joblib.dump(pipe, os.path.join(out_dirs.model_dir, f"event_{name}.joblib"))

        if len(X_test):
            tmp = d.iloc[split["test"]][["iso3c","year","spar_cap_law","delta_tplus1","y_event"]].copy()
            tmp["model"] = name
            tmp["pred_event_prob"] = p_test
            preds.append(tmp)

            if len(np.unique(y_test)) > 1:
                from sklearn.metrics import roc_curve, precision_recall_curve
                fpr, tpr, _ = roc_curve(y_test, p_test)
                prec, rec, _ = precision_recall_curve(y_test, p_test)

                plt.figure(); plt.plot(fpr, tpr)
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name}")
                plot_and_save(os.path.join(fig_dir, f"roc_{name}.png"))

                plt.figure(); plt.plot(rec, prec)
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name}")
                plot_and_save(os.path.join(fig_dir, f"pr_{name}.png"))

                frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10)
                plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
                plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction positives")
                plt.title(f"Calibration — {name}")
                plot_and_save(os.path.join(fig_dir, f"calibration_{name}.png"))

    if preds:
        save_df(pd.concat(preds, ignore_index=True), os.path.join(task_dir, "predictions_test_event.csv"))

    # best by val pr_auc
    best = None
    best_score = -np.inf
    for name, r in results.items():
        pr = r["val"].get("pr_auc")
        if pr is not None and pr > best_score:
            best, best_score = name, pr

    save_json({"models": results, "best": best}, os.path.join(task_dir, "metrics_event_models.json"))
    return {"best": best, "results": results}


def fit_survival(panel: pd.DataFrame, out_dirs: DataPaths, args: argparse.Namespace) -> dict:
    task_dir = os.path.join(out_dirs.tables_dir, "survival_time_to_threshold")
    fig_dir  = os.path.join(out_dirs.fig_dir, "survival_time_to_threshold")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["spar_cap_law","spar_tplus1"]).copy()
    d = d[d["spar_cap_law"] < args.level_threshold].copy()
    d["event_next"] = (d["spar_tplus1"] >= args.level_threshold).astype(int)

    if len(d) < 100:
        warn("Survival: dataset muy chico. Saltando.")
        return {"skipped": True, "n": int(len(d))}

    drop_cols = ["event_next","spar_tplus1","delta_tplus1","spar_delta","spar_cap_law"]
    numeric_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(numeric_cols, cat_cols)

    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    y = d["event_next"].values
    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)

    model = HistGradientBoostingClassifier(random_state=0, max_depth=5, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X.iloc[split["train"]], y[split["train"]])
    joblib.dump(pipe, os.path.join(out_dirs.model_dir, "survival_hazard_hgb.joblib"))

    if len(split["test"]):
        p = pipe.predict_proba(X.iloc[split["test"]])[:,1]
        y_test = y[split["test"]]
        res = {
            "test_roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else None,
            "test_pr_auc": float(average_precision_score(y_test, p)) if len(np.unique(y_test)) > 1 else None,
            "test_brier": float(brier_score_loss(y_test, p))
        }
        tmp = d.iloc[split["test"]][["iso3c","year","spar_cap_law","spar_tplus1","event_next"]].copy()
        tmp["pred_hazard"] = p
        save_df(tmp, os.path.join(task_dir, "predictions_test_hazard.csv"))

        if len(np.unique(y_test)) > 1:
            frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
            plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
            plt.xlabel("Mean predicted hazard"); plt.ylabel("Fraction events")
            plt.title("Hazard calibration (test)")
            plot_and_save(os.path.join(fig_dir, "hazard_calibration.png"))
    else:
        res = {"skipped_test": True}

    save_json(res, os.path.join(task_dir, "metrics_survival.json"))
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, required=True)
    p.add_argument("--train_end_year", type=int, default=2018)
    p.add_argument("--val_end_year", type=int, default=2019)
    p.add_argument("--test_end_year", type=int, default=2023)
    p.add_argument("--event_threshold", type=float, default=2.0)
    p.add_argument("--level_threshold", type=float, default=80.0)
    p.add_argument("--latam_iso3", type=str, default=",".join(LATAM_ISO3_DEFAULT))
    p.add_argument("--shap_max_rows", type=int, default=2000)
    p.add_argument("--no_shap", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    latam = [x.strip().upper() for x in args.latam_iso3.split(",") if x.strip()]
    paths = make_paths(args.base_path)

    panel = build_master_panel(paths, latam)
    save_df(panel, os.path.join(paths.tables_dir, "panel_master_snapshot.csv"))
    data_quality_report(panel, paths)

    save_json({
        "train_end_year": args.train_end_year,
        "val_end_year": args.val_end_year,
        "test_end_year": args.test_end_year,
        "event_threshold": args.event_threshold,
        "level_threshold": args.level_threshold
    }, os.path.join(paths.out_dir, "run_config.json"))

    log("[TASK] Level regression")
    level = fit_and_eval_regression(panel, paths, args)

    log("[TASK] Event classification")
    event = fit_and_eval_event(panel, paths, args)

    log("[TASK] Survival/hazard")
    surv = fit_survival(panel, paths, args)

    save_json({"level": level, "event": event, "survival": surv},
              os.path.join(paths.out_dir, "summary.json"))
    log("[DONE] Outputs en 05_outputs/")


if __name__ == "__main__":
    main()
