#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor ML LATAM — Plug & Play (v2)
====================================
v2 añade:
- Rolling backtest también para clasificación (evento) y hazard (survival discreto).
- Intervalos de predicción para nivel (SPAR t+1) vía cuantiles de residuales (aprox robusta).
- Módulo de forecast multi-año (2024–2030 por defecto) con escenarios:
    - baseline: features carry-forward + covid=0 si existe
    - custom: se aporta un CSV iso3c-year con overrides de features.

Ejemplo:
    python predictor_ml_pipeline_v2.py --base_path "/Users/.../predictor_ml_latam" \
      --forecast_start 2024 --forecast_end 2030 --scenario_mode baseline

Estructura esperada (pero tolerante):
    base_path/
      02_clean_data/   (preferido)
      01_raw_data/     (fallback)
      05_outputs/      (se crea)

Autor: Adela Santos
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

# -----------------------------
# Utils
# -----------------------------
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

def year_split(df: pd.DataFrame, train_end: int, val_end: int, test_end: int) -> Dict[str, np.ndarray]:
    years = df["year"].values
    train_idx = np.where(years <= train_end)[0]
    val_idx   = np.where((years > train_end) & (years <= val_end))[0]
    test_idx  = np.where((years > val_end) & (years <= test_end))[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}

def detect_outcome_col(df_spar: pd.DataFrame) -> str:
    for c in ["spar_cap_law","spar_score","spar_law","y","spar_capability_law"]:
        if c in df_spar.columns:
            return c
    numeric_cols = [c for c in df_spar.columns if c not in ["iso3c","year"]]
    if not numeric_cols:
        raise ValueError("SPAR: no encontré columna outcome.")
    return numeric_cols[0]

def select_feature_columns(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in ["entity","country","country_name","recipient_country"] if c in X.columns], errors="ignore")
    categorical = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical

def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop"
    )


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
        warn(f"[SKIP] Error cargando {name} ({p}): {e}")
        return None

def build_master_panel(paths: DataPaths, latam_iso3: List[str]) -> pd.DataFrame:
    df_spar = load_optional_panel(paths, "SPAR", ["spar_clean_latam.csv", "spar_clean_latam_panel.csv"])
    if df_spar is None:
        raise FileNotFoundError("Necesito SPAR para correr el pipeline.")
    df_spar = filter_latam(df_spar, latam_iso3)
    outcol = detect_outcome_col(df_spar)
    df_spar = df_spar.rename(columns={outcol: "spar_cap_law"})
    df_spar = coerce_numeric(df_spar, ["spar_cap_law"])

    # Optional features
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
        panel = panel.merge(filter_latam(d, latam_iso3), on=["iso3c","year"], how="left")

    panel = panel.sort_values(["iso3c","year"]).reset_index(drop=True)

    # Derived vars
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
    n = len(panel)
    years = (int(panel["year"].min()), int(panel["year"].max()))
    countries = panel["iso3c"].nunique()
    miss = panel.isna().mean().sort_values(ascending=False)

    lines = []
    lines.append("# Data quality report\n")
    lines.append(f"- Rows: {n}")
    lines.append(f"- Countries: {countries}")
    lines.append(f"- Years: {years[0]}–{years[1]}\n")
    lines.append("## Missingness (top 30)\n")
    lines.append(miss.head(30).to_string())
    lines.append("\n## Duplicates check\n")
    dup = panel.duplicated(subset=["iso3c","year"]).sum()
    lines.append(f"- Duplicates iso3c-year: {dup}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Robustness helpers
# -----------------------------
def rolling_backtest_years(df: pd.DataFrame, min_train_years: int = 5) -> List[Tuple[int, int]]:
    years = sorted(df["year"].unique())
    out = []
    for i in range(min_train_years, len(years)-1):
        train_end = years[i-1]
        test_year = years[i]
        out.append((train_end, test_year))
    return out


# -----------------------------
# Task 1: Level regression (+ PI)
# -----------------------------
def fit_level_models(panel: pd.DataFrame, paths: DataPaths, args) -> Dict[str, dict]:
    task_dir = os.path.join(paths.tables_dir, "level_regression")
    fig_dir  = os.path.join(paths.fig_dir, "level_regression")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["spar_tplus1"]).copy()

    drop_cols = ["spar_tplus1","delta_tplus1","spar_delta","spar_cap_law"]
    num_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(num_cols, cat_cols)

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

        pred_val  = pipe.predict(X_val) if len(X_val) else np.array([])
        pred_test = pipe.predict(X_test) if len(X_test) else np.array([])

        metrics = {
            "val_rmse": rmse(y_val, pred_val) if len(X_val) else None,
            "val_mae": float(mean_absolute_error(y_val, pred_val)) if len(X_val) else None,
            "val_r2": float(r2_score(y_val, pred_val)) if len(X_val) else None,
            "test_rmse": rmse(y_test, pred_test) if len(X_test) else None,
            "test_mae": float(mean_absolute_error(y_test, pred_test)) if len(X_test) else None,
            "test_r2": float(r2_score(y_test, pred_test)) if len(X_test) else None,
        }
        results[name] = {"metrics": metrics}
        model_path = os.path.join(paths.model_dir, f"level_{name}.joblib")
        joblib.dump(pipe, model_path)
        results[name]["model_path"] = model_path

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
        preds = pd.concat(pred_rows, ignore_index=True)
        save_df(preds, os.path.join(task_dir, "predictions_test_level.csv"))

    # select best by val_rmse
    best = None
    best_rmse = np.inf
    for name, r in results.items():
        v = r["metrics"]["val_rmse"]
        if v is not None and v < best_rmse:
            best, best_rmse = name, v
    results["best_model"] = {"name": best, "val_rmse": float(best_rmse) if np.isfinite(best_rmse) else None}
    save_json(results, os.path.join(task_dir, "metrics_level_models.json"))

    # Explain best + prediction intervals
    if best is not None and len(X_test):
        best_pipe = joblib.load(results[best]["model_path"])

        explain_dir = os.path.join(task_dir, "explain")
        ensure_dir(explain_dir)
        ran_shap = False
        if not args.no_shap:
            try:
                import shap  # type: ignore
                Xs = X_test.sample(min(args.shap_max_rows, len(X_test)), random_state=0)
                Xt = best_pipe.named_steps["pre"].transform(Xs)
                feature_names = best_pipe.named_steps["pre"].get_feature_names_out()
                model = best_pipe.named_steps["model"]
                explainer = shap.Explainer(model, Xt, feature_names=feature_names)
                shap_vals = explainer(Xt)
                mean_abs = np.abs(shap_vals.values).mean(axis=0)
                imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                save_df(imp, os.path.join(explain_dir, f"shap_importance_{best}.csv"))

                plt.figure()
                top = imp.head(25).iloc[::-1]
                plt.barh(top["feature"], top["mean_abs_shap"])
                plt.title(f"SHAP importance (top 25) — {best}")
                plt.xlabel("Mean |SHAP|")
                plot_and_save(os.path.join(fig_dir, f"shap_bar_{best}.png"))
                ran_shap = True
            except Exception as e:
                warn(f"SHAP falló ({e}). Usaré permutation importance.")

        if not ran_shap:
            pi = permutation_importance(best_pipe, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1)
            feature_names = best_pipe.named_steps["pre"].get_feature_names_out()
            imp = pd.DataFrame({"feature": feature_names, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
            imp = imp.sort_values("importance_mean", ascending=False)
            save_df(imp, os.path.join(explain_dir, f"perm_importance_{best}.csv"))

            plt.figure()
            top = imp.head(25).iloc[::-1]
            plt.barh(top["feature"], top["importance_mean"])
            plt.title(f"Permutation importance (top 25) — {best}")
            plt.xlabel("Mean importance (Δ score)")
            plot_and_save(os.path.join(fig_dir, f"perm_importance_bar_{best}.png"))

        # Prediction intervals (approx): use residual quantiles from val (or train)
        preds_test = best_pipe.predict(X_test)
        resid_base = (y_val - best_pipe.predict(X_val)) if len(X_val) else (y_train - best_pipe.predict(X_train))
        q10, q90 = np.quantile(resid_base, 0.10), np.quantile(resid_base, 0.90)

        pi_df = d.iloc[split["test"]][["iso3c","year","spar_tplus1"]].copy()
        pi_df["pred_mean"] = preds_test
        pi_df["pi_p10"] = preds_test + q10
        pi_df["pi_p90"] = preds_test + q90
        save_df(pi_df, os.path.join(task_dir, "prediction_intervals_test.csv"))

        # Plot: yearly mean with PI band
        g = pi_df.groupby("year", as_index=False).agg(
            actual_mean=("spar_tplus1","mean"),
            pred_mean=("pred_mean","mean"),
            p10=("pi_p10","mean"),
            p90=("pi_p90","mean"),
        )
        plt.figure()
        plt.plot(g["year"], g["actual_mean"], label="Actual mean")
        plt.plot(g["year"], g["pred_mean"], label="Pred mean")
        plt.fill_between(g["year"], g["p10"], g["p90"], alpha=0.2, label="PI (approx)")
        plt.title("Level t+1 — mean by year with prediction interval")
        plt.xlabel("Year")
        plt.ylabel("SPAR t+1")
        plt.legend()
        plot_and_save(os.path.join(fig_dir, "level_yearly_mean_with_pi.png"))

    if not args.no_robustness:
        robust_level(d, paths, pre, X, y, args)

    return results

def robust_level(d: pd.DataFrame, paths: DataPaths, pre: ColumnTransformer, X: pd.DataFrame, y: np.ndarray, args) -> None:
    task_dir = os.path.join(paths.tables_dir, "level_regression")
    fig_dir  = os.path.join(paths.fig_dir, "level_regression")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    model = HistGradientBoostingRegressor(random_state=0, max_depth=6, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])

    # Rolling backtest
    rows=[]
    for train_end, test_year in rolling_backtest_years(d, min_train_years=5):
        tr = d["year"] <= train_end
        te = d["year"] == test_year
        if tr.sum() < 50 or te.sum() < 10:
            continue
        pipe.fit(X.loc[tr], y[tr])
        pred = pipe.predict(X.loc[te])
        rows.append({"train_end": train_end, "test_year": test_year,
                     "rmse": rmse(y[te], pred),
                     "mae": float(mean_absolute_error(y[te], pred)),
                     "r2": float(r2_score(y[te], pred))})
    roll = pd.DataFrame(rows)
    if len(roll):
        save_df(roll, os.path.join(task_dir, "robust_rolling_backtest.csv"))
        plt.figure()
        plt.plot(roll["test_year"], roll["rmse"])
        plt.xlabel("Test year"); plt.ylabel("RMSE")
        plt.title("Rolling backtest RMSE (level)")
        plot_and_save(os.path.join(fig_dir, "rolling_rmse.png"))

    # GroupKFold by country
    groups = d["iso3c"].values
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    g_rows=[]
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        g_rows.append({"fold": fold, "rmse": rmse(y[te], pred),
                       "mae": float(mean_absolute_error(y[te], pred)),
                       "r2": float(r2_score(y[te], pred))})
    gdf=pd.DataFrame(g_rows)
    if len(gdf):
        save_df(gdf, os.path.join(task_dir, "robust_groupkfold_country.csv"))

    # Placebo shuffle within-year
    rng = np.random.default_rng(0)
    y_p = y.copy()
    for yr in d["year"].unique():
        idx = np.where(d["year"].values==yr)[0]
        rng.shuffle(y_p[idx])
    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)
    pipe.fit(X.iloc[split["train"]], y_p[split["train"]])
    pred = pipe.predict(X.iloc[split["test"]]) if len(split["test"]) else np.array([])
    placebo_rmse = rmse(y_p[split["test"]], pred) if len(pred) else None
    save_json({"placebo_test_rmse": placebo_rmse}, os.path.join(task_dir, "robust_placebo.json"))


# -----------------------------
# Task 2: Event classification (+ rolling backtest)
# -----------------------------
def fit_event_models(panel: pd.DataFrame, paths: DataPaths, args) -> Dict[str, dict]:
    task_dir = os.path.join(paths.tables_dir, "event_classification")
    fig_dir  = os.path.join(paths.fig_dir, "event_classification")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = panel.dropna(subset=["delta_tplus1"]).copy()
    d["y_event"] = (d["delta_tplus1"] >= args.event_threshold).astype(int)

    drop_cols = ["spar_tplus1","delta_tplus1","spar_delta","spar_cap_law","y_event"]
    num_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(num_cols, cat_cols)

    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    y = d["y_event"].values
    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)
    X_train, y_train = X.iloc[split["train"]], y[split["train"]]
    X_val,   y_val   = X.iloc[split["val"]],   y[split["val"]]
    X_test,  y_test  = X.iloc[split["test"]],  y[split["test"]]

    models = {
        "logit": LogisticRegression(max_iter=4000, n_jobs=-1),
        "gbc": GradientBoostingClassifier(random_state=0),
        "rf": RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "etc": ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2),
        "hgb": HistGradientBoostingClassifier(random_state=0, max_depth=6, learning_rate=0.06),
    }

    def probs(pipe, X_):
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            return pipe.predict_proba(X_)[:, 1]
        s = pipe.decision_function(X_)
        return 1/(1+np.exp(-s))

    def eval_probs(y_true, y_prob) -> dict:
        y_prob = np.clip(y_prob, 1e-6, 1-1e-6)
        y_hat = (y_prob >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
            "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "brier": float(brier_score_loss(y_true, y_prob))
        }

    results={}
    pred_rows=[]
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        p_val = probs(pipe, X_val) if len(X_val) else np.array([])
        p_test= probs(pipe, X_test) if len(X_test) else np.array([])

        results[name] = {"metrics": {
            "val": eval_probs(y_val, p_val) if len(X_val) else None,
            "test": eval_probs(y_test, p_test) if len(X_test) else None,
            "event_rate_train": float(np.mean(y_train)) if len(y_train) else None
        }}
        mp = os.path.join(paths.model_dir, f"event_{name}.joblib")
        joblib.dump(pipe, mp)
        results[name]["model_path"]=mp

        if len(X_test):
            tmp = d.iloc[split["test"]][["iso3c","year","spar_cap_law","delta_tplus1","y_event"]].copy()
            tmp["model"]=name
            tmp["pred_event_prob"]=p_test
            pred_rows.append(tmp)

            if len(np.unique(y_test))>1:
                from sklearn.metrics import roc_curve, precision_recall_curve
                fpr, tpr, _ = roc_curve(y_test, p_test)
                prec, rec, _ = precision_recall_curve(y_test, p_test)
                plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name}")
                plot_and_save(os.path.join(fig_dir, f"roc_{name}.png"))
                plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name}")
                plot_and_save(os.path.join(fig_dir, f"pr_{name}.png"))
                frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10)
                plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
                plt.xlabel("Mean predicted"); plt.ylabel("Frac positives"); plt.title(f"Calibration — {name}")
                plot_and_save(os.path.join(fig_dir, f"calibration_{name}.png"))

    if pred_rows:
        save_df(pd.concat(pred_rows, ignore_index=True), os.path.join(task_dir, "predictions_test_event.csv"))

    # best by val pr_auc
    best=None; best_pr=-np.inf
    for name, r in results.items():
        m = r["metrics"]["val"]
        if m and m["pr_auc"] is not None and m["pr_auc"]>best_pr:
            best=name; best_pr=m["pr_auc"]
    results["best_model"]={"name": best, "val_pr_auc": float(best_pr) if np.isfinite(best_pr) else None}
    save_json(results, os.path.join(task_dir, "metrics_event_models.json"))

    if not args.no_robustness:
        robust_event(d, paths, pre, X, y, args)

    return results

def robust_event(d: pd.DataFrame, paths: DataPaths, pre: ColumnTransformer, X: pd.DataFrame, y: np.ndarray, args) -> None:
    task_dir = os.path.join(paths.tables_dir, "event_classification")
    fig_dir  = os.path.join(paths.fig_dir, "event_classification")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    model = HistGradientBoostingClassifier(random_state=0, max_depth=6, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])

    # Rolling backtest
    rows=[]
    for train_end, test_year in rolling_backtest_years(d, min_train_years=5):
        tr = d["year"] <= train_end
        te = d["year"] == test_year
        if tr.sum() < 50 or te.sum() < 10:
            continue
        pipe.fit(X.loc[tr], y[tr])
        p = pipe.predict_proba(X.loc[te])[:, 1]
        rows.append({
            "train_end": train_end, "test_year": test_year,
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "pr_auc": float(average_precision_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "brier": float(brier_score_loss(y[te], p))
        })
    roll=pd.DataFrame(rows)
    if len(roll):
        save_df(roll, os.path.join(task_dir, "robust_rolling_backtest.csv"))
        plt.figure(); plt.plot(roll["test_year"], roll["pr_auc"])
        plt.xlabel("Test year"); plt.ylabel("PR-AUC")
        plt.title("Rolling backtest PR-AUC (event)")
        plot_and_save(os.path.join(fig_dir, "rolling_pr_auc.png"))

    # GroupKFold
    groups=d["iso3c"].values
    gkf=GroupKFold(n_splits=min(5, len(np.unique(groups))))
    g_rows=[]
    for fold,(tr,te) in enumerate(gkf.split(X,y,groups=groups), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        p=pipe.predict_proba(X.iloc[te])[:,1]
        g_rows.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "pr_auc": float(average_precision_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "brier": float(brier_score_loss(y[te], p))
        })
    gdf=pd.DataFrame(g_rows)
    if len(gdf):
        save_df(gdf, os.path.join(task_dir, "robust_groupkfold_country.csv"))

    # Placebo shuffle within-year
    rng=np.random.default_rng(0)
    y_p=y.copy()
    for yr in d["year"].unique():
        idx=np.where(d["year"].values==yr)[0]
        rng.shuffle(y_p[idx])
    split=year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)
    pipe.fit(X.iloc[split["train"]], y_p[split["train"]])
    p=pipe.predict_proba(X.iloc[split["test"]])[:,1] if len(split["test"]) else np.array([])
    placebo={
        "placebo_roc_auc": float(roc_auc_score(y_p[split["test"]], p)) if len(p) and len(np.unique(y_p[split["test"]]))>1 else None,
        "placebo_pr_auc": float(average_precision_score(y_p[split["test"]], p)) if len(p) and len(np.unique(y_p[split["test"]]))>1 else None,
    }
    save_json(placebo, os.path.join(task_dir, "robust_placebo.json"))


# -----------------------------
# Task 3: Discrete-time hazard
# -----------------------------
def build_survival_dataset(panel: pd.DataFrame, level_threshold: float) -> pd.DataFrame:
    d = panel.dropna(subset=["spar_cap_law", "spar_tplus1"]).copy()
    d = d[d["spar_cap_law"] < level_threshold].copy()
    d["event_next"] = (d["spar_tplus1"] >= level_threshold).astype(int)
    return d

def fit_survival(panel: pd.DataFrame, paths: DataPaths, args) -> Dict[str, dict]:
    task_dir = os.path.join(paths.tables_dir, "survival_time_to_threshold")
    fig_dir  = os.path.join(paths.fig_dir, "survival_time_to_threshold")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    d = build_survival_dataset(panel, args.level_threshold)
    if len(d) < 100:
        warn("Survival: dataset muy chico. Saltando.")
        return {"skipped": True, "n": int(len(d))}

    drop_cols = ["event_next","spar_tplus1","delta_tplus1","spar_delta","spar_cap_law"]
    num_cols, cat_cols = select_feature_columns(d, drop_cols + ["iso3c","year"])
    pre = make_preprocessor(num_cols, cat_cols)

    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    y = d["event_next"].values

    split = year_split(d, args.train_end_year, args.val_end_year, args.test_end_year)
    X_train, y_train = X.iloc[split["train"]], y[split["train"]]
    X_val,   y_val   = X.iloc[split["val"]],   y[split["val"]]
    X_test,  y_test  = X.iloc[split["test"]],  y[split["test"]]

    model = HistGradientBoostingClassifier(random_state=0, max_depth=5, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    p_val  = pipe.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    p_test = pipe.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])

    def eval_probs(y_true, y_prob):
        if len(y_true)==0: return {}
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else None,
            "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true))>1 else None,
            "brier": float(brier_score_loss(y_true, y_prob)),
        }

    res={"metrics":{
        "val": eval_probs(y_val, p_val),
        "test": eval_probs(y_test, p_test),
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test))
    }}
    mp=os.path.join(paths.model_dir, "survival_hazard_hgb.joblib")
    joblib.dump(pipe, mp)
    res["model_path"]=mp
    save_json(res, os.path.join(task_dir, "metrics_survival.json"))

    if len(X_test):
        tmp=d.iloc[split["test"]][["iso3c","year","spar_cap_law","spar_tplus1","event_next"]].copy()
        tmp["pred_hazard"]=p_test
        save_df(tmp, os.path.join(task_dir, "predictions_test_hazard.csv"))
        if len(np.unique(y_test))>1:
            frac_pos, mean_pred=calibration_curve(y_test, p_test, n_bins=10)
            plt.figure(); plt.plot(mean_pred, frac_pos, marker="o")
            plt.xlabel("Mean predicted hazard"); plt.ylabel("Fraction events")
            plt.title("Hazard calibration (test)")
            plot_and_save(os.path.join(fig_dir, "hazard_calibration.png"))

    if not args.no_robustness:
        robust_survival(d, paths, pre, X, y, args)

    return res

def robust_survival(d: pd.DataFrame, paths: DataPaths, pre: ColumnTransformer, X: pd.DataFrame, y: np.ndarray, args) -> None:
    task_dir = os.path.join(paths.tables_dir, "survival_time_to_threshold")
    fig_dir  = os.path.join(paths.fig_dir, "survival_time_to_threshold")
    ensure_dir(task_dir); ensure_dir(fig_dir)

    model = HistGradientBoostingClassifier(random_state=0, max_depth=5, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])

    rows=[]
    for train_end, test_year in rolling_backtest_years(d, min_train_years=5):
        tr = d["year"] <= train_end
        te = d["year"] == test_year
        if tr.sum() < 50 or te.sum() < 10:
            continue
        pipe.fit(X.loc[tr], y[tr])
        p = pipe.predict_proba(X.loc[te])[:, 1]
        rows.append({
            "train_end": train_end, "test_year": test_year,
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "pr_auc": float(average_precision_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "brier": float(brier_score_loss(y[te], p))
        })
    roll=pd.DataFrame(rows)
    if len(roll):
        save_df(roll, os.path.join(task_dir, "robust_rolling_backtest.csv"))
        plt.figure(); plt.plot(roll["test_year"], roll["pr_auc"])
        plt.xlabel("Test year"); plt.ylabel("PR-AUC")
        plt.title("Rolling backtest PR-AUC (hazard)")
        plot_and_save(os.path.join(fig_dir, "rolling_pr_auc.png"))

    groups=d["iso3c"].values
    gkf=GroupKFold(n_splits=min(5, len(np.unique(groups))))
    g_rows=[]
    for fold,(tr,te) in enumerate(gkf.split(X,y,groups=groups), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        p=pipe.predict_proba(X.iloc[te])[:,1]
        g_rows.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "pr_auc": float(average_precision_score(y[te], p)) if len(np.unique(y[te]))>1 else None,
            "brier": float(brier_score_loss(y[te], p))
        })
    gdf=pd.DataFrame(g_rows)
    if len(gdf):
        save_df(gdf, os.path.join(task_dir, "robust_groupkfold_country.csv"))


# -----------------------------
# Forecast multi-year scenarios
# -----------------------------
def forecast_multiyear(panel: pd.DataFrame, paths: DataPaths, args, level_best_model_path: str) -> Optional[pd.DataFrame]:
    if level_best_model_path is None or not os.path.exists(level_best_model_path):
        warn("Forecast: No encontré el mejor modelo de nivel. Saltando forecast.")
        return None

    out_dir = os.path.join(paths.tables_dir, "forecast")
    fig_dir = os.path.join(paths.fig_dir, "forecast")
    ensure_dir(out_dir); ensure_dir(fig_dir)

    model = joblib.load(level_best_model_path)

    last = panel.dropna(subset=["spar_cap_law"]).sort_values(["iso3c","year"]).groupby("iso3c").tail(1).copy()
    if last.empty:
        warn("Forecast: panel no tiene spar_cap_law observado. Saltando.")
        return None

    start = args.forecast_start
    end   = args.forecast_end
    years = list(range(start, end+1))

    scenario = None
    if args.scenario_mode == "custom":
        if not args.scenario_csv or not os.path.exists(args.scenario_csv):
            warn("Forecast custom: necesitas --scenario_csv con iso3c,year y columnas a sobrescribir.")
        else:
            scenario = pd.read_csv(args.scenario_csv, low_memory=False)
            scenario["iso3c"] = scenario["iso3c"].astype(str).str.upper().str.strip()
            scenario["year"] = pd.to_numeric(scenario["year"], errors="coerce").astype(int)

    forecast_rows = []
    state = {r["iso3c"]: r for _, r in last.iterrows()}

    for y in years:
        for iso in list(state.keys()):
            row = state[iso].copy()
            row["year"] = y

            if args.scenario_mode == "baseline":
                # poner covid=0 si existe
                for c in ["covid_deaths_pm_last12m", "covid_deaths_last12m_clean_latam", "covid_deaths"]:
                    if c in row.index:
                        row[c] = 0.0

            if scenario is not None:
                s = scenario[(scenario["iso3c"]==iso) & (scenario["year"]==y)]
                if len(s):
                    for col in s.columns:
                        if col in ["iso3c","year"]: 
                            continue
                        row[col] = s.iloc[0][col]

            # update lags
            row["spar_lag2"] = row.get("spar_lag1", np.nan)
            row["spar_lag1"] = row.get("spar_cap_law", np.nan)
            row["spar_delta"] = row.get("spar_lag1", np.nan) - row.get("spar_lag2", np.nan)

            row_df = pd.DataFrame([row])
            for c in ["spar_tplus1","delta_tplus1"]:
                if c in row_df.columns:
                    row_df = row_df.drop(columns=[c])

            pred = float(model.predict(row_df)[0])
            forecast_rows.append({"iso3c": iso, "year": y, "pred_spar": pred})

            row["spar_cap_law"] = pred
            state[iso] = row

    fdf = pd.DataFrame(forecast_rows)
    out_csv = os.path.join(out_dir, f"forecast_{args.scenario_mode}_{start}_{end}.csv")
    save_df(fdf, out_csv)

    plt.figure()
    for iso, g in fdf.groupby("iso3c"):
        if len(g) >= 2:
            plt.plot(g["year"], g["pred_spar"], alpha=0.6)
    plt.xlabel("Year"); plt.ylabel("Predicted SPAR (law)")
    plt.title(f"Forecast trajectories ({args.scenario_mode})")
    plot_and_save(os.path.join(fig_dir, f"forecast_trajectories_{args.scenario_mode}.png"))

    mean = fdf.groupby("year", as_index=False)["pred_spar"].mean()
    plt.figure()
    plt.plot(mean["year"], mean["pred_spar"])
    plt.xlabel("Year"); plt.ylabel("Mean predicted SPAR")
    plt.title(f"Forecast mean ({args.scenario_mode})")
    plot_and_save(os.path.join(fig_dir, f"forecast_mean_{args.scenario_mode}.png"))

    return fdf


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

    p.add_argument("--forecast_start", type=int, default=2024)
    p.add_argument("--forecast_end", type=int, default=2030)
    p.add_argument("--scenario_mode", type=str, default="baseline", choices=["baseline","custom"])
    p.add_argument("--scenario_csv", type=str, default=None)

    p.add_argument("--latam_iso3", type=str, default=",".join(LATAM_ISO3_DEFAULT))
    p.add_argument("--shap_max_rows", type=int, default=2000)
    p.add_argument("--no_shap", action="store_true")
    p.add_argument("--no_robustness", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    latam_iso3 = [x.strip().upper() for x in args.latam_iso3.split(",") if x.strip()]

    paths = make_paths(args.base_path)

    panel = build_master_panel(paths, latam_iso3=latam_iso3)
    panel_snapshot = os.path.join(paths.tables_dir, "panel_master_snapshot.csv")
    save_df(panel, panel_snapshot)
    data_quality_report(panel, paths.out_dir)

    run_cfg = {
        "train_end_year": args.train_end_year, "val_end_year": args.val_end_year, "test_end_year": args.test_end_year,
        "event_threshold": args.event_threshold, "level_threshold": args.level_threshold,
        "forecast_start": args.forecast_start, "forecast_end": args.forecast_end,
        "scenario_mode": args.scenario_mode, "scenario_csv": args.scenario_csv,
        "panel_snapshot": panel_snapshot,
    }
    save_json(run_cfg, os.path.join(paths.out_dir, "run_config.json"))

    log("[TASK] 1) Level regression (priority)")
    level_res = fit_level_models(panel, paths, args)

    log("[TASK] 2) Event classification")
    event_res = fit_event_models(panel, paths, args)

    log("[TASK] 3) Survival/hazard (time to threshold)")
    surv_res = fit_survival(panel, paths, args)

    # Forecast with best level model
    best_level = level_res.get("best_model", {}).get("name")
    best_model_path = level_res.get(best_level, {}).get("model_path") if best_level else None
    log("[TASK] 4) Forecast multi-year")
    fdf = forecast_multiyear(panel, paths, args, best_model_path)

    summary = {
        "level_best": level_res.get("best_model", {}),
        "event_best": event_res.get("best_model", {}),
        "survival": surv_res.get("metrics", {}),
        "forecast_rows": int(len(fdf)) if isinstance(fdf, pd.DataFrame) else 0
    }
    save_json(summary, os.path.join(paths.out_dir, "summary.json"))
    log("[DONE] v2 listo — outputs en 05_outputs/")
