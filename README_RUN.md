# Predictor ML LATAM — Cómo correr

## 1) Instalar dependencias
```bash
pip install -r requirements.txt
```

Si SHAP falla:
```bash
python predictor_ml_pipeline.py --base_path "/Users/adelasantos/Documents/predictor_ml_latam" --no_shap
```

## 2) Ejecutar
```bash
python predictor_ml_pipeline.py --base_path "/Users/adelasantos/Documents/predictor_ml_latam"   --train_end_year 2018 --val_end_year 2019 --test_end_year 2023   --event_threshold 2.0 --level_threshold 80.0
```

## 3) Outputs
Se guardan en `05_outputs/`:
- `summary.json`, `run_config.json`, `data_quality_report.md`
- `tables/` (panel snapshot, predicciones, métricas)
- `figures/` (gráficas)
- `models/` (joblib)
