# Loan Default Prediction — Full ML Workflow

This repository contains an end-to-end Jupyter notebook implementing a complete machine-learning workflow to predict loan default using the LendingClub Accepted Loans (2007–2018) dataset.

**Notebook**: Group A Loan Prediction Model.ipynb

**Key points**
- Task: binary classification — predict whether a loan will default before issuance (target: `loan_status` → engineered `target`).
- Dataset: LendingClub Accepted Loans (2007–2018). Public dataset (Kaggle).
- Primary metric: F1‑Macro (balanced evaluation for imbalanced classes). Secondary: AUC‑ROC, PR‑AUC, Brier score.
- Champion model: XGBoost (Optuna‑tuned). Primary challenger: LightGBM. Interpretable fallback: CCP‑pruned Decision Tree.

Contents of the notebook
- Problem definition, EDA, leakage checks, and preprocessing decisions.
- Feature engineering (FICO, credit-history months, ratio features, flags).
- Train/test split (80/20 stratified), scaling, and encoding.
- Baseline models, tuning (Grid/Randomized/Optuna), training, and evaluation.
- Model interpretation: XGBoost gain, permutation importance, SHAP, LIME.
- Decision threshold & business cost analysis, fairness audit, and drift monitoring (PSI).
- Artefact saving and an end-to-end `predict_loan_default()` inference function.

Quick start (Colab / local)
1. Open the notebook: [Group A Loan Prediction Model.ipynb](Group%20A%20Loan%20Prediction%20Model.ipynb)
2. Install dependencies (notebook installs some packages). Recommended minimal requirements:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm optuna shap lime matplotlib seaborn joblib tensorflow
```

3. If running in Colab: mount Google Drive and point the notebook to the compressed CSV (`accepted_2007_to_2018Q4.csv.gz`).
4. Run the cells in order. Long-running steps (Optuna tuning, training on >1M rows) can be done on a subsample for iteration.

Saved artefacts and outputs (created by the notebook)
- `loan_model_artefacts/` — saved models and pickled objects:
  - `xgboost_champion.pkl`, `lightgbm_challenger.pkl`, `random_forest.pkl`, `standard_scaler.pkl`, `feature_names.pkl`, `optimal_threshold.pkl`, `neural_network_model.keras`
- Several diagnostic plots saved as PNG files (EDA, calibration, SHAP, PR/ROC curves, learning curves, etc.).

Using the prediction function
- The notebook provides `predict_loan_default(raw_application: dict)` which:
  - loads the saved artefacts, applies required feature engineering, returns `default_probability`, `decision` (APPROVE/DECLINE), `risk_tier`, and `confidence`.
- Example usage is included near the end of the notebook.

Notes & recommendations
- Data privacy & compliance: The dataset should be used only for research/educational purposes; remove or anonymize any PII before sharing.
- Reproducibility: Training on the full dataset is expensive—use the tuning strategies in the notebook (subsampling, reduced folds) for development.
- Deployment: Wrap `predict_loan_default()` with a REST API (FastAPI/Flask), add monitoring (PSI, input validation), and version model artefacts before serving.

License & attribution
- Data: LendingClub Accepted Loans (2007–2018) — public data (credit to LendingClub / Kaggle dataset). Check dataset license before commercial use.

Questions or changes
- If you want, I can: run a smaller end‑to‑end demo, extract a requirements file, or add a short `serve.md` showing how to wrap the inference function in FastAPI.
