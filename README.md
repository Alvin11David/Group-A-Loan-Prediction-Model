# Group A Loan Prediction Model

End-to-end machine learning project for predicting loan default risk on LendingClub application data. The notebook walks through the full workflow: problem framing, exploratory data analysis, leakage control, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, interpretation, fairness checks, and deployment-oriented artefact saving.

## Project Overview

The goal is to classify a loan application as either a good loan or a bad loan before issuance, using only information available at application time. This is a high-stakes binary classification problem where false negatives are especially costly because approving a risky borrower can lead to capital loss.

The notebook compares multiple models, including a baseline decision tree, random forest, XGBoost, LightGBM, and a neural network. The final workflow emphasises class imbalance handling, probability calibration, explainability, and business threshold selection.

## Key Result

On the held-out test set, the tuned random forest was the best model by macro F1 score:

| Model | Accuracy | Precision | Recall | Specificity | F1 Macro | AUC-ROC | PR-AUC | Brier Score | Log Loss |
|------|----------:|----------:|-------:|------------:|---------:|--------:|-------:|------------:|---------:|
| Random Forest (Tuned) | 0.7461 | 0.4273 | 0.4596 | 0.8267 | 0.6392 | 0.7292 | 0.4272 | 0.1751 | 0.5269 |
| XGBoost (Optuna) | 0.6762 | 0.3686 | 0.6655 | 0.6793 | 0.6203 | 0.7376 | 0.4420 | 0.2034 | 0.5903 |

XGBoost is treated as the champion model for deployment-oriented explainability and threshold analysis, while the tuned random forest is the strongest overall performer in the reported comparison.

## Notebook Workflow

The notebook is organised into the following stages:

1. Problem definition and business framing
2. Dataset loading from LendingClub accepted loans data
3. Exploratory data analysis and missing-value analysis
4. Leakage removal, cleaning, and preprocessing
5. Feature engineering and feature selection demonstrations
6. Train/test split with class imbalance handling
7. Baseline and tuned model training
8. Hyperparameter tuning with GridSearchCV, RandomizedSearchCV, and Optuna
9. Comprehensive model evaluation with confusion matrices, ROC, PR, calibration, and statistical tests
10. Model comparison and selection
11. Explainability with feature importance, SHAP, and LIME
12. Fairness, drift monitoring, model saving, and inference function creation

## Techniques Used

- Missing-value analysis and high-missingness column removal
- Leakage prevention by dropping post-issuance variables
- Target engineering from `loan_status`
- Ordinal, one-hot, and target-guided encoding examples
- Scaling, polynomial features, and interaction term demonstrations
- Filter, wrapper, and embedded feature selection
- Class imbalance handling with `class_weight` and `scale_pos_weight`
- Hyperparameter tuning with grid search, random search, and Optuna
- Probability calibration and Brier score evaluation
- McNemar’s test and bootstrap confidence intervals
- SHAP, permutation importance, and LIME explainability
- Cost-sensitive threshold selection for business use
- PSI-based drift monitoring and model persistence with `joblib`

## Dataset

The project uses the LendingClub Accepted Loans dataset covering 2007–2018 Q4. The notebook describes it as a large tabular dataset with borrower details, loan terms, and repayment outcomes. The target is derived from `loan_status`, where fully paid loans are treated as class 0 and defaults / charged-off loans are treated as class 1.

Because this is a loan approval problem, the notebook explicitly removes columns that are only known after issuance, such as payment totals and recovery-related fields, to avoid leakage.

## Repository Contents

- `Group A Loan Prediction Model.ipynb` - the full analysis and modelling notebook
- `README.md` - project overview and usage guide

Running the notebook also creates model artefacts and visualisations such as:

- `loan_model_artefacts/` for saved models, scaler, feature names, and threshold
- EDA and evaluation figures such as confusion matrices, ROC curves, calibration plots, SHAP plots, and drift charts

## Requirements

The notebook uses common data science libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `optuna`
- `tensorflow`
- `shap`
- `lime`
- `statsmodels`
- `joblib`

The notebook itself includes a pip install step for the extra packages used during modelling.

## How To Run

### 1. Open the notebook

Open `Group A Loan Prediction Model.ipynb` in Jupyter, VS Code, or Google Colab.

### 2. Update the dataset path

The notebook currently loads the dataset from a Google Colab Drive path:

```python
/content/drive/MyDrive/accepted_2007_to_2018Q4.csv.gz
```

If you are running locally, update that path to the location of your CSV or compressed CSV file.

### 3. Run the cells in order

The notebook is designed to be executed top to bottom so that preprocessing, training, evaluation, and artefact saving all happen in sequence.

### 4. Review the outputs

The notebook generates a large set of plots and prints the full evaluation table, including:

- class balance summaries
- learning curves
- tuned model comparison
- ROC and PR curves
- calibration plots
- SHAP and LIME explanations
- fairness and drift checks

## Saved Artefacts

After the final cells run, the notebook saves the following deployment artefacts:

- `loan_model_artefacts/xgboost_champion.pkl`
- `loan_model_artefacts/lightgbm_challenger.pkl`
- `loan_model_artefacts/random_forest.pkl`
- `loan_model_artefacts/standard_scaler.pkl`
- `loan_model_artefacts/feature_names.pkl`
- `loan_model_artefacts/optimal_threshold.pkl`
- `loan_model_artefacts/neural_network_model.keras`

It also defines an end-to-end prediction helper named `predict_loan_default()` that can be wrapped in an API later.

## Business Notes

- False negatives are more expensive than false positives in this lending setting.
- The notebook therefore includes threshold optimisation instead of relying only on the default 0.5 cutoff.
- SHAP is used to support model interpretability and decision transparency.
- A PSI-based drift check is included to support post-deployment monitoring.

## Suggested Next Step

If you want, I can also turn this notebook into a cleaner project README with badges, a short executive summary, and a compact model-results table formatted for GitHub.
