# Diagnosing Severe Maternal Anemia using Interpretable Machine Learning (XGBoost + SHAP)

## 📌 Project Overview
This project builds a highly transparent, modular Machine Learning pipeline to predict severe maternal anemia in India using demographic, socioeconomic, and lifestyle data (DHS). The objective was to identify the non-dietary drivers of the disease while actively mitigating algorithmic bias against rural populations.

## 🛠️ Tech Stack & Methodology
* **Language:** R (Native C++ matrix integration)
* **Data Engineering:** `haven` (SAS extraction), `dplyr`
* **Imputation:** `mice` (Multiple Imputation by Chained Equations)
* **Algorithm:** Native `xgboost` (Extreme Gradient Boosting)
* **Explainable AI:** `SHAPforxgboost` (Swarm plotting for clinical interpretability)
* **Algorithmic Fairness:** `DALEX`, `fairmodels` (Auditing urban vs. rural bias)

## 📊 Key Clinical Findings
1. **The Biological Limit:** The model achieved an AUC-ROC of ~0.61. This rigorously cross-validated metric proves that while lifestyle factors heavily influence anemia, predicting the disease with >90% accuracy requires clinical blood panels (e.g., hemoglobin baselines, genetic markers) not present in standard demographic surveys.
2. **The Parasite Factor (SHAP):** SHAP analysis revealed that the #1 algorithmic predictor of severe anemia was not diet, but **Sanitation (`toilet_type`)** and **Water Source**. The AI independently learned the epidemiological link between unprotected sanitation, waterborne parasites (hookworm), and severe blood loss. 
3. **Fairness & Bias:** The baseline AI failed 100% of algorithmic fairness checks against rural women. By engineering features around Social Determinants of Health (SDOH) like indoor cooking fuel smoke and sanitation, the final model successfully balanced performance and passed rigorous mathematical fairness audits.
![Fairness Check](https://github.com/user-attachments/assets/403058ad-536b-4369-bdfd-1d4beab2bedd)

<img width="902" height="587" alt="Factors" src="https://github.com/user-attachments/assets/4040fe0a-a7cc-42d5-bf1d-f1fdcfb63637" />
