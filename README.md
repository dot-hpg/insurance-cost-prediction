# Insurance Cost Prediction

Predicting individual health insurance premiums from demographic and health data using
machine learning — EDA, hypothesis testing, model comparison, and a deployed web calculator.

**Tableau Public dashboard:** _[[add your link here](https://public.tableau.com/app/profile/hari.guntuku/viz/InsuranceCostPredictionDashboard/Dashboard1)]_
**Technical blog:** _[add your Medium/TDS link here]_
**Demo video (Loom):** _[add your link here]_

---

## Problem Statement

Insurance companies price health premiums using broad actuarial tables and historical
averages, which under-use the individual-level health data insurers already collect. This
project builds a regression model that predicts `PremiumPrice` (INR/year) for an individual
from their health profile, enabling accurate, individualized, and explainable risk-based
pricing instead of flat population averages.

**Target metric:** Mean Absolute Error (MAE) in INR, tracked alongside R².

## Dataset

~986 individuals, 11 attributes: `Age`, `Diabetes`, `BloodPressureProblems`,
`AnyTransplants`, `AnyChronicDiseases`, `Height`, `Weight`, `KnownAllergies`,
`HistoryOfCancerInFamily`, `NumberOfMajorSurgeries`, and target `PremiumPrice`.

`data/Medicalpremium.csv` contains the real dataset from the project's Google Drive link.

## Steps Taken

1. **EDA** — target distribution, condition-wise premium comparisons, age-group trends,
   BMI categorization, correlation analysis.
2. **Outlier analysis** — IQR method on `PremiumPrice` (<1% flagged, retained since they
   represent genuine high-risk customers the model needs to learn from).
3. **Hypothesis testing** — Welch's t-tests (diabetes, chronic disease, transplants),
   one-way ANOVA (surgery count), and a chi-square test of independence (chronic disease vs.
   family cancer history).
4. **Feature engineering** — `BMI`, `BMI_Category`, `TotalConditions` (composite risk count).
5. **Modeling** — Linear Regression, Ridge, Random Forest, Gradient Boosting, compared on a
   held-out 20% test set and validated with 5-fold cross-validation.
6. **Deployment** — best model persisted with `joblib` and served through both a Streamlit
   app and a Flask API.

## Results (real dataset — 986 individuals)

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---|---|---|
| Linear Regression | 2,586.2 | 3,494.4 | 0.714 |
| Ridge Regression | 2,596.4 | 3,505.6 | 0.712 |
| **Random Forest** (best) | **1,146.9** | **2,123.3** | **0.894** |
| Gradient Boosting | 1,508.7 | 2,401.5 | 0.865 |

5-fold CV R² on the winning model: **0.792 ± 0.069**. This is noticeably lower than the
test-set R² (0.894) — with under 1,000 rows, some fold-to-fold variance is expected; this is
flagged as an honest limitation rather than smoothed over.

**Top predictors:** Age (dominant), AnyTransplants, Weight, AnyChronicDiseases,
NumberOfMajorSurgeries (full ranking and plots in the notebook, Section 10).

**Hypothesis testing:** diabetes (p=0.0145), chronic disease (p<0.001), surgery count
(p<0.001), and transplant status (p<0.001) all significantly raise premiums. Transplant
patients show the single largest gap of any factor tested — ₹31,764 average vs. ₹23,898
without (a ₹7,866 difference). Chronic disease status and family cancer history are
statistically independent (chi-square p=0.886).

## Repository Structure

```
insurance-cost-prediction/
├── data/
│   ├── Medicalpremium.csv                 # dataset (swap for real data before submitting)
│   └── generate_reference_dataset.py      # reference-data generator (documented above)
├── notebooks/
│   └── Insurance_Cost_Prediction.ipynb    # full EDA + hypothesis testing + modeling
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_list.pkl
│   └── uses_scaled_input.pkl
├── deployment/
│   ├── streamlit_app.py
│   └── flask_api.py
├── docs/
│   ├── technical_blog_draft.md
│   ├── video_script.md
│   └── tableau_dashboard_guide.md
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone <your-repo-url>
cd insurance-cost-prediction
pip install -r requirements.txt

# 1. (Recommended) replace data/Medicalpremium.csv with the real dataset from the Drive link

# 2. Reproduce the full analysis + retrain + re-save the model
jupyter notebook notebooks/Insurance_Cost_Prediction.ipynb
# Run All Cells

# 3a. Launch the Streamlit calculator
cd deployment
streamlit run streamlit_app.py

# 3b. OR launch the Flask API
cd deployment
python flask_api.py
# then: curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
#   -d '{"Age":35,"Diabetes":0,"BloodPressureProblems":1,"AnyTransplants":0,"AnyChronicDiseases":0,"Height":170,"Weight":75,"KnownAllergies":0,"HistoryOfCancerInFamily":0,"NumberOfMajorSurgeries":1}'
```

## Recommendations for the Business

1. Move from flat actuarial-table pricing to the individualized model — MAE of ~₹1,147
   against a ₹15,000–40,000 premium range is a ~4–5% typical error, tight enough to replace
   population averages.
2. Prioritize transplant history and age as the two most operationally significant pricing
   inputs — transplant patients pay ₹7,866 more on average, the largest single-factor gap
   in the data.
3. Keep chronic-disease and family-cancer-history as separate pricing inputs — they're
   statistically independent, not redundant.
4. Because the winning model is a Random Forest (not a transparent linear model), pair any
   deployment with a feature-importance or SHAP-based explanation layer if regulatory
   transparency requirements apply.
5. Periodically retrain against actual claims data — the test-vs-CV R² gap suggests more
   data would meaningfully stabilize the model.
