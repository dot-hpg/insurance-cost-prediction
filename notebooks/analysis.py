"""
Insurance Cost Prediction — full analysis pipeline.
Run this to reproduce every number, plot, and the trained model.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
sns.set_palette("husl")

ASSETS = "/home/claude/insurance_project/assets"
MODELS = "/home/claude/insurance_project/models"
RESULTS = {}

# ─────────────────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/insurance_project/data/Medicalpremium.csv")
RESULTS["shape"] = df.shape
RESULTS["missing"] = int(df.isnull().sum().sum())
RESULTS["dtypes"] = df.dtypes.astype(str).to_dict()

# ─────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

def bmi_category(b):
    if b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Normal"
    elif b < 30:
        return "Overweight"
    else:
        return "Obese"

df["BMI_Category"] = df["BMI"].apply(bmi_category)
df["TotalConditions"] = (
    df["Diabetes"] + df["BloodPressureProblems"] + df["AnyTransplants"]
    + df["AnyChronicDiseases"] + df["KnownAllergies"] + df["HistoryOfCancerInFamily"]
)
df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 60, 70],
                         labels=["18-30", "31-40", "41-50", "51-60", "61-66"])

# ─────────────────────────────────────────────────────────
# 3. EDA plots
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df["PremiumPrice"], bins=30, color="steelblue", edgecolor="white")
axes[0].axvline(df["PremiumPrice"].median(), color="red", linestyle="--",
                 label=f'Median: {df["PremiumPrice"].median():.0f}')
axes[0].set_title("Premium Price Distribution", fontweight="bold")
axes[0].set_xlabel("Premium Price (INR)")
axes[0].legend()
axes[1].boxplot(df["PremiumPrice"], patch_artist=True,
                 boxprops=dict(facecolor="lightcoral"))
axes[1].set_title("Premium Price — Boxplot", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/01_target_distribution.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
df["BMI_Category"].value_counts().plot(kind="bar", color="mediumseagreen", ax=ax)
ax.set_title("Customer Count by BMI Category", fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{ASSETS}/02_bmi_category_counts.png", dpi=110)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
conditions = ["Diabetes", "BloodPressureProblems", "AnyTransplants",
              "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"]
for ax, col in zip(axes.flat, conditions):
    df.boxplot(column="PremiumPrice", by=col, ax=ax, patch_artist=True)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("")
plt.suptitle("Premium Price by Health Condition", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{ASSETS}/03_conditions_vs_premium.png", dpi=110, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("AgeGroup")["PremiumPrice"].mean().plot(kind="bar", color="coral", ax=ax)
ax.set_title("Average Premium by Age Group", fontweight="bold")
ax.set_ylabel("Avg Premium (INR)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/04_premium_by_agegroup.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["Age"], df["PremiumPrice"], alpha=0.35, s=15, color="steelblue")
z = np.polyfit(df["Age"], df["PremiumPrice"], 1)
ax.plot(df["Age"], np.poly1d(z)(df["Age"]), color="red", linewidth=2)
ax.set_title("Age vs Premium Price", fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Premium Price")
plt.tight_layout()
plt.savefig(f"{ASSETS}/05_age_vs_premium.png", dpi=110)
plt.close()

numeric_cols = ["Age", "Height", "Weight", "BMI", "NumberOfMajorSurgeries",
                 "TotalConditions", "Diabetes", "BloodPressureProblems",
                 "AnyTransplants", "AnyChronicDiseases", "KnownAllergies",
                 "HistoryOfCancerInFamily", "PremiumPrice"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/06_correlation_heatmap.png", dpi=110)
plt.close()

RESULTS["top_correlations"] = (
    corr["PremiumPrice"].drop("PremiumPrice").sort_values(ascending=False).round(3).to_dict()
)

# ─────────────────────────────────────────────────────────
# 4. Outlier detection (IQR, report only — dataset already clean per docs)
# ─────────────────────────────────────────────────────────
Q1, Q3 = df["PremiumPrice"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df["PremiumPrice"] < lower) | (df["PremiumPrice"] > upper)]
RESULTS["iqr_bounds"] = {"Q1": float(Q1), "Q3": float(Q3), "lower": float(lower),
                          "upper": float(upper), "n_outliers": int(len(outliers))}

# ─────────────────────────────────────────────────────────
# 5. Hypothesis Testing
# ─────────────────────────────────────────────────────────
hyp_results = {}

# H1: Diabetics vs non-diabetics — independent t-test
g1 = df[df["Diabetes"] == 1]["PremiumPrice"]
g0 = df[df["Diabetes"] == 0]["PremiumPrice"]
t, p = stats.ttest_ind(g1, g0, equal_var=False)
hyp_results["diabetes_ttest"] = {"t_stat": float(t), "p_value": float(p),
                                  "mean_diabetic": float(g1.mean()), "mean_non_diabetic": float(g0.mean()),
                                  "significant_at_0.05": bool(p < 0.05)}

# H2: Chronic disease presence — independent t-test
g1 = df[df["AnyChronicDiseases"] == 1]["PremiumPrice"]
g0 = df[df["AnyChronicDiseases"] == 0]["PremiumPrice"]
t, p = stats.ttest_ind(g1, g0, equal_var=False)
hyp_results["chronic_disease_ttest"] = {"t_stat": float(t), "p_value": float(p),
                                         "mean_with": float(g1.mean()), "mean_without": float(g0.mean()),
                                         "significant_at_0.05": bool(p < 0.05)}

# H3: Number of major surgeries — one-way ANOVA
groups = [df[df["NumberOfMajorSurgeries"] == k]["PremiumPrice"] for k in sorted(df["NumberOfMajorSurgeries"].unique())]
f, p = stats.f_oneway(*groups)
hyp_results["surgeries_anova"] = {"f_stat": float(f), "p_value": float(p),
                                   "significant_at_0.05": bool(p < 0.05)}

# H4: Chi-square — chronic disease vs family cancer history (association between two categoricals)
ct = pd.crosstab(df["AnyChronicDiseases"], df["HistoryOfCancerInFamily"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
hyp_results["chronic_vs_cancer_chi2"] = {"chi2": float(chi2), "p_value": float(p),
                                          "significant_at_0.05": bool(p < 0.05)}

# H5: Transplant patients — t-test
g1 = df[df["AnyTransplants"] == 1]["PremiumPrice"]
g0 = df[df["AnyTransplants"] == 0]["PremiumPrice"]
t, p = stats.ttest_ind(g1, g0, equal_var=False)
hyp_results["transplant_ttest"] = {"t_stat": float(t), "p_value": float(p),
                                    "mean_with": float(g1.mean()), "mean_without": float(g0.mean()),
                                    "significant_at_0.05": bool(p < 0.05)}

RESULTS["hypothesis_tests"] = hyp_results

# ─────────────────────────────────────────────────────────
# 6. Modeling
# ─────────────────────────────────────────────────────────
TARGET = "PremiumPrice"
FEATURES = ["Age", "Diabetes", "BloodPressureProblems", "AnyTransplants",
            "AnyChronicDiseases", "Height", "Weight", "KnownAllergies",
            "HistoryOfCancerInFamily", "NumberOfMajorSurgeries", "BMI", "TotalConditions"]

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

all_results = []
fitted_models = {}

lr = LinearRegression()
lr.fit(X_train_s, y_train)
all_results.append(evaluate("Linear Regression", y_test, lr.predict(X_test_s)))
fitted_models["Linear Regression"] = lr

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_s, y_train)
all_results.append(evaluate("Ridge Regression", y_test, ridge.predict(X_test_s)))
fitted_models["Ridge Regression"] = ridge

rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=4,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
all_results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))
fitted_models["Random Forest"] = rf

gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                random_state=42)
gb.fit(X_train, y_train)
all_results.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))
fitted_models["Gradient Boosting"] = gb

results_df = pd.DataFrame(all_results).set_index("Model")
RESULTS["model_comparison"] = results_df.round(3).to_dict(orient="index")
best_model_name = results_df["R2"].idxmax()
RESULTS["best_model"] = best_model_name
best_tree = fitted_models[best_model_name]
uses_scaled_input = best_model_name in ("Linear Regression", "Ridge Regression")

# 5-fold CV on the winning model type for robustness
cv = KFold(n_splits=5, shuffle=True, random_state=42)
if uses_scaled_input:
    X_cv = StandardScaler().fit_transform(X)
    cv_scores = cross_val_score(fitted_models[best_model_name], X_cv, y, cv=cv, scoring="r2")
else:
    cv_scores = cross_val_score(fitted_models[best_model_name], X, y, cv=cv, scoring="r2")
RESULTS["best_model_cv_r2_mean"] = float(cv_scores.mean())
RESULTS["best_model_cv_r2_std"] = float(cv_scores.std())

# Feature importance / coefficients (winning model)
if hasattr(best_tree, "feature_importances_"):
    importances = pd.Series(best_tree.feature_importances_, index=FEATURES).sort_values(ascending=False)
else:
    importances = pd.Series(np.abs(best_tree.coef_), index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", color="steelblue", ax=ax)
ax.invert_yaxis()
ax.set_title(f"Feature Importance — {best_model_name}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/07_feature_importance.png", dpi=110)
plt.close()
RESULTS["feature_importance"] = importances.round(4).to_dict()

# Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
labels = results_df.index.tolist()
colors = ["#5b8db8", "#e0a83f", "#6aab72", "#cc5f5f"]
for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    bars = ax.bar(labels, results_df[metric], color=colors)
    ax.set_title(metric, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, results_df[metric]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
plt.suptitle("Model Comparison — Test Set", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/08_model_comparison.png", dpi=110)
plt.close()

# Actual vs predicted for best model
best_pred = best_tree.predict(X_test_s) if uses_scaled_input else best_tree.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, best_pred, alpha=0.4, s=20, color="steelblue")
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
axes[0].set_title(f"Actual vs Predicted — {best_model_name}", fontweight="bold")
axes[0].set_xlabel("Actual Premium")
axes[0].set_ylabel("Predicted Premium")
axes[0].legend()
residuals = y_test - best_pred
axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_title("Residual Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ASSETS}/09_actual_vs_predicted.png", dpi=110)
plt.close()
RESULTS["residuals_mean"] = float(residuals.mean())
RESULTS["residuals_std"] = float(residuals.std())

# ─────────────────────────────────────────────────────────
# 7. Save best model + scaler + feature list for deployment
# ─────────────────────────────────────────────────────────
joblib.dump(best_tree, f"{MODELS}/best_model.pkl")
joblib.dump(FEATURES, f"{MODELS}/feature_list.pkl")
joblib.dump(scaler, f"{MODELS}/scaler.pkl")
joblib.dump(uses_scaled_input, f"{MODELS}/uses_scaled_input.pkl")
RESULTS["saved_model_path"] = "models/best_model.pkl"
RESULTS["uses_scaled_input"] = uses_scaled_input

with open("/home/claude/insurance_project/notebooks/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("DONE.")
print(json.dumps({k: v for k, v in RESULTS.items() if k not in ("dtypes",)}, indent=2, default=str))
