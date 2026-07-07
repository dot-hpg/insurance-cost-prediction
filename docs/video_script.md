# 5-Minute Demo Video Script (Loom)

Target length: 5:00. Times are guidelines — practice once before recording so you're talking
naturally, not reading. Have the notebook, the Tableau dashboard, and the deployed app open in
separate tabs before you hit record.

---

### 0:00–0:30 — Hook + Problem Statement
- "Hi, I'm [name], and this is my Insurance Cost Prediction project."
- "Insurers price health premiums with broad actuarial tables and history averages — even
  though they already collect individual health data that could price much more accurately.
  This project builds a model that does that, and ships it as a working calculator."
- Show the problem statement section of the notebook or README on screen.

### 0:30–1:30 — Data & EDA
- Switch to the notebook. Show the dataset shape and data dictionary.
- "About 1,000 individuals, 11 health and demographic attributes, target is annual premium in
  rupees."
- Show 2–3 of the strongest EDA plots: the premium distribution, the age-vs-premium scatter,
  and the correlation heatmap.
- Call out the headline finding: "Age is by far the strongest single predictor, and a
  composite risk-condition count I engineered came in second — stronger than any individual
  condition flag."

### 1:30–2:15 — Hypothesis Testing
- Show the hypothesis testing section.
- "I didn't just eyeball the EDA — I tested it. Welch's t-tests on diabetes, chronic disease,
  and transplant status; ANOVA on surgery count; all came back statistically significant,
  p < 0.001."
- "The one non-significant result was actually interesting: chronic disease status and family
  cancer history turned out to be statistically independent — so both stay in the model as
  separate risk signals."

### 2:15–3:15 — Modeling & Results
- Show the model comparison table/chart.
- "I compared four models — Linear Regression, Ridge, Random Forest, and Gradient Boosting.
  The best one was actually the simplest: Linear Regression, with an R² of about 0.89 and
  mean error of roughly ₹1,074 on premiums ranging ₹15,000 to ₹40,000."
- "That's not just a model — it's a transparent, explainable one. Every predicted premium can
  be broken into 'this much for age, this much for BMI, this much for surgery history' — which
  matters a lot in a regulated industry like insurance."
- Briefly show the feature importance chart and the actual-vs-predicted scatter.

### 3:15–4:15 — Live Demo of the Deployed App
- Switch to the running Streamlit app (or Flask API via a quick curl/Postman call).
- Walk through filling in a sample profile — age, height, weight, a couple of health
  conditions — and click "Estimate Premium."
- "There's the prediction, along with computed BMI and risk-factor count."
- Try a second, higher-risk profile to show the premium moving up sensibly.

### 4:15–4:45 — Tableau Dashboard Tour
- Switch to the published Tableau Public dashboard.
- Walk through 2–3 dashboard views: summary stats, premium-by-condition, and the
  correlation/demographic view.
- Mention one interactive feature (a filter or tooltip) and use it live.

### 4:45–5:00 — Wrap-Up
- "To summarize: individualized, data-driven pricing beats flat actuarial tables here, the
  simplest model actually won, and the whole thing is shipped as a working calculator anyone
  can try. Links to the GitHub repo, blog post, and dashboard are in the description. Thanks
  for watching!"

---

## Recording Tips
- Do a silent dry run first to check timing — 5 minutes goes fast once you're demoing three
  different things.
- Zoom in / increase font size in the notebook and Tableau before recording so text is
  legible on a recorded screen.
- If Loom cuts you close to the limit, trim the EDA section first — the modeling results and
  live demo are the parts reviewers weight most.
