# Interview Prep: Insurance Cost Prediction Case Study

How to talk about this project convincingly in a data science / ML interview — structured
around what interviewers actually probe for: do you understand *why* you made each decision,
not just *what* you did.

---

## 1. The 60-Second Pitch (memorize this cold)

"I built an end-to-end ML project predicting individual health insurance premiums from
demographic and health data — the goal was replacing flat actuarial-table pricing with
individualized, statistically-justified pricing. I did EDA, ran five formal hypothesis tests
to validate which risk factors actually mattered statistically, compared four model classes
head-to-head, and shipped the winning model behind a Streamlit app and Flask API. The
interesting result: Random Forest beat linear models by a wide margin — R² of 0.89 vs 0.71 —
because a handful of risk factors, especially prior organ transplants, have non-linear,
compounding effects on premium that a linear model structurally can't capture."

That's it. Say this, then let them steer into whatever they want to dig into.

---

## 2. Deep-Dive Q&A — Be Ready to Defend Every Choice

### "Walk me through your EDA process."
Don't just say "I plotted things." Frame it as **hypothesis generation**: "I used EDA to
build intuition an underwriter would recognize — distribution of the target, premium broken
out by each condition flag, and a correlation heatmap. That surfaced candidates (age, a
composite condition count) that I then *formally tested* rather than treating the visual
pattern as proof."

### "Why IQR for outliers instead of Z-score?"
"Z-score assumes approximate normality; my target had a right skew, so IQR is the more
defensible, distribution-free choice." **Follow-up they'll ask: "Why keep the outliers
instead of removing them?"** — "Because in a pricing model, outliers aren't data errors,
they're the highest-risk customers — exactly the population the model most needs to learn
from. Removing them optimizes for the easy average case and blinds the model to the tail risk
that matters most to the business."

### "Why Welch's t-test instead of a standard t-test?"
"Welch's doesn't assume equal variance between groups, which is a safer default — I didn't
want to assume homoscedasticity without checking it."

### "You ran five separate hypothesis tests at α=0.05. Doesn't that inflate your false
positive rate? Why didn't you correct for multiple comparisons (Bonferroni, FDR)?"
**This is a real gap — own it, don't dodge it.** "That's fair — with five tests at 0.05 each,
family-wise error rate is inflated. I'd apply a Bonferroni or Benjamini-Hochberg correction in
a next iteration. That said, four of the five results had p-values well under 0.001, so they'd
survive correction; the one borderline result (diabetes, p=0.0145) is the one I'd want to
re-verify under a corrected threshold before leaning on it in a report."
*(Practicing saying "that's a gap, here's what I'd do" is worth more than pretending it's not
an issue — interviewers are testing intellectual honesty as much as knowledge.)*

### "Why did Random Forest beat Linear Regression by so much? Isn't that unusual for ~1,000 rows?"
"It comes down to *which* feature dominates. A prior organ transplant produces the single
largest premium jump in the dataset, and that effect isn't simply additive with age the way a
generic risk flag might be — it's a threshold/interaction effect. Random Forest can carve out
that kind of 'this factor behaves differently depending on who has it' pattern through
splits; a linear model, by construction, forces every feature into a single global
coefficient. That's a real non-linearity, not overfitting — I confirmed it generalizes with
5-fold CV."

### "Your test R² was 0.894 but your CV R² was 0.792 ± 0.069. That's a big gap — are you
overfitting?"
**Do not get defensive here — this is the single most important answer to nail.**
"I flagged that gap explicitly rather than hiding behind the better number. With under 1,000
rows, fold-to-fold variance is expected — some folds will randomly contain more high-leverage
cases like transplant patients than others. I'd address it two ways: get more data, and
constrain the Random Forest's depth/min-samples-leaf further to reduce variance, accepting a
small bias tradeoff. I'd also want to see the per-fold R² spread, not just mean ± std, before
concluding anything more specific."

### "Why MAE as your primary metric instead of RMSE or R²?"
"MAE is directly interpretable to a non-technical stakeholder — 'our estimate is off by about
₹1,147 on average' — whereas RMSE penalizes large errors quadratically, which matters less
here since I'm not specifically trying to avoid rare catastrophic misses, I'm trying to
communicate typical accuracy to underwriting. I tracked R² alongside it to communicate
variance explained, but MAE was the metric I'd actually report to a business stakeholder."

### "How would you explain this Random Forest model to a regulator who wants to know why a
specific customer got a specific quote?"
"Two-pronged: globally, feature importances tell the underwriting team which factors matter
most in aggregate. For an individual, per-quote explanation, I'd add a SHAP layer — it gives
a local, additive breakdown of exactly which features pushed this specific customer's
premium up or down, which is the closest tree-based equivalent to a linear model's
coefficients. I didn't build that in this version, but flagged it explicitly as next-step
work because regulatory explainability is a real constraint in insurance, not a nice-to-have."

### "Your BMI and TotalConditions are engineered features. Walk me through that decision."
"BMI converts two raw measurements into the single number underwriters actually reason with
clinically. TotalConditions is a composite risk count — during EDA it correlated with premium
more strongly than any individual condition flag, which mirrors how underwriters already think
about 'overall health burden' rather than any single diagnosis."

### "If you had 10x more data, what would you do differently?"
"Two things: first, I'd expect the CV/test R² gap to tighten, since a big chunk of that
variance is small-sample noise. Second, I'd revisit Gradient Boosting and even a small neural
net — with more data, the extra flexibility of boosting/deep learning tends to pay off more
than it did here, where ~1,000 rows likely capped how much Random Forest could even benefit
from its own flexibility."

### "If you had 10x LESS data — say, 100 rows — what would change?"
"I'd trust the Random Forest result a lot less; with that few rows I'd default to the linear
model even if it scored lower on this split, purely because a simpler model is less likely to
be fitting sampling noise. I'd also lean harder on cross-validation with more folds, or
leave-one-out CV, to get a stable estimate at all."

---

## 3. The Question That Actually Separates Candidates: Fairness & Ethics

**"This model prices individuals based on health conditions. Are there fairness concerns
you'd want to check before deploying this in production?"**

This is a favorite at FAANG-adjacent companies for ML roles — they want to see you think past
"the model works" into "should it be deployed as-is."

Good answer: "Yes — a few specific things I'd check before production:
1. **Proxy discrimination** — even without race/gender in the feature set, correlated
   variables (e.g., certain conditions disproportionately affecting one demographic) could
   create disparate impact. I'd run the model's predictions segmented by any available
   demographic variable, even if not used as a feature, to check for that.
2. **Regulatory constraints** — many jurisdictions restrict what health factors can legally be
   used in insurance pricing at all; that's a legal review, not just a modeling one.
3. **Feedback loops** — if higher-risk individuals get priced out and stop buying insurance,
   the training data itself becomes biased over time (survivorship bias in future data).
I'd flag all three to a product/legal stakeholder before shipping, not just optimize for R²
and call it done."

---

## 4. System Design Adjacent: "How would you deploy this at scale?"

If asked to extend past what you built, sketch this verbally:
- **Feature store** — precompute/cache derived features (BMI, TotalConditions) rather than
  recomputing per request
- **Model versioning** — store model artifacts with version tags (you already do this with
  joblib — mention it), so you can roll back if a retrain regresses
- **Monitoring** — track prediction drift (are premium estimates trending in a way that
  doesn't match actual claims?) and data drift (are incoming customer profiles shifting away
  from training distribution?)
- **Retraining cadence** — periodic retrain against actual claims outcomes, with a
  shadow-mode comparison before promoting a new model to serve live traffic
- **A/B testing** — if replacing an existing pricing system, test the new model against the
  old on a traffic split before full cutover, given the direct revenue impact of pricing

---

## 5. Questions to Ask Your Interviewer (always have 2-3 ready)

- "How does your team currently handle the explainability/accuracy tradeoff when a more
  complex model wins, especially in a regulated domain?"
- "What does your model monitoring / drift detection stack look like in production?"
- "How do you typically validate for fairness before shipping a pricing or risk model?"

These signal you think about ML as a *system*, not just a notebook exercise — which is
usually exactly what's being evaluated in a senior-leaning DS/MLE interview.

---

## 6. Weaknesses to Pre-empt (say these before they ask — it lands better)

- Dataset is small (~986 rows) — acknowledge the CV/test gap unprompted, don't wait to be
  caught.
- No multiple-testing correction on the 5 hypothesis tests — mention it as a known next step.
- No fairness/bias audit performed — mention this is flagged, not resolved.
- The model isn't SHAP-explained per-prediction yet — mention it's the natural next step for
  a regulated-industry deployment.

Naming your own limitations unprompted is one of the highest-signal things you can do in a
technical interview — it shows you understand the work at a level beyond "it scored well."
