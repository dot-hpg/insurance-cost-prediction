# Predicting Health Insurance Premiums with Machine Learning: From Actuarial Tables to Individualized Pricing

*A complete walkthrough of an end-to-end data science project — EDA, hypothesis testing,
model comparison, and deployment.*

> **Before publishing:** add your GitHub and deployment links where marked, and swap in your
> own screenshots. This draft is ~2,000 words and structured to match what reviewers look for
> in a portfolio blog: a real problem, a real process, and real numbers — all figures below
> are from the actual trained model on the real dataset.

---

## The Problem: Pricing Risk Without Really Knowing the Individual

Every health insurance premium starts as a bet: the insurer is wagering that the money it
collects from a policyholder this year will, on average across its whole customer base, cover
the claims it has to pay out. Get that bet wrong systematically and one of two bad things
happens — premiums are too high and healthy customers leave for competitors, or premiums are
too low and the insurer loses money on every high-risk customer it accepts.

The traditional way insurers manage this is with actuarial tables: broad buckets built from
historical claims data, segmented by coarse variables like age bands. It works, but it's blunt.
Two 45-year-olds can land in the same actuarial bucket even though one has no health
conditions and the other has diabetes, a chronic illness, and a history of major surgery. The
bucket doesn't know the difference — but the insurer's own intake forms already capture that
difference. The data is sitting there; it's just not being used.

This project asks a simple question: **if we actually use the individual-level health data
insurers already collect, how much more accurately can we price a premium — and which health
factors matter most?**

## The Dataset

The dataset covers roughly 1,000 individuals with 11 attributes: age, height, weight, and six
binary health-condition flags (diabetes, blood pressure problems, any organ transplants, any
chronic disease, known allergies, family history of cancer), plus the count of major surgeries
each person has had. The target variable is `PremiumPrice`, the annual premium in INR, ranging
from ₹15,000 to ₹40,000.

It's a clean dataset — no missing values — which let the project focus on the parts that
actually matter for a business case like this: which features drive cost, whether those
relationships are statistically real or just noise, and how well a model can turn that
understanding into an accurate quote.

## Exploratory Data Analysis: Letting the Data Talk First

Before touching a model, the goal of EDA here was to build intuition an underwriter would
recognize. The premium distribution itself is roughly bell-shaped with a mild right skew —
most customers cluster in the ₹20,000–₹27,000 range, with a smaller group of high-risk
individuals pushing toward ₹35,000+.

Breaking premiums out by each health condition told a consistent story: every single condition
flag — diabetes, blood pressure issues, transplants, chronic disease, allergies, cancer
history — pushed the median premium upward when present. Some of these effects were dramatic
(transplant patients paid noticeably more on average) and some were subtle (known allergies
barely moved the needle). That gap between "dramatic" and "subtle" effects turned out to
matter a lot later, once the modeling stage had to decide which features were actually pulling
their weight.

Age was the standout signal from the very first scatter plot: premium price climbs steadily
and almost linearly with age, and it ended up being the single strongest correlate with the
target in the entire dataset. A composite feature I engineered — `TotalConditions`, a simple
count of how many of the six risk flags a person has — turned out to be the second-strongest
correlate, ahead of any individual condition on its own. That's a useful finding by itself: an
aggregate risk score captures signal that no single flag does, which mirrors how underwriters
already think about "overall health burden" rather than any one diagnosis in isolation.

The correlation heatmap across all numeric and encoded features confirmed the ranking:
age first, `TotalConditions` second, then surgery count and chronic disease status, with
height and known allergies trailing near zero.

## Outlier Handling

I checked for outliers using the IQR method on `PremiumPrice`, since the distribution isn't
perfectly normal and IQR doesn't assume it is. Fewer than 1% of rows fell outside the bounds —
and I chose to keep them. In a pricing model, the "outliers" here aren't data errors; they're
the highest-risk customers, and they're exactly the population the model most needs to learn
to price correctly. Dropping them would have quietly optimized the model for the easy, average
case while leaving it blind to the cases that matter most to an insurer's bottom line.

## Turning Intuition Into Statistics: Hypothesis Testing

EDA suggests patterns, but it doesn't confirm them. That's where formal hypothesis testing
came in, using a 5% significance threshold throughout.

I ran Welch's t-tests (which don't assume equal variance between groups — a safer default than
the standard t-test) comparing premium prices for diabetics vs. non-diabetics, for people with
and without chronic disease, and for people with and without a transplant history. All three
came back statistically significant — chronic disease and transplant status at p < 0.001,
diabetes at p = 0.0145 — the differences visible in the EDA boxplots weren't noise. The
transplant effect was the largest of the three by far: transplant patients pay about ₹7,866
more per year on average.

For surgery count, which has four levels (0 through 3 surgeries) rather than two, I used a
one-way ANOVA instead of a t-test. That also came back significant, confirming that premium
price genuinely shifts across surgery-count groups rather than just between "any surgery" and
"none."

The most interesting result, honestly, was a negative one. I ran a chi-square test of
independence between chronic disease status and family history of cancer — two categorical
health flags — to check whether they were essentially measuring the same underlying risk.
They weren't. The test came back non-significant (p = 0.886), meaning these two risk factors
are statistically independent of each other. For a pricing model, that's actionable: it means
both variables carry unique information and neither can be safely dropped in favor of the
other, even though intuitively they might feel related.

## Building and Comparing the Models

For modeling, I deliberately compared model classes of increasing complexity rather than
jumping straight to the most sophisticated option, because the whole point of an ensemble or
neural approach is that it should *earn* its added complexity by outperforming a simpler
baseline.

- **Linear Regression** — a fully transparent baseline. Every coefficient is directly
  interpretable as "how much this factor moves the premium."
- **Ridge Regression** — the same idea with L2 regularization, to check whether the simpler
  model was overfitting.
- **Random Forest** — an ensemble of decision trees that can capture non-linear interactions
  between risk factors without needing them specified manually.
- **Gradient Boosting** — a sequential boosting ensemble, generally the strongest classical
  approach for structured, tabular data like this.

Features fed into every model included the raw health attributes plus two engineered ones:
`BMI` (derived from height and weight, since BMI is the clinically meaningful combination of
the two) and `TotalConditions` (the composite risk count from the EDA stage). Numeric features
were standardized before training the linear models; the tree-based models used raw features,
since trees are invariant to monotonic scaling.

After training on an 80/20 split and evaluating on the held-out test set:

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---|---|---|
| Linear Regression | ~2,586 | ~3,494 | 0.714 |
| Ridge Regression | ~2,596 | ~3,506 | 0.712 |
| **Random Forest** | ~1,147 | ~2,123 | **0.894** |
| Gradient Boosting | ~1,509 | ~2,402 | 0.865 |

The result that stood out most: **the linear models were clearly outperformed.** Random
Forest beat both Linear Regression and Ridge by a wide margin — 0.894 vs. 0.714 R², nearly a
2.3x reduction in MAE. I validated this wasn't a fluke of the train/test split with 5-fold
cross-validation, which gave R² = 0.792 ± 0.069 — still comfortably ahead of the linear
baselines, though the gap between test R² and CV R² is worth being honest about: with under
1,000 rows, some fold-to-fold variance is expected, and I'm calling that out here rather than
quietly picking the more flattering number.

This makes sense once you look at *which* factor dominates: a prior organ transplant produces
the single largest jump in premium of any variable in the dataset — affected individuals pay
about ₹7,866 more per year on average than those without a transplant history, and that
effect isn't simply additive with age the way, say, an extra risk-condition flag is. Random
Forest can capture that kind of "this factor interacts differently depending on who has it"
pattern; a linear model, by construction, can't. That's a genuinely useful finding: it means
premium pricing here isn't just "sum up your risk factors" — certain conditions compound risk
non-linearly, and a model needs the flexibility to represent that.

There's a real business tradeoff worth naming here too: unlike a linear model, a Random
Forest doesn't hand you a clean "+₹X for age, +₹Y for diabetes" breakdown out of the box. In
an industry where pricing decisions face regulatory scrutiny, that's a real cost of choosing
the more accurate model — one I'd address in production by pairing the model with a
feature-importance or SHAP-based explanation layer, so accuracy and explainability aren't
fully traded off against each other.

## What Actually Drives the Price

Ranking the features by their contribution to the winning Random Forest's predictions
confirmed the EDA story and sharpened it further: age is overwhelmingly the dominant factor,
accounting for the large majority of the model's predictive signal on its own. Behind it,
transplant history, weight, chronic disease status, and surgery count round out the top five.
Known allergies and blood pressure problems contributed the least — useful to know if a
lighter-weight version of the model is ever needed for a simplified product tier.

## Shipping It: From Notebook to Web Calculator

A model that only lives in a notebook doesn't help an underwriter or a customer. The final
step was deploying the trained model behind a simple web interface, built two ways: a
Streamlit app for a quick interactive calculator, and a Flask API for programmatic integration
into an existing quoting system. Both load the same saved model, scaler, and feature list, so
a user enters an age, height, weight, surgery count, and health flags, and gets an instant
premium estimate along with their computed BMI and risk-factor count.

- **GitHub repository:** _[add your link here]_
- **Live deployment:** _[add your Streamlit/Flask deployment link here]_

## Closing Thoughts

The headline result of this project isn't just a model score — it's that properly testing
multiple model classes, rather than assuming either "keep it simple" or "throw the fanciest
model at it," is what actually surfaced the real finding here: health-risk pricing has
genuine non-linear structure, concentrated heavily around a handful of high-impact factors
like transplant history. A linear model would have missed that; testing it against ensembles
and comparing honestly (including being upfront about the CV-vs-test R² gap) is what
surfaced it.

If you're working through a similar structured-data regression problem, my biggest takeaway
is this: don't skip the hypothesis-testing step to jump straight to modeling, and don't assume
you know in advance which model class will win. Test that assumption — the data will tell you
if you're wrong, and sometimes the answer is more interesting than either extreme.

---

*Questions or feedback? Find the full code, notebook, and deployment instructions at the
GitHub link above.*
