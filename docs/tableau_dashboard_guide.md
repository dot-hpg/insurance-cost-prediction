# Tableau Public Dashboard — Build Guide

I can't create a `.twbx` file directly (Tableau isn't something that can be generated as code
the way a notebook or a Flask app can) — but here's exactly what to build, sheet by sheet, so
you can put it together in Tableau Public in under an hour. Connect Tableau directly to
`data/Medicalpremium.csv` (swap in the real dataset first).

## Setup
1. Open Tableau Public → **Connect → Text File** → select `Medicalpremium.csv`.
2. In the Data Source tab, right-click each of the six condition columns
   (`Diabetes`, `BloodPressureProblems`, `AnyTransplants`, `AnyChronicDiseases`,
   `KnownAllergies`, `HistoryOfCancerInFamily`) → **Convert to Discrete** — Tableau will treat
   them as numbers by default, but they're really categories.
3. Create a calculated field `BMI`: `[Weight] / (([Height]/100) * ([Height]/100))`.
4. Create a calculated field `Age Group`:
   ```
   IF [Age] <= 30 THEN "18-30"
   ELSEIF [Age] <= 40 THEN "31-40"
   ELSEIF [Age] <= 50 THEN "41-50"
   ELSEIF [Age] <= 60 THEN "51-60"
   ELSE "61-66"
   END
   ```
5. Create a calculated field `Total Conditions`:
   `[Diabetes] + [BloodPressureProblems] + [AnyTransplants] + [AnyChronicDiseases] + [KnownAllergies] + [HistoryOfCancerInFamily]`

## Sheet 1 — Summary Statistics
- Three-four **KPI text tiles**: Average Premium (`AVG(PremiumPrice)`), Average Age, Count of
  Records, Average BMI. Use Tableau's "Text Table" or a big single-number worksheet for each.
- A **bar chart**: count of individuals with each condition. Put the six condition fields on
  Rows (or pivot them into one field via a Tableau pivot / union if you want a single clean
  bar chart), `CNT([condition]=1)` on Columns.

## Sheet 2 — Premium Pricing Analysis
- **Histogram** of `PremiumPrice` (Tableau: drag PremiumPrice to Columns, right-click →
  Create → Bins, bin size ~1000).
- **Bar chart**: `AVG(PremiumPrice)` by `Age Group`, colored by one condition (e.g., color =
  `AnyChronicDiseases`) to show the compounding effect.
- **Correlation heatmap**: this needs a pre-melted/long-format extract in Tableau (Analysis →
  Create Calculated Field per pair, or use a Tableau "highlight table" with measure names/
  measure values). Simplest approach: build a scatter plot matrix instead — a few key scatter
  plots (Age vs Premium, BMI vs Premium) are usually enough and much faster to build than a
  true correlation matrix in Tableau.

## Sheet 3 — Risk Factors Analysis
- **Bar chart**: `AVG(PremiumPrice)` by `NumberOfMajorSurgeries` (0–3).
- **Stacked bar**: `AVG(PremiumPrice)` split by `AnyChronicDiseases` and `AnyTransplants`
  combined (put both on Color / Detail).
- **Bar chart**: `AVG(PremiumPrice)` by `KnownAllergies` and separately by
  `HistoryOfCancerInFamily`.

## Sheet 4 — Demographic Insights
- **Scatter plot**: `BMI` (Columns) vs `PremiumPrice` (Rows), colored by `Age Group`, sized by
  `Total Conditions`.
- If you have or can infer a region/state field, add a **map** view; if not, skip this tile —
  the brief notes geographic analysis is only relevant "if region data is available."

## Interactivity
- Add a **Filter** action for `Age Group` and each condition field, applied across all four
  sheets via a **Dashboard → Actions → Filter**.
- Add **tooltips** on every mark showing Age, BMI, and PremiumPrice.
- Add a **parameter-driven slider** for BMI range if you want a drill-down feature.

## Assembling the Dashboard
1. Create a new **Dashboard**, size "Automatic" or a fixed 1200×900.
2. Drag all four sheets onto the canvas, arrange in a 2×2 grid.
3. Add a title text box: "Insurance Cost Prediction — Risk & Pricing Dashboard".
4. Test every filter/tooltip before publishing.

## Publishing
1. **Server → Save to Tableau Public As...** (sign in with your Tableau Public account).
2. Once published, copy the public URL — that's the link for your submission document.
3. Double-check the workbook's privacy setting is public, not private, or reviewers won't be
   able to open it.
