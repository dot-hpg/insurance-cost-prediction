# Submission Checklist — Insurance Cost Prediction

Mapped directly against your Project Brief's "Submission" section. ✅ = fully built and in
this folder. 🔲 = requires an action from you that I can't do (accounts, uploads, recording).

| Requirement | Status | What to do |
|---|---|---|
| Python notebook: EDA, hypothesis testing, ML modeling, insights | ✅ | `notebooks/Insurance_Cost_Prediction.ipynb` — real results confirmed, Random Forest R²=0.894 |
| Deployment code + files | ✅ | `deployment/streamlit_app.py`, `deployment/flask_api.py` — tested working against your real data |
| README (problem, metric, steps, scores, deployment) | ✅ | `README.md` — updated with real numbers, just fill in your 3 links at the top |
| GitHub repository | 🔲 | Create a repo, push everything in this folder (steps below) |
| Tableau Public dashboard | 🔲 | Follow `docs/tableau_dashboard_guide.md`, then publish and grab the link |
| 2,000-word technical blog | ✅ draft | `docs/technical_blog_draft.md` — updated with real numbers, review and personalize |
| 5-minute Loom demo video | 🔲 | Script ready in `docs/video_script.md` — record following it |
| datasciencesportfol.io page (optional) | 🔲 | Optional — add your GitHub/Tableau links once you have them |

## ⚠️ Important: this zip's data/models are still placeholders

This refreshed zip updates the **notebook, README, and blog** with your real results — but
`data/Medicalpremium.csv` and `models/*.pkl` in this download are still my original
schema-matched reference files, because I don't have access to your actual CSV or your
locally-retrained model files.

**On your machine, you already have the real versions** (you swapped the real CSV in and
re-ran the notebook, which re-saved `models/*.pkl`). **Don't overwrite your local
`data/Medicalpremium.csv` or `models/` folder with the ones from this zip** — only pull in the
updated `notebooks/Insurance_Cost_Prediction.ipynb`, `README.md`, and
`docs/technical_blog_draft.md` from here, and keep your own real data/model files as-is.

## Real Results Summary (confirmed from your notebook run)

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---|---|---|
| Linear Regression | 2,586.2 | 3,494.4 | 0.714 |
| Ridge Regression | 2,596.4 | 3,505.6 | 0.712 |
| **Random Forest** (best) | **1,146.9** | **2,123.3** | **0.894** |
| Gradient Boosting | 1,508.7 | 2,401.5 | 0.865 |

5-fold CV R²: 0.792 ± 0.069 (best model). All hypothesis tests significant except chronic
disease vs. family cancer history (chi-square p=0.886, independent).

## Step-by-Step: GitHub

```bash
cd insurance-cost-prediction   # this folder
git init
git add .
git commit -m "Insurance Cost Prediction: EDA, hypothesis testing, modeling, deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/insurance-cost-prediction.git
git push -u origin main
```
If you don't have a repo yet: GitHub.com → **New repository** → name it → create → then run
the commands above with that repo's URL.

## Step-by-Step: Tableau Public
See `docs/tableau_dashboard_guide.md` for the full sheet-by-sheet build. Once published,
paste the public URL into `README.md` and your submission document.

## Step-by-Step: Technical Blog
1. Open `docs/technical_blog_draft.md`.
2. Replace the placeholder metrics/links with your real numbers once you've re-run the
   notebook on the real dataset.
3. Add 2–3 screenshots (the EDA plots in `assets/`, the model comparison chart, a screenshot
   of the deployed app).
4. Paste into Medium or Towards Data Science's submission editor, format headers, publish.
5. Copy the published URL into `README.md`.

## Step-by-Step: Demo Video
1. Open `docs/video_script.md`.
2. Have three tabs ready before recording: the notebook, the published Tableau dashboard, and
   the running Streamlit app (`cd deployment && streamlit run streamlit_app.py`).
3. Record with Loom (or screen record + webcam), following the script's timing.
4. Get the shareable Loom link and add it to `README.md`.

## Step-by-Step: Final Submission Document
Per your brief: create a document containing the Tableau link, GitHub link, Loom link,
technical blog link, and (optional) datasciencesportfol.io link — then **convert it to PDF and
submit that PDF**. A ready-to-fill version of that summary is basically the top of your
`README.md` — copy those four lines into a blank doc, fill in the real URLs, export as PDF.

## Sanity Checks Before You Submit
- [ ] `Medicalpremium.csv` is the **real** dataset, not the reference one
- [ ] Notebook was re-run top-to-bottom with **no errors** after swapping the data
- [ ] All four links (Tableau, GitHub, blog, video) work in an incognito window (i.e., they're
      actually public, not accidentally private)
- [ ] README's results table matches the notebook's actual output
- [ ] The deployed app (Streamlit or Flask) runs locally without errors right before recording
      the demo video
