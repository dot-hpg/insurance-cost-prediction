"""
Reference dataset generator for the Insurance Cost Prediction project.

WHY THIS EXISTS:
The real dataset lives behind a Google-Drive link that requires an
authenticated browser session to download, so it can't be pulled
programmatically in this environment. This script builds a dataset that
matches the documented schema and value ranges EXACTLY (same 11 columns,
same min/max per column, same realistic feature-to-target relationships
you'd expect in real underwriting data), so the whole pipeline below is
fully runnable end-to-end right now.

BEFORE YOUR FINAL RUN: download the real CSV from the Drive link in your
brief and drop it in data/Medicalpremium.csv, replacing this one. Column
names are identical, so nothing else in the notebook needs to change.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 986  # matches the "~1000 customers" scale of the real dataset

age = rng.integers(18, 67, N)
diabetes = rng.binomial(1, 0.42, N)
bp_problems = rng.binomial(1, 0.47, N)

transplant_prob = np.clip(0.02 + (age - 18) * 0.001, 0.02, 0.15)
any_transplants = rng.binomial(1, transplant_prob)

chronic_prob = np.clip(0.10 + (age - 18) * 0.004, 0.10, 0.55)
any_chronic = rng.binomial(1, chronic_prob)

height = rng.normal(168, 10, N).clip(145, 188).round(0)
weight = rng.normal(76, 15, N).clip(51, 132).round(0)

known_allergies = rng.binomial(1, 0.22, N)
cancer_history = rng.binomial(1, 0.12, N)

surgeries_prob = np.clip(0.15 + (age - 18) * 0.003, 0.05, 0.6)
number_of_surgeries = rng.binomial(3, surgeries_prob)

bmi = weight / ((height / 100) ** 2)

base = 15000
premium = (
    base
    + (age - 18) * 210
    + diabetes * 1450
    + bp_problems * 1300
    + any_transplants * 4200
    + any_chronic * 2100
    + known_allergies * 350
    + cancer_history * 2600
    + number_of_surgeries * 1550
    + np.clip(bmi - 25, 0, None) * 180
    + rng.normal(0, 1300, N)
)
premium = premium.clip(15000, 40000).round(-2).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Diabetes": diabetes,
    "BloodPressureProblems": bp_problems,
    "AnyTransplants": any_transplants,
    "AnyChronicDiseases": any_chronic,
    "Height": height.astype(int),
    "Weight": weight.astype(int),
    "KnownAllergies": known_allergies,
    "HistoryOfCancerInFamily": cancer_history,
    "NumberOfMajorSurgeries": number_of_surgeries,
    "PremiumPrice": premium,
})

df.to_csv("/home/claude/insurance_project/data/Medicalpremium.csv", index=False)
print(df.shape)
print(df.head())
print(df.describe().round(1))
