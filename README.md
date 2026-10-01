# Preterm birth prediction model for antenatal corticosteroid timing

Reference implementation of the model described in:

> Gulersen M, Lin A, Salleb-Aouissi A, et al. **Development and validation of a
> prediction tool to optimize antenatal corticosteroid timing in patients at risk
> of spontaneous preterm birth.** *Pregnancy* 2026;2:e70217.

The model estimates the probability that a patient presenting at risk of
spontaneous preterm birth will deliver **within 7 days**, so that antenatal
corticosteroids (ACS) can be given in the window where they are most effective.

An online calculator scores one patient at a time. This repository exists so
that investigators can apply the model to **an entire dataset at once** and
evaluate it in their own samples, and so that the derivation of the published
coefficients is inspectable rather than only their values.

## What is here

| Path | Purpose |
|---|---|
| `src/ptb_model.py` | The published coefficients, and a scorer that applies them |
| `src/score.py` | Command-line batch scorer (CSV in, CSV out) |
| `src/fit_model.py` | The fitting code that produced the coefficients |
| `tests/` | Test suite |
| `example_input.csv` | Five synthetic patients showing the input format |

## Quick start

```bash
pip install -r requirements.txt

# score a file of patients
python src/score.py --input example_input.csv --output scored.csv
```

Or from Python:

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from ptb_model import predict

df = pd.read_csv("my_patients.csv")
out = predict(df, cohort="nuMoM2b")
print(out[["risk_within_7d", "risk_more_than_7d"]])
```

## Input schema

One row per patient. Column names must match exactly.

| Column | Units / encoding |
|---|---|
| `maternal_age` | years |
| `bmi` | kg/m² |
| `gestational_age` | weeks at presentation, decimal (30 3/7 → `30.43`) |
| `gravidity` | `1` = first pregnancy, `2` = second, `3` = third or more |
| `preterm_labor` | `1` = yes, `0` = no |
| `pprom` | `1` = membranes ruptured, `0` = intact |
| `effacement` | `1` = 0–25% or >3 cm length, `2` = 26–50% or 2 cm, `3` = 51–75% or 1 cm, `4` = 76–90% or 0.5 cm, `5` = 91–100% or completely effaced |
| `cervical_dilation` | cm |

Any additional columns (e.g. a study ID) are carried through to the output
unchanged. The model performs no imputation: rows with a missing predictor
score as `NaN`, and how to handle missingness is left to the analyst.

## The three model variants

All three were fitted on the nuMoM2b development cohort. They differ only in
which predictors were available in the cohort each was applied to, so pick the
one whose variable set matches your data.

| `cohort=` | Predictors | In the paper |
|---|---|---|
| `nuMoM2b` | all 8 | Development cohort |
| `IU` | all 8 | First validation cohort |
| `TJUH` | 7 (no gravidity) | Second validation cohort |

## The formula

Predictors are standardized, then combined linearly and passed through the
logistic function:

```
z = intercept + Σ  coef_i × (x_i − mean_i) / sd_i

P(delivery > 7 days)     = 1 / (1 + exp(−z))
P(delivery within 7 days) = 1 − P(delivery > 7 days)
```

Note the sign convention, which is the one used in the paper's Supplementary
Table: a **positive** coefficient indicates a higher likelihood of delivery
**more than** 7 days after administration; a negative coefficient, delivery
within 7 days. The quantity clinicians act on is the risk of delivery *within*
7 days, which is `1 − P(>7 days)` — the `risk_within_7d` column.

### Coefficients

Standardization constants (shared by all variants):

| Predictor | Mean | SD |
|---|---|---|
| `maternal_age` | 26.97 | 6.04 |
| `bmi` | 27.18 | 6.87 |
| `gestational_age` | 29.77 | 3.30 |
| `gravidity` | 1.46 | 0.50 |
| `preterm_labor` | 0.85 | 0.36 |
| `pprom` | 0.54 | 0.50 |
| `effacement` | 2.75 | 1.18 |
| `cervical_dilation` | 1.48 | 1.50 |

Coefficients:

| Predictor | nuMoM2b | IU | TJUH |
|---|---|---|---|
| `maternal_age` | −0.49 | −0.45 | −0.42 |
| `bmi` | −0.23 | −0.21 | −0.18 |
| `gestational_age` | −0.26 | −0.24 | −0.04 |
| `gravidity` | 0.07 | 0.20 | — |
| `preterm_labor` | −0.12 | −0.08 | −0.01 |
| `pprom` | −1.88 | −1.74 | −1.39 |
| `effacement` | −0.22 | −0.24 | −0.31 |
| `cervical_dilation` | −0.71 | −0.65 | −0.73 |
| *intercept* | 0.17 | 0.15 | 0.11 |

### A note on the gravidity constants

The table above reproduces the published Supplementary Table exactly, and
matches the deployed calculator. In the development data, however, the gravidity
variable takes values {1, 2, 3} with mean **1.32** and SD **0.57**, while the
smoking variable is binary {1, 2} with mean 1.46 and SD 0.50 — so the published
table appears to carry the mean and SD of these two rows swapped.

The default preserves the published model exactly. Passing
`use_empirical_gravidity=True` (or `--empirical-gravidity` on the command line)
standardizes gravidity with its empirical constants instead. The difference is
small, because the gravidity coefficient is itself small (0.07 and 0.20): across
the example patients it moves the predicted risk by less than 0.05.

## Reproducing the coefficients

`src/fit_model.py` is the fitting code. Predictors are standardized to zero mean
and unit variance; L2-penalized logistic regression is fitted with the
regularization strength and iteration cap selected by 5-fold cross-validated
grid search; the outcome is delivery within 7 days versus more than 7 days.

```bash
python src/fit_model.py --data mycohort.csv              # 8-predictor model
python src/fit_model.py --data mycohort.csv --no-gravidity   # TJUH variant
```

The input needs the predictor columns above plus `interval_days`, the number of
days from ACS administration to delivery.

Run against the development cohort, this reproduces the published
standardization constants exactly (maternal age 26.97/6.04, BMI 27.18/6.87,
gestational age 29.77/3.30, preterm labor 0.85/0.36, PPROM 0.54/0.50,
effacement 2.75/1.18, cervical dilation 1.48/1.50) after restricting to
presentations between 20 and 37 weeks.

Because the published coefficients are rounded to two decimals, scoring with
them is not bit-identical to the fitted estimator. Checked against the fitted
model on the first validation cohort (n = 107), the mean absolute difference in
predicted risk is 0.005 and the maximum is 0.018, with no patient differing by
more than 0.05.

## Data availability

**No patient-level data are included in this repository.** The development
cohort comes from the nuMoM2b study and is available to qualified investigators
through the NICHD Data and Specimen Hub (DASH); the two validation cohorts are
institutional data that cannot be redistributed. `example_input.csv` contains
synthetic rows that illustrate the input format only — the values are not real
patients and should not be used to assess model performance.

## Tests

```bash
python -m pytest tests/ -q
```

## Intended use

This is a research tool. It is not a medical device, has not been prospectively
validated, and must not be used as the sole basis for a treatment decision. The
model was developed in nulliparous patients presenting at risk of *spontaneous*
preterm birth; it should not be extrapolated to patients at risk of
clinician-initiated preterm birth without further evaluation.
