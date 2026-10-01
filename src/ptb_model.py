"""
Preterm birth prediction model: published coefficients and a batch scorer.

This module implements the prediction model described in

    Gulersen M, Lin A, Salleb-Aouissi A, et al. Development and validation of a
    prediction tool to optimize antenatal corticosteroid timing in patients at
    risk of spontaneous preterm birth. Pregnancy 2026;2:e70217.

It exists so that other investigators can apply the model to an entire dataset
at once, rather than entering patients one at a time into the online calculator.

The model is multinomial logistic regression on standardized predictors. Scoring
a patient therefore requires three constants per predictor -- a mean, a standard
deviation, and a coefficient -- all of which are reproduced in `COHORTS` below
exactly as published in the paper's Supplementary Table.

    z = intercept + sum_i  coef_i * (x_i - mean_i) / sd_i
    P(delivery > 7 days) = 1 / (1 + exp(-z))
    P(delivery within 7 days) = 1 - P(delivery > 7 days)

Note the sign convention, which is the one used in the published table: a
POSITIVE coefficient indicates a higher likelihood of delivery MORE than 7 days
after administration, and a negative coefficient a higher likelihood of delivery
within 7 days. The risk clinicians act on -- the risk of delivering within
7 days -- is therefore `1 - P(>7 days)`, which is what `predict` returns in its
`risk_within_7d` column.

Three model variants are published, differing only in which predictors the
corresponding validation cohort recorded. All three were fitted on the nuMoM2b
development cohort.

    "nuMoM2b"  8 predictors, including gravidity (development cohort)
    "IU"       8 predictors, including gravidity (first validation cohort)
    "TJUH"     7 predictors, no gravidity       (second validation cohort)

Usage
-----
    from ptb_model import predict
    import pandas as pd

    df = pd.read_csv("my_patients.csv")
    out = predict(df, cohort="nuMoM2b")
    out[["risk_within_7d", "risk_more_than_7d"]].head()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["COHORTS", "PREDICTORS", "predict", "predict_one", "linear_predictor"]

# Column names expected on the input frame, and the encoding each one uses.
# These encodings are the ones given in the published Supplementary Table.
PREDICTORS = {
    "maternal_age": "years",
    "bmi": "kg/m^2",
    "gestational_age": "weeks, decimal (e.g. 30 3/7 -> 30.43)",
    "gravidity": "1 = first pregnancy, 2 = second, 3 = third or more",
    "preterm_labor": "1 = yes, 0 = no",
    "pprom": "1 = yes (membranes ruptured), 0 = no",
    "effacement": ("1 = 0-25% or >3 cm length, 2 = 26-50% or 2 cm, "
                   "3 = 51-75% or 1 cm, 4 = 76-90% or 0.5 cm, "
                   "5 = 91-100% or completely effaced"),
    "cervical_dilation": "cm",
}

# Published constants. Means and standard deviations are those of the nuMoM2b
# development cohort and are shared by all three variants; the coefficients
# differ by variant. Reproduced verbatim from the Supplementary Table.
_MEANS = {
    "maternal_age": 26.97,
    "bmi": 27.18,
    "gestational_age": 29.77,
    "gravidity": 1.46,
    "preterm_labor": 0.85,
    "pprom": 0.54,
    "effacement": 2.75,
    "cervical_dilation": 1.48,
}

_SDS = {
    "maternal_age": 6.04,
    "bmi": 6.87,
    "gestational_age": 3.30,
    "gravidity": 0.50,
    "preterm_labor": 0.36,
    "pprom": 0.50,
    "effacement": 1.18,
    "cervical_dilation": 1.50,
}

# KNOWN DISCREPANCY (gravidity only).
#
# The values above are reproduced exactly as printed in the published
# Supplementary Table, and match the deployed online calculator. In the
# development data, however, the gravidity variable takes values {1, 2, 3} with
# mean 1.32 and SD 0.57, while the smoking variable is binary {1, 2} with mean
# 1.46 and SD 0.50. The published table appears to carry these two rows'
# mean and SD swapped, so the constants used for gravidity above are in fact the
# smoking variable's.
#
# The effect is small -- the gravidity coefficient is 0.07 (nuMoM2b) and 0.20
# (IU) -- and the default below preserves the published model exactly. To
# standardize gravidity with its empirical constants instead, pass
# `use_empirical_gravidity=True` to `predict`.
_EMPIRICAL_GRAVIDITY = {"mean": 1.32, "sd": 0.57}

COHORTS = {
    "nuMoM2b": {
        "label": "Development cohort (nuMoM2b)",
        "means": _MEANS,
        "sds": _SDS,
        "coefficients": {
            "maternal_age": -0.49,
            "bmi": -0.23,
            "gestational_age": -0.26,
            "gravidity": 0.07,
            "preterm_labor": -0.12,
            "pprom": -1.88,
            "effacement": -0.22,
            "cervical_dilation": -0.71,
        },
        "intercept": 0.17,
    },
    "IU": {
        "label": "First validation cohort (Indiana University)",
        "means": _MEANS,
        "sds": _SDS,
        "coefficients": {
            "maternal_age": -0.45,
            "bmi": -0.21,
            "gestational_age": -0.24,
            "gravidity": 0.20,
            "preterm_labor": -0.08,
            "pprom": -1.74,
            "effacement": -0.24,
            "cervical_dilation": -0.65,
        },
        "intercept": 0.15,
    },
    "TJUH": {
        "label": "Second validation cohort (Thomas Jefferson University Hospital)",
        "means": _MEANS,
        "sds": _SDS,
        # This variant omits gravidity, which was not recorded in that cohort.
        "coefficients": {
            "maternal_age": -0.42,
            "bmi": -0.18,
            "gestational_age": -0.04,
            "preterm_labor": -0.01,
            "pprom": -1.39,
            "effacement": -0.31,
            "cervical_dilation": -0.73,
        },
        "intercept": 0.11,
    },
}


def _spec(cohort):
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {sorted(COHORTS)}, got {cohort!r}")
    return COHORTS[cohort]


def linear_predictor(df: pd.DataFrame, cohort: str = "nuMoM2b",
                     use_empirical_gravidity: bool = False) -> np.ndarray:
    """Return the linear predictor z for each row of `df`.

    See the note on `_EMPIRICAL_GRAVIDITY` for `use_empirical_gravidity`.
    """
    spec = _spec(cohort)
    coefs = spec["coefficients"]

    missing = [c for c in coefs if c not in df.columns]
    if missing:
        raise KeyError(
            f"input is missing required column(s) for cohort {cohort!r}: {missing}. "
            f"Expected: {sorted(coefs)}"
        )

    z = np.full(len(df), float(spec["intercept"]))
    for name, coef in coefs.items():
        x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        mean, sd = spec["means"][name], spec["sds"][name]
        if name == "gravidity" and use_empirical_gravidity:
            mean, sd = _EMPIRICAL_GRAVIDITY["mean"], _EMPIRICAL_GRAVIDITY["sd"]
        z = z + coef * (x - mean) / sd
    return z


def predict(df: pd.DataFrame, cohort: str = "nuMoM2b",
            use_empirical_gravidity: bool = False) -> pd.DataFrame:
    """Score a whole dataframe.

    Returns a copy of `df` with two columns appended:

        risk_within_7d     predicted probability of delivery within 7 days
        risk_more_than_7d  predicted probability of delivery after 7 days

    Rows with a missing or non-numeric predictor yield NaN rather than raising,
    so that one incomplete record does not stop a batch. The model has no
    internal imputation; decide how to handle missing data before calling this.
    """
    z = linear_predictor(df, cohort, use_empirical_gravidity)
    p_more_than_7 = 1.0 / (1.0 + np.exp(-z))

    out = df.copy()
    out["risk_within_7d"] = 1.0 - p_more_than_7
    out["risk_more_than_7d"] = p_more_than_7
    return out


def predict_one(cohort: str = "nuMoM2b", **kwargs) -> float:
    """Score a single patient; returns the risk of delivery within 7 days.

    >>> round(predict_one(maternal_age=28, bmi=26.0, gestational_age=30.0,
    ...                   gravidity=1, preterm_labor=1, pprom=0,
    ...                   effacement=3, cervical_dilation=2.0), 3)
    0.15
    """
    return float(predict(pd.DataFrame([kwargs]), cohort)["risk_within_7d"].iloc[0])
