"""Tests for the published-coefficient scorer.

Run with:  python -m pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptb_model import COHORTS, linear_predictor, predict, predict_one  # noqa: E402


def example_frame():
    return pd.DataFrame([
        # a patient with ruptured membranes and an advanced cervical exam
        dict(maternal_age=24, bmi=23.0, gestational_age=28.0, gravidity=1,
             preterm_labor=1, pprom=1, effacement=5, cervical_dilation=4.0),
        # a patient with intact membranes and a closed cervix
        dict(maternal_age=32, bmi=31.0, gestational_age=34.0, gravidity=3,
             preterm_labor=0, pprom=0, effacement=1, cervical_dilation=0.0),
    ])


@pytest.mark.parametrize("cohort", sorted(COHORTS))
def test_probabilities_are_valid(cohort):
    df = example_frame()
    out = predict(df, cohort=cohort)
    p = out["risk_within_7d"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(out["risk_within_7d"] + out["risk_more_than_7d"], 1.0)


@pytest.mark.parametrize("cohort", sorted(COHORTS))
def test_ruptured_membranes_raises_risk(cohort):
    """PPROM carries the largest negative coefficient in every variant, so the
    first example patient must score higher than the second."""
    p = predict(example_frame(), cohort=cohort)["risk_within_7d"].to_numpy()
    assert p[0] > p[1]


def test_mean_patient_reduces_to_intercept():
    """A patient at the mean of every predictor has z == intercept."""
    spec = COHORTS["nuMoM2b"]
    df = pd.DataFrame([{k: spec["means"][k] for k in spec["coefficients"]}])
    z = linear_predictor(df, "nuMoM2b")
    assert z == pytest.approx(spec["intercept"], abs=1e-12)


def test_tjuh_variant_needs_no_gravidity():
    df = example_frame().drop(columns=["gravidity"])
    out = predict(df, cohort="TJUH")
    assert out["risk_within_7d"].notna().all()


def test_missing_column_raises():
    df = example_frame().drop(columns=["pprom"])
    with pytest.raises(KeyError, match="pprom"):
        predict(df, cohort="nuMoM2b")


def test_missing_value_yields_nan_not_error():
    df = example_frame()
    df.loc[0, "bmi"] = np.nan
    out = predict(df, cohort="nuMoM2b")
    assert np.isnan(out.loc[0, "risk_within_7d"])
    assert not np.isnan(out.loc[1, "risk_within_7d"])


def test_predict_one_matches_predict():
    row = example_frame().iloc[0].to_dict()
    assert predict_one(**row) == pytest.approx(
        predict(example_frame(), "nuMoM2b")["risk_within_7d"].iloc[0])


def test_empirical_gravidity_flag_changes_only_gravidity_term():
    df = example_frame()
    a = predict(df, "IU")["risk_within_7d"].to_numpy()
    b = predict(df, "IU", use_empirical_gravidity=True)["risk_within_7d"].to_numpy()
    assert not np.allclose(a, b)          # it does something
    assert np.max(np.abs(a - b)) < 0.05   # but the effect is small
    # the TJUH variant has no gravidity term, so the flag is inert there
    t = df.drop(columns=["gravidity"])
    assert np.allclose(predict(t, "TJUH")["risk_within_7d"],
                       predict(t, "TJUH", use_empirical_gravidity=True)["risk_within_7d"])


def test_published_constants_are_unchanged():
    """Guards against accidental edits to the published values."""
    assert COHORTS["nuMoM2b"]["coefficients"]["pprom"] == -1.88
    assert COHORTS["IU"]["coefficients"]["pprom"] == -1.74
    assert COHORTS["TJUH"]["coefficients"]["pprom"] == -1.39
    assert COHORTS["nuMoM2b"]["intercept"] == 0.17
    assert COHORTS["IU"]["intercept"] == 0.15
    assert COHORTS["TJUH"]["intercept"] == 0.11
    assert "gravidity" not in COHORTS["TJUH"]["coefficients"]
