"""
Refit the prediction model and print the coefficients.

This is the code that produced the constants published in the Supplementary
Table (and hard-coded in `ptb_model.py`). It is included so that the derivation
of the weights is inspectable, not only their values.

The patient-level data are not distributed with this repository (see README),
so running this requires a local file with one row per patient and the columns
listed in `PREDICTORS` plus an `interval_days` column giving the number of days
from antenatal corticosteroid administration to delivery.

    python src/fit_model.py --data mycohort.csv
    python src/fit_model.py --data mycohort.csv --no-gravidity   # TJUH variant

Method, as published: predictors are standardized (zero mean, unit variance);
L2-penalized logistic regression is fitted with the regularization strength C
and iteration cap chosen by 5-fold cross-validated grid search; the outcome is
dichotomized as delivery within 7 days versus more than 7 days, with the fitted
positive class being "more than 7 days" (hence the published sign convention --
a negative coefficient favors delivery within 7 days).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Predictor order used throughout. Gravidity is dropped for the TJUH variant.
PREDICTORS = [
    "maternal_age",
    "bmi",
    "gestational_age",
    "gravidity",
    "preterm_labor",
    "pprom",
    "effacement",
    "cervical_dilation",
]

PARAM_GRID = [{
    "C": np.logspace(-0.5, 0, 20),
    "solver": ["lbfgs"],
    "max_iter": [50, 60, 70, 80, 90, 100],
}]


def load(path, use_gravidity=True):
    df = pd.read_csv(path)

    cols = list(PREDICTORS) if use_gravidity else [c for c in PREDICTORS if c != "gravidity"]
    missing = [c for c in cols + ["interval_days"] if c not in df.columns]
    if missing:
        sys.exit(f"error: input is missing required column(s): {missing}")

    # Outcome: 1 = delivered more than 7 days after administration.
    y = (pd.to_numeric(df["interval_days"], errors="coerce") > 7).astype(int).to_numpy()

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    # Mode for binary predictors, mean otherwise -- as in the original analysis.
    for c in cols:
        fill = X[c].mode()[0] if X[c].nunique() == 2 else X[c].mean()
        X[c] = X[c].fillna(fill)

    return X, y, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="CSV with one row per patient (see README for the schema)")
    ap.add_argument("--no-gravidity", action="store_true",
                    help="fit the 7-predictor variant (the TJUH model)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the CV split")
    args = ap.parse_args()

    X, y, cols = load(args.data, use_gravidity=not args.no_gravidity)
    print(f"n = {len(y)}  ({int((y == 0).sum())} delivered within 7 days, "
          f"{int(y.sum())} after 7 days)\n")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    search = GridSearchCV(LogisticRegression(random_state=args.seed),
                          param_grid=PARAM_GRID, cv=5, n_jobs=-1)
    search.fit(Xs, y)
    model = search.best_estimator_

    print(f"best C = {model.C:.4f}, max_iter = {model.max_iter}\n")
    print(f"{'predictor':20s} {'mean':>8s} {'sd':>8s} {'coefficient':>12s}")
    for i, c in enumerate(cols):
        print(f"{c:20s} {scaler.mean_[i]:8.2f} {scaler.scale_[i]:8.2f} "
              f"{model.coef_[0][i]:12.2f}")
    print(f"{'intercept':20s} {'':>8s} {'':>8s} {model.intercept_[0]:12.2f}")
    print("\nA positive coefficient indicates a higher likelihood of delivery more "
          "than 7 days\nafter administration; a negative coefficient, within 7 days.")


if __name__ == "__main__":
    main()
