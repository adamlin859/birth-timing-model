"""
Command-line batch scorer.

    python src/score.py --input patients.csv --output scored.csv
    python src/score.py --input patients.csv --cohort TJUH

Reads a CSV with one row per patient, appends the predicted risk of delivery
within 7 days, and writes the result. Columns other than the predictors are
carried through unchanged, so an identifier column can be kept for joining.

See README.md for the input schema and variable encodings.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ptb_model import COHORTS, predict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="input CSV, one row per patient")
    ap.add_argument("--output", help="output CSV (default: stdout)")
    ap.add_argument("--cohort", default="nuMoM2b", choices=sorted(COHORTS),
                    help="which published model variant to apply (default: nuMoM2b)")
    ap.add_argument("--empirical-gravidity", action="store_true",
                    help="standardize gravidity with its empirical mean/SD "
                         "rather than the published values (see README)")
    args = ap.parse_args(argv)

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        sys.exit(f"error: no such file: {args.input}")

    try:
        out = predict(df, cohort=args.cohort,
                      use_empirical_gravidity=args.empirical_gravidity)
    except KeyError as e:
        sys.exit(f"error: {e.args[0]}")

    n_missing = int(out["risk_within_7d"].isna().sum())
    if n_missing:
        print(f"warning: {n_missing} of {len(out)} row(s) had a missing or "
              f"non-numeric predictor and were not scored", file=sys.stderr)

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out)} rows to {args.output}", file=sys.stderr)
    else:
        out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
