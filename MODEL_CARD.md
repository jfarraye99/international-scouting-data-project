# Model Card

## Intended use

The Ridge model demonstrates a transparent, reproducible projection workflow for a portfolio project. It predicts the next-season synthetic performance index from current performance, scouting grades, age, signing bonus, country, league, and position-group context.

## Prohibited use

The model must not be used to evaluate, rank, sign, release, or compensate real players. Its training data are simulated and its outputs have no real-world predictive validity.

## Validation design

- Training observations: seasons through 2023.
- Holdout observations: 2024 features predicting 2025 outcomes.
- Metric: mean absolute error compared with a constant-mean baseline.
- Categorical variables: one-hot encoded with unknown-category handling.
- Numeric variables: median imputation and standardization.
- Regularization: Ridge alpha of 5.0.

## Risks and limitations

- Synthetic relationships can make performance appear more predictable than real international development.
- Country and league categories may encode structural opportunity rather than player talent.
- Signing bonus is partly an organizational decision and should not be treated as an independent skill measure.
- Scouting grades require calibration across evaluators and time.
- Small samples and selection bias can distort apparent archetype track records.
- Production deployment would require source governance, back-testing on real historical cohorts, fairness review, and direct scout feedback.

