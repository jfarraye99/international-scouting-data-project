# International Scouting Data Project

In coaching, I rarely had every piece of player information in one place. Video, pitch data, game results, and staff observations could point in different directions. I built this working prototype to explore the same problem across separate scouting, performance, and signing sources: How do you create a player record people can trust, show what still needs review, and make the result useful to a baseball staff?

## Integrity and scope

All player names, records, metrics, bonuses, and outcomes in this repository are synthetic. They were generated solely to demonstrate system design, validation logic, SQL delivery, analysis, and stakeholder communication. Nothing in the project should be interpreted as an evaluation of a real player, league, country, club, or market.

I defined the baseball problem, data structure, matching priorities, validation rules, and intended outputs. I used AI coding tools to help implement the prototype, then reviewed the workflow, ran the tests, and documented the limitations. The code is reproducible, but the baseball conclusions are only illustrations because the source data are simulated.

## What I built

- Ingestion of three intentionally inconsistent source extracts: scouting reports, performance metrics, and signing records.
- Canonical identity resolution using source IDs, normalized names, and birth dates.
- Data-quality rules covering duplicates, missing critical fields, implausible metrics, invalid 20-80 grades, unmatched identities, and source conflicts.
- A documented exception queue instead of silently correcting questionable records.
- Reproducible Python transformation and a SQLite analytical layer with stakeholder-ready views.
- Archetype track records by country, league, position group, age band, and signing-bonus tier.
- Sample-size suppression and Wilson confidence intervals to reduce overconfidence in small groups.
- A transparent time-based Ridge model used only to test the workflow.
- A dependency-free HTML explorer that non-technical users can open locally.

## How the outcome measure works

A **top-quartile next-season result** means that a player's following-season composite performance index ranked in the top 25% among players of the same role and comparison season. Only player-seasons with a consecutive following season are eligible.

For pitchers, the composite uses strikeout rate, walk rate, average fastball velocity, zone rate, and biomechanics score. For position players, it uses contact rate, isolated power, chase rate, sprint speed, and biomechanics score. Inputs are standardized within role and season before the weighted composite is calculated.

The rate shown for a country, league, position group, age band, or bonus tier is the number of eligible observations reaching that threshold divided by all eligible observations in that group. It is a retrospective comparison of simulated player groups, not model accuracy or a predicted probability of signing success, advancement, or MLB contribution.

## Run the project

From the project folder:

```bash
python src/pipeline.py
python -m unittest discover -s tests -v
```

Required packages are listed in `requirements.txt`. The pipeline uses a fixed random seed, so it produces the same synthetic sources and outputs on every run.

## Key outputs

- `outputs/international_scouting_explorer.html` - self-service browser interface.
- `outputs/international_scouting_poc.sqlite` - SQLite database with curated tables and views.
- `data/processed/data_quality_issues.csv` - row-level exception queue.
- `data/processed/archetype_outcomes.csv` - grouped track records with uncertainty.
- `data/processed/model_scores.csv` - illustrative next-season projections.
- `DECISION_LOG.md` - the reasoning behind the major design choices.

## What I would do next with real data

1. Replace synthetic files with governed vendor and internal feeds.
2. Establish stable cross-system identifiers and stewardship rules.
3. Add source freshness, lineage, schema-drift, and reconciliation monitoring.
4. Validate league translations and outcome definitions with scouts and analysts.
5. Deploy role-based dashboards and an authenticated exception-management workflow.
6. Monitor model calibration, segment stability, and decision impact over time.

## Data sources and references

The project structure was informed by the [Los Angeles Dodgers job description for Manager, International Baseball Strategy and Information](https://www.teamworkonline.com/baseball-jobs/los-angeles-dodgers-jobs/los-angeles-dodgers/manager-international-baseball-strategy-and-information-2190285). Metric concepts follow the public [MLB Statcast glossary](https://www.mlb.com/glossary/statcast). The project does not download, redistribute, or claim to model proprietary Dodgers information.
