"""International scouting data-foundation proof of concept.

This project intentionally uses synthetic players and simulated source systems.
It demonstrates a reproducible Python/SQL workflow without representing any
fictional record or result as a real player evaluation.
"""

from __future__ import annotations

import html
import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


COUNTRIES = {
    "Dominican Republic": ["D.R.", "Dominican Rep.", "DOM"],
    "Venezuela": ["VZ", "VEN", "Venezuela"],
    "Mexico": ["México", "MEX", "Mexico"],
    "Japan": ["JPN", "Japan"],
    "South Korea": ["Korea", "KOR", "South Korea"],
    "Cuba": ["CUB", "Cuba"],
    "Colombia": ["COL", "Colombia"],
    "Panama": ["PAN", "Panamá"],
}

LEAGUES = {
    "Dominican Republic": ["Dominican Summer League", "Complex League"],
    "Venezuela": ["Dominican Summer League", "Complex League"],
    "Mexico": ["Mexican League", "Mexican Pacific League"],
    "Japan": ["NPB Farm", "Industrial League"],
    "South Korea": ["KBO Futures", "Independent"],
    "Cuba": ["Cuban National Series", "Independent"],
    "Colombia": ["Colombian Winter League", "Complex League"],
    "Panama": ["Panamanian League", "Complex League"],
}

FIRST_NAMES = [
    "Adrian", "Alejandro", "Andres", "Carlos", "Daniel", "Diego", "Emilio", "Esteban",
    "Felix", "Gabriel", "Hector", "Isaac", "Javier", "Jorge", "Luis", "Marco", "Mateo",
    "Nicolas", "Rafael", "Santiago", "Takeshi", "Haruto", "Ren", "Kaito", "Min-jun",
    "Ji-ho", "Seong-hyun", "Yun-seo", "Kenji", "Daichi", "Omar", "Ruben",
]

LAST_NAMES = [
    "Alvarez", "Benitez", "Castillo", "Dominguez", "Estrada", "Fernandez", "Garcia",
    "Herrera", "Jimenez", "Lopez", "Martinez", "Navarro", "Ortega", "Perez", "Ramirez",
    "Sanchez", "Torres", "Valdez", "Vargas", "Mori", "Sato", "Tanaka", "Yamamoto",
    "Nakamura", "Kim", "Lee", "Park", "Choi", "Han", "Quintero", "Rojas", "Mendez",
]


@dataclass(frozen=True)
class Paths:
    scout: Path = RAW / "scout_reports.csv"
    performance: Path = RAW / "performance_metrics.csv"
    signing: Path = RAW / "signing_records.csv"
    database: Path = OUTPUTS / "international_scouting_poc.sqlite"
    explorer: Path = OUTPUTS / "international_scouting_explorer.html"


def ensure_directories() -> None:
    for path in (RAW, PROCESSED, OUTPUTS):
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def standardize_country(value: object) -> str:
    token = normalize_text(value)
    for canonical, variants in COUNTRIES.items():
        candidates = [canonical, *variants]
        if token in {normalize_text(item) for item in candidates}:
            return canonical
    return str(value).strip() if not pd.isna(value) else "Unknown"


def age_on_date(birth_date: pd.Timestamp, year: int) -> int:
    return int(year - birth_date.year)


def generate_synthetic_sources(paths: Paths) -> None:
    """Create three intentionally inconsistent source extracts."""
    rng = np.random.default_rng(SEED)
    n_players = 320
    countries = list(COUNTRIES)
    country_probs = np.array([0.31, 0.20, 0.13, 0.10, 0.07, 0.07, 0.07, 0.05])
    positions = ["RHP", "LHP", "C", "IF", "OF"]
    position_probs = [0.27, 0.13, 0.12, 0.25, 0.23]

    players: list[dict[str, object]] = []
    used_names: dict[str, int] = {}
    for idx in range(n_players):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        base_name = f"{first} {last}"
        used_names[base_name] = used_names.get(base_name, 0) + 1
        suffix = "" if used_names[base_name] == 1 else f" {used_names[base_name]}"
        full_name = base_name + suffix
        country = rng.choice(countries, p=country_probs)
        position = rng.choice(positions, p=position_probs)
        birth_year = int(rng.integers(1999, 2008))
        birth_month = int(rng.integers(1, 13))
        birth_day = int(rng.integers(1, 28))
        birth_date = pd.Timestamp(birth_year, birth_month, birth_day)
        sign_year = max(2018, birth_year + int(rng.integers(16, 20)))
        sign_year = min(sign_year, 2025)
        talent = float(rng.normal())
        command = float(rng.normal())
        athleticism = float(rng.normal())
        bonus = max(10_000, np.exp(11.0 + 0.75 * talent + rng.normal(0, 0.8)))
        players.append(
            {
                "player_id": f"INT-{idx + 1:04d}",
                "player_name": full_name,
                "birth_date": birth_date.date().isoformat(),
                "country": country,
                "position": position,
                "bats": rng.choice(["R", "L", "S"], p=[0.58, 0.29, 0.13]),
                "throws": "L" if position == "LHP" else rng.choice(["R", "L"], p=[0.89, 0.11]),
                "sign_date": f"{sign_year}-07-02",
                "signing_bonus_usd": round(float(bonus), -3),
                "source_updated_at": f"2025-{int(rng.integers(1, 10)):02d}-{int(rng.integers(1, 28)):02d}",
                "talent": talent,
                "command": command,
                "athleticism": athleticism,
            }
        )

    canonical = pd.DataFrame(players)

    signing = canonical[
        [
            "player_id", "player_name", "birth_date", "country", "position", "bats", "throws",
            "sign_date", "signing_bonus_usd", "source_updated_at",
        ]
    ].copy()
    # Inject source-system problems: a duplicate, a negative bonus, missing country, and alias values.
    alias_mask = rng.choice(signing.index, size=55, replace=False)
    signing.loc[alias_mask, "country"] = signing.loc[alias_mask, "country"].map(
        lambda c: rng.choice(COUNTRIES[c])
    )
    signing.loc[rng.choice(signing.index, size=8, replace=False), "country"] = np.nan
    signing.loc[rng.choice(signing.index, size=2, replace=False), "signing_bonus_usd"] = -5000
    signing = pd.concat([signing, signing.iloc[[7]]], ignore_index=True)

    perf_rows: list[dict[str, object]] = []
    for row in players:
        start = max(2022, int(str(row["sign_date"])[:4]))
        for season in range(start, 2026):
            country = str(row["country"])
            league = rng.choice(LEAGUES[country])
            age = age_on_date(pd.Timestamp(row["birth_date"]), season)
            development = 0.22 * (season - start)
            role = "Pitcher" if row["position"] in {"RHP", "LHP"} else "Position Player"
            player_ref = row["player_id"]
            if rng.random() < 0.08:
                player_ref = ""
            name = str(row["player_name"])
            if rng.random() < 0.10:
                name = name.upper()
            if rng.random() < 0.05:
                name = "  " + name + " "
            country_value = rng.choice(COUNTRIES[country]) if rng.random() < 0.22 else country
            base = {
                "source_player_id": player_ref,
                "player_name": name,
                "birth_date": row["birth_date"],
                "country": country_value,
                "season": season,
                "league": league,
                "role": role,
                "age": age,
                "sample_size": int(rng.integers(80, 620)),
                "k_pct": np.nan,
                "bb_pct": np.nan,
                "iso": np.nan,
                "contact_pct": np.nan,
                "chase_pct": np.nan,
                "avg_fastball_velocity": np.nan,
                "zone_pct": np.nan,
                "sprint_speed": np.nan,
                "biomech_score": np.nan,
            }
            if role == "Pitcher":
                base.update(
                    {
                        "k_pct": np.clip(0.20 + 0.045 * (row["talent"] + development) + rng.normal(0, 0.035), 0.08, 0.42),
                        "bb_pct": np.clip(0.105 - 0.018 * (row["command"] + development) + rng.normal(0, 0.018), 0.03, 0.21),
                        "avg_fastball_velocity": np.clip(89.0 + 2.1 * (row["talent"] + development) + rng.normal(0, 1.2), 82, 99),
                        "zone_pct": np.clip(0.47 + 0.025 * (row["command"] + development) + rng.normal(0, 0.025), 0.34, 0.60),
                        "biomech_score": np.clip(50 + 8 * row["athleticism"] + 2 * development + rng.normal(0, 5), 20, 80),
                    }
                )
            else:
                base.update(
                    {
                        "k_pct": np.clip(0.24 - 0.025 * (row["command"] + development) + rng.normal(0, 0.025), 0.08, 0.38),
                        "bb_pct": np.clip(0.08 + 0.012 * (row["command"] + development) + rng.normal(0, 0.015), 0.02, 0.18),
                        "iso": np.clip(0.115 + 0.035 * (row["talent"] + development) + rng.normal(0, 0.035), 0.02, 0.32),
                        "contact_pct": np.clip(0.73 + 0.025 * (row["command"] + development) + rng.normal(0, 0.025), 0.58, 0.88),
                        "chase_pct": np.clip(0.30 - 0.022 * (row["command"] + development) + rng.normal(0, 0.025), 0.17, 0.45),
                        "sprint_speed": np.clip(26.0 + 0.8 * row["athleticism"] + rng.normal(0, 0.65), 23.5, 30.5),
                        "biomech_score": np.clip(50 + 8 * row["athleticism"] + 2 * development + rng.normal(0, 5), 20, 80),
                    }
                )
            perf_rows.append(base)

    performance = pd.DataFrame(perf_rows)
    # Inject invalid values and one duplicate record.
    if len(performance) > 20:
        performance.loc[5, "k_pct"] = 1.25
        performance.loc[12, "avg_fastball_velocity"] = 112.0
        performance.loc[19, "sample_size"] = -4
        performance = pd.concat([performance, performance.iloc[[22]]], ignore_index=True)

    scout_rows: list[dict[str, object]] = []
    report_id = 1
    for row in players:
        n_reports = int(rng.integers(1, 4))
        for n in range(n_reports):
            report_year = int(rng.integers(2023, 2026))
            name = str(row["player_name"])
            if rng.random() < 0.08:
                name = name.lower()
            source_id = row["player_id"] if rng.random() > 0.12 else ""
            country = str(row["country"])
            scout_rows.append(
                {
                    "report_id": f"SR-{report_id:05d}",
                    "source_player_id": source_id,
                    "player_name": name,
                    "birth_date": row["birth_date"],
                    "country": rng.choice(COUNTRIES[country]) if rng.random() < 0.18 else country,
                    "position": row["position"],
                    "report_date": f"{report_year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}",
                    "overall_future_value": int(np.clip(45 + 7 * row["talent"] + rng.normal(0, 4), 20, 80) // 5 * 5),
                    "athleticism_grade": int(np.clip(50 + 7 * row["athleticism"] + rng.normal(0, 4), 20, 80) // 5 * 5),
                    "makeup_grade": int(np.clip(50 + 5 * row["command"] + rng.normal(0, 5), 20, 80) // 5 * 5),
                    "risk_grade": rng.choice(["Low", "Medium", "High"], p=[0.20, 0.55, 0.25]),
                    "scout_region": rng.choice(["Caribbean", "Latin America", "Pacific Rim", "Mexico"]),
                    "notes": rng.choice(
                        [
                            "Shows repeatable movement patterns and competitive intent.",
                            "Physical projection remains; performance context requires follow-up.",
                            "Advanced feel for age with present execution and development upside.",
                            "Tools are visible, but sample and competition level need validation.",
                        ]
                    ),
                }
            )
            report_id += 1

    scout = pd.DataFrame(scout_rows)
    scout.loc[3, "overall_future_value"] = 95
    scout.loc[10, "birth_date"] = ""
    scout = pd.concat([scout, scout.iloc[[15]]], ignore_index=True)

    signing.to_csv(paths.signing, index=False)
    performance.to_csv(paths.performance, index=False)
    scout.to_csv(paths.scout, index=False)


def add_issue(issues: list[dict[str, object]], source: str, row_id: object, rule: str, severity: str, detail: str) -> None:
    issues.append(
        {"source": source, "row_id": str(row_id), "rule": rule, "severity": severity, "detail": detail}
    )


def resolve_player_ids(frame: pd.DataFrame, signing: pd.DataFrame, source: str, issues: list[dict[str, object]]) -> pd.DataFrame:
    result = frame.copy()
    signing = signing.copy()
    signing["name_key"] = signing["player_name"].map(normalize_text)
    signing["birth_key"] = signing["birth_date"].fillna("").astype(str)
    signing["country_std"] = signing["country"].map(standardize_country)
    valid_ids = set(signing["player_id"].dropna())
    exact = dict(zip(signing["player_id"], signing["player_id"]))
    composite = dict(zip(signing["name_key"] + "|" + signing["birth_key"], signing["player_id"]))
    unique_name = signing.groupby("name_key")["player_id"].agg(list).to_dict()

    resolved: list[str | None] = []
    methods: list[str] = []
    for idx, row in result.iterrows():
        source_id = str(row.get("source_player_id", "")).strip()
        name_key = normalize_text(row.get("player_name", ""))
        birth_key = str(row.get("birth_date", "")).strip()
        if source_id in valid_ids:
            resolved.append(exact[source_id])
            methods.append("source_id")
            continue
        key = name_key + "|" + birth_key
        if key in composite:
            resolved.append(composite[key])
            methods.append("name_birth_date")
            continue
        candidates = unique_name.get(name_key, [])
        if len(candidates) == 1:
            resolved.append(candidates[0])
            methods.append("unique_normalized_name")
            add_issue(issues, source, idx, "fallback_identity_match", "warning", "Matched on unique normalized name only")
        else:
            resolved.append(None)
            methods.append("unmatched")
            add_issue(issues, source, idx, "unmatched_identity", "error", "No reliable canonical player match")
    result["player_id"] = resolved
    result["match_method"] = methods
    return result


def validate_and_transform(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signing = pd.read_csv(paths.signing)
    performance = pd.read_csv(paths.performance)
    scout = pd.read_csv(paths.scout)
    issues: list[dict[str, object]] = []

    signing["country_std"] = signing["country"].map(standardize_country)
    signing["birth_date"] = pd.to_datetime(signing["birth_date"], errors="coerce")
    signing["sign_date"] = pd.to_datetime(signing["sign_date"], errors="coerce")
    signing["signing_bonus_usd"] = pd.to_numeric(signing["signing_bonus_usd"], errors="coerce")

    for idx in signing.index[signing["player_id"].duplicated(keep=False)]:
        add_issue(issues, "signing", idx, "duplicate_player_id", "error", str(signing.loc[idx, "player_id"]))
    for idx in signing.index[signing["country"].isna()]:
        add_issue(issues, "signing", idx, "missing_country", "error", "Country is required")
    for idx in signing.index[signing["signing_bonus_usd"].fillna(-1) < 0]:
        add_issue(issues, "signing", idx, "invalid_bonus", "error", "Signing bonus must be non-negative")

    # Keep the most recently updated valid row for each canonical ID.
    signing_valid = signing[
        signing["player_id"].notna()
        & ~signing["player_id"].duplicated(keep="last")
        & signing["country"].notna()
        & (signing["signing_bonus_usd"].fillna(-1) >= 0)
    ].copy()

    performance = resolve_player_ids(performance, signing_valid, "performance", issues)
    scout = resolve_player_ids(scout, signing_valid, "scout", issues)
    performance["country_std"] = performance["country"].map(standardize_country)
    scout["country_std"] = scout["country"].map(standardize_country)

    perf_dup_cols = ["player_id", "season", "league"]
    for idx in performance.index[performance.duplicated(perf_dup_cols, keep=False)]:
        add_issue(issues, "performance", idx, "duplicate_player_season_league", "error", "Duplicate grain")
    invalid_perf = (
        (performance["sample_size"].fillna(-1) <= 0)
        | (performance["k_pct"].notna() & ~performance["k_pct"].between(0, 0.60))
        | (performance["avg_fastball_velocity"].notna() & ~performance["avg_fastball_velocity"].between(70, 105))
    )
    for idx in performance.index[invalid_perf]:
        add_issue(issues, "performance", idx, "metric_out_of_range", "error", "One or more metrics failed plausible-range validation")

    scout_dup_cols = ["report_id"]
    for idx in scout.index[scout.duplicated(scout_dup_cols, keep=False)]:
        add_issue(issues, "scout", idx, "duplicate_report_id", "error", str(scout.loc[idx, "report_id"]))
    invalid_grade = ~scout["overall_future_value"].between(20, 80)
    for idx in scout.index[invalid_grade]:
        add_issue(issues, "scout", idx, "invalid_scout_grade", "error", "OFV must be on the 20-80 scale")
    for idx in scout.index[pd.to_datetime(scout["birth_date"], errors="coerce").isna()]:
        add_issue(issues, "scout", idx, "missing_birth_date", "warning", "Birth date unavailable in scout source")

    performance_valid = performance[
        performance["player_id"].notna()
        & ~performance.duplicated(perf_dup_cols, keep="last")
        & ~invalid_perf
    ].copy()
    scout_valid = scout[
        scout["player_id"].notna()
        & ~scout["report_id"].duplicated(keep="last")
        & ~invalid_grade
    ].copy()
    scout_valid["report_date"] = pd.to_datetime(scout_valid["report_date"], errors="coerce")

    # Record conflicts rather than silently treating one source as correct.
    country_lookup = signing_valid.set_index("player_id")["country_std"].to_dict()
    for source_name, frame in (("performance", performance_valid), ("scout", scout_valid)):
        conflict = frame["player_id"].map(country_lookup).ne(frame["country_std"])
        for idx in frame.index[conflict.fillna(False)]:
            add_issue(issues, source_name, idx, "country_conflict", "warning", "Source country differs from canonical signing record")

    issues_df = pd.DataFrame(issues)
    return signing_valid, performance_valid, scout_valid, issues_df


def add_performance_index(performance: pd.DataFrame) -> pd.DataFrame:
    frame = performance.copy()
    frame["performance_index"] = np.nan
    for (role, season), idx in frame.groupby(["role", "season"]).groups.items():
        subset = frame.loc[idx]
        if role == "Pitcher":
            columns = {
                "k_pct": 0.30,
                "bb_pct": -0.20,
                "avg_fastball_velocity": 0.20,
                "zone_pct": 0.15,
                "biomech_score": 0.15,
            }
        else:
            columns = {
                "contact_pct": 0.25,
                "iso": 0.25,
                "chase_pct": -0.15,
                "sprint_speed": 0.15,
                "biomech_score": 0.20,
            }
        score = pd.Series(0.0, index=subset.index)
        for column, weight in columns.items():
            values = pd.to_numeric(subset[column], errors="coerce")
            std = values.std(ddof=0)
            z = (values - values.mean()) / (std if std and not np.isnan(std) else 1)
            score = score.add(weight * z.fillna(0), fill_value=0)
        frame.loc[idx, "performance_index"] = score
    return frame


def bonus_tier(value: float) -> str:
    if value < 100_000:
        return "Under $100K"
    if value < 500_000:
        return "$100K-$499K"
    if value < 1_500_000:
        return "$500K-$1.49M"
    return "$1.5M+"


def age_band(value: float) -> str:
    if value <= 18:
        return "18 and under"
    if value <= 20:
        return "19-20"
    if value <= 22:
        return "21-22"
    return "23+"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def build_analytical_tables(
    signing: pd.DataFrame, performance: pd.DataFrame, scout: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    perf = add_performance_index(performance)
    player = signing.copy()
    latest_scout = scout.sort_values("report_date").groupby("player_id", as_index=False).tail(1)
    scout_cols = [
        "player_id", "overall_future_value", "athleticism_grade", "makeup_grade", "risk_grade", "report_date"
    ]
    player_master = player.merge(latest_scout[scout_cols], on="player_id", how="left")

    perf = perf.merge(
        player_master[["player_id", "position", "signing_bonus_usd", "country_std", "overall_future_value"]],
        on="player_id",
        how="left",
        suffixes=("", "_canonical"),
    )
    perf["position_group"] = np.where(perf["position"].isin(["RHP", "LHP"]), "Pitcher", perf["position"])
    perf["age_band"] = perf["age"].map(age_band)
    perf["bonus_tier"] = perf["signing_bonus_usd"].map(bonus_tier)
    perf["next_season_index"] = perf.groupby("player_id")["performance_index"].shift(-1)
    perf["next_season"] = perf.groupby("player_id")["season"].shift(-1)
    perf.loc[perf["next_season"] != perf["season"] + 1, "next_season_index"] = np.nan
    thresholds = perf.groupby(["role", "season"])["next_season_index"].transform(lambda s: s.quantile(0.75))
    perf["positive_outcome"] = np.where(
        perf["next_season_index"].notna(), (perf["next_season_index"] >= thresholds).astype(int), np.nan
    )

    dims = ["country_std", "league", "position_group", "age_band", "bonus_tier"]
    outcome_rows: list[dict[str, object]] = []
    eligible = perf[perf["positive_outcome"].notna()].copy()
    for dim in dims:
        for value, group in eligible.groupby(dim):
            n = len(group)
            if n < 8:
                continue
            successes = int(group["positive_outcome"].sum())
            low, high = wilson_interval(successes, n)
            outcome_rows.append(
                {
                    "dimension": dim,
                    "segment": value,
                    "n": n,
                    "positive_outcomes": successes,
                    "outcome_rate": successes / n,
                    "ci_low": low,
                    "ci_high": high,
                    "avg_next_season_index": group["next_season_index"].mean(),
                }
            )
    archetypes = pd.DataFrame(outcome_rows).sort_values(["dimension", "outcome_rate"], ascending=[True, False])

    # Transparent time-based projection model. The synthetic nature of the data is disclosed everywhere.
    model_data = eligible.copy()
    numeric = [
        "age", "sample_size", "k_pct", "bb_pct", "iso", "contact_pct", "chase_pct",
        "avg_fastball_velocity", "zone_pct", "sprint_speed", "biomech_score",
        "signing_bonus_usd", "overall_future_value", "performance_index",
    ]
    categorical = ["country_std", "league", "position_group", "age_band", "bonus_tier"]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = Pipeline([("preprocess", pre), ("model", Ridge(alpha=5.0))])
    train = model_data[model_data["season"] <= 2023]
    test = model_data[model_data["season"] == 2024]
    features = numeric + categorical
    model.fit(train[features], train["next_season_index"])
    predictions = model.predict(test[features])
    baseline = np.repeat(train["next_season_index"].mean(), len(test))
    metrics = {
        "test_rows": float(len(test)),
        "model_mae": float(mean_absolute_error(test["next_season_index"], predictions)),
        "baseline_mae": float(mean_absolute_error(test["next_season_index"], baseline)),
    }

    current = perf[perf["season"] == 2025].copy()
    current["projected_next_index"] = model.predict(current[features])
    current["projection_percentile"] = current["projected_next_index"].rank(pct=True)
    projection_cols = [
        "player_id", "player_name", "country_std", "league", "position", "age", "bonus_tier",
        "performance_index", "projected_next_index", "projection_percentile", "overall_future_value",
    ]
    current = current.drop(columns=["player_name"], errors="ignore")
    projections = current.merge(player_master[["player_id", "player_name"]], on="player_id", how="left")[projection_cols]
    return player_master, perf, archetypes, projections, metrics


def create_database(paths: Paths, tables: dict[str, pd.DataFrame]) -> None:
    if paths.database.exists():
        paths.database.unlink()
    with sqlite3.connect(paths.database) as conn:
        for name, frame in tables.items():
            frame.to_sql(name, conn, index=False, if_exists="replace")
        conn.executescript(
            """
            CREATE VIEW v_archetype_track_record AS
            SELECT dimension, segment, n, positive_outcomes,
                   ROUND(outcome_rate, 4) AS outcome_rate,
                   ROUND(ci_low, 4) AS ci_low,
                   ROUND(ci_high, 4) AS ci_high
            FROM archetype_outcomes
            WHERE n >= 8;

            CREATE VIEW v_data_quality_queue AS
            SELECT source, row_id, rule, severity, detail
            FROM data_quality_issues
            ORDER BY CASE severity WHEN 'error' THEN 1 ELSE 2 END, source, row_id;

            CREATE VIEW v_player_projection_board AS
            SELECT player_id, player_name, country_std, league, position, age, bonus_tier,
                   ROUND(performance_index, 3) AS performance_index,
                   ROUND(projected_next_index, 3) AS projected_next_index,
                   ROUND(projection_percentile, 3) AS projection_percentile,
                   overall_future_value
            FROM model_scores;
            """
        )


def render_explorer(paths: Paths, archetypes: pd.DataFrame, issues: pd.DataFrame, summary: dict[str, object]) -> None:
    records = archetypes.replace({np.nan: None}).to_dict(orient="records")
    for record in records:
        for key in ("outcome_rate", "ci_low", "ci_high", "avg_next_season_index"):
            if record.get(key) is not None:
                record[key] = round(float(record[key]), 4)
    issue_counts = issues.groupby(["severity", "rule"]).size().reset_index(name="count").to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=True)
    issue_payload = json.dumps(issue_counts, ensure_ascii=True)
    summary_payload = json.dumps(summary, ensure_ascii=True)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>International Scouting Data Project</title>
<style>
:root{{--navy:#13294b;--blue:#1d5fa7;--sky:#eaf2fb;--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--warn:#9a6700;}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,Helvetica,sans-serif;color:var(--ink);background:#f5f7fa}}
header{{background:var(--navy);color:white;padding:28px 5vw 24px}} header h1{{margin:0 0 8px;font-size:28px}} header p{{margin:0;max-width:920px;line-height:1.45;color:#dbe8f7}}
main{{max-width:1180px;margin:0 auto;padding:24px}} .notice{{background:#fff8e6;border-left:4px solid #d9a400;padding:12px 16px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:18px}} .card{{background:white;border:1px solid var(--line);border-radius:8px;padding:16px}}
.metric{{font-size:28px;font-weight:700;color:var(--navy)}} .label{{color:var(--muted);font-size:13px;margin-top:4px}}
.controls{{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end}} label{{font-size:12px;color:var(--muted);display:block;margin-bottom:5px}}
select,button{{width:100%;padding:10px;border:1px solid #b9c5d2;border-radius:5px;background:white}} button{{background:var(--blue);color:white;border:0;font-weight:700;cursor:pointer}}
table{{width:100%;border-collapse:collapse;margin-top:14px;font-size:14px}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}} th{{background:var(--sky);color:var(--navy)}}
.bar{{height:8px;background:#e7edf4;border-radius:6px;overflow:hidden;min-width:100px}} .bar span{{display:block;height:100%;background:var(--blue)}}
.small{{font-size:12px;color:var(--muted)}} h2{{color:var(--navy);font-size:19px;margin:0 0 12px}} .section{{margin-top:18px}}
.definition{{line-height:1.55;margin:0 0 14px}} details{{margin:0 0 16px;padding:10px 12px;background:#f7f9fc;border:1px solid var(--line);border-radius:5px}} summary{{font-weight:700;color:var(--navy);cursor:pointer}} details p{{margin:8px 0 0;line-height:1.5}}
.guide{{margin-bottom:18px}} .guide p{{line-height:1.5}} .steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .step{{padding:12px;background:#f7f9fc;border:1px solid var(--line);border-radius:5px;line-height:1.45}} .step strong{{display:block;color:var(--navy);margin-bottom:4px}}
.interpretation{{margin-top:14px;padding:12px 14px;background:var(--sky);border-left:4px solid var(--blue);line-height:1.5}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}.controls{{grid-template-columns:1fr}}}}
@media(max-width:900px){{.steps{{grid-template-columns:1fr 1fr}}}} @media(max-width:520px){{.steps{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><h1>International Scouting Data Project</h1><p>A working prototype for bringing separate scouting, performance, and signing information into one dependable player view.</p></header>
<main>
<div class="notice"><strong>About the data:</strong> Every player record and outcome is simulated. The workflow is functional and reproducible; the player rankings are only illustrative.</div>
<div class="grid" id="metrics"></div>
<div class="card guide">
  <h2>How a Baseball Staff Could Use This Tool</h2>
  <p>This prototype is designed to help a staff decide <strong>where deeper scouting, verification, or analysis is warranted</strong>. It organizes the evidence and makes uncertainty visible; it does not replace individual evaluation or make a signing decision.</p>
  <div class="steps">
    <div class="step"><strong>1. Check the data</strong>Review the quality queue and resolve material errors before relying on a comparison.</div>
    <div class="step"><strong>2. Ask one question</strong>Choose a country, league, position, age band, or bonus tier to examine.</div>
    <div class="step"><strong>3. Judge the evidence</strong>Read the sample, observed rate, and uncertainty interval together.</div>
    <div class="step"><strong>4. Choose a follow-up</strong>Use the result to prioritize additional scouting, video, medical, or analytical review.</div>
  </div>
</div>
<div class="card">
  <h2>Historical Outcome Comparison</h2>
  <p class="definition">A <strong>top-quartile next-season result</strong> means that a player's following-season composite performance index ranked in the top 25% among players of the same role and comparison season. The displayed rate is the share of eligible player-seasons within each group that reached that threshold. These results are descriptive, based entirely on simulated data, and should not be interpreted as probabilities of signing success, advancement, or MLB contribution.</p>
  <details><summary>How performance is measured</summary><p><strong>Pitchers:</strong> strikeout rate, walk rate, average fastball velocity, zone rate, and biomechanics score. <strong>Position players:</strong> contact rate, isolated power, chase rate, sprint speed, and biomechanics score. Each input is standardized within role and season before the weighted composite is calculated.</p></details>
  <div class="controls">
    <div><label for="dimension">Explore by</label><select id="dimension"></select></div>
    <div><label for="segment">Segment</label><select id="segment"></select></div>
    <div><button id="reset">Reset filters</button></div>
  </div>
  <div class="interpretation" id="interpretation">Choose a specific segment to generate a plain-language interpretation.</div>
  <table><thead><tr><th>Dimension</th><th>Segment</th><th>Eligible player-seasons</th><th>Top-quartile next seasons</th><th>Top-quartile rate</th><th>95% uncertainty interval</th></tr></thead><tbody id="rows"></tbody></table>
  <p class="small">Only player-seasons with a consecutive following season are eligible. Wilson intervals communicate uncertainty, and groups with fewer than eight observations are suppressed. The rate is descriptive and is not model accuracy.</p>
</div>
<div class="card section"><h2>Data Quality Review Queue</h2><p class="definition">This queue shows records that were missing, duplicated, unmatched, or outside an accepted range. Errors are excluded from the curated analysis; warnings remain visible for review. In production, each exception would also have an owner, status, resolution, and audit history.</p><table><thead><tr><th>Severity</th><th>Rule</th><th>Flagged rows</th></tr></thead><tbody id="issues"></tbody></table></div>
</main>
<script>
const data={payload}; const quality={issue_payload}; const summary={summary_payload};
const dim=document.getElementById('dimension'), seg=document.getElementById('segment'), tbody=document.getElementById('rows');
const labels={{country_std:'Country',league:'League',position_group:'Position group',age_band:'Age band',bonus_tier:'Bonus tier'}};
function option(v,t){{const o=document.createElement('option');o.value=v;o.textContent=t;return o}}
function fillDimensions(){{dim.innerHTML='';dim.appendChild(option('all','All dimensions'));[...new Set(data.map(d=>d.dimension))].forEach(d=>dim.appendChild(option(d,labels[d]||d)));}}
function fillSegments(){{seg.innerHTML='';seg.appendChild(option('all','All segments'));const rows=dim.value==='all'?data:data.filter(d=>d.dimension===dim.value);[...new Set(rows.map(d=>d.segment))].sort().forEach(s=>seg.appendChild(option(s,s)));}}
function pct(v){{return (100*v).toFixed(1)+'%'}}
function render(){{let rows=data;if(dim.value!=='all')rows=rows.filter(d=>d.dimension===dim.value);if(seg.value!=='all')rows=rows.filter(d=>d.segment===seg.value);rows=[...rows].sort((a,b)=>b.outcome_rate-a.outcome_rate);tbody.innerHTML=rows.map(r=>`<tr><td>${{labels[r.dimension]||r.dimension}}</td><td><strong>${{r.segment}}</strong></td><td>${{r.n}}</td><td>${{r.positive_outcomes}}</td><td>${{pct(r.outcome_rate)}}<div class="bar"><span style="width:${{100*r.outcome_rate}}%"></span></div></td><td>${{pct(r.ci_low)}}-${{pct(r.ci_high)}}</td></tr>`).join('');const box=document.getElementById('interpretation');if(seg.value!=='all'&&rows.length===1){{const r=rows[0],base=summary.overall_outcome_rate,direction=r.outcome_rate>base?'above':r.outcome_rate<base?'below':'equal to';const evidence=r.n<20?'an exploratory signal because the sample is small':r.n<50?'directional evidence that still warrants additional validation':'a more stable descriptive comparison, though it is not causal';box.innerHTML=`<strong>${{r.segment}}:</strong> ${{r.positive_outcomes}} of ${{r.n}} eligible player-seasons (${{pct(r.outcome_rate)}}) produced a top-quartile next season. That is ${{direction}} the ${{pct(base)}} overall synthetic baseline. The 95% uncertainty interval is ${{pct(r.ci_low)}}-${{pct(r.ci_high)}}. Treat this as ${{evidence}}. Appropriate next step: review the underlying players and add scouting, video, medical, competition-level, and acquisition-cost context before acting.`;}}else{{box.textContent=`Choose a specific segment to compare its result with the ${{pct(summary.overall_outcome_rate)}} overall synthetic baseline and receive an interpretation.`;}}}}
fillDimensions();fillSegments();render();dim.onchange=()=>{{fillSegments();render()}};seg.onchange=render;document.getElementById('reset').onclick=()=>{{dim.value='all';fillSegments();render()}};
document.getElementById('metrics').innerHTML=[['Players',summary.players],['Source rows',summary.source_rows],['Eligible player-seasons',summary.eligible_player_seasons],['Flagged issues',summary.issues],['Validated match rate',pct(summary.match_rate)],['Overall top-quartile rate',pct(summary.overall_outcome_rate)]].map(x=>`<div class="card"><div class="metric">${{x[1]}}</div><div class="label">${{x[0]}}</div></div>`).join('');
document.getElementById('issues').innerHTML=quality.map(q=>`<tr><td>${{q.severity}}</td><td>${{q.rule.replaceAll('_',' ')}}</td><td>${{q.count}}</td></tr>`).join('');
</script>
</body></html>"""
    paths.explorer.write_text(page, encoding="utf-8")


def main() -> None:
    ensure_directories()
    paths = Paths()
    generate_synthetic_sources(paths)
    signing, performance, scout, issues = validate_and_transform(paths)
    player_master, perf, archetypes, projections, model_metrics = build_analytical_tables(signing, performance, scout)

    player_master.to_csv(PROCESSED / "player_master.csv", index=False)
    perf.to_csv(PROCESSED / "performance_enriched.csv", index=False)
    archetypes.to_csv(PROCESSED / "archetype_outcomes.csv", index=False)
    projections.to_csv(PROCESSED / "model_scores.csv", index=False)
    issues.to_csv(PROCESSED / "data_quality_issues.csv", index=False)

    raw_counts = {path.name: len(pd.read_csv(path)) for path in (paths.signing, paths.performance, paths.scout)}
    raw_rows = sum(raw_counts.values())
    total_matchable = raw_counts[paths.performance.name] + raw_counts[paths.scout.name]
    unmatched_rows = int((issues["rule"] == "unmatched_identity").sum())
    summary = {
        "players": int(len(player_master)),
        "source_rows": int(raw_rows),
        "issues": int(len(issues)),
        "match_rate": float((total_matchable - unmatched_rows) / total_matchable),
        "eligible_player_seasons": int(perf["positive_outcome"].notna().sum()),
        "overall_outcome_rate": float(perf["positive_outcome"].dropna().mean()),
        **model_metrics,
    }
    (PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    create_database(
        paths,
        {
            "player_master": player_master,
            "performance_enriched": perf,
            "archetype_outcomes": archetypes,
            "model_scores": projections,
            "data_quality_issues": issues,
        },
    )
    render_explorer(paths, archetypes, issues, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
