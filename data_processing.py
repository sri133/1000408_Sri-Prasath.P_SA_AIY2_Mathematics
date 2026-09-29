"""
data_processing.py
------------------
Step 2 of the brief: load, clean and feature-engineer the FootLens dataset.

Pipeline (each step is a small, testable function):
    load_dataset()  ->  read CSV, validate columns, convert "N.A." to NaN
    clean()         ->  rename columns, parse dates, tidy text, engineer features

Key engineered features (all documented in the dashboard's "Research & Data" tab):
    rating_before / rating_after / rating_change   (mean player rating in 3 matches)
    gd_before / gd_during / gd_after                (mean team goal difference)
    ppg_*                                           (mean points per game, 3-1-0)
    tpdi   = gd_before - gd_during                  (Team Performance Drop Index)
    tpdi_z = (tpdi - mean) / std                    (standardised z-score)
    days_out, severity (UEFA-style classes), age_band, injury_type, injury_month
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NA_TOKENS = ["N.A.", "N.A", "NA", "N/A", "n/a", "", "-"]
PHASES = ("before", "during", "after")
POINTS = {"win": 3, "draw": 1, "lose": 0, "loss": 0}

BASE_RENAME = {
    "Name": "player",
    "Team Name": "club",
    "Position": "position",
    "Age": "age",
    "Season": "season",
    "FIFA rating": "fifa_rating",
    "Injury": "injury_raw",
    "Date of Injury": "injury_date",
    "Date of return": "return_date",
}
MATCH_PATTERN = re.compile(
    r"^Match(\d)_(before|missed|after)_(?:injury|match)_(Result|Opposition|GD|Player_rating)$"
)
PHASE_NAME = {"before": "before", "missed": "during", "after": "after"}
FIELD_NAME = {"Result": "result", "Opposition": "opp", "GD": "gd", "Player_rating": "rating"}

MONTH_ORDER = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
SEVERITY_ORDER = ["Minimal (<=7d)", "Mild (8-28d)", "Moderate (29-90d)", "Severe (>90d)"]
AGE_BAND_ORDER = ["<=21", "22-25", "26-29", "30+"]

# Ordered rules: the first regex that matches decides the injury family.
INJURY_RULES: list[tuple[str, str]] = [
    ("Illness / COVID", r"covid|coronavirus|\bill\b|illness|fever|\bcold\b|bronchitis|\bflu\b|virus|infection|appendect|depression"),
    ("Head / Concussion", r"concussion|head|facial|cheekbone|nose|eye|jaw"),
    ("Hamstring", r"hamstring"),
    ("Achilles", r"achilles|archilles"),
    ("Ankle", r"ankle"),
    ("Knee / ACL", r"knee|cruciate|\bacl\b|menisc|patell"),
    ("Calf", r"calf"),
    ("Groin / Adductor", r"groin|adductor|hernia"),
    ("Thigh / Quad", r"thigh|quad|dead leg"),
    ("Hip", r"\bhip\b"),
    ("Back", r"\bback\b|spine|spinal|lumbar"),
    ("Shoulder / Arm", r"shoulder|\barm\b|elbow|wrist|hand|collarbone"),
    ("Foot / Toe", r"foot|toe|metatars"),
    ("Chest / Ribs", r"rib|chest|abdominal"),
    ("Fracture (other)", r"broken|fracture|fibula|tibia"),
    ("Muscle (other)", r"muscle|strain|tear|fitness|knock|bruise|sprain|leg"),
]


class DatasetError(Exception):
    """Raised when the CSV is missing, empty or does not have the expected columns."""


@dataclass
class CleaningReport:
    rows_raw: int = 0
    rows_clean: int = 0
    na_tokens_converted: int = 0
    duplicates_removed: int = 0
    dates_unparseable_injury: int = 0
    ongoing_injuries_present: int = 0
    return_dates_invalid: int = 0
    return_years_auto_corrected: int = 0
    injury_labels_before: int = 0
    injury_labels_after: int = 0
    injury_families: int = 0
    rows_without_ratings: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def expected_raw_columns() -> list[str]:
    cols = list(BASE_RENAME)
    for phase in ("before_injury", "missed_match", "after_injury"):
        for i in (1, 2, 3):
            for f in ("Result", "Opposition", "GD", "Player_rating"):
                if phase == "missed_match" and f == "Player_rating":
                    continue
                cols.append(f"Match{i}_{phase}_{f}")
    return cols


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read + validate + clean. Returns (clean_dataframe, cleaning_report_dict)."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found at '{path.as_posix()}'.")
    try:
        raw = pd.read_csv(path, na_values=NA_TOKENS, keep_default_na=True, skipinitialspace=True)
    except Exception as exc:  # unreadable / corrupted file
        raise DatasetError(f"The CSV could not be read: {exc}") from exc
    if raw.empty:
        raise DatasetError("The CSV file is empty.")
    missing = [c for c in expected_raw_columns() if c not in raw.columns]
    if missing:
        raise DatasetError(
            "The CSV is missing required columns: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        )
    raw_text_na = int((pd.read_csv(path, dtype=str, keep_default_na=False).isin(["N.A.", "N.A"])).sum().sum())
    df, report = clean(raw)
    report.na_tokens_converted = raw_text_na
    return df, report.as_dict()


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give every column a short, readable, snake_case name (e.g. before_2_rating)."""
    mapping = dict(BASE_RENAME)
    for col in df.columns:
        m = MATCH_PATTERN.match(col)
        if m:
            i, phase, field = m.groups()
            mapping[col] = f"{PHASE_NAME[phase]}_{i}_{FIELD_NAME[field]}"
    return df.rename(columns=mapping)


def parse_dates(series: pd.Series) -> pd.Series:
    """Robust date parser: handles 'Oct 15,2022', 'January 4, 2023', 'Present', typos."""
    s = series.astype("string").str.strip().str.replace(r",\s*", ", ", regex=True)
    out = pd.to_datetime(s, format="mixed", errors="coerce")
    out = out.where(out.dt.year >= 2000)  # e.g. 'Mar 7, 0202' is a typo -> NaT
    return out


def clean_injury_label(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"^ack injury$", "back injury", regex=True)  # known typo
    return s


def injury_family(label: str) -> str:
    for family, pattern in INJURY_RULES:
        if re.search(pattern, label):
            return family
    return "Other"


def position_group(pos: str) -> str:
    p = pos.lower()
    if "goalkeeper" in p:
        return "Goalkeeper"
    if "back" in p:
        return "Defender"
    if "midfield" in p:
        return "Midfielder"
    if "winger" in p or "forward" in p or "striker" in p:
        return "Forward"
    return "Other"


def _phase_mean(df: pd.DataFrame, phase: str, field: str) -> pd.Series:
    cols = [f"{phase}_{i}_{field}" for i in (1, 2, 3)]
    return df[cols].mean(axis=1, skipna=True)


def _points_frame(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    cols = [f"{phase}_{i}_result" for i in (1, 2, 3)]
    return df[cols].apply(lambda c: c.str.lower().map(POINTS)).astype(float)


# --------------------------------------------------------------------------- #
# Main cleaning + feature engineering
# --------------------------------------------------------------------------- #
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    rep = CleaningReport(rows_raw=len(raw))
    df = rename_columns(raw.copy())

    # ---- text columns -------------------------------------------------------
    for col in ("player", "club", "position", "season"):
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=["player", "club"])
    rep.duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # ---- numeric columns (text -> float) -------------------------------------
    for col in df.columns:
        if col.endswith("_gd") or col.endswith("_rating"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["fifa_rating"] = pd.to_numeric(df["fifa_rating"], errors="coerce")
    for phase in PHASES:
        for i in (1, 2, 3):
            df[f"{phase}_{i}_result"] = df[f"{phase}_{i}_result"].astype("string").str.strip().str.lower()

    # ---- dates ---------------------------------------------------------------
    rep.ongoing_injuries_present = int(df["return_date"].astype("string").str.strip().str.lower().eq("present").sum())
    df["injury_date"] = parse_dates(df["injury_date"])
    df["return_date"] = parse_dates(df["return_date"])
    rep.dates_unparseable_injury = int(df["injury_date"].isna().sum())

    # A return date before the injury date is usually a year typo (Dec 2020 -> Jan 2020).
    bad = df["return_date"] < df["injury_date"]
    plus_year = df["return_date"] + pd.DateOffset(years=1)
    fixable = bad & ((plus_year - df["injury_date"]).dt.days.between(0, 365))
    df.loc[fixable, "return_date"] = plus_year[fixable]
    rep.return_years_auto_corrected = int(fixable.sum())
    still_bad = df["return_date"] < df["injury_date"]
    rep.return_dates_invalid = int(still_bad.sum() + (df["return_date"].isna().sum() - rep.ongoing_injuries_present).clip(min=0))
    df.loc[still_bad, "return_date"] = pd.NaT

    df["days_out"] = (df["return_date"] - df["injury_date"]).dt.days
    df["is_ongoing"] = df["return_date"].isna()

    # ---- injury labels -------------------------------------------------------
    rep.injury_labels_before = int(df["injury_raw"].nunique())
    df["injury"] = clean_injury_label(df["injury_raw"]).str.title()
    rep.injury_labels_after = int(df["injury"].nunique())
    df["injury_type"] = clean_injury_label(df["injury_raw"]).map(injury_family)
    rep.injury_families = int(df["injury_type"].nunique())
    df["position"] = df["position"].str.title()
    df["position_group"] = df["position"].map(position_group)

    # ---- calendar features -----------------------------------------------------
    df["injury_month"] = df["injury_date"].dt.strftime("%b")
    df["injury_year"] = df["injury_date"].dt.year

    # ---- phase summaries (before / during / after) -----------------------------
    df["rating_before"] = _phase_mean(df, "before", "rating")
    df["rating_after"] = _phase_mean(df, "after", "rating")
    df["rating_change"] = df["rating_after"] - df["rating_before"]
    for phase in PHASES:
        df[f"gd_{phase}"] = _phase_mean(df, phase, "gd")
        pts = _points_frame(df, phase)
        df[f"ppg_{phase}"] = pts.mean(axis=1, skipna=True)
        played = pts.notna().sum(axis=1).replace(0, np.nan)
        df[f"winrate_{phase}"] = (pts.eq(3).sum(axis=1) / played)

    # ---- impact metrics ----------------------------------------------------------
    df["tpdi"] = df["gd_before"] - df["gd_during"]          # Team Performance Drop Index
    df["ppg_drop"] = df["ppg_before"] - df["ppg_during"]
    sd = df["tpdi"].std(ddof=1)
    df["tpdi_z"] = (df["tpdi"] - df["tpdi"].mean()) / sd if sd and not np.isnan(sd) else np.nan
    rep.rows_without_ratings = int(df["rating_change"].isna().sum())

    # ---- categorical bins ---------------------------------------------------------
    df["severity"] = pd.cut(
        df["days_out"], bins=[-np.inf, 7, 28, 90, np.inf], labels=SEVERITY_ORDER
    )
    df["age_band"] = pd.cut(df["age"], bins=[0, 21, 25, 29, 100], labels=AGE_BAND_ORDER)
    df["episode_label"] = (
        df["injury"] + " | " + df["injury_date"].dt.strftime("%d %b %Y").fillna("date n/a") + " | " + df["season"]
    )

    rep.rows_clean = len(df)
    return df, rep
