"""
analytics.py
------------
Step 3 of the brief: exploratory data analysis + the mathematics used by the dashboard.

Everything here is pure pandas / NumPy / SciPy (no Streamlit), so it can be unit-tested
and re-used in a notebook. Formulas are documented next to the code that implements them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data_processing import MONTH_ORDER, PHASES, POINTS, SEVERITY_ORDER

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Small maths helpers
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion  p = k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_label(d: float) -> str:
    d = abs(d)
    if np.isnan(d):
        return "n/a"
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def sig_label(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    return "significant" if p < ALPHA else "not significant"


def _ci_mean(x: pd.Series, level: float = 0.95) -> tuple[float, float]:
    x = x.dropna()
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return (x.mean() - t * se, x.mean() + t * se)


# --------------------------------------------------------------------------- #
# KPI + groupby / agg summaries
# --------------------------------------------------------------------------- #
def headline_kpis(df: pd.DataFrame) -> dict:
    rec = record_table(df)
    r = rec.set_index("phase")
    return {
        "injuries": len(df),
        "players": df["player"].nunique(),
        "clubs": df["club"].nunique(),
        "avg_days_out": df["days_out"].mean(),
        "avg_tpdi": df["tpdi"].mean(),
        "avg_rating_change": df["rating_change"].mean(),
        "winrate_before": r.loc["before", "win_rate"] if "before" in r.index else np.nan,
        "winrate_during": r.loc["during", "win_rate"] if "during" in r.index else np.nan,
        "ppg_before": r.loc["before", "ppg"] if "before" in r.index else np.nan,
        "ppg_during": r.loc["during", "ppg"] if "during" in r.index else np.nan,
    }


def top_injuries(df: pd.DataFrame, by: str = "injury_type", n: int = 10, min_cases: int = 3) -> pd.DataFrame:
    """Mean TPDI per injury group with a 95% t-interval; only groups with >= min_cases."""
    g = (
        df.dropna(subset=["tpdi"])
        .groupby(by)
        .agg(
            cases=("tpdi", "size"),
            mean_tpdi=("tpdi", "mean"),
            sd_tpdi=("tpdi", "std"),
            mean_ppg_drop=("ppg_drop", "mean"),
            avg_days_out=("days_out", "mean"),
        )
        .reset_index()
        .rename(columns={by: "injury"})
    )
    g = g[g["cases"] >= min_cases].copy()
    g["sem"] = g["sd_tpdi"] / np.sqrt(g["cases"])
    tcrit = stats.t.ppf(0.975, (g["cases"] - 1).clip(lower=1))
    g["ci95"] = tcrit * g["sem"]
    g = g.sort_values("mean_tpdi", ascending=False).head(n).reset_index(drop=True)
    g.index = g.index + 1
    return g


def worst_episodes(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = ["player", "club", "season", "injury", "days_out", "gd_before", "gd_during", "tpdi", "ppg_drop"]
    out = df.dropna(subset=["tpdi"]).sort_values("tpdi", ascending=False).head(n)[cols].reset_index(drop=True)
    out.index = out.index + 1
    return out


def club_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-club injury burden.  IBI = sum(days_out * FIFA/100) / number_of_seasons."""
    d = df.copy()
    d["star_days"] = d["days_out"] * d["fifa_rating"] / 100.0
    n_seasons = max(d["season"].nunique(), 1)
    g = (
        d.groupby("club")
        .agg(
            injuries=("player", "size"),
            players=("player", "nunique"),
            total_days_out=("days_out", "sum"),
            avg_days_out=("days_out", "mean"),
            mean_tpdi=("tpdi", "mean"),
            mean_ppg_drop=("ppg_drop", "mean"),
            star_days=("star_days", "sum"),
        )
        .reset_index()
    )
    g["ibi_per_season"] = g["star_days"] / n_seasons
    sd = g["ibi_per_season"].std(ddof=1)
    g["ibi_z"] = (g["ibi_per_season"] - g["ibi_per_season"].mean()) / sd if sd and not np.isnan(sd) else 0.0
    return g.sort_values("ibi_per_season", ascending=False).reset_index(drop=True)


def player_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by player and summarise the before / during / after phases."""
    g = (
        df.groupby("player")
        .agg(
            club=("club", lambda s: ", ".join(sorted(set(s)))),
            injuries=("player", "size"),
            total_days_out=("days_out", "sum"),
            rating_before=("rating_before", "mean"),
            rating_after=("rating_after", "mean"),
            rating_change=("rating_change", "mean"),
            gd_before=("gd_before", "mean"),
            gd_during=("gd_during", "mean"),
            gd_after=("gd_after", "mean"),
            ppg_before=("ppg_before", "mean"),
            ppg_during=("ppg_during", "mean"),
            ppg_after=("ppg_after", "mean"),
            mean_tpdi=("tpdi", "mean"),
        )
        .reset_index()
    )
    return g.sort_values(["injuries", "total_days_out"], ascending=False).reset_index(drop=True)


def most_injured(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return player_summary(df).head(n)[["player", "club", "injuries", "total_days_out", "mean_tpdi"]]


def comeback_leaderboard(df: pd.DataFrame, n: int = 15, min_days: int = 0) -> pd.DataFrame:
    """Episodes ranked by rating improvement  (mean rating after - mean rating before)."""
    d = df.dropna(subset=["rating_change"]).copy()
    d = d[d["days_out"].fillna(0) >= min_days]
    sd = d["rating_change"].std(ddof=1)
    d["comeback_z"] = (d["rating_change"] - d["rating_change"].mean()) / sd if sd and not np.isnan(sd) else 0.0
    cols = ["player", "club", "season", "injury_type", "days_out", "rating_before", "rating_after", "rating_change", "comeback_z"]
    out = d.sort_values("rating_change", ascending=False).head(n)[cols].reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "Rank"
    return out


def improvers_decliners(df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    ps = player_summary(df).dropna(subset=["rating_change"])
    cols = ["player", "club", "injuries", "rating_before", "rating_after", "rating_change"]
    return (
        ps.sort_values("rating_change", ascending=False).head(n)[cols].reset_index(drop=True),
        ps.sort_values("rating_change").head(n)[cols].reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Pivot tables
# --------------------------------------------------------------------------- #
def pre_post_pivot(df: pd.DataFrame, index: str = "club") -> pd.DataFrame:
    pv = pd.pivot_table(
        df, index=index, values=["rating_before", "rating_after", "rating_change"], aggfunc="mean", observed=True
    )
    pv = pv[["rating_before", "rating_after", "rating_change"]].round(3)
    return pv.sort_values("rating_change", ascending=False)


def recovery_trend(df: pd.DataFrame, by: str = "severity") -> pd.DataFrame:
    """Recovery statistics by severity / age band / position group."""
    g = (
        df.groupby(by, observed=True)
        .agg(
            episodes=("player", "size"),
            avg_days_out=("days_out", "mean"),
            median_days_out=("days_out", "median"),
            mean_rating_change=("rating_change", "mean"),
            sd_rating_change=("rating_change", "std"),
            pct_improved=("rating_change", lambda s: (s.dropna() > 0).mean() * 100 if s.notna().any() else np.nan),
        )
        .reset_index()
    )
    return g


def month_club_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Injury counts: clubs x months (season order Aug..Jul) via pivot_table."""
    d = df.dropna(subset=["injury_month"])
    pv = pd.pivot_table(d, index="club", columns="injury_month", values="player", aggfunc="count", fill_value=0)
    pv = pv.reindex(columns=MONTH_ORDER, fill_value=0)
    return pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]


# --------------------------------------------------------------------------- #
# Match records (win / draw / loss) by phase
# --------------------------------------------------------------------------- #
def record_table(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """W/D/L, win-rate (Wilson CI) and points-per-game for before / during / after."""
    frames = []
    for phase in PHASES:
        cols = [f"{phase}_{i}_result" for i in (1, 2, 3)]
        keep = ([group] if group else []) + cols
        long = df[keep].melt(id_vars=[group] if group else None, value_name="result").dropna(subset=["result"])
        long["phase"] = phase
        frames.append(long[([group] if group else []) + ["phase", "result"]])
    allm = pd.concat(frames, ignore_index=True)
    allm["pts"] = allm["result"].map(POINTS)
    allm = allm.dropna(subset=["pts"])
    if group is None:
        allm["_all"] = "All"
        keys = ["_all", "phase"]
    else:
        keys = [group, "phase"]
    g = (
        allm.groupby(keys)
        .agg(
            matches=("pts", "size"),
            wins=("result", lambda s: (s == "win").sum()),
            draws=("result", lambda s: (s == "draw").sum()),
            losses=("result", lambda s: (s == "lose").sum()),
            ppg=("pts", "mean"),
        )
        .reset_index()
    )
    g["win_rate"] = g["wins"] / g["matches"]
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(g["wins"], g["matches"])]
    g["win_lo"] = [c[0] for c in ci]
    g["win_hi"] = [c[1] for c in ci]
    order = {p: i for i, p in enumerate(PHASES)}
    g = g.sort_values(keys, key=lambda s: s.map(order) if s.name == "phase" else s).reset_index(drop=True)
    if group is None:
        g = g.drop(columns="_all")
    return g


# --------------------------------------------------------------------------- #
# Inferential statistics ("Maths Lab")
# --------------------------------------------------------------------------- #
def paired_test(a: pd.Series, b: pd.Series, label: str) -> dict:
    """Paired t-test + Wilcoxon signed-rank + Cohen's d_z + 95% CI of the mean difference (a - b)."""
    x = pd.concat([a, b], axis=1).dropna()
    x.columns = ["a", "b"]
    n = len(x)
    out = {"test": label, "n": n}
    if n < 3:
        return {**out, "mean_a": np.nan, "mean_b": np.nan, "mean_diff": np.nan, "t": np.nan, "p_t": np.nan,
                "cohen_d": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "w": np.nan, "p_w": np.nan}
    diff = x["a"] - x["b"]
    sd = diff.std(ddof=1)
    se = sd / np.sqrt(n)
    t_stat, p_t = stats.ttest_rel(x["a"], x["b"])
    tcrit = stats.t.ppf(0.975, n - 1)
    try:
        w, p_w = stats.wilcoxon(diff) if (diff != 0).any() else (np.nan, np.nan)
    except ValueError:
        w, p_w = np.nan, np.nan
    return {
        **out,
        "mean_a": x["a"].mean(),
        "mean_b": x["b"].mean(),
        "mean_diff": diff.mean(),
        "t": t_stat,
        "p_t": p_t,
        "cohen_d": diff.mean() / sd if sd > 0 else np.nan,
        "ci_lo": diff.mean() - tcrit * se,
        "ci_hi": diff.mean() + tcrit * se,
        "w": w,
        "p_w": p_w,
    }


def all_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        paired_test(df["gd_before"], df["gd_during"], "Team GD: before vs during absence"),
        paired_test(df["ppg_before"], df["ppg_during"], "Team PPG: before vs during absence"),
        paired_test(df["gd_after"], df["gd_during"], "Team GD: after return vs during absence"),
        paired_test(df["rating_after"], df["rating_before"], "Player rating: after vs before"),
    ]
    return pd.DataFrame(rows)


def bootstrap_ci(x: pd.Series, n_boot: int = 5000, seed: int = 42, level: float = 0.95) -> dict:
    """Percentile bootstrap CI of the mean:  resample with replacement, take the quantiles."""
    x = x.dropna().to_numpy()
    if len(x) < 3:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": len(x), "boot": np.array([])}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return {"mean": x.mean(), "lo": np.quantile(boots, a), "hi": np.quantile(boots, 1 - a), "n": len(x), "boot": boots}


def correlation_pair(df: pd.DataFrame, x: str = "age", y: str = "tpdi") -> dict:
    d = df[[x, y]].dropna()
    if len(d) < 4 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return {"n": len(d), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(d[x], d[y])
    sr, sp = stats.spearmanr(d[x], d[y])
    return {"n": len(d), "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp}


def simple_regression(df: pd.DataFrame, x: str = "age", y: str = "tpdi") -> dict | None:
    """OLS  y = a + b x  with R^2, p-value and a 95% confidence band for the mean response."""
    d = df[[x, y]].dropna()
    if len(d) < 4 or d[x].nunique() < 2:
        return None
    res = stats.linregress(d[x], d[y])
    n = len(d)
    xbar = d[x].mean()
    sxx = ((d[x] - xbar) ** 2).sum()
    resid = d[y] - (res.intercept + res.slope * d[x])
    s = np.sqrt((resid**2).sum() / (n - 2))
    grid = np.linspace(d[x].min(), d[x].max(), 60)
    fit = res.intercept + res.slope * grid
    se_fit = s * np.sqrt(1 / n + (grid - xbar) ** 2 / sxx)
    tcrit = stats.t.ppf(0.975, n - 2)
    return {
        "n": n, "slope": res.slope, "intercept": res.intercept, "r": res.rvalue, "r2": res.rvalue**2,
        "p": res.pvalue, "stderr": res.stderr, "grid": grid, "fit": fit,
        "lo": fit - tcrit * se_fit, "hi": fit + tcrit * se_fit,
    }


def multiple_regression(df: pd.DataFrame, y: str = "tpdi", xs: tuple[str, ...] = ("age", "fifa_rating", "days_out")) -> dict | None:
    """OLS by the normal equations  beta = (X'X)^-1 X'y  with standard errors, t and p-values."""
    d = df[[y, *xs]].dropna()
    n, k = len(d), len(xs) + 1
    if n <= k + 2:
        return None
    X = np.column_stack([np.ones(n), d[list(xs)].to_numpy(dtype=float)])
    yy = d[y].to_numpy(dtype=float)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ X.T @ yy
    resid = yy - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof
    f_stat = (r2 / (k - 1)) / ((1 - r2) / dof) if r2 < 1 else np.nan
    table = pd.DataFrame({"term": ["Intercept", *xs], "coef": beta, "std_err": se, "t": t, "p_value": p})
    return {"table": table, "r2": r2, "adj_r2": adj, "n": n, "f": f_stat, "f_p": stats.f.sf(f_stat, k - 1, dof)}


def severity_kruskal(df: pd.DataFrame, value: str = "tpdi") -> dict:
    """Kruskal-Wallis H test across severity classes (non-parametric ANOVA) + epsilon^2 effect size."""
    groups = [g[value].dropna().to_numpy() for _, g in df.groupby("severity", observed=True) if g[value].notna().sum() >= 2]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "eps2": np.nan, "k": len(groups), "n": 0}
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {"H": h, "p": p, "eps2": max(0.0, (h - len(groups) + 1) / (n - len(groups))), "k": len(groups), "n": n}


def cluster_tests(df: pd.DataFrame) -> dict:
    """Chi-square tests for injury clustering by month and by club x month."""
    obs = month_club_matrix(df)
    out = {"obs": obs, "residuals": pd.DataFrame(), "month_chi2": np.nan, "month_p": np.nan,
           "chi2": np.nan, "p": np.nan, "dof": np.nan, "cramers_v": np.nan, "peak_month": None}
    if obs.empty or obs.to_numpy().sum() < 10:
        return out
    month_tot = obs.sum(axis=0)
    out["peak_month"] = month_tot.idxmax()
    active = month_tot[month_tot > 0]
    if len(active) > 1:
        c2, p = stats.chisquare(active)
        out["month_chi2"], out["month_p"] = c2, p
    table = obs.loc[:, obs.sum(axis=0) > 0]
    table = table.loc[table.sum(axis=1) > 0]
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p, dof, exp = stats.chi2_contingency(table.to_numpy())
        resid = (table.to_numpy() - exp) / np.sqrt(exp)
        out["residuals"] = pd.DataFrame(resid, index=table.index, columns=table.columns).reindex(columns=MONTH_ORDER)
        n = table.to_numpy().sum()
        out.update({"chi2": chi2, "p": p, "dof": dof,
                    "cramers_v": np.sqrt(chi2 / (n * (min(table.shape) - 1)))})
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["age", "fifa_rating", "days_out", "rating_before", "rating_after", "rating_change", "tpdi", "ppg_drop"]
    return df[cols].corr(method="spearman").round(2)


# --------------------------------------------------------------------------- #
# Auto-generated research answers (all numbers computed from the filtered data)
# --------------------------------------------------------------------------- #
def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def research_answers(df: pd.DataFrame) -> list[dict]:
    ans: list[dict] = []
    top = top_injuries(df, n=3, min_cases=3)
    if len(top):
        t1 = top.iloc[0]
        note = " The interval includes 0, so this is only a hint." if t1["mean_tpdi"] - t1["ci95"] < 0 else ""
        rest = ", ".join(f"{r['injury']} ({r['mean_tpdi']:.2f})" for _, r in top.iloc[1:].iterrows())
        ans.append({
            "q": "Q1. Which injuries led to the biggest team performance drop?",
            "a": f"**{t1['injury']}** had the biggest drop. Goal difference fell by about {t1['mean_tpdi']:.2f} goals per match "
                 f"on average (n = {int(t1['cases'])}, 95% CI \u00b1 {t1['ci95']:.2f}).{note} Next were {rest}.",
            "tab": "Injury Impact"})
    rec = record_table(df).set_index("phase")
    if {"before", "during"} <= set(rec.index):
        b, d = rec.loc["before"], rec.loc["during"]
        pt = paired_test(df["ppg_before"], df["ppg_during"], "ppg")
        ans.append({
            "q": "Q2. What was the team's win/loss record during a player's absence?",
            "a": f"While the player was out, the team won {int(d['wins'])}, drew {int(d['draws'])} and lost {int(d['losses'])}. "
                 f"The win rate was {d['win_rate']*100:.1f}% (it was {b['win_rate']*100:.1f}% before) and points per game were "
                 f"{d['ppg']:.2f} against {b['ppg']:.2f}. The paired t-test on points per game gives {_fmt_p(pt['p_t'])}, "
                 f"which is {sig_label(pt['p_t'])} at the 5% level.",
            "tab": "Team Record"})
    pr = paired_test(df["rating_after"], df["rating_before"], "rating")
    if pr["n"] >= 3:
        ans.append({
            "q": "Q3. How did individual players perform after recovery?",
            "a": f"Player ratings changed by {pr['mean_diff']:+.2f} on average after returning "
                 f"(95% CI {pr['ci_lo']:+.2f} to {pr['ci_hi']:+.2f}, n = {pr['n']}, Cohen's d = {pr['cohen_d']:.2f}, "
                 f"a {cohen_label(pr['cohen_d'])} effect, {_fmt_p(pr['p_t'])}).",
            "tab": "Comebacks"})
    ct = cluster_tests(df)
    if ct["peak_month"]:
        ans.append({
            "q": "Q4. Are there specific months or clubs with frequent injury clusters?",
            "a": f"Injuries peak in **{ct['peak_month']}**. A chi-square test against an even spread across months gives "
                 f"{_fmt_p(ct['month_p'])} ({sig_label(ct['month_p'])}), although June and July are off-season so fewer injuries "
                 f"there is expected. Testing club against month gives chi-square = {ct['chi2']:.1f}, {_fmt_p(ct['p'])}, "
                 f"Cramer's V = {ct['cramers_v']:.2f}.",
            "tab": "Clusters & Clubs"})
    cs = club_summary(df)
    if len(cs):
        c1 = cs.iloc[0]
        ans.append({
            "q": "Q5. Which clubs suffer most due to injuries?",
            "a": f"**{c1['club']}** has the highest Injury Burden Index at {c1['ibi_per_season']:.1f} star-days per season "
                 f"(z = {c1['ibi_z']:+.2f}), with {int(c1['injuries'])} injuries and {int(c1['total_days_out'])} days lost.",
            "tab": "Clusters & Clubs"})
    reg = simple_regression(df)
    if reg:
        ans.append({
            "q": "Q6. Does age explain how much a team suffers?",
            "a": f"The slope is {reg['slope']:+.3f} goals per year of age with R\u00b2 = {reg['r2']:.3f} ({_fmt_p(reg['p'])}), so age is "
                 f"{'a significant' if reg['p'] < ALPHA else 'not a significant'} predictor of the drop index.",
            "tab": "Age & Impact"})
    kr = severity_kruskal(df)
    if not np.isnan(kr["H"]):
        ans.append({
            "q": "Q7. Do longer injuries hurt the team more?",
            "a": f"Kruskal-Wallis test across the severity groups: H = {kr['H']:.2f}, {_fmt_p(kr['p'])} ({sig_label(kr['p'])}), "
                 f"epsilon\u00b2 = {kr['eps2']:.3f}.",
            "tab": "Maths & Stats"})
    return ans
