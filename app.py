"""
FootLens Analytics - The Mathematics of Player Injuries & Team Performance
Streamlit dashboard  |  Mathematics for AI-II  |  Summative Assessment (Scenario 1)

Run locally :  streamlit run app.py
The app reads  player_injuries_impact.csv  from the repository root (same folder as app.py).
If that file is missing or malformed the dashboard does NOT start and shows an error instead.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FootLens | Mathematics of Injuries",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

import analytics as an  # noqa: E402
import charts as ch  # noqa: E402
import data_processing as dp  # noqa: E402
import styles  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "player_injuries_impact.csv"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

styles.inject_css()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
_PLOTLY_PARAMS = inspect.signature(st.plotly_chart).parameters
_STRETCH = {"width": "stretch"} if "width" in _PLOTLY_PARAMS else {"use_container_width": True}


def show(fig) -> None:
    """Render a Plotly figure full-width (works on old and new Streamlit versions)."""
    st.plotly_chart(fig, config=PLOT_CONFIG, **_STRETCH)


def show_df(data: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(data, hide_index=True, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, hide_index=True, use_container_width=True, **kwargs)


def card():
    return st.container(border=True)


def fmt(x, d: int = 2, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def pfmt(p) -> str:
    if p is None or pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


@st.cache_data(show_spinner="Loading and cleaning dataset ...")
def load_data(path: str, mtime: float):
    return dp.load_dataset(path)


# --------------------------------------------------------------------------- #
# Hero + dataset gate (the app only works if the CSV is in the repo)
# --------------------------------------------------------------------------- #
styles.hero()

if not DATA_PATH.exists():
    styles.error_card(
        "Dataset not found",
        f"The dashboard needs the file '{DATA_PATH.relative_to(ROOT).as_posix()}' in the GitHub repository, "
        "but it could not be found, so the analysis has been stopped.",
        [
            "Open your GitHub repository (the main page, next to app.py).",
            "Click Add file > Upload files and upload player_injuries_impact.csv (exact file name).",
            "Reboot the app from the Streamlit Cloud menu (or refresh this page).",
        ],
    )
    st.stop()

try:
    df_all, cleaning_report = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except dp.DatasetError as exc:
    styles.error_card(
        "Dataset problem",
        str(exc),
        [
            "Check that the CSV is the original player_injuries_impact.csv file.",
            "Make sure the column headers were not edited.",
            "Re-upload the file to the data folder and reboot the app.",
        ],
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    styles.brand()
    st.caption("Filter the whole dashboard")
    clubs = sorted(df_all["club"].unique())
    seasons = sorted(df_all["season"].dropna().unique())
    groups = sorted(df_all["position_group"].unique())
    sel_clubs = st.multiselect("Club", clubs, default=clubs)
    sel_seasons = st.multiselect("Season", seasons, default=seasons)
    sel_groups = st.multiselect("Position group", groups, default=groups)
    a_min, a_max = int(df_all["age"].min()), int(df_all["age"].max())
    age_rng = st.slider("Age range", a_min, a_max, (a_min, a_max))
    min_cases = st.slider("Min. cases per injury type", 1, 15, 3, help="Groups with fewer cases are hidden from the injury rankings (small samples are noisy).")
    st.divider()

df = df_all[
    df_all["club"].isin(sel_clubs)
    & df_all["season"].isin(sel_seasons)
    & df_all["position_group"].isin(sel_groups)
    & df_all["age"].between(age_rng[0], age_rng[1])
].copy()

with st.sidebar:
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        df.drop(columns=["episode_label"]).to_csv(index=False).encode("utf-8"),
        file_name="footlens_filtered.csv",
        mime="text/csv",
    )
    st.caption(f"{len(df):,} of {len(df_all):,} injury records selected")
    st.caption("Built with Python · pandas · NumPy · SciPy · Plotly · Streamlit")

if df.empty:
    st.warning("No injuries match the current filters. Widen the filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs([
    "📊 Overview", "🩹 Injury Impact", "🏟️ Team Record", "🔁 Comebacks",
    "🗓️ Clusters & Clubs", "📈 Age & Impact", "🧮 Maths Lab", "📚 Research & Data",
])

# =========================================================================== #
# 1. OVERVIEW
# =========================================================================== #
with tabs[0]:
    k = an.headline_kpis(df)
    d_win = (k["winrate_during"] - k["winrate_before"]) * 100
    d_ppg = k["ppg_during"] - k["ppg_before"]
    cards = [
        styles.kpi("Injury episodes", f"{k['injuries']:,}", f"{k['players']} players · {k['clubs']} clubs", accent="#4F46E5", delay=0.0),
        styles.kpi("Avg. time out", f"{fmt(k['avg_days_out'], 0)} days", "mean per injury", accent="#0EA5E9", delay=0.05),
        styles.kpi("Team drop (TPDI)", fmt(k["avg_tpdi"], 2, True), "goals / match while absent", "down" if k["avg_tpdi"] > 0 else "up", "#F43F5E", 0.10),
        styles.kpi("Win rate during", f"{k['winrate_during']*100:.1f}%", f"{d_win:+.1f} pts vs before", "down" if d_win < 0 else "up", "#F59E0B", 0.15),
        styles.kpi("Points / game", fmt(k["ppg_during"], 2), f"{d_ppg:+.2f} vs before", "down" if d_ppg < 0 else "up", "#8B5CF6", 0.20),
        styles.kpi("Comeback rating", fmt(k["avg_rating_change"], 2, True), "after − before (0-10 scale)", "up" if k["avg_rating_change"] > 0 else "down", "#10B981", 0.25),
    ]
    for col, c in zip(st.columns(6), cards):
        col.markdown(c, unsafe_allow_html=True)

    pt = an.paired_test(df["gd_before"], df["gd_during"], "gd")
    bs = an.bootstrap_ci(df["tpdi"])
    if pt["n"] >= 3:
        verdict = "statistically detectable" if pt["p_t"] < an.ALPHA else "<b>not</b> statistically distinguishable from zero"
        styles.callout(
            f"<b>Headline finding.</b> Across {pt['n']} injuries with match data, the team's goal difference changed by "
            f"<b>{pt['mean_diff']:+.2f} goals/match</b> while the player was absent "
            f"(95% CI {pt['ci_lo']:+.2f} to {pt['ci_hi']:+.2f}; paired t-test p = {pfmt(pt['p_t'])}; Cohen's d = {pt['cohen_d']:.2f}). "
            f"The bootstrap interval for the mean drop is [{bs['lo']:+.2f}, {bs['hi']:+.2f}], so the average effect is {verdict}. "
            "The real value is in <i>which</i> injuries, clubs and players deviate from that average - explore the tabs above.",
            "good" if pt["p_t"] < an.ALPHA else "",
        )

    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("Results before, during and after", "Win rate with Wilson 95% intervals and points per game")
        show(ch.phase_record_chart(an.record_table(df)))
    with c2, card():
        styles.section("Which clubs lose most while a player is out?", "Mean Team Performance Drop Index by club (positive = worse)")
        show(ch.club_drop_chart(an.club_summary(df)))

# =========================================================================== #
# 2. INJURY IMPACT
# =========================================================================== #
with tabs[1]:
    styles.section("Which injuries hurt the team most?", "Ranked by mean Team Performance Drop Index, with 95% confidence intervals", "Q1")
    by_label = st.radio("Group injuries by", ["Injury family", "Exact injury label"], horizontal=True)
    by_col = "injury_type" if by_label == "Injury family" else "injury"
    tbl = an.top_injuries(df, by=by_col, n=10, min_cases=min_cases)
    if tbl.empty:
        st.info("No injury group has enough cases with the current filters. Lower 'Min. cases per injury type' in the sidebar.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1, card():
            styles.section("Top 10 injuries by team performance drop")
            show(ch.top_injuries_bar(tbl))
        with c2, card():
            styles.section("Top 5 highest-impact injuries")
            top5 = tbl.head(5)[["injury", "cases", "mean_tpdi", "ci95", "avg_days_out"]].rename(
                columns={"injury": "Injury", "cases": "Cases", "mean_tpdi": "Mean TPDI", "ci95": "± 95% CI", "avg_days_out": "Avg days out"})
            show_df(top5, column_config={
                "Mean TPDI": st.column_config.NumberColumn(format="%.2f"),
                "± 95% CI": st.column_config.NumberColumn(format="%.2f"),
                "Avg days out": st.column_config.NumberColumn(format="%.0f"),
            })
            styles.callout("Bars to the right of zero mean the team's goal difference <b>fell</b> during the absence. "
                           "If the error bar crosses zero the effect is not statistically reliable.", "warn")
    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("Does injury length change the damage?", "TPDI distribution by severity class (UEFA-style day bands)")
        show(ch.severity_box(df))
    with c2, card():
        styles.section("Ten worst single episodes")
        we = an.worst_episodes(df, 10)[["player", "club", "season", "injury", "days_out", "tpdi"]].rename(
            columns={"player": "Player", "club": "Club", "season": "Season", "injury": "Injury", "days_out": "Days", "tpdi": "TPDI"})
        show_df(we, column_config={"TPDI": st.column_config.NumberColumn(format="%.2f")})
        kr = an.severity_kruskal(df)
        st.caption(f"Kruskal-Wallis across severity classes: H = {fmt(kr['H'])}, p = {pfmt(kr['p'])} → {an.sig_label(kr['p'])}.")

# =========================================================================== #
# 3. TEAM RECORD
# =========================================================================== #
with tabs[2]:
    styles.section("What was the team's record while the player was absent?", "Three matches before, during and after each injury", "Q2")
    rec = an.record_table(df)
    c1, c2 = st.columns([2, 3])
    with c1, card():
        styles.section("Win / Draw / Loss by phase")
        show_df(rec[["phase", "matches", "wins", "draws", "losses", "win_rate", "ppg"]].rename(columns={
            "phase": "Phase", "matches": "Matches", "wins": "W", "draws": "D", "losses": "L", "win_rate": "Win rate", "ppg": "PPG"}),
            column_config={"Win rate": st.column_config.NumberColumn(format="%.3f"),
                           "PPG": st.column_config.NumberColumn(format="%.2f")})
        pp = an.paired_test(df["ppg_before"], df["ppg_during"], "ppg")
        st.caption(f"Paired t-test on points per game (before vs during): t = {fmt(pp['t'])}, p = {pfmt(pp['p_t'])}, d = {fmt(pp['cohen_d'])}.")
    with c2, card():
        styles.section("Points per game by club and phase")
        show(ch.club_record_chart(an.record_table(df, "club")))
    with card():
        styles.section("Record during absence, by club", "Pivot of W / D / L and points per game while the player was missing")
        rc = an.record_table(df, "club")
        rc = rc[rc["phase"] == "during"].drop(columns=["phase", "win_lo", "win_hi"]).sort_values("ppg")
        show_df(rc.rename(columns={"club": "Club", "matches": "Matches", "wins": "W", "draws": "D", "losses": "L", "ppg": "PPG", "win_rate": "Win rate"}),
                column_config={"Win rate": st.column_config.NumberColumn(format="%.3f"), "PPG": st.column_config.NumberColumn(format="%.2f")})

# =========================================================================== #
# 4. COMEBACKS
# =========================================================================== #
with tabs[3]:
    styles.section("How did players perform after recovery?", "Player timelines, comeback leaderboard and recovery statistics", "Q3")
    with card():
        styles.section("Player timeline: before → out → after", "Rating line (left axis) and the team's goal difference in each match (bars, right axis)")
        counts = df.groupby("player").size().sort_values(ascending=False)
        c1, c2 = st.columns(2)
        player = c1.selectbox("Player", counts.index.tolist(), index=0)
        ep = df[df["player"] == player]
        ep_idx = c2.selectbox("Injury episode", ep.index.tolist(), format_func=lambda i: ep.loc[i, "episode_label"])
        row = df.loc[ep_idx]
        m = st.columns(4)
        m[0].markdown(styles.kpi("Rating before", fmt(row["rating_before"]), "mean of 3 matches", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Rating after", fmt(row["rating_after"]), "mean of 3 matches", accent="#10B981"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("Change", fmt(row["rating_change"], 2, True), "after − before", "up" if (row["rating_change"] or 0) > 0 else "down", "#0EA5E9"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Days out", fmt(row["days_out"], 0), f"TPDI {fmt(row['tpdi'], 2, True)}", accent="#F43F5E"), unsafe_allow_html=True)
        show(ch.player_timeline(row))

    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("Comeback leaderboard", "Injury episodes ranked by rating improvement (z-score = standard deviations above the average change)")
        lc1, lc2 = st.columns(2)
        n_rows = lc1.slider("Rows", 5, 30, 15)
        min_days = lc2.slider("Minimum days out", 0, 90, 0)
        lb = an.comeback_leaderboard(df, n_rows, min_days).reset_index()
        top_val = float(lb["rating_change"].max()) if len(lb) else 1.0
        show_df(lb.rename(columns={"player": "Player", "club": "Club", "season": "Season", "injury_type": "Injury", "days_out": "Days out",
                                   "rating_before": "Before", "rating_after": "After", "rating_change": "Δ Rating", "comeback_z": "z"}),
                column_config={
                    "Δ Rating": st.column_config.ProgressColumn(min_value=0.0, max_value=max(top_val, 0.1), format="%+.2f"),
                    "Before": st.column_config.NumberColumn(format="%.2f"), "After": st.column_config.NumberColumn(format="%.2f"),
                    "z": st.column_config.NumberColumn(format="%.2f"), "Days out": st.column_config.NumberColumn(format="%.0f"),
                })
    with c2, card():
        styles.section("Rating change distribution")
        show(ch.rating_change_hist(df))
        up, down = an.improvers_decliners(df, 5)
        st.caption("Biggest improvers")
        show_df(up[["player", "club", "rating_change"]].rename(columns={"player": "Player", "club": "Club", "rating_change": "Δ"}),
                column_config={"Δ": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Biggest decliners")
        show_df(down[["player", "club", "rating_change"]].rename(columns={"player": "Player", "club": "Club", "rating_change": "Δ"}),
                column_config={"Δ": st.column_config.NumberColumn(format="%+.2f")})

    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("Pre- vs post-injury rating (pivot table)")
        idx_label = st.radio("Pivot by", ["club", "position_group", "age_band"], horizontal=True)
        pv = an.pre_post_pivot(df, idx_label).reset_index().rename(columns={
            idx_label: idx_label.replace("_", " ").title(), "rating_before": "Before", "rating_after": "After", "rating_change": "Δ"})
        show_df(pv, column_config={"Before": st.column_config.NumberColumn(format="%.2f"), "After": st.column_config.NumberColumn(format="%.2f"),
                                   "Δ": st.column_config.NumberColumn(format="%+.3f")})
    with c2, card():
        styles.section("Recovery trend by severity")
        rt = an.recovery_trend(df, "severity").rename(columns={
            "severity": "Severity", "episodes": "Episodes", "avg_days_out": "Avg days", "median_days_out": "Median days",
            "mean_rating_change": "Mean Δ rating", "sd_rating_change": "SD Δ", "pct_improved": "% improved"})
        show_df(rt, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Avg days", "Median days", "Mean Δ rating", "SD Δ", "% improved"]})

# =========================================================================== #
# 5. CLUSTERS & CLUBS
# =========================================================================== #
with tabs[4]:
    styles.section("When and where do injuries cluster?", "Heatmap of injuries by club and calendar month (season order)", "Q4 · Q5")
    ct = an.cluster_tests(df)
    with card():
        mode = st.radio("Heatmap shows", ["Injury count", "Standardised residual (χ²)"], horizontal=True)
        if ct["obs"].empty:
            st.info("Not enough dated injuries for a heatmap with these filters.")
        else:
            show(ch.month_club_heatmap(ct["obs"], ct["residuals"], mode))
            st.caption("Standardised residual = (observed − expected) / √expected, where expected = row total × column total / N. "
                       "Values beyond ±2 flag cells with more (red) or fewer (blue) injuries than club and month totals alone would predict.")
        m = st.columns(4)
        m[0].markdown(styles.kpi("Peak month", str(ct["peak_month"] or "n/a"), "most injuries overall", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Uniform-month χ²", fmt(ct["month_chi2"], 1), f"p = {pfmt(ct['month_p'])}", accent="#0EA5E9"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("Club × month χ²", fmt(ct["chi2"], 1), f"p = {pfmt(ct['p'])}", accent="#F59E0B"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Cramér's V", fmt(ct["cramers_v"], 2), "association strength (0-1)", accent="#10B981"), unsafe_allow_html=True)
    cs = an.club_summary(df)
    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("Which clubs suffer most?", "Injury Burden Index: days lost weighted by player quality (FIFA rating), per season")
        show(ch.club_burden_chart(cs))
        st.latex(r"\mathrm{IBI}_{club}=\frac{1}{S}\sum_{i\in club}\text{days\_out}_i\cdot\frac{\text{FIFA}_i}{100}")
    with c2, card():
        styles.section("Club table")
        show_df(cs[["club", "injuries", "players", "total_days_out", "mean_tpdi", "ibi_per_season", "ibi_z"]].rename(columns={
            "club": "Club", "injuries": "Injuries", "players": "Players", "total_days_out": "Days lost", "mean_tpdi": "Mean TPDI",
            "ibi_per_season": "IBI/season", "ibi_z": "IBI z"}),
            column_config={"Mean TPDI": st.column_config.NumberColumn(format="%.2f"), "IBI/season": st.column_config.NumberColumn(format="%.0f"),
                           "IBI z": st.column_config.NumberColumn(format="%+.2f"), "Days lost": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Most frequently injured players")
        show_df(an.most_injured(df, 8).rename(columns={"player": "Player", "club": "Club", "injuries": "Injuries", "total_days_out": "Days", "mean_tpdi": "Mean TPDI"}),
                column_config={"Mean TPDI": st.column_config.NumberColumn(format="%.2f"), "Days": st.column_config.NumberColumn(format="%.0f")})

# =========================================================================== #
# 6. AGE & IMPACT
# =========================================================================== #
with tabs[5]:
    styles.section("Does a player's age change the team's drop?", "Scatter of age vs TPDI (bubble size = days out) with an OLS regression line and 95% confidence band", "Q6")
    reg = an.simple_regression(df)
    cp = an.correlation_pair(df)
    with card():
        show(ch.age_scatter(df, reg))
        m = st.columns(4)
        m[0].markdown(styles.kpi("Pearson r", fmt(cp["pearson_r"], 3, True), f"p = {pfmt(cp['pearson_p'])}", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Spearman ρ", fmt(cp["spearman_rho"], 3, True), f"p = {pfmt(cp['spearman_p'])}", accent="#0EA5E9"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("R²", fmt(reg["r2"], 4) if reg else "n/a", "variance explained by age", accent="#F59E0B"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Slope", fmt(reg["slope"], 3, True) if reg else "n/a", "goals per extra year of age", accent="#10B981"), unsafe_allow_html=True)
        if reg:
            st.latex(rf"\widehat{{\mathrm{{TPDI}}}} = {reg['intercept']:.3f} + {reg['slope']:.4f}\cdot \text{{age}}\qquad R^2={reg['r2']:.4f},\; p={pfmt(reg['p'])}")
            styles.callout(
                f"Interpretation: each extra year of age is associated with a {reg['slope']:+.3f} goal/match change in the team's drop, "
                f"and age explains {reg['r2']*100:.2f}% of the variation. The relationship is "
                f"<b>{'statistically significant' if reg['p'] < an.ALPHA else 'not statistically significant'}</b> at α = 0.05.",
                "good" if reg["p"] < an.ALPHA else "warn")
    with card():
        styles.section("Recovery by age band")
        ra = an.recovery_trend(df, "age_band").rename(columns={
            "age_band": "Age band", "episodes": "Episodes", "avg_days_out": "Avg days", "median_days_out": "Median days",
            "mean_rating_change": "Mean Δ rating", "sd_rating_change": "SD Δ", "pct_improved": "% improved"})
        show_df(ra, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Avg days", "Median days", "Mean Δ rating", "SD Δ", "% improved"]})

# =========================================================================== #
# 7. MATHS LAB
# =========================================================================== #
with tabs[6]:
    styles.section("Maths Lab: is the effect real, or just noise?", "Every formula below is implemented in analytics.py with NumPy / SciPy", "Statistics")
    with card():
        styles.section("Definitions")
        f1, f2 = st.columns(2)
        with f1:
            st.latex(r"\mathrm{TPDI}_i=\overline{GD}_{\text{before},i}-\overline{GD}_{\text{during},i}")
            st.latex(r"z_i=\dfrac{\mathrm{TPDI}_i-\mu}{\sigma}")
            st.latex(r"\text{PPG}=\dfrac{3W+1D+0L}{W+D+L}")
        with f2:
            st.latex(r"t=\dfrac{\bar d}{s_d/\sqrt n},\qquad d_z=\dfrac{\bar d}{s_d}")
            st.latex(r"\bar d\pm t_{0.975,\,n-1}\dfrac{s_d}{\sqrt n}")
            st.latex(r"\tilde p\;\pm\; z\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^2}{4n^2}}\quad(\text{Wilson})")
    with card():
        styles.section("1 · Paired hypothesis tests", "H₀: the mean difference is zero. Positive Δ means the first quantity is larger.")
        pt_df = an.all_paired_tests(df)
        disp = pd.DataFrame({
            "Comparison": pt_df["test"], "n": pt_df["n"],
            "Mean A": pt_df["mean_a"].round(3), "Mean B": pt_df["mean_b"].round(3), "Δ (A−B)": pt_df["mean_diff"].round(3),
            "95% CI of Δ": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(pt_df["ci_lo"], pt_df["ci_hi"])],
            "t": pt_df["t"].round(2), "p (t-test)": [pfmt(p) for p in pt_df["p_t"]],
            "Cohen's d": pt_df["cohen_d"].round(3), "Effect": [an.cohen_label(d) for d in pt_df["cohen_d"]],
            "p (Wilcoxon)": [pfmt(p) for p in pt_df["p_w"]], "Verdict (α = 0.05)": [an.sig_label(p) for p in pt_df["p_t"]],
        })
        show_df(disp)
        styles.callout("<b>Why two tests?</b> The t-test assumes roughly normal differences; the Wilcoxon signed-rank test does not. "
                       "When both agree, the conclusion is robust. Cohen's d tells you how <i>big</i> the effect is, not just whether it exists.")
    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("2 · Bootstrap confidence interval", "5,000 resamples with replacement (seed 42)")
        bs = an.bootstrap_ci(df["tpdi"])
        if len(bs["boot"]):
            show(ch.bootstrap_hist(bs["boot"], bs["lo"], bs["hi"], bs["mean"]))
            st.caption(f"Mean TPDI = {bs['mean']:+.3f}, 95% bootstrap CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}], n = {bs['n']}. "
                       + ("The interval excludes 0." if (bs["lo"] > 0 or bs["hi"] < 0) else "The interval contains 0, so no reliable average effect."))
        else:
            st.info("Not enough data for a bootstrap with the current filters.")
    with c2, card():
        styles.section("3 · Multiple regression (normal equations)")
        st.latex(r"\hat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top y")
        mr = an.multiple_regression(df)
        if mr:
            tb = mr["table"].copy()
            tb["Significant"] = np.where(tb["p_value"] < an.ALPHA, "yes", "no")
            show_df(tb.rename(columns={"term": "Term", "coef": "β", "std_err": "SE", "t": "t", "p_value": "p"}),
                    column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["β", "SE", "t", "p"]})
            st.caption(f"TPDI ~ age + FIFA rating + days out · n = {mr['n']} · R² = {mr['r2']:.4f} · adj. R² = {mr['adj_r2']:.4f} · F-test p = {pfmt(mr['f_p'])}")
        else:
            st.info("Not enough data for the regression with the current filters.")
    with card():
        styles.section("4 · Spearman correlation matrix", "Rank correlation between the engineered metrics (robust to outliers)")
        show(ch.correlation_heatmap(an.correlation_matrix(df)))
    styles.callout(
        "<b>Caveat - regression to the mean.</b> Injured players are often in the squad because they were playing well, so 'before' "
        "matches can look unusually good and 'after' matches drift back to normal. A drop in goal difference is therefore not purely "
        "caused by the absence. This is why the dashboard reports confidence intervals and effect sizes instead of raw averages alone.",
        "warn")

# =========================================================================== #
# 8. RESEARCH & DATA
# =========================================================================== #
with tabs[7]:
    styles.section("Research questions, answered from the (filtered) data", "Answers update live when you change the sidebar filters", "Step 1")
    for a in an.research_answers(df):
        styles.qa_card(a["q"], a["a"], a["tab"])

    with card():
        styles.section("Background research: what sports analysts look for")
        st.markdown(
            "- **Injuries and results.** Club-level studies of professional football (e.g. the UEFA Elite Club Injury Study) link lower injury burden with better league "
            "points and progress in cup competitions, which motivates measuring team performance *during* an absence.\n"
            "- **Severity classification.** Injuries are usually graded by days absent (minimal, mild, moderate, severe); this dashboard uses "
            "≤7, 8-28, 29-90 and >90 days.\n"
            "- **Player impact.** Analysts compare a player's rating before vs after return and weight absences by player quality; hence "
            "the FIFA-rating-weighted **Injury Burden Index**.\n"
            "- **Uncertainty matters.** With only three matches per phase, averages are noisy; confidence intervals, effect sizes and "
            "non-parametric tests prevent over-interpreting small differences.\n"
        )
        st.caption("Sources: Hägglund et al. (2013) Br J Sports Med 47:738-742; Ekstrand et al. (2011) Br J Sports Med 45:553-558; "
                   "Cohen (1988) Statistical Power Analysis for the Behavioral Sciences; reference links supplied in the assessment brief.")

    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("Data cleaning report", "What the pipeline fixed automatically (full dataset)")
        r = cleaning_report
        rep_rows = [
            ("Rows loaded → kept", f"{r['rows_raw']} → {r['rows_clean']}"),
            ("'N.A.' placeholders converted to NaN", r["na_tokens_converted"]),
            ("Duplicate rows removed", r["duplicates_removed"]),
            ("Injury labels standardised", f"{r['injury_labels_before']} → {r['injury_labels_after']} (in {r['injury_families']} families)"),
            ("Return-date year typos auto-corrected", r["return_years_auto_corrected"]),
            ("Impossible return dates set to missing", r["return_dates_invalid"]),
            ("Ongoing injuries ('Present')", r["ongoing_injuries_present"]),
            ("Episodes without rating data", r["rows_without_ratings"]),
        ]
        show_df(pd.DataFrame(rep_rows, columns=["Step", "Result"]).astype(str))
    with c2, card():
        styles.section("Engineered features")
        show_df(pd.DataFrame([
            ("rating_before / after / change", "Mean player rating over 3 matches; change = after − before"),
            ("gd_before / during / after", "Mean team goal difference over the 3 matches of each phase"),
            ("ppg_*, winrate_*", "Points per game (3-1-0) and share of wins per phase"),
            ("tpdi", "Team Performance Drop Index = gd_before − gd_during"),
            ("tpdi_z", "Standardised TPDI (z-score)"),
            ("days_out, severity", "Return − injury date; UEFA-style class"),
            ("injury_type, age_band, position_group", "Grouped categories for analysis"),
        ], columns=["Column", "Definition"]))

    with card():
        styles.section("Cleaned dataset preview")
        preview_cols = ["player", "club", "season", "position", "age", "injury", "injury_date", "days_out", "rating_before", "rating_after",
                        "rating_change", "gd_before", "gd_during", "tpdi", "severity"]
        show_df(df[preview_cols].head(200))
        st.caption("Showing the first 200 rows of the filtered data. Use the sidebar button to download all of it.")
