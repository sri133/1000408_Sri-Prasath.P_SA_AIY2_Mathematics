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
    page_title="FootLens | Injuries and Team Performance",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

import analytics as an  # noqa: E402
import charts as ch  # noqa: E402
import data_processing as dp  # noqa: E402
import report  # noqa: E402
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


@st.cache_data(show_spinner="Loading data...")
def load_data(path: str, mtime: float):
    return dp.load_dataset(path)


# --------------------------------------------------------------------------- #
# Hero + dataset gate (the app only works if the CSV is in the repo)
# --------------------------------------------------------------------------- #
styles.hero_3d()

if not DATA_PATH.exists():
    styles.error_card(
        "Dataset not found",
        f"The file '{DATA_PATH.relative_to(ROOT).as_posix()}' was not found in the repository, so the dashboard can't run.",
        [
            "Open your GitHub repository, on the main page next to app.py.",
            "Click Add file > Upload files and upload player_injuries_impact.csv (exact file name).",
            "Reboot the app from the Streamlit Cloud menu, or refresh this page.",
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
            "Check that this is the original player_injuries_impact.csv file.",
            "Make sure the column headers have not been edited.",
            "Re-upload the file to the data folder and reboot the app.",
        ],
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    styles.brand()
    st.caption("Filters")
    clubs = sorted(df_all["club"].unique())
    seasons = sorted(df_all["season"].dropna().unique())
    groups = sorted(df_all["position_group"].unique())
    sel_clubs = st.multiselect("Club", clubs, default=clubs)
    sel_seasons = st.multiselect("Season", seasons, default=seasons)
    sel_groups = st.multiselect("Position group", groups, default=groups)
    a_min, a_max = int(df_all["age"].min()), int(df_all["age"].max())
    age_rng = st.slider("Age range", a_min, a_max, (a_min, a_max))
    min_cases = st.slider("Min. cases per injury type", 1, 15, 3, help="Injury groups with fewer cases than this are left out of the rankings, because small samples are unreliable.")
    st.divider()

df = df_all[
    df_all["club"].isin(sel_clubs)
    & df_all["season"].isin(sel_seasons)
    & df_all["position_group"].isin(sel_groups)
    & df_all["age"].between(age_rng[0], age_rng[1])
].copy()

with st.sidebar:
    st.download_button(
        "Download filtered data (CSV)",
        df.drop(columns=["episode_label"]).to_csv(index=False).encode("utf-8"),
        file_name="footlens_filtered.csv",
        mime="text/csv",
    )
    st.caption(f"{len(df):,} of {len(df_all):,} injury records selected")
    st.caption("Made with Python, pandas, SciPy, Plotly and Streamlit")

if df.empty:
    st.warning("No records match these filters. Try selecting more clubs or seasons.")
    st.stop()

# Re-injury table: built on the full data (so a player's next injury is never lost), then filtered.
re_all, re_info = an.build_reinjury(df_all)
re_tbl = re_all[re_all.index.isin(df.index)]
re_sum = an.reinjury_summary(re_tbl, df)
STAR = 82   # FIFA rating used for "star player" in the simulator and the brief

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
(t_over, t_inj, t_team, t_come, t_club, t_star, t_what, t_re, t_math, t_brief, t_data) = st.tabs([
    "Overview", "Injury Impact", "Team Record", "Comebacks", "Clubs", "Stars & Age",
    "What-if Tools", "Re-injury Risk", "Maths & Stats", "Manager Brief", "Data & Method",
])

# =========================================================================== #
# 1. OVERVIEW
# =========================================================================== #
with t_over:
    k = an.headline_kpis(df)
    d_win = (k["winrate_during"] - k["winrate_before"]) * 100
    d_ppg = k["ppg_during"] - k["ppg_before"]
    cards = [
        styles.kpi("Injuries", f"{k['injuries']:,}", f"{k['players']} players, {k['clubs']} clubs", accent="#4F46E5", delay=0.0),
        styles.kpi("Avg. time out", f"{fmt(k['avg_days_out'], 0)} days", "average per injury", accent="#0EA5E9", delay=0.05),
        styles.kpi("Team drop (TPDI)", fmt(k["avg_tpdi"], 2, True), "goals per match while out", "down" if k["avg_tpdi"] > 0 else "up", "#F43F5E", 0.10),
        styles.kpi("Win rate while out", f"{k['winrate_during']*100:.1f}%", f"{d_win:+.1f} pts vs before", "down" if d_win < 0 else "up", "#F59E0B", 0.15),
        styles.kpi("Points per game", fmt(k["ppg_during"], 2), f"{d_ppg:+.2f} vs before", "down" if d_ppg < 0 else "up", "#8B5CF6", 0.20),
        styles.kpi("Comeback rating", fmt(k["avg_rating_change"], 2, True), "rating after minus before", "up" if k["avg_rating_change"] > 0 else "down", "#10B981", 0.25),
    ]
    for col, c in zip(st.columns(6), cards):
        col.markdown(c, unsafe_allow_html=True)

    pt = an.paired_test(df["gd_before"], df["gd_during"], "gd")
    bs = an.bootstrap_ci(df["tpdi"])
    if pt["n"] >= 3:
        verdict = "statistically significant" if pt["p_t"] < an.ALPHA else "not statistically significant"
        styles.callout(
            f"Across {pt['n']} injuries with match data, the team's goal difference changed by "
            f"<b>{pt['mean_diff']:+.2f} goals per match</b> while the player was out "
            f"(95% CI {pt['ci_lo']:+.2f} to {pt['ci_hi']:+.2f}; paired t-test p = {pfmt(pt['p_t'])}; Cohen's d = {pt['cohen_d']:.2f}). "
            f"The bootstrap interval for the mean drop is [{bs['lo']:+.2f}, {bs['hi']:+.2f}], so the average effect is {verdict}. "
            "What matters more is which injuries, clubs and players differ from that average, and the other tabs cover that.",
            "good" if pt["p_t"] < an.ALPHA else "",
        )

    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("Results before, during and after", "Win rate (with Wilson 95% intervals) and points per game")
        show(ch.phase_record_chart(an.record_table(df)))
    with c2, card():
        styles.section("Team performance drop by club", "Average TPDI per club. A higher value means a bigger drop.")
        show(ch.club_drop_chart(an.club_summary(df)))

# =========================================================================== #
# 2. INJURY IMPACT
# =========================================================================== #
with t_inj:
    styles.section("Injuries with the biggest team impact", "Ranked by average TPDI, with 95% confidence intervals", "Q1")
    by_label = st.radio("Group injuries by", ["Injury family", "Exact injury label"], horizontal=True)
    by_col = "injury_type" if by_label == "Injury family" else "injury"
    tbl = an.top_injuries(df, by=by_col, n=10, min_cases=min_cases)
    if tbl.empty:
        st.info("No injury group has enough cases with the current filters. Lower 'Min. cases per injury type' in the sidebar.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1, card():
            styles.section("Top 10 injuries by TPDI")
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
            styles.callout("A bar to the right of zero means the team's goal difference got worse while the player was out. "
                           "If the error bar crosses zero, the difference could easily be down to chance.", "warn")
    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("TPDI by injury severity", "Injuries grouped by days out: up to 7, 8-28, 29-90 and over 90")
        show(ch.severity_box(df))
    with c2, card():
        styles.section("10 largest single drops")
        we = an.worst_episodes(df, 10)[["player", "club", "season", "injury", "days_out", "tpdi"]].rename(
            columns={"player": "Player", "club": "Club", "season": "Season", "injury": "Injury", "days_out": "Days", "tpdi": "TPDI"})
        show_df(we, column_config={"TPDI": st.column_config.NumberColumn(format="%.2f")})
        kr = an.severity_kruskal(df)
        st.caption(f"Kruskal-Wallis test across severity groups: H = {fmt(kr['H'])}, p = {pfmt(kr['p'])} ({an.sig_label(kr['p'])}).")

# =========================================================================== #
# 3. TEAM RECORD
# =========================================================================== #
with t_team:
    styles.section("Team record while the player was out", "Based on three matches before, during and after each injury", "Q2")
    rec = an.record_table(df)
    c1, c2 = st.columns([2, 3])
    with c1, card():
        styles.section("Wins, draws and losses by phase")
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
        styles.section("Record by club while the player was out", "Wins, draws, losses and points per game")
        rc = an.record_table(df, "club")
        rc = rc[rc["phase"] == "during"].drop(columns=["phase", "win_lo", "win_hi"]).sort_values("ppg")
        show_df(rc.rename(columns={"club": "Club", "matches": "Matches", "wins": "W", "draws": "D", "losses": "L", "ppg": "PPG", "win_rate": "Win rate"}),
                column_config={"Win rate": st.column_config.NumberColumn(format="%.3f"), "PPG": st.column_config.NumberColumn(format="%.2f")})

# =========================================================================== #
# 4. COMEBACKS
# =========================================================================== #
with t_come:
    styles.section("Performance after returning", "Player timelines, a comeback leaderboard and recovery numbers", "Q3")
    with card():
        styles.section("Player timeline", "Player rating (line, left axis) and team goal difference per match (bars, right axis)")
        counts = df.groupby("player").size().sort_values(ascending=False)
        c1, c2 = st.columns(2)
        player = c1.selectbox("Player", counts.index.tolist(), index=0)
        ep = df[df["player"] == player]
        ep_idx = c2.selectbox("Injury episode", ep.index.tolist(), format_func=lambda i: ep.loc[i, "episode_label"])
        row = df.loc[ep_idx]
        m = st.columns(4)
        m[0].markdown(styles.kpi("Rating before", fmt(row["rating_before"]), "average of 3 matches", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Rating after", fmt(row["rating_after"]), "average of 3 matches", accent="#10B981"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("Change", fmt(row["rating_change"], 2, True), "after minus before", "up" if (row["rating_change"] or 0) > 0 else "down", "#0EA5E9"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Days out", fmt(row["days_out"], 0), f"TPDI {fmt(row['tpdi'], 2, True)}", accent="#F43F5E"), unsafe_allow_html=True)
        show(ch.player_timeline(row))

    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("Comeback leaderboard", "Injuries ranked by rating improvement. The z column shows how far each is above the average change, in standard deviations.")
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
        styles.section("Distribution of rating change")
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
        styles.section("Rating before and after injury (pivot table)")
        idx_label = st.radio("Pivot by", ["club", "position_group", "age_band"], horizontal=True)
        pv = an.pre_post_pivot(df, idx_label).reset_index().rename(columns={
            idx_label: idx_label.replace("_", " ").title(), "rating_before": "Before", "rating_after": "After", "rating_change": "Δ"})
        show_df(pv, column_config={"Before": st.column_config.NumberColumn(format="%.2f"), "After": st.column_config.NumberColumn(format="%.2f"),
                                   "Δ": st.column_config.NumberColumn(format="%+.3f")})
    with c2, card():
        styles.section("Recovery by severity")
        rt = an.recovery_trend(df, "severity").rename(columns={
            "severity": "Severity", "episodes": "Episodes", "avg_days_out": "Avg days", "median_days_out": "Median days",
            "mean_rating_change": "Mean Δ rating", "sd_rating_change": "SD Δ", "pct_improved": "% improved"})
        show_df(rt, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Avg days", "Median days", "Mean Δ rating", "SD Δ", "% improved"]})

# =========================================================================== #
# 5. CLUSTERS & CLUBS
# =========================================================================== #
with t_club:
    styles.section("Injuries by club and month", "Months are in season order, August to July", "Q4, Q5")
    ct = an.cluster_tests(df)
    with card():
        mode = st.radio("Heatmap shows", ["Injury count", "Standardised residual (χ²)"], horizontal=True)
        if ct["obs"].empty:
            st.info("Not enough dated injuries for a heatmap with these filters.")
        else:
            show(ch.month_club_heatmap(ct["obs"], ct["residuals"], mode))
            st.caption("Standardised residual = (observed - expected) / sqrt(expected), where expected = row total x column total / N. "
                       "A value beyond +2 or -2 means that club had noticeably more (red) or fewer (blue) injuries in that month than expected.")
        m = st.columns(4)
        m[0].markdown(styles.kpi("Peak month", str(ct["peak_month"] or "n/a"), "most injuries overall", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Uniform-month χ²", fmt(ct["month_chi2"], 1), f"p = {pfmt(ct['month_p'])}", accent="#0EA5E9"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("Club × month χ²", fmt(ct["chi2"], 1), f"p = {pfmt(ct['p'])}", accent="#F59E0B"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Cramér's V", fmt(ct["cramers_v"], 2), "strength of association, 0 to 1", accent="#10B981"), unsafe_allow_html=True)
    cs = an.club_summary(df)
    c1, c2 = st.columns([3, 2])
    with c1, card():
        styles.section("Injury burden by club", "Injury Burden Index (IBI): days lost, weighted by FIFA rating, per season")
        show(ch.club_burden_chart(cs))
        st.latex(r"\mathrm{IBI}_{club}=\frac{1}{S}\sum_{i\in club}\text{days\_out}_i\cdot\frac{\text{FIFA}_i}{100}")
    with c2, card():
        styles.section("Club table")
        show_df(cs[["club", "injuries", "players", "total_days_out", "mean_tpdi", "ibi_per_season", "ibi_z"]].rename(columns={
            "club": "Club", "injuries": "Injuries", "players": "Players", "total_days_out": "Days lost", "mean_tpdi": "Mean TPDI",
            "ibi_per_season": "IBI/season", "ibi_z": "IBI z"}),
            column_config={"Mean TPDI": st.column_config.NumberColumn(format="%.2f"), "IBI/season": st.column_config.NumberColumn(format="%.0f"),
                           "IBI z": st.column_config.NumberColumn(format="%+.2f"), "Days lost": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Most injured players")
        show_df(an.most_injured(df, 8).rename(columns={"player": "Player", "club": "Club", "injuries": "Injuries", "total_days_out": "Days", "mean_tpdi": "Mean TPDI"}),
                column_config={"Mean TPDI": st.column_config.NumberColumn(format="%.2f"), "Days": st.column_config.NumberColumn(format="%.0f")})

    club_opts = sorted(df["club"].unique())
    with card():
        styles.section("Club vs club", "Put two clubs side by side: results, injury burden and the kind of injuries they suffer")
        if len(club_opts) < 2:
            st.info("Select at least two clubs in the sidebar to use the comparison.")
        else:
            first_club = cs.iloc[0]["club"]
            second_club = cs.iloc[1]["club"] if len(cs) > 1 else club_opts[1]
            cc1, cc2 = st.columns(2)
            club_a = cc1.selectbox("Club A", club_opts, index=club_opts.index(first_club), key="cmp_a")
            club_b = cc2.selectbox("Club B", club_opts, index=club_opts.index(second_club), key="cmp_b")
            if club_a == club_b:
                st.warning("Pick two different clubs.")
            else:
                cmp = an.compare_clubs(df, club_a, club_b)
                d1, d2 = st.columns([2, 3])
                with d1:
                    show_df(cmp["table"])
                    tt = cmp["test"]
                    st.caption(f"Team drop (TPDI), {club_a} vs {club_b}: difference {fmt(tt['diff'], 2, True)}, Welch p = {pfmt(tt['p_welch'])}, "
                               f"Cohen's d = {fmt(tt['cohen_d'])} ({an.sig_label(tt['p_welch'])}).")
                with d2:
                    show(ch.club_record_chart(cmp["record"]))
                show(ch.family_compare(cmp["families"]))
    with card():
        styles.section("How injury burden built up", "Press play to watch cumulative star-days lost, season by season")
        show(ch.burden_race(an.burden_by_season(df)))

# =========================================================================== #
# 6. AGE & IMPACT
# =========================================================================== #
with t_star:
    styles.section("Do star players matter more?", "Injuries split by FIFA rating: does losing a higher-rated player cost the team more?", "Stars")
    thr = st.slider("Star player = FIFA rating of at least", 75, 88, STAR, key="star_thr")
    df_t = df.assign(tier=np.where(df["fifa_rating"] >= thr, f"Star ({thr}+)", "Other players"))
    n_star = int((df["fifa_rating"] >= thr).sum())
    n_other = len(df) - n_star
    if n_star < 5 or n_other < 5:
        st.info("One of the two groups has fewer than 5 injuries with the current filters. Change the rating cut-off or widen the filters.")
    else:
        sv = an.star_vs_squad(df, thr)
        r0 = sv.iloc[0]
        m = st.columns(4)
        m[0].markdown(styles.kpi("Star injuries", str(n_star), f"rated {thr}+", accent="#F59E0B"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Other injuries", str(n_other), f"rated below {thr}", accent="#4F46E5"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("TPDI difference", fmt(r0["diff"], 2, True), "stars minus others (goals/match)", accent="#F43F5E"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Welch t-test", f"p = {pfmt(r0['p_welch'])}", f"Cohen's d = {fmt(r0['cohen_d'])}", accent="#10B981"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1, card():
            styles.section("Team drop by player type", "Box shows the middle 50%, the dashed line is the mean")
            show(ch.star_box(df_t))
        with c2, card():
            styles.section("Points per game by phase")
            show(ch.club_record_chart(an.record_table(df_t, "tier").rename(columns={"tier": "club"})))
        with card():
            styles.section("Star vs other players: tests")
            tst = pd.DataFrame({
                "Measure": sv["metric"], "Stars (n)": sv["n_a"], "Others (n)": sv["n_b"],
                "Stars mean": sv["mean_a"].round(3), "Others mean": sv["mean_b"].round(3), "Difference": sv["diff"].round(3),
                "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(sv["ci_lo"], sv["ci_hi"])],
                "p (Welch)": [pfmt(p) for p in sv["p_welch"]], "p (Mann-Whitney)": [pfmt(p) for p in sv["p_mw"]],
                "Cohen's d": sv["cohen_d"].round(3), "Effect": [an.cohen_label(d) for d in sv["cohen_d"]],
                "Verdict (5%)": [an.sig_label(p) for p in sv["p_welch"]],
            })
            show_df(tst)
            cf = an.correlation_pair(df, "fifa_rating", "tpdi")
            st.latex(r"t=\dfrac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}},\qquad d=\dfrac{\bar x_1-\bar x_2}{s_p}")
            styles.callout(
                f"Welch's t-test does not assume the two groups have equal spread, and the Mann-Whitney test does not assume normality. "
                f"Across the whole squad, the rank correlation between FIFA rating and the team's drop is rho = {fmt(cf['spearman_rho'], 3, True)} "
                f"({'p = ' + pfmt(cf['spearman_p'])}). "
                + ("Higher-rated players do seem to cost the team more." if (r0["p_welch"] < an.ALPHA and r0["diff"] > 0) else
                   "In this data there is no reliable sign that losing a higher-rated player costs the team more."),
                "good" if r0["p_welch"] < an.ALPHA else "warn")
    st.divider()
    styles.section("Age and team performance drop", "Each dot is one injury and bigger dots mean more days out. The dashed line is the regression fit with its 95% band.", "Q6")
    reg = an.simple_regression(df)
    cp = an.correlation_pair(df)
    with card():
        show(ch.age_scatter(df, reg))
        m = st.columns(4)
        m[0].markdown(styles.kpi("Pearson r", fmt(cp["pearson_r"], 3, True), f"p = {pfmt(cp['pearson_p'])}", accent="#4F46E5"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Spearman ρ", fmt(cp["spearman_rho"], 3, True), f"p = {pfmt(cp['spearman_p'])}", accent="#0EA5E9"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("R²", fmt(reg["r2"], 4) if reg else "n/a", "share of variation explained by age", accent="#F59E0B"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Slope", fmt(reg["slope"], 3, True) if reg else "n/a", "goals per extra year of age", accent="#10B981"), unsafe_allow_html=True)
        if reg:
            st.latex(rf"\widehat{{\mathrm{{TPDI}}}} = {reg['intercept']:.3f} + {reg['slope']:.4f}\cdot \text{{age}}\qquad R^2={reg['r2']:.4f},\; p={pfmt(reg['p'])}")
            styles.callout(
                f"Each extra year of age goes with a {reg['slope']:+.3f} goal per match change in the team's drop, "
                f"and age explains {reg['r2']*100:.2f}% of the variation. This is "
                f"{'significant' if reg['p'] < an.ALPHA else 'not significant'} at the 5% level.",
                "good" if reg["p"] < an.ALPHA else "warn")
    with card():
        styles.section("Recovery by age band")
        ra = an.recovery_trend(df, "age_band").rename(columns={
            "age_band": "Age band", "episodes": "Episodes", "avg_days_out": "Avg days", "median_days_out": "Median days",
            "mean_rating_change": "Mean Δ rating", "sd_rating_change": "SD Δ", "pct_improved": "% improved"})
        show_df(ra, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Avg days", "Median days", "Mean Δ rating", "SD Δ", "% improved"]})

# =========================================================================== #
# WHAT-IF TOOLS
# =========================================================================== #
with t_what:
    styles.section("What-if tools", "Two planning tools that use the past patterns in the data", "Planning")
    with card():
        styles.section("Absence simulator", "How many points might a team lose if a player misses a run of matches?")
        s1, s2, s3, s4 = st.columns(4)
        sim_club = s1.selectbox("Club", ["All selected clubs"] + sorted(df["club"].unique()), key="sim_club")
        sim_tier = s2.selectbox("Player type", ["All players", f"Star players only ({STAR}+)", f"Other players (below {STAR})"], key="sim_tier")
        sim_matches = s3.slider("Matches missed", 1, 38, 6, key="sim_matches")
        sim_n = s4.select_slider("Simulations", options=[2000, 5000, 10000, 20000], value=10000, key="sim_n")
        pool = df if sim_club == "All selected clubs" else df[df["club"] == sim_club]
        if sim_tier.startswith("Star"):
            pool = pool[pool["fifa_rating"] >= STAR]
        elif sim_tier.startswith("Other"):
            pool = pool[pool["fifa_rating"] < STAR]
        sim = an.simulate_absence(pool, sim_matches, sim_n)
        if sim is None:
            st.info("Not enough match data for this choice. Pick more clubs or a wider player type.")
        else:
            m = st.columns(4)
            m[0].markdown(styles.kpi("Expected points lost", fmt(sim["mean"], 2, True), f"over {sim_matches} matches", accent="#F43F5E"), unsafe_allow_html=True)
            m[1].markdown(styles.kpi("80% range", f"{sim['lo80']:.0f} to {sim['hi80']:.0f}", "points lost", accent="#4F46E5"), unsafe_allow_html=True)
            m[2].markdown(styles.kpi("Chance of losing points", f"{sim['p_any']*100:.0f}%", "any net loss at all", accent="#F59E0B"), unsafe_allow_html=True)
            m[3].markdown(styles.kpi("Chance of losing 3+", f"{sim['p_3']*100:.0f}%", "the equivalent of one win", accent="#10B981"), unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                show(ch.sim_points_lost(sim))
            with g2:
                show(ch.sim_points_overlay(sim))
            st.latex(r"\mathbf p\sim\mathrm{Dirichlet}(\text{losses}+1,\ \text{draws}+1,\ \text{wins}+1),\qquad \text{Lost}=\text{Pts}_{\text{available}}-\text{Pts}_{\text{absent}}")
            styles.callout(
                f"How it works: results from {sim['n_before']:,} matches with the player available and {sim['n_during']:,} matches without him give the win, draw and loss "
                f"probabilities. Each of the {sim_n:,} simulations draws slightly different probabilities (so the uncertainty in the data is included), plays out "
                f"{sim_matches} matches both ways and compares the points. The simulated mean ({sim['mean']:+.2f}) matches the direct calculation "
                f"{sim_matches} x (PPG before - PPG during) = {sim['analytic']:+.2f}. "
                + ("The 80% range includes zero, so ordinary match luck is bigger than the average effect of the absence." if sim["lo80"] <= 0 <= sim["hi80"]
                   else "The 80% range stays away from zero, which suggests a real cost."), "warn")
    with card():
        styles.section("Expected time out", "How long do similar injuries usually keep a player out? (uses the full dataset, not the sidebar filters)")
        e1, e2, e3 = st.columns(3)
        inj_choice = e1.selectbox("Injury type", df_all["injury_type"].value_counts().index.tolist(), key="est_inj")
        pos_choice = e2.selectbox("Position group", ["Any"] + sorted(df_all["position_group"].unique()), key="est_pos")
        age_choice = e3.selectbox("Age band", ["Any"] + list(dp.AGE_BAND_ORDER), key="est_age")
        est = an.estimate_days_out(df_all, inj_choice, None if pos_choice == "Any" else pos_choice, None if age_choice == "Any" else age_choice)
        if est["n"] < 3:
            st.info("Too few similar injuries in the data for an estimate.")
        else:
            if est["fallback"]:
                st.caption(f"Too few exact matches, so the estimate uses a wider group: {est['basis']}.")
            m = st.columns(4)
            m[0].markdown(styles.kpi("Median time out", f"{est['median']:.0f} days", f"95% CI {est['med_lo']:.0f} to {est['med_hi']:.0f}", accent="#4F46E5"), unsafe_allow_html=True)
            m[1].markdown(styles.kpi("80% range", f"{est['p10']:.0f} to {est['p90']:.0f} days", "10th to 90th percentile", accent="#10B981"), unsafe_allow_html=True)
            m[2].markdown(styles.kpi("Typical range", f"{est['p25']:.0f} to {est['p75']:.0f} days", "middle 50% of cases", accent="#F59E0B"), unsafe_allow_html=True)
            m[3].markdown(styles.kpi("Similar cases", str(est["n"]), est["basis"], accent="#0EA5E9"), unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            with h1:
                show(ch.timeout_hist(est))
            with h2:
                show(ch.timeout_ecdf(est))
            within = pd.DataFrame({"Back within": [f"{k} days" for k in est["within"]], "Share of similar players": [f"{v*100:.0f}%" for v in est["within"].values()]})
            w1, w2 = st.columns([1, 2])
            with w1:
                show_df(within)
            with w2:
                st.latex(rf"\ln(\text{{days}})\sim\mathcal N(\mu,\sigma^2),\quad \mu={est['mu']:.2f},\ \sigma={est['sigma']:.2f}\ \Rightarrow\ \text{{median}}=e^{{\mu}}={np.exp(est['mu']):.0f}\text{{ days}}")
                styles.callout(f"Recovery times are skewed (a few very long injuries), so the median and percentiles are more honest than the mean "
                               f"({est['mean']:.0f} days). The log-normal curve is a smooth fit to the same data. English top-flight teams play roughly one match a "
                               f"week, so a median of {est['median']:.0f} days is about {est['median']/7:.0f} matches. This is a guide from past cases, not a medical prediction.")

# =========================================================================== #
# RE-INJURY RISK
# =========================================================================== #
with t_re:
    styles.section("Re-injury risk", "How long do players stay fit after coming back? Kaplan-Meier survival curves", "Risk")
    if "km" not in re_sum:
        st.info("Not enough return-to-play records with the current filters (at least 10 returns and 3 repeat injuries are needed).")
    else:
        mf = re_sum["median_free"]
        m = st.columns(4)
        m[0].markdown(styles.kpi("Hurt more than once", f"{re_sum['pct_multi']*100:.0f}%", f"of {re_sum['players']} injured players", accent="#F43F5E"), unsafe_allow_html=True)
        m[1].markdown(styles.kpi("Returns followed by a new injury", f"{re_sum['pct_followed']*100:.0f}%", f"{re_sum['events']} of {re_sum['n']} returns", accent="#F59E0B"), unsafe_allow_html=True)
        m[2].markdown(styles.kpi("Same injury type again", f"{re_sum['pct_same_type']*100:.0f}%", "of the repeat injuries", accent="#4F46E5"), unsafe_allow_html=True)
        m[3].markdown(styles.kpi("Median injury-free time", f"{mf:.0f} days" if mf == mf else "not reached", "after returning", accent="#10B981"), unsafe_allow_html=True)
        with card():
            styles.section("Probability of staying injury-free after a return", "The curve drops each time a player gets injured again; shaded bands are 95% confidence limits")
            by = st.radio("Split the curve by", ["Everyone", "Position group", "Age band"], horizontal=True, key="km_by")
            curves = {}
            if by == "Everyone":
                curves["All players"] = re_sum["km"]
            else:
                col = "position_group" if by == "Position group" else "age_band"
                for name, g in re_tbl[re_tbl[col] != "nan"].groupby(col):
                    if len(g) >= 15 and g["event"].sum() >= 3:
                        curves[str(name)] = an.kaplan_meier(g["duration"], g["event"])
            if curves:
                show(ch.km_chart(curves))
            else:
                st.info("No group has enough returns to draw a curve with the current filters.")
            if len(curves) >= 2:
                names = list(curves)
                k1, k2 = st.columns(2)
                ga = k1.selectbox("Compare group", names, index=0, key="lr_a")
                gb = k2.selectbox("with group", names, index=1, key="lr_b")
                if ga != gb:
                    col = "position_group" if by == "Position group" else "age_band"
                    ta, tb = re_tbl[re_tbl[col] == ga], re_tbl[re_tbl[col] == gb]
                    lr = an.logrank_test(ta["duration"], ta["event"], tb["duration"], tb["event"])
                    st.caption(f"Log-rank test, {ga} vs {gb}: chi-square = {fmt(lr['chi2'])}, p = {pfmt(lr['p'])} ({an.sig_label(lr['p'])} at 5%).")
            st.latex(r"\hat S(t)=\prod_{t_i\le t}\left(1-\dfrac{d_i}{n_i}\right),\qquad \chi^2_{\text{log-rank}}=\dfrac{(O_1-E_1)^2}{\mathrm{Var}(O_1-E_1)}")
        c1, c2 = st.columns([2, 3])
        with c1, card():
            styles.section("Chance of a new injury after returning")
            lm = re_sum["landmarks"]
            show_df(pd.DataFrame({"Days after return": lm["days"], "Chance of new injury": [f"{v*100:.0f}%" for v in lm["p_reinjured"]],
                                  "95% range": [f"{lo*100:.0f}% to {hi*100:.0f}%" for lo, hi in zip(lm["lo"], lm["hi"])]}))
        with c2, card():
            styles.section("Players with repeated injuries", "Three or more injuries in the data")
            rp = an.repeat_injury_players(df, re_tbl)
            if rp.empty:
                st.info("No player has three or more injuries with these filters.")
            else:
                show_df(rp.rename(columns={"player": "Player", "club": "Club", "injuries": "Injuries", "total_days_out": "Days out",
                                           "main_injury": "Main injury", "median_days_between": "Median days between injuries"}),
                        column_config={"Days out": st.column_config.NumberColumn(format="%.0f"),
                                       "Median days between injuries": st.column_config.NumberColumn(format="%.0f")})
        styles.callout(
            "Things to keep in mind: the dataset only contains players who were injured at least once, so these are risks for players who have already been hurt, "
            "not for the whole squad. Players still fit at the end of the data are 'censored', which is what the Kaplan-Meier method is designed to handle. "
            f"{re_info['overlap_dropped']} records where the next injury started before the previous return were left out. Several returns from the same player "
            "are not fully independent, so treat the confidence bands as a little optimistic.", "warn")

# =========================================================================== #
# 7. MATHS LAB
# =========================================================================== #
with t_math:
    styles.section("Maths and statistics", "Tests to check whether the differences in the data are real or could be chance. All formulas are coded in analytics.py using NumPy and SciPy.", "Maths")
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
        styles.section("1. Paired hypothesis tests", "Null hypothesis: the mean difference is zero. A positive difference means A is larger than B.")
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
        styles.callout("The t-test assumes the differences are roughly normal. The Wilcoxon signed-rank test does not, so when both agree "
                       "the result is more trustworthy. Cohen's d shows how big the effect is, not just whether there is one.")
    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("2. Bootstrap confidence interval", "5,000 resamples with replacement, seed 42")
        bs = an.bootstrap_ci(df["tpdi"])
        if len(bs["boot"]):
            show(ch.bootstrap_hist(bs["boot"], bs["lo"], bs["hi"], bs["mean"]))
            st.caption(f"Mean TPDI = {bs['mean']:+.3f}, 95% bootstrap CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}], n = {bs['n']}. "
                       + ("The interval does not include 0." if (bs["lo"] > 0 or bs["hi"] < 0) else "The interval includes 0, so there is no reliable average effect."))
        else:
            st.info("Not enough data for a bootstrap with the current filters.")
    with c2, card():
        styles.section("3. Multiple regression")
        st.latex(r"\hat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top y")
        mr = an.multiple_regression(df)
        if mr:
            tb = mr["table"].copy()
            tb["Significant"] = np.where(tb["p_value"] < an.ALPHA, "yes", "no")
            show_df(tb.rename(columns={"term": "Term", "coef": "β", "std_err": "SE", "t": "t", "p_value": "p"}),
                    column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["β", "SE", "t", "p"]})
            st.caption(f"TPDI ~ age + FIFA rating + days out. n = {mr['n']}, R² = {mr['r2']:.4f}, adjusted R² = {mr['adj_r2']:.4f}, F-test p = {pfmt(mr['f_p'])}")
        else:
            st.info("Not enough data for the regression with the current filters.")
    with card():
        styles.section("4. Spearman correlation matrix", "Rank correlation between the main metrics. It is not thrown off by outliers.")
        show(ch.correlation_heatmap(an.correlation_matrix(df)))
    styles.callout(
        "<b>Regression to the mean.</b> Players who get injured were often playing well beforehand, so the 'before' matches can look "
        "better than normal and the 'after' matches drift back. This means a drop in goal difference is not caused only by the absence, "
        "which is why confidence intervals and effect sizes are shown alongside the averages.",
        "warn")

# =========================================================================== #
# MANAGER BRIEF
# =========================================================================== #
with t_brief:
    styles.section("Manager's brief", "A shareable report of the findings for whatever the sidebar filters currently select", "Report")
    filters_text = (f"{len(sel_clubs)} of {len(clubs)} clubs, seasons {', '.join(sel_seasons)}, positions {', '.join(sel_groups)}, "
                    f"ages {age_rng[0]} to {age_rng[1]} ({len(df):,} injuries)")
    with card():
        styles.section("Recommendations preview", "Written from the current data, so they change when you change the filters")
        for r in report.recommendations(df, re_sum):
            st.markdown(f"- {r}")
    with card():
        styles.section("Download")
        st.write("The brief has the key numbers, the recommendations, four charts, the main tables and the statistical tests, in one HTML file. "
                 "Open it in a browser (it needs internet to draw the charts) and choose Print, then Save as PDF, to share it.")
        if st.button("Build the brief", key="build_brief"):
            with st.spinner("Building the report..."):
                st.session_state["brief_html"] = report.build_brief(df, filters_text, re_sum)
                st.session_state["brief_note"] = filters_text
        if "brief_html" in st.session_state:
            st.download_button("Download the brief (HTML)", st.session_state["brief_html"].encode("utf-8"),
                               file_name="footlens_managers_brief.html", mime="text/html", key="dl_brief")
            st.caption(f"Built for: {st.session_state['brief_note']}")

# =========================================================================== #
# 8. RESEARCH & DATA
# =========================================================================== #
with t_data:
    styles.section("Research questions", "These answers are recalculated from whatever the sidebar filters select", "Step 1")
    for a in an.research_answers(df):
        styles.qa_card(a["q"], a["a"], a["tab"])

    with card():
        styles.section("Background research")
        st.markdown(
            "- Studies of professional football, such as the UEFA Elite Club Injury Study, link fewer injuries with better league points "
            "and further progress in cup competitions. That is why this project looks at team results while a player is absent.\n"
            "- Injuries are usually graded by how many days a player misses. This dashboard uses up to 7, 8-28, 29-90 and over 90 days.\n"
            "- Analysts often compare a player's rating before and after returning, and give more weight to better players. "
            "The Injury Burden Index does this by weighting days lost with the FIFA rating.\n"
            "- With only three matches in each phase the averages are noisy, so confidence intervals, effect sizes and non-parametric "
            "tests are used to avoid reading too much into small differences.\n"
        )
        st.caption("Sources: Hägglund et al. (2013) Br J Sports Med 47:738-742; Ekstrand et al. (2011) Br J Sports Med 45:553-558; "
                   "Cohen (1988) Statistical Power Analysis for the Behavioral Sciences; reference links supplied in the assessment brief.")

    c1, c2 = st.columns(2)
    with c1, card():
        styles.section("Data cleaning report", "What was fixed in the full dataset")
        r = cleaning_report
        rep_rows = [
            ("Rows loaded / kept", f"{r['rows_raw']} / {r['rows_clean']}"),
            ("'N.A.' values converted to NaN", r["na_tokens_converted"]),
            ("Duplicate rows removed", r["duplicates_removed"]),
            ("Injury labels cleaned", f"{r['injury_labels_before']} down to {r['injury_labels_after']} ({r['injury_families']} groups)"),
            ("Return date year typos corrected", r["return_years_auto_corrected"]),
            ("Impossible return dates set to missing", r["return_dates_invalid"]),
            ("Ongoing injuries ('Present')", r["ongoing_injuries_present"]),
            ("Injuries with no rating data", r["rows_without_ratings"]),
        ]
        show_df(pd.DataFrame(rep_rows, columns=["Step", "Result"]).astype(str))
    with c2, card():
        styles.section("New columns")
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
        styles.section("Cleaned data (first 200 rows)")
        preview_cols = ["player", "club", "season", "position", "age", "injury", "injury_date", "days_out", "rating_before", "rating_after",
                        "rating_change", "gd_before", "gd_during", "tpdi", "severity"]
        show_df(df[preview_cols].head(200))
        st.caption("Use the button in the sidebar to download all of the filtered data.")
