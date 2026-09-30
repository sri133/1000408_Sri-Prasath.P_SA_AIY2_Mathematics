"""
report.py
---------
Builds the downloadable "Manager's Brief": a one-file HTML report (KPIs, findings, charts, tables and
recommendations) for the currently filtered data. Open it in a browser and use Print > Save as PDF.

Every sentence is generated from the data, so the brief always matches the filters used.
"""
from __future__ import annotations

import datetime as _dt
import html

import numpy as np
import pandas as pd

import analytics as an
import charts as ch

STAR_THRESHOLD = 82


def _p(p) -> str:
    if p is None or pd.isna(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def recommendations(df: pd.DataFrame, reinjury: dict | None = None) -> list[str]:
    """Plain-language, data-driven recommendations (hedged when the evidence is weak)."""
    recs: list[str] = []

    top = an.top_injuries(df, n=3, min_cases=3)
    if len(top):
        t = top.iloc[0]
        solid = t["mean_tpdi"] - t["ci95"] > 0
        recs.append(
            f"Watch {t['injury']} injuries first. The team's goal difference fell by {t['mean_tpdi']:.2f} goals per match "
            f"on average while the player was out ({int(t['cases'])} cases). "
            + ("The confidence interval stays above zero, so this looks like a real effect."
               if solid else "The interval includes zero, so treat this as a hint rather than proof."))

    cs = an.club_summary(df)
    if len(cs):
        c = cs.iloc[0]
        sub = df[df["club"] == c["club"]]
        pos = sub["position_group"].value_counts().index[0]
        typ = sub["injury_type"].value_counts().index[0]
        recs.append(
            f"{c['club']} lost the most player-quality time ({c['ibi_per_season']:.0f} star-days per season). "
            f"Its most injured position group is {pos.lower()}s and its most common injury is {typ.lower()}, "
            "so squad depth in that area is the first thing to check.")

    ct = an.cluster_tests(df)
    if ct["peak_month"]:
        if ct["month_p"] == ct["month_p"] and ct["month_p"] < an.ALPHA:
            recs.append(f"Injuries are not spread evenly through the year ({_p(ct['month_p'])}) and peak in {ct['peak_month']}. "
                        "Consider lighter training loads and more rotation in the weeks before that month. "
                        "The test covers August to May only, since June and July are off-season.")
        else:
            recs.append(f"Injuries peak in {ct['peak_month']}, but the spread across months is not significantly uneven "
                        f"({_p(ct['month_p'])}), so there is no strong calendar effect to plan around.")

    sv = an.star_vs_squad(df, STAR_THRESHOLD).iloc[0]
    if sv["n_a"] >= 5 and sv["n_b"] >= 5 and sv["p_welch"] == sv["p_welch"]:
        if sv["p_welch"] < an.ALPHA:
            recs.append(f"Losing a star player (rated {STAR_THRESHOLD}+) cost the team {sv['diff']:+.2f} goals per match more than losing other "
                        f"players ({_p(sv['p_welch'])}), so cover for key players deserves extra planning.")
        else:
            recs.append(f"Losing a star player (rated {STAR_THRESHOLD}+) did not cost the team significantly more than losing other players "
                        f"({_p(sv['p_welch'])}). Squad depth matters across the whole squad, not only for the best names.")

    pr = an.paired_test(df["rating_after"], df["rating_before"], "rating")
    if pr["n"] >= 3:
        recs.append(f"After returning, player ratings changed by {pr['mean_diff']:+.2f} on average "
                    f"(95% CI {pr['ci_lo']:+.2f} to {pr['ci_hi']:+.2f}), which is {an.sig_label(pr['p_t'])}. "
                    + ("There is no sign that players come back weaker on average." if pr["mean_diff"] >= -0.05 else
                       "Consider a staged return to play."))

    if reinjury and "landmarks" in reinjury:
        lm = reinjury["landmarks"].set_index("days")
        recs.append(f"{reinjury['pct_multi']*100:.0f}% of injured players were hurt more than once, and about "
                    f"{lm.loc[90, 'p_reinjured']*100:.0f}% were injured again within 90 days of returning. "
                    "A staged return-to-play plan for those first weeks is worth considering.")

    sim = an.simulate_absence(df, 6)
    if sim:
        recs.append(f"For a six-match absence the model expects about {sim['mean']:+.1f} points lost, with an 80% range of "
                    f"{sim['lo80']:.0f} to {sim['hi80']:.0f}. Most of that spread is ordinary match luck, so avoid reading a "
                    "single absence as a guaranteed loss of points.")
    return recs


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def _table(frame: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = ""
    for _, row in frame.iterrows():
        cells = ""
        for c in frame.columns:
            v = row[c]
            if c in fmt and pd.notna(v):
                v = format(v, fmt[c])
            elif isinstance(v, float):
                v = "n/a" if pd.isna(v) else f"{v:.2f}"
            cells += f"<td>{html.escape(str(v))}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _fig_html(fig, first: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False})


CSS = """
*{box-sizing:border-box}body{font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#0F172A;margin:0;background:#F8FAFC;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:36px 28px 60px}
header{border-bottom:3px solid #4F46E5;padding-bottom:16px;margin-bottom:22px}
header .eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#4F46E5;font-weight:700}
h1{font-size:28px;margin:6px 0 4px}h2{font-size:18px;margin:30px 0 10px;color:#1E1B4B}
.meta{color:#64748B;font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #E6EAF3;border-radius:12px;padding:12px 16px}
.kpi b{display:block;font-size:22px}.kpi span{font-size:12px;color:#64748B;text-transform:uppercase;letter-spacing:.06em}
ol.recs li{margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:8px 0}
th,td{border:1px solid #E6EAF3;padding:6px 10px;text-align:left}th{background:#EEF2FF}
.card{background:#fff;border:1px solid #E6EAF3;border-radius:12px;padding:8px;margin:10px 0}
.note{font-size:12.5px;color:#475569;background:#FFFBEB;border-left:4px solid #F59E0B;padding:10px 14px;border-radius:8px}
@media print{body{background:#fff}.wrap{padding:0}h2{break-after:avoid}.card,table{break-inside:avoid}}
"""


def build_brief(df: pd.DataFrame, filters_text: str, reinjury: dict | None = None) -> str:
    """Return the complete HTML document for the current selection."""
    k = an.headline_kpis(df)
    recs = recommendations(df, reinjury)
    top = an.top_injuries(df, n=5, min_cases=3)
    cs = an.club_summary(df)
    rec = an.record_table(df)
    tests = an.all_paired_tests(df)
    ct = an.cluster_tests(df)
    today = _dt.date.today().strftime("%d %B %Y")

    def kpi(label, value):
        return f'<div class="kpi"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'

    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Manager's Brief - Injuries and Team Performance</title><style>{CSS}</style></head><body><div class='wrap'>",
        "<header><div class='eyebrow'>FootLens Analytics | Manager's Brief</div><h1>Player Injuries and Team Performance</h1>"
        f"<div class='meta'>Generated {today}. Selection: {html.escape(filters_text)}</div></header>",
        "<div class='kpis'>"
        + kpi("Injuries", f"{k['injuries']:,}") + kpi("Players", f"{k['players']}") + kpi("Average days out", f"{k['avg_days_out']:.0f}")
        + kpi("Team drop (TPDI)", f"{k['avg_tpdi']:+.2f} goals/match") + kpi("Win rate while out", f"{k['winrate_during']*100:.1f}%")
        + kpi("Rating change after return", f"{k['avg_rating_change']:+.2f}") + "</div>",
        "<h2>Recommendations</h2><ol class='recs'>" + "".join(f"<li>{html.escape(r)}</li>" for r in recs) + "</ol>",
    ]

    first = True
    if len(top):
        parts.append("<h2>Injuries with the biggest team impact</h2>")
        parts.append(f"<div class='card'>{_fig_html(ch.top_injuries_bar(an.top_injuries(df, n=10, min_cases=3)), first)}</div>")
        first = False
        t = top[["injury", "cases", "mean_tpdi", "ci95", "avg_days_out"]].rename(
            columns={"injury": "Injury", "cases": "Cases", "mean_tpdi": "Mean TPDI", "ci95": "+/- 95% CI", "avg_days_out": "Avg days out"})
        parts.append(_table(t, {"Avg days out": ".0f", "Cases": ".0f"}))

    parts.append("<h2>Results before, during and after an absence</h2>")
    parts.append(f"<div class='card'>{_fig_html(ch.phase_record_chart(rec), first)}</div>")
    first = False
    parts.append(_table(rec[["phase", "matches", "wins", "draws", "losses", "win_rate", "ppg"]].rename(
        columns={"phase": "Phase", "matches": "Matches", "wins": "W", "draws": "D", "losses": "L", "win_rate": "Win rate", "ppg": "PPG"}),
        {"Win rate": ".3f", "PPG": ".2f", "Matches": ".0f", "W": ".0f", "D": ".0f", "L": ".0f"}))

    if not ct["obs"].empty:
        parts.append("<h2>When injuries happen</h2>")
        parts.append(f"<div class='card'>{_fig_html(ch.month_club_heatmap(ct['obs']), first)}</div>")

    parts.append("<h2>Injury burden by club</h2>")
    parts.append(f"<div class='card'>{_fig_html(ch.club_burden_chart(cs), first)}</div>")
    parts.append(_table(cs[["club", "injuries", "total_days_out", "mean_tpdi", "ibi_per_season"]].rename(
        columns={"club": "Club", "injuries": "Injuries", "total_days_out": "Days lost", "mean_tpdi": "Mean TPDI", "ibi_per_season": "IBI per season"}),
        {"Days lost": ".0f", "IBI per season": ".0f", "Injuries": ".0f"}))

    parts.append("<h2>Statistical tests</h2>")
    tt = pd.DataFrame({
        "Comparison": tests["test"], "n": tests["n"], "Mean difference": tests["mean_diff"].round(3),
        "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(tests["ci_lo"], tests["ci_hi"])],
        "p (t-test)": [_p(p).replace("p = ", "").replace("p ", "") for p in tests["p_t"]],
        "Cohen's d": tests["cohen_d"].round(2), "Verdict": [an.sig_label(p) for p in tests["p_t"]]})
    parts.append(_table(tt, {"n": ".0f"}))

    parts.append("<h2>How to read this</h2><div class='note'>TPDI (Team Performance Drop Index) is the team's average goal difference in the three "
                 "matches before the injury minus the average in the three matches missed; positive means the team did worse without the player. "
                 "Each phase has only three matches, so single injuries are noisy and conclusions rely on many injuries together. Players are often "
                 "injured after good form, so some of the drop may be regression to the mean rather than the absence itself. "
                 "The simulator and time-out numbers describe past patterns in this dataset and are not guarantees.</div>")
    parts.append("</div></body></html>")
    return "".join(parts)
