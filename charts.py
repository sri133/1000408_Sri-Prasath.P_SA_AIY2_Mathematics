"""
charts.py
---------
Step 4 of the brief: every Plotly figure used by the dashboard.
All functions take tidy DataFrames (from analytics.py) and return a plotly Figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_processing import MONTH_ORDER, PHASES, SEVERITY_ORDER

INDIGO, EMERALD, ROSE, AMBER, SKY, SLATE = "#4F46E5", "#10B981", "#F43F5E", "#F59E0B", "#0EA5E9", "#64748B"
PHASE_COLORS = {"before": INDIGO, "during": ROSE, "after": EMERALD}
GROUP_COLORS = {"Defender": INDIGO, "Midfielder": EMERALD, "Forward": ROSE, "Goalkeeper": AMBER, "Other": SLATE}
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"


def _style(fig: go.Figure, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family=FONT, size=13, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT, bordercolor="#E2E8F0"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# 1. Bar chart - top injuries by team performance drop
# --------------------------------------------------------------------------- #
def top_injuries_bar(tbl: pd.DataFrame) -> go.Figure:
    d = tbl.sort_values("mean_tpdi")
    colors = [ROSE if v > 0 else EMERALD for v in d["mean_tpdi"]]
    fig = go.Figure(
        go.Bar(
            x=d["mean_tpdi"], y=d["injury"], orientation="h", marker_color=colors,
            error_x=dict(type="data", array=d["ci95"].fillna(0), color="#94A3B8", thickness=1.4, width=4),
            customdata=np.stack([d["cases"], d["avg_days_out"], d["mean_ppg_drop"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Mean TPDI: %{x:.2f} goals/match<br>Cases: %{customdata[0]}"
                          "<br>Avg days out: %{customdata[1]:.0f}<br>PPG drop: %{customdata[2]:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Team Performance Drop Index (goals per match, error bars = 95% CI)", zeroline=True, zerolinecolor="#94A3B8")
    fig.update_yaxes(title="", showgrid=False)
    return _style(fig, height=max(340, 42 * len(d) + 90), legend=False)


# --------------------------------------------------------------------------- #
# 2. Line chart - player performance timeline (before / out / after)
# --------------------------------------------------------------------------- #
def player_timeline(row: pd.Series) -> go.Figure:
    cats = ["Before 1", "Before 2", "Before 3", "Out 1", "Out 2", "Out 3", "After 1", "After 2", "After 3"]
    ratings = [row.get(f"before_{i}_rating") for i in (1, 2, 3)] + [None] * 3 + [row.get(f"after_{i}_rating") for i in (1, 2, 3)]
    gds = [row.get(f"{p}_{i}_gd") for p in PHASES for i in (1, 2, 3)]
    phases = [p for p in PHASES for _ in range(3)]
    opps = [row.get(f"{p}_{i}_opp") for p in PHASES for i in (1, 2, 3)]
    ratings = [None if (r is None or pd.isna(r)) else float(r) for r in ratings]
    gds = [None if (g is None or pd.isna(g)) else float(g) for g in gds]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_vrect(x0=2.5, x1=5.5, fillcolor=ROSE, opacity=0.06, line_width=0,
                  annotation_text="Player injured", annotation_position="top", annotation_font_color=ROSE)
    fig.add_trace(
        go.Bar(x=cats, y=gds, name="Team goal difference", marker_color=[PHASE_COLORS[p] for p in phases], opacity=0.35,
               customdata=opps, hovertemplate="%{x} vs %{customdata}<br>Team GD: %{y}<extra></extra>"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=cats, y=ratings, name="Player rating", mode="lines+markers", connectgaps=False,
                   line=dict(color=INDIGO, width=3), marker=dict(size=10, color="white", line=dict(color=INDIGO, width=3)),
                   hovertemplate="%{x}<br>Rating: %{y:.1f}<extra></extra>"),
        secondary_y=False,
    )
    fig.update_yaxes(title_text="Player rating (0-10)", range=[3, 10], secondary_y=False)
    fig.update_yaxes(title_text="Team goal difference", showgrid=False, zeroline=True, zerolinecolor="#CBD5E1", secondary_y=True)
    return _style(fig, height=420)


# --------------------------------------------------------------------------- #
# 3. Heatmap - injuries by club x month
# --------------------------------------------------------------------------- #
def month_club_heatmap(mat: pd.DataFrame, residuals: pd.DataFrame | None = None, mode: str = "Injury count") -> go.Figure:
    if mode.startswith("Standardised") and residuals is not None and not residuals.empty:
        z = residuals.reindex(index=mat.index, columns=MONTH_ORDER).fillna(0)
        colorscale, zmid, fmt, title = "RdBu_r", 0, "%{z:.1f}", "Std. residual"
    else:
        z = mat
        colorscale = [[0, "#F5F7FF"], [0.5, "#A5B4FC"], [1, INDIGO]]
        zmid, fmt, title = None, "%{z:.0f}", "Injuries"
    fig = go.Figure(
        go.Heatmap(z=z.to_numpy(), x=list(z.columns), y=list(z.index), colorscale=colorscale, zmid=zmid,
                   texttemplate=fmt, xgap=3, ygap=3, colorbar=dict(title=title, thickness=12),
                   hovertemplate="%{y} - %{x}<br>Value: %{z:.2f}<extra></extra>")
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    return _style(fig, height=max(340, 46 * len(z) + 100), legend=False)


# --------------------------------------------------------------------------- #
# 4. Scatter - age vs performance drop index (+ regression & 95% band)
# --------------------------------------------------------------------------- #
def age_scatter(df: pd.DataFrame, reg: dict | None) -> go.Figure:
    d = df.dropna(subset=["age", "tpdi"]).copy()
    d["marker_size"] = d["days_out"].fillna(d["days_out"].median()).clip(lower=3, upper=200)
    fig = px.scatter(
        d, x="age", y="tpdi", color="position_group", size="marker_size", size_max=20, opacity=0.7,
        color_discrete_map=GROUP_COLORS,
        hover_data={"player": True, "club": True, "injury": True, "days_out": True, "marker_size": False, "age": True, "tpdi": ":.2f"},
        labels={"age": "Player age (years)", "tpdi": "Team Performance Drop Index", "position_group": "Position"},
    )
    if reg:
        fig.add_trace(go.Scatter(x=np.r_[reg["grid"], reg["grid"][::-1]], y=np.r_[reg["hi"], reg["lo"][::-1]], fill="toself",
                                 fillcolor="rgba(79,70,229,0.10)", line=dict(width=0), hoverinfo="skip", name="95% CI band"))
        fig.add_trace(go.Scatter(x=reg["grid"], y=reg["fit"], mode="lines", line=dict(color="#1E1B4B", width=2.5, dash="dash"),
                                 name=f"OLS fit (R\u00b2 = {reg['r2']:.3f})"))
    fig.add_hline(y=0, line_color="#CBD5E1", line_width=1)
    return _style(fig, height=470)


# --------------------------------------------------------------------------- #
# Extra charts
# --------------------------------------------------------------------------- #
def severity_box(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["tpdi", "severity"])
    fig = px.box(d, x="severity", y="tpdi", color="severity", points="all", category_orders={"severity": SEVERITY_ORDER},
                 color_discrete_sequence=[SKY, INDIGO, AMBER, ROSE], hover_data=["player", "club", "injury"],
                 labels={"severity": "Injury severity (days out)", "tpdi": "Team Performance Drop Index"})
    fig.update_traces(marker=dict(size=4, opacity=0.5), jitter=0.4)
    fig.add_hline(y=0, line_color="#CBD5E1", line_width=1)
    return _style(fig, height=400, legend=False)


def phase_record_chart(rec: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Win rate (Wilson 95% CI)", "Points per game"))
    phases = list(rec["phase"])
    cols = [PHASE_COLORS[p] for p in phases]
    fig.add_trace(go.Bar(x=[p.title() for p in phases], y=rec["win_rate"] * 100, marker_color=cols,
                         error_y=dict(type="data", symmetric=False, array=(rec["win_hi"] - rec["win_rate"]) * 100,
                                      arrayminus=(rec["win_rate"] - rec["win_lo"]) * 100, color="#94A3B8"),
                         text=[f"{v*100:.1f}%" for v in rec["win_rate"]], textposition="inside", showlegend=False,
                         hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=[p.title() for p in phases], y=rec["ppg"], marker_color=cols,
                         text=[f"{v:.2f}" for v in rec["ppg"]], textposition="inside", showlegend=False,
                         hovertemplate="%{x}: %{y:.2f} pts/game<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="Win rate (%)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Points / game", rangemode="tozero", row=1, col=2)
    return _style(fig, height=380, legend=False)


def club_record_chart(rec_by_club: pd.DataFrame) -> go.Figure:
    fig = px.bar(rec_by_club, x="club", y="ppg", color="phase", barmode="group",
                 color_discrete_map=PHASE_COLORS, category_orders={"phase": list(PHASES)},
                 labels={"ppg": "Points per game", "club": "", "phase": "Phase"})
    return _style(fig, height=380)


def club_burden_chart(cs: pd.DataFrame) -> go.Figure:
    d = cs.sort_values("ibi_per_season")
    fig = go.Figure(go.Bar(
        x=d["ibi_per_season"], y=d["club"], orientation="h",
        marker=dict(color=d["ibi_z"], colorscale=[[0, "#C7D2FE"], [1, INDIGO]], showscale=False),
        text=[f"z = {z:+.2f}" for z in d["ibi_z"]], textposition="outside",
        customdata=np.stack([d["injuries"], d["total_days_out"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>IBI: %{x:.0f} star-days/season<br>Injuries: %{customdata[0]}<br>Days lost: %{customdata[1]:.0f}<extra></extra>"))
    fig.update_xaxes(title="Injury Burden Index (star-days lost per season)")
    fig.update_yaxes(showgrid=False, title="")
    return _style(fig, height=max(320, 44 * len(d) + 80), legend=False)


def club_drop_chart(cs: pd.DataFrame) -> go.Figure:
    d = cs.sort_values("mean_tpdi", ascending=False)
    fig = go.Figure(go.Bar(x=d["club"], y=d["mean_tpdi"], marker_color=[ROSE if v > 0 else EMERALD for v in d["mean_tpdi"]],
                           hovertemplate="%{x}: %{y:.2f} goals/match<extra></extra>"))
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
    fig.update_yaxes(title="Mean TPDI (goals / match)")
    return _style(fig, height=340, legend=False)


def rating_change_hist(df: pd.DataFrame) -> go.Figure:
    x = df["rating_change"].dropna()
    fig = go.Figure(go.Histogram(x=x, nbinsx=30, marker_color=INDIGO, opacity=0.8,
                                 hovertemplate="Change %{x}<br>Episodes: %{y}<extra></extra>"))
    if len(x):
        fig.add_vline(x=0, line_color="#94A3B8", line_dash="dot")
        fig.add_vline(x=x.mean(), line_color=ROSE, line_width=2, annotation_text=f"mean {x.mean():+.2f}", annotation_font_color=ROSE)
    fig.update_xaxes(title="Rating change after return (after - before)")
    fig.update_yaxes(title="Injury episodes")
    return _style(fig, height=340, legend=False)


def bootstrap_hist(boot: np.ndarray, lo: float, hi: float, mean: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=boot, nbinsx=40, marker_color=SKY, opacity=0.85))
    fig.add_vrect(x0=lo, x1=hi, fillcolor=INDIGO, opacity=0.12, line_width=0, annotation_text="95% CI", annotation_position="top left")
    fig.add_vline(x=mean, line_color=INDIGO, line_width=2)
    fig.add_vline(x=0, line_color=ROSE, line_dash="dash", annotation_text="no effect", annotation_font_color=ROSE)
    fig.update_xaxes(title="Bootstrap distribution of mean TPDI (5,000 resamples)")
    fig.update_yaxes(title="Frequency")
    return _style(fig, height=340, legend=False)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
                               colorscale="RdBu_r", texttemplate="%{z:.2f}", xgap=2, ygap=2,
                               colorbar=dict(title="\u03c1", thickness=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height=470, legend=False)
