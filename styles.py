"""
styles.py
---------
UI layer: a light, clean, "SaaS-style" theme injected as CSS + small HTML component helpers
(hero banner, KPI cards, section headers, call-outs). Pure Python strings - no JavaScript needed.
"""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--indigo:#4F46E5;--indigo-d:#3730A3;--emerald:#10B981;--rose:#F43F5E;--ink:#0F172A;--muted:#64748B;--line:#E6EAF3;}
html, body, .stApp, [class*="css"]{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);}
.stApp{background:radial-gradient(1100px 480px at 8% -8%,#EEF2FF 0%,rgba(238,242,255,0) 62%),radial-gradient(900px 420px at 100% 0%,#ECFDF5 0%,rgba(236,253,245,0) 58%),#F8FAFC;}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1320px;}
h1,h2,h3,h4{letter-spacing:-0.02em;color:var(--ink);}
/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:24px;padding:38px 42px;margin-bottom:22px;background:linear-gradient(135deg,#FFFFFF 0%,#F1F4FF 55%,#ECFDF5 100%);border:1px solid var(--line);box-shadow:0 12px 40px -18px rgba(79,70,229,.30);animation:rise .7s ease both;}
.hero:before,.hero:after{content:"";position:absolute;border-radius:50%;filter:blur(46px);opacity:.55;animation:float 9s ease-in-out infinite;}
.hero:before{width:280px;height:280px;background:#C7D2FE;right:-60px;top:-90px;}
.hero:after{width:220px;height:220px;background:#A7F3D0;right:200px;bottom:-120px;animation-delay:-4s;}
.hero > *{position:relative;z-index:1;}
.eyebrow{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--indigo);background:#EEF2FF;border:1px solid #DDE3FF;padding:6px 12px;border-radius:999px;}
.hero h1{font-size:2.55rem;line-height:1.12;font-weight:800;margin:14px 0 10px 0;background:linear-gradient(90deg,#1E1B4B 0%,#4F46E5 60%,#10B981 120%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{font-size:1.02rem;color:#475569;max-width:760px;margin:0 0 18px 0;line-height:1.6;}
.chips{display:flex;flex-wrap:wrap;gap:10px;}
.chip{font-size:13px;font-weight:600;padding:8px 14px;border-radius:12px;background:#fff;border:1px solid var(--line);color:#334155;box-shadow:0 2px 8px -4px rgba(15,23,42,.15);}
.chip.math{font-family:'Cambria Math','STIX Two Math',Georgia,serif;font-style:italic;color:var(--indigo-d);background:#EEF2FF;border-color:#DDE3FF;}
/* ---------- KPI cards ---------- */
.kpi{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px 18px;box-shadow:0 8px 24px -16px rgba(15,23,42,.25);transition:transform .2s ease, box-shadow .2s ease;animation:rise .6s ease both;position:relative;overflow:hidden;}
.kpi:hover{transform:translateY(-4px);box-shadow:0 18px 34px -16px rgba(79,70,229,.35);}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent,#4F46E5);}
.kpi .l{font-size:11.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.kpi .v{font-size:1.75rem;font-weight:800;margin:4px 0 2px 0;letter-spacing:-.02em;}
.kpi .s{font-size:12.5px;color:var(--muted);}
.kpi .s.up{color:#059669;font-weight:600;} .kpi .s.down{color:#E11D48;font-weight:600;}
/* ---------- section headers, call-outs ---------- */
.sec{margin:6px 0 10px 0;}
.sec .t{font-size:1.25rem;font-weight:700;letter-spacing:-.01em;}
.sec .d{font-size:.93rem;color:var(--muted);margin-top:2px;}
.tag{display:inline-block;font-size:11px;font-weight:700;color:#4338CA;background:#EEF2FF;border-radius:8px;padding:3px 8px;margin-right:8px;vertical-align:middle;}
.callout{border:1px solid #DDE3FF;background:linear-gradient(135deg,#F5F7FF,#FFFFFF);border-left:4px solid var(--indigo);border-radius:14px;padding:14px 18px;margin:8px 0 14px 0;font-size:.95rem;line-height:1.6;color:#334155;}
.callout.warn{border-color:#FDE68A;border-left-color:#F59E0B;background:linear-gradient(135deg,#FFFBEB,#FFFFFF);}
.callout.good{border-color:#BBF7D0;border-left-color:#10B981;background:linear-gradient(135deg,#F0FDF4,#FFFFFF);}
.qa{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 20px;margin-bottom:12px;transition:box-shadow .2s;}
.qa:hover{box-shadow:0 12px 28px -18px rgba(79,70,229,.45);}
.qa .q{font-weight:700;margin-bottom:6px;} .qa .a{color:#334155;line-height:1.6;font-size:.95rem;}
.qa .w{font-size:12px;color:var(--indigo);font-weight:600;margin-top:8px;}
.errbox{max-width:760px;margin:40px auto;background:#fff;border:1px solid #FECDD3;border-radius:22px;padding:34px 38px;box-shadow:0 20px 50px -24px rgba(244,63,94,.45);}
.errbox h2{margin:0 0 8px 0;color:#BE123C;} .errbox code{background:#FFF1F2;color:#9F1239;padding:2px 8px;border-radius:6px;}
/* ---------- Streamlit widgets ---------- */
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--line);}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;margin-bottom:6px;}
.brand .logo{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#4F46E5,#10B981);display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#fff;border:1px solid var(--line);padding:6px;border-radius:16px;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{border-radius:11px;padding:8px 15px;font-weight:600;color:#475569;height:auto;}
.stTabs [aria-selected="true"]{background:#EEF2FF;color:#4338CA;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px;}
div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="border"]{border-radius:18px;}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:600;border:1px solid var(--line);transition:all .2s;}
.stDownloadButton>button:hover,.stButton>button:hover{border-color:var(--indigo);color:var(--indigo);transform:translateY(-1px);}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line);}
@keyframes rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
@keyframes float{0%,100%{transform:translate(0,0);}50%{transform:translate(-24px,18px);}}
@media (max-width:768px){.hero{padding:26px 22px;}.hero h1{font-size:1.75rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _h(text) -> str:
    return html.escape(str(text))


def hero() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="eyebrow">FootLens Analytics &middot; Mathematics for AI-II</span>'
        "<h1>The Mathematics of Player Injuries &amp; Team Performance</h1>"
        "<p>An interactive decision dashboard for technical directors and sports managers: quantify how injuries change "
        "match outcomes, test whether effects are statistically real, and plan rotation and squad depth with evidence.</p>"
        '<div class="chips">'
        '<span class="chip math">TPDI = GD&#772;<sub>before</sub> &minus; GD&#772;<sub>during</sub></span>'
        '<span class="chip">Hypothesis tests &amp; effect sizes</span>'
        '<span class="chip">Bootstrap &amp; regression</span>'
        '<span class="chip">&chi;&sup2; injury clustering</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown('<div class="brand"><div class="logo">&#9917;</div>FootLens</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "", accent: str = "#4F46E5", delay: float = 0.0) -> str:
    return (
        f'<div class="kpi" style="--accent:{accent};animation-delay:{delay}s">'
        f'<div class="l">{_h(label)}</div><div class="v">{_h(value)}</div><div class="s {tone}">{_h(sub)}</div></div>'
    )


def section(title: str, desc: str = "", tag: str = "") -> None:
    tag_html = f'<span class="tag">{_h(tag)}</span>' if tag else ""
    desc_html = f'<div class="d">{_h(desc)}</div>' if desc else ""
    st.markdown(f'<div class="sec"><div class="t">{tag_html}{_h(title)}</div>{desc_html}</div>', unsafe_allow_html=True)


def callout(text_html: str, kind: str = "") -> None:
    """text_html may contain simple <b>/<i> tags written by us (never user data)."""
    st.markdown(f'<div class="callout {kind}">{text_html}</div>', unsafe_allow_html=True)


def qa_card(q: str, a_md: str, where: str) -> None:
    a_html = html.escape(a_md)
    # convert **bold** markers to <b>
    parts = a_html.split("**")
    a_html = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    st.markdown(
        f'<div class="qa"><div class="q">{_h(q)}</div><div class="a">{a_html}</div><div class="w">See tab: {_h(where)}</div></div>',
        unsafe_allow_html=True,
    )


def error_card(title: str, message: str, steps: list[str]) -> None:
    items = "".join(f"<li>{_h(s)}</li>" for s in steps)
    st.markdown(
        f'<div class="errbox"><h2>&#9888;&#65039; {_h(title)}</h2><p>{_h(message)}</p>'
        f"<p><b>How to fix it</b></p><ol>{items}</ol></div>",
        unsafe_allow_html=True,
    )
