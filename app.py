from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ProcessPulse AI | Operations Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM DASHBOARD THEME
# ============================================================



st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #070b16;
        --panel: #101a2d;
        --border: rgba(105, 151, 220, 0.24);
        --text: #f4f7ff;
        --muted: #a6b5d0;
        --cyan: #35e0f2;
        --blue: #4b8bff;
        --purple: #ad7bff;
        --pink: #ff70c6;
        --green: #43e6b0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, rgba(36, 109, 255, .13), transparent 32%),
            radial-gradient(ellipse at 90% 10%, rgba(166, 72, 255, .13), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 11, 22, .92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111d34, #090f1e);
        border-right: 1px solid var(--border);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1800px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #f6f8ff;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #bdc9df;
    }

    /* Brand */

    .pp-brand {
        padding: .8rem 0 1.3rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.3rem;
    }

    .pp-brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
        letter-spacing: -.045em;
    }

    .pp-brand-name span {
        color: var(--cyan);
        text-shadow: 0 0 20px rgba(53, 224, 242, .25);
    }

    .pp-brand-sub, .pp-eyebrow, .pp-section-label {
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .7rem;
        font-weight: 700;
    }

    .pp-brand-sub {
        margin-top: .25rem;
    }

    /* Hero */

    .pp-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(72, 180, 255, .45);
        border-radius: 22px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        background:
            radial-gradient(circle at 85% 10%, rgba(57, 215, 239, .18), transparent 32%),
            linear-gradient(125deg, #111e37 0%, #10152e 55%, #211638 100%);
        box-shadow: 0 15px 55px rgba(26, 106, 255, .10);
    }

    .pp-hero:before {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -100px;
        top: -150px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(173, 123, 255, .3), transparent 70%);
        pointer-events: none;
    }

    .pp-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.4rem, 4vw, 3.7rem);
        font-weight: 700;
        line-height: 1.06;
        letter-spacing: -.065em;
        color: #fff;
    }

    .pp-title span {
        background: linear-gradient(90deg, #35e0f2, #6895ff, #c08aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .pp-subtitle {
        color: #c3cee3;
        max-width: 780px;
        line-height: 1.75;
    }

    .pp-chip {
        display: inline-block;
        padding: .5rem .8rem;
        margin: .2rem .25rem .2rem 0;
        border-radius: 10px;
        border: 1px solid rgba(75, 190, 245, .35);
        background: rgba(18, 39, 68, .8);
        color: #e0efff;
        font-size: .78rem;
    }

    /* KPI cards */

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #14233b, #0c1526);
        border: 1px solid rgba(105, 151, 220, .27);
        border-radius: 16px;
        padding: 1.1rem;
        min-height: 120px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, .17);
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(53, 224, 242, .65);
        box-shadow: 0 10px 32px rgba(53, 224, 242, .08);
    }

    [data-testid="stMetricLabel"] {
        color: #bac9e1 !important;
        font-size: .82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #f6f8ff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    /* Secondary metric panels */

    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
        backdrop-filter: blur(12px);
    }

    /* Tabs */

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .45rem;
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: #c0cce0;
        border: 1px solid transparent;
        border-radius: 10px 10px 0 0;
        padding: .8rem .9rem;
        font-size: .84rem;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--cyan) !important;
        background: linear-gradient(180deg, rgba(53, 224, 242, .12), rgba(53, 224, 242, .025));
        border: 1px solid rgba(53, 224, 242, .3);
        border-bottom: 2px solid var(--cyan);
    }

    /* Insight cards */

    .pp-insight {
        border: 1px solid rgba(53, 224, 242, .3);
        border-radius: 16px;
        padding: 1.2rem;
        background: linear-gradient(125deg, #102d40, #111a30 75%);
        min-height: 135px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, .13);
    }

    .pp-insight-label {
        color: var(--cyan);
        text-transform: uppercase;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .14em;
        margin-bottom: .5rem;
    }

    .pp-insight-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }

    .pp-insight-body {
        color: #c2cde0;
        font-size: .88rem;
        line-height: 1.65;
    }

    /* Chart and table containers */

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(105, 151, 220, .22);
        border-radius: 15px;
        overflow: hidden;
        background: rgba(13, 24, 42, .7);
    }

    /* Inputs */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #111b2d;
        border-color: rgba(130, 164, 212, .3);
        border-radius: 10px;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(53, 224, 242, .42);
        border-radius: 10px;
        background: linear-gradient(100deg, #126c91, #4648a4);
        color: #fff;
        font-weight: 600;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--cyan);
        box-shadow: 0 0 22px rgba(53, 224, 242, .15);
        color: #fff;
        transform: translateY(-1px);
    }

    /* Footer */

    .pp-footer {
        border-top: 1px solid var(--border);
        padding-top: 1.2rem;
        margin-top: 2rem;
        color: #8fa3c2;
        font-size: .78rem;
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        .pp-hero {
            padding: 1.2rem;
        }
        .pp-title {
            font-size: 2.2rem;
        }
    }
    .pp-hero {
    position: relative;
    margin-top: 1.5rem !important;
    margin-bottom: 2.5rem !important;
    padding: 2.5rem !important;
}

/* Add breathing room after the KPI cards */
div[data-testid="stMetric"] {
    min-height: 115px;
    padding: 1.1rem !important;
}

/* Space the secondary KPI row from the main KPI row */
div[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
}

/* Separate insight cards from the secondary KPI row */
.pp-insight {
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}

/* Add space before the dashboard tabs */
div[data-testid="stTabs"] {
    margin-top: 2.5rem !important;
}

/* Give tab content room below the tab labels */
div[data-testid="stTabPanel"] {
    padding-top: 1.5rem !important;
}

/* Respect Streamlit's selected light/dark theme */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
}
    
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "operations_data.csv"
)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    data["delay_minutes"] = (
        data["actual_minutes"] - data["planned_minutes"]
    ).clip(lower=0)
    data["delay_flag"] = (
        data["actual_minutes"]
        > data["planned_minutes"] * 1.15
    ).astype(int)
    data["defect_rate_pct"] = np.where(
        data["units_processed"] > 0,
        data["defect_count"]
        / data["units_processed"] * 100,
        0,
    )

    data["month"] = (
        data["timestamp"].dt.to_period("M").astype(str)
    )

    return data


df = load_data(str(DATA_PATH))
# ============================================================
# SIDEBAR — FILTERS AND NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div class="pp-brand">
        <div class="pp-brand-name">
            ⚙ Process<span>Pulse</span>
        </div>
        <div class="pp-brand-sub">
            Operations Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Workspace")
st.sidebar.caption("Filter the data to investigate operations.")

date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    date_filtered = df[
        df["timestamp"].dt.date.between(
            date_range[0],
            date_range[1],
        )
    ].copy()
else:
    date_filtered = df[
        df["timestamp"].dt.date == date_range
    ].copy()

if date_filtered.empty:
    st.warning(
        "No records fall within this date range. "
        "Please select a wider range."
    )
    st.stop()

steps = st.sidebar.multiselect(
    "Process steps",
    sorted(date_filtered["process_step"].dropna().unique()),
    default=sorted(date_filtered["process_step"].dropna().unique()),
)

machines = st.sidebar.multiselect(
    "Machines",
    sorted(date_filtered["machine"].dropna().unique()),
    default=sorted(date_filtered["machine"].dropna().unique()),
)

shifts = st.sidebar.multiselect(
    "Shifts",
    sorted(date_filtered["shift"].dropna().unique()),
    default=sorted(date_filtered["shift"].dropna().unique()),
)

filtered = date_filtered[
    date_filtered["process_step"].isin(steps)
    & date_filtered["machine"].isin(machines)
    & date_filtered["shift"].isin(shifts)
].copy()

if filtered.empty:
    st.warning(
        "No records match these filters. "
        "Change the sidebar selections."
    )
    st.stop()

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="
        background:rgba(24,42,64,.65);
        border:1px solid rgba(113,160,205,.2);
        border-radius:12px;
        padding:0.9rem;
    ">
        <div style="
            color:#35d7ef;
            font-size:.69rem;
            font-weight:700;
            letter-spacing:.12em;
            text-transform:uppercase;
        ">
            Dataset status
        </div>
        <div style="
            color:#f4f7ff;
            font-size:1.35rem;
            font-weight:700;
            margin-top:.35rem;
        ">
            {len(df):,} records
        </div>
        <div style="
            color:#9aacc5;
            font-size:.77rem;
            margin-top:.25rem;
        ">
            Synthetic portfolio dataset
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("")
st.sidebar.caption(
    "Built with Python · Streamlit · Plotly · scikit-learn"
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    f"""
    <div class="pp-hero">
      <div class="pp-eyebrow">
        ⚡ Operations intelligence platform
      </div>
      <div class="pp-title">
        See the signal.<br>
        <span>Improve the process.</span>
      </div>
      <div class="pp-subtitle">
        Explore delays, defects, downtime, and machine performance
        in one interactive workspace. Use analytics and a
        machine-learning demo to identify where an operations
        team could investigate first.
      </div>
      <div class="pp-chip-row">
        <span class="pp-chip">✦ Interactive analytics</span>
        <span class="pp-chip">◈ Delay-risk classification</span>
        <span class="pp-chip">◎ Synthetic portfolio dataset</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_units = int(filtered["units_processed"].sum())

delay_pct = float(
    filtered["delay_flag"].mean() * 100
)

defect_pct = float(
    filtered["defect_count"].sum()
    / max(filtered["units_processed"].sum(), 1)
    * 100
)

avg_downtime = float(
    filtered["downtime_minutes"].mean()
)

avg_delay = float(
    filtered["delay_minutes"].mean()
)

st.markdown(
    """
    <div class="pp-section-label">
        ◈ Performance overview · Live filter selection
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Operations records",
    f"{len(filtered):,}",
    help="Records matching the selected filters.",
)

k2.metric(
    "Units processed",
    f"{total_units:,}",
    help="Total units represented by the selected records.",
)

k3.metric(
    "High-delay rate",
    f"{delay_pct:.1f}%",
    help=(
        "Percentage of records where actual processing "
        "time exceeds 115% of planned processing time."
    ),
)

k4.metric(
    "Defect rate",
    f"{defect_pct:.2f}%",
    help="Total defects divided by total units processed.",
)

# Secondary statistics

s1, s2, s3 = st.columns(3)

s1.markdown(
    f"""
    <div style="
        border:1px solid rgba(113,160,205,.17);
        background:rgba(14,26,44,.7);
        border-radius:12px;
        padding:.8rem 1rem;
        margin-top:.35rem;
    ">
        <div style="color:#91a7c2;font-size:.75rem;">
            Average delay
        </div>
        <div style="
            color:#f4f7ff;
            font-size:1.25rem;
            font-weight:700;
            margin-top:.2rem;
        ">
            {avg_delay:.1f}
            <span style="font-size:.75rem;color:#91a7c2;">
                min
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

s2.markdown(
    f"""
    <div style="
        border:1px solid rgba(113,160,205,.17);
        background:rgba(14,26,44,.7);
        border-radius:12px;
        padding:.8rem 1rem;
        margin-top:.35rem;
    ">
        <div style="color:#91a7c2;font-size:.75rem;">
            Average machine downtime
        </div>
        <div style="
            color:#f4f7ff;
            font-size:1.25rem;
            font-weight:700;
            margin-top:.2rem;
        ">
            {avg_downtime:.1f}
            <span style="font-size:.75rem;color:#91a7c2;">
                min
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

s3.markdown(
    f"""
    <div style="
        border:1px solid rgba(113,160,205,.17);
        background:rgba(14,26,44,.7);
        border-radius:12px;
        padding:.8rem 1rem;
        margin-top:.35rem;
    ">
        <div style="color:#91a7c2;font-size:.75rem;">
            Records currently analyzed
        </div>
        <div style="
            color:#f4f7ff;
            font-size:1.25rem;
            font-weight:700;
            margin-top:.2rem;
        ">
            {len(filtered):,}
            <span style="font-size:.75rem;color:#91a7c2;">
                / {len(df):,}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTOMATIC OPERATIONAL INSIGHTS
# ============================================================

step_summary = (
    filtered.groupby("process_step", as_index=False)
    .agg(
        avg_delay=("delay_minutes", "mean"),
        high_delay_rate=("delay_flag", "mean"),
        avg_downtime=("downtime_minutes", "mean"),
    )
)

step_summary["high_delay_rate_pct"] = (
    step_summary["high_delay_rate"] * 100
)

leading_step = step_summary.sort_values(
    ["high_delay_rate_pct", "avg_delay"],
    ascending=False,
).iloc[0]

insight_left, insight_right = st.columns([1.65, 1])

with insight_left:
    st.markdown(
    f"""
    <div class="pp-insight">
        <div class="pp-insight-label">✦ Signal to investigate</div>
        <div class="pp-insight-title">{leading_step["process_step"]}</div>
        <div class="pp-insight-body">
            This process step has the highest high-delay rate
            in the current selection:
            <b style="color:#35d7ef;">
                {leading_step["high_delay_rate_pct"]:.1f}%
            </b>.
            Compare it against other process steps and
            machines in the Bottleneck Analysis tab.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with insight_right:
    st.markdown(
    f"""
    <div class="pp-insight"
         style="border-color:rgba(155,108,255,0.35);">
        <div class="pp-insight-label"
             style="color:#b89aff;">
            ◈ Average downtime
        </div>
        <div class="pp-insight-title"
             style="font-size:1.65rem;">
            {avg_downtime:.1f}
            <span style="font-size:.85rem; opacity:.7;">
                min / record
            </span>
        </div>
        <div class="pp-insight-body">
            Average downtime across the selected
            operations records.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
    st.markdown("")
    st.markdown(
    """
    <div style="
        height:1px;
        background:linear-gradient(
            90deg,
            rgba(53,215,239,.4),
            rgba(155,108,255,.15),
            transparent
        );
        margin:1.4rem 0 1rem;
    "></div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊  Operations Overview",
        "🔎  Bottleneck Analysis",
        "🤖  AI Delay-Risk Model",
        "📘  Data & Methodology",
    ]
)
# ============================================================
# TAB 1: OPERATIONS OVERVIEW
# ============================================================

with tab1:

    st.markdown("### Operations at a glance")
    st.caption(
        "Explore processing-time trends, production quality, "
        "throughput, and machine downtime using the selected filters."
    )

    # Shared styling for Plotly charts
    chart_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#b9c9df",
            size=12,
        ),
        title_font=dict(
            family="Space Grotesk, sans-serif",
            size=17,
            color="#f4f7ff",
        ),
        margin=dict(l=15, r=15, t=60, b=20),
        hoverlabel=dict(
            bgcolor="#122138",
            bordercolor="#35d7ef",
            font=dict(color="#f4f7ff"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b9c9df"),
        ),
    )

    # --------------------------------------------------------
    # CHART 1: MONTHLY DELAY TREND
    # --------------------------------------------------------

    left_chart, right_chart = st.columns(2, gap="large")

    with left_chart:

        monthly = (
            filtered.groupby("month", as_index=False)
            .agg(
                average_delay=("delay_minutes", "mean"),
                records=("timestamp", "count"),
            )
            .sort_values("month")
        )

        fig = px.line(
            monthly,
            x="month",
            y="average_delay",
            markers=True,
            title="Average delay by month",
            labels={
                "month": "Month",
                "average_delay": "Average delay (minutes)",
            },
            hover_data={"records": True},
        )

        fig.update_traces(
            line=dict(color="#35d7ef", width=3),
            marker=dict(
                size=8,
                color="#35d7ef",
                line=dict(color="#0b1020", width=2),
            ),
            fill="tozeroy",
            fillcolor="rgba(53,215,239,0.07)",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average delay: %{y:.2f} min"
                "<extra></extra>"
            ),
        )

        fig.update_layout(**chart_layout)
        fig.update_xaxes(
            showgrid=False,
            linecolor="rgba(148,163,184,.18)",
        )
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,.10)",
            zerolinecolor="rgba(148,163,184,.15)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="monthly_delay_chart",
        )

    # --------------------------------------------------------
    # CHART 2: PLANNED VS ACTUAL TIME
    # --------------------------------------------------------

    with right_chart:

        time_summary = (
            filtered.groupby("process_step", as_index=False)
            .agg(
                planned=("planned_minutes", "mean"),
                actual=("actual_minutes", "mean"),
            )
        )

        time_long = time_summary.melt(
            id_vars="process_step",
            value_vars=["planned", "actual"],
            var_name="time_type",
            value_name="minutes",
        )

        fig = px.bar(
            time_long,
            x="process_step",
            y="minutes",
            color="time_type",
            barmode="group",
            title="Planned vs actual processing time",
            labels={
                "process_step": "Process step",
                "minutes": "Average minutes",
                "time_type": "Time type",
            },
            color_discrete_map={
                "planned": "#35d7ef",
                "actual": "#8b78ff",
            },
            category_orders={
                "time_type": ["planned", "actual"],
            },
            custom_data=["time_type"],
        )

        fig.update_traces(
            marker_line_width=0,
            opacity=0.92,
            hovertemplate=(
                "Process: %{x}<br>"
                "Time: %{y:.2f} min"
                "<extra></extra>"
            ),
        )

        fig.update_layout(**chart_layout)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,.10)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="planned_actual_chart",
        )

    # --------------------------------------------------------
    # CHART 3: DEFECT RATE BY PROCESS STEP
    # --------------------------------------------------------

    left_chart, right_chart = st.columns(2, gap="large")

    with left_chart:

        defect_summary = (
            filtered.groupby("process_step", as_index=False)
            .agg(
                defects=("defect_count", "sum"),
                units=("units_processed", "sum"),
            )
        )

        defect_summary["defect_rate_pct"] = (
            defect_summary["defects"]
            / defect_summary["units"].clip(lower=1)
            * 100
        )

        defect_summary = defect_summary.sort_values(
            "defect_rate_pct",
            ascending=False,
        )

        fig = px.bar(
            defect_summary,
            x="process_step",
            y="defect_rate_pct",
            title="Quality signals · Defect rate",
            labels={
                "process_step": "Process step",
                "defect_rate_pct": "Defect rate (%)",
            },
            color="defect_rate_pct",
            color_continuous_scale=[
                "#35d7ef",
                "#3988ff",
                "#9b6cff",
            ],
            hover_data={
                "defects": True,
                "units": True,
                "defect_rate_pct": ":.3f",
            },
        )

        fig.update_traces(
            marker_line_width=0,
            opacity=0.92,
            hovertemplate=(
                "Process: %{x}<br>"
                "Defect rate: %{y:.3f}%"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            **chart_layout,
            coloraxis_showscale=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,.10)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="defect_rate_chart",
        )

    # --------------------------------------------------------
    # CHART 4: MACHINE DOWNTIME
    # --------------------------------------------------------

    with right_chart:

        machine_summary = (
            filtered.groupby("machine", as_index=False)
            .agg(
                average_downtime=("downtime_minutes", "mean"),
                average_delay=("delay_minutes", "mean"),
            )
            .sort_values("average_downtime", ascending=False)
        )

        fig = px.bar(
            machine_summary,
            x="machine",
            y="average_downtime",
            title="Machine downtime comparison",
            labels={
                "machine": "Machine",
                "average_downtime": "Average downtime (minutes)",
            },
            color="average_downtime",
            color_continuous_scale=[
                "#3988ff",
                "#746dff",
                "#b18aff",
            ],
            hover_data={
                "average_delay": ":.2f",
            },
        )

        fig.update_traces(
            marker_line_width=0,
            opacity=0.92,
            hovertemplate=(
                "Machine: %{x}<br>"
                "Average downtime: %{y:.2f} min"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            **chart_layout,
            coloraxis_showscale=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,.10)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="machine_downtime_chart",
        )

    # --------------------------------------------------------
    # OPERATIONS DATA TABLE
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### Operations records")
    st.caption(
        "Review individual records or export the current filtered data."
    )

    display_columns = [
        "timestamp",
        "process_step",
        "machine",
        "shift",
        "planned_minutes",
        "actual_minutes",
        "delay_minutes",
        "units_processed",
        "defect_count",
        "defect_rate_pct",
        "downtime_minutes",
        "delay_flag",
    ]

    display_df = (
        filtered[display_columns]
        .sort_values("timestamp", ascending=False)
        .copy()
    )

    display_df = display_df.rename(
        columns={
            "timestamp": "Timestamp",
            "process_step": "Process step",
            "machine": "Machine",
            "shift": "Shift",
            "planned_minutes": "Planned min",
            "actual_minutes": "Actual min",
            "delay_minutes": "Delay min",
            "units_processed": "Units",
            "defect_count": "Defects",
            "defect_rate_pct": "Defect rate %",
            "downtime_minutes": "Downtime min",
            "delay_flag": "High-delay flag",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=350,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="DD MMM YYYY, HH:mm",
            ),
            "Planned min": st.column_config.NumberColumn(
                "Planned min",
                format="%.1f",
            ),
            "Actual min": st.column_config.NumberColumn(
                "Actual min",
                format="%.1f",
            ),
            "Delay min": st.column_config.NumberColumn(
                "Delay min",
                format="%.1f",
            ),
            "Defect rate %": st.column_config.NumberColumn(
                "Defect rate %",
                format="%.2f%%",
            ),
            "High-delay flag": st.column_config.NumberColumn(
                "High-delay flag",
                help="1 = high delay; 0 = normal delay",
                format="%d",
            ),
        },
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇  Download filtered records (CSV)",
        data=csv_data,
        file_name="processpulse_filtered_operations.csv",
        mime="text/csv",
        key="download_filtered",
    )
    # ============================================================
# TAB 2: BOTTLENECK ANALYSIS
# ============================================================

with tab2:

    st.markdown("### Bottleneck intelligence")
    st.caption(
        "Identify process steps and machines associated with delays, "
        "long processing times, and operational downtime."
    )

    if filtered.empty:

        st.warning(
            "No records match the selected filters. "
            "Adjust the sidebar filters to continue."
        )

    else:

        # ----------------------------------------------------
        # PROCESS-STEP SUMMARY
        # ----------------------------------------------------

        process_summary = (
            filtered.groupby("process_step", as_index=False)
            .agg(
                records=("timestamp", "count"),
                average_planned=("planned_minutes", "mean"),
                average_actual=("actual_minutes", "mean"),
                average_delay=("delay_minutes", "mean"),
                total_delay=("delay_minutes", "sum"),
                high_delay_records=("delay_flag", "sum"),
                total_downtime=("downtime_minutes", "sum"),
                average_downtime=("downtime_minutes", "mean"),
                total_units=("units_processed", "sum"),
                total_defects=("defect_count", "sum"),
            )
        )

        process_summary["high_delay_rate_pct"] = (
            process_summary["high_delay_records"]
            / process_summary["records"].clip(lower=1)
            * 100
        )

        process_summary["defect_rate_pct"] = (
            process_summary["total_defects"]
            / process_summary["total_units"].clip(lower=1)
            * 100
        )

        process_summary["excess_time_pct"] = (
            (
                process_summary["average_actual"]
                - process_summary["average_planned"]
            )
            / process_summary["average_planned"].clip(lower=0.01)
            * 100
        )

        process_summary = process_summary.sort_values(
            "average_delay",
            ascending=False,
        )

        # ----------------------------------------------------
        # HIGHLIGHT THE LEADING BOTTLENECK
        # ----------------------------------------------------

        worst_step = process_summary.iloc[0]

        metric1, metric2, metric3 = st.columns(3, gap="medium")

        with metric1:
            st.metric(
                "Highest average delay",
                f"{worst_step['average_delay']:.1f} min",
                help=(
                    "The process step with the highest average "
                    "recorded delay in the current selection."
                ),
            )

        with metric2:
            st.metric(
                "Highest delay-record rate",
                f"{process_summary['high_delay_rate_pct'].max():.1f}%",
                help=(
                    "Maximum share of records flagged as high-delay "
                    "within any process step."
                ),
            )

        with metric3:
            st.metric(
                "Total recorded downtime",
                f"{filtered['downtime_minutes'].sum():,.0f} min",
                help="Sum of downtime minutes in the filtered records.",
            )

        st.markdown("---")

        # ----------------------------------------------------
        # CHART 1: PROCESS-STEP DELAY RANKING
        # ----------------------------------------------------

        left_chart, right_chart = st.columns(2, gap="large")

        with left_chart:

            st.markdown("#### Process-step delay ranking")

            fig = px.bar(
                process_summary.sort_values(
                    "average_delay",
                    ascending=True,
                ),
                x="average_delay",
                y="process_step",
                orientation="h",
                color="average_delay",
                color_continuous_scale=[
                    "#35d7ef",
                    "#3988ff",
                    "#a277ff",
                ],
                labels={
                    "average_delay": "Average delay (minutes)",
                    "process_step": "Process step",
                },
                hover_data={
                    "records": True,
                    "high_delay_rate_pct": ":.1f",
                    "average_actual": ":.2f",
                },
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#b9c9df"),
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_showscale=False,
            )

            fig.update_traces(
                marker_line_width=0,
                hovertemplate=(
                    "Process: %{y}<br>"
                    "Average delay: %{x:.2f} min"
                    "<extra></extra>"
                ),
            )

            fig.update_xaxes(
                gridcolor="rgba(148,163,184,.10)"
            )
            fig.update_yaxes(showgrid=False)

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="bottleneck_delay_ranking",
            )

        # ----------------------------------------------------
        # CHART 2: DOWNTIME VS DELAY
        # ----------------------------------------------------

        with right_chart:

            st.markdown("#### Machine downtime vs delay")

            machine_summary = (
                filtered.groupby("machine", as_index=False)
                .agg(
                    average_downtime=("downtime_minutes", "mean"),
                    average_delay=("delay_minutes", "mean"),
                    records=("timestamp", "count"),
                    high_delay_records=("delay_flag", "sum"),
                )
            )

            machine_summary["high_delay_rate_pct"] = (
                machine_summary["high_delay_records"]
                / machine_summary["records"].clip(lower=1)
                * 100
            )

            fig = px.scatter(
                machine_summary,
                x="average_downtime",
                y="average_delay",
                size="records",
                color="high_delay_rate_pct",
                hover_name="machine",
                color_continuous_scale=[
                    "#35d7ef",
                    "#3988ff",
                    "#a277ff",
                ],
                labels={
                    "average_downtime": "Average downtime (min)",
                    "average_delay": "Average delay (min)",
                    "high_delay_rate_pct": "High-delay rate (%)",
                    "records": "Records",
                },
            )

            fig.update_traces(
                marker=dict(
                    opacity=0.85,
                    line=dict(
                        color="rgba(255,255,255,0.35)",
                        width=1,
                    ),
                ),
                hovertemplate=(
                    "Machine: %{hovertext}<br>"
                    "Average downtime: %{x:.2f} min<br>"
                    "Average delay: %{y:.2f} min"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#b9c9df"),
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_colorbar=dict(
                    title="High-delay<br>rate (%)"
                ),
            )

            fig.update_xaxes(
                gridcolor="rgba(148,163,184,.10)"
            )
            fig.update_yaxes(
                gridcolor="rgba(148,163,184,.10)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="machine_downtime_delay",
            )

        # ----------------------------------------------------
        # PROCESS-STEP DETAIL TABLE
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("#### Process-step performance")

        bottleneck_table = process_summary[
            [
                "process_step",
                "records",
                "average_planned",
                "average_actual",
                "average_delay",
                "excess_time_pct",
                "high_delay_rate_pct",
                "total_downtime",
                "defect_rate_pct",
            ]
        ].rename(
            columns={
                "process_step": "Process step",
                "records": "Records",
                "average_planned": "Avg planned (min)",
                "average_actual": "Avg actual (min)",
                "average_delay": "Avg delay (min)",
                "excess_time_pct": "Excess time (%)",
                "high_delay_rate_pct": "High-delay rate (%)",
                "total_downtime": "Total downtime (min)",
                "defect_rate_pct": "Defect rate (%)",
            }
        )

        st.dataframe(
            bottleneck_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg planned (min)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Avg actual (min)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Avg delay (min)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Excess time (%)": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "High-delay rate (%)": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Total downtime (min)": st.column_config.NumberColumn(
                    format="%.1f"
                ),
                "Defect rate (%)": st.column_config.NumberColumn(
                    format="%.3f%%"
                ),
            },
        )

        # ----------------------------------------------------
        # AUTOMATED INTERPRETATION
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("#### Operational interpretation")

        top_delay_step = process_summary.iloc[0]
        top_downtime_step = process_summary.loc[
            process_summary["total_downtime"].idxmax()
        ]
        top_high_delay_step = process_summary.loc[
            process_summary["high_delay_rate_pct"].idxmax()
        ]

        insight1, insight2 = st.columns(2, gap="medium")

        with insight1:

            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">DELAY SIGNAL</div>
                    <div class="insight-title">
                        {top_delay_step['process_step']}
                    </div>
                    <div class="insight-body">
                        This process step has the highest average delay
                        in the selected records:
                        <b>{top_delay_step['average_delay']:.2f} minutes</b>.
                        Investigate its workflow, resource availability,
                        and processing requirements.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with insight2:

            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">DOWNTIME SIGNAL</div>
                    <div class="insight-title">
                        {top_downtime_step['process_step']}
                    </div>
                    <div class="insight-body">
                        This process step has the largest total recorded
                        downtime:
                        <b>{top_downtime_step['total_downtime']:.1f} minutes</b>.
                        Review maintenance logs and downtime causes
                        before choosing a corrective action.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.info(
            f"**Additional signal:** "
            f"{top_high_delay_step['process_step']} has the highest "
            f"high-delay-record rate at "
            f"{top_high_delay_step['high_delay_rate_pct']:.1f}%. "
            "These are descriptive signals from the available records, "
            "not proof that a particular factor caused the delays."
        )
        st.caption(
            "Interpretation is calculated from the selected dataset. "
            "The analysis identifies associations and priorities for "
            "investigation; it does not establish causation."
        )
        # ============================================================
# TAB 3: AI DELAY-RISK PREDICTION
# ============================================================
with tab3:
    st.markdown("### AI-powered delay-risk intelligence")
    st.caption(
        "Train a machine-learning model to identify operational "
        "records that may exceed their planned processing time."
    )
    # --------------------------------------------------------
    # PREPARE DATA FOR MACHINE LEARNING
    # --------------------------------------------------------
    model_features = [
        "process_step",
        "machine",
        "shift",
        "planned_minutes",
        "units_processed",
        "downtime_minutes",
    ]
    categorical_features = [
        "process_step",
        "machine",
        "shift",
    ]
    numerical_features = [
        "planned_minutes",
        "units_processed",
        "downtime_minutes",
    ]
    target = "delay_flag"
    model_data = df[model_features + [target]].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[target])
    X = model_data[model_features]
    y = model_data[target].astype(int)
    if len(model_data) < 20 or y.nunique() < 2:
        st.warning(
            "There is not enough data or target-class variation "
            "to train a reliable classification model. "
            "Use a larger dataset containing both delayed and "
            "non-delayed records."
        )
    else:
        # ----------------------------------------------------
        # BUILD THE MACHINE-LEARNING PIPELINE
        # ----------------------------------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
                (
                    "numerical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="median"),
                            ),
                        ]
                    ),
                    numerical_features,
                ),
            ]
        )
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------
        class_counts = y.value_counts()
        can_stratify = (
            y.nunique() == 2
            and class_counts.min() >= 2
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if can_stratify else None,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        classes = list(model.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
        risk_probabilities = probabilities[:, positive_index]
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("#### Model performance")
        score1, score2, score3, score4 = st.columns(4, gap="medium")
        with score1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%",
                help="Fraction of test records classified correctly.",
            )
        with score2:
            st.metric(
                "Precision",
                f"{precision * 100:.1f}%",
                help=(
                    "Of the records predicted as high-delay, "
                    "the fraction that were actually high-delay."
                ),
            )
        with score3:
            st.metric(
                "Recall",
                f"{recall * 100:.1f}%",
                help=(
                    "Of the actual high-delay records, "
                    "the fraction detected by the model."
                ),
            )
        with score4:
            st.metric(
                "F1 score",
                f"{f1 * 100:.1f}%",
                help="Harmonic mean of precision and recall.",
            )
        st.caption(
            f"Training records: {len(X_train):,} | "
            f"Test records: {len(X_test):,} | "
            f"Random state: 42"
        )
        st.info(
            "The target is labelled high-delay when actual processing "
            "time exceeds planned time by more than 15%. "
            "These scores measure performance on a held-out test split; "
            "they do not guarantee future operational performance."
        )
            # ============================================================
    # DATASET AND MODEL LIMITATIONS
    # ============================================================

    st.markdown("#### Dataset and model limitations")

    with st.expander("Understand the data and how to interpret the model"):
        st.markdown(
            """
            **Dataset source**

            This project uses a synthetic operations dataset created for
            demonstration and portfolio purposes. It represents process steps,
            machines, shifts, planned processing times, units processed,
            downtime, and operational delays. It is not real company data.

            **Target definition**

            A record is labelled high-delay when its actual processing time
            exceeds its planned processing time by more than 15%.

            **Model approach**

            A Random Forest classifier uses process, machine, shift, planned
            time, production volume, and downtime features. Performance is
            evaluated on a held-out test split.

            **Important limitations**

            - Synthetic patterns may not reflect real industrial operations.
            - Test-set scores do not guarantee performance on future data.
            - Some features may be unavailable when an early warning is needed.
            - The model identifies patterns; it does not establish causation.
            - Real deployment would require representative historical data,
            leakage checks, external validation, and ongoing monitoring.
            """
        )
        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("#### What influences the model's predictions?")
        fitted_preprocessor = model.named_steps["preprocessor"]
        fitted_classifier = model.named_steps["classifier"]
        try:
            transformed_feature_names = (
                fitted_preprocessor.get_feature_names_out()
            )
            importances = fitted_classifier.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "feature": transformed_feature_names,
                    "importance": importances,
                }
            )
            # Combine one-hot encoded categories back into
            # understandable original feature groups.
            def simplify_feature_name(name):
                if name.startswith("categorical__process_step"):
                    return "Process step"
                if name.startswith("categorical__machine"):
                    return "Machine"
                if name.startswith("categorical__shift"):
                    return "Shift"
                if name.startswith("numerical__planned_minutes"):
                    return "Planned minutes"
                if name.startswith("numerical__units_processed"):
                    return "Units processed"
                if name.startswith("numerical__downtime_minutes"):
                    return "Downtime minutes"
                return name
            importance_df["feature_group"] = (
                importance_df["feature"].apply(simplify_feature_name)
            )
            grouped_importance = (
                importance_df.groupby("feature_group", as_index=False)
                ["importance"]
                .sum()
                .sort_values("importance", ascending=True)
            )
            fig = px.bar(
                grouped_importance,
                x="importance",
                y="feature_group",
                orientation="h",
                color="importance",
                color_continuous_scale=[
                    "#35d7ef",
                    "#3988ff",
                    "#a277ff",
                ],
                labels={
                    "importance": "Relative importance",
                    "feature_group": "Feature",
                },
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#b9c9df"),
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_showscale=False,
            )
            fig.update_xaxes(
                gridcolor="rgba(148,163,184,.10)"
            )
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="model_feature_importance",
            )
            st.caption(
                "Feature importance describes how the trained model "
                "uses these inputs. It does not establish causation."
            )
        except (ValueError, AttributeError) as error:
            st.warning(
                "Feature importance could not be displayed for this "
                f"dataset: {error}"
            )
        # ----------------------------------------------------
        # INTERACTIVE SCENARIO PREDICTOR
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("#### Test an operational scenario")
        st.write(
            "Choose the details of a hypothetical operation to estimate "
            "its probability of exceeding the high-delay threshold."
        )
        process_options = sorted(
            df["process_step"].dropna().astype(str).unique().tolist()
        )
        machine_options = sorted(
            df["machine"].dropna().astype(str).unique().tolist()
        )
        shift_options = sorted(
            df["shift"].dropna().astype(str).unique().tolist()
        )

        if process_options and machine_options and shift_options:

            input_col1, input_col2, input_col3 = st.columns(3)

            with input_col1:

                selected_process = st.selectbox(
                    "Process step",
                    process_options,
                    key="risk_process",
                )

                planned_input = st.number_input(
                    "Planned processing time (min)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=float(
                        max(1, df["planned_minutes"].median())
                    ),
                    step=1.0,
                    key="risk_planned",
                )

            with input_col2:

                selected_machine = st.selectbox(
                    "Machine",
                    machine_options,
                    key="risk_machine",
                )

                units_input = st.number_input(
                    "Units to process",
                    min_value=1,
                    max_value=100000,
                    value=int(
                        max(1, df["units_processed"].median())
                    ),
                    step=1,
                    key="risk_units",
                )

            with input_col3:

                selected_shift = st.selectbox(
                    "Shift",
                    shift_options,
                    key="risk_shift",
                )

                downtime_input = st.number_input(
                    "Expected downtime (min)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(
                        max(0, df["downtime_minutes"].median())
                    ),
                    step=1.0,
                    key="risk_downtime",
                )

            scenario = pd.DataFrame(
                [
                    {
                        "process_step": selected_process,
                        "machine": selected_machine,
                        "shift": selected_shift,
                        "planned_minutes": planned_input,
                        "units_processed": units_input,
                        "downtime_minutes": downtime_input,
                    }
                ]
            )

            scenario_probability = model.predict_proba(scenario)[
                0, positive_index
            ]

            scenario_prediction = model.predict(scenario)[0]

            st.markdown("##### Estimated delay risk")

            risk_col1, risk_col2 = st.columns(
                [1, 2],
                gap="large",
            )

            with risk_col1:

                st.metric(
                    "Estimated probability",
                    f"{scenario_probability * 100:.1f}%",
                )

                if scenario_prediction == 1:
                    st.warning(
                        "The model classifies this scenario as "
                        "high-delay risk."
                    )
                else:
                    st.success(
                        "The model classifies this scenario as "
                        "lower-delay risk."
                    )

            with risk_col2:

                st.progress(
                    float(
                        np.clip(scenario_probability, 0.0, 1.0)
                    )
                )

                st.write(
                    "This probability is the model's estimate for "
                    "the selected inputs. It is not a guarantee, "
                    "and it should not be interpreted as a calibrated "
                    "real-world probability without further validation."
                )

                st.caption(
                    "The model was trained on the complete dataset, "
                    "while the sidebar filters affect the dashboard "
                    "views. Scenario predictions therefore use the "
                    "same trained model regardless of those filters."
                )

        else:

            st.warning(
                "The dataset needs at least one non-empty process step, "
                "machine, and shift value to enable scenario prediction."
            )

        # ----------------------------------------------------
        # TEST-SET PREDICTIONS TABLE
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("#### Held-out test-set predictions")

        prediction_table = X_test.copy().reset_index(drop=True)

        prediction_table["Actual label"] = (
            y_test.reset_index(drop=True).map(
                {0: "Lower delay", 1: "High delay"}
            )
        )

        prediction_table["Predicted label"] = pd.Series(
            predictions
        ).map(
            {0: "Lower delay", 1: "High delay"}
        )

        prediction_table["Estimated risk (%)"] = (
            risk_probabilities * 100
        ).round(2)

        prediction_table["Correct prediction"] = (
            predictions == y_test.to_numpy()
        )

        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True,
        )

        prediction_csv = prediction_table.to_csv(index=False).encode(
            "utf-8"
        )

        st.download_button(
            label="Download test-set predictions",
            data=prediction_csv,
            file_name="processpulse_test_predictions.csv",
            mime="text/csv",
            key="download_test_predictions",
        )
        # ============================================================
# TAB 4: DATA EXPLORER & METHODOLOGY
# ============================================================

with tab4:

    st.markdown("### Data explorer")
    st.caption(
        "Inspect the operational records behind the dashboard, "
        "review data quality, and export the filtered dataset."
    )

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("#### Dataset overview")

    data_col1, data_col2, data_col3, data_col4 = st.columns(4)

    with data_col1:
        st.metric("Total records", f"{len(filtered):,}")

    with data_col2:
        st.metric("Columns", f"{filtered.shape[1]:,}")

    with data_col3:
        missing_count = int(filtered.isna().sum().sum())
        st.metric("Missing values", f"{missing_count:,}")

    with data_col4:
        duplicate_count = int(filtered.duplicated().sum())
        st.metric("Duplicate rows", f"{duplicate_count:,}")

    # --------------------------------------------------------
    # FILTERED DATA TABLE
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("#### Explore operational records")

    search_col, sort_col = st.columns([2, 1])

    with search_col:
        search_text = st.text_input(
            "Search records",
            placeholder="Search machine, process step, shift...",
            key="data_search",
        )

    with sort_col:
        sort_options = [
            "timestamp",
            "delay_minutes",
            "actual_minutes",
            "planned_minutes",
            "downtime_minutes",
            "units_processed",
        ]

        available_sort_options = [
            column
            for column in sort_options
            if column in filtered.columns
        ]

        sort_column = st.selectbox(
            "Sort records by",
            available_sort_options,
            index=(
                available_sort_options.index("timestamp")
                if "timestamp" in available_sort_options
                else 0
            ),
            key="data_sort_column",
        )

    display_data = filtered.copy()

    if search_text.strip():

        searchable_columns = [
            column
            for column in [
                "process_step",
                "machine",
                "shift",
            ]
            if column in display_data.columns
        ]

        if searchable_columns:

            search_mask = pd.Series(
                False,
                index=display_data.index,
            )

            for column in searchable_columns:

                search_mask |= (
                    display_data[column]
                    .astype(str)
                    .str.contains(
                        search_text.strip(),
                        case=False,
                        na=False,
                        regex=False,
                    )
                )

            display_data = display_data.loc[search_mask]

    display_data = display_data.sort_values(
        sort_column,
        ascending=False,
    )

    st.caption(
        f"Showing {len(display_data):,} matching records "
        f"out of {len(filtered):,} filtered records."
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # EXPORT DATA
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("#### Export data")

    export_col1, export_col2 = st.columns(2, gap="medium")

    with export_col1:

        filtered_csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download filtered dataset",
            data=filtered_csv,
            file_name="processpulse_filtered_data.csv",
            mime="text/csv",
            key="download_filtered_dataset",
            use_container_width=True,
        )

    with export_col2:

        full_csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download complete dataset",
            data=full_csv,
            file_name="processpulse_complete_data.csv",
            mime="text/csv",
            key="download_complete_dataset",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("#### Data quality checks")

    quality_data = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Missing values": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Missing (%)": [
                round(
                    df[column].isna().mean() * 100,
                    2,
                )
                for column in df.columns
            ],
            "Unique values": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    st.dataframe(
        quality_data,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("#### Methodology")

    method_col1, method_col2 = st.columns(2, gap="large")

    with method_col1:

        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">01 · DATA PROCESSING</div>
                <div class="insight-title">Operational data pipeline</div>
                <div class="insight-body">
                    The dashboard loads operational records, derives
                    delay-related measures, calculates defect rates,
                    and applies interactive filters to support
                    descriptive analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">02 · BOTTLENECK ANALYSIS</div>
                <div class="insight-title">Process performance</div>
                <div class="insight-body">
                    Records are aggregated by process step and machine
                    to compare average delay, downtime, and the share
                    of records exceeding the delay threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with method_col2:

        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">03 · MACHINE LEARNING</div>
                <div class="insight-title">Random Forest classifier</div>
                <div class="insight-body">
                    A preprocessing pipeline encodes categorical
                    variables and imputes missing feature values.
                    A Random Forest classifier predicts whether a
                    record crosses the high-delay threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-label">04 · MODEL EVALUATION</div>
                <div class="insight-title">Held-out test set</div>
                <div class="insight-body">
                    Accuracy, precision, recall, and F1 score
                    summarize classification performance on the
                    test split. Further validation is necessary
                    before using predictions in live operations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # IMPORTANT PROJECT DISCLAIMER
    # --------------------------------------------------------

    st.markdown("---")

    st.warning(
        "**Dataset disclaimer:** This project uses the operational "
        "dataset included with the project. If the data is synthetic, "
        "the resulting patterns and model metrics demonstrate the "
        "analytics workflow, not verified performance in a real factory."
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="pp-footer">
        <b style="color:#b7c8e4">ProcessPulse AI</b>
        · Operations analytics and delay-risk demonstration
        <br>
        Built with Python, Streamlit, Pandas, Plotly, and
        scikit-learn. Synthetic data only; not for production use.
    </div>
    """,
    unsafe_allow_html=True,
)
