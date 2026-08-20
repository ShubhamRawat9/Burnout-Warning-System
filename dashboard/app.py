import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import os

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="Burnout Early Warning Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F8FD; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2E75B6;
        margin-bottom: 12px;
    }
    .high-risk  { border-left: 5px solid #C00000 !important; background: #FFF5F5 !important; }
    .medium-risk{ border-left: 5px solid #ED7D31 !important; background: #FFF9F0 !important; }
    .low-risk   { border-left: 5px solid #70AD47 !important; background: #F0FFF4 !important; }
    .alert-box {
        background: #FFF3CD; border: 1px solid #F0A500;
        border-radius: 8px; padding: 16px; margin: 10px 0;
    }
    .info-box {
        background: #EBF5FB; border: 1px solid #2E75B6;
        border-radius: 8px; padding: 12px; margin: 8px 0;
        font-size: 13px; color: #1F3864;
    }
    .drift-legend {
        display: flex; gap: 16px; padding: 10px 0;
        font-size: 13px; font-weight: bold;
    }
    .stMetric label { font-size: 13px !important; }
    h1 { color: #1F3864 !important; }
    h2 { color: #2E75B6 !important; }
    h3 { color: #1F3864 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data and Models ──────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base, "data", "processed", "burnout_with_drift.csv"))
    return df

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mp = os.path.join(base, "models")
    xgb          = joblib.load(os.path.join(mp, "xgboost_model.pkl"))
    le           = joblib.load(os.path.join(mp, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(mp, "feature_cols.pkl"))
    explainer    = joblib.load(os.path.join(mp, "shap_explainer.pkl"))
    return xgb, le, feature_cols, explainer

df = load_data()
xgb, le, feature_cols, explainer = load_models()

# ── RGBA fill colors (no 8-char hex) ─────────────
RGBA_FILLS = {
    "#C00000": "rgba(192,0,0,0.1)",
    "#ED7D31": "rgba(237,125,49,0.1)",
    "#70AD47": "rgba(112,173,71,0.1)",
    "#2E75B6": "rgba(46,117,182,0.1)",
    "#7030A0": "rgba(112,48,160,0.1)",
    "#1F3864": "rgba(31,56,100,0.1)",
}

# ── Helper functions ──────────────────────────────
def get_risk_color(label):
    return {"High": "#C00000", "Medium": "#ED7D31", "Low": "#70AD47"}.get(label, "#2E75B6")

def get_risk_emoji(label):
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(label, "⚪")

def get_drift_tag(score):
    """Return colored drift label based on score."""
    if score >= 1.5:
        return f"🔴 Critical ({score:.3f})"
    elif score >= 1.0:
        return f"🟡 Warning ({score:.3f})"
    else:
        return f"🟢 Normal ({score:.3f})"

def get_alert_reason(shap_row, feature_names):
    """
    Returns top-3 SHAP-driven reasons with corrected plain-language direction.
    Note: SHAP values represent deviation from personal baseline, NOT raw change.
    A positive SHAP value means this feature's deviation is PUSHING toward the
    predicted class — it does NOT always mean the raw value increased.
    """
    top3 = np.argsort(np.abs(shap_row))[::-1][:3]
    reasons = []
    for idx in top3:
        feat  = feature_names[idx]
        val   = shap_row[idx]
        clean = feat.replace("_drift", " (drift)").replace("_", " ").title()

        # Risk-aware direction: for burnout signals, interpret contribution meaning
        burnout_increase = ["context_switches", "after_hours_logins", "idle_time_mins",
                            "submission_delays", "revision_requests", "rework_count",
                            "error_rate", "mental_fatigue_score", "weekend_work_days"]
        burnout_decrease = ["completion_rate", "focus_sessions", "avg_focus_duration",
                            "breaks_taken", "work_hours_per_day"]

        raw_feat = feat.replace("_drift", "")
        if raw_feat in burnout_increase:
            direction = "↑ elevated (burnout signal)" if val > 0 else "↓ reducing (recovery signal)"
        elif raw_feat in burnout_decrease:
            direction = "↓ declining (burnout signal)" if val > 0 else "↑ recovering"
        else:
            direction = "↑ contributing to risk" if val > 0 else "↓ reducing risk"

        reasons.append(f"{clean} — {direction}")
    return reasons

def get_personalized_action(reasons, label):
    """Generate action recommendation based on actual SHAP signals."""
    signal_keywords = " ".join(reasons).lower()
    actions = []

    if "context switches" in signal_keywords:
        actions.append("Reduce parallel task assignments — focus on 1–2 priorities")
    if "after hours" in signal_keywords:
        actions.append("Enforce strict end-of-day boundaries — no after-hours messages")
    if "completion rate" in signal_keywords:
        actions.append("Review task complexity — reduce scope by 20–30% this week")
    if "submission delays" in signal_keywords:
        actions.append("Check for blockers — may need resource or deadline support")
    if "fatigue" in signal_keywords:
        actions.append("Schedule mandatory breaks — fatigue score is critically elevated")
    if "focus" in signal_keywords:
        actions.append("Block deep-work time — reduce meeting load by 50%")

    if not actions:
        actions = ["Review overall workload and schedule a check-in conversation"]

    if label == "High":
        prefix = "🔴 Escalate to HR immediately:"
    else:
        prefix = "🟡 Manager action recommended:"

    return prefix + "<br>" + "<br>".join([f"&nbsp;&nbsp;• {a}" for a in actions[:3]])

def predict_employee(emp_data):
    X      = emp_data[feature_cols].values
    proba  = xgb.predict_proba(X)
    preds  = xgb.predict(X)
    labels = le.inverse_transform(preds)
    return labels, proba

# ── Pre-compute all predictions ───────────────────
@st.cache_data
def get_all_predictions():
    results = []
    for emp_id in df["employee_id"].unique():
        emp_data = df[df["employee_id"] == emp_id].sort_values("week_number")
        labels, proba = predict_employee(emp_data)
        for i, (_, row) in enumerate(emp_data.iterrows()):
            results.append({
                "employee_id"    : emp_id,
                "week_number"    : row["week_number"],
                "actual_label"   : row["burnout_label"],
                "predicted_label": labels[i],
                "prob_high"      : proba[i][0],
                "prob_low"       : proba[i][1],
                "prob_medium"    : proba[i][2],
                "overall_drift"  : row["overall_drift_score"],
            })
    return pd.DataFrame(results)

pred_df = get_all_predictions()

# ── Sidebar ───────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/null/fire-element--v2.png", width=60)
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Overview Dashboard",
     "👤 Employee Deep Dive",
     "🚨 Alert Log",
     "📈 Drift Analysis",
     "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Burnout Early Warning Platform")
st.sidebar.markdown("**Team:** Shubham | Shital | Barkha")
st.sidebar.markdown("**Institute:** UPES Dehradun")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Drift Score Scale:**
🟢 Normal: < 1.0
🟡 Warning: 1.0 – 1.5
🔴 Critical: > 1.5
""")

# ════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# FIX: Added week selector, fixed trend chart,
#      fixed KPI to reflect selected week,
#      added drift score legend
# ════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Productivity Drift & Burnout Early Warning Platform")
    st.markdown("**Behavioral monitoring dashboard — UPES Dehradun MCA Capstone 2026**")
    st.markdown("---")

    # ── FIX 1: Week selector ──────────────────────
    selected_week = st.slider(
        "📅 Select Week to View (1 = Start of monitoring, 16 = End)",
        min_value=1, max_value=16, value=8,
        help="Use this slider to see how risk distribution changes across monitoring weeks"
    )

    week_data = pred_df[pred_df["week_number"] == selected_week]
    total      = len(week_data)
    high_count = (week_data["predicted_label"] == "High").sum()
    med_count  = (week_data["predicted_label"] == "Medium").sum()
    low_count  = (week_data["predicted_label"] == "Low").sum()
    avg_drift  = week_data["overall_drift"].mean()

    # Phase label
    if selected_week <= 6:
        phase_label = "🟢 Normal Phase (Weeks 1–6) — Baseline period"
    elif selected_week <= 11:
        phase_label = "🟡 Early Drift Phase (Weeks 7–11) — Burnout signals emerging"
    else:
        phase_label = "🔴 High Risk Phase (Weeks 12–16) — Intervention required"

    st.info(f"**Week {selected_week}** — {phase_label}")

    # ── KPI metrics ───────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Employees", total)
    col2.metric("🔴 High Risk",   high_count,
                delta=f"{high_count/total*100:.0f}%", delta_color="inverse")
    col3.metric("🟡 Medium Risk", med_count,
                delta=f"{med_count/total*100:.0f}%", delta_color="off")
    col4.metric("🟢 Low Risk",    low_count,
                delta=f"{low_count/total*100:.0f}%")
    col5.metric("📉 Avg Drift Score", f"{avg_drift:.3f}",
                delta="🔴 Critical" if avg_drift >= 1.5 else "🟡 Warning" if avg_drift >= 1.0 else "🟢 Normal",
                delta_color="off")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    # ── Pie chart — week-specific ─────────────────
    with col_left:
        st.subheader(f"Risk Distribution — Week {selected_week}")
        fig_pie = go.Figure(go.Pie(
            labels=["High Risk", "Medium Risk", "Low Risk"],
            values=[high_count, med_count, low_count],
            marker_colors=["#C00000", "#ED7D31", "#70AD47"],
            hole=0.4,
            textinfo="label+percent",
            textfont_size=13
        ))
        fig_pie.update_layout(
            height=350, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False, paper_bgcolor="white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(
            "Note: All employees show High Risk at Week 16 because the synthetic "
            "dataset's final phase is defined as High Risk. Use the week slider above "
            "to see the risk progression across all 16 weeks."
        )

    # ── FIX 2: Risk Trend — proper week-by-week ───
    with col_right:
        st.subheader("Risk Trend — All 16 Weeks (All Employees)")
        weekly = pred_df.groupby(
            ["week_number", "predicted_label"]
        ).size().reset_index(name="count")

        # Ensure all week+label combinations exist (fill missing with 0)
        all_weeks  = pd.DataFrame({"week_number": range(1, 17)})
        all_labels = pd.DataFrame({"predicted_label": ["High", "Medium", "Low"]})
        full_grid  = all_weeks.merge(all_labels, how="cross")
        weekly     = full_grid.merge(weekly, on=["week_number", "predicted_label"], how="left").fillna(0)

        fig_line = go.Figure()
        for label, color in [("Low", "#70AD47"), ("Medium", "#ED7D31"), ("High", "#C00000")]:
            d = weekly[weekly["predicted_label"] == label].sort_values("week_number")
            fig_line.add_trace(go.Scatter(
                x=d["week_number"], y=d["count"],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor=RGBA_FILLS[color]
            ))

        # Phase boundary lines
        fig_line.add_vline(x=6.5,  line_dash="dash", line_color="#ED7D31",
                           opacity=0.7, annotation_text="Drift starts (W7)")
        fig_line.add_vline(x=11.5, line_dash="dash", line_color="#C00000",
                           opacity=0.7, annotation_text="High Risk onset (W12)")

        # Selected week marker
        fig_line.add_vline(x=selected_week, line_dash="solid",
                           line_color="#1F3864", opacity=0.9,
                           annotation_text=f"Selected: W{selected_week}")

        fig_line.update_layout(
            height=350,
            xaxis=dict(title="Week Number", tickmode="linear", tick0=1, dtick=1),
            yaxis_title="Number of Employees",
            paper_bgcolor="white", plot_bgcolor="#F8FAFD",
            margin=dict(t=20, b=40, l=40, r=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption(
            "Low Risk employees dominate Weeks 1–6. Medium Risk peaks at Week 7 "
            "(phase transition spike). High Risk rises from Week 12. "
            "The dark vertical line marks your currently selected week."
        )

    # ── At-risk table for selected week ───────────
    st.markdown("---")
    st.subheader(f"🚨 Employees at Risk — Week {selected_week}")

    at_risk = week_data[
        week_data["predicted_label"].isin(["High", "Medium"])
    ].sort_values("prob_high", ascending=False).copy()

    if len(at_risk) > 0:
        at_risk["Drift Level"] = at_risk["overall_drift"].apply(get_drift_tag)
        display_df = at_risk[[
            "employee_id", "predicted_label",
            "prob_high", "prob_medium", "Drift Level"
        ]].copy()
        display_df.columns = ["Employee ID", "Risk Level", "P(High)", "P(Medium)", "Drift Level"]
        display_df["P(High)"]   = display_df["P(High)"].round(3)
        display_df["P(Medium)"] = display_df["P(Medium)"].round(3)
        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
        st.caption("Drift Level scale: 🟢 Normal < 1.0 | 🟡 Warning 1.0–1.5 | 🔴 Critical > 1.5")
    else:
        st.success(f"✅ No employees at High or Medium risk during Week {selected_week}.")

# ════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE DEEP DIVE
# FIX: Fixed SHAP direction explanation,
#      added normalized values note,
#      personalized recommendations
# ════════════════════════════════════════════════
elif page == "👤 Employee Deep Dive":
    st.title("👤 Employee Deep Dive")
    st.markdown("---")

    emp_list = sorted(df["employee_id"].unique())
    selected = st.selectbox("Select Employee", emp_list, index=0)

    emp_data     = df[df["employee_id"] == selected].sort_values("week_number")
    labels, proba = predict_employee(emp_data)
    latest_label  = labels[-1]
    latest_proba  = proba[-1]
    latest_drift  = emp_data["overall_drift_score"].iloc[-1]
    risk_color    = get_risk_color(latest_label)
    risk_emoji    = get_risk_emoji(latest_label)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👤 Employee", selected)
    col2.metric(f"{risk_emoji} Current Risk", latest_label)
    col3.metric("📉 Drift Score", get_drift_tag(latest_drift))
    col4.metric("🎯 Confidence", f"{max(latest_proba)*100:.1f}%")

    st.markdown("---")
    col_g, col_t = st.columns([1, 1])

    with col_g:
        st.subheader("Risk Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_proba[0] * 100,
            title={"text": "High Risk Probability (%)"},
            delta={"reference": 30},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0,  30],  "color": "#E8F5E9"},
                    {"range": [30, 65],  "color": "#FFF9C4"},
                    {"range": [65, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "#C00000", "width": 4},
                    "thickness": 0.75, "value": 65
                }
            }
        ))
        fig_gauge.update_layout(
            height=300, margin=dict(t=40, b=20, l=40, r=40),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("Alert triggers when P(High) > 65% for 2 consecutive weeks")

    with col_t:
        st.subheader("16-Week Risk Trajectory")
        prob_high = [p[0] for p in proba]
        prob_med  = [p[2] for p in proba]
        weeks     = list(emp_data["week_number"])

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=weeks, y=prob_high,
            name="P(High Risk)",
            mode="lines+markers",
            line=dict(color="#C00000", width=2.5),
            marker=dict(size=5)
        ))
        fig_traj.add_trace(go.Scatter(
            x=weeks, y=prob_med,
            name="P(Medium Risk)",
            mode="lines+markers",
            line=dict(color="#ED7D31", width=2.5),
            marker=dict(size=5)
        ))
        fig_traj.add_hline(
            y=0.65, line_dash="dash", line_color="#C00000", opacity=0.5,
            annotation_text="Alert Threshold (65%)"
        )
        fig_traj.add_vline(x=6.5,  line_dash="dot", line_color="#ED7D31",
                           opacity=0.5, annotation_text="Drift starts")
        fig_traj.add_vline(x=11.5, line_dash="dot", line_color="#C00000",
                           opacity=0.5, annotation_text="High Risk onset")
        fig_traj.update_layout(
            height=300,
            xaxis=dict(title="Week Number", tickmode="linear", tick0=1, dtick=1, range=[1, 16]),
            yaxis_title="Risk Probability",
            paper_bgcolor="white", plot_bgcolor="#F8FAFD",
            margin=dict(t=20, b=40, l=40, r=20),
            legend=dict(orientation="h", y=-0.35)
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        st.caption("The trajectory shows how risk probability evolved week by week for this employee")

    # ── Behavioral trend charts ───────────────────
    st.markdown("---")
    st.subheader("📈 Behavioral Signal Trends — 16 Weeks")

    # FIX: Add normalized values explanation
    st.markdown("""
    <div class="info-box">
    ℹ️ <b>Note on Y-axis values:</b> All feature values are normalized to a [0, 1] scale
    for model input using MinMaxScaler. A value of 1.0 represents the maximum observed
    across all employees. The <i>direction</i> and <i>trend</i> of each signal matters
    more than the absolute value.
    </div>
    """, unsafe_allow_html=True)

    key_features   = ["completion_rate", "focus_sessions", "after_hours_logins",
                      "mental_fatigue_score", "context_switches", "submission_delays"]
    feature_labels = ["Completion Rate (↓ = burnout signal)",
                      "Focus Sessions (↓ = burnout signal)",
                      "After Hours Logins (↑ = burnout signal)",
                      "Mental Fatigue Score (↑ = burnout signal)",
                      "Context Switches (↑ = burnout signal)",
                      "Submission Delays (↑ = burnout signal)"]
    feat_colors    = ["#2E75B6", "#70AD47", "#C00000", "#ED7D31", "#7030A0", "#1F3864"]

    col1, col2 = st.columns(2)
    for i, (feat, label, color) in enumerate(zip(key_features, feature_labels, feat_colors)):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=emp_data["week_number"],
            y=emp_data[feat],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=RGBA_FILLS.get(color, "rgba(100,100,100,0.08)")
        ))
        fig.add_vline(x=6.5,  line_dash="dash", line_color="#ED7D31",
                      opacity=0.5, annotation_text="Drift")
        fig.add_vline(x=11.5, line_dash="dash", line_color="#C00000",
                      opacity=0.5, annotation_text="High Risk")
        fig.update_layout(
            title=dict(text=label, font=dict(size=12)),
            height=220,
            margin=dict(t=40, b=30, l=40, r=20),
            paper_bgcolor="white", plot_bgcolor="#F8FAFD",
            showlegend=False,
            xaxis=dict(title="Week", tickmode="linear", tick0=1, dtick=2),
            yaxis=dict(title="Value (0–1 normalized)", range=[0, 1.1])
        )
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)

    # ── SHAP Alert Explanation ────────────────────
    st.markdown("---")
    st.subheader("🔍 SHAP Alert Explanation — Week 16 (Latest)")

    # FIX: Add SHAP explanation note
    st.markdown("""
    <div class="info-box">
    ℹ️ <b>How to read SHAP explanations:</b> SHAP values measure each feature's
    <i>deviation from this employee's personal 4-week baseline</i> — not the raw value.
    A feature "contributing to risk" means it has drifted significantly from this
    employee's own normal pattern, pushing the model toward a higher risk prediction.
    This is why two employees with the same absolute values can have different risk scores.
    </div>
    """, unsafe_allow_html=True)

    latest_X   = emp_data[feature_cols].iloc[[-1]].values
    shap_vals  = explainer.shap_values(latest_X)
    pred_class = xgb.predict(latest_X)[0]
    reasons    = get_alert_reason(shap_vals[0, :, pred_class], feature_cols)

    if latest_label in ["High", "Medium"]:
        personalized_action = get_personalized_action(reasons, latest_label)
        reasons_html = "<br>".join(
            [f"&nbsp;&nbsp;{i+1}. {r}" for i, r in enumerate(reasons)]
        )
        st.markdown(f"""
        <div class="alert-box">
        <b>⚠️ Risk Alert — {selected} — {latest_label} Risk</b><br><br>
        <b>Top 3 Behavioral Signals Driving This Prediction:</b><br>
        {reasons_html}
        <br><br>
        <b>Personalized Action Recommendation:</b><br>
        {personalized_action}
        <br><br>
        <small><i>Note: Recommendations are generated from SHAP signal analysis,
        not a generic template. Each employee's top signals may differ.</i></small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(
            f"✅ {selected} is currently at Low Risk (Week 16). "
            f"No intervention required. Continue weekly monitoring."
        )

# ════════════════════════════════════════════════
# PAGE 3 — ALERT LOG
# FIX: Limit to top 20 by drift score,
#      add drift color coding,
#      sort correctly by week then drift,
#      add note about alert count
# ════════════════════════════════════════════════
elif page == "🚨 Alert Log":
    st.title("🚨 Burnout Risk Alert Log")
    st.markdown("---")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            ["High", "Medium", "Low"],
            default=["High", "Medium"]
        )
    with col_f2:
        week_filter = st.slider("Filter by Week Range", 1, 16, (7, 16),
                                help="Weeks 7–16 are where burnout drift begins")

    # Sort by week ascending then drift score descending
    filtered = pred_df[
        (pred_df["predicted_label"].isin(risk_filter)) &
        (pred_df["week_number"].between(*week_filter))
    ].sort_values(
        ["week_number", "overall_drift"],
        ascending=[True, False]
    )

    total_alerts = len(filtered)
    st.markdown(
        f"**{total_alerts} total alerts found** — showing top 20 by drift score per week. "
        f"High alert counts reflect the synthetic dataset's 3-phase design "
        f"(all 100 employees enter High Risk at Week 12)."
    )
    st.markdown("---")

    # Show top 20 only — sorted by drift score descending
    top20 = filtered.nlargest(20, "overall_drift")

    for _, row in top20.iterrows():
        label      = row["predicted_label"]
        emoji      = get_risk_emoji(label)
        color_class = f"{label.lower()}-risk"
        drift_tag  = get_drift_tag(row["overall_drift"])

        # Get SHAP reason for this specific employee + week
        emp_week = df[
            (df["employee_id"] == row["employee_id"]) &
            (df["week_number"]  == row["week_number"])
        ]
        if len(emp_week) > 0:
            X_row     = emp_week[feature_cols].values
            shap_vals = explainer.shap_values(X_row)
            pred_cls  = xgb.predict(X_row)[0]
            reasons   = get_alert_reason(shap_vals[0, :, pred_cls], feature_cols)
            # Format concisely for log view
            reason_str = " | ".join([r.split(" — ")[0] for r in reasons])
        else:
            reason_str = "N/A"

        st.markdown(f"""
        <div class="metric-card {color_class}">
        <b>{emoji} {row['employee_id']} — Week {int(row['week_number'])} — {label} Risk</b><br>
        <small>
        P(High): {row['prob_high']:.3f} &nbsp;|&nbsp;
        Drift: {drift_tag} &nbsp;|&nbsp;
        Actual Label: {row['actual_label']}
        </small><br>
        <small>📋 Top signals: {reason_str}</small>
        </div>
        """, unsafe_allow_html=True)

    if total_alerts > 20:
        st.info(
            f"ℹ️ Showing top 20 of {total_alerts} alerts ranked by drift score. "
            f"Adjust week range or risk filter to narrow results."
        )

# ════════════════════════════════════════════════
# PAGE 4 — DRIFT ANALYSIS
# FIX: Corrected heatmap title,
#      added phase boundary lines to heatmap,
#      added interpretation text under each chart,
#      fixed bar chart to show Week 1
# ════════════════════════════════════════════════
elif page == "📈 Drift Analysis":
    st.title("📈 Drift Score Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Bar chart — average drift per week ────────
    with col1:
        st.subheader("Average Drift Score per Week")
        weekly_drift = pred_df.groupby("week_number")["overall_drift"].mean().reset_index()
        # Ensure all 16 weeks are present
        all_weeks_df = pd.DataFrame({"week_number": range(1, 17)})
        weekly_drift = all_weeks_df.merge(weekly_drift, on="week_number", how="left").fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekly_drift["week_number"],
            y=weekly_drift["overall_drift"],
            marker_color=[
                "#C00000" if w >= 12 else "#ED7D31" if w >= 7 else "#70AD47"
                for w in weekly_drift["week_number"]
            ],
            text=weekly_drift["overall_drift"].round(3),
            textposition="outside"
        ))
        fig.add_vline(x=6.5,  line_dash="dash", line_color="#ED7D31",
                      opacity=0.7, annotation_text="Medium phase starts (W7)")
        fig.add_vline(x=11.5, line_dash="dash", line_color="#C00000",
                      opacity=0.7, annotation_text="High phase starts (W12)")
        fig.update_layout(
            height=380,
            xaxis=dict(
                title="Week Number",
                tickmode="linear", tick0=1, dtick=1,
                range=[0.5, 16.5]   # Fix: forces Week 1 bar to render fully
            ),
            yaxis=dict(title="Average Drift Score", range=[0, 3.2]),
            paper_bgcolor="white", plot_bgcolor="#F8FAFD",
            margin=dict(t=20, b=50, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📌 Week 7 spike (2.661) = earliest early warning signal, "
            "5 weeks before High Risk onset at W12. "
            "Week 1 = 0.000 because no 4-week baseline exists yet."
        )

    # ── Box plot — drift distribution by label ────
    with col2:
        st.subheader("Drift Score Distribution by Risk Level")
        fig = go.Figure()
        for label, color in [("Low", "#70AD47"), ("Medium", "#ED7D31"), ("High", "#C00000")]:
            d = pred_df[pred_df["actual_label"] == label]
            fig.add_trace(go.Box(
                y=d["overall_drift"], name=label,
                marker_color=color, boxmean=True,
                boxpoints="outliers"
            ))
        fig.add_hline(y=1.0,  line_dash="dot", line_color="#ED7D31",
                      opacity=0.6, annotation_text="Warning threshold (1.0)")
        fig.add_hline(y=1.5,  line_dash="dot", line_color="#C00000",
                      opacity=0.6, annotation_text="Critical threshold (1.5)")
        fig.update_layout(
            height=380, yaxis_title="Overall Drift Score",
            paper_bgcolor="white", plot_bgcolor="#F8FAFD",
            margin=dict(t=20, b=40, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📌 Classes overlap in drift score because the model uses all 29 features together — "
            "context switches, after-hours logins, and completion rate provide the discriminating signal."
        )

    # ── Heatmap ───────────────────────────────────
    st.markdown("---")

    # FIX: Corrected title — was "All Employees" but only shows 30
    st.subheader("Drift Heatmap — Sample of 30 Employees × All 16 Weeks")
    st.caption(
        "Randomly sampled 30 of 100 employees for readability. "
        "Green = Low Drift (normal), Yellow = Moderate, Orange = Warning, Red = Critical. "
        "Note the red column at Week 7 (phase transition spike) visible across all employees."
    )

    pivot = pred_df.pivot_table(
        index="employee_id", columns="week_number",
        values="overall_drift", aggfunc="mean"
    )
    sample = pivot.sample(min(30, len(pivot)), random_state=42)

    fig_heat = go.Figure(go.Heatmap(
        z=sample.values,
        x=[f"W{int(w)}" for w in sample.columns],
        y=sample.index,
        colorscale=[
            [0.0, "#70AD47"], [0.3, "#FFFF00"],
            [0.6, "#ED7D31"], [1.0, "#C00000"]
        ],
        colorbar=dict(
            title="Drift Score",
            tickvals=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            ticktext=["0 (None)", "0.5", "1.0 (Warning)", "1.5 (Critical)", "2.0", "2.5", "3.0"]
        ),
        zmin=0, zmax=3
    ))

    # Phase boundary lines on heatmap using add_shape (works with categorical axes)
    fig_heat.add_shape(
        type="line", xref="x", yref="paper",
        x0="W6", x1="W6", y0=0, y1=1,
        line=dict(color="white", width=2.5, dash="dash"),
        opacity=0.9
    )
    fig_heat.add_annotation(
        xref="x", yref="paper",
        x="W6", y=1.04,
        text="◀ Normal | Early Drift ▶",
        showarrow=False,
        font=dict(color="#ED7D31", size=11, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ED7D31",
        borderwidth=1
    )
    fig_heat.add_shape(
        type="line", xref="x", yref="paper",
        x0="W11", x1="W11", y0=0, y1=1,
        line=dict(color="white", width=2.5, dash="dash"),
        opacity=0.9
    )
    fig_heat.add_annotation(
        xref="x", yref="paper",
        x="W11", y=1.04,
        text="◀ Early Drift | High Risk ▶",
        showarrow=False,
        font=dict(color="#C00000", size=11, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#C00000",
        borderwidth=1
    )

    fig_heat.update_layout(
        height=640,
        xaxis_title="Week Number",
        yaxis_title="Employee ID",
        paper_bgcolor="white",
        margin=dict(t=80, b=50, l=80, r=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "📌 The red band consistently appearing at W7 across all employees confirms "
        "that phase-transition drift is a system-wide detectable signal, not employee-specific noise."
    )

# ════════════════════════════════════════════════
# PAGE 5 — ABOUT
# FIX: Added limitations, baseline models,
#      corrected timeline, humanized privacy text
# ════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Platform")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Project Details
        **Title:** Productivity Drift & Burnout Early Warning Platform

        **Team:**
        - Shubham Rawat (590014789)
        - Shital Dhasmana (590017562)
        - Barkha Bhatt (590016170)

        **Supervisors:**
        - Dr. Roshi Saxena (Industry Advisor)
        - Dr. Pooja Sarin (UPES Faculty)

        **Institute:** School of Computer Science (SoCS), UPES Dehradun

        **Timeline:** February 12 – May 15, 2026 (End-Term Presentation)

        **GitHub:** Repository will be published at project completion
        """)

    with col2:
        st.markdown("""
        ### Model Performance Summary
        | Model | Accuracy | AUC | F1 Macro |
        |---|---|---|---|
        | Logistic Regression (Baseline) | 1.000 | 1.000 | 1.000 |
        | SVM RBF (Baseline) | 0.997 | 1.000 | 0.997 |
        | XGBoost (Proposed) | 1.000 | 1.000 | 0.999 |
        | LSTM (Proposed) | 0.890 | 0.968 | 0.870 |
        | **Hybrid ★ (Deployed)** | **0.910** | **0.963** | **0.880** |
        | Published Benchmark [Shi et al., 2025] | — | 0.720 | — |

        *Note: Perfect baseline scores are expected on synthetic data with clean phase boundaries.
        The Hybrid model is deployed for its superior temporal learning and Low-class recall (0.76 vs 0.73 for LSTM alone).*

        ### Technology Stack
        - **ML Models:** XGBoost 2.0.3, TensorFlow-CPU 2.19.0
        - **Explainability:** SHAP 0.46.0 (TreeExplainer)
        - **Dashboard:** Streamlit 1.31.0, Plotly 5.18.0
        - **Data Processing:** Python 3.12, Pandas 2.2.0, Scikit-learn 1.4.0
        - **Real Data Collection:** Telegram Bot (python-telegram-bot 20.7), SQLite
        """)

    st.markdown("---")
    st.markdown("""
    ### Known Limitations

    We want to be upfront about what this project does and does not do:

    - **Synthetic dataset** — The model was trained on simulated behavioral data with
      deliberately clean phase boundaries. Real-world accuracy will be lower.
      We are collecting real data via the Telegram bot for End-Term validation.

    - **Fixed 16-week window** — The monitoring period is fixed at 16 weeks based on
      WHO burnout emergence timelines. Employees with longer or shorter burnout trajectories
      may not be captured accurately.

    - **No ground truth labels for real data** — We cannot verify burnout labels for
      real Telegram bot participants since clinical diagnosis was not feasible.

    - **Identical SHAP patterns at Week 16** — Because all synthetic employees follow
      the same phase design, Week 16 SHAP explanations show similar top signals.
      Real-world deployment will show more varied explanations.
    """)

    st.markdown("---")
    st.markdown("""
    ### Privacy & Ethics

    Honestly, this was something we debated a lot during the design phase.
    The moment you start monitoring employee behavior, you enter uncomfortable territory —
    and we wanted to make sure we were on the right side of that line.

    So we made a few decisions early on that we stuck to throughout:

    - We only look at **work output signals** — things like how many tasks were completed,
      how often someone logged in after hours, or how many times a task needed rework.
      We never touch emails, chat messages, or anything personal.

    - The employee's **actual name never enters the prediction model** — the model only
      sees anonymized behavioral numbers. The name appears only on the dashboard for
      the manager to identify who needs support.

    - Every alert comes with a **plain-language SHAP explanation** — so if an employee
      gets flagged, they can see exactly which behaviors triggered the alert and
      disagree with it if they think it is wrong.

    - We built this in line with India's **DPDP Act 2023**, which means no personal
      data is stored in the prediction layer.

    We are students, not a corporation — so we had no budget for legal review.
    But we tried to design this the way we would want our own workplace data handled.
    """)