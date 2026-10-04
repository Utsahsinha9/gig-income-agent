import streamlit as st
import pandas as pd
import forecast
from agent import run_agent

st.set_page_config(page_title="Gig Income Budgeting Agent", layout="wide")
st.title("Irregular-Income Budgeting Agent")
st.caption("Forecasts safe-to-spend amounts for gig workers with volatile income — synthetic demo data")

st.markdown("""
<style>
    /* Metric cards get a subtle border, padding, and rounded corners */
    [data-testid="stMetric"] {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #1e40af;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #1e293b;
    }

    /* Section headers get a bit more breathing room and a bottom border */
    h2, h3 {
        margin-top: 1.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #eff6ff;
    }

    /* Sidebar background slightly differentiated */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Button styling */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: persona picker ---
df = pd.read_csv("synthetic_gig_income.csv", parse_dates=["date"])
personas = sorted(df["persona"].unique())
selected_persona = st.sidebar.selectbox("Choose a persona", personas)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Personas:**\n"
    "- `steady_delivery_partner`: low volatility, regular payouts\n"
    "- `spiky_freelancer`: high volatility, irregular timing\n"
    "- `new_gig_worker`: thin history (tests low-confidence fallback)"
)

# --- Main: run the agent for the selected persona ---
with st.spinner("Computing forecast..."):
    result = run_agent(selected_persona)  # no question yet — just the numbers

# --- Income history chart ---
st.subheader("Income History")
persona_weekly = forecast.weekly_totals(
    df[df["persona"] == selected_persona]
).reset_index()
persona_weekly.columns = ["week", "weekly_income"]
st.line_chart(persona_weekly.set_index("week")["weekly_income"])

# --- Forecast band (single-week method) ---
st.subheader("Forecast — This Week")
col1, col2, col3 = st.columns(3)
col1.metric("Floor (conservative)", f"₹{result['week_forecast']['floor']:,.0f}")
col2.metric("Typical", f"₹{result['week_forecast']['typical']:,.0f}")
col3.metric("Optimistic", f"₹{result['week_forecast']['optimistic']:,.0f}")

if not result["confident"]:
    st.warning(
        "Low confidence — this persona doesn't have enough income history yet. "
        "Showing a wider, more cautious estimate."
    )

# --- Safe-to-spend recommendation ---
st.subheader("Safe-to-Spend Recommendation")
col1, col2 = st.columns(2)
col1.metric("Safe to spend this week", f"₹{result['safe_to_spend']:,.0f}")
col2.metric("Recommended buffer", f"₹{result['buffer_amount']:,.0f}")

st.caption(
    "Safe-to-spend blends the conservative floor with the typical forecast, "
    "since the floor can be honestly ₹0 for volatile income patterns."
)

# --- Q&A chat ---
st.subheader("Ask About Your Budget")
question = st.text_input("e.g. \"Can I afford to spend ₹5000 this week?\"")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        qa_result = run_agent(selected_persona, question=question)
    st.markdown(f"**Answer:** {qa_result['answer']}")