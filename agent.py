import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

import forecast  # reuses weekly_totals() / forecast_persona_horizon()

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


class AgentState(TypedDict):
    persona: str
    question: Optional[str]
    week_forecast: Optional[dict]
    safe_to_spend: Optional[float]
    safe_to_spend_weekly: Optional[float]
    buffer_amount: Optional[float]
    confident: Optional[bool]
    answer: Optional[str]


def ingest_node(state: AgentState) -> AgentState:
    df = forecast.pd.read_csv("synthetic_gig_income.csv", parse_dates=["date"])
    weekly = forecast.weekly_totals(df[df["persona"] == state["persona"]]).reset_index()
    weekly.columns = ["week", "weekly_income"]

    fc = forecast.forecast_persona_horizon(weekly)  # multi-week horizon, not single week
    latest = fc.iloc[-1]

    state["week_forecast"] = {
        "floor": float(latest["floor"]),
        "typical": float(latest["typical"]),
        "optimistic": float(latest["optimistic"]),
    }
    state["confident"] = bool(latest["confident"])
    return state


def safe_to_spend_node(state: AgentState) -> AgentState:
    floor = state["week_forecast"]["floor"]

    if not state["confident"]:
        safe_total = floor * 0.5
        buffer_total = floor * 0.5
    else:
        safe_total = floor * 0.7
        buffer_total = floor * 0.3

    state["safe_to_spend"] = safe_total
    state["buffer_amount"] = buffer_total
    state["safe_to_spend_weekly"] = safe_total / forecast.HORIZON_WEEKS
    return state


def qa_node(state: AgentState) -> AgentState:
    if not state.get("question"):
        state["answer"] = None
        return state

    context = f"""
You are a budgeting assistant for a gig worker with irregular income.
Forecast for the next {forecast.HORIZON_WEEKS} weeks: floor=₹{state['week_forecast']['floor']:.0f},
typical=₹{state['week_forecast']['typical']:.0f},
optimistic=₹{state['week_forecast']['optimistic']:.0f}.
Recommended safe-to-spend over the next {forecast.HORIZON_WEEKS} weeks: ₹{state['safe_to_spend']:.0f} (~₹{state['safe_to_spend_weekly']:.0f}/week).
Recommended buffer to set aside: ₹{state['buffer_amount']:.0f}.
Confidence: {"normal" if state['confident'] else "LOW — limited income history, being extra conservative"}.

Answer the user's question using ONLY these numbers. Be direct and brief.
If the question can't be answered from these numbers, say so honestly.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.3,
    )
    state["answer"] = response.choices[0].message.content
    return state


graph = StateGraph(AgentState)
graph.add_node("ingest", ingest_node)
graph.add_node("safe_to_spend", safe_to_spend_node)
graph.add_node("qa", qa_node)

graph.set_entry_point("ingest")
graph.add_edge("ingest", "safe_to_spend")
graph.add_edge("safe_to_spend", "qa")
graph.add_edge("qa", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "persona": "spiky_freelancer",
        "question": "Can I afford to spend ₹5000 this week?",
    })

    print("\nForecast:", result["week_forecast"])
    print("Confident:", result["confident"])
    print(f"Safe to spend (total, {forecast.HORIZON_WEEKS} weeks):", round(result["safe_to_spend"]))
    print("Safe to spend (weekly):", round(result["safe_to_spend_weekly"]))
    print("Buffer:", round(result["buffer_amount"]))
    print("\nAnswer:", result["answer"])