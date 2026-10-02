import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

import forecast

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

    fc = forecast.forecast_persona(weekly)  # back to the original single-week method
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
    typical = state["week_forecast"]["typical"]

    # Blend floor and typical instead of relying on floor alone —
    # floor can be honestly ₹0 for a volatile persona, which would
    # otherwise zero out the whole recommendation.
    if not state["confident"]:
        safe_total = 0.5 * floor + 0.2 * typical
        buffer_total = typical - safe_total if typical > safe_total else 0.0
    else:
        safe_total = 0.3 * floor + 0.4 * typical
        buffer_total = typical - safe_total if typical > safe_total else 0.0

    state["safe_to_spend"] = safe_total
    state["buffer_amount"] = buffer_total
    state["safe_to_spend_weekly"] = safe_total  # single week now, no horizon division
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
Recommended safe-to-spend this week: ₹{state['safe_to_spend']:.0f}.
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


def run_agent(persona: str, question: Optional[str] = None) -> dict:
    """
    Callable entry point for the UI (or anything else) to use.
    Returns the full result dict — persona's forecast, safe-to-spend
    figures, confidence, and the LLM's answer if a question was asked.
    """
    return app.invoke({"persona": persona, "question": question})


if __name__ == "__main__":
    result = run_agent("spiky_freelancer", "Can I afford to spend ₹5000 this week?")
    print("\nForecast:", result["week_forecast"])
    print("Confident:", result["confident"])
    print("Safe to spend (weekly):", round(result["safe_to_spend"]))
    print("Buffer:", round(result["buffer_amount"]))
    print("\nAnswer:", result["answer"])