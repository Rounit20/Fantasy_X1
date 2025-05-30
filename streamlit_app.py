import os
import warnings
import streamlit as st
from utils.retriever import Retriever
from utils.live_data import get_live_match_data
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timezone
import requests

# Suppress the torch.classes warning
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize Retriever (which loads player_stats & match_conditions & FAQs)
retriever = Retriever(
    player_stats_path="data/player_stats.json",
    match_conditions_path="data/match_conditions.json",
    faqs_path="data/faqs.json",
)

# ─── Helpers ──────────────────────────────────────────────────

def get_today_str():
    return datetime.now(timezone.utc).date().isoformat()

def get_todays_matches(matches):
    today = get_today_str()
    return [m for m in matches if m.get("start_time", "").startswith(today)]

def format_score(score_list):
    if not score_list:
        return "No score available"
    lines = []
    for inn in score_list:
        lines.append(f"**{inn.get('inning','Inning')}**: {inn.get('r',0)}/{inn.get('w',0)} in {inn.get('o',0)} overs")
    return "\n".join(lines)

# ─── Scorecard & Live XI ──────────────────────────────────────

def get_scorecard(match_id):
    api_key = os.getenv("CRICKET_API_KEY")
    url = "https://api.cricapi.com/v1/match_scorecard"
    params = {"apikey": api_key, "id": match_id}
    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        if data.get("status") != "success":
            st.error(f"Scorecard API error: {data.get('reason')}")
            return None
        return data["data"]
    except Exception as e:
        st.error(f"Error fetching scorecard: {e}")
        return None

def select_fantasy_xi(scorecard):
    innings = scorecard.get("scorecard", [])
    stats = {}
    # accumulate batting & bowling
    for inn in innings:
        for b in inn.get("batting", []):
            nm = b.get("batsman", {}).get("name")
            if not nm: continue
            stats.setdefault(nm, {"runs":0,"wickets":0})
            stats[nm]["runs"] += b.get("r",0)
        for bw in inn.get("bowling", []):
            nm = bw.get("bowler", {}).get("name")
            if not nm: continue
            stats.setdefault(nm, {"runs":0,"wickets":0})
            stats[nm]["wickets"] += bw.get("w",0)
    # build and sort
    players = [
        {"name":nm, "runs":v["runs"], "wickets":v["wickets"], "score": v["runs"] + v["wickets"]*20}
        for nm,v in stats.items()
    ]
    players.sort(key=lambda x: x["score"], reverse=True)
    xi = players[:11]
    if xi:
        xi[0]["captain"] = True
    if len(xi)>1:
        xi[1]["vice_captain"] = True
    return xi

# ─── Prediction for Upcoming Matches ───────────────────────────

def predict_fantasy_xi(match):
    """
    Very basic predictor: uses historical average_runs & average_wickets 
    plus a venue bonus if available.
    """
    t1, t2 = match["team1"], match["team2"]
    venue = match.get("venue","")
    cond = retriever.match_conditions
    stats_db = {p["player"]: p for p in retriever.player_stats}

    def score_player(p):
        s = p.get("average_runs",0) + p.get("average_wickets",0)*20
        # bonus for venue if stored in match_conditions
        vmap = p.get("venue_performance",{})
        s += vmap.get(venue,{}).get("bonus",0)
        return s

    candidates = []
    for nm,p in stats_db.items():
        if p.get("team") in (t1,t2):
            sc = score_player(p)
            candidates.append({
                "name": nm,
                "team": p.get("team"),
                "runs": p.get("average_runs",0),
                "wickets": p.get("average_wickets",0),
                "score": sc
            })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    xi = candidates[:11]
    if xi:
        xi[0]["captain"] = True
    if len(xi)>1:
        xi[1]["vice_captain"] = True
    return xi

# ─── Streamlit UI ─────────────────────────────────────────────

st.title("🏏 Fantasy Cricket Chatbot Assistant")

# Sidebar
with st.sidebar:
    st.header("Try Asking:")
    for q in [
      "Who is a good differential pick today?",
      "Should I pick Rohit or Gill?",
      "Who performs well on Mumbai pitches?",
      "What’s the scoring rule for bowlers?",
    ]:
        st.markdown(f"- {q}")
    st.markdown("---")

    all_matches = get_live_match_data()

    # Live Matches
    st.subheader("Live Matches")
    live = [m for m in all_matches if m["status"].lower() not in ["scheduled","upcoming","not started"]]
    if live:
        for m in live:
            st.markdown(f"**{m['team1']} vs {m['team2']}**")
            st.markdown(format_score(m.get("score",[])))
            st.markdown(f"Status: {m['status']}")
            st.markdown(f"Match ID: `{m['match_id']}`")
            st.markdown("---")
    else:
        st.write("No live matches.")

    # Upcoming Matches
    st.subheader("Upcoming Matches")
    upcoming = [m for m in all_matches if m["status"].lower() in ["scheduled","upcoming","not started"]]
    if upcoming:
        # show a selector for prediction
        choices = {f"{m['team1']} vs {m['team2']} @ {m['venue']}": m for m in upcoming}
        sel = st.selectbox("Select upcoming match", list(choices.keys()))
        if st.button("Predict XI"):
            pred = predict_fantasy_xi(choices[sel])
            st.success("Predicted Fantasy XI:")
            for i,p in enumerate(pred,1):
                tag = "(C)" if p.get("captain") else "(VC)" if p.get("vice_captain") else ""
                st.markdown(f"**{i}. {p['name']}** {tag} — {p['team']}")
                st.markdown(f"- AvgRuns: {p['runs']}, AvgWkts: {p['wickets']}, Score: {p['score']}")
                st.markdown("---")
    else:
        st.write("No upcoming matches.")

# Chat interface
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("Ask me anything about Fantasy Cricket!")
    go = st.form_submit_button("Send")

if go and user_q:
    docs = retriever.retrieve(user_q)
    today = get_todays_matches(all_matches)
    ctx = "\n".join(docs)
    if today:
        ctx += "\n\nToday's matches:\n" + "\n".join(f"{m['team1']} vs {m['team2']}" for m in today)
    messages = [
        {"role":"system","content":"You are a helpful fantasy cricket assistant. Use context to answer."},
        {"role":"user","content":f"Context:\n{ctx}\n\nQ: {user_q}"},
    ]
    r = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, max_tokens=500
    )
    st.session_state.history.append({"user":user_q, "bot":r.choices[0].message.content.strip()})

for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])

# Fantasy XI — live/completed matches
st.markdown("---")
st.subheader("🧠 Fantasy XI from Live Match")
mid = st.text_input("Enter Match ID", key="live_mid")
if st.button("Fetch Live XI") and mid:
    sc = get_scorecard(mid.strip())
    if sc:
        st.markdown(f"**Match:** {sc.get('name','N/A')}")
        st.markdown(f"**Status:** {sc.get('status','N/A')}")
        st.markdown(f"**Teams:** {', '.join(sc.get('teams',[]))}")
        st.markdown("---")
        xi = select_fantasy_xi(sc)
        if xi:
            for i,p in enumerate(xi,1):
                flag = "(C)" if p.get("captain") else "(VC)" if p.get("vice_captain") else ""
                st.markdown(f"**{i}. {p['name']}** {flag}")
                st.markdown(f"- Runs: {p['runs']}, Wkts: {p['wickets']} (Score: {p['score']})")
        else:
            st.error("No XI could be built.")
    else:
        st.error("Invalid Match ID or API error.")
