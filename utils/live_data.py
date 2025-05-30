import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_live_match_data():
    url = "https://api.cricapi.com/v1/currentMatches"
    api_key = os.getenv("CRICKET_API_KEY")
    params = {"apikey": api_key, "offset": 0}  # Use query params, not headers

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if not data.get("status") == "success":
            print(f"❌ API error: {data.get('reason')}")
            return []

        matches = []
        for match in data.get("data", []):
            match_info = {
                "team1": match.get("teamInfo", [{}])[0].get("name", "N/A"),
                "team2": match.get("teamInfo", [{}])[1].get("name", "N/A") if len(match.get("teamInfo", [])) > 1 else "N/A",
                "venue": match.get("venue", "N/A"),
                "status": match.get("status", "N/A"),
                "start_time": match.get("dateTimeGMT", "N/A"),
                "match_id": match.get("id", "N/A"),
                "score": match.get("score", [])
            }
            matches.append(match_info)
        return matches

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code} - {e.response.reason}")
        print("Response text:", e.response.text)
    except Exception as e:
        print(f"❌ Error fetching live data: {e}")
    return []


def get_scheduled_matches():
    """
    Filters matches to return those which are upcoming.
    """
    all_matches = get_live_match_data()
    return [
        m for m in all_matches
        if m.get("status", "").lower() in ["not started", "scheduled", "upcoming"]
    ]
