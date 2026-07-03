"""Daily odds refresh — fetches fresh odds and saves to .odds_cache.json."""
import requests, json, os
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
today_str = datetime.now(TZ).strftime("%Y-%m-%d")

print(f"=== Health Check {today_str} ===")

# Check if cache already from today
try:
    with open(".odds_cache.json") as f:
        d = json.load(f)
    if d.get("date") == today_str:
        print("Cache already from today, skipping")
        exit(0)
except:
    pass

if not ODDS_API_KEY:
    print("No ODDS_API_KEY, skipping")
    exit(0)

# Fetch fresh odds
print("Fetching odds...")
try:
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}",
        timeout=10
    )
    if r.status_code == 200:
        data = r.json()
        cache = {"date": today_str, "data": data}
        with open(".odds_cache.json", "w") as f:
            json.dump(cache, f)
        print(f"Updated: {len(data)} games | Requests remaining: {r.headers.get('x-requests-remaining','?')}")
    else:
        print(f"Odds API error: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
