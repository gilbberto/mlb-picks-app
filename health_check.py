"""Daily health check: pre-fetch odds and validate schedule. Runs at 6 AM Chihuahua via GitHub Actions."""
import requests, json, os, sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

try:
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_CACHE_PATH = os.path.join(os.path.dirname(__file__) or ".", ".odds_cache.json")
MLB_API = "https://statsapi.mlb.com/api/v1"

today_str = datetime.now(TZ).strftime("%Y-%m-%d")
today_api = datetime.now(TZ).strftime("%m/%d/%Y")

print(f"=== Health Check {today_str} ===")

# 1. Check MLB schedule
print("1. Schedule...")
try:
    r = requests.get(f"{MLB_API}/schedule?sportId=1&date={today_api}&hydrate=probablePitcher", timeout=10)
    if r.status_code == 200:
        games = []
        for d in r.json().get("dates", []):
            games.extend(d.get("games", []))
        print(f"   ✅ {len(games)} juegos hoy")
    else:
        print(f"   ❌ MLB API error {r.status_code}")
except Exception as e:
    print(f"   ❌ {e}")

# 2. Check/fetch odds
print("2. Odds...")
try:
    with open(ODDS_CACHE_PATH) as f:
        cached = json.load(f)
    cache_date = cached.get("date", "") if isinstance(cached, dict) else ""
except:
    cache_date = ""

if cache_date == today_str:
    print(f"   ✅ Cache ya es de hoy")
else:
    print(f"   ⏳ Cache es de {cache_date or 'N/A'}, refrescando...")
    if ODDS_API_KEY:
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}",
                timeout=10
            )
            if r.status_code == 200:
                odds = r.json()
                cache = {"date": today_str, "data": odds}
                with open(ODDS_CACHE_PATH, "w") as f:
                    json.dump(cache, f)
                print(f"   ✅ {len(odds)} juegos con odds actualizados")
            else:
                print(f"   ❌ Odds API error {r.status_code}")
        except Exception as e:
            print(f"   ❌ {e}")
    else:
        print(f"   ❌ Sin ODDS_API_KEY")

print("=== Health Check Done ===")
