"""Daily odds refresh — uses GitHub API to update .odds_cache.json directly (no git conflicts)."""
import requests, json, os, base64
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

try:
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
REPO = "gilbberto/mlb-picks-app"
FILE_PATH = ".odds_cache.json"
today_str = datetime.now(TZ).strftime("%Y-%m-%d")

print(f"=== Health Check {today_str} ===")

# 1. Fetch fresh odds
print("1. Odds API...")
if not ODDS_API_KEY:
    print("   ❌ Sin ODDS_API_KEY")
    exit(1)

r = requests.get(
    f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}",
    timeout=10
)
if r.status_code != 200:
    print(f"   ❌ Odds API error {r.status_code}")
    exit(1)

odds = r.json()
cache = {"date": today_str, "data": odds}
content = json.dumps(cache, indent=2)
print(f"   ✅ {len(odds)} juegos | Requests: {r.headers.get('x-requests-remaining','?')}")

# 2. Update GitHub via API (no git needed)
print("2. GitHub update...")
if not GH_TOKEN:
    print("   ⚠️ Sin GH_TOKEN, guardando localmente")
    with open(FILE_PATH, "w") as f:
        f.write(content)
    print(f"   ✅ Guardado local: {len(odds)} juegos")
    exit(0)

headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
api_url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"

# Get current file SHA
sha = None
try:
    get_r = requests.get(api_url, headers=headers, timeout=10)
    if get_r.status_code == 200:
        sha = get_r.json().get("sha")
except:
    pass

# Put new content
payload = {
    "message": f"auto-health: refresh odds cache for {today_str}",
    "content": base64.b64encode(content.encode()).decode(),
    "branch": "main"
}
if sha:
    payload["sha"] = sha

put_r = requests.put(api_url, headers=headers, json=payload, timeout=10)
if put_r.status_code in (200, 201):
    print(f"   ✅ GitHub actualizado: {len(odds)} juegos")
else:
    print(f"   ❌ GitHub error {put_r.status_code}: {put_r.text[:200]}")

print("=== Done ===")
