"""
bankroll.py — Kelly Criterion sizing + P&L tracker + auto-settlement.
Lee/escribe picks.json para llevar control de resultados.
"""
import json, os, requests
from datetime import datetime, timedelta, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "picks.json")
MLB_API = "https://statsapi.mlb.com/api/v1"

# Team name mapping: abbreviation -> full name (MLB API)
TEAM_NAMES = {
    "ATL": "Atlanta Braves", "AZ": "Arizona Diamondbacks",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}
REV_TEAM = {v.lower(): k for k, v in TEAM_NAMES.items()}

# ─── Calibration ───
# Based on XGBoost validation on 262 games (2026 season)
def calibrate_ml(prob):
    """Calibrate ML probability. XGBoost (8174 games) validated 262 games.
    Smooth monotonic piecewise linear. Overall ML acc: 57.6%."""
    if prob < 0.50:
        return 1.0 - calibrate_ml(1.0 - prob)
    # Target points from validation: prob -> calibrated
    # 0.525→0.574, 0.575→~0.58*, 0.625→0.604, 0.675→0.615, 0.725→0.667, 0.775→0.714
    # *55-60% bucket was noisy (N=51, actual=45.1% — treated as outlier); smoothed
    if prob < 0.55:
        t = (prob - 0.50) / 0.05
        return 0.555 + t * 0.025  # 0.50->0.555, 0.55->0.580
    if prob < 0.60:
        t = (prob - 0.55) / 0.05
        return 0.580 + t * 0.010  # 0.55->0.580, 0.60->0.590
    if prob < 0.70:
        t = (prob - 0.60) / 0.10
        return 0.590 + t * 0.050  # 0.60->0.590, 0.70->0.640
    if prob < 0.80:
        t = (prob - 0.70) / 0.10
        return 0.640 + t * 0.060  # 0.70->0.640, 0.80->0.700
    return min(0.700 + (prob - 0.80) * 0.20, 0.78)  # gentle slope beyond 80%

def calibrate_rl(prob):
    """Calibrate RL -1.5 probability. XGBoost (8174 games) validated 262 games.
    Overall RL -1.5 cover rate: ~36.6%. Smooth monotonic shrinkage."""
    # Smooth shrinkage target: 0.366 (mean), with confidence-dependent adjustment
    # Low predictions (<15%) are very underconfident; high ones are overconfident
    if prob < 0.08:
        return 0.05 + prob * 0.625  # 0->0.05, 0.08->0.10
    if prob < 0.15:
        t = (prob - 0.08) / 0.07
        return 0.10 + t * 0.10  # 0.08->0.10, 0.15->0.20
    if prob < 0.25:
        t = (prob - 0.15) / 0.10
        return 0.20 + t * 0.10  # 0.15->0.20, 0.25->0.30
    if prob < 0.35:
        t = (prob - 0.25) / 0.10
        return 0.30 + t * 0.06  # 0.25->0.30, 0.35->0.36
    if prob < 0.50:
        t = (prob - 0.35) / 0.15
        return 0.36 + t * 0.12  # 0.35->0.36, 0.50->0.48
    return min(0.48 + (prob - 0.50) * 0.30, 0.65)  # 0.50->0.48, 0.65->0.53

# ─── Kelly Criterion ───
def american_to_prob(odds):
    if odds is None or odds == 0: return None
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def kelly_fraction(model_prob, odds_american):
    """Full Kelly fraction. Returns 0 if no edge."""
    if model_prob is None or odds_american is None or odds_american == 0:
        return 0
    if odds_american > 0:
        b = odds_american / 100.0
    else:
        b = 100.0 / abs(odds_american)
    p = model_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f)

def recommend_stake(model_prob, odds_american, bankroll=100, kelly_frac=0.25):
    """Returns (stake_amount, units, label)."""
    f = kelly_fraction(model_prob, odds_american)
    if f <= 0:
        return (0, 0, "No bet")
    stake = round(bankroll * f * kelly_frac, 2)
    units = round(stake / 10, 1) if stake > 0 else 0
    if f * kelly_frac >= 0.03:
        label = "🔥 High"
    elif f * kelly_frac >= 0.015:
        label = "⭐ Med"
    else:
        label = "Low"
    return (stake, units, label)

# ─── P&L Tracker ───
def load_picks():
    if not os.path.exists(DB_PATH):
        return {"bankroll": 1000, "history": []}
    try:
        with open(DB_PATH) as f:
            d = json.load(f)
        if "history" not in d: d["history"] = []
        return d
    except:
        return {"bankroll": 1000, "history": []}

def save_picks(data):
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

def add_pick(date, game, market, model_prob, odds, stake, bankroll, label="", team="", detail=""):
    data = load_picks()
    # Parse team abbreviation from "AWAY @ HOME" if team not given
    if not team and " @ " in game:
        parts = game.split(" @ ")
        if market in ("ML", "RL"):
            # Default to first team (away) if not specified
            team = parts[0]
    pick = {
        "id": len(data["history"]) + 1,
        "date": date,
        "game": game,
        "market": market,
        "team": team,
        "detail": detail,
        "model_prob": round(model_prob, 3),
        "odds": odds,
        "stake": round(stake, 2),
        "bankroll_before": round(bankroll, 2),
        "result": None,
        "profit": None,
        "label": label,
        "settled": False,
    }
    data["history"].append(pick)
    save_picks(data)
    return pick["id"]

def settle_pick(pick_id, won):
    """Mark a pick as won/lost and update bankroll."""
    data = load_picks()
    bankroll = data["bankroll"]
    for p in data["history"]:
        if p["id"] == pick_id and not p.get("settled"):
            p["result"] = "W" if won else "L"
            if won:
                if p["odds"] > 0:
                    profit = p["stake"] * (p["odds"] / 100.0)
                else:
                    profit = p["stake"] * (100.0 / abs(p["odds"]))
            else:
                profit = -p["stake"]
            p["profit"] = round(profit, 2)
            p["settled"] = True
            data["bankroll"] = round(bankroll + profit, 2)
            save_picks(data)
            return p["profit"]
    return None

def get_pnl():
    """Return summary stats."""
    data = load_picks()
    h = data["history"]
    settled = [p for p in h if p.get("settled")]
    wins = [p for p in settled if p["result"] == "W"]
    losses = [p for p in settled if p["result"] == "L"]
    total_profit = sum(p.get("profit", 0) for p in settled)
    total_staked = sum(p.get("stake", 0) for p in settled)
    return {
        "bankroll": data["bankroll"],
        "total": len(settled),
        "wins": len(wins),
        "losses": len(losses),
        "pct": round(len(wins) / len(settled) * 100, 1) if settled else 0,
        "profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "open": len([p for p in h if not p.get("settled")]),
    }

def today_checks():
    """Print pending picks from today/yesterday not yet settled."""
    data = load_picks()
    pending = [p for p in data["history"] if not p.get("settled")]
    return pending

# ─── Auto-Settlement ───
def _same_team(team, abbr, full_name):
    """Check if team string matches abbreviation or full name."""
    return team == abbr or team.lower() == full_name.lower()

def _fetch_mlb_games(date_str):
    """Get completed MLB games for a given date."""
    url = f"{MLB_API}/schedule?date={date_str}&sportId=1&hydrate=linescore,team"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return []
        games = []
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("codedGameState") != "F":
                    continue
                away = g["teams"]["away"]
                home = g["teams"]["home"]
                away_name = away["team"]["name"]
                home_name = home["team"]["name"]
                away_abbr = REV_TEAM.get(away_name.lower(), away_name)
                home_abbr = REV_TEAM.get(home_name.lower(), home_name)
                games.append({
                    "away_abbr": away_abbr,
                    "home_abbr": home_abbr,
                    "away_name": away_name,
                    "home_name": home_name,
                    "away_runs": away.get("score", 0),
                    "home_runs": home.get("score", 0),
                    "label": f"{away_abbr} @ {home_abbr}",
                })
        return games
    except:
        return []

def auto_settle():
    """Auto-settle pending picks against MLB API results."""
    data = load_picks()
    settled_count = 0
    errors = []

    # Collect unique dates with pending picks
    pending_dates = set()
    for p in data["history"]:
        if not p.get("settled") and p.get("date"):
            pending_dates.add(p["date"])

    for date_str in sorted(pending_dates):
        games = _fetch_mlb_games(date_str)
        for p in data["history"]:
            if p.get("settled") or p.get("date") != date_str:
                continue
            game_label = p.get("game", "")
            market = p.get("market", "")
            team = p.get("team", "")

            # Find matching game
            match = None
            for g in games:
                if g["label"] == game_label:
                    match = g
                    break
            if not match:
                continue

            try:
                won = None
                if market == "ML":
                    if not team: continue
                    if _same_team(team, match["away_abbr"], match["away_name"]):
                        won = match["away_runs"] > match["home_runs"]
                    elif _same_team(team, match["home_abbr"], match["home_name"]):
                        won = match["home_runs"] > match["away_runs"]

                elif market in ("RL -1.5", "RL +1.5"):
                    if not team: continue
                    is_away = _same_team(team, match["away_abbr"], match["away_name"])
                    team_runs = match["away_runs"] if is_away else match["home_runs"]
                    opp_runs = match["home_runs"] if is_away else match["away_runs"]
                    diff = team_runs - opp_runs
                    if market == "RL -1.5":
                        won = diff >= 1.5
                    else:  # RL +1.5
                        won = diff >= -1.5

                elif market == "O/U":
                    detail = p.get("detail", "")
                    total = match["away_runs"] + match["home_runs"]
                    pick_side = team  # "Over" or "Under"
                    # Parse line from detail (e.g., "o8.5", "u8.5", "o7", "u7")
                    line = None
                    if detail:
                        try:
                            line = float(detail.replace("o","").replace("u",""))
                        except:
                            pass
                    if line is not None and pick_side:
                        if pick_side.lower() == "over":
                            won = total > line
                        elif pick_side.lower() == "under":
                            won = total < line

                if won is not None:
                    profit = settle_pick(p["id"], won)
                    if profit is not None:
                        settled_count += 1
            except Exception as e:
                errors.append(f"Pick #{p['id']}: {e}")

    return settled_count, errors

if __name__ == "__main__":
    # Demo / test
    print("=== Kelly Demo ===")
    for prob, odds in [(0.79, -120), (0.77, -115), (0.57, +132)]:
        stake, units, label = recommend_stake(prob, odds, bankroll=1000)
        print(f"  prob={prob:.0%}, odds={odds:+d}: stake=${stake:.2f} ({units}u) {label}")

    print("\n=== P&L Status ===")
    pnl = get_pnl()
    print(f"  Bankroll: ${pnl['bankroll']:.0f}")
    print(f"  Record: {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
    print(f"  Profit: ${pnl['profit']:.2f} ({pnl['roi']}%)")
    print(f"  Open: {pnl['open']}")
