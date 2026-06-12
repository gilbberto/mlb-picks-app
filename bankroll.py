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
    """Calibrate ML probability. XGBoost (35 features, 8198 games 2023-2026).
    Light calibration — XGBoost binary:logistic tends to be well-calibrated.
    Revisit after ≥50 settled picks."""
    if prob < 0.50:
        return 1.0 - calibrate_ml(1.0 - prob)
    # Gentle piecewise linear (50% of old calibration strength)
    if prob < 0.55:
        t = (prob - 0.50) / 0.05
        return 0.525 + t * 0.035  # 0.50->0.525, 0.55->0.560
    if prob < 0.65:
        t = (prob - 0.55) / 0.10
        return 0.560 + t * 0.060  # 0.55->0.560, 0.65->0.620
    if prob < 0.80:
        t = (prob - 0.65) / 0.15
        return 0.620 + t * 0.105  # 0.65->0.620, 0.80->0.725
    return min(0.725 + (prob - 0.80) * 0.25, 0.85)  # 0.80->0.725, 0.95->0.762

def calibrate_rl(prob):
    """Calibrate RL probability. XGBoost (35 features, 8198 games 2023-2026).
    Base RL -1.5 cover rate: ~36.6%. Light shrinkage toward mean."""
    if prob < 0.10:
        return 0.06 + prob * 0.40  # 0->0.06, 0.10->0.10
    if prob < 0.20:
        t = (prob - 0.10) / 0.10
        return 0.10 + t * 0.08  # 0.10->0.10, 0.20->0.18
    if prob < 0.35:
        t = (prob - 0.20) / 0.15
        return 0.18 + t * 0.14  # 0.20->0.18, 0.35->0.32
    if prob < 0.55:
        t = (prob - 0.35) / 0.20
        return 0.32 + t * 0.14  # 0.35->0.32, 0.55->0.46
    return min(0.46 + (prob - 0.55) * 0.30, 0.70)  # 0.55->0.46, 0.70->0.51

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
    stake = min(stake, 100)
    units = round(stake / 10, 1) if stake > 0 else 0
    if f * kelly_frac >= 0.03:
        label = "🔥 High"
    elif f * kelly_frac >= 0.015:
        label = "⭐ Med"
    else:
        label = "Low"
    return (stake, units, label)

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

# ─── P&L Tracker ───
def _current_week_start():
    """Return Monday 00:00 of current week as datetime."""
    today = datetime.now(TZ)
    return today - timedelta(days=today.weekday())

def check_weekly_reset():
    """Reset weekly bankroll when a new week starts (Monday)."""
    data = load_picks()
    now = datetime.now(TZ)
    current_start = _current_week_start().strftime("%Y-%m-%d")
    saved_start = data.get("weekly_start", "")
    last_reset_date = data.get("last_weekly_reset", "")
    if saved_start == current_start or last_reset_date == current_start:
        return
    ws = saved_start or current_start
    h = data["history"]
    weekly_picks = [p for p in h if p.get("date", "") >= ws]
    weekly_settled = [p for p in weekly_picks if p.get("settled")]
    weekly_w = sum(1 for p in weekly_settled if p["result"] == "W")
    weekly_l = sum(1 for p in weekly_settled if p["result"] == "L")
    weekly_profit = sum(p.get("profit") or 0 for p in weekly_picks if p.get("profit") is not None)
    weekly_history = data.get("weekly_history", [])
    weekly_history.append({
        "week_start": ws,
        "week_end": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "bankroll_start": data.get("weekly_bankroll", 1000),
        "profit": round(weekly_profit, 2),
        "wins": weekly_w, "losses": weekly_l,
        "picks": len(weekly_settled),
    })
    data["weekly_history"] = weekly_history
    data["weekly_bankroll"] = 1000
    data["weekly_start"] = current_start
    data["last_weekly_reset"] = current_start
    data.pop("cash_adjust", None)
    save_picks(data)
    print(f"  Weekly bankroll reset to $1000 ({current_start})")

def load_picks():
    if not os.path.exists(DB_PATH):
        return {"bankroll": 1000, "history": [], "weekly_bankroll": 1000,
                "weekly_start": _current_week_start().strftime("%Y-%m-%d")}
    try:
        with open(DB_PATH) as f:
            d = json.load(f)
        if "history" not in d: d["history"] = []
        if "weekly_bankroll" not in d:
            d["weekly_bankroll"] = 1000
            d["weekly_start"] = _current_week_start().strftime("%Y-%m-%d")
        return d
    except:
        return {"bankroll": 1000, "history": [], "weekly_bankroll": 1000,
                "weekly_start": _current_week_start().strftime("%Y-%m-%d")}

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
    """Mark a pick as won/lost/push and update bankroll. won=True/False/None (push)."""
    data = load_picks()
    bankroll = data["bankroll"]
    for p in data["history"]:
        if p["id"] == pick_id and not p.get("settled"):
            if won == "cancel":
                p["result"] = "C"
                profit = 0
                data["bankroll"] = round(bankroll + p["stake"], 2)  # Return stake
            elif won is None:
                p["result"] = "P"
                profit = 0  # No gain/loss
                data["bankroll"] = round(bankroll + p["stake"], 2)  # Return stake
            elif won:
                p["result"] = "W"
                if p["odds"] > 0:
                    profit = p["stake"] * (p["odds"] / 100.0)
                else:
                    profit = p["stake"] * (100.0 / abs(p["odds"]))
            else:
                p["result"] = "L"
                profit = -p["stake"]
            p["profit"] = round(profit, 2)
            p["settled"] = True
            data["bankroll"] = round(bankroll + profit, 2)
            save_picks(data)
            return p["profit"]
    return None

CASH_ADJUST = None  # Se carga desde data["cash_adjust"]

def get_pnl():
    """Return summary stats."""
    data = load_picks()
    h = data["history"]
    cash_adj = data.get("cash_adjust", 0) or 0
    settled = [p for p in h if p.get("settled")]
    wins = [p for p in settled if p["result"] == "W"]
    losses = [p for p in settled if p["result"] == "L"]
    total_staked = sum(p.get("stake", 0) for p in settled)
    total_profit = sum(p.get("profit") or 0 for p in h if p.get("profit") is not None)
    open_stakes = sum(p.get("stake", 0) for p in h if not p.get("settled"))
    ws = data.get("weekly_start", _current_week_start().strftime("%Y-%m-%d"))
    try:
        ws_dt = datetime.strptime(ws, "%Y-%m-%d").replace(tzinfo=TZ) if TZ else datetime.strptime(ws, "%Y-%m-%d")
    except:
        ws_dt = _current_week_start()
    weekly_picks = [p for p in h if p.get("date", "") >= ws]
    weekly_settled = [p for p in weekly_picks if p.get("settled")]
    weekly_w = sum(1 for p in weekly_settled if p["result"] == "W")
    weekly_l = sum(1 for p in weekly_settled if p["result"] == "L")
    weekly_profit = sum(p.get("profit") or 0 for p in weekly_picks if p.get("profit") is not None)
    weekly_bankroll_start = data.get("weekly_bankroll", 1000)
    weekly_bankroll = round(weekly_bankroll_start + weekly_profit, 2)
    hist_profit = sum(w.get("profit", 0) for w in data.get("weekly_history", [])) + weekly_profit
    cash_adj = data.get("cash_adjust", 0) or 0
    adj_bankroll = round(1000 + hist_profit - open_stakes + cash_adj, 2)
    adj_profit = round(total_profit + cash_adj, 2)
    return {
        "bankroll": adj_bankroll,
        "total": len(settled), "wins": len(wins), "losses": len(losses),
        "pct": round(len(wins) / len(settled) * 100, 1) if settled else 0,
        "profit": adj_profit,
        "roi": round(adj_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "open": len([p for p in h if not p.get("settled")]),
        "weekly_bankroll": weekly_bankroll,
        "weekly_profit": round(weekly_profit, 2),
        "weekly_wins": weekly_w, "weekly_losses": weekly_l,
        "weekly_start": ws,
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
                # Check if game was cancelled/postponed
                if g.get("status", {}).get("detailedState") in ("Postponed", "Cancelled", "Suspended"):
                    games.append({
                        "away_abbr": away_abbr,
                        "home_abbr": home_abbr,
                        "away_name": away_name,
                        "home_name": home_name,
                        "away_runs": 0,
                        "home_runs": 0,
                        "label": f"{away_abbr} @ {home_abbr}",
                        "cancelled": True,
                    })
                    continue
                if g.get("status", {}).get("codedGameState") not in ("F", "O"):
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
                if match.get("cancelled"):
                    profit = settle_pick(p["id"], "cancel")
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
                    line = None
                    if detail:
                        try:
                            line = float(detail.replace("o","").replace("u",""))
                        except:
                            pass
                    if line is not None and pick_side:
                        if total == line:
                            won = None  # Push
                        elif pick_side.lower() == "over":
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

PRED_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.json")

def log_predictions(picks, today=None):
    """Save all model predictions for the day to predictions_log.json."""
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(PRED_PATH) as f:
            data = json.load(f)
    except:
        data = {"predictions": []}

    existing_ids = {p["id"] for p in data["predictions"] if p.get("date") == today}
    new_count = 0
    for pick in picks:
        gid = pick.get("game_id", "")
        for mkt_key, label in [("moneyline", "ML"), ("spread_minus", "RL -1.5"),
                                ("spread_plus", "RL +1.5"), ("total", "O/U")]:
            entry = pick.get(mkt_key)
            if not entry:
                continue
            pick_name = entry.get("pick", "")
            if not pick_name:
                continue
            pid = f"{gid}_{mkt_key}"
            if pid in existing_ids:
                continue
            ha = pick.get("home_abbrev", "?")
            aa = pick.get("away_abbrev", "?")
            data["predictions"].append({
                "id": pid,
                "date": today,
                "game": f"{aa} @ {ha}",
                "away_abbrev": aa,
                "home_abbrev": ha,
                "market": label,
                "pick": entry.get("pick", ""),
                "prob": entry.get("prob", 0),
                "odds": entry.get("odds", "N/A"),
                "edge": entry.get("edge"),
                "detail": entry.get("detail", ""),
                "result": None,
                "settled": False,
            })
            new_count += 1
    if new_count > 0:
        with open(PRED_PATH, "w") as f:
            json.dump(data, f, indent=2)
    return new_count

def settle_predictions():
    """Settle unsettled predictions against MLB API results."""
    from datetime import datetime
    try:
        with open(PRED_PATH) as f:
            data = json.load(f)
    except:
        return 0, []

    unsettled = [p for p in data["predictions"] if not p.get("settled")]
    if not unsettled:
        return 0, []

    # Group by date
    dates = set(p["date"] for p in unsettled)
    settled_count = 0
    errors = []

    for dt in dates:
        games = _fetch_mlb_games(dt)
        if not games:
            continue
        for p in unsettled:
            pid = p["id"]
            # Match game
            gl = f"{p.get('away_abbrev','')} @ {p.get('home_abbrev','')}"
            game = None
            for g in games:
                if g["label"] == gl:
                    game = g
                    break
            if not game:
                continue

            market = p["market"]
            pick_side = p["pick"]
            detail = p.get("detail", "")
            away_runs = game["away_runs"]
            home_runs = game["home_runs"]

            try:
                won = None
                if market == "ML":
                    if pick_side == game["home_name"]:
                        won = home_runs > away_runs
                    else:
                        won = away_runs > home_runs
                elif market in ("RL -1.5", "RL +1.5"):
                    margin = home_runs - away_runs
                    if market == "RL -1.5":
                        won = (pick_side == game["home_name"] and margin >= 2) or \
                              (pick_side == game["away_name"] and margin <= -2)
                    else:
                        won = (pick_side == game["home_name"] and margin >= -1) or \
                              (pick_side == game["away_name"] and margin <= 1)
                elif market == "O/U":
                    total = home_runs + away_runs
                    line = None
                    if detail:
                        try:
                            line = float(detail.replace("o","").replace("u",""))
                        except:
                            pass
                    if line:
                        if pick_side.lower() == "over":
                            won = total > line
                        elif pick_side.lower() == "under":
                            won = total < line

                if won is not None and pick_side:
                    p["result"] = "W" if won else "L"
                    p["settled"] = True
                    settled_count += 1
            except Exception as e:
                errors.append(f"Pred #{pid}: {e}")

    if settled_count > 0:
        with open(PRED_PATH, "w") as f:
            json.dump(data, f, indent=2)
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
