"""
bankroll.py — Kelly Criterion sizing + P&L tracker.
Lee/escribe picks.json para llevar control de resultados.
"""
import json, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "picks.json")

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

def add_pick(date, game, market, model_prob, odds, stake, bankroll, label=""):
    data = load_picks()
    pick = {
        "id": len(data["history"]) + 1,
        "date": date,
        "game": game,
        "market": market,
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
