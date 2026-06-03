"""
settle_and_notify.py — GitHub Actions: auto-settle + Telegram notification.
Usage: python3 settle_and_notify.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import auto_settle, settle_predictions, load_picks, get_pnl, REV_TEAM, MLB_API
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
NOTIFIED_PATH = os.path.join(os.path.dirname(__file__), "game_starts_notified.json")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado — salteando")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("  Notificación enviada a Telegram")
        else:
            print(f"  Error Telegram: {r.text}")
    except Exception as e:
        print(f"  Error Telegram: {e}")

def load_state():
    try:
        with open(NOTIFIED_PATH) as f:
            return json.load(f)
    except:
        return {"notified_starts": [], "scores": {}}

def save_state(state):
    with open(NOTIFIED_PATH, "w") as f:
        json.dump(state, f, indent=2)

def inning_icon(state):
    if state == "Top": return "🔝"
    if state == "Bottom": return "🔽"
    return ""

def get_todays_pick_games():
    """Return set of game labels (e.g. 'TEX @ STL') for today's registered picks."""
    try:
        data = load_picks()
        today = datetime.now().strftime("%Y-%m-%d")
        return {p["game"] for p in data["history"] if p.get("date") == today}
    except:
        return set()

def check_game_starts_and_scores():
    print("=== Verificando juegos en vivo ===")
    pick_games = get_todays_pick_games()
    if not pick_games:
        print("  Sin picks registrados hoy — salteando")
        return
    print(f"  Picks de hoy: {pick_games}")

    state = load_state()
    notified_starts = set(state.get("notified_starts", []))
    scores = state.get("scores", {})
    today = datetime.now().strftime("%Y-%m-%d")

    url = f"{MLB_API}/schedule?date={today}&sportId=1&hydrate=linescore,team"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"  Error MLB API: {r.status_code}")
            return
    except Exception as e:
        print(f"  Error MLB API: {e}")
        return

    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            state_code = g.get("status", {}).get("codedGameState", "")
            if state_code not in ("L", "I"):
                continue
            gid = str(g["gamePk"])
            away = g["teams"]["away"]
            home = g["teams"]["home"]
            away_name = away["team"]["name"]
            home_name = home["team"]["name"]
            away_abbr = REV_TEAM.get(away_name.lower(), away_name)
            home_abbr = REV_TEAM.get(home_name.lower(), home_name)
            label = f"{away_abbr} @ {home_abbr}"

            # Solo notificar juegos donde tengo picks registrados
            if label not in pick_games:
                continue

            away_runs = away.get("score", 0)
            home_runs = home.get("score", 0)

            linescore = g.get("linescore") or {}
            inning = linescore.get("currentInning")
            inn_state = linescore.get("inningState", "")
            inn_ord = linescore.get("currentInningOrdinal", f"{inning or ''}")

            # Game start notification
            if gid not in notified_starts:
                msg = f"⚾ *JUEGO INICIADO*\n{label}"
                print(f"  {label} — notificando inicio")
                send_telegram(msg)
                notified_starts.add(gid)

            # Score change detection
            prev = scores.get(gid)
            if prev is None or prev.get("away") != away_runs or prev.get("home") != home_runs:
                if prev is not None:
                    parts = []
                    if away_runs > prev.get("away", 0):
                        diff = away_runs - prev.get("away", 0)
                        parts.append(f"{away_abbr} ({'+' if diff > 0 else ''}{diff})")
                    if home_runs > prev.get("home", 0):
                        diff = home_runs - prev.get("home", 0)
                        parts.append(f"{home_abbr} ({'+' if diff > 0 else ''}{diff})")
                    who = ", ".join(parts)
                    icon = inning_icon(inn_state)
                    inn_display = f"{icon} {inn_ord}" if inn_ord else ""
                    msg = f"⚾ *CARRERA!* {label}\n{away_abbr} {away_runs} - {home_runs} {home_abbr}"
                    if inn_display:
                        msg += f"  ({inn_display})"
                    if who:
                        msg += f"\nAnotó: {who}"
                    print(f"  {label} — {away_runs}-{home_runs} (cambio)")
                    send_telegram(msg)
                scores[gid] = {"away": away_runs, "home": home_runs}

    state["notified_starts"] = list(notified_starts)
    state["scores"] = scores
    save_state(state)

def main():
    print("=== Auto-Settlement + Telegram ===")
    before = load_picks()
    pending_before = {p["id"] for p in before["history"] if not p.get("settled")}
    print(f"  Pendientes antes: {len(pending_before)}")

    count, errors = auto_settle()
    print(f"  Liquidados: {count}")
    if errors:
        for e in errors:
            print(f"  ⚠️ {e}")

    if count == 0:
        print("  Sin cambios — salteando notificación de resultados")
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            send_telegram("⚾ *MLB Picks Bot* ✅ Activo — sin juegos nuevos liquidados.")
    else:
        after = load_picks()
        pnl = get_pnl()
        lines = ["⚾ *Resultados MLB*"]
        for p in after["history"]:
            if p["id"] in pending_before and p.get("settled"):
                profit = p.get("profit", 0)
                icon = "✅" if p.get("result") == "W" else "❌"
                result = "GANADA" if p.get("result") == "W" else "PERDIDA"
                lines.append(f"{icon} *{p['game']}* → {p.get('market','?')} {p.get('team','?')} {icon} *{result}* (${profit:+.2f})")
        lines.append("")
        lines.append(f"💰 *Bankroll:* ${after['bankroll']:.2f}")
        lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
        lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")
        msg = "\n".join(lines)
        print(f"\nMensaje:\n{msg}")
        send_telegram(msg)

    pred_count, pred_errors = settle_predictions()
    if pred_count > 0:
        print(f"  Predicciones liquidadas: {pred_count}")
        if pred_errors:
            for e in pred_errors:
                print(f"  ⚠️ {e}")

    check_game_starts_and_scores()

if __name__ == "__main__":
    main()
