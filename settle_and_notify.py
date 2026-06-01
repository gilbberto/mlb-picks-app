"""
settle_and_notify.py — GitHub Actions: auto-settle + Telegram notification.
Usage: python3 settle_and_notify.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import auto_settle, load_picks, get_pnl, REV_TEAM, MLB_API
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

def load_notified():
    try:
        with open(NOTIFIED_PATH) as f:
            return set(json.load(f))
    except:
        return set()

def save_notified(games):
    with open(NOTIFIED_PATH, "w") as f:
        json.dump(sorted(games), f)

def check_game_starts():
    print("=== Verificando juegos en vivo ===")
    notified = load_notified()
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

    new_live = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            state = g.get("status", {}).get("codedGameState", "")
            if state not in ("L", "I"):
                continue
            gid = str(g["gamePk"])
            if gid in notified:
                continue
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            away_abbr = REV_TEAM.get(away.lower(), away)
            home_abbr = REV_TEAM.get(home.lower(), home)
            new_live.append((gid, away_abbr, home_abbr, away, home))

    if not new_live:
        print("  Sin juegos nuevos en vivo")
        return

    for gid, away_abbr, home_abbr, away_name, home_name in new_live:
        msg = f"⚾ *JUEGO INICIADO*\n{away_abbr} @ {home_abbr}\n{away_name} vs {home_name}"
        print(f"  {away_abbr} @ {home_abbr} — notificando")
        send_telegram(msg)
        notified.add(gid)

    save_notified(notified)

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
                icon = "✅" if p.get("won") else "❌"
                result = "GANADA" if p.get("won") else "PERDIDA"
                lines.append(f"{icon} *{p['game']}* → {p.get('market','?')} {p.get('team','?')} {icon} *{result}* (${profit:+.2f})")
        lines.append("")
        lines.append(f"💰 *Bankroll:* ${after['bankroll']:.2f}")
        lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
        lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")
        msg = "\n".join(lines)
        print(f"\nMensaje:\n{msg}")
        send_telegram(msg)

    check_game_starts()

if __name__ == "__main__":
    main()
