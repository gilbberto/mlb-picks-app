"""
settle_and_notify.py — GitHub Actions: auto-settle + Telegram notification.
Usage: python3 settle_and_notify.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import auto_settle, load_picks, get_pnl
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

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
        print("  Sin cambios — salteando notificación")
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            send_telegram("⚾ *MLB Picks Bot* ✅ Activo — sin juegos nuevos liquidados.")
        return

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

if __name__ == "__main__":
    main()
