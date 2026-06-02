"""
morning_summary.py — GitHub Actions: daily summary of today's picks via Telegram.
Usage: python3 morning_summary.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import load_picks, get_pnl
from datetime import datetime, timezone, timedelta
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("Notificación enviada")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    data = load_picks()
    pnl = get_pnl()

    todays_picks = [p for p in data["history"] if p.get("date") == today]

    # Predictions log (model predictions for today)
    pred_path = os.path.join(os.path.dirname(__file__), "predictions_log.json")
    try:
        with open(pred_path) as f:
            pred_data = json.load(f)
        todays_preds = [p for p in pred_data.get("predictions", []) if p.get("date") == today]
    except:
        todays_preds = []

    pred_line = ""
    if todays_preds:
        settled = [p for p in todays_preds if p.get("settled")]
        pending = [p for p in todays_preds if not p.get("settled")]
        pw = sum(1 for p in settled if p["result"] == "W")
        pl = sum(1 for p in settled if p["result"] == "L")
        pt = pw + pl
        if pt > 0:
            pred_line = f"📊 *Modelo:* {pw}-{pl} ({round(pw/pt*100)}%) en {pt} liquidados"
        if pending:
            pred_line += f" | {len(pending)} pendientes"

    if not todays_picks:
        if pred_line:
            msg = f"☀️ *Buenos días! — {today}*\n{pred_line}\n\nSin picks registrados. Revisa las predicciones del modelo."
        else:
            msg = f"☀️ *Buenos días! — {today}*\n\nNo hay actividad para hoy."
    else:
        lines = [f"☀️ *Resumen del día — {today}*\n"]
        total_stake = 0
        for p in todays_picks:
            market = p.get("market", "?")
            team = p.get("team", "?")
            odds = p.get("odds", 0)
            stake = p.get("stake", 0)
            total_stake += stake
            detail = p.get("detail", "")
            if detail:
                team = f"{team} {detail.lstrip('ou~')}"
            lines.append(f"• *{p['game']}* → {market} {team}  ${odds:+d}  (${stake:.0f})")
        lines.append("")
        lines.append(f"💰 *Bankroll:* ${data['bankroll']:.2f}")
        lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
        lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")
        lines.append(f"🎯 *Stake total hoy:* ${total_stake:.2f}")
        if pred_line:
            lines.append("")
            lines.append(pred_line)
        msg = "\n".join(lines)

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    main()
