"""
morning_summary.py — Daily summary using predictions.py (same picks as web).
"""
import os, sys
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from predictions import generate_recommendations
from bankroll import load_picks, get_pnl, recommend_stake

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

_QUIET = False

def send_telegram(msg):
    if _QUIET:
        return
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado")
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("Notificacion enviada")
        else:
            print(f"Error Telegram: {r.text}")
    except Exception as e:
        print(f"Error Telegram: {e}")

def fmt_odds(odds):
    if odds is None: return "N/A"
    return f"{odds:+d}"

def main():
    print("=== Morning Summary — Usando predictions.py ===")
    today_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    data = load_picks()
    pnl = get_pnl()

    print("  Generando recomendaciones...")
    top = generate_recommendations()

    lines = [f"☀️ *Buenos dias! — {today}*\n"]
    lines.append(f"🏟️ *Recomendaciones del modelo*\n")

    if not top:
        lines.append("_Sin recomendaciones con valor positivo hoy._")
    else:
        for r in top:
            detail = f" {r['detail']}" if r["detail"] else ""
            flames = "🔥" if r["edge"] >= 8 else ("⭐" if r["edge"] >= 5 else "📝")
            stake_info = ""
            try:
                stake_amt = r.get("stake", 0)
                if stake_amt > 0:
                    stake_info = f"  💰 ${stake_amt:.0f}"
            except:
                pass
            lines.append(f"{flames} {r['game']}")
            lines.append(f"   {r['market']} {r['team']}{detail}")
            lines.append(f"   Prob: {r['prob']:.0f}%  |  Odds: {fmt_odds(r['odds'])}  |  Edge: +{r['edge']:.1f}%{stake_info}")
            lines.append("")

    lines.append("┅" * 10)
    lines.append(f"💰 *Bankroll:* ${data['bankroll']:.2f}")
    lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
    lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")

    msg = "\n".join(lines)
    print(f"\nMensaje:\n{msg}")
    send_telegram(msg)
    return top

if __name__ == "__main__":
    h = datetime.now(TZ).hour
    if h < 6 or h >= 12:
        print(f"Fuera de horario matutino ({h}:00 Chihuahua) — salteando")
    else:
        try:
            main()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error en morning_summary: {e}\n{tb}")
            try:
                send_telegram(f"❌ Error en morning_summary: {e}")
            except: pass
