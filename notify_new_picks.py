"""
notify_new_picks.py — Railway: check for new high-edge picks and notify.
Runs every ~2h during game hours. Only sends Telegram if new 🔥 picks appear.
"""
import os, sys, json, hashlib

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CWD)

COMMAND_FILE = os.path.join(CWD, ".notified_new_picks.json")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def _load_sent():
    if not os.path.isfile(COMMAND_FILE):
        return []
    try:
        with open(COMMAND_FILE) as f:
            return json.load(f)
    except:
        return []

def _save_sent(ids):
    with open(COMMAND_FILE, "w") as f:
        json.dump(ids, f)

def _pick_id(p):
    raw = f"{p['game']}|{p['market']}|{p['team']}|{p.get('detail','')}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def _send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("  Telegram no configurado")
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    if r.status_code == 200:
        print("  Notificacion enviada")
    else:
        print(f"  Error Telegram: {r.text}")

def check():
    import morning_summary as ms
    ms._QUIET = True
    from predictions import _MODELS_LOADED

    if not _MODELS_LOADED:
        print("  Modelos no cargados — salteando")
        return

    top = ms.main()
    if not top:
        print("  Sin recomendaciones con valor")
        return

    sent = _load_sent()
    new_picks = []
    for p in top:
        pid = _pick_id(p)
        if p["edge"] is not None and p["edge"] >= 8 and pid not in sent:
            new_picks.append(p)

    if not new_picks:
        print("  Sin nuevos picks 🔥")
        return

    print(f"  Nuevos picks 🔥: {len(new_picks)}")
    from bankroll import recommend_stake, get_pnl
    br = get_pnl()["bankroll"]

    lines = ["🆕 *NUEVOS PICKS RECOMENDADOS*\n"]
    for p in new_picks:
        detail = f" {p['detail']}" if p.get("detail") else ""
        stake_amt = 0
        try:
            stake_amt, _, _ = recommend_stake(p["prob"]/100, p["odds"], bankroll=br)
        except:
            pass
        stake_str = f"  💰 ${stake_amt:.0f}" if stake_amt > 0 else ""
        lines.append(f"🔥 {p['game']}")
        lines.append(f"   {p['market']} {p['team']}{detail}")
        lines.append(f"   Prob: {p['prob']:.0f}%  |  Edge: +{p['edge']:.1f}%{stake_str}")

        if len(new_picks) == 1:
            reason = p.get("reason", "")
            if reason:
                lines.append(f"   {reason[:200]}")

    msg = "\n".join(lines)
    _send_telegram(msg)

    # Update sent list
    all_ids = sent + [_pick_id(p) for p in new_picks]
    _save_sent(all_ids)
    print(f"  Guardados {len(all_ids)} picks notificados")

if __name__ == "__main__":
    try:
        check()
    except Exception as e:
        import traceback
        print(f"Error en notify_new_picks: {e}\n{traceback.format_exc()}")
