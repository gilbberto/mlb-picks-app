"""
settle_and_notify.py — GitHub Actions: auto-settle + Telegram notification.
Usage: python3 settle_and_notify.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json, math
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import auto_settle, settle_predictions, load_picks, get_pnl, REV_TEAM, MLB_API
import requests

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
NOTIFIED_PATH = os.path.join(os.path.dirname(__file__), "game_starts_notified.json")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado — salteando")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        print(f"  Enviando a Telegram (len={len(msg)} chars)...")
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("  Notificación enviada a Telegram")
        else:
            print(f"  Error Telegram ({r.status_code}): {r.text[:200]}")
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
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        return {p["game"] for p in data["history"] if p.get("date") == today}
    except:
        return set()

def get_picks_for_game(game_label):
    """Return list of today's picks matching a game label."""
    try:
        data = load_picks()
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        return [p for p in data["history"] if p.get("date") == today and p.get("game") == game_label]
    except:
        return []

def estimate_team_win_pct(run_diff, inning):
    """Win probability for the team with positive run_diff based on inning."""
    k = {1: 0.30, 2: 0.32, 3: 0.35, 4: 0.38, 5: 0.42,
         6: 0.47, 7: 0.53, 8: 0.62, 9: 0.75}.get(min(inning, 9), 0.80)
    return 1.0 / (1.0 + math.exp(-k * run_diff))

def pick_win_pct(pick, away_runs, home_runs, away_abbr, home_abbr, inning):
    """Return estimated win % for this pick given current score."""
    team = pick.get("team", "")
    market = pick.get("market", "")
    detail = pick.get("detail", "")
    team_lower = team.lower()

    # Determine if pick team is home or away
    is_home = any(name in team_lower for name in [home_abbr.lower(), pick.get("game","").split("@")[1].strip().lower()])
    team_runs = home_runs if is_home else away_runs
    opp_runs = away_runs if is_home else home_runs
    run_diff = team_runs - opp_runs

    if market == "ML":
        return round(estimate_team_win_pct(run_diff, inning) * 100 if run_diff >= 0 else (1 - estimate_team_win_pct(-run_diff, inning)) * 100, 1)

    elif market in ("RL -1.5", "RL +1.5"):
        spread = -1.5 if market == "RL -1.5" else 1.5
        eff_diff = run_diff - spread
        # Rough approximation: use team win prob scaled
        base = estimate_team_win_pct(abs(eff_diff), inning) if eff_diff >= 0 else 1 - estimate_team_win_pct(abs(eff_diff), inning)
        return round(base * 100, 1)

    elif market == "O/U":
        try:
            line = float(detail.replace("o","").replace("u",""))
        except:
            return None
        total = away_runs + home_runs
        remaining = max(0, 9 - inning) * 0.5  # ~0.5 runs per remaining inning
        expected = total + remaining
        if team == "Over":
            prob = 1.0 / (1.0 + math.exp(-(total - line) / max(remaining, 0.5)))
        else:
            prob = 1.0 / (1.0 + math.exp((total - line) / max(remaining, 0.5)))
        return round(prob * 100, 1)

    return None

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
    today = datetime.now(TZ).strftime("%Y-%m-%d")

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
            changed = prev is None or prev.get("away") != away_runs or prev.get("home") != home_runs
            print(f"    prev={prev}, current={away_runs}-{home_runs}, changed={changed}")
            if changed:
                msg_parts = [f"⚾ *CARRERA!* {label}", f"{away_abbr} {away_runs} - {home_runs} {home_abbr}"]
                if inn_ord:
                    msg_parts.append(f"  ({inning_icon(inn_state)} {inn_ord})")
                if prev is not None:
                    sc_lines = []
                    if away_runs > prev.get("away", 0):
                        diff = away_runs - prev.get("away", 0)
                        sc_lines.append(f"{away_abbr} ({'+' if diff > 0 else ''}{diff})")
                    if home_runs > prev.get("home", 0):
                        diff = home_runs - prev.get("home", 0)
                        sc_lines.append(f"{home_abbr} ({'+' if diff > 0 else ''}{diff})")
                    if sc_lines:
                        msg_parts.append("Anotó: " + ", ".join(sc_lines))
                picks_in_game = get_picks_for_game(label)
                if picks_in_game:
                    wp_lines = []
                    for pk in picks_in_game:
                        wp = pick_win_pct(pk, away_runs, home_runs, away_abbr, home_abbr, inning or 1)
                        if wp is not None:
                            pm = pk.get("market", "")
                            pt = pk.get("team", "")
                            pd = pk.get("detail", "")
                            lp = f"{pm} {pt}" + (f" {pd}" if pd else "")
                            wp_lines.append(f"  {lp}: {wp:.0f}%")
                    if wp_lines:
                        msg_parts.append("\n".join(wp_lines))
                msg = "\n".join(msg_parts)
                print(f"  {label} — {away_runs}-{home_runs} {'(inicial)' if prev is None else '(cambio)'}")
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

    # Morning summary entre 6-11 AM (una vez por dia)
    h = datetime.now(TZ).hour
    if 6 <= h <= 11:
        flag = os.path.join(os.path.dirname(__file__), ".morning_sent")
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        try:
            already = open(flag).read().strip() == today
        except:
            already = False
        if not already:
            print("  Morning summary pendiente — ejecutando...")
            import subprocess
            r = subprocess.run(["python3", "morning_summary.py"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            if r.returncode == 0:
                with open(flag, "w") as f:
                    f.write(today)
                print("  Morning summary enviado")
            else:
                print(f"  Error morning summary: {r.stderr[:200]}")

def _git_pull():
    """Sync local state from git before each cycle."""
    import subprocess
    subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], capture_output=True, cwd=os.path.dirname(__file__))

def _git_commit():
    """Save game_starts_notified.json to git after each cycle."""
    import subprocess
    cwd = os.path.dirname(__file__)
    subprocess.run(["git", "config", "user.name", "MLB Picks Bot"], capture_output=True, cwd=cwd)
    subprocess.run(["git", "config", "user.email", "bot@mlb-picks.local"], capture_output=True, cwd=cwd)
    subprocess.run(["git", "add", "game_starts_notified.json", ".morning_sent"], capture_output=True, cwd=cwd)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True, cwd=cwd)
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "sync state"], capture_output=True, cwd=cwd)
        subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], capture_output=True, cwd=cwd)
        subprocess.run(["git", "push"], capture_output=True, cwd=cwd)

if __name__ == "__main__":
    import time
    in_actions = bool(os.environ.get("GITHUB_ACTIONS"))
    if in_actions:
        _git_pull()
    main()
    if in_actions:
        _git_commit()
    for _ in range(10):
        h = datetime.now(TZ).hour
        if h >= 22 or h < 6:
            sleep_min = 120
        elif 12 <= h < 18:
            sleep_min = 30
        else:
            sleep_min = 15
        print(f"  Durmiendo {sleep_min} min (hora Chihuahua: {h})...")
        time.sleep(sleep_min * 60)
        if in_actions:
            _git_pull()
        main()
        if in_actions:
            _git_commit()
