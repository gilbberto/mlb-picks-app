"""
settle_and_notify.py — GitHub Actions: auto-settle + Telegram notification.
Usage: python3 settle_and_notify.py
Envs: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json, math, base64
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
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
NOTIFIED_PATH = os.path.join(os.path.dirname(__file__), "game_starts_notified.json")
TG_OFFSET_PATH = os.path.join(os.path.dirname(__file__), ".telegram_offset")
PICKS_PATH = os.path.join(os.path.dirname(__file__), "picks.json")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado — salteando")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        print(f"  Enviando a Telegram (len={len(msg)} chars)...")
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("  Notificación enviada a Telegram")
            return True
        else:
            print(f"  Error Telegram ({r.status_code}): {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  Error Telegram: {e}")
        return False

def _load_tg_offset():
    try:
        with open(TG_OFFSET_PATH) as f:
            return int(f.read().strip())
    except:
        return 0

def _save_tg_offset(offset):
    with open(TG_OFFSET_PATH, "w") as f:
        f.write(str(offset))

def _sync_picks_from_github():
    """Fetch latest picks.json from GitHub to get fresh bankroll."""
    url = f"https://api.github.com/repos/gilbberto/mlb-picks-app/contents/picks.json?ref=main"
    try:
        headers = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"]).decode()
            with open(PICKS_PATH, "w") as f:
                f.write(content)
    except:
        pass

def _build_resultados_response():
    """Build current game results message for registered picks."""
    try:
        _sync_picks_from_github()
        data = load_picks()
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        picks = [p for p in data["history"] if p.get("date") == today]
        if not picks:
            return "📋 No hay picks registrados hoy."
        lines = ["📋 *Resultados del Día*\n"]
        url = f"{MLB_API}/schedule?date={today}&sportId=1&hydrate=linescore,team"
        r = requests.get(url, timeout=15)
        games_map = {}
        if r.status_code == 200:
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    away = g["teams"]["away"]
                    home = g["teams"]["home"]
                    away_abbr = REV_TEAM.get(away["team"]["name"].lower(), away["team"]["name"])
                    home_abbr = REV_TEAM.get(home["team"]["name"].lower(), home["team"]["name"])
                    label = f"{away_abbr} @ {home_abbr}"
                    sc = g.get("status", {}).get("codedGameState", "")
                    sd = g.get("status", {}).get("detailedState", "Programado")
                    ls = g.get("linescore", {})
                    inning = ""
                    if sc in ("F", "O"):
                        inning = "Final"
                    elif sc in ("L", "I"):
                        inn = ls.get("currentInningOrdinal", "")
                        side = ls.get("inningState", "")
                        inning = f"{'🔝' if side=='Top' else '🔽'} {inn}" if inn else sd
                    games_map[label] = {
                        "state": sc, "state_str": sd, "inning": inning,
                        "away_runs": away.get("score", 0),
                        "home_runs": home.get("score", 0),
                    }
        for p in picks:
            gl = p.get("game", "")
            market = p.get("market", "")
            team = p.get("team", "")
            detail = p.get("detail", "")
            lp = f"{market} {team}" + (f" {detail}" if detail else "")
            if p.get("settled"):
                icon = "✅" if p.get("result") == "W" else "❌"
                result = "GANADA" if p.get("result") == "W" else "PERDIDA"
                profit = p.get("profit") or 0
                lines.append(f"{icon} {gl} → {lp}: *{result}* (${profit:+.2f})")
            elif gl in games_map and games_map[gl]["state"] in ("L", "I"):
                gm = games_map[gl]
                inn = gm.get("inning", "")
                lines.append(f"⚾ {gl} → {lp}: *{gm['away_runs']}-{gm['home_runs']}* {inn}")
            elif gl in games_map and games_map[gl]["state"] in ("F", "O"):
                gm = games_map[gl]
                lines.append(f"⏳ {gl} → {lp}: *Final* {gm['away_runs']}-{gm['home_runs']} (pendiente)")
            else:
                gs = games_map.get(gl, {}).get("state_str", "Programado")
                lines.append(f"⏳ {gl} → {lp}: {gs}")
        pnl = get_pnl()
        lines.append(f"\n💰 *Bankroll (semanal):* ${pnl['weekly_bankroll']:.2f}")
        lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error al obtener resultados: {e}"

def _check_telegram_commands():
    """Poll Telegram for user commands like /resultados."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        offset = _load_tg_offset()
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        params = {"timeout": 5, "offset": offset + 1}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return
        max_id = offset
        for update in r.json().get("result", []):
            update_id = update.get("update_id", 0)
            if update_id > max_id:
                max_id = update_id
            msg = update.get("message", {}) or update.get("callback_query", {}).get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text = (msg.get("text") or "").strip().lower()
            if not chat_id or chat_id != int(CHAT_ID):
                continue
            if text in ("resultados", "/resultados"):
                resp = _build_resultados_response()
                send_telegram(resp)
            elif text in ("/status", "/diagnostico"):
                send_telegram(_cmd_status(text == "/diagnostico"))
            elif text == "/reiniciar_worker":
                msg = _cmd_restart("449aff70-31f8-4e67-88ab-c6ccedcc1546", "Worker")
                send_telegram(msg)
            elif text == "/reiniciar_web":
                msg = _cmd_restart("e5af2645-5349-4176-a305-419ce60353da", "Web")
                send_telegram(msg)
            elif text == "/reiniciar_webhook":
                msg = _cmd_restart("e722f196-dd7f-48f9-9654-2c9335ad0c0f", "Webhook")
                send_telegram(msg)
            elif text in ("/rendimiento", "/modelo"):
                try:
                    from predictions import compute_model_stats
                    send_telegram(compute_model_stats())
                except Exception as e:
                    send_telegram(f"❌ Error: {e}")
        if max_id > offset:
            _save_tg_offset(max_id)
    except Exception as e:
        print(f"  Error polling Telegram: {e}")

def _railway_headers():
    tok = os.environ.get("RAILWAY_TOKEN", "")
    if not tok:
        return None
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

def _railway_graphql(query):
    headers = _railway_headers()
    if not headers:
        return None
    try:
        r = requests.post("https://backboard.railway.app/graphql/v2", json={"query": query}, headers=headers, timeout=15)
        return r.json()
    except:
        return None

def _cmd_status(verbose=False):
    q = '{ project(id: "aea9b11d-3919-4278-b429-77dc91ebadb9") { services { edges { node { name serviceInstances { edges { node { activeDeployments { status } } } } } } } } }'
    data = _railway_graphql(q)
    if not data:
        return "❌ No pude conectar con Railway API"
    lines = ["🚦 *Estado del Sistema*\n"]
    ok = True
    for e in data.get("data", {}).get("project", {}).get("services", {}).get("edges", []):
        srv = e["node"]
        statuses = [dep["status"] for inst in srv["serviceInstances"]["edges"] for dep in inst["node"]["activeDeployments"]]
        s = statuses[0] if statuses else "DOWN"
        icon = "✅" if s == "SUCCESS" else ("⚠️" if s in ("QUEUED","BUILDING") else "❌")
        if s != "SUCCESS":
            ok = False
        lines.append(f"{icon} *{srv['name']}*: {s}")
    lines.append(f"\n{'✅ Todo bien' if ok else '⚠️ Hay servicios caídos'}")
    if verbose:
        lines.append("\nComandos disponibles:")
        lines.append("/reiniciar-worker — Reinicia el Worker")
        lines.append("/reiniciar-web — Reinicia la Web")
        lines.append("/reiniciar-webhook — Reinicia el Webhook")
    return "\n".join(lines)

def _cmd_restart(service_id, name):
    q = f'mutation {{ serviceInstanceRedeploy(environmentId: "4d3a8996-f03c-4ab3-bcc4-dbb394d6f057", serviceId: "{service_id}") }}'
    data = _railway_graphql(q)
    if data and data.get("data", {}).get("serviceInstanceRedeploy"):
        return f"🔄 *{name}* reiniciándose..."
    return f"❌ Error al reiniciar {name}"

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
        if team == "Over":
            if total > line:
                return 100.0
        else:
            if total > line:
                return 0.0
        remaining = max(0, 9 - inning) * 0.5  # ~0.5 runs per remaining inning
        prob = 1.0 / (1.0 + math.exp(-(total - line) / max(remaining, 0.5)))
        return round(prob * 100, 1)

    return None

def _game_ended_msg(label, away_abbr, home_abbr, away_runs, home_runs):
    """Build juego terminado message with pick results."""
    picks = get_picks_for_game(label)
    result_lines = []
    for pk in picks:
        market = pk.get("market", "")
        team = pk.get("team", "")
        detail = pk.get("detail", "")
        settled = pk.get("settled")
        if settled:
            icon = "✅" if pk.get("result") == "W" else "❌"
            result = "GANADA" if pk.get("result") == "W" else "PERDIDA"
            profit = pk.get("profit") or 0
            lp = f"{market} {team}" + (f" {detail}" if detail else "")
            result_lines.append(f"{icon} {lp}: *{result}* (${profit:+.2f})")
        else:
            lp = f"{market} {team}" + (f" {detail}" if detail else "")
            result_lines.append(f"⏳ {lp}: pendiente")
    parts = [f"⚾ *JUEGO TERMINADO*\n{label}", f"{away_abbr} {away_runs} - {home_runs} {home_abbr}"]
    if result_lines:
        parts.append("")
        parts.extend(result_lines)
    return "\n".join(parts)

def check_game_starts_and_scores():
    print("=== Verificando juegos en vivo ===")
    pick_games = get_todays_pick_games()
    if not pick_games:
        print("  Sin picks registrados hoy — salteando")
        return
    print(f"  Picks de hoy: {pick_games}")

    state = load_state()
    notified_starts = set(state.get("notified_starts", []))
    notified_ended = set(state.get("notified_ended", []))
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
            gid = str(g["gamePk"])
            away = g["teams"]["away"]
            home = g["teams"]["home"]
            away_name = away["team"]["name"]
            home_name = home["team"]["name"]
            away_abbr = REV_TEAM.get(away_name.lower(), away_name)
            home_abbr = REV_TEAM.get(home_name.lower(), home_name)
            label = f"{away_abbr} @ {home_abbr}"

            if label not in pick_games:
                continue

            away_runs = away.get("score", 0)
            home_runs = home.get("score", 0)

            # Juego terminado
            if state_code in ("F", "O"):
                if gid not in notified_ended:
                    msg = _game_ended_msg(label, away_abbr, home_abbr, away_runs, home_runs)
                    print(f"  {label} — JUEGO TERMINADO ({away_runs}-{home_runs})")
                    if send_telegram(msg):
                        notified_ended.add(gid)
                continue

            if state_code not in ("L", "I"):
                continue

            # Game start notification (solo una vez)
            if gid not in notified_starts:
                msg = f"⚾ *JUEGO INICIADO*\n{label}"
                print(f"  {label} — notificando inicio")
                if send_telegram(msg):
                    notified_starts.add(gid)

    state["notified_starts"] = list(notified_starts)
    state["notified_ended"] = list(notified_ended)
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

    new_ids = {p["id"] for p in load_picks()["history"] if p["id"] in pending_before and p.get("settled")}
    if count == 0:
        print("  Sin cambios — salteando notificación de resultados")
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            send_telegram("⚾ *MLB Picks Bot* ✅ Activo — sin juegos nuevos liquidados.")
    elif new_ids:
        # Avoid notifying same pick across parallel zombie workflows
        already_notified = set(load_state().get("notified_settled", []))
        fresh_ids = new_ids - already_notified
        if not fresh_ids:
            print("  IDs ya notificados — salteando")
        else:
            after = load_picks()
            pnl = get_pnl()
            lines = ["⚾ *Resultados MLB*"]
            for p in after["history"]:
                if p["id"] in fresh_ids:
                    profit = p.get("profit") or 0
                    icon = "✅" if p.get("result") == "W" else "❌"
                    result = "GANADA" if p.get("result") == "W" else "PERDIDA"
                    lines.append(f"{icon} *{p['game']}* → {p.get('market','?')} {p.get('team','?')} {icon} *{result}* (${profit:+.2f})")
            lines.append("")
            lines.append(f"💰 *Bankroll (semanal):* ${pnl['weekly_bankroll']:.2f}")
            lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
            lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")
            msg = "\n".join(lines)
            print(f"\nMensaje:\n{msg}")
            if send_telegram(msg):
                # Mark these IDs as notified so other workflows skip
                st = load_state()
                st.setdefault("notified_settled", [])
                st["notified_settled"] = list(set(st["notified_settled"]) | fresh_ids)
                save_state(st)

    pred_count, pred_errors = settle_predictions()
    if pred_count > 0:
        print(f"  Predicciones liquidadas: {pred_count}")
        if pred_errors:
            for e in pred_errors:
                print(f"  ⚠️ {e}")

    # Log all predictions for today (runs once per day via dedup)
    try:
        from predictions import log_all_todays_predictions
        logged = log_all_todays_predictions()
        if logged > 0:
            print(f"  Predicciones guardadas hoy: {logged}")
    except Exception as e:
        print(f"  Error guardando predicciones: {e}")

    check_game_starts_and_scores()
    _check_telegram_commands()

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
    for f in ("picks.json", "game_starts_notified.json", "predictions_log.json", ".morning_sent", ".telegram_offset"):
        fp = os.path.join(cwd, f)
        if os.path.isfile(fp):
            subprocess.run(["git", "add", f], capture_output=True, cwd=cwd)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True, cwd=cwd)
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "sync state"], capture_output=True, cwd=cwd)
        subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], capture_output=True, cwd=cwd)
        subprocess.run(["git", "push"], capture_output=True, cwd=cwd)

if __name__ == "__main__":
    if os.environ.get("GITHUB_ACTIONS"):
        _git_pull()
    main()
    if os.environ.get("GITHUB_ACTIONS"):
        _git_commit()
