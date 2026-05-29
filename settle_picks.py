"""
settle_picks.py — liquidar picks pendientes después de los juegos.

Uso: python3 settle_picks.py list           # ver pendientes
     python3 settle_picks.py settle <id> W   # marcar pick como ganado
     python3 settle_picks.py settle <id> L   # marcar pick como perdido
     python3 settle_picks.py auto            # auto-liquidar (no impl.)
"""
import sys, requests
from datetime import datetime, timezone
from bankroll import load_picks, save_picks, get_pnl, today_checks

MLB_API = "https://statsapi.mlb.com/api/v1"

def list_pending():
    data = load_picks()
    pending = [p for p in data["history"] if not p.get("settled")]
    if not pending:
        print("No hay picks pendientes.")
        return
    print(f"\n{'ID':>3}  {'Fecha':<10}  {'Juego':<30}  {'Mercado':<12}  {'Prob':<6}  {'Cuota':<6}  {'Stake':<8}")
    print("-" * 85)
    for p in pending:
        print(f"{p['id']:>3}  {p['date']:<10}  {p['game']:<30}  {p['market']:<12}  {p['model_prob']:.0%}  ${p['odds']:<+4}  ${p['stake']:<.2f}")

def auto_settle():
    """Auto-settle: consulta MLB API para juegos de ayer y marca resultados."""
    print("Auto-liquidación — consultando MLB API...")
    yesterday = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{MLB_API}/schedule?date={yesterday}&sportId=1&hydrate=linescore,team"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print("  Error consultando MLB API")
            return
        data = r.json()
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("codedState") == "F":
                    away = g["teams"]["away"]["team"]["name"]
                    home = g["teams"]["home"]["team"]["name"]
                    a_runs = g["teams"]["away"].get("score", 0)
                    h_runs = g["teams"]["home"].get("score", 0)
                    label = f"{away} @ {home}"
                    winner = "away" if a_runs > h_runs else "home" if h_runs > a_runs else None
                    text = f"{away} {a_runs}-{h_runs} {home}"
                    games.append({
                        "label": label,
                        "winner_team": away if winner == "away" else home if winner == "home" else None,
                        "winner_side": "away" if winner == "away" else "home",
                        "a_runs": a_runs, "h_runs": h_runs,
                        "text": text,
                        "away_team": away, "home_team": home,
                    })
            for g in games:
                print(f"  {g['text']}")

        # Match pending picks to results
        data = load_picks()
        settled = 0
        for p in data["history"]:
            if p.get("settled"): continue
            for g in games:
                # Check if game matches (simplified: look for team name in game string)
                g_teams = g["label"].lower()
                p_game_teams = p["game"].lower()
                # Match by checking if teams are mentioned
                teams_in_pick = set(p_game_teams.split(" @ "))
                teams_in_game = set(g_teams.split(" @ "))
                if teams_in_pick == teams_in_game:
                    # Determine result based on market
                    result = None
                    p_team_lower = p["game"].split(" @ ")[0] if "away" in p["market"].lower() else \
                                  p["game"].split(" @ ")[-1] if "home" in p["market"].lower() else None
                    if p_team_lower:
                        # Moneyline or Run Line: check if the team matches winner
                        pass
                    break
        print(f"  Liquidados: {settled}")

    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 settle_picks.py list|settle <id> W|L|auto")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        list_pending()

    elif cmd == "settle" and len(sys.argv) >= 4:
        pick_id = int(sys.argv[2])
        result = sys.argv[3].upper()
        if result not in ("W", "L"):
            print("Resultado debe ser W o L")
            sys.exit(1)
        from bankroll import settle_pick
        profit = settle_pick(pick_id, result == "W")
        if profit is not None:
            print(f"Pick #{pick_id} → {'GANADA' if result == 'W' else 'PERDIDA'} (${profit:+.2f})")
            pnl = get_pnl()
            print(f"Bankroll actual: ${pnl['bankroll']:.2f}")
        else:
            print(f"Pick #{pick_id} no encontrado o ya liquidado")

    elif cmd == "auto":
        auto_settle()

    else:
        print("Comando desconocido")
