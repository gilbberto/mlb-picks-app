"""
morning_summary.py — GitHub Actions: daily summary of model predictions via Telegram.
Corre a las 6:00 AM Chihuahua (12:00 UTC).
Genera predicciones del modelo y muestra las mejores recomendaciones del dia.
"""
import os, sys, json, math, pickle, requests, numpy as np
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from bankroll import load_picks, get_pnl, REV_TEAM, MLB_API, calibrate_ml, calibrate_rl, recommend_stake

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "3988754e84aac800a8ee2eeca88cb085")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CURRENT_SEASON = 2026

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

BASE = os.path.join(os.path.dirname(__file__), "")
CACHE = {}

def cget(url, ttl=300):
    now = datetime.now().timestamp()
    if url in CACHE and now - CACHE[url]["ts"] < ttl:
        return CACHE[url]["data"]
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            CACHE[url] = {"data": r.json(), "ts": now}
            return r.json()
    except:
        pass
    return None

def safe_float(v, d=0.0):
    if v is None: return d
    try: return float(v)
    except: return d

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram no configurado")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code == 200:
            print("Notificacion enviada")
        else:
            print(f"Error Telegram: {r.text}")
    except Exception as e:
        print(f"Error Telegram: {e}")

def load_models():
    global _xgb_hw, _xgb_rd, _xgb_tot, _cols, _MODELS_LOADED
    try:
        import xgboost as xgb
        with open(BASE + "xgb_hw.pkl", "rb") as f: _xgb_hw = pickle.load(f)
        with open(BASE + "xgb_rd.pkl", "rb") as f: _xgb_rd = pickle.load(f)
        with open(BASE + "xgb_tot.pkl", "rb") as f: _xgb_tot = pickle.load(f)
        with open(BASE + "xgb_cols.pkl", "rb") as f: _cols = pickle.load(f)
        _MODELS_LOADED = True
        print("  Modelos XGBoost cargados")
    except Exception as e:
        print(f"  Error cargando modelos: {e}")
        _MODELS_LOADED = False

load_models()

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

def fetch_todays_schedule():
    today = datetime.now(TZ).strftime("%m/%d/%Y")
    data = cget(f"{MLB_API_BASE}/schedule?sportId=1&date={today}&hydrate=probablePitcher", ttl=600)
    games = []
    if data:
        for de in data.get("dates", []):
            for g in de.get("games", []):
                games.append(g)
    return games

def fetch_team_stats(tid):
    data = cget(f"{MLB_API_BASE}/teams/{tid}/stats?season={CURRENT_SEASON}&group=hitting,pitching&stats=season", ttl=3600)
    result = {"hitting": {}, "pitching": {}}
    if data:
        for sg in data.get("stats", []):
            g = sg.get("group", {}).get("displayName", "").lower()
            splits = sg.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                if g == "hitting":
                    result["hitting"] = {k: safe_float(s.get(k)) for k in ["avg","runs","hr","obp","slg","ops","hits","strikeOuts","baseOnBalls"]}
                elif g == "pitching":
                    result["pitching"] = {k: safe_float(s.get(k)) for k in ["era","whip","runs","strikeouts","baseOnBalls","homeRuns"]}
    return result

def fetch_recent_games(tid, ng=20):
    today = datetime.now(TZ)
    end = today.strftime("%m/%d/%Y")
    start = (today - timedelta(days=45)).strftime("%m/%d/%Y")
    data = cget(f"{MLB_API_BASE}/schedule?sportId=1&teamId={tid}&startDate={start}&endDate={end}", ttl=600)
    games = []
    if data:
        for de in data.get("dates", []):
            for g in de.get("games", []):
                if g.get("status", {}).get("codedGameState") == "F":
                    games.append(g)
    return games[-ng:]

def compute_form(games, tid):
    if not games:
        return {"wp": 0.5, "rs": 4.5, "ra": 4.5, "rest": 3}
    games_sorted = sorted(games, key=lambda x: x.get("gameDate", ""), reverse=True)
    w, rs, ra = 0, [], []
    for i, g in enumerate(games_sorted):
        t = g["teams"]
        is_h = t["home"]["team"]["id"] == tid
        if not is_h and t["away"]["team"]["id"] != tid:
            continue
        ms = safe_float(t["home"]["score"] if is_h else t["away"]["score"])
        os_ = safe_float(t["away"]["score"] if is_h else t["home"]["score"])
        rs.append(ms); ra.append(os_)
        if ms > os_: w += 1
    n = len(rs) or 1
    rest = 3
    if games_sorted:
        try:
            ld = datetime.strptime(games_sorted[0].get("gameDate","").split("T")[0], "%Y-%m-%d")
            rest = max((datetime.now().date() - ld.date()).days, 1)
        except:
            pass
    return {"wp": w/n, "rs": np.mean(rs) if rs else 4.5, "ra": np.mean(ra) if ra else 4.5, "rest": rest}

def fetch_pitcher_stats(pid):
    if not pid: return {}
    data = cget(f"{MLB_API_BASE}/people/{pid}/stats?stats=season&season={CURRENT_SEASON}&group=pitching", ttl=3600)
    if not data: return {}
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits: return {}
    s = splits[0].get("stat", {})
    ip = s.get("inningsPitched", "0")
    ip_val = 0
    if isinstance(ip, str) and "." in ip:
        parts = ip.split(".")
        ip_val = int(parts[0]) + int(parts[1]) / 3.0 if len(parts) > 1 else float(parts[0])
    else:
        ip_val = float(ip or 0)
    hr = safe_float(s.get("homeRuns")); bb = safe_float(s.get("baseOnBalls"))
    so = safe_float(s.get("strikeOuts")); h = safe_float(s.get("hits"))
    ab = safe_float(s.get("atBats")); sf = safe_float(s.get("sacFlies"))
    hbp = safe_float(s.get("hitByPitch")); go = safe_float(s.get("groundOuts")); ao = safe_float(s.get("airOuts"))
    fip = 3.10
    if ip_val > 0:
        fip = ((13 * hr) + (3 * (bb + hbp)) - (2 * so)) / ip_val + 3.10
    babip = 0.300
    if (ab - so - hr + sf) > 0:
        babip = (h - hr) / (ab - so - hr + sf)
    kbb = so / bb if bb > 0 else so
    gb_rate = go / (go + ao) if (go + ao) > 0 else 0.44
    return {"era": safe_float(s.get("era")), "whip": safe_float(s.get("whip")), "ip": ip_val,
            "k9": safe_float(s.get("strikeoutsPer9Inn")), "bb9": safe_float(s.get("walksPer9Inn")),
            "hr9": safe_float(s.get("homeRunsPer9")), "fip": fip, "babip": babip, "kbb": kbb, "gb_rate": gb_rate}

def fetch_pitcher_recent_form(pid, n_starts=5):
    if not pid: return {}
    data = cget(f"{MLB_API_BASE}/people/{pid}/stats?stats=gameLog&season={CURRENT_SEASON}&group=pitching", ttl=3600)
    if not data: return {}
    splits = data.get("stats", [{}])[0].get("splits", [])
    starts = [s for s in splits if s.get("stat", {}).get("inningsPitched", "0") != "0" and
              s.get("game", {}).get("gameType") == "R"]
    recent = starts[-n_starts:]
    if not recent: return {}
    eras, k9s, bbs, hrs = [], [], [], []
    for s in recent:
        st = s.get("stat", {})
        ip_s = st.get("inningsPitched", "0")
        ipv = 0
        if isinstance(ip_s, str) and "." in ip_s:
            p2 = ip_s.split(".")
            ipv = int(p2[0]) + int(p2[1]) / 3.0 if len(p2) > 1 else float(p2[0])
        else:
            ipv = float(ip_s or 0)
        er = safe_float(st.get("earnedRuns"))
        eras.append(9 * er / ipv if ipv > 0 else 4.5)
        k9s.append(safe_float(st.get("strikeoutsPer9Inn")))
        bbs.append(safe_float(st.get("walksPer9Inn")))
        hrs.append(safe_float(st.get("homeRunsPer9")))
    return {"rec_era": np.mean(eras), "rec_k9": np.mean(k9s),
            "rec_bb9": np.mean(bbs), "rec_hr9": np.mean(hrs)}

def compute_elo(home_games, away_games, home_id, away_id, k=32):
    seen, ordered = set(), []
    for g in sorted(home_games + away_games, key=lambda x: x.get("gameDate", "")):
        pk = g.get("gamePk")
        if pk not in seen:
            seen.add(pk); ordered.append(g)
    elos = {}
    for g in ordered:
        t = g["teams"]
        try:
            hg, ag = t["home"]["team"]["id"], t["away"]["team"]["id"]
            hs, aws_s = int(t["home"].get("score",0)), int(t["away"].get("score",0))
        except: continue
        if hs == 0 and aws_s == 0: continue
        for eid in (hg, ag): elos.setdefault(eid, 1500)
        he, ae = elos[hg], elos[ag]
        exp_h = 1 / (1 + 10 ** ((ae - he - 50) / 400))
        act_h = 1 if hs > aws_s else (0 if hs < aws_s else 0.5)
        marg = min(np.log(abs(hs - aws_s) + 1) / 2.2, 1.5)
        elos[hg] += k * marg * (act_h - exp_h)
        elos[ag] += k * marg * ((1 - act_h) - (1 - exp_h))
    h_elo = elos.get(home_id, 1500); a_elo = elos.get(away_id, 1500)
    return round(h_elo), round(a_elo)

PARK_FACTORS = {
    "Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
    "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
    "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
    "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
    "Oracle Park": 0.98, "Petco Park": 0.97, "Oakland Coliseum": 0.97,
    "Tropicana Field": 0.96, "Target Field": 0.96, "PNC Park": 0.97,
}

def build_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f):
    f = {
        "h_elo": h_elo, "a_elo": a_elo,
        "h_wp": hf.get("wp", 0.5), "a_wp": af.get("wp", 0.5),
        "h_rs": hf.get("rs", 4.5), "a_rs": af.get("rs", 4.5),
        "h_ra": hf.get("ra", 4.5), "a_ra": af.get("ra", 4.5),
        "h_rest": hf.get("rest", 3), "a_rest": af.get("rest", 3),
        "h_ops": hs.get("hitting",{}).get("ops", 0.700),
        "a_ops": aws.get("hitting",{}).get("ops", 0.700),
        "h_whip": hs.get("pitching",{}).get("whip", 1.35),
        "a_whip": aws.get("pitching",{}).get("whip", 1.35),
        "h_era": hs.get("pitching",{}).get("era", 4.5),
        "a_era": aws.get("pitching",{}).get("era", 4.5),
        "park": park_f,
        "hp_era": hpitch.get("era", 4.5) if hpitch else 4.5,
        "hp_k9": hpitch.get("k9", 8.0) if hpitch else 8.0,
        "hp_bb9": hpitch.get("bb9", 3.0) if hpitch else 3.0,
        "hp_hr9": hpitch.get("hr9", 1.2) if hpitch else 1.2,
        "hp_v": 1 if (hpitch and hpitch.get("ip",0) >= 10) else 0,
        "hp_fip": hpitch.get("fip", 4.5) if hpitch else 4.5,
        "hp_babip": hpitch.get("babip", 0.300) if hpitch else 0.300,
        "hp_kbb": hpitch.get("kbb", 3.0) if hpitch else 3.0,
        "hp_gb_rate": hpitch.get("gb_rate", 0.44) if hpitch else 0.44,
        "ap_era": apitch.get("era", 4.5) if apitch else 4.5,
        "ap_k9": apitch.get("k9", 8.0) if apitch else 8.0,
        "ap_bb9": apitch.get("bb9", 3.0) if apitch else 3.0,
        "ap_hr9": apitch.get("hr9", 1.2) if apitch else 1.2,
        "ap_v": 1 if (apitch and apitch.get("ip",0) >= 10) else 0,
        "ap_fip": apitch.get("fip", 4.5) if apitch else 4.5,
        "ap_babip": apitch.get("babip", 0.300) if apitch else 0.300,
        "ap_kbb": apitch.get("kbb", 3.0) if apitch else 3.0,
        "ap_gb_rate": apitch.get("gb_rate", 0.44) if apitch else 0.44,
    }
    return f

def norm_cdf(x, mu=0, sigma=1):
    return 0.5 * (1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

def compute_ev(prob, odds):
    if odds is None: return None
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return round((prob/100 * dec) - 1, 4)

def american_to_prob(odds):
    if odds is None or odds == 0: return None
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)

def match_game(odds_list, home_name, away_name):
    hn = home_name.replace("Los Angeles","LA").replace("New York","NY").replace("San Francisco","SF")
    an = away_name.replace("Los Angeles","LA").replace("New York","NY").replace("San Francisco","SF")
    for og in odds_list:
        oh, oa = og.get("home_team",""), og.get("away_team","")
        if (home_name in oh or hn in oh or oh in home_name) and (away_name in oa or an in oa or oa in away_name):
            return og
        if (home_name in oa or hn in oa or oa in home_name) and (away_name in oh or an in oh or oh in away_name):
            return og
    return None

def extract_market_odds(game_odds, market_key, outcome_name=None, expect_point=None):
    if not game_odds: return None, None, None
    best_price, best_book, best_point = None, None, None
    for book in game_odds.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != market_key: continue
            for oc in mkt.get("outcomes", []):
                if outcome_name and oc.get("name") != outcome_name: continue
                point = oc.get("point")
                if expect_point is not None and point != expect_point: continue
                price = oc.get("price")
                if best_price is None or (price is not None and abs(price) > abs(best_price)):
                    best_price = price; best_book = book.get("title", "Unknown"); best_point = point
    return best_price, best_book, best_point

def fmt_odds(odds):
    if odds is None: return "N/A"
    return f"{odds:+d}"

def edge_pct(prob, odds):
    ip = american_to_prob(odds)
    if ip is None or ip == 0: return None
    return round((prob/100 - ip) * 100, 1)

def main():
    print("=== Morning Summary — Generando predicciones ===")
    today_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    data = load_picks()
    pnl = get_pnl()

    # Fetch schedule + odds
    games = fetch_todays_schedule()
    print(f"  Juegos hoy: {len(games)}")

    odds_raw = []
    if ODDS_API_KEY:
        try:
            r = requests.get(f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}", timeout=10)
            if r.status_code == 200:
                odds_raw = r.json()
        except:
            pass
    print(f"  Partidos con odds: {len(odds_raw)}")

    picks = []
    for g in games:
        try:
            t = g["teams"]
            hid = t["home"]["team"]["id"]; aid = t["away"]["team"]["id"]
            hn = t["home"]["team"]["name"]; an = t["away"]["team"]["name"]
            venue = g.get("venue", {}).get("name", "")
            gstatus = g.get("status", {}).get("detailedState", "")
            if gstatus == "Final":
                continue
            hp_info = (g.get("probablePitcher") or {}).get("home") or g.get("probablePitcher", {})
            ap_info = (g.get("probablePitcher") or {}).get("away") or g.get("probablePitcher", {})
        except:
            continue

        print(f"\n  Procesando: {an} @ {hn}")

        hs = fetch_team_stats(hid)
        aws = fetch_team_stats(aid)
        hr_games = fetch_recent_games(hid)
        ar_games = fetch_recent_games(aid)
        hf = compute_form(hr_games, hid)
        af = compute_form(ar_games, aid)
        hpitch = fetch_pitcher_stats(hp_info.get("id"))
        apitch = fetch_pitcher_stats(ap_info.get("id"))
        h_elo, a_elo = compute_elo(hr_games, ar_games, hid, aid)
        park_f = PARK_FACTORS.get(venue, 1.0)

        if not _MODELS_LOADED:
            print("  Modelos no cargados — salteando")
            continue

        row = build_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f)
        x = np.array([[row[c] for c in _cols]])

        hw_prob = _xgb_hw.predict_proba(x)[0, 1]
        exp_rdiff = _xgb_rd.predict(x)[0]
        exp_total = _xgb_tot.predict(x)[0]

        n_sims = 5000
        rdiff_sims = np.random.normal(exp_rdiff, 3.0, n_sims)
        total_sims = np.random.normal(exp_total, 3.2, n_sims)

        mc_hw = np.mean(rdiff_sims > 0)
        mc_home_minus = np.mean(rdiff_sims >= 1.5)
        mc_home_plus = np.mean(rdiff_sims >= -1.5)
        mc_over = np.mean(total_sims > 8.5)

        ml_hp = calibrate_ml(round(float(mc_hw), 4))
        ml_ap = 1.0 - ml_hp
        spr_home_minus = calibrate_rl(round(float(mc_home_minus), 4))
        spr_home_plus = calibrate_rl(round(float(mc_home_plus), 4))
        spr_away_minus = calibrate_rl(round(float(1.0 - mc_home_plus), 4))
        spr_away_plus = calibrate_rl(round(float(1.0 - mc_home_minus), 4))
        over_prob = calibrate_ml(round(float(mc_over), 4))
        under_prob = 1.0 - over_prob

        # Match odds
        og = match_game(odds_raw, hn, an)
        home_abbr = REV_TEAM.get(hn.lower(), hn[:3].upper())
        away_abbr = REV_TEAM.get(an.lower(), an[:3].upper())
        game_label = f"{away_abbr} @ {home_abbr}"

        # Justification helpers
        h_fip = hpitch.get("fip", 4.5) if hpitch else 4.5
        a_fip = apitch.get("fip", 4.5) if apitch else 4.5
        elo_diff = h_elo - a_elo
        h_wp = hf.get("wp", 0.5); a_wp = af.get("wp", 0.5)
        h_rs10 = hf.get("rs", 4.5); a_rs10 = af.get("rs", 4.5)

        def ml_reason(team):
            is_h = team == hn
            fip_own = h_fip if is_h else a_fip
            fip_opp = a_fip if is_h else h_fip
            elo = elo_diff if is_h else -elo_diff
            margin = exp_rdiff if is_h else -exp_rdiff
            if fip_own < fip_opp - 0.3:
                return f"Los {team} tienen mejor abridor hoy (FIP {fip_own:.2f} vs {fip_opp:.2f}), lo que inclina la balanza a su favor."
            if abs(elo) > 20:
                return f"Los {team} llegan con mejor rendimiento general (Elo +{abs(elo)}) y el modelo proyecta su victoria por {margin:+.1f} carreras."
            return f"El modelo favorece a los {team} con un {ml_prob*100:.0f}% de probabilidad de ganar el juego."

        def ou_reason(line):
            avg_rs = (h_rs10 + a_rs10) / 2
            if abs(park_f - 1.0) > 0.04:
                label = "un parque favorable a la ofensiva" if park_f > 1.0 else "un parque que favorece a los lanzadores"
                if avg_rs > 4.5:
                    return f"Se espera un juego con ~{exp_total:.0f} carreras, superando la línea de {line}. Además, {label} y ambos equipos vienen promediando {avg_rs:.1f} carreras por juego."
                return f"Se espera un juego con ~{exp_total:.0f} carreras, superando la línea de {line}. También {label}."
            if avg_rs > 4.5:
                return f"El modelo espera muchas carreras (~{exp_total:.0f}) superando la línea de {line}. Los dos equipos vienen anotando bien: {avg_rs:.1f} carreras por juego cada uno."
            return f"El modelo proyecta ~{exp_total:.0f} carreras totales, por encima de la línea de {line}."

        game_picks = []

        # ML — pick the favorite
        if ml_hp >= 0.5:
            ml_team = hn; ml_prob = ml_hp
        else:
            ml_team = an; ml_prob = ml_ap

        ml_price, ml_book, _ = extract_market_odds(og, "h2h", ml_team)

        if ml_price is not None:
            game_picks.append({
                "game": game_label, "market": "ML", "team": ml_team, "detail": "",
                "prob": round(ml_prob * 100, 1), "odds": ml_price,
                "edge": edge_pct(round(ml_prob * 100, 1), ml_price),
                "reason": ml_reason(ml_team),
            })

        # RL -1.5 (favorite)
        if ml_hp >= 0.5:
            rl_fav_team = hn; rl_fav_prob = spr_home_minus
        else:
            rl_fav_team = an; rl_fav_prob = spr_away_minus
        spr_price, spr_book, _ = extract_market_odds(og, "spreads", rl_fav_team, expect_point=-1.5)
        def rl_reason(team, is_fav):
            if is_fav:
                margin = exp_rdiff if team == hn else -exp_rdiff
                return f"Modelo proyecta margen de {margin:+.1f} carreras, suficiente para cubrir -1.5."
            return f"Modelo proyecta {ml_prob*100:.0f}% de ganar el juego directamente."

        if spr_price is not None:
            game_picks.append({
                "game": game_label, "market": "RL -1.5", "team": rl_fav_team, "detail": "-1.5",
                "prob": round(rl_fav_prob * 100, 1), "odds": spr_price,
                "edge": edge_pct(round(rl_fav_prob * 100, 1), spr_price),
                "reason": rl_reason(rl_fav_team, True),
            })

        # RL +1.5 (underdog)
        if ml_hp >= 0.5:
            rl_dog_team = an; rl_dog_prob = spr_away_plus
        else:
            rl_dog_team = hn; rl_dog_prob = spr_home_plus
        spr_dog_price, spr_dog_book, _ = extract_market_odds(og, "spreads", rl_dog_team, expect_point=1.5)
        if spr_dog_price is not None:
            game_picks.append({
                "game": game_label, "market": "RL +1.5", "team": rl_dog_team, "detail": "+1.5",
                "prob": round(rl_dog_prob * 100, 1), "odds": spr_dog_price,
                "edge": edge_pct(round(rl_dog_prob * 100, 1), spr_dog_price),
                "reason": rl_reason(rl_dog_team, False),
            })

        # O/U
        ov_price, ov_book, ov_point = extract_market_odds(og, "totals")
        if ov_price is not None and ov_point is not None:
            ov_prob = norm_cdf(exp_total - ov_point, 0, 3.2)
            ov_prob_cal = calibrate_ml(round(float(ov_prob), 4))
            ov_reason = ou_reason(ov_point)
            if ov_prob_cal >= 0.5:
                game_picks.append({
                    "game": game_label, "market": "O/U", "team": "Over",
                    "detail": f"o{ov_point}",
                    "prob": round(ov_prob_cal * 100, 1), "odds": ov_price,
                    "edge": edge_pct(round(ov_prob_cal * 100, 1), ov_price),
                    "reason": ov_reason,
                })
            else:
                un_price, un_book, _ = extract_market_odds(og, "totals", "Under")
                if un_price is not None:
                    game_picks.append({
                        "game": game_label, "market": "O/U", "team": "Under",
                        "detail": f"u{ov_point}",
                        "prob": round((1-ov_prob_cal)*100, 1), "odds": un_price,
                        "edge": edge_pct(round((1-ov_prob_cal)*100, 1), un_price),
                        "reason": ov_reason,
                    })

        picks.extend(game_picks)

    # Filter: only edges > 2%, one per game (best edge)
    recs = [p for p in picks if p["edge"] is not None and p["edge"] > 2]
    best_per_game = {}
    for p in recs:
        g = p["game"]
        if g not in best_per_game or p["edge"] > best_per_game[g]["edge"]:
            best_per_game[g] = p
    recs = sorted(best_per_game.values(), key=lambda x: x["edge"], reverse=True)
    top = recs[:4]

    # Format message
    lines = [f"☀️ *Buenos dias! — {today}*\n"]
    lines.append(f"🏟️ *Recomendaciones del modelo* ({len(games)} juegos)\n")

    if not top:
        lines.append("_Sin recomendaciones con valor positivo hoy._")
    else:
        for r in top:
            detail = f" {r['detail']}" if r["detail"] else ""
            flames = "🔥" if r["edge"] >= 8 else ("⭐" if r["edge"] >= 5 else "📝")
            stake_info = ""
            try:
                br = data.get("bankroll", 1000)
                stake_amt, _, _ = recommend_stake(r["prob"]/100, r["odds"], bankroll=br)
                if stake_amt > 0:
                    stake_info = f"  💰 ${stake_amt:.0f}"
            except:
                pass
            lines.append(f"{flames} *{r['game']}*")
            lines.append(f"   {r['market']} {r['team']}{detail}")
            lines.append(f"   Prob: {r['prob']:.0f}%  |  Odds: {fmt_odds(r['odds'])}  |  Edge: +{r['edge']:.1f}%{stake_info}")
            if r.get("reason"):
                lines.append(f"   _{r['reason']}_")
            lines.append("")

    # P&L summary
    lines.append("┅" * 10)
    lines.append(f"💰 *Bankroll:* ${data['bankroll']:.2f}")
    lines.append(f"📊 *Record:* {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
    lines.append(f"📈 *Profit:* ${pnl['profit']:.2f} ({pnl['roi']}%)")

    msg = "\n".join(lines)
    print(f"\nMensaje:\n{msg}")
    send_telegram(msg)

if __name__ == "__main__":
    main()
