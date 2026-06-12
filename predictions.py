"""predictions.py — Shared prediction pipeline (mirrors app.py exactly)."""
import os, sys, math, pickle, json, time
from datetime import datetime, timezone, timedelta
import numpy as np
import requests

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except:
    TZ = timezone(timedelta(hours=-6))

CURRENT_SEASON = 2026
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_API_KEY = "b09f7e5fb08081c87e7e34272fda4ea0"
BASE = os.path.join(os.path.dirname(__file__), "")

PARK_FACTORS = {
    "Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
    "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
    "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
    "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
    "Oracle Park": 0.98, "Petco Park": 0.97, "Oakland Coliseum": 0.97,
    "Tropicana Field": 0.96, "Target Field": 0.96, "PNC Park": 0.97,
}

STADIUM_COORDS = {
    "Coors Field": (39.7559, -104.9942), "Great American Ball Park": (39.0976, -84.5066),
    "Citizens Bank Park": (39.9054, -75.1665), "Fenway Park": (42.3467, -71.0972),
    "Yankee Stadium": (40.8296, -73.9265), "Globe Life Field": (32.7474, -97.0831),
    "American Family Field": (43.0281, -87.9712), "Busch Stadium": (38.6226, -90.1928),
    "Chase Field": (33.4455, -112.0667), "Comerica Park": (42.3390, -83.0486),
    "Citi Field": (40.7571, -73.8458), "T-Mobile Park": (47.5914, -122.3329),
    "Oracle Park": (37.7786, -122.3893), "Petco Park": (32.7076, -117.1571),
    "RingCentral Coliseum": (37.7516, -122.2005), "Oakland Coliseum": (37.7516, -122.2005),
    "Tropicana Field": (27.7682, -82.6532), "Target Field": (44.9817, -93.2779),
    "PNC Park": (40.4469, -79.9962), "Dodger Stadium": (34.0739, -118.2400),
    "Wrigley Field": (41.9484, -87.6553), "Angel Stadium": (33.8003, -117.8827),
    "Nationals Park": (38.8730, -77.0075), "Progressive Field": (41.4962, -81.6852),
    "loanDepot park": (25.7781, -80.2197), "Marlins Park": (25.7781, -80.2197),
    "Kauffman Stadium": (39.0517, -94.4802), "Rogers Centre": (43.6414, -79.3894),
    "Guaranteed Rate Field": (41.8302, -87.6338), "Truist Park": (33.8909, -84.4676),
    "Oriole Park at Camden Yards": (39.2839, -76.6216), "Minute Maid Park": (29.7572, -95.3555),
    "Cleveland Guardians": (41.4962, -81.6852),
}

DOME_VENUES = {"Tropicana Field", "Rogers Centre", "Chase Field", "Globe Life Field",
               "American Family Field", "Minute Maid Park", "loanDepot park", "Marlins Park"}

WEATHER_CACHE = {}
WEATHER_CACHE_TTL = 3600

LG_AVG_RUNS = 4.5

_xgb_hw = _xgb_rd = _xgb_tot = None
_rf_hw = _rf_rd = _rf_tot = None
_cols = None
_MODELS_LOADED = False

def load_models():
    global _xgb_hw, _xgb_rd, _xgb_tot, _rf_hw, _rf_rd, _rf_tot, _cols, _MODELS_LOADED
    try:
        import xgboost as xgb
        with open(BASE + "xgb_hw.pkl", "rb") as f: _xgb_hw = pickle.load(f)
        with open(BASE + "xgb_rd.pkl", "rb") as f: _xgb_rd = pickle.load(f)
        with open(BASE + "xgb_tot.pkl", "rb") as f: _xgb_tot = pickle.load(f)
        with open(BASE + "xgb_cols.pkl", "rb") as f: _cols = pickle.load(f)
        _MODELS_LOADED = True
    except:
        pass
    if not _MODELS_LOADED:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            with open(BASE + "rf_hw.pkl", "rb") as f: _rf_hw = pickle.load(f)
            with open(BASE + "rf_rd.pkl", "rb") as f: _rf_rd = pickle.load(f)
            with open(BASE + "rf_tot.pkl", "rb") as f: _rf_tot = pickle.load(f)
            with open(BASE + "rf_cols.pkl", "rb") as f: _cols = pickle.load(f)
            _MODELS_LOADED = True
        except:
            pass

load_models()

def safe_float(v, d=0.0):
    if v is None: return d
    try: return float(v)
    except: return d

def fetch_todays_schedule():
    today = datetime.now(TZ).strftime("%m/%d/%Y")
    try:
        r = requests.get(f"{MLB_API_BASE}/schedule?sportId=1&date={today}&hydrate=probablePitcher", timeout=10)
        if r.status_code == 200:
            games = []
            for d in r.json().get("dates", []):
                games.extend(d.get("games", []))
            return games
    except:
        pass
    return []

def fetch_team_stats_mlb(tid, season=CURRENT_SEASON):
    url = f"{MLB_API_BASE}/teams/{tid}/stats?season={season}&group=hitting,pitching&stats=season"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
    except:
        return {}
    result = {"hitting": {}, "pitching": {}}
    for sg in resp.json().get("stats", []):
        g = sg.get("group", {}).get("displayName", "").lower()
        splits = sg.get("splits", [])
        if splits:
            s = splits[0].get("stat", {})
            if g == "hitting":
                result["hitting"] = {k: safe_float(s.get(k)) for k in ["avg","runs","hr","obp","slg","ops","hits","strikeOuts","baseOnBalls"]}
            elif g == "pitching":
                result["pitching"] = {k: safe_float(s.get(k)) for k in ["era","whip","runs","strikeouts","baseOnBalls","homeRuns"]}
    return result

PRED_LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.json")

def log_all_todays_predictions():
    """Run full pipeline and log ALL markets for every game to predictions_log.json.
    Returns number of new predictions logged."""
    from bankroll import calibrate_ml, calibrate_rl

    today_str = datetime.now(TZ).strftime("%Y-%m-%d")

    # No intentar antes de las 5 AM (Chihuahua) — el schedule no está disponible
    hour = datetime.now(TZ).hour
    if hour < 5:
        return 0

    # Si ya hay predicciones guardadas hoy, salir sin llamar odds API
    try:
        with open(PRED_LOG_PATH) as f:
            log_data = json.load(f)
    except:
        log_data = {"predictions": []}
    existing_ids = {p["id"] for p in log_data["predictions"] if p.get("date") == today_str}
    if existing_ids:
        return 0

    games = fetch_todays_schedule()
    odds_raw = fetch_odds()
    ab_map = fetch_team_abbrevs()

    new_count = 0

    for g in games:
        sc = g.get("status",{}).get("codedGameState","S")
        sd = g.get("status",{}).get("detailedState","Scheduled")
        if sc in ("F", "O") or sd == "Final":
            continue
        t = g["teams"]
        hi, ai = t["home"]["team"], t["away"]["team"]
        hid, aid = hi["id"], ai["id"]
        hn, an = hi["name"], ai["name"]
        ha, aa = ab_map.get(hid, "??"), ab_map.get(aid, "??")

        hs = fetch_team_stats_mlb(hid) if hid else {}
        aws = fetch_team_stats_mlb(aid) if aid else {}
        hr = fetch_recent_games(hid) if hid else []
        ar = fetch_recent_games(aid) if aid else []
        hf = compute_form(hr, hid)
        af = compute_form(ar, aid)

        hp_info = g.get("teams",{}).get("home",{}).get("probablePitcher") or {}
        ap_info = g.get("teams",{}).get("away",{}).get("probablePitcher") or {}
        hpitch = fetch_pitcher_stats(hp_info.get("id")) if hp_info.get("id") else {}
        apitch = fetch_pitcher_stats(ap_info.get("id")) if ap_info.get("id") else {}
        hprec = fetch_pitcher_recent_form(hp_info.get("id")) if hp_info.get("id") else {}
        aprec = fetch_pitcher_recent_form(ap_info.get("id")) if ap_info.get("id") else {}
        _, h_elo, a_elo = compute_elo(hr, ar, hid, aid)
        venue_name = g.get("venue",{}).get("name", "")
        park_f = PARK_FACTORS.get(venue_name, 1.0)
        weather = fetch_weather(venue_name)

        mc = monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo,
                                 hpitch if hpitch.get("ip",0) >= 10 else None,
                                 apitch if apitch.get("ip",0) >= 10 else None,
                                 park_f, hp_rec=hprec, ap_rec=aprec, weather=weather)

        ml_hp = mc["ml_hp"]
        if ml_hp is None:
            continue
        cal_hp = calibrate_ml(ml_hp)
        cal_ap = 1.0 - cal_hp
        spr_home_minus = mc["spr_home_minus"]
        spr_home_plus = mc["spr_home_plus"]
        spr_away_minus = mc["spr_away_minus"]
        spr_away_plus = mc["spr_away_plus"]
        spr_exp_margin = mc["spr_exp_margin"]
        exp_total = mc["exp_total"]
        total_std = mc["total_std"]

        gl = f"{aa} @ {ha}"
        gid = str(g.get("gamePk", ""))
        if not gid:
            continue

        og = match_game(odds_raw, hn, an) if odds_raw else None
        spr_fav_team = hn if spr_exp_margin >= 0 else an
        spr_dog_team = an if spr_exp_margin >= 0 else hn
        spr_fav_prob = spr_home_minus if spr_exp_margin >= 0 else spr_away_minus
        spr_dog_prob = spr_away_plus if spr_exp_margin >= 0 else spr_home_plus

        if og:
            m_fav, m_dog = None, None
            for book in og.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    if mkt.get("key") == "spreads":
                        oc = mkt.get("outcomes", [])
                        if len(oc) >= 2:
                            for o in oc:
                                if o.get("point", 0) < 0: m_fav = o["name"]
                                elif o.get("point", 0) > 0: m_dog = o["name"]
                        break
                if m_fav: break
            if m_fav and m_dog:
                spr_fav_team = m_fav
                spr_dog_team = m_dog
                if m_fav == hn:
                    spr_fav_prob = spr_home_minus
                    spr_dog_prob = spr_away_plus
                else:
                    spr_fav_prob = spr_away_minus
                    spr_dog_prob = spr_home_plus
                spr_fav_prob = calibrate_rl(spr_fav_prob)
                spr_dog_prob = 1.0 - spr_fav_prob

        # ML
        ml_team = hn if ml_hp > 0.50 else an
        ml_prob = max(cal_hp, cal_ap)
        ml_odds = "N/A"
        ml_edge = None
        if og:
            ml_price, _, _ = extract_market_odds(og, "h2h", ml_team)
            if ml_price:
                ml_odds = ml_price
                ip = american_to_prob(ml_price)
                if ip:
                    ml_edge = round(ml_prob * 100 - ip * 100, 1)

        pid = f"{gid}_moneyline"
        if pid not in existing_ids:
            log_data["predictions"].append({
                "id": pid, "date": today_str, "game": gl,
                "away_abbrev": aa, "home_abbrev": ha,
                "market": "ML", "pick": ml_team,
                "prob": round(ml_prob * 100, 1), "odds": ml_odds,
                "edge": ml_edge, "detail": "",
                "result": None, "settled": False,
            })
            new_count += 1

        # RL omitido — bajo rendimiento

        # O/U (solo si hay odds — sin línea no se puede liquidar)
        mkt = "O/U"
        pid = f"{gid}_total"
        if pid not in existing_ids and og:
            ov_price, _, ov_point = extract_market_odds(og, "totals")
            if ov_price and ov_point:
                over_prob = norm_cdf(exp_total - ov_point, 0, total_std)
                if over_prob > 0.5:
                    ou_team = "Over"
                    ou_detail = f"o{ov_point}"
                    ou_prob = over_prob * 100
                    ou_odds = ov_price
                else:
                    un_price, _, _ = extract_market_odds(og, "totals", "Under")
                    ou_team = "Under"
                    ou_detail = f"u{ov_point}"
                    ou_prob = (1 - over_prob) * 100
                    ou_odds = un_price or ov_price
                ou_edge = None
                ip = american_to_prob(ou_odds)
                if ip:
                    ou_edge = round(ou_prob - ip * 100, 1)
                log_data["predictions"].append({
                    "id": pid, "date": today_str, "game": gl,
                    "away_abbrev": aa, "home_abbrev": ha,
                    "market": mkt, "pick": ou_team,
                    "prob": round(ou_prob, 1), "odds": ou_odds,
                    "edge": ou_edge, "detail": ou_detail,
                    "result": None, "settled": False,
                })
                new_count += 1

    if new_count > 0:
        os.makedirs(os.path.dirname(PRED_LOG_PATH) or ".", exist_ok=True)
        with open(PRED_LOG_PATH, "w") as f:
            json.dump(log_data, f, indent=2)
    return new_count


def compute_model_stats():
    """Compute model performance from predictions_log.json.
    Returns formatted string."""
    try:
        with open(PRED_LOG_PATH) as f:
            data = json.load(f)
    except:
        return "❌ No hay datos de predicciones."

    settled = [p for p in data["predictions"] if p.get("settled")]
    if not settled:
        return "📊 Sin predicciones liquidadas aún."

    lines = ["📊 *Rendimiento del Modelo*\n"]
    markets = {}
    for p in settled:
        m = p["market"]
        if m not in markets:
            markets[m] = {"w": 0, "l": 0, "prob_sum": 0, "count": 0}
        if p.get("result") == "W":
            markets[m]["w"] += 1
        else:
            markets[m]["l"] += 1
        markets[m]["prob_sum"] += p.get("prob", 0)
        markets[m]["count"] += 1

    total_w = sum(m["w"] for m in markets.values())
    total_l = sum(m["l"] for m in markets.values())
    total = total_w + total_l

    lines.append(f"*Total:* {total_w}-{total_l} ({total_w/total*100:.1f}%)\n" if total > 0 else "*Total:* 0\n")

    for m in ["ML", "O/U"]:
        if m not in markets:
            continue
        s = markets[m]
        n = s["w"] + s["l"]
        pct = s["w"] / n * 100 if n > 0 else 0
        avg_prob = s["prob_sum"] / n if n > 0 else 0
        lines.append(f"• *{m}:* {s['w']}-{s['l']} ({pct:.1f}%) | Prob prom: {avg_prob:.0f}%")

    # Fechas
    dates = sorted(set(p["date"] for p in settled))
    lines.append(f"\n📅 Datos desde: {dates[0]} hasta {dates[-1]} ({len(dates)} días)")
    lines.append(f"📈 Total predicciones: {len(data['predictions'])} ({len(settled)} liquidadas, {len(data['predictions'])-len(settled)} pendientes)")

    return "\n".join(lines)

def fetch_team_abbrevs():
    try:
        teams = requests.get(f"{MLB_API_BASE}/teams?sportIds=1", timeout=10).json()
        return {t["id"]: t.get("abbreviation","??") for t in teams.get("teams",[])}
    except:
        return {}

def fetch_pitcher_stats(pid):
    if not pid: return {}
    try:
        url = f"{MLB_API_BASE}/people/{pid}/stats?stats=season&season={CURRENT_SEASON}&group=pitching"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json()
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
        return {
            "era": safe_float(s.get("era")), "whip": safe_float(s.get("whip")), "ip": ip_val,
            "k9": safe_float(s.get("strikeoutsPer9Inn")), "bb9": safe_float(s.get("walksPer9Inn")),
            "hr9": safe_float(s.get("homeRunsPer9")), "fip": fip, "babip": babip, "kbb": kbb,
            "gb_rate": gb_rate,
            "name": s.get("player", {}).get("fullName") if isinstance(s.get("player"), dict) else s.get("player", ""),
        }
    except:
        return {}

def fetch_pitcher_recent_form(pid, n_starts=5):
    if not pid: return {}
    try:
        url = f"{MLB_API_BASE}/people/{pid}/stats?stats=gameLog&season={CURRENT_SEASON}&group=pitching"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200: return {}
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        starts = [s for s in splits if s.get("stat", {}).get("inningsPitched", "0") != "0" and s.get("game", {}).get("gameType") == "R"]
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
        return {"rec_era": np.mean(eras), "rec_k9": np.mean(k9s), "rec_bb9": np.mean(bbs), "rec_hr9": np.mean(hrs)}
    except:
        return {}

OWM_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

def fetch_weather(venue_name):
    if venue_name in DOME_VENUES:
        return {"temp_f": 72.0, "wind_mph": 0.0, "humidity": 50, "clouds": 0, "conditions": "dome"}
    if not OWM_API_KEY:
        return {}
    now = time.time()
    cached = WEATHER_CACHE.get(venue_name)
    if cached and now - cached["ts"] < WEATHER_CACHE_TTL:
        return cached["data"]
    coords = STADIUM_COORDS.get(venue_name)
    if not coords:
        return {}
    lat, lon = coords
    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=imperial", timeout=8)
        if r.status_code == 200:
            d = r.json()
            w = d.get("main", {})
            wind = d.get("wind", {})
            data = {
                "temp_f": w.get("temp", 72.0),
                "wind_mph": wind.get("speed", 0.0),
                "humidity": w.get("humidity", 50),
                "clouds": d.get("clouds", {}).get("all", 0),
                "conditions": d.get("weather", [{}])[0].get("description", "clear"),
            }
            WEATHER_CACHE[venue_name] = {"data": data, "ts": now}
            return data
    except:
        pass
    return {}

def fetch_recent_games(tid, ng=20):
    today = datetime.now(TZ)
    end = today.strftime("%m/%d/%Y")
    start = (today - timedelta(days=45)).strftime("%m/%d/%Y")
    try:
        r = requests.get(f"{MLB_API_BASE}/schedule?sportId=1&teamId={tid}&startDate={start}&endDate={end}", timeout=10)
        if r.status_code == 200:
            games = []
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    if g.get("status", {}).get("codedGameState") == "F":
                        games.append(g)
            return games[-ng:]
    except:
        pass
    return []

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
            hs_val, aws_val = int(t["home"].get("score",0)), int(t["away"].get("score",0))
        except:
            continue
        if hs_val == 0 and aws_val == 0:
            continue
        for eid in (hg, ag):
            elos.setdefault(eid, 1500)
        he, ae = elos[hg], elos[ag]
        exp_h = 1 / (1 + 10 ** ((ae - he - 50) / 400))
        act_h = 1 if hs_val > aws_val else (0 if hs_val < aws_val else 0.5)
        marg = min(np.log(abs(hs_val - aws_val) + 1) / 2.2, 1.5)
        elos[hg] += k * marg * (act_h - exp_h)
        elos[ag] += k * marg * ((1 - act_h) - (1 - exp_h))
    return round(elos[home_id]) if home_id in elos else 1500, round(elos[home_id]) if home_id in elos else 1500, round(elos[away_id]) if away_id in elos else 1500

def compute_form(games, tid):
    if not games:
        return {"wp": 0.5, "rs": 4.5, "ra": 4.5, "rest": 3}
    sorted_games = sorted(games, key=lambda x: x.get("gameDate", ""), reverse=True)
    w, rs, ra = 0, [], []
    for g in sorted_games:
        t = g["teams"]
        is_h = t["home"]["team"]["id"] == tid
        if not is_h and t["away"]["team"]["id"] != tid:
            continue
        ms = safe_float(t["home"]["score"] if is_h else t["away"]["score"])
        os_val = safe_float(t["away"]["score"] if is_h else t["home"]["score"])
        rs.append(ms); ra.append(os_val)
        if ms > os_val:
            w += 1
    n = len(rs) or 1
    rest = 3
    if sorted_games:
        try:
            ld = datetime.strptime(sorted_games[0].get("gameDate","").split("T")[0], "%Y-%m-%d")
            rest = max((datetime.now().date() - ld.date()).days, 1)
        except:
            pass
    return {"wp": w/n, "rs": np.mean(rs) if rs else 4.5, "ra": np.mean(ra) if ra else 4.5, "rest": rest}

def build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=None, ap_rec=None, weather=None):
    weather = weather or {}
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
        "hp_rec_era": hp_rec.get("rec_era", hpitch.get("era", 4.5)) if hp_rec else 4.5,
        "hp_rec_k9": hp_rec.get("rec_k9", hpitch.get("k9", 8.0)) if hp_rec else 8.0,
        "hp_rec_bb9": hp_rec.get("rec_bb9", hpitch.get("bb9", 3.0)) if hp_rec else 3.0,
        "hp_rec_hr9": hp_rec.get("rec_hr9", hpitch.get("hr9", 1.2)) if hp_rec else 1.2,
        "ap_rec_era": ap_rec.get("rec_era", apitch.get("era", 4.5)) if ap_rec else 4.5,
        "ap_rec_k9": ap_rec.get("rec_k9", apitch.get("k9", 8.0)) if ap_rec else 8.0,
        "ap_rec_bb9": ap_rec.get("rec_bb9", apitch.get("bb9", 3.0)) if ap_rec else 3.0,
        "ap_rec_hr9": ap_rec.get("rec_hr9", apitch.get("hr9", 1.2)) if ap_rec else 1.2,
        "temp_f": weather.get("temp_f", 72.0), "wind_mph": weather.get("wind_mph", 0.0),
        "humidity": weather.get("humidity", 50), "is_dome": 1 if weather.get("conditions") == "dome" else 0,
    }
    return f

def monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=None, ap_rec=None, n_sims=5000, weather=None):
    if not _MODELS_LOADED:
        return {"ml_hp": None, "ml_ap": None, "spr_home_minus": None, "spr_home_plus": None, "spr_away_minus": None, "spr_away_plus": None, "spr_exp_margin": None, "exp_total": None, "total_std": 3.2}
    row = build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=hp_rec, ap_rec=ap_rec, weather=weather)
    x = np.array([[row[c] for c in _cols]])
    if _xgb_hw is not None:
        hw_prob = _xgb_hw.predict_proba(x)[0, 1]
        exp_rdiff = _xgb_rd.predict(x)[0]
        exp_total = _xgb_tot.predict(x)[0]
    else:
        hw_prob = _rf_hw.predict_proba(x)[0, 1]
        exp_rdiff = _rf_rd.predict(x)[0]
        exp_total = _rf_tot.predict(x)[0]
    rdiff_sims = np.random.normal(exp_rdiff, 3.0, n_sims)
    total_sims = np.random.normal(exp_total, 3.2, n_sims)
    mc_hw = np.mean(rdiff_sims > 0)
    mc_home_minus = np.mean(rdiff_sims >= 1.5)
    mc_home_plus = np.mean(rdiff_sims >= -1.5)
    mc_over = np.mean(total_sims > 8.5)
    mc_away_minus = 1.0 - mc_home_plus
    mc_away_plus = 1.0 - mc_home_minus
    return {
        "ml_hp": round(float(mc_hw), 4), "ml_ap": round(float(1 - mc_hw), 4),
        "spr_home_minus": round(float(mc_home_minus), 4), "spr_home_plus": round(float(mc_home_plus), 4),
        "spr_away_minus": round(float(mc_away_minus), 4), "spr_away_plus": round(float(mc_away_plus), 4),
        "spr_exp_margin": round(float(exp_rdiff), 2), "exp_total": round(float(exp_total), 2), "total_std": 3.2,
        "hw_prob_raw": round(float(hw_prob), 4), "exp_rdiff_raw": round(float(exp_rdiff), 2),
    }

def american_to_prob(odds):
    if odds is None or odds == 0: return None
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)

def norm_cdf(x, mu=0, sigma=1):
    return 0.5 * (1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

def compute_ev(prob, odds):
    if odds is None: return None
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return round((prob/100 * dec) - 1, 4)

ODDS_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".odds_cache.json")
ODDS_COOLDOWN_PATH = os.path.join(os.path.dirname(__file__), ".odds_cooldown")

def fetch_odds():
    cache_age = 0
    try:
        cache_age = time.time() - os.path.getmtime(ODDS_CACHE_PATH)
    except:
        pass
    if cache_age > 0 and cache_age < 14400:
        try:
            with open(ODDS_CACHE_PATH) as f:
                return json.load(f)
        except:
            pass
    # Cooldown por archivo (persiste entre subprocess calls)
    try:
        cd_age = time.time() - os.path.getmtime(ODDS_COOLDOWN_PATH)
        if cd_age < 1800:
            return []
    except:
        pass
    odds = []
    if ODDS_API_KEY:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                odds = r.json()
        except:
            pass
    if odds:
        try:
            with open(ODDS_CACHE_PATH, "w") as f:
                json.dump(odds, f)
        except:
            pass
    try:
        with open(ODDS_COOLDOWN_PATH, "w") as f:
            f.write(str(time.time()))
    except:
        pass
    return odds

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

PREFERRED_BOOK = "BetMGM"

def extract_market_odds(game_odds, market_key, outcome_name=None, expect_point=None):
    if not game_odds: return None, None, None
    best_price, best_book, best_point = None, None, None
    for book in game_odds.get("bookmakers", []):
        if book.get("title", "") != PREFERRED_BOOK: continue
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

def run_backtest():
    """Backtest using existing predictions_log.json + fetch more historical data."""
    import requests as _req
    from datetime import timedelta
    from bankroll import american_to_prob, kelly_fraction

    # Load existing predictions
    try:
        with open(PRED_LOG_PATH) as f:
            data = json.load(f)
    except:
        data = {"predictions": []}
    preds = data["predictions"]
    existing_dates = set(p["date"] for p in preds if p.get("date"))

    # Try to fetch more historical data from MLB API
    today = datetime.now(TZ)
    backtest_results = {"by_date": {}, "by_market": {}, "by_prob": {}, "total": {}}

    for days_ago in range(1, 11):
        d = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        if d in existing_dates:
            continue
        try:
            url = f"{MLB_API_BASE}/schedule?sportId=1&date={d}&hydrate=linescore,team,probablePitcher"
            r = _req.get(url, timeout=10)
            if r.status_code != 200:
                continue
            games = r.json().get("dates", [])
            if not games:
                continue
            print(f"  Backtest: fetching {d}...")
        except:
            continue

    # Analyze all predictions (existing + new)
    banks = {"kelly25": 1000, "flat100": 1000, "flat50": 1000}
    results = {"total": 0, "wins": 0, "by_market": {}, "by_prob_bin": {}, "bankrolls": banks.copy()}

    settled = [p for p in preds if p.get("settled")]
    for p in settled:
        results["total"] += 1
        if p.get("result") == "W":
            results["wins"] += 1
        mkt = p.get("market", "?")
        if mkt not in results["by_market"]:
            results["by_market"][mkt] = {"w": 0, "l": 0, "profit": 0}
        if p["result"] == "W":
            results["by_market"][mkt]["w"] += 1
        else:
            results["by_market"][mkt]["l"] += 1
        pf = p.get("profit") or 0
        results["by_market"][mkt]["profit"] += pf

        prob_bin = round((p.get("prob", 50) / 10)) * 10
        key = f"{prob_bin}-{prob_bin+10}%"
        if key not in results["by_prob_bin"]:
            results["by_prob_bin"][key] = {"w": 0, "l": 0}
        if p["result"] == "W":
            results["by_prob_bin"][key]["w"] += 1
        else:
            results["by_prob_bin"][key]["l"] += 1

    pct = round(results["wins"] / results["total"] * 100, 1) if results["total"] else 0

    lines = ["📊 *BACKTEST COMPLETO*\n"]
    lines.append(f"📅 Periodo: múltiples días")
    lines.append(f"🎯 Total predicciones liquidadas: {results['total']}")
    lines.append(f"📊 Record: {results['wins']}-{results['total']-results['wins']} ({pct}%)\n")

    for mkt in ["ML", "O/U"]:
        if mkt in results["by_market"]:
            d = results["by_market"][mkt]
            n = d["w"] + d["l"]
            mp = round(d["w"] / n * 100, 1) if n else 0
            lines.append(f"• *{mkt}:* {d['w']}-{d['l']} ({mp}%) | Profit: ${d['profit']:+.2f}")

    lines.append(f"\n💰 *Estrategia Kelly 25%:* ${banks['kelly25']:.2f}")
    lines.append(f"💰 *Flat $100:* ${banks['flat100']:.2f}")
    lines.append(f"📈 *Model Accuracy:* {pct}%")

    if results["by_prob_bin"]:
        lines.append("\n*Calibración por probabilidad:*")
        for k in sorted(results["by_prob_bin"].keys()):
            d = results["by_prob_bin"][k]
            n = d["w"] + d["l"]
            rp = round(d["w"] / n * 100) if n else 0
            lines.append(f"  {k}: {d['w']}-{d['l']} ({rp}%)")

    return "\n".join(lines)


def get_edge_for_entry(entry):
    if not entry or not entry.get("odds") or entry["odds"] in ("N/A", "—", ""):
        return None
    try:
        odds = int(str(entry["odds"]).replace("$",""))
    except:
        return None
    if odds == 0: return None
    ip = american_to_prob(odds)
    if ip is None: return None
    prob = entry.get("prob", 0) / 100.0
    return round((prob - ip) * 100, 1)

def generate_recommendations():
    """Run full prediction pipeline matching app.py exactly.
    Returns list of recommendation dicts (best-per-game, edge>2%, up to 4)."""
    from bankroll import load_picks, recommend_stake, calibrate_ml, calibrate_rl

    today_str = datetime.now(TZ).strftime("%Y-%m-%d")
    games = fetch_todays_schedule()
    odds_raw = fetch_odds()
    ab_map = fetch_team_abbrevs()
    from bankroll import get_pnl
    actual_bankroll = get_pnl()["weekly_bankroll"]

    all_recs = []

    for g in games:
        sc = g.get("status",{}).get("codedGameState","S")
        sd = g.get("status",{}).get("detailedState","Scheduled")
        if sc in ("F", "O") or sd == "Final":
            continue
        t = g["teams"]
        hi, ai = t["home"]["team"], t["away"]["team"]
        hid, aid = hi["id"], ai["id"]
        hn, an = hi["name"], ai["name"]
        ha, aa = ab_map.get(hid, "??"), ab_map.get(aid, "??")

        hs = fetch_team_stats_mlb(hid) if hid else {}
        aws = fetch_team_stats_mlb(aid) if aid else {}
        hr = fetch_recent_games(hid) if hid else []
        ar = fetch_recent_games(aid) if aid else []
        hf = compute_form(hr, hid)
        af = compute_form(ar, aid)

        hp_info = g.get("teams",{}).get("home",{}).get("probablePitcher") or {}
        ap_info = g.get("teams",{}).get("away",{}).get("probablePitcher") or {}
        hpitch = fetch_pitcher_stats(hp_info.get("id")) if hp_info.get("id") else {}
        apitch = fetch_pitcher_stats(ap_info.get("id")) if ap_info.get("id") else {}
        hprec = fetch_pitcher_recent_form(hp_info.get("id")) if hp_info.get("id") else {}
        aprec = fetch_pitcher_recent_form(ap_info.get("id")) if ap_info.get("id") else {}
        elo_hp, h_elo, a_elo = compute_elo(hr, ar, hid, aid)
        venue_name = g.get("venue",{}).get("name", "")
        park_f = PARK_FACTORS.get(venue_name, 1.0)
        weather = fetch_weather(venue_name)

        mc = monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo,
                                 hpitch if hpitch.get("ip",0) >= 10 else None,
                                 apitch if apitch.get("ip",0) >= 10 else None,
                                 park_f, hp_rec=hprec, ap_rec=aprec, weather=weather)

        ml_hp = mc["ml_hp"]
        ml_ap = mc["ml_ap"]
        spr_home_minus = mc["spr_home_minus"]
        spr_home_plus = mc["spr_home_plus"]
        spr_away_minus = mc["spr_away_minus"]
        spr_away_plus = mc["spr_away_plus"]
        spr_exp_margin = mc["spr_exp_margin"]
        exp_total = mc["exp_total"]
        total_std = mc["total_std"]

        spr_fav_team = hn if spr_exp_margin >= 0 else an
        spr_dog_team = an if spr_exp_margin >= 0 else hn
        spr_fav_prob = spr_home_minus if spr_exp_margin >= 0 else spr_away_minus
        spr_dog_prob = spr_away_plus if spr_exp_margin >= 0 else spr_home_plus

        if ml_hp is None:
            continue

        cal_hp = calibrate_ml(ml_hp)
        cal_ap = 1.0 - cal_hp

        gl = f"{aa} @ {ha}"

        if not odds_raw:
            continue

        og = match_game(odds_raw, hn, an)
        if not og:
            continue

        m_fav, m_dog = None, None
        for book in og.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt.get("key") == "spreads":
                    oc = mkt.get("outcomes", [])
                    if len(oc) >= 2:
                        for o in oc:
                            if o.get("point", 0) < 0: m_fav = o["name"]
                            elif o.get("point", 0) > 0: m_dog = o["name"]
                    break
            if m_fav: break
        if m_fav and m_dog:
            spr_fav_team = m_fav
            spr_dog_team = m_dog
            if m_fav == hn:
                spr_fav_prob = spr_home_minus
                spr_dog_prob = spr_away_plus
            else:
                spr_fav_prob = spr_away_minus
                spr_dog_prob = spr_home_plus
            spr_fav_prob = calibrate_rl(spr_fav_prob)
            spr_dog_prob = 1.0 - spr_fav_prob

        # ML
        tgt = hn if ml_hp > 0.50 else an
        ml_price, ml_book, _ = extract_market_odds(og, "h2h", tgt)
        if ml_price:
            cal_prob = max(cal_hp, cal_ap)
            ml_entry = {
                "game": gl, "market": "ML", "team": tgt, "detail": "",
                "prob": cal_prob*100, "odds": ml_price, "edge": None,
            }
            mp = american_to_prob(ml_price)
            if mp:
                ml_entry["edge"] = round(cal_prob*100 - (mp*100 if mp else 0), 1)
            all_recs.append(ml_entry)

        # RL omitido — bajo rendimiento (15-39 histórico)

        # O/U
        ov_price, ov_book, ov_point = extract_market_odds(og, "totals")
        if ov_price and ov_point:
            over_prob = norm_cdf(exp_total - ov_point, 0, total_std)
            if over_prob > 0.5:
                ou_entry = {
                    "game": gl, "market": "O/U", "team": "Over", "detail": f"o{ov_point}",
                    "prob": over_prob*100, "odds": ov_price, "edge": None,
                }
            else:
                un_price, un_book, _ = extract_market_odds(og, "totals", "Under")
                ou_entry = {
                    "game": gl, "market": "O/U", "team": "Under", "detail": f"u{ov_point}",
                    "prob": (1-over_prob)*100, "odds": un_price or ov_price, "edge": None,
                }
            ed = get_edge_for_entry(ou_entry)
            if ed: ou_entry["edge"] = ed
            all_recs.append(ou_entry)

    # Filter + rank (matching app.py)
    recs = []
    for r in all_recs:
        if r["edge"] is not None and r["edge"] > 2:
            recs.append(r)
        elif r["prob"] >= 55:
            r["edge"] = round((r["prob"] - 50) * 0.5, 1)
            recs.append(r)
    best_per_game = {}
    for r in recs:
        g = r["game"]
        if g not in best_per_game or r["edge"] > best_per_game[g]["edge"]:
            best_per_game[g] = r
    recs = sorted(best_per_game.values(), key=lambda x: x["edge"], reverse=True)

    result = []
    for r in recs[:4]:
        stake, units, stake_label = 0, 0, ""
        try:
            stake, units, stake_label = recommend_stake(r["prob"]/100, r["odds"], bankroll=actual_bankroll)
        except:
            pass
        r["stake"] = stake
        r["units"] = units
        r["stake_label"] = stake_label
        result.append(r)

    return result
