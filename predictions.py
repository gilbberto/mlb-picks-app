"""predictions.py — Shared prediction pipeline (mirrors app.py exactly)."""
import os, sys, math, pickle, json
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
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "3988754e84aac800a8ee2eeca88cb085")
BASE = os.path.join(os.path.dirname(__file__), "")

PARK_FACTORS = {
    "Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
    "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
    "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
    "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
    "Oracle Park": 0.98, "Petco Park": 0.97, "Oakland Coliseum": 0.97,
    "Tropicana Field": 0.96, "Target Field": 0.96, "PNC Park": 0.97,
}

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
        _xgb_hw = _xgb_rd = _xgb_tot = None
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
    try:
        url = f"{MLB_API_BASE}/teams/{tid}/stats?season={season}&group=hitting,pitching&stats=season"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
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
    except:
        return {}

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

def build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=None, ap_rec=None):
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
    }
    return {k: v for k, v in f.items() if not k.startswith(("hp_rec_", "ap_rec_"))}

def monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=None, ap_rec=None, n_sims=5000):
    if not _MODELS_LOADED:
        return {"ml_hp": None, "ml_ap": None, "spr_home_minus": None, "spr_home_plus": None, "spr_away_minus": None, "spr_away_plus": None, "spr_exp_margin": None, "exp_total": None, "total_std": 3.2}
    row = build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, hp_rec=hp_rec, ap_rec=ap_rec)
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

def fetch_odds():
    odds = []
    if ODDS_API_KEY:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                odds = r.json()
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
    bk_data = load_picks()
    actual_bankroll = bk_data["bankroll"]

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

        mc = monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo,
                                 hpitch if hpitch.get("ip",0) >= 10 else None,
                                 apitch if apitch.get("ip",0) >= 10 else None,
                                 park_f, hp_rec=hprec, ap_rec=aprec)

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

        # RL -1.5
        spr_price, spr_book, _ = extract_market_odds(og, "spreads", spr_fav_team, expect_point=-1.5)
        if spr_price and spr_fav_prob > 0:
            rl_entry = {
                "game": gl, "market": "RL -1.5", "team": spr_fav_team, "detail": "-1.5",
                "prob": spr_fav_prob*100, "odds": spr_price, "edge": None,
            }
            ed = get_edge_for_entry(rl_entry)
            if ed: rl_entry["edge"] = ed
            all_recs.append(rl_entry)

        # RL +1.5
        spr_dog_price, spr_dog_book, _ = extract_market_odds(og, "spreads", spr_dog_team, expect_point=1.5)
        if spr_dog_price and spr_dog_prob > 0:
            rl_dog_entry = {
                "game": gl, "market": "RL +1.5", "team": spr_dog_team, "detail": "+1.5",
                "prob": spr_dog_prob*100, "odds": spr_dog_price, "edge": None,
            }
            ed = get_edge_for_entry(rl_dog_entry)
            if ed: rl_dog_entry["edge"] = ed
            all_recs.append(rl_dog_entry)

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

    # Filter + rank identical to app.py
    recs = [r for r in all_recs if r["edge"] is not None and r["edge"] > 2]
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
