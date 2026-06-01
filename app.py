import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import pickle
import math
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Chihuahua")
except Exception:
    TZ = timezone(timedelta(hours=-6))  # fallback: UTC-6 fijo

load_dotenv()
if not load_dotenv():
    pass

st.set_page_config(page_title="MLB Picks AI", page_icon="⚾", layout="wide", initial_sidebar_state="expanded")

# ─── Dark mode + Mobile CSS ───
st.markdown("""
<style>
    /* Force dark background */
    .stApp, .main, .block-container, .stApp > header {
        background-color: #0a0a0a !important;
    }
    .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: #e0e0e0 !important;
    }
    a { color: #58a6ff !important; }
    /* Cards */
    div[data-testid="column"] {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 12px;
        margin: 4px 0;
        border: 1px solid #2a2a2a;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111 !important; }
    section[data-testid="stSidebar"] * { color: #ccc !important; }
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 8px 12px;
        border: 1px solid #2a2a2a;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
        color: #e0e0e0 !important;
    }
    /* DataFrames */
    div[data-testid="stDataFrame"] { background: #1a1a1a !important; }
    div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
        color: #e0e0e0 !important;
        background: #1a1a1a !important;
    }
    /* Expanders */
    div[data-testid="stExpander"] { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; }
    /* Dividers */
    hr { border-color: #2a2a2a !important; }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container { padding: 6px 2px !important; max-width: 100% !important; }
        div[data-testid="column"] { padding: 6px !important; }
        h1 { font-size: 20px !important; }
        h2 { font-size: 16px !important; }
        h3 { font-size: 14px !important; }
        p, li, .stMarkdown, span, div { font-size: 13px !important; }
        div[data-testid="stMetric"] { padding: 4px 6px !important; }
        div[data-testid="stMetric"] label { font-size: 11px !important; }
        div[data-testid="stMetric"] div { font-size: 16px !important; }
        /* Make buttons touch-friendly */
        button, .stButton button { min-height: 36px !important; font-size: 13px !important; }
        /* Card containers scroll horizontally if needed */
        div[data-testid="column"] > div { overflow-x: auto !important; }
        /* DataFrames full width with scroll */
        div[data-testid="stDataFrame"] { width: 100% !important; overflow-x: auto !important; }
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
            font-size: 12px !important; padding: 2px 4px !important; white-space: nowrap !important;
        }
        /* Smaller emoji in headings */
        h1 img, h2 img { height: 28px !important; }
        /* Expanders more compact */
        div[data-testid="stExpander"] details { padding: 4px !important; }
        /* Reduce chart height */
        div[data-testid="stAltairChart"] { max-height: 160px !important; }
    }
    @media (max-width: 480px) {
        .block-container { padding: 4px 1px !important; }
        div[data-testid="column"] { padding: 4px !important; margin: 2px 0 !important; }
        p, .stMarkdown, span, div { font-size: 12px !important; }
        button, .stButton button { min-height: 40px !important; font-size: 12px !important; }
        h3 { font-size: 13px !important; }
        div[data-testid="stMetric"] div { font-size: 14px !important; }
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
            font-size: 11px !important; padding: 1px 3px !important;
        }
    }
    /* Spinner */
    div[data-testid="stSpinner"] { color: #58a6ff !important; }
    /* Action buttons: blue accent */
    .stButton button { color: #58a6ff !important; border-color: #58a6ff !important; background: #0d1b2a !important; }
    .stButton button:hover { background: #1a3a5c !important; color: #7cbfff !important; }
</style>
""", unsafe_allow_html=True)

try:
    ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY", "")

# ─── ML Models (XGBoost first, fallback RandomForest + Monte Carlo) ───
_MODELS_LOADED = False
_xgb_hw = _xgb_rd = _xgb_tot = None
_rf_hw = _rf_rd = _rf_tot = None
_cols = None
_model_type = ""

def load_models():
    global _xgb_hw, _xgb_rd, _xgb_tot, _rf_hw, _rf_rd, _rf_tot, _cols, _MODELS_LOADED, _model_type
    base = os.path.join(os.path.dirname(__file__), "")
    # Try XGBoost first
    try:
        import xgboost as xgb
        with open(base + "xgb_hw.pkl", "rb") as f: _xgb_hw = pickle.load(f)
        with open(base + "xgb_rd.pkl", "rb") as f: _xgb_rd = pickle.load(f)
        with open(base + "xgb_tot.pkl", "rb") as f: _xgb_tot = pickle.load(f)
        with open(base + "xgb_cols.pkl", "rb") as f: _cols = pickle.load(f)
        _model_type = "XGBoost"
        _MODELS_LOADED = True
    except Exception as e:
        print(f"XGBoost models not loaded: {e}, trying RF...")
        try:
            with open(base + "rf_hw.pkl", "rb") as f: _rf_hw = pickle.load(f)
            with open(base + "rf_rd.pkl", "rb") as f: _rf_rd = pickle.load(f)
            with open(base + "rf_tot.pkl", "rb") as f: _rf_tot = pickle.load(f)
            with open(base + "rf_cols.pkl", "rb") as f: _cols = pickle.load(f)
            _model_type = "RandomForest"
            _MODELS_LOADED = True
        except Exception as e2:
            print(f"RF models also not loaded: {e2}")

load_models()

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
CURRENT_SEASON = 2026
LG_AVG_RUNS = 4.5

C = {
    "value_high": "#00cc66", "value_med": "#88cc00", "value_low": "#cccc00",
    "no_value": "#ff4444", "card_bg": "#1a1d2e", "accent": "#4da6ff",
}

# ─── MLB Stats API ───

@st.cache_data(ttl=600)
def fetch_todays_schedule():
    today = datetime.now(TZ).strftime("%m/%d/%Y")
    url = f"{MLB_API_BASE}/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    games = []
    for de in resp.json().get("dates", []):
        for g in de.get("games", []):
            games.append(g)
    return games


@st.cache_data(ttl=3600)
def fetch_team_stats_mlb(tid, season=CURRENT_SEASON):
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


@st.cache_data(ttl=86400)
def fetch_team_abbrevs():
    teams = requests.get(f"{MLB_API_BASE}/teams?sportIds=1", timeout=10).json()
    return {t["id"]: t.get("abbreviation","??") for t in teams.get("teams",[])}

@st.cache_data(ttl=3600)
def fetch_pitcher_stats(pid):
    if not pid:
        return {}
    try:
        url = f"{MLB_API_BASE}/people/{pid}/stats?stats=season&season={CURRENT_SEASON}&group=pitching"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return {}
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if splits:
            s = splits[0].get("stat", {})
            ip = s.get("inningsPitched", "0")
            ip_val = 0
            if isinstance(ip, str) and "." in ip:
                parts = ip.split(".")
                ip_val = int(parts[0]) + int(parts[1]) / 3.0 if len(parts) > 1 else float(parts[0])
            else:
                ip_val = float(ip or 0)
            return {"era": safe_float(s.get("era")), "whip": safe_float(s.get("whip")), "ip": ip_val,
                    "k9": safe_float(s.get("strikeoutsPer9Inn")), "bb9": safe_float(s.get("walksPer9Inn")),
                    "hr9": safe_float(s.get("homeRunsPer9")), "name": splits[0].get("player",{}).get("fullName","")}
    except Exception:
        return {}
    return {}

@st.cache_data(ttl=600)
def fetch_recent_games(tid, ng=20):
    today = datetime.now(TZ)
    end = today.strftime("%m/%d/%Y")
    start = (today - timedelta(days=45)).strftime("%m/%d/%Y")
    url = f"{MLB_API_BASE}/schedule?sportId=1&teamId={tid}&startDate={start}&endDate={end}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return []
    games = []
    for de in resp.json().get("dates", []):
        for g in de.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                games.append(g)
    return games[-ng:]


# ─── ESPN API ───

@st.cache_data(ttl=3600)
def fetch_espn_standings():
    try:
        resp = requests.get(f"{ESPN_API_BASE}/standings", timeout=10)
        if resp.status_code != 200:
            return {}
        teams = {}
        for entry in resp.json().get("children", []):
            for ce in entry.get("children", []):
                for te in ce.get("standings", {}).get("entries", []):
                    tid = te.get("team", {}).get("id")
                    if not tid:
                        continue
                    sts = {s["name"]: s["value"] for s in te.get("stats", [])}
                    teams[int(tid)] = {
                        "wins": int(sts.get("wins", 0)), "losses": int(sts.get("losses", 0)),
                        "win_pct": safe_float(sts.get("winPercent", 0)),
                        "streak": sts.get("streak", ""), "last_10": sts.get("last10", "0-0"),
                        "runs_for": int(sts.get("runsScored", 0)), "runs_against": int(sts.get("runsAllowed", 0)),
                    }
        return teams
    except requests.RequestException:
        return {}


# ─── PyBaseball (opcional) ───

@st.cache_data(ttl=7200)
def fetch_advanced_stats():
    try:
        from pybaseball import batting_stats, pitching_stats
        hitters = batting_stats(CURRENT_SEASON, qual=200)
        pitchers = pitching_stats(CURRENT_SEASON, qual=50)
        ba, pa = {}, {}
        if hitters is not None and not hitters.empty:
            for _, r in hitters.iterrows():
                tm = r.get("Team", "")
                ba.setdefault(tm, {"wrc_plus":[],"woba":[],"war":[]})
                ba[tm]["wrc_plus"].append(safe_float(r.get("wRC+")))
                ba[tm]["woba"].append(safe_float(r.get("wOBA")))
                ba[tm]["war"].append(safe_float(r.get("WAR")))
        if pitchers is not None and not pitchers.empty:
            for _, r in pitchers.iterrows():
                tm = r.get("Team", "")
                pa.setdefault(tm, {"fip":[],"xfip":[],"siera":[],"war":[],"k_9":[],"bb_9":[]})
                pa[tm]["fip"].append(safe_float(r.get("FIP")))
                pa[tm]["xfip"].append(safe_float(r.get("xFIP")))
                pa[tm]["siera"].append(safe_float(r.get("SIERA")))
                pa[tm]["war"].append(safe_float(r.get("WAR")))
        return {
            "batting": {t: {k: np.mean(v) if v else 0 for k, v in sts.items()} for t, sts in ba.items()},
            "pitching": {t: {k: np.mean(v) if v else 0 for k, v in sts.items()} for t, sts in pa.items()},
        }
    except Exception:
        return None


# ─── Odds APIs ───

@st.cache_data(ttl=300)
def fetch_odds():
    odds = []
    if ODDS_API_KEY:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_API_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                odds = r.json()
        except Exception:
            pass
    if not odds and SHARPAPI_KEY:
        try:
            url = f"https://api.sharpapi.com/v1/odds/baseball_mlb?apiKey={SHARPAPI_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                odds = r.json().get("data", [])
        except Exception:
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


def extract_market_odds(game_odds, market_key, outcome_name=None):
    if not game_odds:
        return None, None
    best_price, best_book = None, None
    for book in game_odds.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for oc in mkt.get("outcomes", []):
                if outcome_name and oc.get("name") != outcome_name:
                    continue
                price = oc.get("price")
                point = oc.get("point")
                if best_price is None or (price is not None and abs(price) > abs(best_price)):
                    best_price = price
                    best_book = book.get("title", "Unknown")
                    best_point = point
    return best_price, best_book, best_point if market_key != "h2h" else (best_price, best_book)


# ─── Cálculos ───

PARK_FACTORS = {
    "Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
    "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
    "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
    "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
    "Oracle Park": 0.98, "Petco Park": 0.97, "Oakland Coliseum": 0.97,
    "Tropicana Field": 0.96, "Target Field": 0.96, "PNC Park": 0.97,
}

def safe_float(v, d=0.0):
    if v is None: return d
    try: return float(v)
    except: return d


def compute_elo(home_games, away_games, home_id, away_id, k=32):
    seen, ordered = set(), []
    for g in sorted(home_games + away_games, key=lambda x: x.get("gameDate", "")):
        pk = g.get("gamePk")
        if pk not in seen:
            seen.add(pk)
            ordered.append(g)
    elos = {}
    for g in ordered:
        t = g["teams"]
        try:
            hg, ag = t["home"]["team"]["id"], t["away"]["team"]["id"]
            hs, aws_s = int(t["home"].get("score",0)), int(t["away"].get("score",0))
        except (KeyError, ValueError, TypeError):
            continue
        if hs == 0 and aws_s == 0:
            continue
        for eid in (hg, ag):
            elos.setdefault(eid, 1500)
        he, ae = elos[hg], elos[ag]
        exp_h = 1 / (1 + 10 ** ((ae - he - 50) / 400))
        act_h = 1 if hs > aws_s else (0 if hs < aws_s else 0.5)
        marg = min(np.log(abs(hs - aws_s) + 1) / 2.2, 1.5)
        elos[hg] += k * marg * (act_h - exp_h)
        elos[ag] += k * marg * ((1 - act_h) - (1 - exp_h))
    h_elo = elos.get(home_id, 1500)
    a_elo = elos.get(away_id, 1500)
    home_prob = 1 / (1 + 10 ** ((a_elo - h_elo - 50) / 400))
    return home_prob, round(h_elo), round(a_elo)


def compute_form(games, tid):
    if not games:
        return {"wp": 0.5, "rs": 4.5, "ra": 4.5, "rest": 3, "wl": 0, "n": 0}
    games_sorted = sorted(games, key=lambda x: x.get("gameDate", ""), reverse=True)
    w, rs, ra = 0, [], []
    w_exp, rs_exp, ra_exp = 0, 0, 0
    total_w = 0
    for i, g in enumerate(games_sorted):
        t = g["teams"]
        is_h = t["home"]["team"]["id"] == tid
        is_a = t["away"]["team"]["id"] == tid
        if not is_h and not is_a:
            continue
        ms = safe_float(t["home"]["score"] if is_h else t["away"]["score"])
        os = safe_float(t["away"]["score"] if is_h else t["home"]["score"])
        rs.append(ms); ra.append(os)
        if ms > os: w += 1
        wt = max(1.5 - i * 0.12, 0.4) if i < 10 else 0.3
        w_exp += wt * (1 if ms > os else 0)
        rs_exp += wt * ms
        ra_exp += wt * os
        total_w += wt
    n = len(rs) or 1
    last_date = games_sorted[0].get("gameDate", "") if games_sorted else ""
    rest = 3
    if last_date:
        from datetime import datetime
        try:
            ld = datetime.strptime(last_date.split("T")[0], "%Y-%m-%d")
            rest = (datetime.now().date() - ld.date()).days if hasattr(ld, 'date') else 3
        except: pass
    return {"wp": w/n, "rs": np.mean(rs) if rs else 4.5, "ra": np.mean(ra) if ra else 4.5,
            "rest": max(rest, 1), "wp_exp": w_exp / total_w if total_w else 0.5,
            "rs_exp": rs_exp / total_w if total_w else 4.5,
            "ra_exp": ra_exp / total_w if total_w else 4.5}


def log5(a, b):
    d = a*(1-b)+(1-a)*b
    return (a*(1-b))/d if d else 0.5


def pyth(rs, ra):
    return (rs**1.83)/(rs**1.83+ra**1.83) if rs+ra > 0 else 0.5


def norm_cdf(x, mu=0, sigma=1):
    return 0.5 * (1 + __import__("math").erf((x-mu)/(sigma*__import__("math").sqrt(2))))


def compute_ev(prob, odds):
    if odds is None: return None
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return round((prob/100 * dec) - 1, 4)


# ─── RandomForest + Monte Carlo ───

def build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f):
    """Build a feature dict matching the training format."""
    return {
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
        "ap_era": apitch.get("era", 4.5) if apitch else 4.5,
        "ap_k9": apitch.get("k9", 8.0) if apitch else 8.0,
        "ap_bb9": apitch.get("bb9", 3.0) if apitch else 3.0,
        "ap_hr9": apitch.get("hr9", 1.2) if apitch else 1.2,
        "ap_v": 1 if (apitch and apitch.get("ip",0) >= 10) else 0,
    }


@st.cache_data(ttl=3600)
def monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f, n_sims=5000):
    """Run Monte Carlo simulation using trained models. Returns dict."""
    if not _MODELS_LOADED:
        return {"ml_hp": None, "ml_ap": None,
                "spr_fav_prob": None, "spr_dog_prob": None, "spr_exp_margin": None,
                "exp_total": None, "total_std": 3.2}

    row = build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f)
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
    mc_cover_minus = np.mean(rdiff_sims >= 1.5)
    mc_cover_plus = np.mean(rdiff_sims >= -1.5)
    mc_over = np.mean(total_sims > 8.5)

    return {
        "ml_hp": round(float(mc_hw), 4),
        "ml_ap": round(float(1 - mc_hw), 4),
        "spr_fav_prob": round(float(mc_cover_minus), 4),
        "spr_dog_prob": round(float(mc_cover_plus), 4),
        "spr_exp_margin": round(float(exp_rdiff), 2),
        "exp_total": round(float(exp_total), 2),
        "total_std": 3.2,
        "hw_prob_raw": round(float(hw_prob), 4),
        "exp_rdiff_raw": round(float(exp_rdiff), 2),
    }


def american_to_prob(odds):
    if odds is None or odds == 0: return None
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)


def ev_label(ev):
    if ev is None: return "📊 Sin odds", C["no_value"]
    if ev > 0.15: return "🔥 HIGH VALUE", C["value_high"]
    if ev > 0.08: return "✅ VALUE", C["value_med"]
    if ev > 0.03: return "⚠️ LOW VALUE", C["value_low"]
    return "❌ NO VALUE", C["no_value"]


# ─── Telegram ───

def send_telegram(msg):
    try:
        tok = st.secrets.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_TOKEN", ""))
        cid = st.secrets.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID", ""))
        if not tok or not cid: return
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def notify_pick(gl, market, team, stake, odds, bankroll):
    try:
        s = f"📝 *Pick registrado*\n{gl} → {market} {team}\nApuesta: ${stake:.2f} @ {odds:+d}\nBankroll: ${bankroll:.2f}"
        send_telegram(s)
    except: pass


# ─── GitHub sync ───

def sync_picks_to_github():
    try:
        tok = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
        repo = st.secrets.get("REPO", os.environ.get("REPO", ""))
        branch = st.secrets.get("BRANCH", os.environ.get("BRANCH", "main"))
        if not tok or not repo: return
        owner, repo_name = repo.split("/")
        path = "picks.json"
        with open(path, "r") as f:
            content = f.read()
        import base64
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
        headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
        r = requests.get(url + f"?ref={branch}", headers=headers, timeout=10)
        sha = r.json().get("sha", "") if r.ok else ""
        data = {"message": "sync picks.json from app", "content": base64.b64encode(content.encode()).decode(), "branch": branch}
        if sha: data["sha"] = sha
        requests.put(url, json=data, headers=headers, timeout=10)
    except: pass


# ─── Modelos de predicción ───

def predict_moneyline(hs, aws, hf, af, hname, aname, h_adv=None, a_adv=None, hpitch=None, apitch=None, elo_hp=None, park_factor=1.0):
    HFA = 0.04

    l5 = log5(hf["wp"], af["wp"])
    l5_exp = log5(hf.get("wp_exp", hf["wp"]), af.get("wp_exp", af["wp"]))
    hp_pyth = pyth(hf.get("rs_exp", hf["rs"]), hf.get("ra_exp", hf["ra"]))
    ap_pyth = pyth(af.get("rs_exp", af["rs"]), af.get("ra_exp", af["ra"]))
    pyth_r = log5(hp_pyth, ap_pyth)

    ho = hs.get("hitting",{}).get("ops", 0.700) * (1 / (park_factor ** 0.3))
    ao = aws.get("hitting",{}).get("ops", 0.700) * (1 / (park_factor ** 0.3))
    off_r = log5(ho/2, ao/2) if ho+ao > 0 else 0.5

    hp_ops = hs.get("pitching",{}).get("ops", 0.700) * park_factor ** 0.3
    ap_ops = aws.get("pitching",{}).get("ops", 0.700) * park_factor ** 0.3
    def_r = log5(ap_ops/2, hp_ops/2) if hp_ops+ap_ops > 0 else 0.5

    hw = hs.get("pitching",{}).get("whip", 1.35)
    aw_whip = aws.get("pitching",{}).get("whip", 1.35)
    whip_r = aw_whip / (hw + aw_whip) if hw+aw_whip > 0 else 0.5

    sp_r = 0.5
    if hpitch and apitch:
        he_k = max(1 - (hpitch.get("k9") or 7) / 15, 0.1)
        ae_k = max(1 - (apitch.get("k9") or 7) / 15, 0.1)
        he_bb = (hpitch.get("bb9") or 3) / 6
        ae_bb = (apitch.get("bb9") or 3) / 6
        he_hr = (hpitch.get("hr9") or 1.2) / 3
        ae_hr = (apitch.get("hr9") or 1.2) / 3
        he_comp = 0.5 * (hpitch.get("era", 4.5) / 5) + 0.3 * he_k + 0.1 * he_bb + 0.1 * he_hr
        ae_comp = 0.5 * (apitch.get("era", 4.5) / 5) + 0.3 * ae_k + 0.1 * ae_bb + 0.1 * ae_hr
        sp_r = ae_comp / (he_comp + ae_comp) if he_comp + ae_comp > 0 else 0.5

    adv_r = 0.5
    if h_adv and a_adv:
        hwrc = h_adv.get("batting",{}).get("wrc_plus", 100)
        awrc = a_adv.get("batting",{}).get("wrc_plus", 100)
        if hwrc and awrc and hwrc+awrc > 0:
            adv_r = log5(hwrc/200, awrc/200)

    elo_r = elo_hp if elo_hp is not None else 0.5

    rest_adv = 0
    hr = hf.get("rest", 3)
    ar = af.get("rest", 3)
    if hr > ar + 1: rest_adv = 0.015
    elif ar > hr + 1: rest_adv = -0.015

    use_sp = hpitch and apitch and hpitch.get("ip",0) >= 10 and apitch.get("ip",0) >= 10
    base = 0.5
    if use_sp and elo_hp is not None:
        base = 0.22*elo_r + 0.14*pyth_r + 0.10*l5_exp + 0.14*sp_r + 0.12*off_r + 0.10*def_r + 0.10*whip_r + 0.05*adv_r + 0.03*l5
    elif elo_hp is not None:
        base = 0.26*elo_r + 0.20*pyth_r + 0.14*l5_exp + 0.12*off_r + 0.10*def_r + 0.10*whip_r + 0.05*adv_r + 0.03*l5
    elif use_sp:
        base = 0.18*l5 + 0.18*pyth_r + 0.14*sp_r + 0.12*off_r + 0.10*def_r + 0.10*whip_r + 0.10*hf.get("wp_exp",hf["wp"]) + 0.05*adv_r + 0.03*l5_exp
    else:
        base = 0.25*l5 + 0.22*pyth_r + 0.15*off_r + 0.12*def_r + 0.12*whip_r + 0.08*hf.get("wp_exp",hf["wp"]) + 0.03*adv_r + 0.03*l5_exp

    home_p = min(max(base + HFA + rest_adv, 0.01), 0.99)
    return home_p, 1-home_p


def predict_spread(hf, af, park_factor=1.0):
    h_exp = (hf.get("rs_exp", hf["rs"]) * af.get("ra_exp", af["ra"])) / LG_AVG_RUNS * park_factor
    a_exp = (af.get("rs_exp", af["rs"]) * hf.get("ra_exp", hf["ra"])) / LG_AVG_RUNS * park_factor
    exp_margin = h_exp - a_exp
    std = 3.0
    abs_margin = abs(exp_margin)
    prob_fav_minus = 1 - norm_cdf(1.5, abs_margin, std)
    prob_dog_plus = 1 - norm_cdf(-1.5, abs_margin, std)
    return prob_fav_minus, prob_dog_plus, exp_margin


def predict_totals(hf, af, hs, aws, h_adv=None, a_adv=None, park_factor=1.0):
    adj = 1.0
    if h_adv:
        hfip = h_adv.get("pitching",{}).get("xfip", 4.5)
        afip = a_adv.get("pitching",{}).get("xfip", 4.5)
        if hfip and afip:
            adj = (hfip + afip) / 9.0
    h_exp = (hf["rs"] * af["ra"]) / LG_AVG_RUNS
    a_exp = (af["rs"] * hf["ra"]) / LG_AVG_RUNS
    exp_total = (h_exp + a_exp) * adj * park_factor

    std = 3.2
    return round(exp_total, 2), std


# ─── UI ───

def _log_pick_fn(pick, mkt_key, mkt_label, entry):
    """Helper to log a pick to the tracker. Returns True on success."""
    try:
        from bankroll import add_pick, load_picks, recommend_stake
        data = load_picks()
        bk = data["bankroll"]
        gl = f"{pick['away_abbrev']} @ {pick['home_abbrev']}"
        odds_str = entry.get("odds", "N/A")
        try: odds_int = int(str(odds_str).replace("$",""))
        except: odds_int = 0
        prob = entry.get("prob", 50) / 100.0
        stake, units, slabel = recommend_stake(prob, odds_int, bankroll=bk)
        if stake <= 0:
            return False
        pick_team = entry.get("pick", "")
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        pid = add_pick(today, gl, mkt_label, prob, odds_int, stake, bk, slabel, pick_team)
        notify_pick(gl, mkt_label, pick_team, stake, odds_int, bk)
        sync_picks_to_github()
        return True
    except Exception as e:
        return False

def render_card(pick, key_suffix="", game_idx=0):
    hn = pick["home_team"]
    an = pick["away_team"]
    hp = pick["ml_home_prob"]
    ap = pick["ml_away_prob"]

    srcs = ["MLB"]
    if pick.get("espn_data"): srcs.append("ESPN")
    if pick.get("advanced_used"): srcs.append("SABR")

    border_c = pick.get("border_color", "#444")

    # Count recommended markets (positive EV)
    rec_count = 0
    for mk in ("moneyline", "spread", "total"):
        m = pick.get(mk, {})
        if m.get("ev") is not None and m["ev"] > 0:
            rec_count += 1

    with st.container():
        pitcher_line = ""
        hp_name = pick.get("home_pitcher", "")
        ap_name = pick.get("away_pitcher", "")
        venue = pick.get("venue", "")
        if hp_name or ap_name:
            pitcher_line = f"  \n`🔄` {ap_name or '?'} @ {hp_name or '?'}"
        if venue:
            pitcher_line += f"  \n`🏟️` {venue}"
        badge = f" ⭐ **{rec_count}**" if rec_count else ""
        cgs = pick.get("coded_game_state", "")
        game_dt = pick.get("game_dt")
        now_tz = datetime.now(TZ)
        status_label = ""
        status_col = ""
        if cgs == "I":
            status_label = "🔴 EN VIVO "
            status_col = "#ff4444"
        elif cgs != "F" and game_dt is not None:
            mins_to_start = (game_dt - now_tz).total_seconds() / 60.0
            if 0 <= mins_to_start <= 15:
                status_label = "⏳ POR INICIAR "
                status_col = "#ffaa00"
            elif mins_to_start < 0:
                status_label = "🔴 EN VIVO "
                status_col = "#ff4444"
        score_str = f"**{pick['final']}** " if pick.get("final") else ""
        time_str = f"🕐 {pick['game_time']}  " if pick.get("game_time") else ""
        st.markdown(f"### {time_str}{status_label}{score_str}**{an}** @ **{hn}**" + "".join(f" `{s}`" for s in srcs) + badge + pitcher_line)
        mkt_list = [("moneyline", "ML", "💰")]
        mkt_list += [("spread_minus", "RL -1.5", "📏"), ("spread_plus", "RL +1.5", "📏")]
        mkt_list += [("total", "O/U", "📈")]
        for mkt_key, mkt_label, mkt_icon in mkt_list:
            p = pick.get(mkt_key, {})
            if not p:
                continue
            ev = p.get("ev")
            ev_str = f"{ev:.1%}" if ev is not None else "N/A"
            ev_lbl, ev_clr = ev_label(ev)
            pick_name = p.get("pick", "—")
            prob_val = p.get("prob")
            prob_str = f"{prob_val:.0f}%" if prob_val is not None else ""
            odds = p.get("odds", "N/A")
            book = p.get("book", "")
            detail = p.get("detail", "")
            edge = p.get("edge")
            recommended = ev is not None and ev > 0

            col_a, col_b, col_c = st.columns([1.4, 0.9, 1.0])
            with col_a:
                rec_tag = " ⭐" if recommended else ""
                detail_display = f" {detail}" if detail and mkt_label != "O/U" else ""
                st.markdown(f"**{mkt_icon} {mkt_label}{rec_tag}**  \n`{pick_name}`{detail_display}")
            with col_b:
                line = f"**{prob_str}**" if prob_str else ""
                if ev is not None:
                    line += f"  \n{ev_str}"
                st.markdown(line if line else "")
            with col_c:
                if odds and odds != "N/A":
                    st.markdown(f"`{odds}`")
                log_key = f"lg_{pick.get('game_id','')}_{mkt_key}"
                if st.session_state.get(log_key, False):
                    st.markdown("<span style='color:#00cc66'>✅</span>", unsafe_allow_html=True)
                elif edge is not None and edge > 2:
                    btn = st.button("📝", key=log_key)
                    if btn:
                        try:
                            from bankroll import add_pick, load_picks, recommend_stake
                            d = load_picks(); bk = d["bankroll"]
                            gl = f"{pick['away_abbrev']} @ {pick['home_abbrev']}"
                            os_ = str(odds) if odds and odds != "N/A" else ""
                            oi = int(str(os_).replace("$","")) if os_ not in ("N/A","—","") else 0
                            pv = prob_val/100.0 if prob_val is not None else 0.5
                            sk,_,sl = recommend_stake(pv, oi, bankroll=bk)
                            if sk > 0:
                                ts = datetime.now(TZ).strftime("%Y-%m-%d")
                                add_pick(ts, gl, mkt_label, pv, oi, sk, bk, sl, pick_name, detail)
                                notify_pick(gl, mkt_label, pick_name, sk, oi, bk)
                                sync_picks_to_github()
                                st.session_state[log_key] = True
                            else:
                                st.caption("⚠️ Kelly=0")
                        except Exception as ex:
                            st.caption(f"❌ {ex}")
                elif recommended:
                    st.markdown("<span style='color:#ffcc00'>⭐</span>", unsafe_allow_html=True)
                elif edge is not None:
                    ec = "#00cc66" if edge > 5 else "#88cc00" if edge > 2 else "#cccc00"
                    st.markdown(f"<span style='color:{ec}'>{edge:+.1f}%</span>", unsafe_allow_html=True)
                elif book:
                    st.markdown(f"{book}" if book else "")
        st.markdown(f"*Prob: {hn} {hp:.0f}% / {an} {ap:.0f}%*")
        st.divider()


def _check_pybaseball():
    try:
        import pybaseball
        return True
    except ImportError:
        return False


def american_to_decimal(odds):
    if odds is None: return None
    return 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)


def generate_parlays(picks, top_n=3):
    legs = []
    for p in picks:
        if p.get("status") == "Final":
            continue
        gid = p.get("game_id", "")
        for mkt_key, mkt_label in [("moneyline","ML"), ("spread_minus","RL -1.5"), ("spread_plus","RL +1.5"), ("total","O/U")]:
            m = p.get(mkt_key, {})
            ev = m.get("ev")
            odds = m.get("odds")
            prob = m.get("prob")

            if prob is None or prob <= 0:
                continue

            prob_dec = prob / 100.0

            has_odds = odds is not None and odds != "N/A"
            has_ev = ev is not None and ev > 0
            threshold = 45 if mkt_label in ("RL -1.5","RL +1.5") else 50
            has_confidence = prob >= threshold

            if not has_ev and not has_confidence:
                continue

            score = ev if has_ev else (prob_dec * 100 - 50)

            if has_odds and odds != "N/A":
                dec_val = american_to_decimal(odds)
            else:
                dec_val = 1.0 / prob_dec if prob_dec > 0 else 1.0

            legs.append({
                "game_id": gid,
                "matchup": f"{p['away_abbrev']} @ {p['home_abbrev']}",
                "market": mkt_label,
                "team": m.get("pick", ""),
                "detail": m.get("detail", ""),
                "odds": odds if (has_odds and odds != "N/A") else "—",
                "decimal": dec_val,
                "prob": prob_dec,
                "ev": ev,
                "score": score,
                "home_team": p["home_team"],
                "away_team": p["away_team"],
            })

    if len(legs) < 3:
        return []

    has_any_ev = any(l["ev"] is not None for l in legs)
    sort_key = "ev" if has_any_ev else "score"
    legs.sort(key=lambda x: x[sort_key] or 0, reverse=True)

    parlays = []

    # Parlay: Best value (EV) — only when real odds exist
    if has_any_ev:
        selected, seen = [], set()
        for leg in legs:
            if leg["game_id"] not in seen and len(selected) < 3 and leg.get("ev") and leg["ev"] > 0:
                selected.append(leg)
                seen.add(leg["game_id"])
        if len(selected) == 3:
            dec = 1.0
            jp = 1.0
            for leg in selected:
                dec *= leg["decimal"]
                jp *= leg["prob"]
            parlays.append({
                "name": "🔥 Best Value",
                "desc": "Top 3 picks por valor esperado (distintos juegos)",
                "legs": selected,
                "decimal_odds": round(dec, 2),
                "joint_prob": jp,
                "american": int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1))),
            })

    return parlays[:3]


def render_parlay(parlay, idx):
    payout_10 = round(10 * parlay["decimal_odds"] - 10, 2)
    payout_25 = round(25 * parlay["decimal_odds"] - 25, 2)
    payout_50 = round(50 * parlay["decimal_odds"] - 50, 2)
    ev_parlay = parlay['joint_prob']*parlay['decimal_odds'] - 1

    rows = []
    for leg in parlay["legs"]:
        prob_pct = f"{leg['prob']*100:.0f}%"
        detail_str = f" {leg['detail']}" if leg.get("detail") and leg.get("market") != "O/U" else ""
        rows.append({
            "Partido": leg['matchup'],
            "Mercado": leg['market'],
            "Pick": f"{leg['team']}{detail_str}",
            "Cuota": leg['odds'],
            "Prob": prob_pct,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"**Parlay {parlay['american']:+d}** · Dec: {parlay['decimal_odds']}x · Prob: {parlay['joint_prob']*100:.1f}% · EV: {ev_parlay:.1%}  \n"
               f"$10 → ${payout_10:.2f}  |  $25 → ${payout_25:.2f}  |  $50 → ${payout_50:.2f}")


# ─── Main ───

def main():
    # ── Auto-settlement al iniciar ──
    try:
        from bankroll import auto_settle
        settled, errors = auto_settle()
        if settled > 0:
            st.success(f"✅ {settled} picks liquidados automáticamente contra resultados MLB")
        if errors:
            with st.expander(f"⚠️ {len(errors)} errores de liquidación"):
                for e in errors:
                    st.caption(e)
    except ImportError:
        pass
    except Exception as _ae:
        st.caption(f"⚠️ Auto-settle: {type(_ae).__name__}")

    with st.sidebar:
        st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", width=55)
        st.markdown("### MLB Picks AI")
        st.markdown(f"**Modelo:** {_model_type} + Monte Carlo (27 vars)" if _model_type else "**Modelo:** No cargado")
        st.markdown("**3 mercados:** Moneyline · Run Line · Over/Under")
        st.divider()
        min_conf = st.selectbox("Filtrar por confianza", ["Todas","🔥 HIGH VALUE","✅ VALUE","⚠️ LOW VALUE"], index=0)
        require_odds = st.checkbox("Requiere odds del mercado", value=True)
        use_adv = st.checkbox("Usar sabermetrics (pybaseball)", value=True)
        st.divider()
        st.markdown("#### ⚙️ APIs")
        for n, ok, d in [
            ("MLB Stats API", True, "Siempre disponible"),
            ("ESPN API", True, "Siempre disponible"),
            ("PyBaseball", _check_pybaseball(), "Opcional"),
            ("The Odds API", bool(ODDS_API_KEY), f"{'✅' if ODDS_API_KEY else '❌'}"),
            ("SharpAPI", bool(SHARPAPI_KEY), f"{'✅' if SHARPAPI_KEY else '❌'}"),
        ]:
            st.markdown(f"{'✅' if ok else '⬜'} **{n}** — {d}")
        if not ODDS_API_KEY and not SHARPAPI_KEY:
            st.divider()
            st.markdown("🔌 APIs de odds gratis: the-odds-api.com · sharpapi.io")
        st.divider()
        st.markdown("#### 📊 Metodología")
        with st.expander("Ver"):
            st.markdown(f"""
            **{_model_type} + Monte Carlo**: 27 features (Elo, forma, OPS/WHIP/ERA, park factor, pitcher real).
            **Entrenado**: 2023–2026 (8,174 juegos, 1,385 pitcher-season stats).
            **Calibración ML**: Ajuste lineal por tramos según validación.
            **Kelly Criterion**: 25% fraccional para sizing de apuestas.

            **Value**: Cuando la prob del modelo supera la prob implícita de la cuota.
            """)
        try:
            from bankroll import get_pnl
            pnl = get_pnl()
            st.divider()
            st.markdown("#### 💰 P&L Tracker")
            c1, c2 = st.columns(2)
            c1.metric("Bankroll", f"${pnl['bankroll']:.0f}")
            c2.metric("Profit", f"${pnl['profit']:+.0f}", delta=f"{pnl['roi']:+.0f}%")
            st.caption(f"{pnl['wins']}-{pnl['losses']} ({pnl['pct']}%) · {pnl['open']} pendientes")
        except ImportError:
            pass
        st.divider()
        st.caption(f"🕐 {datetime.now(TZ).strftime('%H:%M')} Chihuahua")
        st.caption("MLB Picks AI v3.0 · Solo informativo · No garantiza ganancias")

    st.markdown("""
    <div style="text-align:center;padding:16px 0 6px;">
        <div style="display:flex;align-items:center;justify-content:center;gap:12px;">
            <img src="https://www.mlbstatic.com/team-logos/league-on-dark/1.svg" style="height:42px;">
            <h1 style="font-size:40px;margin:0;letter-spacing:-1px;">MLB Picks AI</h1>
        </div>
        <p style="color:#888;font-size:15px;margin-top:4px;">
            Predicciones: Moneyline · Run Line (-1.5) · Over/Under
        </p>
    </div>
    """, unsafe_allow_html=True)

    today_str = datetime.now(TZ).strftime("%A, %d %B %Y")
    st.markdown(f'<div style="text-align:center;margin-bottom:16px;"><span style="color:#888;font-size:13px;">{today_str}</span></div>', unsafe_allow_html=True)

    with st.spinner("🔄 Cargando juegos..."):
        games = fetch_todays_schedule()
    if games is None or games == []:
        st.warning("No hay juegos hoy.")
        return

    c1, c2, c3 = st.columns(3)
    live_count = sum(1 for g in games if g.get("status",{}).get("codedGameState") == "I")
    soon_count = 0
    alive_count = 0
    for g in games:
        if g.get("status",{}).get("codedGameState") == "F":
            continue
        gd = g.get("gameDate","")
        try:
            gt = datetime.fromisoformat(gd.replace("Z","+00:00")).astimezone(TZ)
            mins = (gt - datetime.now(TZ)).total_seconds() / 60.0
            if 0 <= mins <= 15:
                soon_count += 1
            elif mins < 0:
                alive_count += 1
        except:
            pass
    with c1: st.metric("Juegos", len(games))
    with c2: st.metric("Temporada", CURRENT_SEASON)
    with c3: st.metric("En vivo / Por iniciar", f"🔴{live_count + alive_count} ⏳{soon_count}" if live_count or soon_count or alive_count else "0")

    with st.spinner("📡 Cargando datos..."):
        odds_raw = fetch_odds()
        espn_std = fetch_espn_standings()
        adv = fetch_advanced_stats() if use_adv and _check_pybaseball() else None

    if adv: st.success("✅ Sabermetrics activas (wRC+, FIP, SIERA)")
    elif use_adv and not _check_pybaseball():
        st.info("ℹ️ pip install pybaseball para métricas avanzadas")

    st.divider()

    picks = []
    for g in games:
        sc = g.get("status",{}).get("codedGameState","S")
        sd = g.get("status",{}).get("detailedState","Scheduled")
        t = g["teams"]
        hi, ai = t["home"]["team"], t["away"]["team"]
        hid, aid = hi["id"], ai["id"]
        hn, an = hi["name"], ai["name"]
        ab_map = fetch_team_abbrevs()
        ha, aa = ab_map.get(hid, "??"), ab_map.get(aid, "??")

        with st.spinner(f"Analizando {aa} @ {ha}..."):
            hs = fetch_team_stats_mlb(hid) if hid else {}
            aws = fetch_team_stats_mlb(aid) if aid else {}
            hr = fetch_recent_games(hid) if hid else []
            ar = fetch_recent_games(aid) if aid else []
            hf = compute_form(hr, hid)
            af = compute_form(ar, aid)

            h_adv_fg, a_adv_fg = {}, {}
            adv_used = False
            if adv:
                ha_upper = ha.upper()
                aa_upper = aa.upper()
                hb = adv.get("batting",{}).get(ha_upper,{})
                ab = adv.get("batting",{}).get(aa_upper,{})
                hp2 = adv.get("pitching",{}).get(ha_upper,{})
                ap2 = adv.get("pitching",{}).get(aa_upper,{})
                h_adv_fg = {"batting": hb, "pitching": hp2}
                a_adv_fg = {"batting": ab, "pitching": ap2}
                if hb or ab or hp2 or ap2: adv_used = True

            # ── Starting pitchers ──
            hp_info = g.get("teams",{}).get("home",{}).get("probablePitcher") or {}
            ap_info = g.get("teams",{}).get("away",{}).get("probablePitcher") or {}
            hpitch = fetch_pitcher_stats(hp_info.get("id")) if hp_info.get("id") else {}
            apitch = fetch_pitcher_stats(ap_info.get("id")) if ap_info.get("id") else {}
            h_pitcher_name = hpitch.get("name", hp_info.get("fullName", ""))
            a_pitcher_name = apitch.get("name", ap_info.get("fullName", ""))

            # ── Elo ratings ──
            elo_hp, h_elo, a_elo = compute_elo(hr, ar, hid, aid)

            # ── Park factor ──
            venue_name = g.get("venue",{}).get("name", "")
            park_f = PARK_FACTORS.get(venue_name, 1.0)

            # ── RandomForest + Monte Carlo ──
            mc = monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo,
                                     hpitch if hpitch.get("ip",0) >= 10 else None,
                                     apitch if apitch.get("ip",0) >= 10 else None,
                                     park_f)

            ml_hp = mc["ml_hp"]
            ml_ap = mc["ml_ap"]
            spr_fav_prob = mc["spr_fav_prob"]
            spr_dog_prob = mc["spr_dog_prob"]
            spr_exp_margin = mc["spr_exp_margin"]
            exp_total = mc["exp_total"]
            total_std = mc["total_std"]

            spr_fav_team = hn if spr_exp_margin >= 0 else an
            spr_dog_team = an if spr_exp_margin >= 0 else hn

            if ml_hp is None:
                # Fallback: use old model
                ml_hp, ml_ap = predict_moneyline(hs, aws, hf, af, hn, an,
                                                 h_adv_fg if adv_used else None,
                                                 a_adv_fg if adv_used else None,
                                                 hpitch if hpitch.get("ip",0) >= 10 else None,
                                                 apitch if apitch.get("ip",0) >= 10 else None,
                                                 elo_hp, park_f)
                spr_fav_prob, spr_dog_prob, spr_exp_margin = predict_spread(hf, af, park_f)
                spr_fav_team = hn if spr_exp_margin >= 0 else an
                spr_dog_team = an if spr_exp_margin >= 0 else hn
                exp_total, total_std = predict_totals(hf, af, hs, aws,
                                                      h_adv_fg if adv_used else None,
                                                      a_adv_fg if adv_used else None,
                                                      park_f)
            else:
                ml_ap = 1 - ml_hp

            # Calibrate ML and RL probabilities based on validation data
            try:
                from bankroll import calibrate_ml, calibrate_rl
                if ml_hp is not None:
                    cal_hp = calibrate_ml(ml_hp)
                    cal_ap = 1.0 - cal_hp
                else:
                    cal_hp, cal_ap = None, None
                spr_fav_prob = calibrate_rl(spr_fav_prob)
                spr_dog_prob = 1.0 - spr_fav_prob
            except ImportError:
                cal_hp, cal_ap = ml_hp, ml_ap

            over_prob = norm_cdf(exp_total - 8.5, 0, total_std)
            ov_verdict = "Over" if over_prob > 0.5 else "Under"
            ov_pct = round(max(over_prob, 1 - over_prob) * 100, 1)

            gd = g.get("gameDate","")
            game_dt = None
            game_time_str = ""
            try:
                utc_dt = datetime.fromisoformat(gd.replace("Z","+00:00"))
                game_dt = utc_dt.astimezone(TZ)
                game_time_str = game_dt.strftime("%I:%M %p").lstrip("0")
            except:
                pass

            pick_entry = {
                "home_team": hn, "away_team": an,
                "home_abbrev": ha, "away_abbrev": aa,
                "ml_home_prob": round(ml_hp*100,1),
                "ml_away_prob": round(ml_ap*100,1),
                "ml_home_prob_cal": round(cal_hp*100, 1) if cal_hp else None,
                "ml_away_prob_cal": round(cal_ap*100, 1) if cal_ap else None,
                "status": sd, "coded_game_state": sc, "game_id": g.get("gamePk",""), "game_time": game_time_str, "game_dt": game_dt,
                "advanced_used": adv_used, "espn_data": bool(espn_std),
                "exp_total": exp_total, "total_std": total_std,
                "spr_fav_team": spr_fav_team, "spr_fav_prob": spr_fav_prob,
                "spr_dog_team": spr_dog_team, "spr_dog_prob": spr_dog_prob,
                "spr_exp_margin": spr_exp_margin,
                "home_pitcher": h_pitcher_name, "away_pitcher": a_pitcher_name,
                "home_elo": h_elo, "away_elo": a_elo,
                "venue": venue_name,
            }

            if sc == "F":
                pick_entry["final"] = f"{t['away']['score']} - {t['home']['score']}"
            elif sc == "I" and t.get("away",{}).get("score") is not None:
                pick_entry["final"] = f"{t['away']['score']} - {t['home']['score']}"
            else:
                pick_entry["final"] = None

            # ── Odds ──
            border = "#444"
            if odds_raw:
                og = match_game(odds_raw, hn, an)
                if og:
                    # Moneyline odds
                    tgt = hn if ml_hp > 0.50 else an
                    ml_price, ml_book, _ = extract_market_odds(og, "h2h", tgt)
                    if ml_price:
                        # Use calibrated prob for edge calculation
                        cal_prob = max(cal_hp, cal_ap) if cal_hp else max(ml_hp, ml_ap)
                        ev_ml = compute_ev(cal_prob*100, ml_price)
                        mp = american_to_prob(ml_price)
                        edge = round(cal_prob*100 - (mp*100 if mp else 0), 1) if mp else None
                        pick_entry["moneyline"] = {
                            "pick": tgt, "prob": cal_prob*100,
                            "raw_prob": max(ml_hp, ml_ap)*100,
                            "ev": ev_ml, "odds": ml_price, "book": ml_book, "edge": edge,
                        }
                        l, lc = ev_label(ev_ml)
                        if l in ("🔥 HIGH VALUE", "✅ VALUE"): border = lc
                    else:
                        pick_entry["moneyline"] = {"pick": tgt, "prob": max(cal_hp, cal_ap)*100 if cal_hp else max(ml_hp, ml_ap)*100, "raw_prob": max(ml_hp, ml_ap)*100, "ev": None, "odds": "N/A", "book": "", "edge": None}

                    # Spread odds — favorite -1.5
                    spr_price, spr_book, _ = extract_market_odds(og, "spreads", spr_fav_team)
                    if spr_price and spr_fav_prob > 0:
                        ev_spr = compute_ev(spr_fav_prob*100, spr_price)
                        pick_entry["spread_minus"] = {
                            "pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5",
                            "ev": ev_spr, "odds": spr_price, "book": spr_book,
                        }
                        l_spr, _ = ev_label(ev_spr)
                        if l_spr in ("🔥 HIGH VALUE", "✅ VALUE"): border = spr_book and "#00cc66" or border
                    else:
                        pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": ""}

                    # Spread odds — underdog +1.5
                    spr_dog_price, spr_dog_book, _ = extract_market_odds(og, "spreads", spr_dog_team)
                    if spr_dog_price and spr_dog_prob > 0:
                        ev_spr_dog = compute_ev(spr_dog_prob*100, spr_dog_price)
                        pick_entry["spread_plus"] = {
                            "pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5",
                            "ev": ev_spr_dog, "odds": spr_dog_price, "book": spr_dog_book,
                        }
                    else:
                        pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": ""}

                    # Totals odds
                    ov_price, ov_book, ov_point = extract_market_odds(og, "totals")
                    if ov_price and ov_point:
                        over_prob = norm_cdf(exp_total - ov_point, 0, total_std)
                        if over_prob > 0.5:
                            pick_entry["total"] = {
                                "pick": "Over", "prob": over_prob*100,
                                "detail": f"o{ov_point}", "odds": ov_price, "book": ov_book,
                                "ev": compute_ev(over_prob*100, ov_price),
                            }
                        else:
                            un_price, un_book, _ = extract_market_odds(og, "totals", "Under")
                            pick_entry["total"] = {
                                "pick": "Under", "prob": (1-over_prob)*100,
                                "detail": f"u{ov_point}", "odds": un_price or ov_price,
                                "book": un_book or ov_book,
                                "ev": compute_ev((1-over_prob)*100, un_price or ov_price),
                            }
                    else:
                        pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"~{exp_total:.1f}", "ev": None, "odds": "N/A", "book": ""}
                else:
                    cal_p = max(cal_hp, cal_ap) if cal_hp else max(ml_hp, ml_ap)
                    raw_p = max(ml_hp, ml_ap)
                    pick_entry["moneyline"] = {"pick": hn if ml_hp > 0.50 else an, "prob": cal_p*100, "raw_prob": raw_p*100, "ev": None, "odds": "N/A", "book": "", "edge": None}
                    pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": ""}
                    pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": ""}
                    pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"~{exp_total:.1f}", "ev": None, "odds": "N/A", "book": ""}
            else:
                cal_p = max(cal_hp, cal_ap) if cal_hp else max(ml_hp, ml_ap)
                raw_p = max(ml_hp, ml_ap)
                pick_entry["moneyline"] = {"pick": hn if ml_hp > 0.50 else an, "prob": cal_p*100, "raw_prob": raw_p*100, "ev": None, "odds": "N/A", "book": "", "edge": None}
                pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": ""}
                pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": ""}
                pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"~{exp_total:.1f}", "ev": None, "odds": "N/A", "book": ""}

            pick_entry["border_color"] = border
            picks.append(pick_entry)

    if not picks:
        return

    try:
        df = pd.DataFrame(picks)
    except Exception as _e:
        st.warning(f"⚠️ Error generando tabla ({type(_e).__name__})")
        _show_dbg = st.checkbox("Ver detalle del error", value=False, key="dbg_picks")
        if _show_dbg:
            st.exception(_e)
        return

    ev_filter_map = {
        "🔥 HIGH VALUE": lambda x: x is not None and x > 0.15,
        "✅ VALUE": lambda x: x is not None and x > 0.08,
        "⚠️ LOW VALUE": lambda x: x is not None and x > 0.03,
        "Todas": lambda x: True,
    }

    def best_ev(row):
        for m in ["moneyline","spread","total"]:
            p = row.get(m, {})
            if p and p.get("ev") is not None:
                return p["ev"]
        return None

    if min_conf != "Todas":
        f = ev_filter_map[min_conf]
        df = df[df.apply(lambda r: f(best_ev(r)), axis=1)]

    if require_odds and odds_raw:
        upcoming = df[df["status"] != "Final"]
    else:
        upcoming = df

    completed = df[df["status"] == "Final"]

    if len(upcoming) > 0:
        st.markdown(f"### 📋 Picks del Día ({len(upcoming)} juegos)")
        flat_rows = []
        now_tz = datetime.now(TZ)
        for _, r in upcoming.iterrows():
            gl = f"{r['away_abbrev']} @ {r['home_abbrev']}"
            cgs = r.get("coded_game_state", "")
            game_dt = r.get("game_dt")
            score = r.get("final", "")
            time_str = r.get("game_time", "")
            if cgs == "I" and score:
                t = f"🔴 {score}"
            elif cgs == "I":
                t = "🔴 EN VIVO"
            elif game_dt is not None:
                mins = (game_dt - now_tz).total_seconds() / 60.0
                if 0 <= mins <= 15:
                    t = f"⏳ {time_str}"
                elif mins < 0:
                    t = f"🔴 {score}" if score else "🔴 EN VIVO"
                else:
                    t = time_str
            else:
                t = time_str
            for mk, ml in [("moneyline","ML"),("spread_minus","RL-1.5"),("spread_plus","RL+1.5"),("total","O/U")]:
                p = r.get(mk)
                if not p: continue
                odds_val = p.get("odds","N/A")
                prob_val = p.get("prob")
                edge = None
                if prob_val and odds_val not in ("N/A","—",""):
                    try:
                        oi = int(str(odds_val).replace("$",""))
                        ip = american_to_prob(oi)
                        if ip: edge = round(prob_val - ip * 100, 1)
                    except: pass
                pick_name = p.get("pick","—")
                display_pick = f"🔥 {pick_name}" if (edge is not None and edge > 2) else pick_name
                flat_rows.append({
                    "Juego": gl, "Hora": t, "M": ml,
                    "Pick": display_pick,
                    "Prob": f"{prob_val:.0f}%" if prob_val else "",
                    "Odds": odds_val,
                    "EV": f"{p['ev']:.1%}" if p.get("ev") is not None else "",
                })
        if flat_rows:
            st.dataframe(pd.DataFrame(flat_rows), column_config={
                "Juego": st.column_config.TextColumn("Juego", width="medium"),
                "Hora": st.column_config.TextColumn("Hora", width="small"),
                "M": st.column_config.TextColumn("M", width="small"),
                "Pick": st.column_config.TextColumn("Pick", width="medium"),
                "Prob": st.column_config.TextColumn("Prob", width="small"),
                "Odds": st.column_config.TextColumn("Odds", width="small"),
                "EV": st.column_config.TextColumn("EV", width="small"),
            }, hide_index=True, use_container_width=True)
            st.caption("Usa la sección 🏆 Recomendaciones para registrar tus picks.")
            st.divider()
            st.markdown("##### ✏️ Registrar Picks")
            from bankroll import recommend_stake, load_picks, add_pick
            bk_data = load_picks(); act_bk = bk_data["bankroll"]
            reg_rows = []
            for _, r in upcoming.iterrows():
                gl = f"{r['away_abbrev']} @ {r['home_abbrev']}"
                gid = r.get("game_id","")
                row = {"Juego": gl}
                for mk, base_ml in [("moneyline","ML"),("spread_minus","RL-1.5"),("spread_plus","RL+1.5"),("total","O/U")]:
                    p_dat = r.get(mk)
                    if not p_dat:
                        row[base_ml] = "—"
                    else:
                        pick_name = p_dat.get("pick", "")
                        if mk == "total":
                            ml = pick_name
                        else:
                            abb = (r["home_abbrev"] if pick_name == r.get("home_team","") else
                                   r["away_abbrev"] if pick_name == r.get("away_team","") else "")
                            ml = f"{abb} {base_ml}" if abb else base_ml
                        lk = f"lgs_{gid}_{mk}"
                        if st.session_state.get(lk, False):
                            row[base_ml] = "✅"
                        else:
                            odds = p_dat.get("odds","N/A")
                            os_ = str(odds) if odds and odds!="N/A" else ""
                            oi = int(os_.replace("$","")) if os_ not in ("N/A","—","") else 0
                            prob = p_dat.get("prob",0)/100.0
                            stk,_,_ = recommend_stake(prob, oi, bankroll=act_bk)
                            edge = None
                            if oi != 0 and prob:
                                ip = american_to_prob(oi)
                                if ip: edge = round(prob*100 - ip*100, 1)
                            flames = ""
                            if edge and edge > 2:
                                flames = "🔥" if edge <= 5 else "🔥🔥" if edge <= 8 else "🔥🔥🔥"
                            lbl = f"{flames} {ml}" if flames else ml
                            row[base_ml] = lbl if stk > 0 else "—"
                reg_rows.append(row)
            if reg_rows:
                st.dataframe(pd.DataFrame(reg_rows), hide_index=True, use_container_width=True)
                for _, r in upcoming.iterrows():
                    gl = f"{r['away_abbrev']} @ {r['home_abbrev']}"
                    gid = r.get("game_id","")
                    btns = []
                    for mk, base_ml in [("moneyline","ML"),("spread_minus","RL-1.5"),("spread_plus","RL+1.5"),("total","O/U")]:
                        if not st.session_state.get(f"lgs_{gid}_{mk}", False):
                            p_dat = r.get(mk)
                            if p_dat:
                                pick_name = p_dat.get("pick", "")
                                if mk == "total":
                                    ml = pick_name
                                else:
                                    abb = (r["home_abbrev"] if pick_name == r.get("home_team","") else
                                           r["away_abbrev"] if pick_name == r.get("away_team","") else "")
                                    ml = f"{abb} {base_ml}" if abb else base_ml
                                odds = p_dat.get("odds","N/A")
                                os_ = str(odds) if odds and odds!="N/A" else ""
                                oi = int(os_.replace("$","")) if os_ not in ("N/A","—","") else 0
                                prob = p_dat.get("prob",0)/100.0
                                stk,_,_ = recommend_stake(prob, oi, bankroll=act_bk)
                                if stk > 0:
                                    edge = None
                                    if oi != 0 and prob:
                                        ip = american_to_prob(oi)
                                        if ip: edge = round(prob*100 - ip*100, 1)
                                    flames = ""
                                    if edge and edge > 2:
                                        flames = "🔥" if edge <= 5 else "🔥🔥" if edge <= 8 else "🔥🔥🔥"
                                    lbl = f"{flames}{ml}" if flames else ml
                                    btns.append((mk, base_ml, lbl))
                    if not btns: continue
                    cols = st.columns([1.5]+[1]*len(btns))
                    with cols[0]: st.markdown(f"**{gl}**")
                    for ci,(mk,clean_ml,lbl) in enumerate(btns):
                        with cols[ci+1]:
                            lk = f"lgs_{gid}_{mk}"
                            if st.button(lbl, key=lk, use_container_width=True):
                                p_dat = r[mk]
                                odds = p_dat.get("odds","N/A")
                                os_ = str(odds) if odds and odds!="N/A" else ""
                                oi = int(os_.replace("$","")) if os_ not in ("N/A","—","") else 0
                                prob = p_dat.get("prob",0)/100.0
                                stk,_,sl = recommend_stake(prob, oi, bankroll=act_bk)
                                ts = datetime.now(TZ).strftime("%Y-%m-%d")
                                add_pick(ts, gl, clean_ml, prob, oi, stk, act_bk, sl,
                                         p_dat.get("pick",""), p_dat.get("detail",""))
                                notify_pick(gl, clean_ml, p_dat.get("pick",""), stk, oi, act_bk)
                                sync_picks_to_github()
                                st.session_state[lk] = True
                                st.rerun()
        else:
            st.info("No hay picks disponibles para mostrar.")

    # ── Parlays ──
    parlays = generate_parlays(picks)
    if parlays:
        st.divider()
        st.markdown("## 🎯 Parlays Recomendados")
        st.caption("Top 3 picks por valor esperado. Todos deben ganar para pagar el parlay.")
        for i, parlay in enumerate(parlays):
            render_parlay(parlay, i)

    # ── Recomendaciones ──
    recs = []
    try:
        from bankroll import recommend_stake, load_picks
        bk_data = load_picks()
        actual_bankroll = bk_data["bankroll"]
        # Helper: compute edge from entry if it has odds + prob
        def get_edge(entry):
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

        for p in picks:
            if p.get("status") == "Final":
                continue
            gl = f"{p['away_abbrev']} @ {p['home_abbrev']}"

            for mkt_key, label in [("moneyline", "ML"), ("spread_minus", "RL -1.5"),
                                   ("spread_plus", "RL +1.5"), ("total", "O/U")]:
                entry = p.get(mkt_key)
                if not entry: continue
                edge = entry.get("edge") if mkt_key == "moneyline" else get_edge(entry)
                if edge is None or edge <= 2: continue

                pick_team = entry.get("pick", "")
                prob = entry.get("prob", 0)
                odds_str = entry.get("odds", "N/A")
                odds_int = 0
                try: odds_int = int(str(odds_str).replace("$",""))
                except: pass

                stake, units, stake_label = recommend_stake(prob/100, odds_int, bankroll=actual_bankroll)

                recs.append({
                    "game": gl,
                    "market": label,
                    "pick": pick_team,
                    "prob": prob,
                    "odds": odds_str,
                    "edge": edge,
                    "stake": stake,
                    "units": units,
                    "stake_label": stake_label,
                    "pick_dict": p,
                    "mkt_key": mkt_key,
                    "entry": entry,
                })

        if recs:
            # One recommendation per game (best edge)
            best_per_game = {}
            for r in recs:
                g = r["game"]
                if g not in best_per_game or r["edge"] > best_per_game[g]["edge"]:
                    best_per_game[g] = r
            recs = sorted(best_per_game.values(), key=lambda x: x["edge"], reverse=True)
            st.divider()
            st.markdown("## 🏆 Recomendaciones del Día")
            st.caption(f"Top {min(len(recs),4)} de {len(recs)} — Kelly Criterion (25% fraccional, bankroll ${actual_bankroll:,.0f}).")
            rec_table = []
            for i, r in enumerate(recs[:4]):
                icon = "🔥" if r["edge"] > 8 else "⭐" if r["edge"] > 5 else "✅"
                rec_table.append({
                    "Juego": r["game"], "Mercado": r["market"],
                    "Pick": r["pick"], "Prob": f"{r['prob']:.0f}%",
                    "Odds": r["odds"], "Edge": f"{icon} {r['edge']:+.1f}%",
                    "Stake": f"${r['stake']:.0f}  {r['units']}u" if r["stake"] > 0 else "—",
                })
            st.dataframe(pd.DataFrame(rec_table), hide_index=True, use_container_width=True)
            avail = [(i, r) for i, r in enumerate(recs[:4]) if not st.session_state.get(f"done_rg_{i}_{r['mkt_key']}", False)]
            if avail:
                cols = st.columns(len(avail))
                for ci, (i, r) in enumerate(avail):
                    with cols[ci]:
                        lk = f"rg_{i}_{r['mkt_key']}"
                        if st.button(f"📝 {r['game'][:7]} {r['market']}", key=lk):
                            try:
                                from bankroll import add_pick, load_picks, recommend_stake
                                d = load_picks(); bk = d["bankroll"]
                                gl = f"{r['pick_dict']['away_abbrev']} @ {r['pick_dict']['home_abbrev']}"
                                os_ = r["odds"]
                                oi = int(str(os_).replace("$","")) if os_ not in ("N/A","—","") else 0
                                pv = r["prob"]/100.0
                                sk,_,sl = recommend_stake(pv, oi, bankroll=bk)
                                if sk > 0:
                                    pt = r["pick"]
                                    dtl = r.get("entry", {}).get("detail", "")
                                    ts = datetime.now(TZ).strftime("%Y-%m-%d")
                                    add_pick(ts, gl, r["market"], pv, oi, sk, bk, sl, pt, dtl)
                                    notify_pick(gl, r["market"], pt, sk, oi, bk)
                                    sync_picks_to_github()
                                    st.session_state[f"done_{lk}"] = True
                                    st.rerun()
                            except Exception as ex:
                                st.caption(f"❌ {ex}")
    except ImportError:
        pass

    # ── Mis Picks Registrados ──
    try:
        from bankroll import get_pnl, load_picks
        pnl = get_pnl()
        data = load_picks()
        st.divider()
        st.markdown("## 📋 Mis Picks Registrados")

        # Build game status lookup from current schedule
        game_status = {}
        for g in games:
            try:
                t = g["teams"]
                hid = t["home"]["team"]["id"]
                aid = t["away"]["team"]["id"]
                ha = ab_map.get(hid, "??")
                aa = ab_map.get(aid, "??")
                key = f"{aa} @ {ha}"
                sc = g.get("status",{}).get("codedGameState","")
                sd = g.get("status",{}).get("detailedState","")
                gd = g.get("gameDate","")
                gt = None
                try:
                    utc_dt = datetime.fromisoformat(gd.replace("Z","+00:00"))
                    gt = utc_dt.astimezone(TZ)
                except:
                    pass
                game_status[key] = (sc, sd, gt)
            except:
                pass

        if data["history"]:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Bankroll", f"${pnl['bankroll']:.0f}")
            mc2.metric("Profit", f"${pnl['profit']:+.0f}", delta=f"{pnl['roi']:+.0f}%")
            mc3.metric("Record", f"{pnl['wins']}-{pnl['losses']}", delta=f"{pnl['pct']}%")
            mc4.metric("Pendientes", pnl["open"])

            # Bankroll chart
            if len(data["history"]) >= 2:
                try:
                    import altair as alt
                    chart_data = []
                    br = 1000
                    for p in data["history"]:  # chronological
                        if p.get("settled") and p.get("profit") is not None:
                            br += p["profit"]
                        chart_data.append({"#": len(chart_data), "Bankroll": br, "Fecha": p.get("date","")})
                    cdf = pd.DataFrame(chart_data)
                    chart = alt.Chart(cdf).mark_line(point=True, color="#00cc66").encode(
                        x=alt.X("#:Q", title="Pick #", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("Bankroll:Q", title="Bankroll ($)", scale=alt.Scale(zero=False)),
                        tooltip=["Fecha:N", "Bankroll:Q"],
                    ).properties(height=200)
                    st.altair_chart(chart, use_container_width=True)
                except:
                    pass

            total_profit = 0
            rows = []
            for p in reversed(data["history"]):
                result = p.get("result")
                if result == "W":
                    r_icon = "✅ Ganado"
                elif result == "L":
                    r_icon = "❌ Perdido"
                else:
                    gk = p.get("game", "")
                    if gk in game_status:
                        sc, sd, gt = game_status[gk]
                        if sc == "I":
                            r_icon = "🔴 En Vivo"
                        elif sc != "F" and gt is not None:
                            now_tz = datetime.now(TZ)
                            mins_to_start = (gt - now_tz).total_seconds() / 60.0
                            if 0 <= mins_to_start <= 15:
                                r_icon = "⏳ Por Iniciar"
                            elif mins_to_start < 0:
                                r_icon = "🔴 En Vivo"
                            else:
                                r_icon = "⏳ Pendiente"
                        else:
                            r_icon = "⏳ Pendiente"
                    else:
                        r_icon = "⏳ Pendiente"

                profit = p.get("profit")
                if profit is not None:
                    total_profit += profit
                profit_str = f"${profit:+.0f}" if profit is not None else "—"
                prob_str = f"{p.get('model_prob', 0):.0%}"
                odds_str = f"${p.get('odds', 0):+d}"
                stake_str = f"${p.get('stake', 0):.0f}"

                rows.append({
                    "Fecha": p.get("date",""),
                    "Juego": p.get("game",""),
                    "Mercado": p.get("market",""),
                    "Pick": p.get("team",""),
                    "Prob": prob_str,
                    "Cuota": odds_str,
                    "Stake": stake_str,
                    "Estado": r_icon,
                    "Profit": profit_str,
                })

            # Dataframe con scroll horizontal
            if rows:
                _df = pd.DataFrame(rows)
                st.dataframe(
                    _df,
                    use_container_width=True,
                    hide_index=True,
                )

                # Delete pick — per-row buttons
                st.markdown("**Eliminar picks:**")
                del_cols = st.columns(min(len(data["history"]), 5))
                for i, p in enumerate(data["history"]):
                    col = del_cols[i % len(del_cols)]
                    pid = p.get("id", i + 1)
                    lbl = f"#{pid} {p.get('game','')[:12]}"
                    if col.button(lbl, key=f"del_{pid}", use_container_width=True):
                        from bankroll import save_picks, load_picks
                        d = load_picks()
                        d["history"] = [x for x in d["history"] if x.get("id") != pid]
                        save_picks(d)
                        st.rerun()
                if st.button("🗑️ Limpiar todo", key="clear_all", type="secondary"):
                    from bankroll import save_picks
                    save_picks({"bankroll": 1000, "history": []})
                    st.session_state.clear()
                    st.toast("✅ Historial limpiado", icon="🗑️")

            green = total_profit >= 0
            st.markdown(f"Profit total: <span style='color:{'#00cc66' if green else '#ff4444'}'><b>${total_profit:+.2f}</b></span>", unsafe_allow_html=True)
        else:
            st.info("💡 Aún no has registrado picks. Usa el botón **📝** en las tarjetas o recomendaciones para empezar.")
    except ImportError:
        pass

    st.divider()
    with st.expander("🔬 Tabla detallada"):
        cols_avail = [c for c in ["away_team","home_team","ml_home_prob","ml_away_prob","exp_total","spr_team","spr_prob"] if c in df.columns]
        if cols_avail:
            sd2 = df[cols_avail].copy()
            st.dataframe(sd2, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
