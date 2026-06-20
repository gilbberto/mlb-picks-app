import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import json
import pickle
import math
import time
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

# ─── PlayDoIt Style ───
st.markdown("""
<style>
    :root {
        --bg: #FFFFFF;
        --card: #F8F9FA;
        --accent: #E53935;
        --btn: #2E7D32;
        --btn-hover: #388E3C;
        --text: #212121;
        --sub: #666666;
        --border: #E0E0E0;
    }
    .stApp, .main, .block-container, .stApp > header {
        background-color: var(--bg) !important;
    }
    #MainMenu, header[data-testid="stHeader"], .stApp > header, .stApp header {
        background: var(--bg) !important;
    }
    section[data-testid="stSidebar"] + div, .main > div:first-child {
        background: var(--bg) !important;
    }
    .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, div {
        color: var(--text) !important;
    }
    span, .stCaption { color: var(--sub) !important; }
    a { color: var(--accent) !important; }
    /* Cards */
    div[data-testid="column"] {
        background: var(--card) !important;
        border-radius: 8px !important;
        padding: 14px !important;
        margin: 6px 0 !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #FAFAFA !important; border-right: 1px solid var(--border) !important; }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] .stCaption { color: var(--sub) !important; }
    /* Metrics */
    div[data-testid="stMetric"] {
        background: var(--card) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="stMetric"] label { color: var(--sub) !important; }
    div[data-testid="stMetric"] div { color: var(--text) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--accent) !important; }
    /* Tables */
    div[data-testid="stDataFrame"] { background: #FFF !important; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] thead th, div[data-testid="stDataFrame"] thead td {
        background: var(--accent) !important; color: #FFF !important;
        font-weight: 600 !important; text-transform: uppercase; font-size: 12px !important;
        border-bottom: 2px solid #C62828 !important;
    }
    div[data-testid="stDataFrame"] tbody td {
        color: var(--text) !important; background: #FFF !important;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background: #FAFAFA !important;
    }
    /* Buttons */
    .stButton button {
        background: var(--btn) !important; color: #FFF !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; transition: 0.2s;
    }
    .stButton button:hover { background: var(--btn-hover) !important; }
    /* Inputs */
    input, textarea, .stTextInput > div > div {
        background: #FFF !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    /* Expanders */
    div[data-testid="stExpander"] {
        background: #FFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary {
        background: var(--accent) !important; color: #FFF !important;
        border-radius: 8px 8px 0 0 !important; padding: 8px 12px !important; font-weight: 600 !important;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stExpander"] details summary p, div[data-testid="stExpander"] details summary span {
        color: #FFF !important;
    }
    div[data-testid="stExpander"] details summary svg { fill: #FFF !important; }
    /* Dividers */
    hr { border-color: var(--border) !important; }
    /* Spinner */
    div[data-testid="stSpinner"] { color: var(--accent) !important; }
    div.stSpinner > div { background: rgba(255,255,255,0.9) !important; }
    /* Form */
    div[data-testid="stForm"] { border: 1px solid var(--border) !important; border-radius: 8px; padding: 16px; }
    @media (max-width: 768px) {
        .block-container { padding: 4px 2px !important; }
        div[data-testid="column"] { padding: 8px !important; margin: 3px 0 !important; }
        h1 { font-size: 20px !important; } h2 { font-size: 16px !important; } h3 { font-size: 14px !important; }
        .stButton button { min-height: 38px !important; font-size: 13px !important; }
    }
</style>
""", unsafe_allow_html=True)

ODDS_API_KEY = "b09f7e5fb08081c87e7e34272fda4ea0"
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY", "")
PREFERRED_BOOK = "BetMGM"

# ─── ML Models (Ensemble XGBoost, fallback single XGBoost, fallback RF) ───
_MODELS_LOADED = False
_xgb_models_hw = []
_xgb_models_rd = []
_xgb_models_tot = []
_rf_hw = _rf_rd = _rf_tot = None
_cols = None
_model_type = ""

def _load_xgb_seeds(base, prefix):
    models = []
    for seed in [42, 123, 456]:
        try:
            with open(base + f"{prefix}_s{seed}.pkl", "rb") as f:
                models.append(pickle.load(f))
        except:
            pass
    return models

def load_models():
    global _xgb_models_hw, _xgb_models_rd, _xgb_models_tot, _rf_hw, _rf_rd, _rf_tot, _cols, _MODELS_LOADED, _model_type
    base = os.path.join(os.path.dirname(__file__), "")
    try:
        import xgboost as xgb
        _xgb_models_hw = _load_xgb_seeds(base, "xgb_hw")
        _xgb_models_rd = _load_xgb_seeds(base, "xgb_rd")
        _xgb_models_tot = _load_xgb_seeds(base, "xgb_tot")
        if _xgb_models_hw:
            with open(base + "xgb_cols.pkl", "rb") as f: _cols = pickle.load(f)
            _model_type = f"XGBoost Ensemble ({len(_xgb_models_hw)} seeds)"
            _MODELS_LOADED = True
        else:
            print("No ensemble models, trying single XGBoost...")
            with open(base + "xgb_hw.pkl", "rb") as f: _xgb_models_hw = [pickle.load(f)]
            with open(base + "xgb_rd.pkl", "rb") as f: _xgb_models_rd = [pickle.load(f)]
            with open(base + "xgb_tot.pkl", "rb") as f: _xgb_models_tot = [pickle.load(f)]
            with open(base + "xgb_cols.pkl", "rb") as f: _cols = pickle.load(f)
            _model_type = "XGBoost"
            _MODELS_LOADED = True
    except Exception as e:
        print(f"XGBoost not loaded: {e}, trying RF...")
        try:
            with open(base + "rf_hw.pkl", "rb") as f: _rf_hw = pickle.load(f)
            with open(base + "rf_rd.pkl", "rb") as f: _rf_rd = pickle.load(f)
            with open(base + "rf_tot.pkl", "rb") as f: _rf_tot = pickle.load(f)
            with open(base + "rf_cols.pkl", "rb") as f: _cols = pickle.load(f)
            _model_type = "RandomForest"
            _MODELS_LOADED = True
        except Exception as e2:
            print(f"RF also not loaded: {e2}")

load_models()

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
CURRENT_SEASON = 2026
LG_AVG_RUNS = 4.5

C = {
    "value_high": "#E53935", "value_med": "#88cc00", "value_low": "#cccc00",
    "no_value": "#FF6B6B", "card_bg": "#1a1d2e", "accent": "#4da6ff",
}

# ─── MLB Stats API ───

@st.cache_data(ttl=3600)
def fetch_todays_schedule(date_str=""):
    today = date_str or datetime.now(TZ).strftime("%m/%d/%Y")
    url = f"{MLB_API_BASE}/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    games = []
    for de in resp.json().get("dates", []):
        for g in de.get("games", []):
            games.append(g)
    return games


@st.cache_data(ttl=86400)
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
                ip = s.get("inningsPitched", "0")
                ipv = 0
                if isinstance(ip, str) and "." in ip:
                    parts = ip.split(".")
                    ipv = int(parts[0]) + int(parts[1]) / 3.0 if len(parts) > 1 else float(parts[0])
                else:
                    ipv = float(ip or 0)
                result["pitching"] = {k: safe_float(s.get(k)) for k in ["era","whip","runs","strikeouts","baseOnBalls","homeRuns",
                                              "hits","earnedRuns"]}
                result["pitching"]["ip"] = ipv
                result["pitching"]["er"] = safe_float(s.get("earnedRuns"))
                result["pitching"]["bb"] = safe_float(s.get("baseOnBalls"))
                result["pitching"]["so"] = safe_float(s.get("strikeOuts"))
                result["pitching"]["h"] = safe_float(s.get("hits"))
    return result


@st.cache_data(ttl=86400)
def fetch_team_abbrevs():
    teams = requests.get(f"{MLB_API_BASE}/teams?sportIds=1", timeout=10).json()
    return {t["id"]: t.get("abbreviation","??") for t in teams.get("teams",[])}

@st.cache_data(ttl=86400)
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
            hr = safe_float(s.get("homeRuns"))
            bb = safe_float(s.get("baseOnBalls"))
            so = safe_float(s.get("strikeOuts"))
            h = safe_float(s.get("hits"))
            ab = safe_float(s.get("atBats"))
            sf = safe_float(s.get("sacFlies"))
            hbp = safe_float(s.get("hitByPitch"))
            go = safe_float(s.get("groundOuts"))
            ao = safe_float(s.get("airOuts"))
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
                    "hr9": safe_float(s.get("homeRunsPer9")), "fip": fip, "babip": babip,
                    "kbb": kbb, "gb_rate": gb_rate,
                    "er": safe_float(s.get("earnedRuns")), "bb": bb, "so": so, "h": h,
                    "name": splits[0].get("player",{}).get("fullName","")}
    except Exception:
        return {}
    return {}


@st.cache_data(ttl=86400)
def fetch_pitcher_recent_form(pid, n_starts=5):
    if not pid:
        return {}
    try:
        url = f"{MLB_API_BASE}/people/{pid}/stats?stats=gameLog&season={CURRENT_SEASON}&group=pitching"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return {}
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        starts = [s for s in splits if s.get("stat", {}).get("inningsPitched", "0") != "0" and
                  s.get("game", {}).get("gameType") == "R"]
        recent = starts[-n_starts:]
        if not recent:
            return {}
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
    except Exception:
        return {}

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=86400)
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

ODDS_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".odds_cache.json")

def _sync_odds_from_github():
    """Download .odds_cache.json from GitHub ONLY if it's from today."""
    try:
        owner, repo_name, branch, headers = _gh_headers()
        if not owner: return
        today_str = datetime.now(TZ).strftime("%Y-%m-%d")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/.odds_cache.json?ref={branch}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            import base64
            content = base64.b64decode(r.json()["content"]).decode()
            cached = json.loads(content)
            gh_date = cached.get("date", "") if isinstance(cached, dict) else ""
            if gh_date == today_str:
                with open(ODDS_CACHE_PATH, "w") as f:
                    f.write(content)
    except:
        pass

ODDS_COOLDOWN = os.path.join(os.path.dirname(__file__), ".odds_cooldown")

def _check_odds_cache():
    """Return cached odds if cache date is today. Uses date field inside JSON (not file mtime)."""
    try:
        with open(ODDS_CACHE_PATH) as f:
            cached = json.load(f)
        cache_date = cached.get("date", "") if isinstance(cached, dict) else ""
        today_str = datetime.now(TZ).strftime("%Y-%m-%d")
        if cache_date == today_str:
            data = cached.get("data", cached) if isinstance(cached, dict) else cached
            if data:
                return data
    except:
        pass
    return None

def _save_odds_cache(odds):
    try:
        cache = {"date": datetime.now(TZ).strftime("%Y-%m-%d"), "data": odds}
        with open(ODDS_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except:
        pass

@st.cache_data(ttl=600)
def fetch_odds():
    cached = _check_odds_cache()
    if cached is not None:
        return cached
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
    if odds:
        _save_odds_cache(odds)
    else:
        # API failed — load whatever is in cache (even if stale)
        try:
            with open(ODDS_CACHE_PATH) as f:
                cached = json.load(f)
            odds = cached.get("data", cached) if isinstance(cached, dict) else cached
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
    if not game_odds:
        return None, None
    best_price, best_book, best_point = None, None, None
    for book in game_odds.get("bookmakers", []):
        if book.get("title", "") != PREFERRED_BOOK:
            continue
        for mkt in book.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for oc in mkt.get("outcomes", []):
                if outcome_name and oc.get("name") != outcome_name:
                    continue
                point = oc.get("point")
                if expect_point is not None and point != expect_point:
                    continue
                price = oc.get("price")
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

def build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f,
                           hp_rec=None, ap_rec=None, weather=None):
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


@st.cache_data(ttl=86400)
def monte_carlo_predict(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f,
                         hp_rec=None, ap_rec=None, n_sims=5000, weather=None, total_std=3.2):
    """Run Monte Carlo simulation using trained models. Returns dict."""
    if not _MODELS_LOADED:
        return {"ml_hp": None, "ml_ap": None,
                "spr_home_minus": None, "spr_home_plus": None,
                "spr_away_minus": None, "spr_away_plus": None, "spr_exp_margin": None,
                "exp_total": None, "total_std": total_std}

    row = build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f,
                                hp_rec=hp_rec, ap_rec=ap_rec, weather=weather)
    x = np.array([[row[c] for c in _cols]])

    if _xgb_models_hw:
        hw_probs = [m.predict_proba(x)[0, 1] for m in _xgb_models_hw]
        hw_prob = sum(hw_probs) / len(hw_probs)
        exp_rdiffs = [m.predict(x)[0] for m in _xgb_models_rd]
        exp_rdiff = sum(exp_rdiffs) / len(exp_rdiffs)
        exp_totals = [m.predict(x)[0] for m in _xgb_models_tot]
        exp_total = sum(exp_totals) / len(exp_totals)
    else:
        hw_prob = _rf_hw.predict_proba(x)[0, 1]
        exp_rdiff = _rf_rd.predict(x)[0]
        exp_total = _rf_tot.predict(x)[0]

    rdiff_sims = np.random.normal(exp_rdiff, 3.0, n_sims)
    total_sims = np.random.normal(exp_total, 3.2, n_sims)

    mc_hw = np.mean(rdiff_sims > 0)
    mc_home_minus = np.mean(rdiff_sims >= 1.5)   # home cubre -1.5
    mc_home_plus  = np.mean(rdiff_sims >= -1.5)  # home cubre +1.5
    mc_over = np.mean(total_sims > 8.5)

    # Probs complementarias para el visitante
    mc_away_minus = 1.0 - mc_home_plus   # visita cubre -1.5
    mc_away_plus  = 1.0 - mc_home_minus  # visita cubre +1.5

    return {
        "ml_hp": round(float(mc_hw), 4),
        "ml_ap": round(float(1 - mc_hw), 4),
        "spr_home_minus": round(float(mc_home_minus), 4),
        "spr_home_plus": round(float(mc_home_plus), 4),
        "spr_away_minus": round(float(mc_away_minus), 4),
        "spr_away_plus": round(float(mc_away_plus), 4),
        "spr_exp_margin": round(float(exp_rdiff), 2),
        "exp_total": round(float(exp_total), 2),
        "total_std": total_std,
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

def _secret(key, default=""):
    try:
        try:
            return st.secrets.get(key, os.environ.get(key, default))
        except Exception:
            pass
        return os.environ.get(key, default)
    except Exception:
        return default


def send_telegram(msg):
    try:
        tok = _secret("TELEGRAM_TOKEN")
        cid = _secret("TELEGRAM_CHAT_ID")
        if not tok or not cid: return None
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                       json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        if r.ok:
            return r.json().get("result", {}).get("message_id")
    except: pass
    return None

def fmt_ou(pick_name, detail):
    if not detail:
        return pick_name
    line = detail[1:] if len(detail) > 1 and detail[0] in "ou" else detail
    return f"{pick_name} {line}"


def notify_pick(gl, market, team, stake, odds, bankroll, pick_id=None):
    try:
        s = f"📝 *Pick registrado*\n{gl} → {market} {team}\nApuesta: ${stake:.2f} @ {odds:+d}\nBankroll: ${bankroll:.2f}"
        mid = send_telegram(s)
        if mid and pick_id:
            from bankroll import load_picks, save_picks
            d = load_picks()
            for p in d["history"]:
                if p["id"] == pick_id:
                    p["telegram_msg_id"] = mid
                    break
            save_picks(d)
    except: pass


# ─── GitHub sync ───

def _gh_headers():
    tok = _secret("GITHUB_TOKEN")
    repo = _secret("REPO")
    if not tok or not repo: return None, None, None, None
    owner, repo_name = repo.split("/")
    branch = _secret("BRANCH", "main")
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    return owner, repo_name, branch, headers

def sync_picks_from_github():
    """Pull latest picks.json from GitHub (important for Railway where web + worker are separate)."""
    try:
        owner, repo_name, branch, headers = _gh_headers()
        if not owner: return
        import base64
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/picks.json?ref={branch}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"]).decode()
            with open("picks.json", "w") as f:
                f.write(content)
    except: pass

def _merge_picks(local_str, remote_str):
    import json
    remote = json.loads(remote_str)
    local = json.loads(local_str)
    local_ids = {p.get("id") for p in local.get("history", [])}
    remote["history"] = [p for p in remote["history"] if p.get("id") in local_ids]
    local_by_id = {p.get("id"): p for p in local.get("history", [])}
    for p in remote["history"]:
        lp = local_by_id.get(p.get("id"))
        if lp and lp.get("settled") and not p.get("settled"):
            p.update({k: lp[k] for k in ("result", "profit", "settled") if k in lp})
    remote_ids = {p.get("id") for p in remote.get("history", [])}
    for p in local.get("history", []):
        if p.get("id") not in remote_ids:
            remote["history"].append(p)
    remote["history"].sort(key=lambda x: x.get("id", 0))
    stakes = sum(p.get("stake", 0) for p in remote["history"])
    profits = sum(p.get("profit") or 0 for p in remote["history"] if p.get("profit") is not None)
    remote["bankroll"] = round(1000 - stakes + profits, 2)
    for k in ("weekly_bankroll", "weekly_start", "weekly_history", "last_weekly_reset", "cash_adjust"):
        if k in local:
            remote[k] = local[k]
    return json.dumps(remote, indent=2)

def sync_picks_to_github():
    try:
        owner, repo_name, branch, headers = _gh_headers()
        if not owner: return
        import base64
        with open("picks.json", "r") as f:
            local = f.read()
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/picks.json"
        for attempt in range(3):
            r = requests.get(url + f"?ref={branch}", headers=headers, timeout=10)
            if r.status_code == 200:
                remote = r.json()
                local = _merge_picks(local, base64.b64decode(remote["content"]).decode())
                sha = remote["sha"]
            else:
                sha = ""
            data = {"message": "sync picks.json from app", "content": base64.b64encode(local.encode()).decode(), "branch": branch}
            if sha: data["sha"] = sha
            r2 = requests.put(url, json=data, headers=headers, timeout=10)
            if r2.status_code == 409:
                time.sleep(0.5); continue
            st.session_state["sync_status"] = f"Sync intento {attempt+1}: {r2.status_code}"
            break
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
            status_col = "#FF6B6B"
        elif cgs == "F":
            status_label = "✅ FINAL "
            status_col = "#888"
        elif pick.get("final", ""):
            status_label = "✅ FINAL "
            status_col = "#888"
        elif game_dt is not None:
            mins_to_start = (game_dt - now_tz).total_seconds() / 60.0
            if 0 <= mins_to_start <= 15:
                status_label = "⏳ POR INICIAR "
                status_col = "#ffaa00"
        score_str = f"**{pick['final']}** " if pick.get("final") else ""
        time_str = f"🕐 {pick['game_time']}  " if pick.get("game_time") else ""
        st.markdown(f"### {time_str}{status_label}{score_str}**{an}** @ **{hn}**" + "".join(f" `{s}`" for s in srcs) + badge + pitcher_line)
        mkt_list = [("moneyline", "ML", "💰")]
        mkt_list += [("spread_plus", "RL +1.5", "📏")]
        mkt_list += [("total", "O/U", "📈")]
        # Solo mostrar el mercado con mayor confianza
        _best_mkt = None
        _best_score = -1
        for mkt_key, mkt_label, mkt_icon in mkt_list:
            p = pick.get(mkt_key, {})
            if not p: continue
            edge_val = p.get("edge")
            prob_val = p.get("prob")
            if edge_val is not None and edge_val > 2:
                score = 5 if edge_val > 8 else (4 if edge_val > 5 else 3)
            elif edge_val is None and prob_val is not None and prob_val >= 65:
                score = 5 if prob_val >= 75 else 4
            else:
                score = 0
            if score > _best_score:
                _best_score = score
                _best_mkt = (mkt_key, mkt_label, mkt_icon)
        mkt_list = [_best_mkt] if _best_mkt else []
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
                detail_display = f" {detail}" if detail else ""
                st.markdown(f"**{mkt_icon} {mkt_label}{rec_tag}**  \n`{pick_name}`{detail_display}")
            with col_b:
                line = f"**{prob_str}**" if prob_str else ""
                if ev is not None:
                    line += f"  \n{ev_str}"
                st.markdown(line if line else "")
            with col_c:
                if odds and odds != "N/A":
                    st.markdown(f"`{odds}`")
                if edge is not None and edge > 2:
                    if edge > 8: flame = "🔥🔥🔥"
                    elif edge > 5: flame = "🔥🔥"
                    else: flame = "🔥"
                elif edge is None and prob_val is not None and prob_val >= 65:
                    if prob_val >= 85: flame = "🔥🔥🔥"
                    elif prob_val >= 75: flame = "🔥🔥🔥"
                    elif prob_val >= 65: flame = "🔥🔥"
                else:
                    flame = ""
                if flame:
                    st.markdown(f"<span style='font-size:18px'>{flame}</span>", unsafe_allow_html=True)
                log_key = f"lg_{pick.get('game_id','')}_{mkt_key}"
                if role == "admin" and edge is not None and edge > 2:
                    if st.button("📝", key=log_key):
                        try:
                            from bankroll import add_pick, get_pnl, recommend_stake
                            bk = get_pnl()["weekly_bankroll"]
                            gl = f"{pick['away_abbrev']} @ {pick['home_abbrev']}"
                            os_ = str(odds) if odds and odds != "N/A" else ""
                            oi = int(str(os_).replace("$","")) if os_ not in ("N/A","—","") else 0
                            pv = prob_val/100.0 if prob_val is not None else 0.5
                            sk,_,sl = recommend_stake(pv, oi, bankroll=bk)
                            if sk > 0:
                                ts = datetime.now(TZ).strftime("%Y-%m-%d")
                                pid = add_pick(ts, gl, mkt_label, pv, oi, sk, bk, sl, pick_name, detail)
                                notify_pick(gl, mkt_label, pick_name, sk, oi, bk, pick_id=pid)
                                sync_picks_to_github()
                                st.session_state[log_key] = True
                            else:
                                st.caption("⚠️ Kelly=0")
                        except Exception as ex:
                            st.caption(f"❌ {ex}")
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
        for mkt_key, mkt_label in [("moneyline","ML"), ("total","O/U")]:
            m = p.get(mkt_key, {})
            ev = m.get("ev")
            odds = m.get("odds")
            prob = m.get("prob")

            if prob is None or prob <= 0:
                continue

            prob_dec = prob / 100.0

            has_odds = odds is not None and odds != "N/A"
            has_ev = ev is not None and ev > 0
            threshold = 50
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
        detail_str = f" {leg['detail']}" if leg.get("detail") else ""
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


# ─── Auth ───

USERS_PATH = os.path.join(os.path.dirname(__file__), "users.json")
MODULES = [("daily_picks", "Picks del Día"), ("recommendations", "Recomendaciones"),
           ("calibration", "Calibración"), ("model_stats", "Estadísticas"),
           ("detailed_table", "Tabla Detallada")]

def _default_perms():
    return {k: v for k, v in [("daily_picks", True), ("recommendations", True),
                                ("calibration", False), ("model_stats", False),
                                ("detailed_table", False)]}

def _expires_at(days):
    """Return ISO date string N days from now for expiration."""
    return (datetime.now(TZ) + timedelta(days=days)).strftime("%Y-%m-%d")

def _is_expired(user_data):
    """Check if a user's access has expired."""
    exp = user_data.get("expires_at")
    if not exp or user_data.get("role") == "admin":
        return False
    return datetime.now(TZ).strftime("%Y-%m-%d") > exp

def _days_left(user_data):
    """Return days remaining for a user."""
    exp = user_data.get("expires_at")
    if not exp or user_data.get("role") == "admin":
        return None
    try:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=TZ)
        return (exp_dt - datetime.now(TZ)).days
    except:
        return None

def _load_users():
    try:
        with open(USERS_PATH) as f:
            return json.load(f)
    except:
        return {"admin": {"password": _secret("ADMIN_PASSWORD", "admin2024"), "role": "admin"}}

def _sync_users_from_github():
    try:
        owner, repo_name, branch, headers = _gh_headers()
        if not owner:
            return
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/users.json?ref={branch}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            import base64
            content = base64.b64decode(r.json()["content"]).decode()
            with open(USERS_PATH, "w") as f:
                f.write(content)
    except:
        pass

def _save_users(u):
    for name, data in u.items():
        if data.get("role") != "admin" and "expires_at" not in data:
            data["expires_at"] = _expires_at(30)
    with open(USERS_PATH, "w") as f:
        json.dump(u, f, indent=2)
    _sync_users_to_github()

def _sync_users_to_github():
    try:
        owner, repo_name, branch, headers = _gh_headers()
        if not owner:
            return
        import base64
        with open(USERS_PATH, "r") as f:
            content = f.read()
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/users.json"
        r = requests.get(url + f"?ref={branch}", headers=headers, timeout=10)
        sha = r.json().get("sha", "") if r.ok else ""
        data = {"message": "sync users.json from app", "content": base64.b64encode(content.encode()).decode(), "branch": branch}
        if sha:
            data["sha"] = sha
        requests.put(url, json=data, headers=headers, timeout=10)
    except:
        pass

def _get_perms(username):
    users = _load_users()
    u = users.get(username, {})
    if u.get("role") == "admin":
        return {k: True for k, _ in MODULES}
    return {**_default_perms(), **u.get("permissions", {})}

def _login_form():
    _, c, _ = st.columns([3, 1, 3])
    with c:
        st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", use_container_width=True)
        st.markdown("<h2 style='text-align:center;margin:5px 0 0 0'>MLB Picks AI</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;font-size:13px;margin-bottom:20px'>We don't promise to win every day.<br>We promise to be on the right side of the numbers. ⚾📈💰</p>", unsafe_allow_html=True)
        # Rate limit check
        attempts = st.session_state.get("login_attempts", [])
        attempts = [t for t in attempts if time.time() - t < 30]
        st.session_state.login_attempts = attempts
        if len(attempts) >= 5:
            wait = int(30 - (time.time() - attempts[0]))
            st.error(f"⏳ Demasiados intentos. Espera {wait}s.")
        with st.form("login_form"):
            u = st.text_input("Usuario")
            p = st.text_input("Contraseña", type="password")
            if st.form_submit_button("🔑 Entrar", use_container_width=True):
                if len(st.session_state.get("login_attempts", [])) >= 5:
                    st.rerun()
                user = u.strip().lower()
                pwd = p
                users = _load_users()
                if user in users and users[user]["password"] == pwd:
                    if _is_expired(users[user]):
                        st.error("❌ Acceso expirado.")
                    else:
                        st.session_state.user = user
                        st.session_state.role = users[user]["role"]
                        st.session_state.login_time = time.time()
                        st.query_params["u"] = user
                        st.rerun()
                else:
                    st.session_state.login_attempts = st.session_state.get("login_attempts", []) + [time.time()]
                    st.error("Usuario o contraseña incorrectos")

def _admin_panel():
    users = _load_users()
    st.sidebar.divider()
    st.sidebar.markdown(f"### ⚙️ Admin")
    st.sidebar.markdown(f"👤 **{st.session_state.user}** (Admin)")

    with st.sidebar.expander("👥 Usuarios", expanded=False):
        for uname, udata in users.items():
            if uname == "admin":
                continue
            cols = st.columns([4, 1])
            exp_days = _days_left(udata)
            exp_str = f" — ⏳{exp_days}d" if exp_days and exp_days > 0 else (" — ❌Expirado" if exp_days is not None and exp_days <= 0 else " — ♾️")
            cols[0].markdown(f"**{uname}**{exp_str}")
            if cols[1].button("🗑", key=f"del_{uname}"):
                del users[uname]
                _save_users(users)
                st.rerun()

        st.divider()
        st.markdown("**Nuevo usuario**")
        nu = st.text_input("Usuario", key="nu_name").strip().lower()
        np = st.text_input("Contraseña", type="password", key="nu_pass")
        nr = st.selectbox("Rol", ["viewer"], key="nu_role")
        nd = st.selectbox("Acceso por", [7, 15, 30, 60, 90], index=2, key="nu_dur", format_func=lambda x: f"{x} días")
        if st.button("➕ Crear") and nu and np:
            if nu in users:
                st.error("Ya existe")
            else:
                user_data = {"password": np, "role": nr, "permissions": _default_perms()}
                user_data["expires_at"] = _expires_at(nd)
                users[nu] = user_data
                _save_users(users)
                st.rerun()

    with st.sidebar.expander("🔧 Permisos", expanded=False):
        sel = st.selectbox("Usuario", [u for u in users if u != "admin"], key="perm_user")
        if sel:
            perms = users[sel].setdefault("permissions", _default_perms())
            changed = False
            for key, label in MODULES:
                val = st.checkbox(label, value=perms.get(key, False), key=f"p_{sel}_{key}")
                if val != perms.get(key):
                    perms[key] = val
                    changed = True
            if changed:
                _save_users(users)
            st.divider()
            st.markdown("**⏱ Expiración**")
            cur_exp = users[sel].get("expires_at", "")
            st.caption(f"Actual: {cur_exp}" if cur_exp else "Sin expiración")
            new_dur = st.selectbox("Extender por", [7, 15, 30, 60, 90], index=2, key=f"ext_{sel}",
                                    format_func=lambda x: f"{x} días")
            if st.button("🔄 Actualizar", key=f"upd_{sel}"):
                users[sel]["expires_at"] = _expires_at(new_dur)
                _save_users(users)
                st.rerun()

    if st.sidebar.button("🔒 Cerrar sesión"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()


# ─── Main ───

def main():
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
    _sync_users_from_github()
    _sync_odds_from_github()
    
    # ─── Health Check: verify odds are from today (no API call) ───
    if _get_perms(st.session_state.user).get("daily_picks", True):
        try:
            with open(ODDS_CACHE_PATH) as f:
                oc = json.load(f)
            cache_date = oc.get("date", "") if isinstance(oc, dict) else ""
            today_str = datetime.now(TZ).strftime("%Y-%m-%d")
            if cache_date != today_str:
                st.warning(f"⏳ Odds del {cache_date}. Se actualizarán pronto (1 llamada/día).")
        except:
            pass
    # ─── End Health Check ───
    if st.session_state.get("login_time") and time.time() - st.session_state.login_time > 28800:
        st.session_state.clear()
        st.rerun()
    if not st.session_state.role:
        _u = st.query_params.get("u")
        if _u:
            users = _load_users()
            if _u in users and not _is_expired(users[_u]):
                st.session_state.user = _u
                st.session_state.role = users[_u]["role"]
                st.session_state.login_time = time.time()
    if not st.session_state.role:
        _login_form()
        return
    role = st.session_state.role

    # ── Procesar delete de pick antes de cualquier render ──
    _del_pid = st.query_params.get("del_pick")
    if _del_pid is not None:
        _proc_key = f"processed_del_{_del_pid}"
        if not st.session_state.get(_proc_key, False):
            st.session_state[_proc_key] = True
            try:
                _pid = int(_del_pid)
                from bankroll import save_picks, load_picks
                _d = load_picks()
                _pick = next((p for p in _d["history"] if p.get("id") == _pid), None)
                if _pick and _pick.get("telegram_msg_id"):
                    _tok = _secret("TELEGRAM_TOKEN")
                    _cid = _secret("TELEGRAM_CHAT_ID")
                    if _tok and _cid:
                        requests.post(f"https://api.telegram.org/bot{_tok}/deleteMessage",
                                      json={"chat_id": _cid, "message_id": _pick["telegram_msg_id"]}, timeout=5)
                _d["history"] = [x for x in _d["history"] if x.get("id") != _pid]
                save_picks(_d)
                sync_picks_to_github()
            except:
                pass

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
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", width=45)
        with c2:
            st.markdown("### MLB Picks AI")
        st.caption(f"👤 **{st.session_state.user}** ({role})")

        try:
            from bankroll import get_pnl
            pnl = get_pnl()
            st.divider()
            st.markdown("#### 📊 Mi Rendimiento")
            c1, c2 = st.columns(2)
            c1.metric("Semanal", f"${pnl['weekly_bankroll']:.0f}", delta=f"${pnl['weekly_profit']:+.0f}")
            c2.metric("Histórico", f"${pnl['profit']:+.0f}", delta=f"{pnl['roi']:+.0f}%")
            st.caption(f"Semana: {pnl['weekly_wins']}-{pnl['weekly_losses']} | Total: {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%) | {pnl['open']} pendientes")
            try:
                from bankroll import load_picks
                _wh = load_picks().get("weekly_history", [])
                if _wh:
                    with st.expander("📆 Semanas anteriores"):
                        _wh_rows = [{"Semana": w["week_start"], "Profit": f"${w['profit']:+.0f}", "Record": f"{w['wins']}-{w['losses']}", "Picks": w["picks"]} for w in reversed(_wh[-10:])]
                        st.dataframe(pd.DataFrame(_wh_rows), hide_index=True, use_container_width=True)
            except: pass
        except ImportError: pass

        if role == "admin":
            st.divider()
            with st.expander("⚙️ Herramientas", expanded=False):
                min_conf = st.selectbox("Filtrar por confianza", ["Todas","🔥 HIGH VALUE","✅ VALUE","⚠️ LOW VALUE"], index=0)
                require_odds = st.checkbox("Requiere odds del mercado", value=True)
                use_adv = st.checkbox("Usar sabermetrics (pybaseball)", value=True)
                st.divider()
                st.markdown(f"**Modelo:** {_model_type}" if _model_type else "**Modelo:** No cargado")
                st.caption("MLB Stats ✅ | ESPN ✅ | PyBaseball " + ("✅" if _check_pybaseball() else "⬜"))
                st.caption(f"Odds API: {'✅' if ODDS_API_KEY else '❌'} | SharpAPI: {'✅' if SHARPAPI_KEY else '❌'}")
            _admin_panel()
        else:
            min_conf = "Todas"
            require_odds = True
            use_adv = False
            if role == "viewer":
                if st.sidebar.button("🔒 Cerrar sesión"):
                    st.session_state.clear()
                    st.query_params.clear()
                    st.rerun()

        st.divider()
        st.caption(f"🕐 {datetime.now(TZ).strftime('%H:%M')} Chihuahua")
        st.caption("MLB Picks AI v3.4 · Solo informativo · No garantiza ganancias")

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
        games = fetch_todays_schedule(datetime.now(TZ).strftime("%m/%d/%Y"))
    if games is None or games == []:
        st.warning("No hay juegos hoy.")
        return

    c1, c2, c3 = st.columns(3)
    live_count = sum(1 for g in games if g.get("status",{}).get("codedGameState") == "I")
    soon_count = 0
    alive_count = 0
    for g in games:
        cgs = g.get("status",{}).get("codedGameState", "")
        if cgs == "F":
            continue
        if g.get("teams",{}).get("away",{}).get("score") is not None and g.get("teams",{}).get("home",{}).get("score") is not None:
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
            # Recent form (last 5 starts) — fetched but not yet in features (needs model retrain)
            hprec = fetch_pitcher_recent_form(hp_info.get("id")) if hp_info.get("id") else {}
            aprec = fetch_pitcher_recent_form(ap_info.get("id")) if ap_info.get("id") else {}

            # ── Elo ratings ──
            elo_hp, h_elo, a_elo = compute_elo(hr, ar, hid, aid)

            # ── Park factor ──
            venue_name = g.get("venue",{}).get("name", "")
            park_f = PARK_FACTORS.get(venue_name, 1.0)

            # ── Weather ──
            weather = fetch_weather(venue_name)

            # ── RandomForest + Monte Carlo ──
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

            # Asignación inicial basada en el modelo (se sobrescribe con el mercado si hay odds)
            spr_fav_team = hn if spr_exp_margin >= 0 else an
            spr_dog_team = an if spr_exp_margin >= 0 else hn
            spr_fav_prob = spr_home_minus if spr_exp_margin >= 0 else spr_away_minus
            spr_dog_prob = spr_away_plus if spr_exp_margin >= 0 else spr_home_plus

            if ml_hp is None:
                # Fallback: use old model
                ml_hp, ml_ap = predict_moneyline(hs, aws, hf, af, hn, an,
                                                 h_adv_fg if adv_used else None,
                                                 a_adv_fg if adv_used else None,
                                                 hpitch if hpitch.get("ip",0) >= 10 else None,
                                                 apitch if apitch.get("ip",0) >= 10 else None,
                                                 elo_hp, park_f)
                spr_fav_prob_old, spr_dog_prob_old, spr_exp_margin_old = predict_spread(hf, af, park_f)
                spr_fav_prob, spr_dog_prob, spr_exp_margin = spr_fav_prob_old, spr_dog_prob_old, spr_exp_margin_old
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
                "spr_home_minus": spr_home_minus,
                "spr_home_plus": spr_home_plus,
                "spr_away_minus": spr_away_minus,
                "spr_away_plus": spr_away_plus,
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
                    # Sobrescribir asignación RL con la del mercado (quien está en -1.5 / +1.5)
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
                        # Recalcular probs según qué equipo es favorito del mercado
                        hm = pick_entry.get("spr_home_minus")
                        if hm is not None:
                            # XGBoost path: usar raw MC probs
                            if m_fav == hn:
                                spr_fav_prob = hm
                                spr_dog_prob = pick_entry["spr_away_plus"]
                            else:
                                spr_fav_prob = pick_entry["spr_away_minus"]
                                spr_dog_prob = pick_entry["spr_home_plus"]
                        else:
                            # Fallback path: recalcular con margen firmado
                            h_exp = (hf.get("rs_exp", hf["rs"]) * af.get("ra_exp", af["ra"])) / LG_AVG_RUNS * park_f
                            a_exp = (af.get("rs_exp", af["rs"]) * hf.get("ra_exp", hf["ra"])) / LG_AVG_RUNS * park_f
                            exp_margin = h_exp - a_exp
                            std = 3.0
                            if m_fav == hn:
                                spr_fav_prob = 1 - norm_cdf(1.5, exp_margin, std)
                                spr_dog_prob = norm_cdf(1.5, exp_margin, std)
                            else:
                                spr_fav_prob = norm_cdf(-1.5, exp_margin, std)
                                spr_dog_prob = 1 - norm_cdf(-1.5, exp_margin, std)
                        try:
                            from bankroll import calibrate_rl
                            spr_fav_prob = calibrate_rl(spr_fav_prob)
                            spr_dog_prob = 1.0 - spr_fav_prob
                        except ImportError:
                            pass
                        pick_entry["spr_fav_team"] = spr_fav_team
                        pick_entry["spr_dog_team"] = spr_dog_team
                        pick_entry["spr_fav_prob"] = spr_fav_prob
                        pick_entry["spr_dog_prob"] = spr_dog_prob

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
                    spr_price, spr_book, _ = extract_market_odds(og, "spreads", spr_fav_team, expect_point=-1.5)
                    if spr_price and spr_fav_prob > 0:
                        ev_spr = compute_ev(spr_fav_prob*100, spr_price)
                        spr_p = american_to_prob(spr_price)
                        spr_edge = round(spr_fav_prob*100 - (spr_p*100 if spr_p else 0), 1) if spr_p else None
                        pick_entry["spread_minus"] = {
                            "pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5",
                            "ev": ev_spr, "odds": spr_price, "book": spr_book, "edge": spr_edge,
                        }
                        l_spr, _ = ev_label(ev_spr)
                        if l_spr in ("🔥 HIGH VALUE", "✅ VALUE"): border = spr_book and "#E53935" or border
                    else:
                        pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}

                    # Spread odds — underdog +1.5
                    spr_dog_price, spr_dog_book, _ = extract_market_odds(og, "spreads", spr_dog_team, expect_point=1.5)
                    if spr_dog_price and spr_dog_prob > 0:
                        ev_spr_dog = compute_ev(spr_dog_prob*100, spr_dog_price)
                        spr_dog_p = american_to_prob(spr_dog_price)
                        spr_dog_edge = round(spr_dog_prob*100 - (spr_dog_p*100 if spr_dog_p else 0), 1) if spr_dog_p else None
                        pick_entry["spread_plus"] = {
                            "pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5",
                            "ev": ev_spr_dog, "odds": spr_dog_price, "book": spr_dog_book, "edge": spr_dog_edge,
                        }
                    else:
                        pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}

                    # Totals odds
                    ov_price, ov_book, ov_point = extract_market_odds(og, "totals")
                    if ov_price and ov_point:
                        over_prob = norm_cdf(exp_total - ov_point, 0, total_std)
                        if over_prob > 0.5:
                            ov_p = american_to_prob(ov_price)
                            ov_edge = round(over_prob*100 - (ov_p*100 if ov_p else 0), 1) if ov_p else None
                            pick_entry["total"] = {
                                "pick": "Over", "prob": over_prob*100,
                                "detail": f"o{ov_point}", "odds": ov_price, "book": ov_book,
                                "ev": compute_ev(over_prob*100, ov_price), "edge": ov_edge,
                            }
                        else:
                            un_price, un_book, _ = extract_market_odds(og, "totals", "Under")
                            un_p = american_to_prob(un_price or ov_price)
                            un_edge = round((1-over_prob)*100 - (un_p*100 if un_p else 0), 1) if un_p else None
                            pick_entry["total"] = {
                                "pick": "Under", "prob": (1-over_prob)*100,
                                "detail": f"u{ov_point}", "odds": un_price or ov_price,
                                "book": un_book or ov_book,
                                "ev": compute_ev((1-over_prob)*100, un_price or ov_price), "edge": un_edge,
                            }
                    else:
                        pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"(est ~{exp_total:.1f})", "ev": None, "odds": "N/A", "book": "", "edge": None}
                else:
                    cal_p = max(cal_hp, cal_ap) if cal_hp else max(ml_hp, ml_ap)
                    raw_p = max(ml_hp, ml_ap)
                    pick_entry["moneyline"] = {"pick": hn if ml_hp > 0.50 else an, "prob": cal_p*100, "raw_prob": raw_p*100, "ev": None, "odds": "N/A", "book": "", "edge": None}
                    pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}
                    pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}
                    pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"~{exp_total:.1f}", "ev": None, "odds": "N/A", "book": "", "edge": None}
            else:
                cal_p = max(cal_hp, cal_ap) if cal_hp else max(ml_hp, ml_ap)
                raw_p = max(ml_hp, ml_ap)
                pick_entry["moneyline"] = {"pick": hn if ml_hp > 0.50 else an, "prob": cal_p*100, "raw_prob": raw_p*100, "ev": None, "odds": "N/A", "book": "", "edge": None}
                pick_entry["spread_minus"] = {"pick": spr_fav_team, "prob": spr_fav_prob*100, "detail": "-1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}
                pick_entry["spread_plus"] = {"pick": spr_dog_team, "prob": spr_dog_prob*100, "detail": "+1.5", "ev": None, "odds": "N/A", "book": "", "edge": None}
                pick_entry["total"] = {"pick": ov_verdict, "prob": ov_pct, "detail": f"~{exp_total:.1f}", "ev": None, "odds": "N/A", "book": "", "edge": None}

            pick_entry["border_color"] = border
            picks.append(pick_entry)

    # Log all predictions for tracking
    try:
        from bankroll import log_predictions
        log_predictions(picks)
    except:
        pass

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
        for m in ["moneyline","spread_plus","spread_minus","total"]:
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

    if _get_perms(st.session_state.user).get("daily_picks", True) and len(upcoming) > 0:
        has_odds = bool(odds_raw)
        def _high_conf(game_row):
            for mk in ("moneyline", "spread_plus", "total"):
                e = game_row.get(mk)
                if not isinstance(e, dict): continue
                prob_val = e.get("prob")
                odds_val = e.get("odds", "N/A")
                if odds_val in ("N/A", "—", "", None): continue
                edge_val = e.get("edge")
                if edge_val is not None and edge_val > 8: return True
                if prob_val is not None and prob_val >= 75: return True
            return False
        high_conf_mask = upcoming.apply(_high_conf, axis=1)
        hc_count = high_conf_mask.sum()
        if hc_count < len(upcoming):
            st.caption(f"🎯 Solo picks con alta confianza (edge > 8% o prob ≥ 75%)")
            upcoming = upcoming[high_conf_mask]
        st.markdown(f"### 📋 Picks del Día ({len(upcoming)} juegos)")
        flat_rows = []
        now_tz = datetime.now(TZ)
        _existing_picks = set()
        try:
            from bankroll import load_picks
            _today_str = datetime.now(TZ).strftime("%Y-%m-%d")
            for ep in load_picks().get("history", []):
                if ep.get("date", "") == _today_str:
                    _existing_picks.add((ep.get("game", "").strip(), ep.get("market", "").strip(), ep.get("team", "").strip()))
        except:
            pass
        for _, r in upcoming.iterrows():
            gl = f"{r['away_abbrev']} @ {r['home_abbrev']}"
            cgs = r.get("coded_game_state", "")
            game_dt = r.get("game_dt")
            score = r.get("final", "")
            if isinstance(score, float) and math.isnan(score):
                score = ""
            time_str = r.get("game_time", "")
            import math
            if isinstance(time_str, float) and math.isnan(time_str):
                time_str = ""
            if cgs == "F":
                t = score if score else "Final"
            elif cgs == "I":
                t = f"🔴 {score}" if score else "🔴 EN VIVO"
            elif game_dt is not None:
                mins = (game_dt - now_tz).total_seconds() / 60.0
                if 0 <= mins <= 15:
                    t = f"⏳ {time_str}"
                elif mins < 0:
                    t = time_str
                else:
                    t = time_str
            else:
                t = time_str
            _best_row = None
            _best_score = -1
            for mk, ml in [("moneyline","ML"),("spread_plus","RL +1.5"),("total","O/U")]:
                p = r.get(mk)
                if not p: continue
                _ev = p.get("ev")
                prob_val = p.get("prob")
                odds_val = p.get("odds","N/A")
                edge = None
                if prob_val and odds_val not in ("N/A","—",""):
                    try:
                        oi = int(str(odds_val).replace("$",""))
                        ip = american_to_prob(oi)
                        if ip: edge = round(prob_val - ip * 100, 1)
                    except: pass
                score = 0
                if edge is not None and edge > 2 and prob_val is not None:
                    score = prob_val + edge * 0.3
                elif prob_val is not None:
                    score = prob_val
                if prob_val is not None and (prob_val < 60 or prob_val > 89):
                    score = 0
                if mk == "spread_plus":
                    score = score * 0.7   # heavier penalty — RL +1.5 dominates picks
                if mk == "total" and p.get("detail","").startswith("o"):
                    score = 0  # skip Over — model Under bias 62% vs Over 49%
                if score > _best_score:
                    _best_score = score
                    pick_name = p.get("pick","—")
                    if mk == "total":
                        display_name = fmt_ou(pick_name, p.get("detail",""))
                    elif mk == "spread_plus":
                        display_name = f"{pick_name} (+1.5)"
                    else:
                        display_name = pick_name
                    flames = ""
                    if (edge is not None and edge > 8) or (prob_val is not None and prob_val >= 75):
                        flames = "🔥🔥🔥"
                    elif (edge is not None and edge > 5) or (prob_val is not None and prob_val >= 65):
                        flames = "🔥🔥"
                    elif (edge is not None and edge > 3) or (prob_val is not None and prob_val >= 60):
                        flames = "🔥"
                    display_pick = f"{flames} {display_name}" if flames else display_name
                    _best_row = {
                        "Juego": gl, "Hora": t, "M": ml,
                        "Pick": display_pick,
                        "Prob": f"{prob_val:.0f}%" if prob_val else "",
                        "Odds": odds_val,
                        "EV": f"{_ev:.1%}" if _ev is not None else "",
                    }
            if _best_row:
                flat_rows.append(_best_row)
        # Sort by probability (highest first) — backtest shows 60-89% range is optimal
        flat_rows.sort(key=lambda r: float(r['Prob'].replace('%','')), reverse=True)
        flat_rows = flat_rows[:6]  # Top 6 picks only
        if flat_rows:
            html = """<div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;color:#212121;font-size:14px">
                <thead><tr style="background:#E53935;color:#FFF">
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Juego</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Hora</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">M</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Pick</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Prob</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Odds</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">EV</th>
                </tr></thead><tbody>"""
            for i, row in enumerate(flat_rows):
                bg = "#F8F9FA" if i % 2 == 0 else "#FFF"
                html += f"""<tr style="background:{bg}">
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['Juego']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['Hora']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['M']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['Pick']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['Prob']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['Odds']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{row['EV']}</td>
                </tr>"""
            html += "</tbody></table></div>"
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Usa la sección 🏆 Recomendaciones para registrar tus picks.")
        else:
            st.info("No hay picks disponibles para mostrar.")

    # ── Recomendaciones ──
    recs = []
    try:
        from bankroll import recommend_stake, get_pnl
        actual_bankroll = get_pnl()["weekly_bankroll"]
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

            for mkt_key, label in [("moneyline", "ML"), ("total", "O/U")]:
                entry = p.get(mkt_key)
                if not entry: continue
                edge = entry.get("edge") if mkt_key == "moneyline" else get_edge(entry)
                prob = entry.get("prob", 0)
                if edge is not None and edge > 8:
                    pass  # 🔥🔥🔥 high confidence
                elif edge is not None and edge > 5:
                    pass  # 🔥🔥 medium confidence
                elif prob >= 75:
                    pass  # 🔥🔥🔥 high probability
                elif prob >= 65 and edge is not None and edge > 3:
                    pass  # 🔥 decent
                else:
                    continue

                pick_team = entry.get("pick", "")
                odds_str = entry.get("odds", "N/A")
                odds_int = 0
                has_real_odds = False
                try:
                    odds_int = int(str(odds_str).replace("$",""))
                    has_real_odds = True
                except: pass

                if has_real_odds:
                    stake, units, stake_label = recommend_stake(prob/100, odds_int, bankroll=actual_bankroll)
                else:
                    stake, units, stake_label = 0, 0, "—"

                home = p.get("home_team", "")
                away = p.get("away_team", "")
                venue = p.get("venue", "su estadio")
                hp = p.get("home_pitcher", "TBD")
                ap = p.get("away_pitcher", "TBD")
                if label == "O/U":
                    detail = entry.get("detail", "")
                    if detail.startswith("o"):
                        continue  # skip Over — model has strong Under bias (62% Under vs 49% Over)
                    side = "Under"
                    exp_t = p.get('exp_total', 0)
                    vars_ou = [
                        f"El duelo {ap} vs {hp} pinta cerrado. Proyectamos ~{exp_t:.1f} carreras, muy por debajo de {detail}. {side} es el lado con valor.",
                        f"{ap} y {hp} son abridores que limitan el daño. En juegos con este perfil, el {side} {detail} suele cumplirse. Proyectamos {exp_t:.1f} carreras.",
                        f"Ofensivas frías + pitcheo sólido = juego de pocas carreras. Esperamos solo {exp_t:.1f} anotaciones. {side} {detail}.",
                        f"En {venue}, juegos como este promedian menos de {detail}. Con {ap} vs {hp}, duelo de lanzadores. {side} {detail}.",
                    ]
                    reason = vars_ou[len(recs) % len(vars_ou)]
                else:
                    vars_ml = [
                        f"{pick_team} con {ap if pick_team == away else hp} en la loma tiene ventaja. Su ofensiva ha estado encendida. Pick con fundamentos.",
                        f"El lineup de {pick_team} ha castigado a lanzadores como {hp if pick_team == away else ap}. Buen spot para confiar en ellos.",
                        f"{pick_team} llega enrachado y su abridor {ap if pick_team == away else hp} domina a este rival. Valor claro.",
                        f"Los números dan a {pick_team} como favorito. Su bullpen cierra juegos y el lineup responde en momentos clave.",
                    ]
                    reason = vars_ml[len(recs) % len(vars_ml)]
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
                    "reason": reason,
                })

        if recs and _get_perms(st.session_state.user).get("recommendations", True):
            # One recommendation per game (best edge)
            best_per_game = {}
            for r in recs:
                g = r["game"]
                if g not in best_per_game or r["edge"] > best_per_game[g]["edge"]:
                    best_per_game[g] = r
            recs = sorted(best_per_game.values(), key=lambda x: x["prob"], reverse=True)
            st.divider()
            st.markdown("## 🏆 Recomendaciones del Día")
            has_real_odds = any(r.get("odds","N/A") not in ("N/A","—","") for r in recs)
            if has_real_odds:
                st.caption(f"Top {min(len(recs),6)} de {len(recs)} — Kelly Criterion (25% fraccional, bankroll ${actual_bankroll:,.0f}). Stakes aumentan con el bankroll.")
            else:
                st.caption(f"Top {min(len(recs),4)} de {len(recs)} — Basado en probabilidad del modelo (sin odds disponibles).")

            # HTML table with inline form buttons (safe: requires click, no bot auto-follow)
            from bankroll import load_picks
            today_str = datetime.now(TZ).strftime("%Y-%m-%d")
            existing = set()
            for ep in load_picks().get("history", []):
                if ep.get("date", "") != today_str:
                    continue
                epg = ep.get("game", "").strip()
                epm = ep.get("market", "").strip()
                ept = ep.get("team", "").strip()
                if epg and epm and ept:
                    existing.add((epg, epm, ept))
            html_rows = ""
            _reasons_list = []
            for i, r in enumerate(recs[:4]):
                is_regd = (r["game"].strip(), r["market"].strip(), r["pick"].strip()) in existing
                icon = "🔥🔥🔥" if (r.get("edge") or 0) > 8 or r["prob"] >= 75 else "🔥🔥" if (r.get("edge") or 0) > 5 or r["prob"] >= 65 else "🔥"
                pick_str = fmt_ou(r["pick"], r.get("entry",{}).get("detail",""))
                # Warning for O/U when predicted total is close to the line
                warn = ""
                if r["market"] == "O/U" and r.get("entry",{}).get("detail",""):
                    try:
                        ov_point = float(r["entry"]["detail"].replace("o","").replace("u",""))
                        exp_tot = r.get("pick_dict",{}).get("exp_total")
                        if exp_tot and abs(exp_tot - ov_point) < 0.5:
                            warn = "⚠️ "
                    except: pass
                stake_str = f"${r['stake']:.0f}" if r["stake"] > 0 else "—"
                if role != "admin":
                    btn = "<span style='color:#666'>🔒</span>"
                elif is_regd:
                    btn = "<span style='color:#E53935'>✅</span>"
                else:
                    btn = f"<form action='' method='GET' style='display:inline;margin:0;padding:0'><input type='hidden' name='reg_pick' value='{i}'><input type='hidden' name='u' value='{st.session_state.user}'><button type='submit' style='background:none;border:none;cursor:pointer;font-size:18px;padding:0;color:#E53935' title='Registrar'>📝</button></form>"
                _reason = r.get("reason", "")
                html_rows += f"""<tr style="background:{'#F8F9FA' if i%2==0 else '#EEEEEE'}">
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{r['game']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{r['market']}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{warn}{pick_str}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{icon} {r['edge']:+.1f}%</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0">{stake_str}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #E0E0E0;text-align:center">{btn}</td>
                </tr>"""
                _reasons_list.append(f"**{r['game']}** ({r['market']}): {_reason}")
        st.markdown(f"""<div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;color:#212121;font-size:14px">
                <thead><tr style="background:#E53935;color:#FFF">
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Juego</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Mercado</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Pick</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Edge</th>
                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Stake</th>
                    <th style="padding:8px;text-align:center;border-bottom:2px solid #E0E0E0">Reg</th>
                </tr></thead>
                <tbody>{html_rows}</tbody>
            </table></div>""", unsafe_allow_html=True)
        if _reasons_list:
            st.markdown("**📖 Explicación:**")
            for line in _reasons_list:
                st.markdown(f"- {line}")

            # Handle form submission via query param
            reg_idx = st.query_params.get("reg_pick") if role == "admin" else None
            if reg_idx is not None:
                proc_key = f"processed_reg_{reg_idx}"
                if not st.session_state.get(proc_key, False):
                    st.session_state[proc_key] = True
                    idx = int(reg_idx)
                    if 0 <= idx < len(recs[:4]):
                        r = recs[idx]
                        sk = r["stake"]
                        sl = r["stake_label"]
                        bk = get_pnl()["weekly_bankroll"]
                        if sk > 0:
                            from bankroll import add_pick
                            pt = r["pick"]
                            dtl = r.get("entry", {}).get("detail", "")
                            gl = f"{r['pick_dict']['away_abbrev']} @ {r['pick_dict']['home_abbrev']}"
                            os_ = r["odds"]
                            oi = int(str(os_).replace("$","")) if os_ not in ("N/A","—","") else 0
                            pv = r["prob"]/100.0
                            ts = datetime.now(TZ).strftime("%Y-%m-%d")
                            pid = add_pick(ts, gl, r["market"], pv, oi, sk, bk, sl, pt, dtl)
                            notify_pick(gl, r["market"], pt, sk, oi, bk, pick_id=pid)
                            sync_picks_to_github()
                    _u = st.query_params.get("u", "")
                    st.query_params.clear()
                    if _u:
                        st.query_params["u"] = _u
                    st.rerun()
    except ImportError:
        pass

    # ── Mis Picks Registrados ──
    try:
        from bankroll import get_pnl, load_picks
        pnl = get_pnl()
        data = load_picks()
        st.divider()
        st.markdown("## 📋 Mis Picks Registrados")
        if role != "admin":
            st.info("👁️ Sección visible solo para administradores")
            st.divider()
        else:
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
                wk_start = data.get("weekly_start", "2026-01-01")
                weekly_picks = [p for p in data["history"] if p.get("date", "") >= wk_start]
                st.write(f"DEBUG: {len(data['history'])} total picks, {len(weekly_picks)} this week")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("📅 Semanal", f"${pnl['weekly_bankroll']:.0f}", delta=f"${pnl['weekly_profit']:+.0f}")
                mc2.metric("💰 Histórico", f"${pnl['profit']:+.0f}", delta=f"{pnl['roi']:+.0f}%")
                mc3.metric("Record Semanal", f"{pnl['weekly_wins']}-{pnl['weekly_losses']}")
                mc4.metric("Pendientes", sum(1 for p in weekly_picks if not p.get("settled")))

                # ── Win rate by market ──
                settled = [p for p in data["history"] if p.get("result") in ("W", "L", "P", "C")]
                if settled:
                    markets = {}
                    for p in settled:
                        mkt = p.get("market", "?")
                        if mkt not in markets:
                            markets[mkt] = {"w": 0, "l": 0, "profit": 0.0}
                        if p["result"] == "W":
                            markets[mkt]["w"] += 1
                        else:
                            markets[mkt]["l"] += 1
                        markets[mkt]["profit"] += p.get("profit", 0) or 0
                    mr = []
                    for mkt in ("ML", "RL +1.5", "O/U"):
                        d = markets.get(mkt)
                        if d and (d["w"] + d["l"]) > 0:
                            tot = d["w"] + d["l"]
                            pct = round(d["w"] / tot * 100)
                            pf = d["profit"]
                            icon = "🟢" if pf > 0 else "🔴" if pf < 0 else "⚪"
                            mr.append(f"{icon} *{mkt}*: {d['w']}-{d['l']} ({pct}%) ${pf:+.0f}")
                    if mr:
                        with st.expander("📊 Win rate por mercado", expanded=False):
                            for line in mr:
                                st.markdown(line)

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
                        chart = alt.Chart(cdf).mark_line(point=True, color="#E53935").encode(
                            x=alt.X("#:Q", title="Pick #", axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y("Bankroll:Q", title="Bankroll ($)", scale=alt.Scale(zero=False)),
                            tooltip=["Fecha:N", "Bankroll:Q"],
                        ).properties(height=200)
                        st.altair_chart(chart, use_container_width=True)
                    except:
                        pass

                total_profit = 0
                rows = []
                for p in reversed(weekly_picks):
                    result = p.get("result")
                    r_icon = "⏳ Pendiente"
                    if result == "W":
                        r_icon = "✅ Ganado"
                    elif result == "L":
                        r_icon = "❌ Perdido"
                    elif result == "P":
                        r_icon = "🤝 Push"
                    elif result == "C":
                        r_icon = "❌ Cancelado"
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
                                else:
                                    r_icon = "⏳ Pendiente"
                        else:
                            r_icon = "⏳ Pendiente"

                    profit = p.get("profit")
                    profit_str = "—"
                    if profit is not None:
                        total_profit += profit
                        profit_str = f"${profit:+.0f}"
                    prob_str = f"{p.get('model_prob', 0):.0%}"
                    odds_str = f"${p.get('odds', 0):+d}"
                    stake_str = f"${p.get('stake', 0):.0f}"

                    rows.append({
                        "Fecha": p.get("date",""),
                        "Juego": p.get("game",""),
                        "Mercado": p.get("market",""),
                        "Pick": fmt_ou(p.get("team",""), p.get("detail","")),
                        "Prob": prob_str,
                        "Cuota": odds_str,
                        "Stake": stake_str,
                        "Estado": r_icon,
                        "Profit": profit_str,
                    })

                # Tabla de picks con botón eliminar inline
                if rows:
                    html = """<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;color:#212121;font-size:13px">
                    <thead><tr style="background:#E53935;color:#FFF">
                        <th style="padding:6px 8px;text-align:left;border-bottom:2px solid #E0E0E0">Fecha</th>
                        <th style="padding:6px 8px;text-align:left;border-bottom:2px solid #E0E0E0">Juego</th>
                        <th style="padding:6px 8px;text-align:left;border-bottom:2px solid #E0E0E0">Mercado</th>
                        <th style="padding:6px 8px;text-align:left;border-bottom:2px solid #E0E0E0">Pick</th>
                        <th style="padding:6px 8px;text-align:center;border-bottom:2px solid #E0E0E0">Estado</th>
                        <th style="padding:6px 8px;text-align:right;border-bottom:2px solid #E0E0E0">Profit</th>
                        <th style="padding:6px 8px;text-align:center;border-bottom:2px solid #E0E0E0">Del</th>
                    </tr></thead><tbody>"""
                    for i, (row, p) in enumerate(zip(rows, list(reversed(data["history"])))):
                        pid = p.get("id", i + 1)
                        is_settled = p.get("result") in ("W", "L")
                        bg = "#F8F9FA" if i % 2 == 0 else "#EEEEEE"
                        profit_str = row["Profit"]
                        estado = row["Estado"]
                        if is_settled:
                            del_btn = "—"
                        else:
                            del_btn = f"<form action='' method='GET' style='display:inline;margin:0;padding:0'><input type='hidden' name='del_pick' value='{pid}'><input type='hidden' name='u' value='{st.session_state.user}'><button type='submit' style='background:none;border:none;cursor:pointer;font-size:16px;padding:0;color:#FF6B6B' title='Eliminar'>✕</button></form>"
                        html += f"""<tr style="background:{bg}">
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0">{row['Fecha']}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0">{row['Juego']}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0">{row['Mercado']}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0">{row['Pick']}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0;text-align:center">{estado}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0;text-align:right">{profit_str}</td>
                            <td style="padding:4px 6px;border-bottom:1px solid #E0E0E0;text-align:center">{del_btn}</td>
                        </tr>"""
                    html += "</tbody></table></div>"
                    st.markdown(html, unsafe_allow_html=True)

                green = total_profit >= 0
                st.markdown(f"Profit total: <span style='color:{'#E53935' if green else '#FF6B6B'}'><b>${total_profit:+.2f}</b></span>", unsafe_allow_html=True)

                # ── Ajuste de profit real (casino) ──
                # ── Model calibration ──
                if settled:
                    bins = [(50, 55), (55, 60), (60, 65), (65, 70), (70, 100)]
                    cal_rows = []
                    for lo, hi in bins:
                        pool = [p for p in settled if lo <= p.get("model_prob", 0.5) * 100 < hi]
                        if not pool:
                            continue
                        wins = sum(1 for p in pool if p["result"] == "W")
                        tot = len(pool)
                        actual = round(wins / tot * 100)
                        predicted = round((lo + hi) / 2)
                        cal_rows.append({
                            "Rango": f"{lo}-{hi}%", "Picks": tot, "Ganados": wins,
                            "Real": f"{actual}%", "Esperado": f"{predicted}%", "Diff": f"{actual - predicted:+.0f}%",
                        })
                    if _get_perms(st.session_state.user).get("calibration", False):
                        with st.expander("📐 Calibración del modelo", expanded=False):
                            _html = """<table style="width:100%;border-collapse:collapse;color:#212121;font-size:14px">
                                <thead><tr style="background:#E53935;color:#FFF">
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Rango</th>
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Picks</th>
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Ganados</th>
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Real</th>
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Esperado</th>
                                    <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Diff</th>
                                </tr></thead><tbody>"""
                            for i, cr in enumerate(cal_rows):
                                bg = "#F8F9FA" if i%2==0 else "#FFF"
                                _html += f"<tr style='background:{bg}'><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Rango']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Picks']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Ganados']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Real']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Esperado']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{cr['Diff']}</td></tr>"
                            _html += "</tbody></table>"
                            st.markdown(_html, unsafe_allow_html=True)
                            if len(cal_rows) >= 2:
                                import altair as alt
                                cd = pd.DataFrame(cal_rows)
                                cd["Real"] = cd["Real"].str.replace("%","").astype(int)
                                cd["Esperado"] = cd["Esperado"].str.replace("%","").astype(int)
                                cd["Rango"] = cd["Rango"].str.replace("%","")
                                chart = alt.Chart(cd).mark_line(point=True).encode(
                                    x=alt.X("Rango:N", title="Probabilidad estimada"),
                                    y=alt.Y("Real:Q", title="Frecuencia real (%)", scale=alt.Scale(zero=False)),
                                    color=alt.value("#E53935"),
                                ).properties(height=200)
                                chart += alt.Chart(cd).mark_line(strokeDash=[4,4], color="#888").encode(
                                    x="Rango:N", y="Esperado:Q")
                                st.altair_chart(chart, use_container_width=True)

                # ── Backtest completo ──
                if settled and _get_perms(st.session_state.user).get("model_stats", True):
                    with st.expander("📈 Backtest completo", expanded=False):
                        if st.button("🔄 Ejecutar Backtest"):
                            with st.spinner("Analizando predicciones..."):
                                try:
                                    from predictions import run_backtest
                                    msg = run_backtest()
                                    st.markdown(msg.replace("\n", "  \n"))
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            total_staked = sum(p.get("stake", 0) for p in settled)
                            flat_profit = sum(p.get("profit") or 0 for p in settled)
                            kelly_roi = round(flat_profit / total_staked * 100, 1) if total_staked > 0 else 0
                            st.markdown(f"""
                            **Kelly Criterion (25% fraccional)**
                            - Picks: {len(settled)}
                            - Profit: ${flat_profit:+.2f}
                            - ROI: {kelly_roi}%
                            - Bankroll final: ${pnl['bankroll']:.2f}
                            """)

            # ── Predicciones history ──
            if _get_perms(st.session_state.user).get("model_stats", True):
                try:
                    with open(os.path.join(os.path.dirname(__file__), "predictions_log.json")) as f:
                        pred_data = json.load(f)
                    all_preds = pred_data.get("predictions", [])
                    settled_preds = [p for p in all_preds if p.get("settled")]
                    if settled_preds:
                        pw = sum(1 for p in settled_preds if p["result"] == "W")
                        pl = sum(1 for p in settled_preds if p["result"] == "L")
                        pt = pw + pl
                        pct = round(pw / pt * 100) if pt > 0 else 0
                        st.markdown(f"**📊 Rendimiento del modelo:** {pw}-{pl} ({pct}%) en {pt} predicciones liquidadas")
                        # By market
                        mkt_rows = []
                        for mkt in ("ML", "RL +1.5", "O/U"):
                            pool = [p for p in settled_preds if p["market"] == mkt]
                            if not pool:
                                continue
                            mw = sum(1 for p in pool if p["result"] == "W")
                            mt = len(pool)
                            mp = round(mw / mt * 100)
                            mkt_rows.append({"Mercado": mkt, "G-P": f"{mw}-{mt-mw}", "Picks": mt, "%": f"{mp}%"})
                        if mkt_rows:
                            with st.expander("📊 Por mercado", expanded=False):
                                _html = """<table style="width:100%;border-collapse:collapse;color:#212121;font-size:14px">
                                    <thead><tr style="background:#E53935;color:#FFF">
                                        <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Mercado</th>
                                        <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">G-P</th>
                                        <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">Picks</th>
                                        <th style="padding:8px;text-align:left;border-bottom:2px solid #E0E0E0">%</th>
                                    </tr></thead><tbody>"""
                                for i, mr in enumerate(mkt_rows):
                                    bg = "#F8F9FA" if i%2==0 else "#FFF"
                                    _html += f"<tr style='background:{bg}'><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{mr['Mercado']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{mr['G-P']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{mr['Picks']}</td><td style='padding:6px 8px;border-bottom:1px solid #E0E0E0'>{mr['%']}</td></tr>"
                                _html += "</tbody></table>"
                                st.markdown(_html, unsafe_allow_html=True)
                except: pass
            else:
                st.info("💡 Aún no has registrado picks. Usa el botón **📝** en las tarjetas o recomendaciones para empezar.")
    except Exception as _me:
        st.error(f"Error en Mis Picks: {_me}")
    
    if _get_perms(st.session_state.user).get("detailed_table", False):
        st.divider()
        with st.expander("🔬 Tabla detallada"):
            cols_avail = [c for c in ["away_team","home_team","ml_home_prob","ml_away_prob","exp_total","spr_team","spr_prob"] if c in df.columns]
            if cols_avail:
                sd2 = df[cols_avail].copy()
                _html = """<table style="width:100%;border-collapse:collapse;color:#212121;font-size:13px">
                    <thead><tr style="background:#E53935;color:#FFF">"""
                for col in cols_avail:
                    _html += f"<th style='padding:6px 8px;text-align:left;border-bottom:2px solid #E0E0E0'>{col}</th>"
                _html += "</tr></thead><tbody>"
                for i, (_, row) in enumerate(sd2.iterrows()):
                    bg = "#F8F9FA" if i%2==0 else "#FFF"
                    _html += f"<tr style='background:{bg}'>"
                    for col in cols_avail:
                        val = row[col]
                        if isinstance(val, float):
                            val = f"{val:.1f}"
                        _html += f"<td style='padding:4px 6px;border-bottom:1px solid #E0E0E0'>{val}</td>"
                    _html += "</tr>"
                _html += "</tbody></table>"
                st.markdown(_html, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        sync_picks_from_github()
    except: pass
    main()
