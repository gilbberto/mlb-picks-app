"""
calibrate.py — Calibración del modelo RF+MC.
Compara probabilidades predichas vs resultados reales en juegos completados.
"""
import requests, numpy as np, math, pickle, os
from datetime import datetime, timedelta
from collections import defaultdict

MLB_API = "https://statsapi.mlb.com/api/v1"
BASE = os.path.join(os.path.dirname(__file__), "")
CACHE = {}

def cget(url):
    if url in CACHE: return CACHE[url]
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            d = r.json(); CACHE[url] = d; return d
    except: pass
    return None

def sf(v, d=0.0):
    try: return float(v) if v is not None else d
    except: return d

PARK = {"Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
        "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
        "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
        "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
        "Oracle Park": 0.98, "Petco Park": 0.97, "PNC Park": 0.97,
        "Tropicana Field": 0.96, "Target Field": 0.96, "Oakland Coliseum": 0.97}

def team_id_map():
    d = cget(f"{MLB_API}/teams?sportIds=1")
    return {t["id"]: t.get("abbreviation","??") for t in d.get("teams",[])}

def get_standings(date_str):
    d = cget(f"{MLB_API}/standings?leagueId=103,104&season=2026&date={date_str}")
    rows = {}
    for r in d.get("records",[]):
        for t in r.get("teamRecords",[]):
            rows[t["team"]["id"]] = {
                "wp": sf(t.get("wins",0)) / max(sf(t.get("wins",0)) + sf(t.get("losses",0)), 1),
                "rs": sf(t.get("runsScored",0)) / max(sf(t.get("gamesPlayed",0)), 1),
                "ra": sf(t.get("runsAllowed",0)) / max(sf(t.get("gamesPlayed",0)), 1),
            }
    return rows

def team_stats(tid, info):
    s = info.get(tid, {})
    return (s.get("wp",0.5), s.get("rs",4.5), s.get("ra",4.5))

def recent_games(tid, up_to):
    pr = []
    for g in ALL_GAMES.get(tid, []):
        if g.get("gameDate","") < up_to:
            pr.append(g)
    return pr

def form(games, tid):
    r10 = sorted(games, key=lambda x: x.get("gameDate",""), reverse=True)[:10]
    if not r10: return (0.5, 4.5, 4.5)
    w, rs, ra = 0, [], []
    for g in r10:
        home = g["teams"]["home"]["team"]["id"] == tid
        ms = sf(g["teams"]["home"]["score"] if home else g["teams"]["away"]["score"])
        os_ = sf(g["teams"]["away"]["score"] if home else g["teams"]["home"]["score"])
        rs.append(ms); ra.append(os_)
        if ms > os_: w += 1
    return (w/len(rs), np.mean(rs), np.mean(ra)) if rs else (0.5,4.5,4.5)

def elo_rating(roster, tid, default=1500):
    r10 = sorted(roster, key=lambda x: x.get("gameDate",""), reverse=True)[:10]
    score = 1500
    for g in r10:
        home = g["teams"]["home"]["team"]["id"] == tid
        ms = sf(g["teams"]["home"]["score"] if home else g["teams"]["away"]["score"])
        os_ = sf(g["teams"]["away"]["score"] if home else g["teams"]["home"]["score"])
        expected = 1 / (1 + 10**((score - (score + (ms - os_)*20)) / 400))
        score += 32 * (1 if ms > os_ else 0 - expected)
    return score

def build_feature(game, pitcher=None):
    if pitcher is None: pitcher = {}
    gd = game.get("gameDate","").split("T")[0]
    up_to = gd
    hid = game["teams"]["home"]["team"]["id"]
    aid = game["teams"]["away"]["team"]["id"]
    stat_info = get_standings(gd)

    h_wp, h_rs, h_ra = team_stats(hid, stat_info)
    a_wp, a_rs, a_ra = team_stats(aid, stat_info)
    hg = recent_games(hid, up_to)
    ag = recent_games(aid, up_to)
    hf = form(hg, hid)
    af = form(ag, aid)
    h_elo = elo_rating(hg, hid)
    a_elo = elo_rating(ag, aid)
    pf = PARK.get(game.get("venue",{}).get("name",""), 1.0)

    return {
        "h_elo": h_elo, "a_elo": a_elo,
        "h_wp": h_wp, "a_wp": a_wp,
        "h_rs": h_rs, "a_rs": a_rs,
        "h_ra": h_ra, "a_ra": a_ra,
        "h_ops": 0.720 + (h_rs - 4.5)*0.02, "a_ops": 0.720 + (a_rs - 4.5)*0.02,
        "h_whip": 1.35 + (4.5 - h_ra)*0.02, "a_whip": 1.35 + (4.5 - a_ra)*0.02,
        "h_era": 4.5 + (4.5 - h_ra)*0.5, "a_era": 4.5 + (4.5 - a_ra)*0.5,
        "park": pf,
        "hp_era": pitcher.get("hp_era", 4.2), "hp_k9": pitcher.get("hp_k9", 8.0),
        "hp_bb9": pitcher.get("hp_bb9", 3.2), "hp_hr9": pitcher.get("hp_hr9", 1.2),
        "hp_v": 0,
        "ap_era": pitcher.get("ap_era", 4.2), "ap_k9": pitcher.get("ap_k9", 8.0),
        "ap_bb9": pitcher.get("ap_bb9", 3.2), "ap_hr9": pitcher.get("ap_hr9", 1.2),
        "ap_v": 0,
        "h_rest": 1, "a_rest": 1,
        "hw": 1 if sf(game["teams"]["home"]["score"]) > sf(game["teams"]["away"]["score"]) else 0,
        "rd": sf(game["teams"]["home"]["score"]) - sf(game["teams"]["away"]["score"]),
        "tot": sf(game["teams"]["home"]["score"]) + sf(game["teams"]["away"]["score"]),
    }

# ─── Load models (XGBoost first, fallback RF) ───
print("=== Loading models ===")
model_type = "RF"
try:
    import xgboost as xgb
    with open(BASE + "xgb_hw.pkl", "rb") as f: m_hw = pickle.load(f)
    with open(BASE + "xgb_rd.pkl", "rb") as f: m_rd = pickle.load(f)
    with open(BASE + "xgb_tot.pkl", "rb") as f: m_tot = pickle.load(f)
    with open(BASE + "xgb_cols.pkl", "rb") as f: cols = pickle.load(f)
    model_type = "XGBoost"
except Exception:
    with open(BASE + "rf_hw.pkl", "rb") as f: m_hw = pickle.load(f)
    with open(BASE + "rf_rd.pkl", "rb") as f: m_rd = pickle.load(f)
    with open(BASE + "rf_tot.pkl", "rb") as f: m_tot = pickle.load(f)
    with open(BASE + "rf_cols.pkl", "rb") as f: cols = pickle.load(f)
print(f"  {model_type} models loaded. {len(cols)} features: {cols}")

# ─── Fetch games ───
print("\n=== Fetching completed games ===")
raw = cget(f"{MLB_API}/schedule?sportId=1&startDate=2026-03-27&endDate=2026-06-01")
all_games = []
for d in raw.get("dates", []):
    for g in d.get("games", []):
        if g.get("status", {}).get("codedGameState") == "F":
            all_games.append(g)
all_games.sort(key=lambda x: x.get("gameDate", ""))

# Build team index for recent_games
ALL_GAMES = defaultdict(list)
for g in all_games:
    hid = g["teams"]["home"]["team"]["id"]
    aid = g["teams"]["away"]["team"]["id"]
    ALL_GAMES[hid].append(g)
    ALL_GAMES[aid].append(g)

print(f"  {len(all_games)} completed games")
if len(all_games) == 0:
    print("No games found!")
    exit()

# Use last 30% of games as validation set
TEST_SIZE = min(300, int(len(all_games) * 0.3))
test_games = all_games[-TEST_SIZE:]
print(f"  Validation set: {len(test_games)} games")

# ─── Fetch starting pitchers from boxscores (parallel) ───
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_starter(game_pk, team_side):
    try:
        d = cget(f"{MLB_API}/game/{game_pk}/boxscore")
        if d:
            pitchers = d.get("teams", {}).get(team_side, {}).get("pitchers", [])
            if pitchers: return pitchers[0]
    except: pass
    return None

def fetch_pitcher_stats(pid):
    if not pid: return {}
    d = cget(f"{MLB_API}/people/{pid}/stats?stats=season&season=2026&group=pitching")
    if not d: return {}
    s = d.get("stats", [{}])[0].get("splits", [])
    if not s: return {}
    st = s[0].get("stat", {})
    ip_raw = st.get("inningsPitched", "0")
    ip_val = 0
    if ip_raw and "." in str(ip_raw):
        parts = str(ip_raw).split(".")
        ip_val = int(parts[0]) + int(parts[1]) / 3.0 if len(parts) > 1 else float(parts[0])
    else:
        ip_val = float(ip_raw or 0)
    return {"era": sf(st.get("era")), "k9": sf(st.get("strikeoutsPer9Inn")),
            "bb9": sf(st.get("walksPer9Inn")), "hr9": sf(st.get("homeRunsPer9")),
            "ip": ip_val}

print("  Fetching boxscores for starting pitchers...")
pitchers_cache = {}
def get_starter_stats(game_pk, side):
    pid = get_starter(game_pk, side)
    if pid and pid not in pitchers_cache:
        pitchers_cache[pid] = fetch_pitcher_stats(pid)
    return pitchers_cache.get(pid, {})

with ThreadPoolExecutor(max_workers=20) as ex:
    futures = {}
    for g in test_games:
        pk = g["gamePk"]
        futures[ex.submit(get_starter, pk, "home")] = (pk, "home")
        futures[ex.submit(get_starter, pk, "away")] = (pk, "away")
    for f in as_completed(futures):
        pk, side = futures[f]
        try:
            pid = f.result()
            if pid and pid not in pitchers_cache:
                pitchers_cache[pid] = fetch_pitcher_stats(pid)
        except:
            pass
print(f"  {len(pitchers_cache)} unique pitchers loaded")

# ─── Team abbreviation map ───
td = cget(f"{MLB_API}/teams?sportIds=1")
ab_map = {t["id"]: t.get("abbreviation","??") for t in td.get("teams",[])} if td else {}

# ─── Features & Predictions ───
print(f"\n=== Predicting on {len(test_games)} games ===")

records = []
for i, g in enumerate(test_games):
    if i % 50 == 0 and i > 0:
        print(f"  {i}/{len(test_games)}")
    pk = g["gamePk"]
    hpid = get_starter(pk, "home")
    apid = get_starter(pk, "away")
    hp_stats = pitchers_cache.get(hpid, {})
    ap_stats = pitchers_cache.get(apid, {})
    pdata = {
        "hp_era": hp_stats.get("era", 4.2), "hp_k9": hp_stats.get("k9", 8.0),
        "hp_bb9": hp_stats.get("bb9", 3.2), "hp_hr9": hp_stats.get("hr9", 1.2),
        "ap_era": ap_stats.get("era", 4.2), "ap_k9": ap_stats.get("k9", 8.0),
        "ap_bb9": ap_stats.get("bb9", 3.2), "ap_hr9": ap_stats.get("hr9", 1.2),
    }
    try:
        feat = build_feature(g, pdata)
    except:
        continue

    x = np.array([[feat[c] for c in cols]])
    hp = m_hw.predict_proba(x)[0, 1]
    er = m_rd.predict(x)[0]
    et = m_tot.predict(x)[0]

    # Monte Carlo
    ns = 2000
    rs = np.random.normal(er, 3.0, ns)
    ts = np.random.normal(et, 3.2, ns)

    ml_fav_prob = max(hp, 1-hp)
    fav_is_home = hp > 0.5
    rl_fav_prob = np.mean(rs >= 1.5)
    rl_dog_prob = np.mean(rs >= -1.5)
    over_prob = np.mean(ts > 8.5)

    actual_home_win = feat["hw"]
    actual_rd = feat["rd"]
    actual_tot = feat["tot"]

    records.append({
        "date": g.get("gameDate","").split("T")[0],
        "game": f"{ab_map.get(g['teams']['away']['team']['id'], g['teams']['away']['team']['name'])} @ {ab_map.get(g['teams']['home']['team']['id'], g['teams']['home']['team']['name'])}",
        "ml_fav_prob": ml_fav_prob,
        "rl_fav_prob": rl_fav_prob,
        "rl_dog_prob": rl_dog_prob,
        "over_prob": over_prob,
        "ml_fav_won": 1 if (fav_is_home and actual_home_win) or (not fav_is_home and not actual_home_win) else 0,
        "rl_fav_covered": 1 if actual_rd >= 1.5 else 0,
        "rl_dog_covered": 1 if actual_rd >= -1.5 else 0,
        "over_hit": 1 if actual_tot > 8.5 else 0,
    })

# ─── Calibration report ───
def bucket_report(data, prob_key, result_key, label):
    if not data: return
    all_probs = [r[prob_key] for r in data]
    p_min, p_max = min(all_probs), max(all_probs)
    # Create dynamic buckets based on data range
    lo = int(p_min * 20) / 20.0
    hi = int(p_max * 20 + 1) / 20.0
    buckets = [(i/20, (i+1)/20) for i in range(int(lo*20), int(hi*20))]
    print(f"\n  {label}  (probs: {p_min:.1%}–{p_max:.1%}, n={len(data)})")
    print(f"  {'Bucket':<14} {'N':>5} {'Actual':>8} {'Pred':>8} {'Diff':>8}")
    print(f"  {'-'*45}")
    total_mae = 0
    total_n = 0
    for lo, hi in buckets:
        group = [r for r in data if lo <= r[prob_key] < hi]
        if not group: continue
        n = len(group)
        actual_rate = sum(r[result_key] for r in group) / n
        pred_rate = sum(r[prob_key] for r in group) / n
        diff = actual_rate - pred_rate
        total_mae += abs(diff) * n
        total_n += n
        bar = "█" * int(abs(diff) * 50) if abs(diff) > 0.01 else ""
        sign = "+" if diff > 0 else ""
        print(f"  {lo*100:>3.0f}-{hi*100:<3.0f}%   {n:>5}  {actual_rate:>7.1%}  {pred_rate:>7.1%}  {sign}{diff:>+7.1%}  {bar}")
    if total_n:
        print(f"  {'-'*45}")
        mae = total_mae / total_n
        print(f"  MAE: {mae:.3f}  ({'✅ bien calibrado' if mae < 0.03 else '⚠️ regular' if mae < 0.06 else '❌ mal calibrado'})")

bucket_report(records, "ml_fav_prob", "ml_fav_won", "Moneyline (favorito)")
bucket_report(records, "rl_fav_prob", "rl_fav_covered", "Run Line -1.5 (favorito)")
bucket_report(records, "rl_dog_prob", "rl_dog_covered", "Run Line +1.5 (dog)")
bucket_report(records, "over_prob", "over_hit", "Over 8.5")

# ─── Overall accuracy ───
print(f"\n=== Overall Accuracy ({len(records)} games) ===")
ml_correct = sum(r["ml_fav_won"] for r in records)
rl_fav_correct = sum(r["rl_fav_covered"] for r in records)
rl_dog_correct = sum(r["rl_dog_covered"] for r in records)
over_correct = sum(r["over_hit"] for r in records)
n = len(records)
print(f"  Moneyline:     {ml_correct}/{n} ({ml_correct/n*100:.1f}%)")
print(f"  RL -1.5:       {rl_fav_correct}/{n} ({rl_fav_correct/n*100:.1f}%)")
print(f"  RL +1.5:       {rl_dog_correct}/{n} ({rl_dog_correct/n*100:.1f}%)")
print(f"  Over 8.5:      {over_correct}/{n} ({over_correct/n*100:.1f}%)")

# ─── Edge-based accuracy ───
print(f"\n=== Edge-based Accuracy (picks with value) ===")
for threshold, label in [(0.02, "edge >2%"), (0.05, "edge >5%"), (0.08, "edge >8%")]:
    group = [r for r in records if r["ml_fav_prob"] > 0.5 + threshold]
    if group:
        correct = sum(r["ml_fav_won"] for r in group)
        n_g = len(group)
        avg_prob = sum(r["ml_fav_prob"] for r in group) / n_g
        print(f"  ML {label}: {correct}/{n_g} ({correct/n_g*100:.1f}%)  avg_prob={avg_prob:.1%}")

# ─── Summary ───
print(f"\n{'='*50}")
if records:
    from bankroll import get_pnl
    pnl = get_pnl()
    print(f"  Bankroll P&L tracker: {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%), ${pnl['profit']:+.0f}")
print(f"  Model trained on {len(all_games)} games, calibrated on {len(records)} validation games")
