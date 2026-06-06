"""
train_v3.py — Multi-season XGBoost training (2023+2024+2025+2026).
Real starting pitcher data from boxscores, per-season pitcher stats.
"""
import requests, numpy as np, math, pickle, json, time, os
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error

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

def parse_ip(ip):
    if not ip: return 0
    if isinstance(ip, str) and "." in ip:
        p = ip.split(".")
        return int(p[0]) + int(p[1]) / 3.0 if len(p) > 1 else float(p[0])
    return float(ip or 0)

PARK = {"Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
        "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
        "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
        "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
        "Oracle Park": 0.98, "Petco Park": 0.97, "PNC Park": 0.97,
        "Tropicana Field": 0.96, "Target Field": 0.96, "Oakland Coliseum": 0.97}
DOME_VENUES = {"Tropicana Field", "Rogers Centre", "Chase Field", "Globe Life Field",
               "American Family Field", "Minute Maid Park", "loanDepot park", "Marlins Park"}

# ─── Step 1: Fetch games for all seasons ───
print("=== Step 1: Fetching games ===")
SEASONS = {
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-27", "2026-10-01"),
}

all_games = []
for year, (start, end) in SEASONS.items():
    raw = cget(f"{MLB_API}/schedule?sportId=1&startDate={start}&endDate={end}")
    if not raw:
        print(f"  Failed to fetch {year}")
        continue
    games = []
    for d in raw.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                g["_season"] = year
                games.append(g)
    print(f"  {year}: {len(games)} completed games")
    all_games.extend(games)

all_games.sort(key=lambda x: x.get("gameDate", ""))
print(f"  Total: {len(all_games)} games")

# ─── Step 2: Fetch starting pitchers from boxscores (parallel, cached) ───
PITCHER_CACHE_FILE = BASE + "pitcher_cache.json"
game_pitchers = {}
pitcher_stats_db = {}  # (pid, season) -> stats
pitcher_rec_form = {}  # (pid, season) -> rec form stats

if os.path.exists(PITCHER_CACHE_FILE):
    with open(PITCHER_CACHE_FILE) as f:
        cached = json.load(f)
    game_pitchers = {int(k): v for k, v in cached.get("game_pitchers", {}).items()}
    pitcher_stats_db = {tuple(k.split("_")): v for k, v in cached.get("pitcher_stats", {}).items()}
    pitcher_rec_form = {tuple(k.split("_")): v for k, v in cached.get("rec_form", {}).items()}
    print(f"\n=== Loaded cached data: {len(game_pitchers)} games, {len(pitcher_stats_db)} pitcher-stats, {len(pitcher_rec_form)} rec-form ===")

print("\n=== Step 2: Fetching starting pitchers ===")

def get_boxscore_starter(game_pk, team_side):
    try:
        d = cget(f"{MLB_API}/game/{game_pk}/boxscore")
        if d:
            pitchers = d.get("teams", {}).get(team_side, {}).get("pitchers", [])
            if pitchers: return pitchers[0]
    except: pass
    return None

def fetch_season_pitcher_stats(pid, season):
    key = (str(pid), str(season))
    if key in pitcher_stats_db:
        return
    d = cget(f"{MLB_API}/people/{pid}/stats?stats=season&season={season}&group=pitching")
    if d:
        sl = d.get("stats", [])
        if sl and sl[0].get("splits"):
            s = sl[0]["splits"][0].get("stat", {})
            ipv = parse_ip(s.get("inningsPitched", "0"))
            pitcher_stats_db[key] = {
                "era": sf(s.get("era")), "ip": ipv,
                "k9": sf(s.get("strikeoutsPer9Inn")),
                "bb9": sf(s.get("walksPer9Inn")),
                "hr9": sf(s.get("homeRunsPer9")),
                "hr": sf(s.get("homeRuns")),
                "bb": sf(s.get("baseOnBalls")),
                "so": sf(s.get("strikeOuts")),
                "h": sf(s.get("hits")),
                "ab": sf(s.get("atBats")),
                "sf": sf(s.get("sacFlies")),
                "hbp": sf(s.get("hitByPitch")),
                "go": sf(s.get("groundOuts")),
                "ao": sf(s.get("airOuts")),
            }

def fetch_pitcher_rec_form(pid, season):
    key = (str(pid), str(season))
    if key in pitcher_rec_form:
        return
    try:
        d = cget(f"{MLB_API}/people/{pid}/stats?stats=gameLog&season={season}&group=pitching")
        if d:
            splits = d.get("stats", [{}])[0].get("splits", [])
            starts = [s for s in splits if sf(s.get("stat", {}).get("inningsPitched", "0")) > 0]
            recent = starts[-5:]
            if recent:
                eras, k9s, bbs, hrs = [], [], [], []
                for s in recent:
                    st = s.get("stat", {})
                    ip_s = st.get("inningsPitched", "0")
                    ipv = parse_ip(ip_s)
                    er = sf(st.get("earnedRuns"))
                    eras.append(9 * er / ipv if ipv > 0 else 4.5)
                    k9s.append(sf(st.get("strikeoutsPer9Inn")))
                    bbs.append(sf(st.get("walksPer9Inn")))
                    hrs.append(sf(st.get("homeRunsPer9")))
                pitcher_rec_form[key] = {"rec_era": np.mean(eras), "rec_k9": np.mean(k9s), "rec_bb9": np.mean(bbs), "rec_hr9": np.mean(hrs)}
    except:
        pass

# Build pid_seasons from game_pitchers (cached or just fetched)
pid_seasons = set()
for g in all_games:
    gp = game_pitchers.get(g["gamePk"], {})
    season = g.get("_season", 2026)
    if gp.get("h"):
        pid_seasons.add((str(gp["h"]), str(season)))
    if gp.get("a"):
        pid_seasons.add((str(gp["a"]), str(season)))

# Fetch boxscores for games not yet fetched
game_pks_to_fetch = [g["gamePk"] for g in all_games if g["gamePk"] not in game_pitchers]
if game_pks_to_fetch:
    print(f"  Fetching boxscores for {len(game_pks_to_fetch)} games...")
    all_pids = set()
    def fetch_game_starters(g):
        pk = g["gamePk"]
        hpid = get_boxscore_starter(pk, "home")
        apid = get_boxscore_starter(pk, "away")
        return pk, hpid, apid

    batch_size = 50
    total = len(game_pks_to_fetch)
    with ThreadPoolExecutor(max_workers=25) as ex:
        games_subset = [g for g in all_games if g["gamePk"] in game_pks_to_fetch]
        futures = [ex.submit(fetch_game_starters, g) for g in games_subset]
        done = 0
        for f in as_completed(futures):
            pk, hpid, apid = f.result()
            game_pitchers[pk] = {"h": hpid, "a": apid}
            if hpid: all_pids.add(hpid)
            if apid: all_pids.add(apid)
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{total}")

    covered = sum(1 for v in game_pitchers.values() if v["h"] and v["a"])
    print(f"  Games with both starters: {covered}/{len(all_games)}")

    # ─── Step 3: Fetch pitcher stats per season ───
    print(f"\n=== Step 3: Fetching pitcher stats ({len(all_pids)} unique) ===")
    
    to_fetch = [ps for ps in pid_seasons if tuple(ps) not in pitcher_stats_db]
    print(f"  {len(to_fetch)} pitcher-season combos to fetch ({len(pid_seasons) - len(to_fetch)} cached)")
    
    batch = 1
    for pid, season in to_fetch:
        fetch_season_pitcher_stats(int(pid), int(season))
        if batch % 100 == 0:
            print(f"    {batch}/{len(to_fetch)}")
        batch += 1
    print(f"  Total pitcher-season stats: {len(pitcher_stats_db)}")

# ─── Step 3b: Fetch pitcher recent form (rec_* features) ───
print(f"\n=== Step 3b: Fetching pitcher game logs (rec_*) ===")
rec_to_fetch = [ps for ps in pid_seasons if tuple(ps) not in pitcher_rec_form]
print(f"  {len(rec_to_fetch)} pitcher-seasons to fetch ({len(pid_seasons) - len(rec_to_fetch)} cached)")
batch = 1
for pid, season in rec_to_fetch:
    fetch_pitcher_rec_form(int(pid), int(season))
    if batch % 100 == 0:
        print(f"    {batch}/{len(rec_to_fetch)}")
    batch += 1
print(f"  Total pitcher rec-form: {len(pitcher_rec_form)}")

# Save cache
serializable = {
    "game_pitchers": {str(k): v for k, v in game_pitchers.items()},
    "pitcher_stats": {"_".join(k): v for k, v in pitcher_stats_db.items()},
    "rec_form": {"_".join(k): v for k, v in pitcher_rec_form.items()},
}
with open(PITCHER_CACHE_FILE, "w") as f:
    json.dump(serializable, f)
print(f"  Cache saved to {PITCHER_CACHE_FILE}")

def get_pitcher_feats(pid, season):
    key = (str(pid), str(season))
    ps = pitcher_stats_db.get(key)
    if ps and ps.get("ip", 0) >= 10:
        fip = ((13 * ps.get("hr", 0)) + (3 * (ps.get("bb", 0) + ps.get("hbp", 0))) - (2 * ps.get("so", 0))) / ps["ip"] + 3.10 if ps["ip"] > 0 else 4.5
        babip = (ps.get("h", 0) - ps.get("hr", 0)) / (ps.get("ab", 0) - ps.get("so", 0) - ps.get("hr", 0) + ps.get("sf", 0)) if (ps.get("ab", 0) - ps.get("so", 0) - ps.get("hr", 0) + ps.get("sf", 0)) > 0 else 0.300
        kbb = ps.get("so", 0) / ps.get("bb", 0) if ps.get("bb", 0) > 0 else ps.get("so", 0)
        gb = ps.get("go", 0) / (ps.get("go", 0) + ps.get("ao", 0)) if (ps.get("go", 0) + ps.get("ao", 0)) > 0 else 0.44
        return (ps["era"], ps["k9"], ps["bb9"], ps["hr9"], 1, fip, babip, kbb, gb)
    return (4.5, 8.0, 3.0, 1.2, 0, 4.5, 0.300, 3.0, 0.44)

# ─── Step 4: Build features ───
print("\n=== Step 4: Building features ===")

# Team lookup for recent form: group by team, sorted chronologically
team_games = defaultdict(list)
for g in all_games:
    team_games[g["teams"]["home"]["team"]["id"]].append(g)
    team_games[g["teams"]["away"]["team"]["id"]].append(g)

# Elo ratings
elos = defaultdict(lambda: 1500)
features = []
N = len(all_games)
batch = max(N // 20, 1)

for i, g in enumerate(all_games):
    if i % batch == 0:
        print(f"    {i}/{N} ({i*100//N}%)")
    try:
        t = g["teams"]
        htid = t["home"]["team"]["id"]; atid = t["away"]["team"]["id"]
        gd = g.get("gameDate", "").split("T")[0]
        hs = int(t.get("home", {}).get("score", 0))
        as_ = int(t.get("away", {}).get("score", 0))
        season = g.get("_season", 2026)
    except: continue
    if hs == 0 and as_ == 0: continue

    h_elo = elos[htid]; a_elo = elos[atid]

    def recent_form(tid, dt_str):
        prior = [x for x in team_games[tid] if x.get("gameDate", "").split("T")[0] < dt_str]
        r10 = sorted(prior, key=lambda x: x.get("gameDate", ""), reverse=True)[:10]
        if not r10: return (0.5, 4.5, 4.5)
        rs, ra, w = [], [], 0
        for x in r10:
            xh = x["teams"]["home"]["team"]["id"] == tid
            ms = sf(x["teams"]["home"]["score"] if xh else x["teams"]["away"]["score"])
            os_ = sf(x["teams"]["away"]["score"] if xh else x["teams"]["home"]["score"])
            rs.append(ms); ra.append(os_)
            if ms > os_: w += 1
        n = len(rs) or 1
        return (w / n, np.mean(rs), np.mean(ra))

    hf = recent_form(htid, gd); af = recent_form(atid, gd)

    def rest(tid, dt_str):
        prior = [x for x in team_games[tid] if x.get("gameDate", "").split("T")[0] < dt_str]
        if prior:
            try:
                ld = datetime.strptime(prior[-1].get("gameDate", "").split("T")[0], "%Y-%m-%d")
                return min((datetime.strptime(dt_str, "%Y-%m-%d") - ld).days, 5)
            except: pass
        return 3

    hr = rest(htid, gd); ar = rest(atid, gd)
    vname = g.get("venue", {}).get("name", "")
    pf = PARK.get(vname, 1.0)

    # Fetch team stats for the correct season
    def get_team_stats(tid, ssn):
        d = cget(f"{MLB_API}/teams/{tid}/stats?season={ssn}&group=hitting,pitching&stats=season")
        r = {"ops": 0.700, "whip": 1.35, "era": 4.5}
        if d:
            for sg in d.get("stats", []):
                g_ = sg.get("group", {}).get("displayName", "").lower()
                sp = sg.get("splits", [])
                if sp:
                    s = sp[0].get("stat", {})
                    if g_ == "hitting": r["ops"] = sf(s.get("ops"), 0.700)
                    elif g_ == "pitching":
                        ipv = parse_ip(s.get("inningsPitched", "0"))
                        er = sf(s.get("earnedRuns"))
                        wh = (sf(s.get("walks", 0)) + sf(s.get("hits", 0))) / ipv if ipv > 0 else 0
                        r["whip"] = min(wh, 3.0)
                        r["era"] = 9 * er / ipv if ipv > 0 else 4.5
        return r

    ts_h = get_team_stats(htid, season)
    ts_a = get_team_stats(atid, season)

    # REAL pitcher data from boxscore + correct season
    gp = game_pitchers.get(g["gamePk"], {})
    hpe = get_pitcher_feats(gp.get("h"), season)
    ape = get_pitcher_feats(gp.get("a"), season)
    hp_rec = pitcher_rec_form.get((str(gp.get("h")), str(season)), {})
    ap_rec = pitcher_rec_form.get((str(gp.get("a")), str(season)), {})

    features.append({
        "h_elo": h_elo, "a_elo": a_elo,
        "h_wp": hf[0], "a_wp": af[0],
        "h_rs": hf[1], "a_rs": af[1],
        "h_ra": hf[2], "a_ra": af[2],
        "h_rest": hr, "a_rest": ar,
        "h_ops": ts_h["ops"], "a_ops": ts_a["ops"],
        "h_whip": ts_h["whip"], "a_whip": ts_a["whip"],
        "h_era": ts_h["era"], "a_era": ts_a["era"],
        "park": pf,
        "hp_era": hpe[0], "hp_k9": hpe[1], "hp_bb9": hpe[2], "hp_hr9": hpe[3], "hp_v": hpe[4],
        "hp_fip": hpe[5], "hp_babip": hpe[6], "hp_kbb": hpe[7], "hp_gb_rate": hpe[8],
        "ap_era": ape[0], "ap_k9": ape[1], "ap_bb9": ape[2], "ap_hr9": ape[3], "ap_v": ape[4],
        "ap_fip": ape[5], "ap_babip": ape[6], "ap_kbb": ape[7], "ap_gb_rate": ape[8],
        "hp_rec_era": hp_rec.get("rec_era", hpe[0]), "hp_rec_k9": hp_rec.get("rec_k9", hpe[1]),
        "hp_rec_bb9": hp_rec.get("rec_bb9", hpe[2]), "hp_rec_hr9": hp_rec.get("rec_hr9", hpe[3]),
        "ap_rec_era": ap_rec.get("rec_era", ape[0]), "ap_rec_k9": ap_rec.get("rec_k9", ape[1]),
        "ap_rec_bb9": ap_rec.get("rec_bb9", ape[2]), "ap_rec_hr9": ap_rec.get("rec_hr9", ape[3]),
        "temp_f": 72.0, "wind_mph": 0.0, "humidity": 50,
        "is_dome": 1 if vname in DOME_VENUES else 0,
        "hw": 1 if hs > as_ else 0,
        "rd": hs - as_,
        "tot": hs + as_,
    })

    # Update Elo (sequential for all seasons)
    eh = 1 / (1 + 10 ** ((a_elo - h_elo - 50) / 400))
    ah = 1 if hs > as_ else (0 if hs < as_ else 0.5)
    mg = min(math.log(abs(hs - as_) + 1) / 2.2, 1.5)
    elos[htid] += 32 * mg * (ah - eh)
    elos[atid] += 32 * mg * ((1 - ah) - (1 - eh))

print(f"  Done! {len(features)} rows")

# ─── Step 5: Train XGBoost models ───
print("\n=== Step 5: Training XGBoost ===")
cols = ["h_elo", "a_elo", "h_wp", "a_wp", "h_rs", "a_rs", "h_ra", "a_ra",
        "h_rest", "a_rest",
        "h_ops", "a_ops", "h_whip", "a_whip", "h_era", "a_era",
        "park",
        "hp_era", "hp_k9", "hp_bb9", "hp_hr9", "hp_v",
        "hp_fip", "hp_babip", "hp_kbb", "hp_gb_rate",
        "ap_era", "ap_k9", "ap_bb9", "ap_hr9", "ap_v",
        "ap_fip", "ap_babip", "ap_kbb", "ap_gb_rate",
        "hp_rec_era", "hp_rec_k9", "hp_rec_bb9", "hp_rec_hr9",
        "ap_rec_era", "ap_rec_k9", "ap_rec_bb9", "ap_rec_hr9",
        "temp_f", "wind_mph", "humidity", "is_dome"]

X = np.array([[f[c] for c in cols] for f in features])
y_hw = np.array([f["hw"] for f in features])
y_rd = np.array([f["rd"] for f in features])
y_tot = np.array([f["tot"] for f in features])

# 80/20 time-based split (last 20% by date order)
sp = int(len(X) * 0.8)
Xt, Xv = X[:sp], X[sp:]
yht, yhv = y_hw[:sp], y_hw[sp:]
yrt, yrv = y_rd[:sp], y_rd[sp:]
ytt, ytv = y_tot[:sp], y_tot[sp:]
print(f"  Train: {len(Xt)}, Val: {len(Xv)}")

# XGBoost HW classifier
m_hw = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1.0,  # balanced enough with 4 seasons
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric='logloss',
)
m_hw.fit(Xt, yht, eval_set=[(Xv, yhv)], verbose=False)
p_hw = m_hw.predict(Xv)
p_hw_proba = m_hw.predict_proba(Xv)[:, 1]
print(f"  HW acc:  {accuracy_score(yhv, p_hw):.3f}")

# XGBoost RD regressor
m_rd = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric='mae',
)
m_rd.fit(Xt, yrt, eval_set=[(Xv, yrv)], verbose=False)
p_rd = m_rd.predict(Xv)
print(f"  RD MAE:  {mean_absolute_error(yrv, p_rd):.3f}")

# XGBoost Total regressor
m_tot = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric='mae',
)
m_tot.fit(Xt, ytt, eval_set=[(Xv, ytv)], verbose=False)
p_tot = m_tot.predict(Xv)
print(f"  Tot MAE: {mean_absolute_error(ytv, p_tot):.3f}")

# ─── Ensemble: train 3 seeds ───
SEEDS = [42, 123, 456]
ensemble_models = {}
for seed in SEEDS:
    print(f"\n=== Training seed={seed} ===")
    s_hw = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0, eval_metric='logloss')
    s_hw.fit(Xt, yht, eval_set=[(Xv, yhv)], verbose=False)
    s_rd = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mae')
    s_rd.fit(Xt, yrt, eval_set=[(Xv, yrv)], verbose=False)
    s_tot = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mae')
    s_tot.fit(Xt, ytt, eval_set=[(Xv, ytv)], verbose=False)
    print(f"  HW acc: {accuracy_score(yhv, s_hw.predict(Xv)):.3f} | RD MAE: {mean_absolute_error(yrv, s_rd.predict(Xv)):.3f} | Tot MAE: {mean_absolute_error(ytv, s_tot.predict(Xv)):.3f}")
    with open(BASE + f"xgb_hw_s{seed}.pkl", "wb") as f: pickle.dump(s_hw, f)
    with open(BASE + f"xgb_rd_s{seed}.pkl", "wb") as f: pickle.dump(s_rd, f)
    with open(BASE + f"xgb_tot_s{seed}.pkl", "wb") as f: pickle.dump(s_tot, f)

with open(BASE + "xgb_cols.pkl", "wb") as f: pickle.dump(cols, f)
print(f"\n✅ Ensemble ({len(SEEDS)} seeds) saved to {BASE}")
