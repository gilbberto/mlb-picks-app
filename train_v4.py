"""
train_v4.py — Dedicated O/U model with bullpen stats, park-weather interactions.
Also trains improved HW and RD models with bullpen features.
2020-2026 data, 3-seed XGBoost ensemble.
"""
import requests, numpy as np, math, pickle, json, time, os
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

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
               "American Family Field", "Minute Maid Park", "loanDepot park", "Marlins Park",
               "loanDepot Park", "Sutter Health Park"}

# Step 1: Fetch games for all seasons
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

# Step 2: Fetch starting pitchers from boxscores (parallel, cached)
PITCHER_CACHE_FILE = BASE + "pitcher_cache.json"
game_pitchers = {}
pitcher_stats_db = {}
pitcher_rec_form = {}

if os.path.exists(PITCHER_CACHE_FILE):
    with open(PITCHER_CACHE_FILE) as f:
        cached = json.load(f)
    game_pitchers = {int(k): v for k, v in cached.get("game_pitchers", {}).items()}
    pitcher_stats_db = {tuple(k.split("_")): v for k, v in cached.get("pitcher_stats", {}).items()}
    pitcher_rec_form = {tuple(k.split("_")): v for k, v in cached.get("rec_form", {}).items()}
    print(f"\n=== Loaded cached data: {len(game_pitchers)} games, {len(pitcher_stats_db)} pitcher-stats ===")

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
                "er": sf(s.get("earnedRuns")),
                "r": sf(s.get("runs")),
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

pid_seasons = set()
for g in all_games:
    gp = game_pitchers.get(g["gamePk"], {})
    season = g.get("_season", 2026)
    if gp.get("h"):
        pid_seasons.add((str(gp["h"]), str(season)))
    if gp.get("a"):
        pid_seasons.add((str(gp["a"]), str(season)))

game_pks_to_fetch = [g["gamePk"] for g in all_games if g["gamePk"] not in game_pitchers]
if game_pks_to_fetch:
    print(f"  Fetching boxscores for {len(game_pks_to_fetch)} games...")
    def fetch_game_starters(g):
        pk = g["gamePk"]
        hpid = get_boxscore_starter(pk, "home")
        apid = get_boxscore_starter(pk, "away")
        return pk, hpid, apid

    with ThreadPoolExecutor(max_workers=25) as ex:
        games_subset = [g for g in all_games if g["gamePk"] in game_pks_to_fetch]
        futures = [ex.submit(fetch_game_starters, g) for g in games_subset]
        done = 0
        for f in as_completed(futures):
            pk, hpid, apid = f.result()
            game_pitchers[pk] = {"h": hpid, "a": apid}
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(game_pks_to_fetch)}")

    covered = sum(1 for v in game_pitchers.values() if v["h"] and v["a"])
    print(f"  Games with both starters: {covered}/{len(all_games)}")

    print(f"\n=== Step 3: Fetching pitcher stats ===")
    to_fetch = [ps for ps in pid_seasons if tuple(ps) not in pitcher_stats_db]
    print(f"  {len(to_fetch)} pitcher-season combos to fetch")
    batch = 1
    for pid, season in to_fetch:
        fetch_season_pitcher_stats(int(pid), int(season))
        if batch % 100 == 0:
            print(f"    {batch}/{len(to_fetch)}")
        batch += 1
    print(f"  Total pitcher-season stats: {len(pitcher_stats_db)}")

# Step 3b: Fetch pitcher recent form
print(f"\n=== Step 3b: Fetching pitcher game logs (rec_*) ===")
rec_to_fetch = [ps for ps in pid_seasons if tuple(ps) not in pitcher_rec_form]
print(f"  {len(rec_to_fetch)} pitcher-seasons to fetch")
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
        return {"era": ps["era"], "k9": ps["k9"], "bb9": ps["bb9"], "hr9": ps["hr9"], "v": 1,
                "fip": fip, "babip": babip, "kbb": kbb, "gb_rate": gb,
                "ip": ps["ip"], "er": ps.get("er", 0), "r": ps.get("r", 0),
                "bb": ps.get("bb", 0), "so": ps.get("so", 0), "h": ps.get("h", 0)}
    return {"era": 4.5, "k9": 8.0, "bb9": 3.0, "hr9": 1.2, "v": 0, "fip": 4.5,
            "babip": 0.300, "kbb": 3.0, "gb_rate": 0.44,
            "ip": 0, "er": 0, "r": 0, "bb": 0, "so": 0, "h": 0}

# Step 4: Build features
print("\n=== Step 4: Building features ===")

team_games = defaultdict(list)
for g in all_games:
    team_games[g["teams"]["home"]["team"]["id"]].append(g)
    team_games[g["teams"]["away"]["team"]["id"]].append(g)

elos = defaultdict(lambda: 1500)

# Cache team-season stats
team_season_cache = {}
def get_team_stats_full(tid, ssn):
    key = (str(tid), str(ssn))
    if key in team_season_cache:
        return team_season_cache[key]
    d = cget(f"{MLB_API}/teams/{tid}/stats?season={ssn}&group=hitting,pitching&stats=season")
    r = {"ops": 0.700, "whip": 1.35, "era": 4.5, "er": 0, "ip": 0, "bb": 0, "so": 0, "h": 0, "r": 0, "hr": 0}
    if d:
        for sg in d.get("stats", []):
            g_ = sg.get("group", {}).get("displayName", "").lower()
            sp = sg.get("splits", [])
            if sp:
                s = sp[0].get("stat", {})
                if g_ == "hitting":
                    r["ops"] = sf(s.get("ops"), 0.700)
                    r["hr"] = sf(s.get("homeRuns"))
                elif g_ == "pitching":
                    ipv = parse_ip(s.get("inningsPitched", "0"))
                    er = sf(s.get("earnedRuns"))
                    wh = (sf(s.get("walks", 0)) + sf(s.get("hits", 0))) / ipv if ipv > 0 else 0
                    r["whip"] = min(wh, 3.0)
                    r["era"] = 9 * er / ipv if ipv > 0 else 4.5
                    r["er"] = er
                    r["ip"] = ipv
                    r["bb"] = sf(s.get("baseOnBalls"))
                    r["so"] = sf(s.get("strikeOuts"))
                    r["h"] = sf(s.get("hits"))
                    r["r"] = sf(s.get("runs"))
    team_season_cache[key] = r
    return r

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
    is_dome = 1 if vname in DOME_VENUES else 0

    ts_h = get_team_stats_full(htid, season)
    ts_a = get_team_stats_full(atid, season)

    gp = game_pitchers.get(g["gamePk"], {})
    hpe = get_pitcher_feats(gp.get("h"), season)
    ape = get_pitcher_feats(gp.get("a"), season)
    hp_rec = pitcher_rec_form.get((str(gp.get("h")), str(season)), {})
    ap_rec = pitcher_rec_form.get((str(gp.get("a")), str(season)), {})

    # Bullpen stats: team totals minus starter contribution
    # Home bullpen
    h_bp_ip = max(ts_h["ip"] - hpe["ip"], 1.0)
    h_bp_er = max(ts_h["er"] - hpe["er"], 0)
    h_bp_era = 9 * h_bp_er / h_bp_ip if h_bp_ip > 0 else 4.5
    h_bp_h = max(ts_h["h"] - hpe["h"], 0)
    h_bp_bb = max(ts_h["bb"] - hpe["bb"], 0)
    h_bp_so = max(ts_h["so"] - hpe["so"], 0)
    h_bp_whip = (h_bp_bb + h_bp_h) / h_bp_ip if h_bp_ip > 0 else 1.35
    h_bp_k9 = 9 * h_bp_so / h_bp_ip if h_bp_ip > 0 else 8.0
    h_bp_bb9 = 9 * h_bp_bb / h_bp_ip if h_bp_ip > 0 else 3.0

    # Away bullpen
    a_bp_ip = max(ts_a["ip"] - ape["ip"], 1.0)
    a_bp_er = max(ts_a["er"] - ape["er"], 0)
    a_bp_era = 9 * a_bp_er / a_bp_ip if a_bp_ip > 0 else 4.5
    a_bp_h = max(ts_a["h"] - ape["h"], 0)
    a_bp_bb = max(ts_a["bb"] - ape["bb"], 0)
    a_bp_so = max(ts_a["so"] - ape["so"], 0)
    a_bp_whip = (a_bp_bb + a_bp_h) / a_bp_ip if a_bp_ip > 0 else 1.35
    a_bp_k9 = 9 * a_bp_so / a_bp_ip if a_bp_ip > 0 else 8.0
    a_bp_bb9 = 9 * a_bp_bb / a_bp_ip if a_bp_ip > 0 else 3.0

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
        "hp_era": hpe["era"], "hp_k9": hpe["k9"], "hp_bb9": hpe["bb9"], "hp_hr9": hpe["hr9"], "hp_v": hpe["v"],
        "hp_fip": hpe["fip"], "hp_babip": hpe["babip"], "hp_kbb": hpe["kbb"], "hp_gb_rate": hpe["gb_rate"],
        "ap_era": ape["era"], "ap_k9": ape["k9"], "ap_bb9": ape["bb9"], "ap_hr9": ape["hr9"], "ap_v": ape["v"],
        "ap_fip": ape["fip"], "ap_babip": ape["babip"], "ap_kbb": ape["kbb"], "ap_gb_rate": ape["gb_rate"],
        "hp_rec_era": hp_rec.get("rec_era", hpe["era"]), "hp_rec_k9": hp_rec.get("rec_k9", hpe["k9"]),
        "hp_rec_bb9": hp_rec.get("rec_bb9", hpe["bb9"]), "hp_rec_hr9": hp_rec.get("rec_hr9", hpe["hr9"]),
        "ap_rec_era": ap_rec.get("rec_era", ape["era"]), "ap_rec_k9": ap_rec.get("rec_k9", ape["k9"]),
        "ap_rec_bb9": ap_rec.get("rec_bb9", ape["bb9"]), "ap_rec_hr9": ap_rec.get("rec_hr9", ape["hr9"]),
        "h_bp_era": h_bp_era, "h_bp_whip": h_bp_whip, "h_bp_k9": h_bp_k9, "h_bp_bb9": h_bp_bb9,
        "a_bp_era": a_bp_era, "a_bp_whip": a_bp_whip, "a_bp_k9": a_bp_k9, "a_bp_bb9": a_bp_bb9,
        "h_hr": ts_h["hr"], "a_hr": ts_a["hr"],
        "temp_f": 72.0, "wind_mph": 0.0, "humidity": 50,
        "is_dome": is_dome,
        "hw": 1 if hs > as_ else 0,
        "rd": hs - as_,
        "tot": hs + as_,
    })

    # Update Elo
    eh = 1 / (1 + 10 ** ((a_elo - h_elo - 50) / 400))
    ah = 1 if hs > as_ else (0 if hs < as_ else 0.5)
    mg = min(math.log(abs(hs - as_) + 1) / 2.2, 1.5)
    elos[htid] += 32 * mg * (ah - eh)
    elos[atid] += 32 * mg * ((1 - ah) - (1 - eh))

print(f"  Done! {len(features)} rows")

# Step 5: Train XGBoost models
print("\n=== Step 5: Training XGBoost ===")

# Base features for all models
cols_base = ["h_elo", "a_elo", "h_wp", "a_wp", "h_rs", "a_rs", "h_ra", "a_ra",
             "h_rest", "a_rest",
             "h_ops", "a_ops", "h_whip", "a_whip", "h_era", "a_era",
             "park",
             "hp_era", "hp_k9", "hp_bb9", "hp_hr9", "hp_v",
             "hp_fip", "hp_babip", "hp_kbb", "hp_gb_rate",
             "ap_era", "ap_k9", "ap_bb9", "ap_hr9", "ap_v",
             "ap_fip", "ap_babip", "ap_kbb", "ap_gb_rate",
             "hp_rec_era", "hp_rec_k9", "hp_rec_bb9", "hp_rec_hr9",
             "ap_rec_era", "ap_rec_k9", "ap_rec_bb9", "ap_rec_hr9",
             "h_bp_era", "h_bp_whip", "h_bp_k9", "h_bp_bb9",
             "a_bp_era", "a_bp_whip", "a_bp_k9", "a_bp_bb9",
             "temp_f", "wind_mph", "humidity", "is_dome"]

# O/U-specific extra features
cols_ou = cols_base + ["h_hr", "a_hr"]  # HR + recent runs for totals

X_base = np.array([[f[c] for c in cols_base] for f in features])
X_ou = np.array([[f[c] for c in cols_ou] for f in features])
y_hw = np.array([f["hw"] for f in features])
y_rd = np.array([f["rd"] for f in features])
y_tot = np.array([f["tot"] for f in features])

# Time-based split
sp = int(len(X_base) * 0.8)
Xt, Xv = X_base[:sp], X_base[sp:]
Xot, Xov = X_ou[:sp], X_ou[sp:]
yht, yhv = y_hw[:sp], y_hw[sp:]
yrt, yrv = y_rd[:sp], y_rd[sp:]
ytt, ytv = y_tot[:sp], y_tot[sp:]
print(f"  Train: {len(Xt)}, Val: {len(Xv)}")

# Train HW (same as before but with bullpen features)
print("\n--- Moneyline (HW) ---")
m_hw = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss')
m_hw.fit(Xt, yht, eval_set=[(Xv, yhv)], verbose=False)
p_hw = m_hw.predict(Xv)
print(f"  HW acc: {accuracy_score(yhv, p_hw):.3f}")

# Train RD regressor
print("\n--- Run Differential (RD) ---")
m_rd = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='mae')
m_rd.fit(Xt, yrt, eval_set=[(Xv, yrv)], verbose=False)
p_rd = m_rd.predict(Xv)
rd_mae = mean_absolute_error(yrv, p_rd)
print(f"  RD MAE: {rd_mae:.3f}")

# Train dedicated O/U model (Total runs regressor with OU-specific features + tuning)
print("\n--- Total Runs (O/U) — Dedicated Model ---")
m_tot = xgb.XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='mae')
m_tot.fit(Xot, ytt, eval_set=[(Xov, ytv)], verbose=False)
p_tot = m_tot.predict(Xov)
tot_mae = mean_absolute_error(ytv, p_tot)
tot_r2 = r2_score(ytv, p_tot)
print(f"  Tot MAE: {tot_mae:.3f}, R²: {tot_r2:.3f}")

# O/U classification accuracy: predict over/under median (~8.5)
median_total = np.median(y_tot)
ou_pred = (p_tot > median_total).astype(int)
ou_true = (ytv > median_total).astype(int)
ou_acc = accuracy_score(ou_true, ou_pred)
print(f"  O/U > {median_total:.1f} acc: {ou_acc:.3f}")

# Ensemble: 3 seeds
print("\n=== Ensemble: 3 seeds ===")
SEEDS = [42, 123, 456]
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    s_hw = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0, eval_metric='logloss')
    s_hw.fit(Xt, yht, eval_set=[(Xv, yhv)], verbose=False)
    s_rd = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mae')
    s_rd.fit(Xt, yrt, eval_set=[(Xv, yrv)], verbose=False)
    s_tot = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mae')
    s_tot.fit(Xot, ytt, eval_set=[(Xov, ytv)], verbose=False)
    sp_hw = s_hw.predict(Xv); sp_tot = s_tot.predict(Xov)
    print(f"  HW acc: {accuracy_score(yhv, sp_hw):.3f} | RD MAE: {mean_absolute_error(yrv, s_rd.predict(Xv)):.3f} | Tot MAE: {mean_absolute_error(ytv, sp_tot):.3f} | O/U acc: {accuracy_score(ou_true, (sp_tot > median_total).astype(int)):.3f}")
    with open(BASE + f"xgb_hw_s{seed}.pkl", "wb") as f: pickle.dump(s_hw, f)
    with open(BASE + f"xgb_rd_s{seed}.pkl", "wb") as f: pickle.dump(s_rd, f)
    with open(BASE + f"xgb_tot_s{seed}.pkl", "wb") as f: pickle.dump(s_tot, f)

# Save column names for both feature sets
with open(BASE + "xgb_cols.pkl", "wb") as f: pickle.dump(cols_base, f)
with open(BASE + "xgb_ou_cols.pkl", "wb") as f: pickle.dump(cols_ou, f)
print(f"\n  Models saved. Base cols: {len(cols_base)}, OU cols: {len(cols_ou)}")

# ─── Platt Scaling Calibration ───
print("\n=== Calibration (Platt scaling) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Use seed-42 model for calibration (ensembles average the probas anyway)
calib_hw = m_hw
calib_tot = m_tot

# HW calibration: Platt scale on validation set
hw_probas = calib_hw.predict_proba(Xv)[:, 1]
platt_hw = LogisticRegression(C=1.0, solver='lbfgs')
platt_hw.fit(hw_probas.reshape(-1, 1), yhv)
cal_hw = platt_hw.predict_proba(hw_probas.reshape(-1, 1))[:, 1]
cal_hw_acc = accuracy_score(yhv, (cal_hw > 0.5).astype(int))
print(f"  HW calibration: {len(yhv)} pts, acc={cal_hw_acc:.3f}")

# Show calibration curves (before/after)
for name, probs in [("raw", hw_probas), ("calibrated", cal_hw)]:
    frac_pos, mean_pred = calibration_curve(yhv, probs, n_bins=10, strategy='uniform')
    print(f"  {name:12s}: bins={[f'{fp:.2f}' for fp in frac_pos[:5]]}...")

# O/U calibration: predict total → over_prob → Platt scale
tot_preds = calib_tot.predict(Xov)
median_total = np.median(y_tot)
# For O/U: use the market line median as threshold to get binary outcomes
ou_binary = (ytv > median_total).astype(int)
# Raw O/U probability via normal CDF (std=3.2)
from scipy.stats import norm
ou_raw_probs = norm.cdf(tot_preds - median_total, 0, 3.2)
platt_ou = LogisticRegression(C=1.0, solver='lbfgs')
platt_ou.fit(ou_raw_probs.reshape(-1, 1), ou_binary)
cal_ou = platt_ou.predict_proba(ou_raw_probs.reshape(-1, 1))[:, 1]
cal_ou_acc = accuracy_score(ou_binary, (cal_ou > 0.5).astype(int))
print(f"  O/U calibration: {len(ytv)} pts, acc={cal_ou_acc:.3f}")

# Save calibrators
with open(BASE + "calib_hw.pkl", "wb") as f: pickle.dump(platt_hw, f)
with open(BASE + "calib_ou.pkl", "wb") as f: pickle.dump(platt_ou, f)
print(f"  Calibrators saved: calib_hw.pkl, calib_ou.pkl")
print(f"  Previous Tot MAE ~4.0 → New Tot MAE: {tot_mae:.3f}")
