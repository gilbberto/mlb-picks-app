"""
train_models.py — Train RandomForest models for MLB predictions using pre-game data only.

Usage: python3 train_models.py
Output: rf_home_win.pkl, rf_run_diff.pkl, rf_total_runs.pkl, rf_feature_names.pkl
"""
import requests, numpy as np, math, json, os, pickle
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

MLB_API = "https://statsapi.mlb.com/api/v1"
CACHE = {}

def cget(url, ttl=300):
    if url in CACHE:
        return CACHE[url]
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        data = r.json()
        CACHE[url] = data
        return data
    return None

def sf(v, d=0.0):
    try: return float(v) if v is not None else d
    except: return d

def parse_ip(ip):
    if not ip: return 0
    if isinstance(ip, str) and "." in ip:
        p = ip.split(".")
        return int(p[0]) + int(p[1])/3.0 if len(p)>1 else float(p[0])
    return float(ip or 0)

def fetch_team_abbrevs():
    data = cget(f"{MLB_API}/teams?sportIds=1", 86400)
    if not data: return {}
    return {t["id"]: t.get("abbreviation","??") for t in data.get("teams",[])}

def fetch_team_log(tid, start_date, end_date):
    """Fetch all games for a team in a date range."""
    sd = start_date.strftime("%m/%d/%Y")
    ed = end_date.strftime("%m/%d/%Y")
    url = f"{MLB_API}/schedule?sportId=1&teamId={tid}&startDate={sd}&endDate={ed}"
    data = cget(url, 3600)
    if not data: return []
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status",{}).get("codedGameState") == "F":
                games.append(g)
    return games

def fetch_stats_ytd(tid, up_to_date):
    """Fetch team stats through day before up_to_date (YTD pre-game)."""
    end = (up_to_date - timedelta(days=1)).strftime("%Y-%m-%d")
    start = "2026-03-27"
    url = f"{MLB_API}/teams/{tid}/stats?startDate={start}&endDate={end}&group=hitting,pitching&stats=season"
    data = cget(url, 3600)
    if not data: return None
    res = {"hitting": {}, "pitching": {}}
    for sg in data.get("stats", []):
        g = sg.get("group",{}).get("displayName","").lower()
        sp = sg.get("splits", [])
        if sp:
            s = sp[0].get("stat", {})
            if g == "hitting":
                res["hitting"] = {k: sf(s.get(k)) for k in ["avg","runs","hr","obp","slg","ops"]}
            elif g == "pitching":
                ipv = parse_ip(s.get("inningsPitched","0"))
                er = sf(s.get("earnedRuns"))
                wh = (sf(s.get("walks",0)) + sf(s.get("hits",0))) / ipv if ipv > 0 else 0
                res["pitching"] = {
                    "era": 9*er/ipv if ipv>0 else 0, "whip": min(wh, 3.0),
                    "runs": sf(s.get("runs")), "strikeouts": sf(s.get("strikeOuts")),
                    "ops": sf(s.get("ops"))
                }
    return res

def compute_elo_from_log(team_log, team_ids):
    """Compute Elo ratings for all teams after processing team_log."""
    elos = {tid: 1500 for tid in team_ids}
    ordered = sorted(team_log, key=lambda x: x.get("gameDate",""))
    for g in ordered:
        t = g["teams"]
        try:
            hg = t["home"]["team"]["id"]
            ag = t["away"]["team"]["id"]
            hs = int(t["home"].get("score",0))
            aws = int(t["away"].get("score",0))
        except: continue
        if hs == 0 and aws == 0: continue
        he = elos.get(hg, 1500)
        ae = elos.get(ag, 1500)
        eh = 1 / (1 + 10 ** ((ae - he - 50) / 400))
        ah = 1 if hs > aws else (0 if hs < aws else 0.5)
        mg = min(math.log(abs(hs-aws)+1)/2.2, 1.5)
        elos[hg] += 32 * mg * (ah - eh)
        elos[ag] += 32 * mg * ((1-ah) - (1-eh))
    return elos

def compute_recent_form(team_id, all_games, up_to_date):
    """Compute recent form from games before up_to_date."""
    recent = [g for g in all_games if g.get("gameDate","").split("T")[0] < up_to_date.strftime("%Y-%m-%d")]
    recent = sorted(recent, key=lambda x: x.get("gameDate",""), reverse=True)[:20]
    if not recent: return {"wp":0.5, "rs":4.5, "ra":4.5}
    rs, ra = [], []
    w = 0
    for g in recent:
        t = g["teams"]
        try:
            is_h = t["home"]["team"]["id"] == team_id
            is_a = t["away"]["team"]["id"] == team_id
            if not is_h and not is_a: continue
            ms = sf(t["home"]["score"] if is_h else t["away"]["score"])
            os = sf(t["away"]["score"] if is_h else t["home"]["score"])
            rs.append(ms); ra.append(os)
            if ms > os: w += 1
        except: continue
    n = len(rs) or 1
    return {"wp": w/n, "rs": np.mean(rs) if rs else 4.5, "ra": np.mean(ra) if ra else 4.5}


# ─── 1. Fetch all games ───
print("Fetching all 2026 games...")
today = datetime.now()
all_dates = []
d = datetime(2026, 3, 27)
while d < today:
    all_dates.append(d)
    d += timedelta(days=1)

all_games = []
team_ids = set()
for dt in all_dates:
    ds = dt.strftime("%m/%d/%Y")
    data = cget(f"{MLB_API}/schedule?sportId=1&date={ds}&hydrate=probablePitcher", 3600)
    if not data: continue
    for de in data.get("dates", []):
        for g in de.get("games", []):
            if g.get("status",{}).get("codedGameState") == "F":
                all_games.append(g)
                try:
                    team_ids.add(g["teams"]["home"]["team"]["id"])
                    team_ids.add(g["teams"]["away"]["team"]["id"])
                except: pass
print(f"  Total games: {len(all_games)}, teams: {len(team_ids)}")

# ─── 2. Compute Elo ratings after each game ───
print("Computing Elo ratings...")
elo_history = []
elos = {tid: 1500 for tid in team_ids}
ordered_games = sorted(all_games, key=lambda x: x.get("gameDate",""))

# Track elo before each game
game_features = []
for g in ordered_games:
    try:
        t = g["teams"]
        htid = t["home"]["team"]["id"]
        atid = t["away"]["team"]["id"]
        gdate_str = g.get("gameDate","").split("T")[0]
        gdate = datetime.strptime(gdate_str, "%Y-%m-%d")
        hs = int(t["home"].get("score",0))
        aws = int(t["away"].get("score",0))
    except: continue
    if hs == 0 and aws == 0: continue

    # Store elo before this game
    h_elo_before = elos.get(htid, 1500)
    a_elo_before = elos.get(atid, 1500)

    # Compute features from pre-game data
    # Skip games too early in season (limited data)
    stats_h = fetch_stats_ytd(htid, gdate)
    stats_a = fetch_stats_ytd(atid, gdate)

    # Only add game if we have stats
    if not stats_h or not stats_a:
        # Update elo anyway
        eh = 1 / (1 + 10 ** ((a_elo_before - h_elo_before - 50) / 400))
        ah = 1 if hs > aws else (0 if hs < aws else 0.5)
        mg = min(math.log(abs(hs-aws)+1)/2.2, 1.5)
        elos[htid] += 32 * mg * (ah - eh)
        elos[atid] += 32 * mg * ((1-ah) - (1-eh))
        continue

    # Recent form (from games before this date)
    h_form = compute_recent_form(htid, ordered_games, gdate)
    a_form = compute_recent_form(atid, ordered_games, gdate)

    # Rest days
    h_last = [x for x in ordered_games if x.get("gameDate","").split("T")[0] < gdate_str and 
              (x["teams"]["home"]["team"]["id"]==htid or x["teams"]["away"]["team"]["id"]==htid)]
    a_last = [x for x in ordered_games if x.get("gameDate","").split("T")[0] < gdate_str and
              (x["teams"]["home"]["team"]["id"]==atid or x["teams"]["away"]["team"]["id"]==atid)]
    h_rest = 3
    a_rest = 3
    if h_last:
        try:
            ld = datetime.strptime(h_last[-1].get("gameDate","").split("T")[0], "%Y-%m-%d")
            h_rest = (gdate - ld).days
        except: pass
    if a_last:
        try:
            ld = datetime.strptime(a_last[-1].get("gameDate","").split("T")[0], "%Y-%m-%d")
            a_rest = (gdate - ld).days
        except: pass

    # Pitchers
    hp_data = g.get("teams",{}).get("home",{}).get("probablePitcher",{})
    ap_data = g.get("teams",{}).get("away",{}).get("probablePitcher",{})
    hpitch = {}
    apitch = {}
    if hp_data:
        pid = hp_data.get("id")
        if pid:
            try:
                end = (gdate - timedelta(days=1)).strftime("%Y-%m-%d")
                start = "2026-03-27"
                pd = cget(f"{MLB_API}/people/{pid}/stats?startDate={start}&endDate={end}&group=pitching&stats=season", 3600)
                if pd:
                    sp = pd.get("stats",[{}])[0].get("splits",[])
                    if sp:
                        s = sp[0].get("stat",{})
                        ipv = parse_ip(s.get("inningsPitched","0"))
                        hpitch = {"era": sf(s.get("era")), "ip": ipv,
                                  "k9": sf(s.get("strikeoutsPer9Inn")), "bb9": sf(s.get("walksPer9Inn")),
                                  "hr9": sf(s.get("homeRunsPer9"))}
            except: pass
    if ap_data:
        pid = ap_data.get("id")
        if pid:
            try:
                end = (gdate - timedelta(days=1)).strftime("%Y-%m-%d")
                start = "2026-03-27"
                pd = cget(f"{MLB_API}/people/{pid}/stats?startDate={start}&endDate={end}&group=pitching&stats=season", 3600)
                if pd:
                    sp = pd.get("stats",[{}])[0].get("splits",[])
                    if sp:
                        s = sp[0].get("stat",{})
                        ipv = parse_ip(s.get("inningsPitched","0"))
                        apitch = {"era": sf(s.get("era")), "ip": ipv,
                                  "k9": sf(s.get("strikeoutsPer9Inn")), "bb9": sf(s.get("walksPer9Inn")),
                                  "hr9": sf(s.get("homeRunsPer9"))}
            except: pass

    # Park factor
    PARK_FACTORS = {
        "Coors Field": 1.18, "Great American Ball Park": 1.05, "Citizens Bank Park": 1.04,
        "Fenway Park": 1.03, "Yankee Stadium": 1.03, "Globe Life Field": 1.02,
        "American Family Field": 1.02, "Busch Stadium": 1.01, "Chase Field": 1.01,
        "Comerica Park": 0.99, "Citi Field": 0.99, "T-Mobile Park": 0.98,
        "Oracle Park": 0.98, "Petco Park": 0.97, "Oakland Coliseum": 0.97,
        "Tropicana Field": 0.96, "Target Field": 0.96, "PNC Park": 0.97,
    }
    park_f = PARK_FACTORS.get(g.get("venue",{}).get("name",""), 1.0)

    f = {
        "h_elo": h_elo_before,
        "a_elo": a_elo_before,
        "h_wp": h_form["wp"],
        "a_wp": a_form["wp"],
        "h_rs_avg": h_form["rs"],
        "a_rs_avg": a_form["rs"],
        "h_ra_avg": h_form["ra"],
        "a_ra_avg": a_form["ra"],
        "h_rest": min(h_rest, 5),
        "a_rest": min(a_rest, 5),
        "h_ops": stats_h["hitting"].get("ops", 0.700),
        "a_ops": stats_a["hitting"].get("ops", 0.700),
        "h_whip": stats_h["pitching"].get("whip", 1.35),
        "a_whip": stats_a["pitching"].get("whip", 1.35),
        "h_era": stats_h["pitching"].get("era", 4.5),
        "a_era": stats_a["pitching"].get("era", 4.5),
        "park_factor": park_f,
    }

    # Pitcher features (smoothed for missing data)
    if hpitch and hpitch.get("ip",0) >= 10:
        f["h_p_era"] = hpitch["era"]
        f["h_p_k9"] = hpitch["k9"]
        f["h_p_bb9"] = hpitch["bb9"]
        f["h_p_hr9"] = hpitch["hr9"]
        f["h_p_valid"] = 1
    else:
        f["h_p_era"] = 4.5
        f["h_p_k9"] = 8.0
        f["h_p_bb9"] = 3.0
        f["h_p_hr9"] = 1.2
        f["h_p_valid"] = 0

    if apitch and apitch.get("ip",0) >= 10:
        f["a_p_era"] = apitch["era"]
        f["a_p_k9"] = apitch["k9"]
        f["a_p_bb9"] = apitch["bb9"]
        f["a_p_hr9"] = apitch["hr9"]
        f["a_p_valid"] = 1
    else:
        f["a_p_era"] = 4.5
        f["a_p_k9"] = 8.0
        f["a_p_bb9"] = 3.0
        f["a_p_hr9"] = 1.2
        f["a_p_valid"] = 0

    f["home_win"] = 1 if hs > aws else 0
    f["run_diff"] = hs - aws
    f["total_runs"] = hs + aws
    f["home_team"] = htid
    f["away_team"] = atid
    f["game_date"] = gdate_str

    game_features.append(f)

    # Update Elo
    eh = 1 / (1 + 10 ** ((a_elo_before - h_elo_before - 50) / 400))
    ah = 1 if hs > aws else (0 if hs < aws else 0.5)
    mg = min(math.log(abs(hs-aws)+1)/2.2, 1.5)
    elos[htid] += 32 * mg * (ah - eh)
    elos[atid] += 32 * mg * ((1-ah) - (1-eh))

print(f"  Feature rows: {len(game_features)}")

# ─── 3. Train XGBoost models ───
print("\nTraining models...")

feature_cols = [
    "h_elo", "a_elo", "h_wp", "a_wp",
    "h_rs_avg", "a_rs_avg", "h_ra_avg", "a_ra_avg",
    "h_rest", "a_rest",
    "h_ops", "a_ops", "h_whip", "a_whip", "h_era", "a_era",
    "park_factor",
    "h_p_era", "h_p_k9", "h_p_bb9", "h_p_hr9", "h_p_valid",
    "a_p_era", "a_p_k9", "a_p_bb9", "a_p_hr9", "a_p_valid",
]

X = np.array([[f[c] for c in feature_cols] for f in game_features])
y_win = np.array([f["home_win"] for f in game_features])
y_rdiff = np.array([f["run_diff"] for f in game_features])
y_total = np.array([f["total_runs"] for f in game_features])

# Train/test split (chronological: use first 80% for train, last 20% for test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_win_train, y_win_test = y_win[:split], y_win[split:]
y_rdiff_train, y_rdiff_test = y_rdiff[:split], y_rdiff[split:]
y_total_train, y_total_test = y_total[:split], y_total[split:]

print(f"  Train set: {len(X_train)}, Test set: {len(X_test)}")

# --- Home Win model (classification) ---
model_win = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
model_win.fit(X_train, y_win_train)
pred_win = model_win.predict(X_test)
acc_win = accuracy_score(y_win_test, pred_win)
print(f"  Home Win model: accuracy={acc_win:.3f}")

# --- Run Diff model (regression) ---
model_rdiff = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    random_state=42, n_jobs=-1,
)
model_rdiff.fit(X_train, y_rdiff_train)
pred_rdiff = model_rdiff.predict(X_test)
mae_rdiff = mean_absolute_error(y_rdiff_test, pred_rdiff)
print(f"  Run Diff model: MAE={mae_rdiff:.3f}")

# --- Total Runs model (regression) ---
model_total = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    random_state=42, n_jobs=-1,
)
model_total.fit(X_train, y_total_train)
pred_total = model_total.predict(X_test)
mae_total = mean_absolute_error(y_total_test, pred_total)
print(f"  Total Runs model: MAE={mae_total:.3f}")

# ─── 4. Save models ───
print("\nSaving models...")
with open("rf_home_win.pkl", "wb") as f:
    pickle.dump(model_win, f)
with open("rf_run_diff.pkl", "wb") as f:
    pickle.dump(model_rdiff, f)
with open("rf_total_runs.pkl", "wb") as f:
    pickle.dump(model_total, f)
with open("rf_feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("  Saved: rf_home_win.pkl, rf_run_diff.pkl, rf_total_runs.pkl, rf_feature_names.pkl")

# ─── 5. Monte Carlo simulation test ───
print("\nTesting Monte Carlo simulation on last 10 test games...")
n_sims = 5000
for i in range(min(10, len(X_test))):
    feats = X_test[i]
    feats_2d = feats.reshape(1, -1)

    # Get model predictions
    home_win_prob = model_win.predict_proba(feats_2d)[0, 1]
    exp_rdiff = model_rdiff.predict(feats_2d)[0]
    exp_total = model_total.predict(feats_2d)[0]

    # Monte Carlo: sample run diff from predicted distribution
    # Use predicted mean, with historical std
    rdiff_samples = np.random.normal(exp_rdiff, 3.0, n_sims)
    total_samples = np.random.normal(exp_total, 3.2, n_sims)

    mc_home_win = np.mean(rdiff_samples > 0)
    mc_cover_minus = np.mean(rdiff_samples >= 1.5)
    mc_cover_plus = np.mean(rdiff_samples >= -1.5)
    mc_over = np.mean(total_samples > 8.5)

    actual_hw = y_win_test[i]
    actual_rd = y_rdiff_test[i]
    actual_tot = y_total_test[i]
    actual_over = 1 if actual_tot > 8.5 else 0
    actual_cover_minus = 1 if actual_rd >= 1.5 else 0
    actual_cover_plus = 1 if actual_rd >= -1.5 else 0

    hw_ok = (home_win_prob > 0.5) == bool(actual_hw)
    cm_ok = (mc_cover_minus > 0.5) == bool(actual_cover_minus)
    cp_ok = (mc_cover_plus > 0.5) == bool(actual_cover_plus)
    ov_ok = (mc_over > 0.5) == bool(actual_over)

    print(f"  Game {i+1}: HW={home_win_prob:.2f}({'✅' if hw_ok else '❌'}) "
          f"RD={exp_rdiff:.2f}(act={actual_rd:+d}) "
          f"CV-={mc_cover_minus:.2f}({'✅' if cm_ok else '❌'}) "
          f"CV+={mc_cover_plus:.2f}({'✅' if cp_ok else '❌'}) "
          f"OV={mc_over:.2f}({'✅' if ov_ok else '❌'})"
    )

print("\nDone! Models ready for integration.")
