"""
train_v2.py — Retrain RF models with REAL starting pitcher data from boxscores.
Fetches starting pitchers for ALL 794 games via boxscore endpoint (parallel).
"""
import requests, numpy as np, math, pickle, json, time
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

MLB_API = "https://statsapi.mlb.com/api/v1"
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

# ─── Step 1: Fetch all games ───
print("=== Step 1: Fetching games ===")
raw = cget(f"{MLB_API}/schedule?sportId=1&startDate=2026-03-27&endDate=2026-06-01")
all_games = []
for d in raw.get("dates", []):
    for g in d.get("games", []):
        if g.get("status", {}).get("codedGameState") == "F":
            all_games.append(g)
all_games.sort(key=lambda x: x.get("gameDate", ""))
print(f"  {len(all_games)} completed games")

# ─── Step 2: Fetch starting pitchers from boxscores (parallel) ───
print("\n=== Step 2: Fetching starting pitchers ===")
game_pitchers = {}  # gamePk -> {home_pitcher_id, away_pitcher_id}

def get_starter(game_pk, team_side):
    """Get first pitcher ID from boxscore for given team side."""
    try:
        d = cget(f"{MLB_API}/game/{game_pk}/boxscore")
        if d:
            pitchers = d.get("teams", {}).get(team_side, {}).get("pitchers", [])
            if pitchers: return pitchers[0]
    except: pass
    return None

# Collect all gamePks
game_pks = [g["gamePk"] for g in all_games]
print(f"  Fetching boxscores for {len(game_pks)} games...")

# Use thread pool for parallel fetching
def fetch_game_starters(g):
    pk = g["gamePk"]
    hpid = get_starter(pk, "home")
    apid = get_starter(pk, "away")
    return pk, hpid, apid

batch_size = 50
all_pids = set()
total = len(all_games)

with ThreadPoolExecutor(max_workers=20) as ex:
    futures = [ex.submit(fetch_game_starters, g) for g in all_games]
    done = 0
    for f in as_completed(futures):
        pk, hpid, apid = f.result()
        game_pitchers[pk] = {"h": hpid, "a": apid}
        if hpid: all_pids.add(hpid)
        if apid: all_pids.add(apid)
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total}")

covered = sum(1 for v in game_pitchers.values() if v["h"] and v["a"])
print(f"  Games with both starters: {covered}/{total}")

# ─── Step 3: Fetch season stats for all pitchers ───
print(f"\n=== Step 3: Fetching pitcher stats ({len(all_pids)} unique) ===")
pstats = {}
batch = 1
for pid in sorted(all_pids):
    d = cget(f"{MLB_API}/people/{pid}/stats?stats=season&season=2026&group=pitching")
    if d:
        sl = d.get("stats", [])
        if sl and sl[0].get("splits"):
            s = sl[0]["splits"][0].get("stat", {})
            ipv = parse_ip(s.get("inningsPitched", "0"))
            pstats[pid] = {"era": sf(s.get("era")), "ip": ipv,
                           "k9": sf(s.get("strikeoutsPer9Inn")),
                           "bb9": sf(s.get("walksPer9Inn")),
                           "hr9": sf(s.get("homeRunsPer9"))}
    if batch % 50 == 0:
        print(f"    {batch}/{len(all_pids)}")
    batch += 1
print(f"  Pitchers with stats: {len(pstats)}/{len(all_pids)}")

def get_pitcher_feats(pid):
    ps = pstats.get(pid)
    if ps and ps.get("ip", 0) >= 10:
        return (ps["era"], ps["k9"], ps["bb9"], ps["hr9"], 1)
    return (4.5, 8.0, 3.0, 1.2, 0)

# ─── Step 4: Pre-fetch team stats ───
print("\n=== Step 4: Fetching team stats ===")
team_ids = set()
for g in all_games:
    team_ids.add(g["teams"]["home"]["team"]["id"])
    team_ids.add(g["teams"]["away"]["team"]["id"])
team_stats = {}
for tid in team_ids:
    d = cget(f"{MLB_API}/teams/{tid}/stats?season=2026&group=hitting,pitching&stats=season")
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
    team_stats[tid] = r
print(f"  {len(team_stats)} teams")

# ─── Step 5: Build features ───
print("\n=== Step 5: Building features ===")

# Team lookup for recent form
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
    pf = PARK.get(g.get("venue", {}).get("name", ""), 1.0)
    ts_h = team_stats.get(htid, {"ops": 0.700, "whip": 1.35, "era": 4.5})
    ts_a = team_stats.get(atid, {"ops": 0.700, "whip": 1.35, "era": 4.5})

    # REAL pitcher data from boxscore
    gp = game_pitchers.get(g["gamePk"], {})
    hpe = get_pitcher_feats(gp.get("h"))
    ape = get_pitcher_feats(gp.get("a"))

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
        "ap_era": ape[0], "ap_k9": ape[1], "ap_bb9": ape[2], "ap_hr9": ape[3], "ap_v": ape[4],
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

# ─── Step 6: Train models ───
print("\n=== Step 6: Training ===")
cols = ["h_elo", "a_elo", "h_wp", "a_wp", "h_rs", "a_rs", "h_ra", "a_ra",
        "h_rest", "a_rest",
        "h_ops", "a_ops", "h_whip", "a_whip", "h_era", "a_era",
        "park",
        "hp_era", "hp_k9", "hp_bb9", "hp_hr9", "hp_v",
        "ap_era", "ap_k9", "ap_bb9", "ap_hr9", "ap_v"]

X = np.array([[f[c] for c in cols] for f in features])
y_hw = np.array([f["hw"] for f in features])
y_rd = np.array([f["rd"] for f in features])
y_tot = np.array([f["tot"] for f in features])

sp = int(len(X) * 0.8)
Xt, Xv = X[:sp], X[sp:]
yht, yhv = y_hw[:sp], y_hw[sp:]
yrt, yrv = y_rd[:sp], y_rd[sp:]
ytt, ytv = y_tot[:sp], y_tot[sp:]
print(f"  Train: {len(Xt)}, Val: {len(Xv)}")

m_hw = RandomForestClassifier(n_estimators=300, max_depth=8,
                              min_samples_leaf=20, class_weight='balanced',
                              random_state=42, n_jobs=-1)
m_hw.fit(Xt, yht)
p_hw = m_hw.predict(Xv)
print(f"  HW acc:  {accuracy_score(yhv, p_hw):.3f}")

m_rd = RandomForestRegressor(n_estimators=300, max_depth=8,
                             min_samples_leaf=20, random_state=42, n_jobs=-1)
m_rd.fit(Xt, yrt)
p_rd = m_rd.predict(Xv)
print(f"  RD MAE:  {mean_absolute_error(yrv, p_rd):.3f}")

m_tot = RandomForestRegressor(n_estimators=300, max_depth=8,
                              min_samples_leaf=20, random_state=42, n_jobs=-1)
m_tot.fit(Xt, ytt)
p_tot = m_tot.predict(Xv)
print(f"  Tot MAE: {mean_absolute_error(ytv, p_tot):.3f}")

# ─── Step 7: Quick MC validation ───
print(f"\n=== Step 7: MC Validation ({min(50, len(Xv))} games) ===")
ns = 2000
hw = r15 = rn15 = ov = 0
n = min(50, len(Xv))
for i in range(n):
    x = Xv[i].reshape(1, -1)
    hp = m_hw.predict_proba(x)[0, 1]
    er = m_rd.predict(x)[0]
    et = m_tot.predict(x)[0]
    rs = np.random.normal(er, 3.0, ns)
    ts = np.random.normal(et, 3.2, ns)
    hw += int((np.mean(rs > 0) > .5) == bool(yhv[i]))
    r15 += int((np.mean(rs >= 1.5) > .5) == bool(yrv[i] >= 1.5))
    rn15 += int((np.mean(rs >= -1.5) > .5) == bool(yrv[i] >= -1.5))
    ov += int((np.mean(ts > 8.5) > .5) == bool(ytv[i] > 8.5))

print(f"  HW: {hw}/{n} ({hw/n*100:.1f}%)")
print(f"  RL Fav -1.5: {r15}/{n} ({r15/n*100:.1f}%)")
print(f"  RL Dog +1.5: {rn15}/{n} ({rn15/n*100:.1f}%)")
print(f"  O/U: {ov}/{n} ({ov/n*100:.1f}%)")

# ─── Save ───
base_out = "/Users/comic/Documents/mlb-picks-app/"
with open(base_out + "rf_hw.pkl", "wb") as f: pickle.dump(m_hw, f)
with open(base_out + "rf_rd.pkl", "wb") as f: pickle.dump(m_rd, f)
with open(base_out + "rf_tot.pkl", "wb") as f: pickle.dump(m_tot, f)
with open(base_out + "rf_cols.pkl", "wb") as f: pickle.dump(cols, f)
print(f"\n✅ Models saved to {base_out}")
