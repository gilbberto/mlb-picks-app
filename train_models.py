"""
train_models.py — Fast: builds features from schedule only, no per-game API calls.
Trains RandomForest for Monte Carlo simulation.
"""
import requests, numpy as np, math, pickle
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

MLB_API = "https://statsapi.mlb.com/api/v1"
CACHE = {}
def cget(url):
    if url in CACHE: return CACHE[url]
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        d = r.json()
        CACHE[url] = d
        return d
    return None

def sf(v,d=0.0):
    try: return float(v) if v is not None else d
    except: return d

# === 1. Fetch all games in one call ===
print("Fetching games Mar 27 – Jun 1...")
data = cget(f"{MLB_API}/schedule?sportId=1&startDate=2026-03-27&endDate=2026-06-01&hydrate=probablePitcher")
all_games = []
for d in data.get("dates",[]):
    for g in d.get("games",[]):
        if g.get("status",{}).get("codedGameState")=="F":
            all_games.append(g)
all_games.sort(key=lambda x: x.get("gameDate",""))
print(f"  {len(all_games)} games")

# === 2. Teams & caching ===
team_ids = set()
for g in all_games:
    try:
        team_ids.add(g["teams"]["home"]["team"]["id"])
        team_ids.add(g["teams"]["away"]["team"]["id"])
    except: pass

team_games = defaultdict(list)
for g in all_games:
    try:
        h = g["teams"]["home"]["team"]["id"]
        a = g["teams"]["away"]["team"]["id"]
        team_games[h].append(g)
        team_games[a].append(g)
    except: pass

PARK = {"Coors Field":1.18,"Great American Ball Park":1.05,"Citizens Bank Park":1.04,
        "Fenway Park":1.03,"Yankee Stadium":1.03,"Globe Life Field":1.02,
        "American Family Field":1.02,"Busch Stadium":1.01,"Chase Field":1.01,
        "Comerica Park":0.99,"Citi Field":0.99,"T-Mobile Park":0.98,
        "Oracle Park":0.98,"Petco Park":0.97,"PNC Park":0.97,
        "Tropicana Field":0.96,"Target Field":0.96,"Oakland Coliseum":0.97}

# === 3. Single-pass feature builder ===
print("Building features...")

# For looking up latest season stats per team (fetched once per team)
TEAM_STATS_CACHE = {}
def get_season_stats(tid, before_date):
    """Get YTD stats once per team, reuse for all games on same date."""
    key = (tid, before_date.strftime("%Y-%m-%d"))
    if key in TEAM_STATS_CACHE: return TEAM_STATS_CACHE[key]

    end = (before_date - timedelta(days=1)).strftime("%Y-%m-%d")
    if end < "2026-03-27":
        TEAM_STATS_CACHE[key] = None
        return None
    url = f"{MLB_API}/teams/{tid}/stats?startDate=2026-03-27&endDate={end}&group=hitting,pitching&stats=season"
    d = cget(url)
    r = {"ops":0.700, "whip":1.35, "era":4.5}
    if d:
        for sg in d.get("stats",[]):
            g = sg.get("group",{}).get("displayName","").lower()
            sp = sg.get("splits",[])
            if sp:
                s = sp[0].get("stat",{})
                if g == "hitting":
                    r["ops"] = sf(s.get("ops"),0.700)
                elif g == "pitching":
                    ip = s.get("inningsPitched","0")
                    ipv = 0
                    if isinstance(ip,str) and "." in ip:
                        p = ip.split("."); ipv = int(p[0])+int(p[1])/3.0 if len(p)>1 else float(p[0])
                    else: ipv = float(ip or 0)
                    er = sf(s.get("earnedRuns"))
                    wh = (sf(s.get("walks",0))+sf(s.get("hits",0)))/ipv if ipv>0 else 0
                    r["whip"] = min(wh,3.0)
                    r["era"] = 9*er/ipv if ipv>0 else 4.5
    TEAM_STATS_CACHE[key] = r
    return r

def get_pitcher(pid, before_date):
    """Get pitcher season stats before a date."""
    if not pid: return None
    key = ("p",pid,before_date.strftime("%Y-%m-%d"))
    if key in TEAM_STATS_CACHE: return TEAM_STATS_CACHE[key]
    end = (before_date - timedelta(days=1)).strftime("%Y-%m-%d")
    if end < "2026-03-27":
        TEAM_STATS_CACHE[key] = None
        return None
    url = f"{MLB_API}/people/{pid}/stats?startDate=2026-03-27&endDate={end}&group=pitching&stats=season"
    d = cget(url)
    r = None
    if d:
        sl = d.get("stats",[])
        if sl:
            sp = sl[0].get("splits",[])
            if sp:
                s = sp[0].get("stat",{})
                ip = s.get("inningsPitched","0")
                ipv = 0
                if isinstance(ip,str) and "." in ip:
                    p = ip.split("."); ipv = int(p[0])+int(p[1])/3.0 if len(p)>1 else float(p[0])
                else: ipv = float(ip or 0)
                r = {"era":sf(s.get("era")),"ip":ipv,
                     "k9":sf(s.get("strikeoutsPer9Inn")),
                     "bb9":sf(s.get("walksPer9Inn")),
                     "hr9":sf(s.get("homeRunsPer9"))}
    TEAM_STATS_CACHE[key] = r
    return r

elos = {tid: 1500 for tid in team_ids}
features = []
N = len(all_games)
batch = max(N//20, 1)

for i, g in enumerate(all_games):
    if i % batch == 0:
        print(f"  {i}/{N} ({i*100//N}%)")

    try:
        t = g["teams"]
        htid, atid = t["home"]["team"]["id"], t["away"]["team"]["id"]
        gd = g.get("gameDate","").split("T")[0]
        gdate = datetime.strptime(gd, "%Y-%m-%d")
        hs = int(t["home"].get("score",0))
        aws = int(t["away"].get("score",0))
    except: continue
    if hs == 0 and aws == 0: continue

    h_elo = elos[htid]; a_elo = elos[atid]

    # Recent form: last 10 games before this one
    prior = [x for x in team_games[htid] if x.get("gameDate","").split("T")[0] < gd]
    r10 = sorted(prior, key=lambda x:x.get("gameDate",""), reverse=True)[:10]
    hf_w, hf_rs, hf_ra = 0.5, 4.5, 4.5
    if r10:
        rs, ra, w = [], [], 0
        for x in r10:
            xh = x["teams"]["home"]["team"]["id"]==htid
            ms = sf(x["teams"]["home"]["score"] if xh else x["teams"]["away"]["score"])
            os_ = sf(x["teams"]["away"]["score"] if xh else x["teams"]["home"]["score"])
            rs.append(ms); ra.append(os_)
            if ms > os_: w += 1
        hf_w = w/len(rs); hf_rs = np.mean(rs); hf_ra = np.mean(ra)

    prior_a = [x for x in team_games[atid] if x.get("gameDate","").split("T")[0] < gd]
    r10a = sorted(prior_a, key=lambda x:x.get("gameDate",""), reverse=True)[:10]
    af_w, af_rs, af_ra = 0.5, 4.5, 4.5
    if r10a:
        rs, ra, w = [], [], 0
        for x in r10a:
            xh = x["teams"]["home"]["team"]["id"]==atid
            ms = sf(x["teams"]["home"]["score"] if xh else x["teams"]["away"]["score"])
            os_ = sf(x["teams"]["away"]["score"] if xh else x["teams"]["home"]["score"])
            rs.append(ms); ra.append(os_)
            if ms > os_: w += 1
        af_w = w/len(rs); af_rs = np.mean(rs); af_ra = np.mean(ra)

    # Rest
    def rest_days(tid, gs, dt):
        p = [x for x in gs if x.get("gameDate","").split("T")[0] < dt]
        if p:
            try:
                ld = datetime.strptime(p[-1].get("gameDate","").split("T")[0], "%Y-%m-%d")
                return min((gdate-ld).days, 5)
            except: pass
        return 3
    hr = rest_days(htid, team_games[htid], gd)
    ar = rest_days(atid, team_games[atid], gd)

    # Season stats (lazy, only 1 call per team per date)
    ss_h = get_season_stats(htid, gdate)
    ss_a = get_season_stats(atid, gdate)

    pit_h = None
    pit_a = None
    hpd = g.get("teams",{}).get("home",{}).get("probablePitcher",{})
    apd = g.get("teams",{}).get("away",{}).get("probablePitcher",{})
    if hpd: pit_h = get_pitcher(hpd.get("id"), gdate)
    if apd: pit_a = get_pitcher(apd.get("id"), gdate)

    pf = PARK.get(g.get("venue",{}).get("name",""), 1.0)

    def pfeat(p):
        if p and p.get("ip",0)>=10:
            return (p["era"], p["k9"], p["bb9"], p["hr9"], 1)
        return (4.5, 8.0, 3.0, 1.2, 0)

    hpe = pfeat(pit_h); ape = pfeat(pit_a)

    features.append({
        "h_elo": h_elo, "a_elo": a_elo,
        "h_wp": hf_w, "a_wp": af_w,
        "h_rs": hf_rs, "a_rs": af_rs,
        "h_ra": hf_ra, "a_ra": af_ra,
        "h_rest": hr, "a_rest": ar,
        "h_ops": ss_h["ops"] if ss_h else 0.700,
        "a_ops": ss_a["ops"] if ss_a else 0.700,
        "h_whip": ss_h["whip"] if ss_h else 1.35,
        "a_whip": ss_a["whip"] if ss_a else 1.35,
        "h_era": ss_h["era"] if ss_h else 4.5,
        "a_era": ss_a["era"] if ss_a else 4.5,
        "park": pf,
        "hp_era": hpe[0], "hp_k9": hpe[1], "hp_bb9": hpe[2], "hp_hr9": hpe[3], "hp_v": hpe[4],
        "ap_era": ape[0], "ap_k9": ape[1], "ap_bb9": ape[2], "ap_hr9": ape[3], "ap_v": ape[4],
        "hw": 1 if hs>aws else 0,
        "rd": hs-aws,
        "tot": hs+aws,
    })

    # Update Elo
    eh = 1/(1+10**((a_elo-h_elo-50)/400))
    ah = 1 if hs>aws else (0 if hs<aws else 0.5)
    mg = min(math.log(abs(hs-aws)+1)/2.2, 1.5)
    elos[htid] += 32*mg*(ah-eh)
    elos[atid] += 32*mg*((1-ah)-(1-eh))

print(f"  Done! {len(features)} rows")

# === 4. Train ===
print("\nTraining...")
cols = ["h_elo","a_elo","h_wp","a_wp","h_rs","a_rs","h_ra","a_ra",
        "h_rest","a_rest","h_ops","a_ops","h_whip","a_whip","h_era","a_era",
        "park","hp_era","hp_k9","hp_bb9","hp_hr9","hp_v","ap_era","ap_k9","ap_bb9","ap_hr9","ap_v"]
X = np.array([[f[c] for c in cols] for f in features])
y_hw = np.array([f["hw"] for f in features])
y_rd = np.array([f["rd"] for f in features])
y_tot = np.array([f["tot"] for f in features])

sp = int(len(X)*0.8)
Xt, Xv = X[:sp], X[sp:]
yht, yhv = y_hw[:sp], y_hw[sp:]
yrt, yrv = y_rd[:sp], y_rd[sp:]
ytt, ytv = y_tot[:sp], y_tot[sp:]
print(f"  Train: {len(Xt)}, Val: {len(Xv)}")

m_hw = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1)
m_hw.fit(Xt, yht)
p = m_hw.predict(Xv)
print(f"  Home Win: acc={accuracy_score(yhv, p):.3f}")

m_rd = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
m_rd.fit(Xt, yrt)
pr = m_rd.predict(Xv)
print(f"  Run Diff: MAE={mean_absolute_error(yrv, pr):.3f}")

m_tot = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
m_tot.fit(Xt, ytt)
pt = m_tot.predict(Xv)
print(f"  Total:    MAE={mean_absolute_error(ytv, pt):.3f}")

# === 5. Save ===
with open("rf_hw.pkl","wb") as f: pickle.dump(m_hw, f)
with open("rf_rd.pkl","wb") as f: pickle.dump(m_rd, f)
with open("rf_tot.pkl","wb") as f: pickle.dump(m_tot, f)
with open("rf_cols.pkl","wb") as f: pickle.dump(cols, f)
print("  Saved: rf_hw.pkl, rf_rd.pkl, rf_tot.pkl, rf_cols.pkl")

# === 6. Monte Carlo quick test ===
print(f"\nMonte Carlo test ({min(30,len(Xv))} val games, 2000 sims)...")
n_sims = 2000
def mc_test():
    hw=rd=cp=ov=tot=0
    for i in range(min(30,len(Xv))):
        x = Xv[i].reshape(1,-1)
        hp = m_hw.predict_proba(x)[0,1]
        er = m_rd.predict(x)[0]
        et = m_tot.predict(x)[0]
        rs = np.random.normal(er, 3.0, n_sims)
        ts = np.random.normal(et, 3.2, n_sims)
        def ok(p, a): return int((p>.5)==bool(a))
        hw += ok(np.mean(rs>0), yhv[i])
        rd += ok(np.mean(rs>=1.5), yrv[i]>=1.5)
        cp += ok(np.mean(rs>=-1.5), yrv[i]>=-1.5)
        ov += ok(np.mean(ts>8.5), ytv[i]>8.5)
        tot += 1
    n = tot
    print(f"  HW: {hw}/{n} ({hw/n*100:.1f}%)  RD-1.5: {rd}/{n} ({rd/n*100:.1f}%)  CP+1.5: {cp}/{n} ({cp/n*100:.1f}%)  O/U: {ov}/{n} ({ov/n*100:.1f}%)")
mc_test()

print("\n✅ Done! Models ready for integration.")
