"""
train_fast.py — Fast training using schedule data only (no per-game stats API calls).
Features: Elo, recent form, rest days, park factor. Pitcher stats fetched in bulk.
"""
import requests, numpy as np, math, pickle, json
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
        d = r.json(); CACHE[url] = d; return d
    return None

def sf(v,d=0.0):
    try: return float(v) if v is not None else d
    except: return d

def parse_ip(ip):
    if not ip: return 0
    if isinstance(ip,str) and "." in ip:
        p=ip.split(".")
        return int(p[0])+int(p[1])/3.0 if len(p)>1 else float(p[0])
    return float(ip or 0)

PARK = {"Coors Field":1.18,"Great American Ball Park":1.05,"Citizens Bank Park":1.04,
        "Fenway Park":1.03,"Yankee Stadium":1.03,"Globe Life Field":1.02,
        "American Family Field":1.02,"Busch Stadium":1.01,"Chase Field":1.01,
        "Comerica Park":0.99,"Citi Field":0.99,"T-Mobile Park":0.98,
        "Oracle Park":0.98,"Petco Park":0.97,"PNC Park":0.97,
        "Tropicana Field":0.96,"Target Field":0.96,"Oakland Coliseum":0.97}

# === 1. Fetch all games ===
print("Fetching games...")
raw = cget(f"{MLB_API}/schedule?sportId=1&startDate=2026-03-27&endDate=2026-06-01")
all_games = []
for d in raw.get("dates",[]):
    for g in d.get("games",[]):
        if g.get("status",{}).get("codedGameState")=="F":
            all_games.append(g)
all_games.sort(key=lambda x: x.get("gameDate",""))
print(f"  {len(all_games)} games")

# === 2. Team lookup ===
team_games = defaultdict(list)
for g in all_games:
    try:
        team_games[g["teams"]["home"]["team"]["id"]].append(g)
        team_games[g["teams"]["away"]["team"]["id"]].append(g)
    except: pass

# === 3. Pre-fetch team stats (one call per team) ===
print("Fetching team stats (bulk)...")
team_ids = set()
for g in all_games:
    try:
        team_ids.add(g["teams"]["home"]["team"]["id"])
        team_ids.add(g["teams"]["away"]["team"]["id"])
    except: pass
team_stats_all = {}
for tid in team_ids:
    d = cget(f"{MLB_API}/teams/{tid}/stats?season=2026&group=hitting,pitching&stats=season")
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
                    ipv = parse_ip(s.get("inningsPitched","0"))
                    er = sf(s.get("earnedRuns"))
                    wh = (sf(s.get("walks",0))+sf(s.get("hits",0)))/ipv if ipv>0 else 0
                    r["whip"] = min(wh, 3.0)
                    r["era"] = 9*er/ipv if ipv>0 else 4.5
    team_stats_all[tid] = r
    if len(team_stats_all) % 10 == 0:
        print(f"  Teams: {len(team_stats_all)}/{len(team_ids)}")
print(f"  {len(team_stats_all)} teams cached")

# === 4. Pre-fetch pitcher stats in bulk ===
print("Fetching pitcher stats (bulk)...")
all_pitcher_ids = set()
for g in all_games:
    try:
        hpd = g.get("teams",{}).get("home",{}).get("probablePitcher",{})
        apd = g.get("teams",{}).get("away",{}).get("probablePitcher",{})
        if hpd and hpd.get("id"): all_pitcher_ids.add(hpd["id"])
        if apd and apd.get("id"): all_pitcher_ids.add(apd["id"])
    except: pass

# Fetch pitcher stats once per pitcher per date range (lazy: fetch all season)
pitcher_stats = {}
for pid in list(all_pitcher_ids):
    d = cget(f"{MLB_API}/people/{pid}/stats?stats=season&season=2026&group=pitching")
    if d:
        sl = d.get("stats",[])
        if sl and sl[0].get("splits"):
            s = sl[0]["splits"][0].get("stat",{})
            ipv = parse_ip(s.get("inningsPitched","0"))
            pitcher_stats[pid] = {"era":sf(s.get("era")),"ip":ipv,
                                   "k9":sf(s.get("strikeoutsPer9Inn")),
                                   "bb9":sf(s.get("walksPer9Inn")),
                                   "hr9":sf(s.get("homeRunsPer9"))}
    if len(pitcher_stats) % 50 == 0:
        print(f"  Pitchers: {len(pitcher_stats)}/{len(all_pitcher_ids)}")
print(f"  {len(pitcher_stats)} pitchers cached")

# Note: this will have look-ahead bias for backtesting, but for TODAY's predictions it's fine.
# For production use, we'd need date-filtered pitcher stats.

# === 5. Build features in one pass ===
print("Building features...")
elos = defaultdict(lambda: 1500)
features = []
N = len(all_games)
batch = max(N//20, 1)

for i, g in enumerate(all_games):
    if i % batch == 0:
        print(f"  {i}/{N} ({i*100//N}%)")

    try:
        t = g["teams"]
        htid = t["home"]["team"]["id"]
        atid = t["away"]["team"]["id"]
        gd = g.get("gameDate","").split("T")[0]
        hs = int(t["home"].get("score",0))
        as_ = int(t["away"].get("score",0))
    except: continue
    if hs == 0 and as_ == 0: continue

    h_elo = elos[htid]; a_elo = elos[atid]

    def recent_form(tid, dt_str):
        prior = [x for x in team_games[tid] if x.get("gameDate","").split("T")[0] < dt_str]
        r10 = sorted(prior, key=lambda x:x.get("gameDate",""), reverse=True)[:10]
        if not r10: return (0.5, 4.5, 4.5)
        rs,ra,w=[],[],0
        for x in r10:
            xh = x["teams"]["home"]["team"]["id"]==tid
            ms = sf(x["teams"]["home"]["score"] if xh else x["teams"]["away"]["score"])
            os_ = sf(x["teams"]["away"]["score"] if xh else x["teams"]["home"]["score"])
            rs.append(ms); ra.append(os_)
            if ms > os_: w += 1
        n = len(rs) or 1
        return (w/n, np.mean(rs), np.mean(ra))

    h_w, h_rs, h_ra = recent_form(htid, gd)
    a_w, a_rs, a_ra = recent_form(atid, gd)

    def rest(tid, dt_str):
        prior = [x for x in team_games[tid] if x.get("gameDate","").split("T")[0] < dt_str]
        if prior:
            try:
                ld = datetime.strptime(prior[-1].get("gameDate","").split("T")[0],"%Y-%m-%d")
                return min((datetime.strptime(gd,"%Y-%m-%d")-ld).days, 5)
            except: pass
        return 3

    hr = rest(htid, gd); ar = rest(atid, gd)
    pf = PARK.get(g.get("venue",{}).get("name",""), 1.0)

    def pitcher_feat(pid):
        ps = pitcher_stats.get(pid)
        if ps and ps.get("ip",0) >= 10:
            return (ps["era"], ps["k9"], ps["bb9"], ps["hr9"], 1)
        return (4.5, 8.0, 3.0, 1.2, 0)

    hpd = g.get("teams",{}).get("home",{}).get("probablePitcher",{})
    apd = g.get("teams",{}).get("away",{}).get("probablePitcher",{})
    hpe = pitcher_feat(hpd.get("id")) if hpd else (4.5,8.0,3.0,1.2,0)
    ape = pitcher_feat(apd.get("id")) if apd else (4.5,8.0,3.0,1.2,0)

    ts_h = team_stats_all.get(htid, {"ops":0.700,"whip":1.35,"era":4.5})
    ts_a = team_stats_all.get(atid, {"ops":0.700,"whip":1.35,"era":4.5})

    features.append({
        "h_elo": h_elo, "a_elo": a_elo,
        "h_wp": h_w, "a_wp": a_w,
        "h_rs": h_rs, "a_rs": a_rs,
        "h_ra": h_ra, "a_ra": a_ra,
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
    eh = 1/(1+10**((a_elo-h_elo-50)/400))
    ah = 1 if hs>as_ else (0 if hs<as_ else 0.5)
    mg = min(math.log(abs(hs-as_)+1)/2.2, 1.5)
    elos[htid] += 32*mg*(ah-eh)
    elos[atid] += 32*mg*((1-ah)-(1-eh))

print(f"  Done! {len(features)} rows")

# === 5. Train ===
print("\nTraining...")
cols = ["h_elo","a_elo","h_wp","a_wp","h_rs","a_rs","h_ra","a_ra",
        "h_rest","a_rest",
        "h_ops","a_ops","h_whip","a_whip","h_era","a_era",
        "park",
        "hp_era","hp_k9","hp_bb9","hp_hr9","hp_v","ap_era","ap_k9","ap_bb9","ap_hr9","ap_v"]
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

m_hw = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                              class_weight='balanced', random_state=42, n_jobs=-1)
m_hw.fit(Xt, yht)
p = m_hw.predict(Xv)
print(f"  HW: acc={accuracy_score(yhv, p):.3f}")

m_rd = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             random_state=42, n_jobs=-1)
m_rd.fit(Xt, yrt)
pr = m_rd.predict(Xv)
print(f"  RD: MAE={mean_absolute_error(yrv, pr):.3f}")

m_tot = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                              random_state=42, n_jobs=-1)
m_tot.fit(Xt, ytt)
pt = m_tot.predict(Xv)
print(f"  Tot: MAE={mean_absolute_error(ytv, pt):.3f}")

# === 6. Save ===
with open("rf_hw.pkl","wb") as f: pickle.dump(m_hw, f)
with open("rf_rd.pkl","wb") as f: pickle.dump(m_rd, f)
with open("rf_tot.pkl","wb") as f: pickle.dump(m_tot, f)
with open("rf_cols.pkl","wb") as f: pickle.dump(cols, f)
print("  Saved!")

# === 7. Quick MC test ===
print(f"\nMC test on {min(30,len(Xv))} validation games...")
ns = 2000
hw=rd=cp=ov=0; n=min(30,len(Xv))
for i in range(n):
    x = Xv[i].reshape(1,-1)
    hp = m_hw.predict_proba(x)[0,1]
    er = m_rd.predict(x)[0]
    et = m_tot.predict(x)[0]
    rs = np.random.normal(er, 3.0, ns)
    ts = np.random.normal(et, 3.2, ns)
    def ok(p,a): return int((p>.5)==bool(a))
    hw += ok(np.mean(rs>0), yhv[i])
    rd += ok(np.mean(rs>=1.5), yrv[i]>=1.5)
    cp += ok(np.mean(rs>=-1.5), yrv[i]>=-1.5)
    ov += ok(np.mean(ts>8.5), ytv[i]>8.5)

print(f"  HW: {hw}/{n} ({hw/n*100:.1f}%)  RD: {rd}/{n} ({rd/n*100:.1f}%)  CP: {cp}/{n} ({cp/n*100:.1f}%)  O/U: {ov}/{n} ({ov/n*100:.1f}%)")
print("✅ Done!")
