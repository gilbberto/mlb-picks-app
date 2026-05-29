"""
predict_today.py — Predicciones del día con RF+MC v2 (con pitchers reales) + Odds API.
"""
import requests, pickle, numpy as np, math, os, json
from datetime import datetime, timedelta
from collections import defaultdict
from bankroll import recommend_stake, get_pnl, add_pick

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_KEY = "3988754e84aac800a8ee2eeca88cb085"
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

PARK = {"Coors Field":1.18,"Great American Ball Park":1.05,"Citizens Bank Park":1.04,"Fenway Park":1.03,"Yankee Stadium":1.03,"Globe Life Field":1.02,"American Family Field":1.02,"Busch Stadium":1.01,"Chase Field":1.01,"Comerica Park":0.99,"Citi Field":0.99,"T-Mobile Park":0.98,"Oracle Park":0.98,"Petco Park":0.97,"PNC Park":0.97,"Tropicana Field":0.96,"Target Field":0.96,"Oakland Coliseum":0.97}

# ─── Load models ───
base = "/Users/comic/Documents/mlb-picks-app/"
with open(base + "rf_hw.pkl", "rb") as f: rf_hw = pickle.load(f)
with open(base + "rf_rd.pkl", "rb") as f: rf_rd = pickle.load(f)
with open(base + "rf_tot.pkl", "rb") as f: rf_tot = pickle.load(f)
with open(base + "rf_cols.pkl", "rb") as f: cols = pickle.load(f)

today = "2026-05-27"

# ─── Fetch today's schedule ───
raw = cget(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=probablePitcher")
games = raw.get("dates", [{}])[0].get("games", []) if raw.get("dates") else []

# ─── Season data ───
raw_s = cget(f"{MLB_API}/schedule?sportId=1&startDate=2026-03-27&endDate={today}")
ag = []
for d in raw_s.get("dates", []):
    for g in d.get("games", []):
        if g.get("status", {}).get("codedGameState") == "F": ag.append(g)
ag.sort(key=lambda x: x.get("gameDate", ""))
tg = defaultdict(list)
for g in ag: tg[g["teams"]["home"]["team"]["id"]].append(g); tg[g["teams"]["away"]["team"]["id"]].append(g)

# ─── Elo (pre-today) ───
elos = defaultdict(lambda: 1500)
for g in ag:
    try:
        t = g["teams"]; htid, atid = t["home"]["team"]["id"], t["away"]["team"]["id"]
        gd = g.get("gameDate", "").split("T")[0]
        if gd >= today: continue
        hs = int(t.get("home", {}).get("score", 0)); as_ = int(t.get("away", {}).get("score", 0))
        if hs == 0 and as_ == 0: continue
        he, ae = elos[htid], elos[atid]
        eh = 1 / (1 + 10 ** ((ae - he - 50) / 400))
        ah = 1 if hs > as_ else (0.5 if hs == as_ else 0)
        mg = min(math.log(abs(hs - as_) + 1) / 2.2, 1.5)
        elos[htid] += 32 * mg * (ah - eh); elos[atid] += 32 * mg * ((1 - ah) - (1 - eh))
    except: pass

# ─── Team stats ───
tids = set()
for g in ag + games: tids.add(g["teams"]["home"]["team"]["id"]); tids.add(g["teams"]["away"]["team"]["id"])
ts = {}
for tid in tids:
    d = cget(f"{MLB_API}/teams/{tid}/stats?season=2026&group=hitting,pitching&stats=season")
    r = {"ops": 0.700, "whip": 1.35, "era": 4.5}
    if d:
        for sg in d.get("stats", []):
            g_ = sg.get("group", {}).get("displayName", "").lower(); sp = sg.get("splits", [])
            if sp:
                s = sp[0].get("stat", {})
                if g_ == "hitting": r["ops"] = sf(s.get("ops"), 0.700)
                elif g_ == "pitching":
                    ipv = parse_ip(s.get("inningsPitched", "0")); er = sf(s.get("earnedRuns"))
                    wh = (sf(s.get("walks", 0)) + sf(s.get("hits", 0))) / ipv if ipv > 0 else 0
                    r["whip"] = min(wh, 3.0); r["era"] = 9 * er / ipv if ipv > 0 else 4.5
    ts[tid] = r

# ─── Pitcher stats ───
all_pids = set()
for g in games + ag:
    for side in ["home", "away"]:
        pp = g.get("teams", {}).get(side, {}).get("probablePitcher", {})
        if pp and pp.get("id"): all_pids.add(pp["id"])
pstats = {}
for pid in all_pids:
    d = cget(f"{MLB_API}/people/{pid}/stats?stats=season&season=2026&group=pitching")
    if d:
        sl = d.get("stats", [])
        if sl and sl[0].get("splits"):
            s = sl[0]["splits"][0].get("stat", {})
            ipv = parse_ip(s.get("inningsPitched", "0"))
            pstats[pid] = {"era": sf(s.get("era")), "ip": ipv, "k9": sf(s.get("strikeoutsPer9Inn")),
                           "bb9": sf(s.get("walksPer9Inn")), "hr9": sf(s.get("homeRunsPer9"))}

def pf(pid):
    ps = pstats.get(pid)
    if ps and ps.get("ip", 0) >= 10: return (ps["era"], ps["k9"], ps["bb9"], ps["hr9"], 1)
    return (4.5, 8.0, 3.0, 1.2, 0)

def tab(tid):
    d = cget(f"{MLB_API}/teams/{tid}")
    return d.get("teams", [{}])[0].get("abbreviation", "?") if d else "?"

def tname(tid):
    d = cget(f"{MLB_API}/teams/{tid}")
    return d.get("teams", [{}])[0].get("name", "?") if d else "?"

def rf(tid, dt):
    pr = [x for x in tg[tid] if x.get("gameDate", "").split("T")[0] < dt]
    r10 = sorted(pr, key=lambda x: x.get("gameDate", ""), reverse=True)[:10]
    if not r10: return (0.5, 4.5, 4.5)
    rs, ra, w = [], [], 0
    for x in r10:
        xh = x["teams"]["home"]["team"]["id"] == tid
        ms = sf(x["teams"]["home"]["score"] if xh else x["teams"]["away"]["score"])
        os_ = sf(x["teams"]["away"]["score"] if xh else x["teams"]["home"]["score"])
        rs.append(ms); ra.append(os_)
        if ms > os_: w += 1
    return (w / len(rs), np.mean(rs), np.mean(ra)) if rs else (0.5, 4.5, 4.5)

def rest(tid, dt):
    pr = [x for x in tg[tid] if x.get("gameDate", "").split("T")[0] < dt]
    if pr:
        try:
            ld = datetime.strptime(pr[-1].get("gameDate", "").split("T")[0], "%Y-%m-%d")
            return min((datetime.strptime(dt, "%Y-%m-%d") - ld).days, 5)
        except: pass
    return 3

def american_to_prob(price):
    if price is None or price == 0: return None
    if price > 0: return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)

# ─── Odds API ───
odds_raw = {}
for mkt in ["h2h", "spreads", "totals"]:
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
            f"?apiKey={ODDS_KEY}&regions=us&markets={mkt}"
            f"&bookmakers=fanduel,draftkings,betmgm&oddsFormat=american",
            timeout=10
        )
        if r.status_code == 200: odds_raw[mkt] = r.json()
    except: pass

def find_odds(hteam, ateam, mkt):
    if mkt not in odds_raw: return []
    res = []
    for e in odds_raw[mkt]:
        if hteam.lower() in e.get("home_team","").lower() and ateam.lower() in e.get("away_team","").lower():
            for bk in e.get("bookmakers", []):
                for o in bk.get("markets", [{}])[0].get("outcomes", []):
                    res.append((bk["title"], o["name"], o["price"], o.get("point"), american_to_prob(o["price"])))
    return res

def best_price(results, target_name, target_point=None):
    best = None
    for bk, name, price, point, prob in results:
        if name.lower() == target_name.lower():
            if target_point is not None and (point is None or abs(point - target_point) > 0.1): continue
            if best is None:
                best = (bk, price, prob)
            else:
                cp = best[1]; np_ = price
                if (np_ > 0 and (cp < 0 or np_ > cp)) or (np_ < 0 and cp < 0 and np_ > cp):
                    best = (bk, price, prob)
    return best

def best_total(results):
    best = {}
    for bk, name, price, point, prob in results:
        is_over = "over" in name.lower()
        pt = point or 8.5
        key = ("over" if is_over else "under", pt)
        if key not in best:
            best[key] = (bk, price, prob)
        else:
            cp = best[key][1]; np_ = price
            if (np_ > 0 and (cp < 0 or np_ > cp)) or (np_ < 0 and cp < 0 and np_ > cp):
                best[key] = (bk, price, prob)
    return best

CITY_MAP = {"Yankees":"New York Yankees","Mets":"New York Mets","Cubs":"Chicago Cubs","White Sox":"Chicago White Sox",
    "Dodgers":"Los Angeles Dodgers","Angels":"Los Angeles Angels","Giants":"San Francisco Giants",
    "Athletics":"Athletics","Red Sox":"Boston Red Sox","Braves":"Atlanta Braves","Astros":"Houston Astros",
    "Rangers":"Texas Rangers","Phillies":"Philadelphia Phillies","Cardinals":"St. Louis Cardinals",
    "Brewers":"Milwaukee Brewers","Padres":"San Diego Padres","Mariners":"Seattle Mariners",
    "Diamondbacks":"Arizona Diamondbacks","Orioles":"Baltimore Orioles","Rays":"Tampa Bay Rays",
    "Blue Jays":"Toronto Blue Jays","Twins":"Minnesota Twins","Guardians":"Cleveland Guardians",
    "Tigers":"Detroit Tigers","Royals":"Kansas City Royals","Pirates":"Pittsburgh Pirates",
    "Reds":"Cincinnati Reds","Rockies":"Colorado Rockies","Marlins":"Miami Marlins","Nationals":"Washington Nationals"}

def oname(mlb_name):
    for short, full in CITY_MAP.items():
        if short in mlb_name: return full
    return mlb_name

# ═══════════════════════════════════════════
print(f"\n{'='*76}")
print(f"  MLB PICKS — {today} | Modelo v2 (con pitchers reales, {len(games)} juegos)")
print(f"{'='*76}\n")

for g in games:
    try:
        t = g["teams"]; htid = t["home"]["team"]["id"]; atid = t["away"]["team"]["id"]
        gd = g.get("gameDate", "").split("T")[0]
    except: continue
    ha = tab(htid); aa = tab(atid)
    hn = tname(htid); an = tname(atid)
    venue = g.get("venue", {}).get("name", "N/A")
    gt = g.get("gameDate", "").split("T")[1][:5] if "T" in g.get("gameDate", "") else ""

    h_elo = elos[htid]; a_elo = elos[atid]
    hf = rf(htid, gd); af = rf(atid, gd)
    hr = rest(htid, gd); ar = rest(atid, gd)
    pf_ = PARK.get(venue, 1.0)
    ts_h = ts.get(htid, {"ops": 0.700, "whip": 1.35, "era": 4.5})
    ts_a = ts.get(atid, {"ops": 0.700, "whip": 1.35, "era": 4.5})

    hpd = t.get("home", {}).get("probablePitcher", {})
    apd = t.get("away", {}).get("probablePitcher", {})
    hpn = hpd.get("fullName", "?") if hpd else "?"
    apn = apd.get("fullName", "?") if apd else "?"
    hpe = pf(hpd.get("id")) if hpd else (4.5, 8.0, 3.0, 1.2, 0)
    ape = pf(apd.get("id")) if apd else (4.5, 8.0, 3.0, 1.2, 0)

    row = {"h_elo": h_elo, "a_elo": a_elo, "h_wp": hf[0], "a_wp": af[0],
           "h_rs": hf[1], "a_rs": af[1], "h_ra": hf[2], "a_ra": af[2],
           "h_rest": hr, "a_rest": ar,
           "h_ops": ts_h["ops"], "a_ops": ts_a["ops"],
           "h_whip": ts_h["whip"], "a_whip": ts_a["whip"],
           "h_era": ts_h["era"], "a_era": ts_a["era"],
           "park": pf_,
           "hp_era": hpe[0], "hp_k9": hpe[1], "hp_bb9": hpe[2], "hp_hr9": hpe[3], "hp_v": hpe[4],
           "ap_era": ape[0], "ap_k9": ape[1], "ap_bb9": ape[2], "ap_hr9": ape[3], "ap_v": ape[4]}

    x = np.array([[row[c] for c in cols]])
    hw_p = rf_hw.predict_proba(x)[0, 1]
    er = rf_rd.predict(x)[0]; et = rf_tot.predict(x)[0]

    ns = 5000
    rs_ = np.random.normal(er, 3.0, ns); ts_ = np.random.normal(et, 3.2, ns)
    mc_hw = np.mean(rs_ > 0); mc_rd15 = np.mean(rs_ >= 1.5); mc_rdn15 = np.mean(rs_ >= -1.5); mc_ov = np.mean(ts_ > 8.5)

    # Favorite
    fav_home = mc_hw >= 0.5
    fav_p = mc_hw if fav_home else 1 - mc_hw
    fav_team = ha if fav_home else aa
    dog_team = aa if fav_home else ha
    rl_fav_p = mc_rd15 if fav_home else 1 - mc_rdn15
    rl_dog_p = mc_rdn15 if fav_home else 1 - mc_rd15

    # Odds
    o_hn = oname(hn); o_an = oname(an)
    o_h2h = find_odds(o_hn, o_an, "h2h")
    o_sp = find_odds(o_hn, o_an, "spreads")
    o_tot = find_odds(o_hn, o_an, "totals")

    fav_odd = best_price(o_h2h, oname(fav_team if fav_home else tname(htid) if fav_home else tname(atid)))
    if not fav_odd and o_h2h: fav_odd = best_price(o_h2h, oname(hn if fav_home else an))

    # Actually let's be smarter about odds matching
    on_fav = oname(hn if fav_home else an)
    on_dog = oname(an if fav_home else hn)
    fo = best_price(o_h2h, on_fav)
    do = best_price(o_h2h, on_dog)

    # Spread odds
    fspo = best_price(o_sp, on_fav, -1.5)
    dspo = best_price(o_sp, on_dog, 1.5)

    # Total odds
    to = best_total(o_tot) if o_tot else {}

    # Edge labels
    def edge(mp, odp):
        if odp is None: return ""
        e = mp - odp
        if e > 0.08: return "🔥 VALUE"
        if e > 0.05: return "⭐ VALUE"
        if e > 0.02: return "✓ edge"
        return ""

    def edge_with_stake(mp, odp, odds_num):
        """Edge label + Kelly stake info."""
        lbl = edge(mp, odp)
        if not lbl: return ""
        _, units, _ = recommend_stake(mp, odds_num, bankroll=1000)
        if units <= 0: return lbl
        return f"{lbl} {units:.1f}u"

    def fmt_odd(x):
        if x is None: return ""
        b, p, prob = x
        return f"${p:+d}"

    lines = []
    lines.append(f"┌{'─'*72}┐")
    lines.append(f"│ {aa} @ {ha}  │  {gt}  │  {venue}")
    lines.append(f"│ {an} @ {hn}")
    lines.append(f"│ {apn} ({ape[0]:.2f} ERA, {ape[2]:.1f} BB/9, {ape[1]:.1f} K/9) vs {hpn} ({hpe[0]:.2f} ERA, {hpe[2]:.1f} BB/9, {hpe[1]:.1f} K/9)")
    lines.append(f"│ Elo: {a_elo:.0f} @ {h_elo:.0f}  │  Forma: {af[0]:.0%} vs {hf[0]:.0%}  │  Park: {pf_:.2f}")
    lines.append(f"├{'─'*72}┤")

    # ML
    fav_str = f"{fav_team} {fav_p:.1%}"
    dog_str = f"{dog_team} {1-fav_p:.1%}"
    fav_odd_str = fmt_odd(fo) if fo else ""
    dog_odd_str = fmt_odd(do) if do else ""
    e_ml = edge_with_stake(fav_p, fo[2] if fo else None, fo[1] if fo else None)
    e_dog = edge_with_stake(1-fav_p, do[2] if do else None, do[1] if do else None)
    lines.append(f"│ ML  │ {fav_str:<18} {fav_odd_str:<10} {e_ml:<12} │ {dog_str:<18} {dog_odd_str:<10} {e_dog:<12}│")

    # RL
    rl_fav_str = f"{fav_team} -1.5 {rl_fav_p:.1%}"
    rl_dog_str = f"{dog_team} +1.5 {rl_dog_p:.1%}"
    fspo_str = fmt_odd(fspo) if fspo else ""
    dspo_str = fmt_odd(dspo) if dspo else ""
    e_rlf = edge_with_stake(rl_fav_p, fspo[2] if fspo else None, fspo[1] if fspo else None)
    e_rld = edge_with_stake(rl_dog_p, dspo[2] if dspo else None, dspo[1] if dspo else None)
    lines.append(f"│ RL  │ {rl_fav_str:<18} {fspo_str:<10} {e_rlf:<12} │ {rl_dog_str:<18} {dspo_str:<10} {e_rld:<12}│")

    # O/U
    ov_str = f"Over 8.5 {mc_ov:.1%}"
    un_str = f"Under 8.5 {1-mc_ov:.1%}"
    ov_odd_str = fmt_odd(to.get(("over", 8.5))) if to else ""
    un_odd_str = fmt_odd(to.get(("under", 8.5))) if to else ""
    e_ov = edge_with_stake(mc_ov, to.get(("over", 8.5), [None, None, None])[2] if to else None,
                           to.get(("over", 8.5), [None, None, None])[1] if to else None)
    e_un = edge_with_stake(1-mc_ov, to.get(("under", 8.5), [None, None, None])[2] if to else None,
                           to.get(("under", 8.5), [None, None, None])[1] if to else None)
    lines.append(f"│ O/U │ {ov_str:<18} {ov_odd_str:<10} {e_ov:<12} │ {un_str:<18} {un_odd_str:<10} {e_un:<12}│")

    lines.append(f"│ Pred: {er:+.2f} RD, {et:.2f} Total (σ=3.0/3.2), n={ns} sims")
    lines.append(f"└{'─'*72}┘")
    lines.append("")

    print("\n".join(lines))

# P&L Summary
pnl = get_pnl()
print("\n" + "=" * 60)
print(f"  Bankroll: ${pnl['bankroll']:>8.2f}  |  Record: {pnl['wins']}-{pnl['losses']} ({pnl['pct']}%)")
print(f"  P&L: ${pnl['profit']:>+8.2f} ({pnl['roi']:>+7.1f}% ROI)  |  Pendientes: {pnl['open']}")
if pnl["open"] > 0:
    print("  ⚠️  Hay picks de hoy sin resolver — corre settle_picks.py mañana")
print("=" * 60)

print("Leyenda: 🔥 VALUE = edge >8%  ⭐ VALUE = edge >5%  ✓ edge = 2-5%")
print("         X.Xu = unidades Kelly (25% fraccional, bankroll=$1,000)")
print("         Precios de FanDuel/DraftKings/BetMGM via The Odds API")
print(f"         Modelo v2: RandomForest (809 games) + Monte Carlo con pitchers reales")
