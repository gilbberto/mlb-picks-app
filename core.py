"""
core.py — Shared functions used by both app.py and predictions.py.
Single source of truth for feature building, odds matching, and math.
"""
import os, json, math, requests
import numpy as np

ODDS_API_KEY = "b09f7e5fb08081c87e7e34272fda4ea0"
SHARPAPI_KEY = os.environ.get("SHARPAPI_KEY", "")
PREFERRED_BOOK = "BetMGM"

# ─── Feature Building ───
# THIS FUNCTION MUST BE IDENTICAL IN app.py AND predictions.py
# If you modify it here, it affects both files.

def build_rf_feature_row(hs, aws, hf, af, h_elo, a_elo, hpitch, apitch, park_f,
                          hp_rec=None, ap_rec=None, weather=None, home_abbrev=None, away_abbrev=None):
    weather = weather or {}
    sc_h = _sc_defaults()
    sc_a = _sc_defaults()
    if home_abbrev or away_abbrev:
        try:
            base = os.path.join(os.path.dirname(__file__) or ".", "")
            with open(base + "statcast_2026.json") as f:
                sc_data = json.load(f)
            if home_abbrev:
                h_sc = sc_data.get(home_abbrev, _sc_defaults())
                sc_h = [h_sc[0], h_sc[1], h_sc[2], h_sc[3], h_sc[4], h_sc[5]]
            if away_abbrev:
                a_sc = sc_data.get(away_abbrev, _sc_defaults())
                sc_a = [a_sc[0], a_sc[1], a_sc[2], a_sc[3], a_sc[4], a_sc[5]]
        except:
            pass
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
        "h_sc_ev": sc_h[0], "h_sc_barrel": sc_h[1], "h_sc_hardhit": sc_h[2],
        "h_sc_xwoba": sc_h[3], "h_sc_batspeed": sc_h[4], "h_sc_la": sc_h[5],
        "a_sc_ev": sc_a[0], "a_sc_barrel": sc_a[1], "a_sc_hardhit": sc_a[2],
        "a_sc_xwoba": sc_a[3], "a_sc_batspeed": sc_a[4], "a_sc_la": sc_a[5],
    }
    return f

def _sc_defaults():
    return [82.65, 7.51, 24.46, 0.37, 70.72, 18.06]

# ─── Odds Matching ───

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
        return None, None, None
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
    if market_key == "h2h":
        return best_price, best_book
    return best_price, best_book, best_point

# ─── Math ───

def american_to_prob(odds):
    if odds is None or odds == 0: return None
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def norm_cdf(x, mu=0, sigma=1):
    return 0.5 * (1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

def compute_ev(prob, odds):
    if odds is None: return None
    dec = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return round((prob/100 * dec) - 1, 4)
