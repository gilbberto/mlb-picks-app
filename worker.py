"""
worker.py — Railway background process.
Uses GitHub API (no git CLI needed) for state sync.
Runs settle_and_notify.py every 30s for instant Telegram response.
Checks for new high-edge picks every ~2h during game hours.
"""
import subprocess, time, os, json, base64, requests
from datetime import datetime, timezone, timedelta
try: from zoneinfo import ZoneInfo; TZ = ZoneInfo("America/Chihuahua")
except: TZ = timezone(timedelta(hours=-6))

CWD = os.path.dirname(os.path.abspath(__file__))
ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"

REPO = "gilbberto/mlb-picks-app"
BRANCH = "main"
FILES_TO_SYNC = ("picks.json", "users.json", "game_starts_notified.json", "predictions_log.json", ".morning_sent", ".telegram_offset", ".notified_new_picks.json")

def _gh_headers():
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        return None
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}

def _gh_get(path):
    """Get a file from GitHub. Returns (content_str, sha) or (None, None)."""
    headers = _gh_headers()
    if not headers:
        return None, None
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"]).decode()
        return content, data.get("sha", "")
    return None, None

def _gh_put(path, content, sha=None):
    """Write a file to GitHub. Returns True on success."""
    headers = _gh_headers()
    if not headers:
        return False
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    data = {"message": f"sync {path} from worker", "content": base64.b64encode(content.encode()).decode(), "branch": BRANCH}
    if sha:
        data["sha"] = sha
    r = requests.put(url, json=data, headers=headers, timeout=10)
    return r.status_code in (200, 201)

def sync_from_github():
    """Pull latest state files from GitHub."""
    for fname in FILES_TO_SYNC:
        content, _ = _gh_get(fname)
        if content:
            with open(os.path.join(CWD, fname), "w") as f:
                f.write(content)

def _merge_picks(local_str, remote_str):
    """Merge picks: keep remote entries + add any local entries not in remote."""
    import json
    remote = json.loads(remote_str)
    local = json.loads(local_str)
    remote_ids = {p.get("id") for p in remote.get("history", [])}
    for p in local.get("history", []):
        if p.get("id") not in remote_ids:
            remote["history"].append(p)
    remote["bankroll"] = min(remote.get("bankroll", 1000), local.get("bankroll", 1000))
    remote["history"].sort(key=lambda x: x.get("id", 0))
    return json.dumps(remote, indent=2)

def sync_to_github():
    """Push changed state files to GitHub (merge picks.json, overwrite others)."""
    for fname in FILES_TO_SYNC:
        fp = os.path.join(CWD, fname)
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            local_content = f.read()
        remote_content, sha = _gh_get(fname)
        if remote_content == local_content:
            continue
        if fname == "picks.json" and remote_content:
            local_content = _merge_picks(local_content, remote_content)
        _gh_put(fname, local_content, sha)

def main():
    print("=== Worker iniciado en Railway ===")
    sync_from_github()
    cycle = 0
    while True:
        sync_from_github()
        subprocess.run(["python3", "settle_and_notify.py"], cwd=CWD, env=ENV)
        sync_to_github()
        if cycle == 0 or cycle % 240 == 0:
            h = datetime.now(TZ).hour
            if 6 <= h < 12:
                pass  # morning_summary handles this window
            elif 12 <= h <= 23:
                print("=== Verificando nuevos picks ===")
                subprocess.run(["python3", "notify_new_picks.py"], cwd=CWD, env=ENV)
                sync_to_github()
        time.sleep(30)
        cycle += 1

if __name__ == "__main__":
    main()
