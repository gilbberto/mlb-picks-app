"""
worker.py — Railway background process.
Runs settle_and_notify.py in a continuous loop (30s interval)
for instant Telegram responses and 24/7 auto-settlement.
"""
import subprocess, time, os

CWD = os.path.dirname(os.path.abspath(__file__))
ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"

def setup_git():
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        url = f"https://gilbberto:{token}@github.com/gilbberto/mlb-picks-app.git"
        subprocess.run(["git", "remote", "set-url", "origin", url], cwd=CWD, capture_output=True)
    subprocess.run(["git", "config", "user.name", "MLB Picks Bot"], cwd=CWD, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@mlb-picks.local"], cwd=CWD, capture_output=True)

def git_pull():
    subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], cwd=CWD, capture_output=True)

def git_commit():
    for f in ("picks.json", "game_starts_notified.json", "predictions_log.json", ".morning_sent", ".telegram_offset"):
        fp = os.path.join(CWD, f)
        if os.path.isfile(fp):
            subprocess.run(["git", "add", f], cwd=CWD, capture_output=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=CWD, capture_output=True)
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "sync state"], cwd=CWD, capture_output=True)
        subprocess.run(["git", "pull", "--rebase", "-X", "theirs"], cwd=CWD, capture_output=True)
        subprocess.run(["git", "push"], cwd=CWD, capture_output=True)

def main():
    print("=== Worker iniciado en Railway ===")
    setup_git()
    git_pull()
    cycle = 0
    while True:
        if cycle > 0 and cycle % 20 == 0:
            git_pull()
        subprocess.run(["python3", "settle_and_notify.py"], cwd=CWD, env=ENV)
        git_commit()
        time.sleep(30)
        cycle += 1

if __name__ == "__main__":
    main()
