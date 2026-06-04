"""
webhook.py — Telegram webhook handler for instant /resultados response.
Railway expone este servicio como HTTPS para que Telegram lo llame.
"""
import os, sys
from flask import Flask, request

sys.path.insert(0, os.path.dirname(__file__))
from settle_and_notify import _build_resultados_response, send_telegram

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

@app.route(f"/webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = request.json
    msg = (update or {}).get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    text = (msg.get("text") or "").strip().lower()

    if chat_id and str(chat_id) == CHAT_ID and text in ("resultados", "/resultados"):
        resp = _build_resultados_response()
        send_telegram(resp)

    return {"ok": True}

@app.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
