"""
webhook.py — Telegram webhook handler for instant /resultados response.
Railway expone este servicio como HTTPS para que Telegram lo llame.
"""
import os, sys, hashlib
from flask import Flask, request

sys.path.insert(0, os.path.dirname(__file__))
from settle_and_notify import _build_resultados_response, send_telegram, _cmd_status, _cmd_restart

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
WEBHOOK_PATH = hashlib.sha256(TELEGRAM_TOKEN.encode()).hexdigest()[:16]

@app.route(f"/webhook/{WEBHOOK_PATH}", methods=["POST"])
def webhook():
    update = request.json
    msg = (update or {}).get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    text = (msg.get("text") or "").strip().lower()

    if chat_id and str(chat_id) == CHAT_ID:
        if text in ("resultados", "/resultados"):
            resp = _build_resultados_response()
            send_telegram(resp)
        elif text in ("/status", "/diagnostico"):
            send_telegram(_cmd_status(text == "/diagnostico"))
        elif text == "/reiniciar_worker":
            send_telegram(_cmd_restart("449aff70-31f8-4e67-88ab-c6ccedcc1546", "Worker"))
        elif text == "/reiniciar_web":
            send_telegram(_cmd_restart("e5af2645-5349-4176-a305-419ce60353da", "Web"))
        elif text == "/reiniciar_webhook":
            send_telegram(_cmd_restart("e722f196-dd7f-48f9-9654-2c9335ad0c0f", "Webhook"))

    return {"ok": True}

@app.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
