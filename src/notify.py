"""Telegram notifier for DoesItRunDoom?"""

import os
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8798400513:AAHVGh4T2dtsEXZML6zmtXLNLVPM4lpAcZE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "631196199")


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID

    def send(self, message: str, buttons: list = None) -> bool:
        if not self.token or not self.chat_id:
            print(f"[notify] No token/chat_id configured. Message: {message}")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        if buttons:
            payload["reply_markup"] = {
                "inline_keyboard": [[{"text": b["text"], "callback_data": b.get("callback", b["text"])}] for b in buttons]
            }
        try:
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print(f"[notify] Error: {e}")
            return False

    def send_confirmation(self, scenario: str, folder: str, is_existing: bool) -> bool:
        status = "wird weitertrainiert ⚠️" if is_existing else "wird neu erstellt 🆕"
        message = (
            f"🏎️  <b>{scenario}</b>
"
            f"📁 {folder}
"
            f"🧠 {status}
"
            f"
"
            f"Starte Training?"
        )
        buttons = [
            [{"text": "✅ Ja, starte", "callback": f"doom_start_yes_{scenario}"}],
            [{"text": "❌ Nein, abbrechen", "callback": f"doom_start_no_{scenario}"}]
        ]
        return self.send(message, buttons)
