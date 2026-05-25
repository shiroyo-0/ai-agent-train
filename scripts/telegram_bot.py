#!/usr/bin/env python3
"""Telegram bot for Shiro Nb.1.0 - connects to local API."""

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

API = "http://localhost:8080"
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Get from @BotFather


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Shiro Nb.1.0 ready! Send me any message.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = httpx.get(f"{API}/training/status", timeout=5)
    d = r.json()
    if d.get("cycles"):
        l = d["latest"]
        await update.message.reply_text(f"🎓 Training: {d['cycles']} cycles\nLast: {l['examples']} examples, score {l['avg_score']}/10\nModel: {l['model']}")
    else:
        await update.message.reply_text("No training data yet.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    msg = update.message.text

    await update.message.chat.send_action("typing")

    try:
        r = httpx.post(f"{API}/chat", json={"message": msg, "session_id": f"tg_{user_id}"}, timeout=300)
        response = r.json()["response"]
    except Exception as e:
        response = f"⚠️ Error: {e}"

    # Telegram max message length = 4096
    if len(response) > 4000:
        for i in range(0, len(response), 4000):
            await update.message.reply_text(response[i:i+4000])
    else:
        await update.message.reply_text(response)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Telegram bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
