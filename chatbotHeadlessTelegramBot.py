"""Headless Telegram runtime for the Support Chatbot.

This script exposes the same plugin-driven conversation flow used by
`chatbotChatter.py`, but over Telegram using long polling.

Core design:
- Reuses the same handler/state/meta passoff model as terminal runtime.
- Stores one session object per Telegram chat id to avoid cross-user leakage.
- Writes JSONL activity logs for auditing and debugging.
- Masks password input in logs during login password state.

How to run:
    python chatbotHeadlessTelegramBot.py

Token setup (choose one):
1. Environment variable `TELEGRAM_BOT_TOKEN`
2. `.env` file in this folder with a line like:
   TELEGRAM_BOT_TOKEN=123456:ABCDEF...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import runtimeFlowPlugins
from runtimeSubmodules.chatbotNLP import predict_class


DEBUGGING = True
_BASE_DIR = Path(__file__).resolve().parent
_ENV_PATH = _BASE_DIR / ".env"
_HEADLESS_LOG_DIR = _BASE_DIR / "headlessLog"
_DEFAULT_IDLE_TIMEOUT_SECONDS = 600
_TELEGRAM_MAX_TEXT = 3900

# Session state is isolated per Telegram chat id so multiple users can talk to
# the bot simultaneously without sharing login state or quiz progress context.
_SESSIONS: dict[int, dict[str, Any]] = {}


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a simple .env file into os.environ if missing."""
    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # If .env cannot be read, the bot can still run using existing env vars.
        return


def _ensure_log_dir() -> None:
    """Create headless log directory if it does not exist."""
    _HEADLESS_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_activity_log(chat_id: int, user_id: int | None, direction: str, text: str) -> None:
    """Append one JSON line activity record to today's log file.

    Parameters:
    - chat_id: Telegram chat identifier.
    - user_id: Telegram user identifier when available.
    - direction: `received` or `sent`.
    - text: message content after optional masking.
    """
    _ensure_log_dir()
    now_utc = datetime.now(timezone.utc)
    log_path = _HEADLESS_LOG_DIR / f"telegram_{now_utc.strftime('%Y-%m-%d')}.jsonl"

    entry = {
        "timestamp_utc": now_utc.isoformat(),
        "chat_id": chat_id,
        "user_id": user_id,
        "direction": direction,
        "text": text,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logging.warning("Failed to write headless activity log: %s", exc)


def _sanitize_received_text_for_log(session: dict[str, Any], text: str) -> str:
    """Mask sensitive user input when logger is at password-entry state."""
    handler = str(session.get("handler", ""))
    state = str(session.get("state", ""))
    if handler == "LoginHandler" and state == "awaiting_password" and text.strip():
        return "<masked_password>"
    return text


def _split_telegram_text(text: str, max_len: int = _TELEGRAM_MAX_TEXT) -> list[str]:
    """Split long text into Telegram-safe chunks while preferring paragraph breaks."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:max_len]
            split_at = max_len

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


async def _reply_text_safely(update: Update, text: str) -> None:
    """Send long responses in chunks so Telegram length limits do not fail sends."""
    if update.message is None:
        return

    for part in _split_telegram_text(text):
        await update.message.reply_text(part)


def _new_session() -> dict[str, Any]:
    """Create a fresh conversation session mirroring chatbotChatter defaults."""
    return {
        "handler": "LoginHandler",
        "state": "start",
        "meta": {},
        "last_active": time.time(),
    }


def _get_session(chat_id: int) -> dict[str, Any]:
    """Get or initialize a per-chat conversation session."""
    if chat_id not in _SESSIONS:
        _SESSIONS[chat_id] = _new_session()
    return _SESSIONS[chat_id]


def _is_session_idle(session: dict[str, Any]) -> bool:
    """Return True when session has exceeded idle timeout."""
    idle_timeout_seconds = _get_idle_timeout_seconds()
    if idle_timeout_seconds <= 0:
        return False

    last_active = float(session.get("last_active", 0.0))
    if last_active <= 0:
        return False
    return (time.time() - last_active) > idle_timeout_seconds


def _touch_session(session: dict[str, Any]) -> None:
    """Update session heartbeat timestamp to current time."""
    session["last_active"] = time.time()


def _predict_intent(message: str) -> str:
    """Predict the top intent label for a user message."""
    if not message.strip():
        return ""

    intents = predict_class(message)
    return intents[0]["intent"] if intents else "Not Detected"


def _call_flow(
    handler: str,
    state: str,
    meta: dict[str, Any],
    input_text: str,
    predicted_intent: str,
) -> dict[str, Any]:
    """Call one registered runtime plugin and return standardized outcomes."""
    try:
        plugin = runtimeFlowPlugins.require(handler)
        return plugin(state, meta, input_text, predicted_intent)
    except ValueError as exc:
        return {
            "response": f"Error: {exc}",
            "next_handler": handler,
            "next_state": state,
            "meta_update": meta,
        }


def _run_turn(session: dict[str, Any], message: str) -> str:
    """Run one full turn including auto-passoff chaining and state updates."""
    handler = str(session.get("handler", "LoginHandler"))
    state = str(session.get("state", "start"))
    meta = dict(session.get("meta", {}))

    predicted_intent = _predict_intent(message) if message.strip() else ""
    if DEBUGGING:
        logging.info("Predicted intent=%s handler=%s state=%s", predicted_intent, handler, state)

    outcomes = _call_flow(handler, state, meta, message, predicted_intent)
    response_chain: list[str] = []
    if outcomes.get("response"):
        response_chain.append(str(outcomes["response"]))

    # Guard prevents accidental infinite handoff loops if a plugin keeps
    # returning `passoff` without eventually moving to a stable state.
    passoff_guard = 0
    while outcomes.get("next_state") == "passoff" and passoff_guard < 10:
        handler = str(outcomes.get("next_handler") or handler)
        state = str(outcomes.get("next_state") or state)
        meta = dict(outcomes.get("meta_update") or meta)

        passoff_guard += 1
        outcomes = _call_flow(handler, state, meta, "", "")
        if outcomes.get("response"):
            response_chain.append(str(outcomes["response"]))

    if outcomes.get("next_state") == "passoff" and passoff_guard >= 10:
        response_chain.append("Internal handoff loop limit reached.")

    # Persist session for the next incoming message.
    session["handler"] = str(outcomes.get("next_handler") or handler)
    session["state"] = str(outcomes.get("next_state") or state)
    session["meta"] = dict(outcomes.get("meta_update") or meta)
    _touch_session(session)

    if not response_chain:
        return "I am here. Please send a message to continue."
    return "\n\n".join(response_chain)


async def _run_turn_with_typing(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    session: dict[str, Any],
    message: str,
) -> str:
    """Run one turn while periodically showing Telegram typing status."""
    # Run the flow in a worker thread so the event loop can keep sending typing actions.
    task = asyncio.create_task(asyncio.to_thread(_run_turn, session, message))
    while True:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
        except asyncio.TimeoutError:
            # If response generation is still running, keep the typing signal alive.
            continue


async def start_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset session and trigger initial login greeting for this user."""
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user is not None else None
    _append_activity_log(chat_id, user_id, "received", "/start")

    session = _new_session()
    _SESSIONS[chat_id] = session

    first_response = await _run_turn_with_typing(_context, chat_id, session, "")
    _append_activity_log(chat_id, user_id, "sent", first_response)
    await _reply_text_safely(update, first_response)


async def reset_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear session state and start from the login flow again."""
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user is not None else None
    _append_activity_log(chat_id, user_id, "received", "/reset")

    session = _new_session()
    _SESSIONS[chat_id] = session

    first_response = await _run_turn_with_typing(_context, chat_id, session, "")
    reset_response = "Session reset.\n\n" + first_response
    _append_activity_log(chat_id, user_id, "sent", reset_response)
    await _reply_text_safely(update, reset_response)


async def text_message(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages through the plugin conversation pipeline."""
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user is not None else None
    text = update.message.text or ""
    session = _get_session(chat_id)
    safe_text = _sanitize_received_text_for_log(session, text)
    _append_activity_log(chat_id, user_id, "received", safe_text)

    if _is_session_idle(session):
        # If a user comes back after a long pause, restart from login state to
        # avoid resuming stale flow context unexpectedly.
        session = _new_session()
        _SESSIONS[chat_id] = session
        greeting = await _run_turn_with_typing(_context, chat_id, session, "")
        expiry_response = (
            "Session expired due to inactivity. I reset the conversation for you.\n\n"
            + greeting
        )
        _append_activity_log(chat_id, user_id, "sent", expiry_response)
        await _reply_text_safely(update, expiry_response)
        return

    try:
        response = await _run_turn_with_typing(_context, chat_id, session, text)
    except Exception as exc:
        logging.exception("Chat turn failed for chat_id=%s user_id=%s", chat_id, user_id)
        response = (
            "I hit an internal error while generating a reply. "
            "Please try again or type 'exit' and re-enter chat mode."
        )
        _append_activity_log(chat_id, user_id, "sent", f"<internal_error: {exc}>")

    _append_activity_log(chat_id, user_id, "sent", response)
    await _reply_text_safely(update, response)


def _resolve_bot_token() -> str:
    """Resolve Telegram bot token from env/.env and fail with clear guidance."""
    _load_env_file(_ENV_PATH)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        return token

    raise RuntimeError(
        "Missing TELEGRAM_BOT_TOKEN. Add it to environment variables or .env file."
    )


def _get_idle_timeout_seconds() -> int:
    """Resolve idle timeout from env with safe fallback."""
    raw = os.getenv("TELEGRAM_IDLE_TIMEOUT_SECONDS", str(_DEFAULT_IDLE_TIMEOUT_SECONDS)).strip()
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_IDLE_TIMEOUT_SECONDS


def main() -> None:
    """Build Telegram app handlers and start long-polling."""
    logging.basicConfig(
        level=logging.INFO if DEBUGGING else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    idle_timeout_seconds = _get_idle_timeout_seconds()
    if idle_timeout_seconds > 0:
        logging.info("Idle reset enabled at %s seconds", idle_timeout_seconds)
    else:
        logging.info("Idle reset disabled")

    token = _resolve_bot_token()
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    logging.info("Telegram bot started. Waiting for messages...")
    print("Telegram bot is ON and polling for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

