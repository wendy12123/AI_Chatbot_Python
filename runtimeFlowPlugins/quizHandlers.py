from pathlib import Path
import random
from difflib import SequenceMatcher
import yaml
import runtimeFlowPlugins

from .encouragementGenerator import encouragement_switch


BASEPATH = Path(__file__).resolve().parent.parent
QUIZ_DIR = BASEPATH / "runtimeInfo" / "quiz"
USERINFO_DIR = BASEPATH / "runtimeInfo" / "userInfo"
QUIZ_MENU_HOLD_STATE = "hold_before_menu"


def _outcome(response: str, next_handler: str, next_state: str, meta: dict) -> dict:
    return {
        "response": response,
        "next_handler": next_handler,
        "next_state": next_state,
        "meta_update": meta,
    }


def _reset_quiz_runtime(meta: dict) -> dict:
    next_meta = dict(meta)
    next_meta["quiz_active"] = None
    next_meta["quiz_index"] = 0
    next_meta["quiz_score"] = 0
    return next_meta


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=False)


def _quiz_files() -> list[Path]:
    return sorted(QUIZ_DIR.glob("*.yaml"))


def _all_set_metas() -> list[dict]:
    return [_quiz_meta(p) for p in _quiz_files()]


def _quiz_key(path: Path) -> str:
    return path.stem


def _quiz_meta(path: Path) -> dict:
    payload = _load_yaml(path)
    return {
        "key": _quiz_key(path),
        "name": payload.get("name", _quiz_key(path)),
        "description": payload.get("description", "No description."),
        "questions": payload.get("questions", []),
        "path": str(path),
    }


def _user_file(username: str) -> Path:
    return USERINFO_DIR / f"{username}.yaml"


def _get_user_data(username: str) -> dict:
    return _load_yaml(_user_file(username))


def _get_progress(username: str) -> dict:
    progress = _get_user_data(username).get("quiz_progress", {})
    if not isinstance(progress, dict):
        return {}
    return progress


def _is_completed(progress: dict, quiz_key: str) -> bool:
    if not isinstance(progress, dict):
        return False
    entry = progress.get(quiz_key, {})
    return bool(entry.get("completed"))


def _random_not_completed_sets(username: str, count: int = 3) -> list[dict]:
    progress = _get_progress(username)
    metas = _all_set_metas()
    pending = [m for m in metas if not _is_completed(progress, m["key"])]
    if len(pending) <= count:
        return pending
    return random.sample(pending, count)


def _format_set_choices(choices: list[dict]) -> str:
    if not choices:
        return "You have completed all available sets. Type 'all sets' to review your records or 'exit' to return to the menu."
    lines = ["Pick a quiz set by number (or type 'all sets', 'encourage me', or 'exit'):"]
    idx = 1
    for item in choices:
        lines.append(f"{idx}. {item['name']} - {item['description']}")
        idx += 1
    return "\n".join(lines)


def _format_all_sets_status(username: str) -> str:
    progress = _get_progress(username)
    lines = ["All available quiz sets:"]
    for meta in _all_set_metas():
        record = progress.get(meta["key"], {})
        if record.get("completed"):
            score = record.get("score", 0)
            total = record.get("total", 0)
            lines.append(f"- {meta['name']}: completed ({score}/{total})")
        else:
            lines.append(f"- {meta['name']}: not completed")
    return "\n".join(lines)


def _is_exit_text(input_text: str) -> bool:
    text = (input_text or "").strip().lower()
    return text in {"exit", "quit", "back", "menu", "stop"}


def _wants_encouragement(input_text: str, predicted_intent: str) -> bool:
    text = (input_text or "").strip().lower().replace("'", "")
    if predicted_intent == "encouragement":
        return True

    if "encourag" in text or "encorag" in text or "motivat" in text:
        return True

    encourage_phrases = {
        "encourage me",
        "encorage me",
        "give me encouragement",
        "i need encouragement",
        "i need support",
        "pep talk",
    }
    unsure_phrases = {
        "dunno",
        "dont know",
        "do not know",
        "idk",
        "i dont know",
        "i do not know",
        "not sure",
        "unsure",
    }

    for phrase in encourage_phrases:
        if phrase in text:
            return True

    for phrase in unsure_phrases:
        if phrase in text:
            return True

    return False


def _wants_all_sets(input_text: str) -> bool:
    text = (input_text or "").strip().lower()
    return text in {"all", "all sets", "sets", "list", "list sets", "what sets are there", "show sets", "available sets"}


def _normalize_set_token(text: str) -> str:
    lowered = (text or "").strip().lower()
    return "".join(ch for ch in lowered if ch.isalnum())


def _trailing_digits(token: str) -> str:
    digits = []
    for ch in reversed(token):
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    return "".join(reversed(digits))


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _select_choice(input_text: str, choices: list[dict]) -> dict | None:
    text = (input_text or "").strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    lowered = text.lower()
    normalized = _normalize_set_token(text)
    normalized_digits = _trailing_digits(normalized)

    # 1) Exact and alias matches.
    for item in choices:
        key_lower = item["key"].lower()
        name_lower = item["name"].lower()
        if lowered in {key_lower, name_lower}:
            return item
        key_norm = _normalize_set_token(item["key"])
        name_norm = _normalize_set_token(item["name"])
        if normalized in {key_norm, name_norm}:
            return item
        # Accept alias terms like "quiz 1" for an item named "set 1".
        if normalized_digits and normalized_digits in {_trailing_digits(key_norm), _trailing_digits(name_norm)}:
            return item

    # 2) Partial normalized contains match.
    if normalized:
        for item in choices:
            key_norm = _normalize_set_token(item["key"])
            name_norm = _normalize_set_token(item["name"])
            if normalized in key_norm or normalized in name_norm:
                return item

    # 3) Fuzzy match by similarity score with a conservative threshold.
    best_item = None
    best_score = 0.0
    if normalized:
        for item in choices:
            key_norm = _normalize_set_token(item["key"])
            name_norm = _normalize_set_token(item["name"])
            score = max(_similarity(normalized, key_norm), _similarity(normalized, name_norm))
            if score > best_score:
                best_score = score
                best_item = item

    if best_item is not None and best_score >= 0.65:
        return best_item

    return None


def _format_question(question: dict, number: int, total: int) -> str:
    q_text = question.get("question", "")
    q_type = question.get("type", "fill-in-the-blank")
    lines = [f"Question {number}/{total}:", str(q_text)]
    if q_type == "multiple-choice":
        options = question.get("options", [])
        idx = 1
        for opt in options:
            lines.append(f"{idx}. {str(opt)}")
            idx += 1
        lines.append("Reply with option number or text.")
    else:
        lines.append("Reply with your answer.")
    return "\n".join(lines)


def _is_correct_fill_blank(question: dict, input_text: str) -> bool:
    candidates = question.get("answer", [])
    if not isinstance(candidates, list):
        candidates = [candidates]
    answer = (input_text or "").strip()
    mode = question.get("capitalization", "disregard")
    for candidate in candidates:
        expected = str(candidate).strip()
        if mode == "regard":
            if answer == expected:
                return True
        else:
            if answer.casefold() == expected.casefold():
                return True
    return False


def _is_correct_mc(question: dict, input_text: str) -> bool:
    raw_answer = question.get("answer")
    options = question.get("options", [])
    text = (input_text or "").strip()

    if isinstance(raw_answer, int):
        if text.isdigit() and int(text) == raw_answer:
            return True
        idx = raw_answer - 1
        if 0 <= idx < len(options) and text.casefold() == str(options[idx]).strip().casefold():
            return True
        return False

    expected = str(raw_answer).strip()
    if text.casefold() == expected.casefold():
        return True
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(options):
            return str(options[idx]).strip().casefold() == expected.casefold()
    return False


def _is_correct_answer(question: dict, input_text: str) -> bool:
    if question.get("type") == "multiple-choice":
        return _is_correct_mc(question, input_text)
    return _is_correct_fill_blank(question, input_text)


def _format_correct_answer(question: dict) -> str:
    if question.get("type") == "multiple-choice":
        options = question.get("options", [])
        answer = question.get("answer")
        if isinstance(answer, int):
            idx = answer - 1
            if 0 <= idx < len(options):
                return f"Correct answer: {answer}. {options[idx]}"
            return f"Correct answer: option {answer}"
        return f"Correct answer: {answer}"

    answer = question.get("answer", [])
    if isinstance(answer, list):
        if len(answer) == 1:
            return f"Correct answer: {answer[0]}"
        return "Correct answers: " + ", ".join(str(a) for a in answer)
    return f"Correct answer: {answer}"


def _format_wrong_feedback(question: dict) -> str:
    base = "Not quite. " + _format_correct_answer(question)
    reason = question.get("reason")
    if isinstance(reason, str) and reason.strip():
        return base + "\nReason: " + reason.strip()
    return base


def _save_quiz_result(username: str, quiz_key: str, quiz_name: str, score: int, total: int) -> None:
    path = _user_file(username)
    user_data = _load_yaml(path)
    progress = user_data.get("quiz_progress")
    if not isinstance(progress, dict):
        progress = {}
    progress[quiz_key] = {
        "name": quiz_name,
        "completed": True,
        "score": score,
        "total": total,
    }
    user_data["quiz_progress"] = progress
    _save_yaml(path, user_data)


def _start_menu(meta: dict) -> dict:
    username = str(meta.get("username", "")).strip()
    if not username:
        return _outcome(
            "I couldn't find your user profile. Returning to main menu.",
            "WelcomeHandler",
            "passoff",
            meta,
        )

    choices = _random_not_completed_sets(username, count=3)
    all_choices = _all_set_metas()
    next_meta = _reset_quiz_runtime(meta)
    next_meta["quiz_choices"] = choices
    next_meta["quiz_all_choices"] = all_choices
    return _outcome(_format_set_choices(choices), "QuizHandler", "awaiting_set_choice", next_meta)


@runtimeFlowPlugins.register("QuizHandler")
def quiz_handler(state, meta, inputText, predictedIntent):
    next_meta = dict(meta)

    if state == "passoff":
        return _start_menu(next_meta)

    if state == QUIZ_MENU_HOLD_STATE:
        # Wait for one user input before handing off back to main menu.
        return _outcome("", "WelcomeHandler", "passoff", next_meta)

    if state == "awaiting_set_choice":
        username = str(next_meta.get("username", "")).strip()

        if _wants_encouragement(inputText, predictedIntent):
            encouragement = encouragement_switch("any")
            suffix = "\nYou can continue the quiz, ask for all sets, or type exit."
            return _outcome(encouragement + suffix, "QuizHandler", state, next_meta)

        if _wants_all_sets(inputText):
            all_choices = next_meta.get("quiz_all_choices", [])
            if not all_choices:
                all_choices = _all_set_metas()
                next_meta["quiz_all_choices"] = all_choices

            # Switch active menu choices to all sets so completed sets are selectable.
            next_meta["quiz_choices"] = all_choices
            response = _format_all_sets_status(username) + "\n\n" + _format_set_choices(all_choices)
            return _outcome(response, "QuizHandler", "awaiting_set_choice", next_meta)

        choices = next_meta.get("quiz_choices", [])
        selected = _select_choice(inputText, choices)
        if selected is not None:
            total = len(selected.get("questions", []))
            if total == 0:
                return _outcome("That set has no questions yet. Pick another set.", "QuizHandler", "passoff", next_meta)

            next_meta["quiz_active"] = selected
            next_meta["quiz_index"] = 0
            next_meta["quiz_score"] = 0
            first_question = selected["questions"][0]
            response = f"Starting {selected['name']}!\n" + _format_question(first_question, 1, total)
            return _outcome(response, "QuizHandler", "awaiting_answer", next_meta)

        if _is_exit_text(inputText):
            return _outcome("Exiting quiz and returning to main menu.", "QuizHandler", QUIZ_MENU_HOLD_STATE, next_meta)

        if selected is None:
            return _outcome(
                "Please choose one of the displayed set numbers, or type 'all sets'.",
                "QuizHandler",
                "awaiting_set_choice",
                next_meta,
            )

    if _is_exit_text(inputText):
        return _outcome("Exiting quiz and returning to main menu.", "QuizHandler", QUIZ_MENU_HOLD_STATE, next_meta)

    if _wants_encouragement(inputText, predictedIntent):
        encouragement = encouragement_switch("any")
        suffix = "\nYou can continue the quiz, ask for all sets, or type exit."
        return _outcome(encouragement + suffix, "QuizHandler", state, next_meta)

    if state == "awaiting_answer":
        active = next_meta.get("quiz_active")
        if not active:
            return _start_menu(next_meta)

        questions = active.get("questions", [])
        total = len(questions)
        idx = int(next_meta.get("quiz_index", 0))

        if idx >= total:
            return _start_menu(next_meta)

        current_question = questions[idx]
        correct = _is_correct_answer(current_question, inputText)
        if correct:
            next_meta["quiz_score"] = int(next_meta.get("quiz_score", 0)) + 1

        next_idx = idx + 1
        next_meta["quiz_index"] = next_idx

        if next_idx < total:
            if correct:
                feedback = "Correct!"
            else:
                feedback = _format_wrong_feedback(current_question)
            next_question = questions[next_idx]
            response = feedback + "\n\n" + _format_question(next_question, next_idx + 1, total)
            return _outcome(response, "QuizHandler", "awaiting_answer", next_meta)

        username = str(next_meta.get("username", "")).strip()
        final_score = int(next_meta.get("quiz_score", 0))
        _save_quiz_result(username, active["key"], active["name"], final_score, total)

        final_feedback = ""
        if not correct:
            final_feedback = _format_wrong_feedback(current_question) + "\n"

        summary = f"Quiz complete: {active['name']}\nYour score: {final_score}/{total}."
        next_meta = _reset_quiz_runtime(next_meta)
        return _outcome(final_feedback + summary + " Returning to the main menu.", "QuizHandler", QUIZ_MENU_HOLD_STATE, next_meta)

    return _start_menu(next_meta)
