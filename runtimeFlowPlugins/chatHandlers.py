import math
import re
from pathlib import Path
from typing import Any
import runtimeFlowPlugins
import yaml
from runtimeSubmodules.chatbotNLP import predict_class
import torch


BASEPATH = Path(__file__).resolve().parent.parent
RAG_KNOWLEDGE_ROOT = BASEPATH / "runtimeInfo" / "ragKnowledge"
CHAT_SETTINGS_PATH = BASEPATH / "runtimeInfo" / "chatSettings.yaml"
CHAT_ACTIVE_STATE = "chatting"
CHAT_HOLD_STATE = "hold_before_menu"
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B"
_LLM_DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
_LLM_MAX_INPUT_TOKENS = 1200
_LLM_MAX_NEW_TOKENS = 180

_DEFAULT_CHAT_SETTINGS = {
    "exit_intent_check_enabled": True,
    "exit_intent_threshold": 0.6,
    "exit_intent_names": ["exit"],
}

_LLM_CACHE = {
    "tokenizer": None,
    "model": None,
    "torch": None,
    "load_attempted": False,
    "load_error": None,
}


def _load_chat_settings() -> dict:
    if not CHAT_SETTINGS_PATH.exists():
        return dict(_DEFAULT_CHAT_SETTINGS)
    try:
        with open(CHAT_SETTINGS_PATH, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
    except OSError:
        return dict(_DEFAULT_CHAT_SETTINGS)
    if not isinstance(parsed, dict):
        return dict(_DEFAULT_CHAT_SETTINGS)

    settings = dict(_DEFAULT_CHAT_SETTINGS)
    settings.update(parsed)
    return settings


_CHAT_SETTINGS = _load_chat_settings()
_EXIT_INTENT_ENABLED = bool(_CHAT_SETTINGS.get("exit_intent_check_enabled", True))
try:
    _EXIT_INTENT_THRESHOLD = float(_CHAT_SETTINGS.get("exit_intent_threshold", 0.6))
except (TypeError, ValueError):
    _EXIT_INTENT_THRESHOLD = 0.6

_raw_exit_names = _CHAT_SETTINGS.get("exit_intent_names", ["exit"])
if isinstance(_raw_exit_names, list):
    _EXIT_INTENT_NAMES = {str(name).strip().lower() for name in _raw_exit_names if str(name).strip()}
else:
    _EXIT_INTENT_NAMES = {"exit"}
if not _EXIT_INTENT_NAMES:
    _EXIT_INTENT_NAMES = {"exit"}


def _outcome(response: str, next_handler: str, next_state: str, meta: dict) -> dict:
    return {
        "response": response,
        "next_handler": next_handler,
        "next_state": next_state,
        "meta_update": meta,
    }


def _is_exit_text(input_text: str) -> bool:
    text = (input_text or "").strip().lower()
    return text in {"exit", "quit", "back", "menu", "stop", "bye"}


def _detect_exit_intent(input_text: str) -> tuple[bool, float, str]:
    try:
        predictions = predict_class(input_text)
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return False, 0.0, ""

    best_name = ""
    best_score = 0.0
    for item in predictions:
        intent_name = str(item.get("intent", "")).strip().lower()
        try:
            confidence = float(item.get("probability", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence > best_score:
            best_score = confidence
            best_name = intent_name

        if intent_name in _EXIT_INTENT_NAMES and confidence >= _EXIT_INTENT_THRESHOLD:
            return True, confidence, intent_name
    return False, best_score, best_name


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _iter_knowledge_files() -> list[Path]:
    files: list[Path] = []
    if not RAG_KNOWLEDGE_ROOT.exists():
        return files

    for path in RAG_KNOWLEDGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".yaml", ".yml", ".txt", ".md"} or path.name.upper() == "README":
            files.append(path)
    return sorted(files)


def _chunk_text(text: str, max_lines: int = 10) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    for line in lines:
        cur.append(line)
        if len(cur) >= max_lines:
            chunks.append("\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _build_index() -> tuple[list[dict], dict[str, float]]:
    docs: list[dict] = []
    df: dict[str, int] = {}

    for path in _iter_knowledge_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_path = str(path.relative_to(BASEPATH))
        for chunk in _chunk_text(content):
            tokens = _tokenize(chunk)
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            docs.append({"source": rel_path, "chunk": chunk, "tf": tf, "tokens": set(tokens)})
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

    n_docs = max(1, len(docs))
    idf = {tok: math.log((1 + n_docs) / (1 + freq)) + 1.0 for tok, freq in df.items()}
    return docs, idf


_RAG_DOCS, _RAG_IDF = _build_index()


def _load_local_llm() -> tuple[Any | None, Any | None, str | None]:
    if _LLM_CACHE["tokenizer"] is not None and _LLM_CACHE["model"] is not None:
        return _LLM_CACHE["tokenizer"], _LLM_CACHE["model"], None
    if _LLM_CACHE["load_attempted"]:
        return None, None, _LLM_CACHE["load_error"]

    _LLM_CACHE["load_attempted"] = True

    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        _LLM_CACHE["torch"] = _torch
        _LLM_CACHE["tokenizer"] = _AutoTokenizer.from_pretrained(_LLM_CHECKPOINT)
        _LLM_CACHE["model"] = _AutoModelForCausalLM.from_pretrained(_LLM_CHECKPOINT)
        return _LLM_CACHE["tokenizer"], _LLM_CACHE["model"], None
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        _LLM_CACHE["load_error"] = str(exc)
        return None, None, _LLM_CACHE["load_error"]


def _score_query(query_tokens: list[str], doc: dict, idf: dict[str, float]) -> float:
    if not query_tokens:
        return 0.0
    tf = doc["tf"]
    doc_tokens = doc["tokens"]
    score = 0.0
    for tok in query_tokens:
        if tok in doc_tokens:
            score += (1.0 + math.log(1 + tf.get(tok, 0))) * idf.get(tok, 1.0)
    return score


def _retrieve(query: str, top_k: int = 2) -> list[dict]:
    q_tokens = _tokenize(query)
    if not q_tokens or not _RAG_DOCS:
        return []

    scored = []
    for doc in _RAG_DOCS:
        score = _score_query(q_tokens, doc, _RAG_IDF)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def _build_extractive_response(hits: list[dict]) -> str:
    lines = ["Here is what I found:"]
    for idx, hit in enumerate(hits, start=1):
        snippet = hit["chunk"]
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."
        lines.append(f"{idx}. Source: {hit['source']}")
        lines.append(snippet)
    lines.append("Type 'exit' to return to the main menu.")
    return "\n\n".join(lines)


def _build_synthesis_prompt(query: str, hits: list[dict]) -> str:
    context_lines = []
    for idx, hit in enumerate(hits, start=1):
        context_lines.append(f"[{idx}] Source: {hit['source']}")
        context_lines.append(hit["chunk"])
        context_lines.append("")

    context_block = "\n".join(context_lines).strip()
    return (
        "You are a course support assistant. Answer the question using only the provided context. "
        "If the context is insufficient, say so clearly and suggest a more specific question. "
        "Keep the answer concise and include source numbers in square brackets like [1], [2].\n\n"
        f"Question: {query}\n\n"
        "Context:\n"
        f"{context_block}\n\n"
        "Answer:"
    )


def _clean_llm_answer(decoded_text: str, prompt: str) -> str:
    text = (decoded_text or "").strip()
    if not text:
        return ""

    # Most reliable case: model output starts with the prompt exactly.
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Fallback: many causal models echo prompt-like blocks; keep only the tail after
    # the last "Answer:" marker so we avoid duplicated Question/Context sections.
    marker = "Answer:"
    if marker in text:
        text = text.rsplit(marker, 1)[1].strip()

    # If the model still starts with prompt section labels, trim up to first sentence-ish line.
    noisy_prefixes = ("Question:", "Context:", "Sources:")
    while any(text.startswith(prefix) for prefix in noisy_prefixes):
        split_idx = text.find("\n\n")
        if split_idx == -1:
            break
        text = text[split_idx + 2 :].strip()

    return text


def _synthesize_with_local_llm(query: str, hits: list[dict]) -> str | None:
    tokenizer, model, _err = _load_local_llm()
    if tokenizer is None or model is None or _LLM_CACHE["torch"] is None:
        return None

    prompt = _build_synthesis_prompt(query, hits)
    try:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_LLM_MAX_INPUT_TOKENS,
        )
        model_inputs = model_inputs.to(_LLM_DEVICE)
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=_LLM_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decoded = _clean_llm_answer(decoded, prompt)

        if not decoded:
            return None

        source_lines = [f"[{idx}] {hit['source']}" for idx, hit in enumerate(hits, start=1)]
        return decoded + "\n\nSources:\n" + "\n".join(source_lines) + "\n\nType 'exit' to return to the main menu."
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return None


def _build_rag_response(query: str) -> str:
    hits = _retrieve(query, top_k=2)
    if not hits:
        return (
            "I could not find strong matching notes in my local knowledge files. "
            "Try rephrasing your question with specific keywords.\n"
            "Type 'exit' to return to the main menu."
        )

    synthesized = _synthesize_with_local_llm(query, hits)
    if synthesized:
        return synthesized
    return _build_extractive_response(hits)


@runtimeFlowPlugins.register("ChatHandler")
def chat_handler(state, meta, inputText, _predictedIntent):
    next_meta = dict(meta)

    if state == "passoff":
        return _outcome(
            "You are now in chat mode. Ask me anything about the available course notes and quiz content. "
            "Type 'exit' any time to return to the main menu.",
            "ChatHandler",
            CHAT_ACTIVE_STATE,
            next_meta,
        )

    if state == CHAT_HOLD_STATE:
        return _outcome("", "WelcomeHandler", "passoff", next_meta)

    if _is_exit_text(inputText):
        return _outcome("Leaving chat mode and returning to main menu.", "ChatHandler", CHAT_HOLD_STATE, next_meta)

    if _EXIT_INTENT_ENABLED:
        is_exit_intent, confidence, _intent_name = _detect_exit_intent(inputText)
        if is_exit_intent:
            return _outcome(
                f"Detected exit intent with confidence {confidence:.2f} (>= {_EXIT_INTENT_THRESHOLD:.2f}). Returning to main menu.",
                "ChatHandler",
                CHAT_HOLD_STATE,
                next_meta,
            )

    if state != CHAT_ACTIVE_STATE:
        return _outcome("", "ChatHandler", "passoff", next_meta)

    response = _build_rag_response(inputText)
    return _outcome(response, "ChatHandler", CHAT_ACTIVE_STATE, next_meta)
