"""Chat flow plugin with local retrieval and optional local LLM synthesis.

Design summary:
- Retrieves relevant snippets from local files in `runtimeInfo/ragKnowledge`.
- Optionally synthesizes a concise answer using a local Hugging Face model.
- Keeps conversation state in plugin-style outcomes for runtime handoff.
"""

import math
import re
from pathlib import Path
from typing import Any, cast
import runtimeFlowPlugins
import yaml
from runtimeSubmodules.chatbotNLP import predict_class
from .welcomeHandlers import get_return_to_menu_message
import torch as _torch


BASEPATH = Path(__file__).resolve().parent.parent
RAG_KNOWLEDGE_ROOT = BASEPATH / "runtimeInfo" / "ragKnowledge"
CHAT_SETTINGS_PATH = BASEPATH / "runtimeInfo" / "chatSettings.yaml"
CHAT_ACTIVE_STATE = "chatting"
CHAT_HOLD_STATE = "hold_before_menu"
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B"
_LLM_DEVICE = "cpu" if not _torch.cuda.is_available() else "cuda"
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
    "load_attempted": False,
    "load_error": None,
}

_PREFLIGHT_STATUS = {
    "rag_docs": 0,
    "rag_sources": 0,
    "llm_ready": False,
    "llm_error": None,
}

_LAST_LLM_RUNTIME_ERROR: str | None = None


def _model_is_quantized(model: Any) -> bool:
    """Detect whether the loaded model is already managed by a quantized device map."""
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "hf_device_map", None)
    )


def _resolve_model_device(model: Any, fallback: str) -> str:
    """Pick the device where inputs should be placed for the current model."""
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        first_device = next(iter(hf_device_map.values()))
        if isinstance(first_device, str) and first_device:
            return first_device

    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return fallback


def _set_last_llm_runtime_error(msg: str | None) -> None:
    """Store the latest synthesis/runtime error note for fallback reporting."""
    globals()["_LAST_LLM_RUNTIME_ERROR"] = msg


def _load_chat_settings() -> dict:
    """Load chat settings YAML and merge it with safe defaults."""
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
    """Create the standard flow outcome object used by the runtime loop."""
    return {
        "response": response,
        "next_handler": next_handler,
        "next_state": next_state,
        "meta_update": meta,
    }


def _is_exit_text(input_text: str) -> bool:
    """Detect explicit user commands that should leave chat mode."""
    text = (input_text or "").strip().lower()
    return text in {"exit", "quit", "back", "menu", "stop", "bye"}


def _detect_exit_intent(input_text: str) -> tuple[bool, float, str]:
    """Use intent model confidence to infer exit intent from free-form text."""
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
    """Tokenize text into lowercase word tokens for simple retrieval scoring."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _iter_knowledge_files() -> list[Path]:
    """List knowledge files that are eligible for retrieval indexing."""
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
    """Split text into compact line-based chunks for retrieval."""
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
    """Build a lightweight TF-IDF-style index over local knowledge chunks."""
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
    """Lazy-load tokenizer/model for local synthesis and cache the result."""
    if _LLM_CACHE["tokenizer"] is not None and _LLM_CACHE["model"] is not None:
        return _LLM_CACHE["tokenizer"], _LLM_CACHE["model"], None
    if _LLM_CACHE["load_attempted"]:
        return None, None, _LLM_CACHE["load_error"]

    _LLM_CACHE["load_attempted"] = True

    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
        model_kwargs: dict[str, Any] = {}

        if _LLM_DEVICE == "cuda":
            try:
                from transformers import BitsAndBytesConfig as _BitsAndBytesConfig

                model_kwargs["device_map"] = "auto"
                model_kwargs["quantization_config"] = _BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=_torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except (ImportError, ValueError, RuntimeError, TypeError):
                # Fall back to the unquantized path if the quantization stack is unavailable.
                pass

        _LLM_CACHE["tokenizer"] = _AutoTokenizer.from_pretrained(_LLM_CHECKPOINT, **tokenizer_kwargs)
        _LLM_CACHE["model"] = _AutoModelForCausalLM.from_pretrained(_LLM_CHECKPOINT, **model_kwargs)
        _set_last_llm_runtime_error(None)
        return _LLM_CACHE["tokenizer"], _LLM_CACHE["model"], None
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        _LLM_CACHE["load_error"] = str(exc)
        _set_last_llm_runtime_error(f"{type(exc).__name__}: {exc}")
        return None, None, _LLM_CACHE["load_error"]


def _run_startup_preflight() -> None:
    """Capture startup diagnostics for retrieval docs and LLM readiness."""
    # RAG files are already read during module import when _build_index() runs.
    _PREFLIGHT_STATUS["rag_docs"] = len(_RAG_DOCS)
    _PREFLIGHT_STATUS["rag_sources"] = len({doc["source"] for doc in _RAG_DOCS})

    tokenizer, model, load_error = _load_local_llm()
    _PREFLIGHT_STATUS["llm_ready"] = tokenizer is not None and model is not None
    _PREFLIGHT_STATUS["llm_error"] = load_error


_run_startup_preflight()


def _score_query(query_tokens: list[str], doc: dict, idf: dict[str, float]) -> float:
    """Score one document chunk against query tokens using weighted overlap."""
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
    """Return top-k retrieved chunks for a query from the local index."""
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
    """Format retrieval hits directly when synthesis is unavailable."""
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
    """Build a constrained prompt so synthesis only uses retrieved context."""
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
    """Remove prompt echo/noise from raw causal-model decoded output."""
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


def _is_cuda_runtime_error(exc: Exception) -> bool:
    """Detect CUDA runtime failures where retrying on CPU is safer."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cuda error",
        "acceleratorerror",
        "cudaerrorinvalidkernelimage",
        "device kernel image is invalid",
    )
    return any(m in msg for m in markers)


def _synthesize_with_local_llm(query: str, hits: list[dict]) -> str | None:
    """Generate a concise grounded answer with the local causal model."""
    tokenizer, model, _err = _load_local_llm()
    if tokenizer is None or model is None:
        if _err:
            _set_last_llm_runtime_error(str(_err))
        return None

    tok = cast(Any, tokenizer)
    mdl = cast(Any, model)

    if not callable(getattr(tok, "__call__", None)):
        _set_last_llm_runtime_error("Loaded tokenizer is not callable")
        return None
    if not callable(getattr(tok, "decode", None)):
        _set_last_llm_runtime_error("Loaded tokenizer has no callable decode()")
        return None
    if not callable(getattr(mdl, "generate", None)):
        _set_last_llm_runtime_error("Loaded model has no callable generate()")
        return None

    tokenizer_call = cast(Any, getattr(tok, "__call__"))
    tokenizer_decode = cast(Any, getattr(tok, "decode"))
    model_generate = cast(Any, getattr(mdl, "generate"))

    prompt = _build_synthesis_prompt(query, hits)

    def _generate_once(target_device: str) -> str | None:
        # Keep model and input tensors on the same device to avoid runtime
        # errors like "Expected all tensors to be on the same device".
        model_device = _resolve_model_device(mdl, target_device)
        if not _model_is_quantized(mdl):
            mdl.to(model_device)
        mdl.eval()

        model_inputs = tokenizer_call(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_LLM_MAX_INPUT_TOKENS,
        )
        model_inputs = cast(Any, model_inputs).to(model_device)
        
        # Debug: log device placement to verify GPU usage
        try:
            first_param = next(mdl.parameters())
            model_device = str(first_param.device)
        except StopIteration:
            model_device = model_device if isinstance(model_device, str) else "unknown"
        input_device = str(cast(Any, model_inputs).get("input_ids", _torch.tensor([])).device)
        print(f"[LLM] Model device: {model_device}, Input device: {input_device}, Target: {target_device}")
        
        with _torch.inference_mode():
            output_ids = model_generate(
                **model_inputs,
                max_new_tokens=_LLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=getattr(tok, "eos_token_id", None),
            )
        decoded_text = str(tokenizer_decode(cast(Any, output_ids)[0], skip_special_tokens=True))
        decoded_text = _clean_llm_answer(decoded_text, prompt)
        return decoded_text if decoded_text else None

    try:
        decoded = _generate_once(_LLM_DEVICE)
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        # If CUDA runtime is unstable on this machine, retry once on CPU and
        # keep CPU as default for future turns.
        if _LLM_DEVICE == "cuda" and _is_cuda_runtime_error(exc) and not _model_is_quantized(mdl):
            try:
                decoded = _generate_once("cpu")
                globals()["_LLM_DEVICE"] = "cpu"
                _set_last_llm_runtime_error(None)
            except (RuntimeError, ValueError, TypeError, AttributeError) as retry_exc:
                _set_last_llm_runtime_error(f"{type(retry_exc).__name__}: {retry_exc}")
                return None
        else:
            _set_last_llm_runtime_error(f"{type(exc).__name__}: {exc}")
            return None

    if not decoded:
        _set_last_llm_runtime_error("Empty decoded response from local LLM")
        return None

    source_lines = [f"[{idx}] {hit['source']}" for idx, hit in enumerate(hits, start=1)]
    _set_last_llm_runtime_error(None)
    return decoded + "\n\nSources:\n" + "\n".join(source_lines) + "\n\nType 'exit' to return to the main menu."


def _build_rag_response(query: str) -> str:
    """Return a final chat reply using retrieval plus optional synthesis."""
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

    fallback = _build_extractive_response(hits)
    if _LAST_LLM_RUNTIME_ERROR:
        return (
            fallback
            + "\n\n[Runtime note] Local LLM synthesis failed. "
            + f"Reason: {_LAST_LLM_RUNTIME_ERROR}"
        )
    return fallback


@runtimeFlowPlugins.register("ChatHandler")
def chat_handler(state, meta, inputText, _predictedIntent):
    """Run one step of chat-mode state handling and return next flow outcome."""
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
        return _outcome(get_return_to_menu_message(), "WelcomeHandler", "passoff", next_meta)

    if _is_exit_text(inputText):
        return _outcome(
            f"{get_return_to_menu_message()} (you typed 'exit')",
            "ChatHandler",
            CHAT_HOLD_STATE,
            next_meta,
        )

    if _EXIT_INTENT_ENABLED:
        is_exit_intent, confidence, _intent_name = _detect_exit_intent(inputText)
        if is_exit_intent:
            return _outcome(
                f"{get_return_to_menu_message()} (detected exit intent with confidence {confidence:.2f}).",
                "ChatHandler",
                CHAT_HOLD_STATE,
                next_meta,
            )

    if state != CHAT_ACTIVE_STATE:
        return _outcome("", "ChatHandler", "passoff", next_meta)

    response = _build_rag_response(inputText)
    return _outcome(response, "ChatHandler", CHAT_ACTIVE_STATE, next_meta)
