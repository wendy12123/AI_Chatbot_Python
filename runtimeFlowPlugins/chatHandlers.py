"""Chat flow plugin with local retrieval and optional LLM synthesis.

Design summary:
- Retrieves relevant snippets from local files in `runtimeInfo/ragKnowledge`.
- Synthesizes answers via either an external OpenAI-compatible API (e.g. LM Studio)
  or an in-process Hugging Face model, controlled by `llm_mode` in chatSettings.
- Keeps conversation state in plugin-style outcomes for runtime handoff.
"""

import math
import re
import logging
from pathlib import Path
from typing import Any, cast
import requests
import runtimeFlowPlugins
import yaml
from runtimeSubmodules.chatbotNLP import predict_class
from .welcomeHandlers import get_return_to_menu_message
import torch as _torch
import os
import json


BASEPATH = Path(__file__).resolve().parent.parent
RAG_KNOWLEDGE_ROOT = BASEPATH / "runtimeInfo" / "ragKnowledge"
CHAT_SETTINGS_PATH = BASEPATH / "runtimeInfo" / "chatSettings.yaml"
CHAT_ACTIVE_STATE = "chatting"
CHAT_HOLD_STATE = "hold_before_menu"
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_LLM_CHECKPOINT = "google/gemma-4-E4B"
_LLM_DEVICE = "cpu" if not _torch.cuda.is_available() else "cuda"
_LLM_MAX_INPUT_TOKENS = 1200
_LLM_MAX_NEW_TOKENS = 180
CHAT_MODE_CHOICE_STATE = "awaiting_chat_mode_choice"

_DEFAULT_CHAT_SETTINGS = {
    "exit_intent_check_enabled": True,
    "exit_intent_threshold": 0.6,
    "exit_intent_names": ["exit"],
    "llm_mode": "external",
    "external_llm": {
        "base_url": "http://localhost:1234/v1",
        "model": "",
        "timeout": 60,
        "fail_soft": True,
    },
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

# ✨ 新增 1：通用聊天的 Prompt
def _build_general_prompt(query: str) -> str:
    """Build a simple prompt for general LLM querying without context."""
    return (
        "You are an intelligent AI assistant. Please answer the user's question clearly and concisely.\n\n"
        f"User: {query}\n\n"
        "Assistant:"
    )

# ✨ 新增 2：通用聊天的處理函數
def _build_general_response(query: str) -> str:
    """直接調用外部 API 進行通用回答，不使用本地知識庫。"""
    prompt = _build_general_prompt(query)
    
    # 這裡我們複用你已經寫好的、連接 free.v36.cm 的邏輯
    free_api_key = os.getenv("FREE_CHAT_API_KEY")
    if not free_api_key:
        return "Sorry, the General Chat mode is currently unavailable (API key missing)."

    headers = {
        "Authorization": f"Bearer {free_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _EXTERNAL_LLM_MAX_TOKENS,
    }

    try:
        resp = requests.post("https://free.v36.cm/v1/chat/completions", json=payload, headers=headers, timeout=_EXTERNAL_LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
             return "Sorry, I couldn't generate a response right now."
             
        return content.strip() + "\n\nType 'exit' to return to the main menu."

    except requests.RequestException as exc:
        return f"Network error during general chat: {exc}"

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

_LLM_MODE = str(_CHAT_SETTINGS.get("llm_mode", "internal")).strip().lower()
if _LLM_MODE not in ("external", "internal", "disabled"):
    _LLM_MODE = "internal"

_ext_cfg = _CHAT_SETTINGS.get("external_llm", {})
if not isinstance(_ext_cfg, dict):
    _ext_cfg = {}
_EXTERNAL_LLM_BASE_URL = str(_ext_cfg.get("base_url", "http://localhost:1234/v1")).rstrip("/")
_EXTERNAL_LLM_MODEL = str(_ext_cfg.get("model", ""))
try:
    _EXTERNAL_LLM_TIMEOUT = int(_ext_cfg.get("timeout", 60))
except (TypeError, ValueError):
    _EXTERNAL_LLM_TIMEOUT = 60
_EXTERNAL_LLM_FAIL_SOFT = bool(_ext_cfg.get("fail_soft", False))
try:
    _EXTERNAL_LLM_MAX_TOKENS = int(_ext_cfg.get("max_tokens", 1024))
except (TypeError, ValueError):
    _EXTERNAL_LLM_MAX_TOKENS = 1024


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


_ACRONYM_EXPANSIONS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "ani": "artificial narrow intelligence",
    "agi": "artificial general intelligence",
    "asi": "artificial super intelligence",
    "gpu": "graphics processing unit",
    "tpu": "tensor processing unit",
    "dfs": "depth first search",
    "bfs": "breadth first search",
    "gbfs": "greedy best first search",
    "a*": "a star search",
    "mabp": "multi armed bandit problem",
    "ctr": "click through rate",
    "ann": "artificial neural network",
    "lifo": "last in first out",
    "fifo": "first in first out",
    "relu": "rectified linear unit",
    "gd": "gradient descent",
    "sgd": "stochastic gradient descent",
    "cnn": "convolutional neural network",
    "rgb": "red green blue",
    "nlp": "natural language processing",
    "nltk": "natural language toolkit",
    "bow": "bag of words",
    "oov": "out of vocabulary",
    "rnn": "recurrent neural network",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "svm": "support vector machine",
    "aws": "amazon web services",
    "oop": "object oriented programming",
    "pep": "python enhancement proposal",
    "pep8": "python enhancement proposal 8",
    "cpu": "central processing unit",
    "numpy": "numerical python",
}


def _preprocess_query(query: str) -> str:
    """Expand known acronyms in the query to improve TF-IDF matching."""
    text = (query or "").strip()
    if not text:
        return text
    lower = text.lower()
    expansions = []
    for acronym, expansion in _ACRONYM_EXPANSIONS.items():
        if acronym in lower:
            expansions.append(expansion)
    if expansions:
        return text + " " + " ".join(expansions)
    return text


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


def _chunk_text(text: str, max_lines: int = 20) -> list[str]:
    """Split markdown text on heading boundaries; sub-chunk if still too long."""
    lines = (text or "").splitlines()
    sections: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("##") and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)

    chunks: list[str] = []
    for section in sections:
        non_empty = [ln for ln in section if ln.strip()]
        if len(non_empty) <= max_lines:
            chunk = "\n".join(section).strip()
            if chunk:
                chunks.append(chunk)
        else:
            sub_lines = [ln.strip() for ln in section if ln.strip()]
            cur: list[str] = []
            for ln in sub_lines:
                cur.append(ln)
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
        stem_tokens = set(_tokenize(path.stem.lower()))
        for chunk in _chunk_text(content):
            tokens = _tokenize(chunk)
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            docs.append({"source": rel_path, "chunk": chunk, "tf": tf, "tokens": set(tokens), "stem_tokens": stem_tokens})
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


def _ping_external_llm() -> tuple[bool, str]:
    """Check if the external LLM API is reachable and return status."""
    try:
        resp = requests.get(
            f"{_EXTERNAL_LLM_BASE_URL}/v1/models",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        model_names = [m.get("id", "unknown") for m in models] if isinstance(models, list) else []
        return True, ", ".join(model_names) if model_names else "no models listed"
    except requests.RequestException as exc:
        return False, str(exc)


def _run_startup_preflight() -> None:
    """Capture startup diagnostics for retrieval docs and LLM readiness."""
    _PREFLIGHT_STATUS["rag_docs"] = len(_RAG_DOCS)
    _PREFLIGHT_STATUS["rag_sources"] = len({doc["source"] for doc in _RAG_DOCS})

    if _LLM_MODE == "external":
        reachable, detail = _ping_external_llm()
        _PREFLIGHT_STATUS["llm_ready"] = reachable
        _PREFLIGHT_STATUS["llm_error"] = None if reachable else f"External LLM unreachable: {detail}"
        if reachable:
            logging.info("[ChatHandler] External LLM connected at %s (models: %s)", _EXTERNAL_LLM_BASE_URL, detail)
        elif not _EXTERNAL_LLM_FAIL_SOFT:
            raise RuntimeError(
                f"External LLM unreachable at {_EXTERNAL_LLM_BASE_URL}: {detail}\n"
                f"Ensure LM Studio (or compatible server) is running and accessible.\n"
                f"Set 'fail_soft: true' in chatSettings.yaml to suppress this and fall back to extractive-only."
            )
        else:
            logging.warning(
                "[ChatHandler] External LLM unreachable (%s). Synthesis disabled; extractive fallback active.",
                detail,
            )
    elif _LLM_MODE == "internal":
        tokenizer, model, load_error = _load_local_llm()
        _PREFLIGHT_STATUS["llm_ready"] = tokenizer is not None and model is not None
        _PREFLIGHT_STATUS["llm_error"] = load_error
    else:
        _PREFLIGHT_STATUS["llm_ready"] = False
        _PREFLIGHT_STATUS["llm_error"] = "LLM synthesis disabled (llm_mode=disabled)"


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
    stem_tokens = doc.get("stem_tokens", set())
    if any(tok in stem_tokens for tok in query_tokens):
        score *= 1.5
    return score


def _retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Return top-k retrieved chunks for a query from the local index."""
    q_tokens = _tokenize(_preprocess_query(query))
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
    """Strip prompt echo from causal-model output and return the rest."""
    text = (decoded_text or "").strip()
    if not text:
        return ""
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
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

        # Use the tokenizer's chat template if available (Gemma requires it).
        chat_prompt = prompt
        if hasattr(tok, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError, AttributeError):
                pass

        model_inputs = tokenizer_call(
            chat_prompt,
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
        raw_decoded = str(tokenizer_decode(cast(Any, output_ids)[0], skip_special_tokens=True))
        print(f"[LLM] Raw output (first 200 chars): {raw_decoded[:200]}")
        decoded_text = _clean_llm_answer(raw_decoded, chat_prompt)
        if not decoded_text:
            print(f"[LLM] Cleaning produced empty; returning raw fallback")
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


_REASONING_PATTERNS = [
    (r"Thinking Process:.*?(?=\n\n)", re.DOTALL),
    (r"<think>.*?</think>", re.DOTALL),
]


def _strip_reasoning_artifacts(text: str) -> str:
    """Remove internal reasoning blocks from reasoning-model output."""
    cleaned = text
    for pattern, flags in _REASONING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=flags)
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        for i, line in enumerate(lines):
            if not re.match(r"^\d+\.\s+\*\*", line) and not re.match(r"^\d+\.\s", line):
                return "\n".join(lines[i:]).strip()
    return cleaned.strip()


def _synthesize_with_external_llm(query: str, hits: list[dict]) -> str | None:
    """Generate a concise grounded answer via an OpenAI-compatible external API."""
    prompt = _build_synthesis_prompt(query, hits)

    # 1. 嘗試獲取 free.v36.cm 的 API Key
    free_api_key = os.getenv("FREE_CHAT_API_KEY")
    
    # 2. 決定使用的 API 參數
    if free_api_key:
        # 如果有 Key，使用 free.v36.cm 的免費服務
        api_base_url = "https://free.v36.cm/v1" 
        api_key = free_api_key
        # 使用它支援的免費模型
        model_name = "gpt-4o-mini" 
    else:
        # 如果沒有 Key，退回到你原有的本地 LM Studio 設定
        api_base_url = _EXTERNAL_LLM_BASE_URL
        api_key = "not-needed" # 本地服務通常不需要 key
        model_name = _EXTERNAL_LLM_MODEL

    # 3. 準備請求標頭 (Header)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 4. 現在，將所有邏輯都放在一個 try/ except 區塊中
    try:
        payload: dict[str, Any] = {
            "model": model_name, # 使用我們決定的模型名稱
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _EXTERNAL_LLM_MAX_TOKENS,
        }

        # 發送請求
        resp = requests.post(
            f"{api_base_url}/chat/completions",
            json=payload,
            headers=headers, 
            timeout=_EXTERNAL_LLM_TIMEOUT,
        )
        resp.raise_for_status() # 如果 API 返回錯誤 (如 4xx, 5xx)，這裡會拋出異常
        data = resp.json()

        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")

        if not content or not content.strip():
            content = msg.get("reasoning_content", "")
            
        if content:
            content = _strip_reasoning_artifacts(content)
            
        if not content or not content.strip():
            _set_last_llm_runtime_error("Empty response from external LLM")
            return None

        source_lines = [f"[{idx}] {hit['source']}" for idx, hit in enumerate(hits, start=1)]
        _set_last_llm_runtime_error(None)
        return content.strip() + "\n\nSources:\n" + "\n".join(source_lines) + "\n\nType 'exit' to return to the main menu."
    
    except requests.RequestException as exc:
        _set_last_llm_runtime_error(f"External API request failed: {exc}")
        return None
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _set_last_llm_runtime_error(f"External API response parse error: {exc}")
        return None


def _synthesize(query: str, hits: list[dict]) -> str | None:
    """Dispatch synthesis to the configured LLM backend."""
    if _LLM_MODE == "external":
        return _synthesize_with_external_llm(query, hits)
    if _LLM_MODE == "internal":
        return _synthesize_with_local_llm(query, hits)
    return None


def _build_rag_response(query: str) -> str:
    """Return a final chat reply using retrieval plus optional synthesis."""
    hits = _retrieve(query, top_k=2)
    if not hits:
        return (
            "I could not find strong matching notes in my local knowledge files. "
            "Try rephrasing your question with specific keywords.\n"
            "Type 'exit' to return to the main menu."
        )

    synthesized = _synthesize(query, hits)
    if synthesized:
        return synthesized

    fallback = _build_extractive_response(hits)
    if _LAST_LLM_RUNTIME_ERROR:
        mode_label = "External" if _LLM_MODE == "external" else "Local"
        return (
            fallback
            + "\n\n[Runtime note] " + mode_label + " LLM synthesis failed. "
            + f"Reason: {_LAST_LLM_RUNTIME_ERROR}"
        )
    return fallback

# 註冊這個函數為 "ChatHandler" 插件
@runtimeFlowPlugins.register("ChatHandler")
def chat_handler(state, meta, inputText, _predictedIntent):
    """
    運行聊天模式的一步狀態處理，並返回下一個流程的結果。
    
    參數:
    - state: 當前系統所處的狀態（例如 "passoff", "CHAT_MODE_CHOICE_STATE" 等）。
    - meta: 包含用戶資訊和對話歷史的字典。
    - inputText: 用戶剛剛輸入的文字。
    - _predictedIntent: NLP 模型預測的意圖（在這個 handler 中我們主要依賴自定義的邏輯，所以加上下劃線表示暫不使用）。
    """
    
    # 創建一個 meta 的副本，以防止意外修改原始數據
    next_meta = dict(meta)

    # =========================================================================
    # --- 1. 通用退出指令處理 (優先級最高) ---
    # 在聊天模式的任何狀態下，用戶都有權利隨時退出。
    # =========================================================================
    
    # 如果當前的狀態是 CHAT_HOLD_STATE（通常是在準備退出前的短暫停留狀態）
    if state == CHAT_HOLD_STATE:
        # 發出信號：不產生任何回應文字 ("")，將控制權交給 "WelcomeHandler"，
        # 並且將狀態設定為 "passoff"（這會觸發主選單的歡迎詞）。
        return _outcome(get_return_to_menu_message(), "WelcomeHandler", "passoff", next_meta)

    # 檢查用戶的輸入是否為明確的退出指令（如 "exit", "quit" 等）
    if _is_exit_text(inputText):
        # 如果是，發出退出信號，將狀態切換到 CHAT_HOLD_STATE 準備離開
        return _outcome(f"{get_return_to_menu_message()} (you typed 'exit')", "ChatHandler", CHAT_HOLD_STATE, next_meta)

    # 如果在設定中啟用了「意圖識別退出」功能
    if _EXIT_INTENT_ENABLED:
        # 使用 AI 模型來判斷用戶輸入的是否有退出的意圖，並返回信心分數
        is_exit_intent, confidence, _intent_name = _detect_exit_intent(inputText)
        if is_exit_intent:
            # 如果 AI 判斷用戶想退出（且信心足夠高），則發出退出信號
            return _outcome(f"{get_return_to_menu_message()} (detected exit intent).", "ChatHandler", CHAT_HOLD_STATE, next_meta)


    # =========================================================================
    # --- 2. 核心狀態處理 ---
    # 根據系統當前所處的狀態，執行不同的業務邏輯。
    # =========================================================================
    
    # 狀態: 剛從主選單進入 Chat 模式時的初始狀態
    if state == "passoff":
        # 準備一段歡迎文字，並詢問用戶想要哪種聊天模式
        response = (
            "You are now in chat mode.\n"
            "Would you like a **General Chat** (ask me anything) or a **Course Chat** (questions strictly based on python lecture notes)?\n"
            "(Type 'general' or 'course', or 'exit' to leave)"
        )
        # 返回這段文字，並將狀態推進到 "CHAT_MODE_CHOICE_STATE"（等待用戶選擇模式）
        return _outcome(response, "ChatHandler", CHAT_MODE_CHOICE_STATE, next_meta)

    # ✨ 狀態: 處理用戶對「聊天模式」的選擇
    if state == CHAT_MODE_CHOICE_STATE:
        # 清理用戶的輸入文字，轉為小寫以方便比對
        choice = inputText.strip().lower()
        
        # 如果用戶輸入了包含 "general" 的字眼
        if "general" in choice:
            # 在 meta 中記錄用戶選擇了 "general" 模式
            next_meta["chat_mode"] = "general"
            # 告訴用戶模式已啟動，並將狀態推進到 "CHAT_ACTIVE_STATE"（開始實際聊天）
            return _outcome("General Chat activated. Ask me anything!", "ChatHandler", CHAT_ACTIVE_STATE, next_meta)
            
        # 如果用戶輸入了包含 "course" 或 "deep" 的字眼
        elif "course" in choice or "deep" in choice:
            # 在 meta 中記錄用戶選擇了 "course" 模式
            next_meta["chat_mode"] = "course"
            # 告訴用戶模式已啟動，並將狀態推進到 "CHAT_ACTIVE_STATE"
            return _outcome("Course Chat activated. Ask me about your Python notes!", "ChatHandler", CHAT_ACTIVE_STATE, next_meta)
            
        # 如果用戶輸入了無法識別的內容
        else:
            # 提示用戶重新輸入，並且「保持」在當前的 CHAT_MODE_CHOICE_STATE 狀態
            return _outcome("Please type 'general' or 'course'.", "ChatHandler", CHAT_MODE_CHOICE_STATE, next_meta)

    # ✨ 狀態: 用戶已經選擇了模式，正在進行實際的問答聊天
    if state == CHAT_ACTIVE_STATE:
        # 從 meta 中讀取用戶之前選擇的模式。如果找不到，預設為 "course" 模式。
        current_mode = next_meta.get("chat_mode", "course") 
        
        if current_mode == "general":
            # 如果是通用模式：調用 _build_general_response 函數，
            # 該函數會直接連接外部 LLM API 回答問題，不使用本地筆記。
            response = _build_general_response(inputText)
        else:
            # 如果是課程模式 (預設)：調用原有的 _build_rag_response 函數，
            # 該函數會先檢索本地 Markdown 筆記，然後再交給 LLM 進行總結。
            response = _build_rag_response(inputText)
            
        # 返回 LLM 生成的回應文字，並「保持」在 CHAT_ACTIVE_STATE 狀態，等待用戶的下一個問題
        return _outcome(response, "ChatHandler", CHAT_ACTIVE_STATE, next_meta)

    # =========================================================================
    # --- 3. 備用邏輯 (Fallback) ---
    # 如果系統進入了一個未知的狀態，執行這裡的代碼以防止崩潰。
    # =========================================================================
    
    # 產生一個空的 response，並強制將狀態重置回 "passoff"（即重新詢問用戶要選擇哪種聊天模式）
    return _outcome("", "ChatHandler", "passoff", next_meta)

