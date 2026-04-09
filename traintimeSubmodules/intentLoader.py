"""Intent loader module for chatbot."""    

from pathlib import Path
import yaml


def load_intents():
    """Load intent definitions from YAML files in the runtimeInfo/Intents directory.

    Returns:
        list of dict: A list of intent definitions, where each intent is a dict
        with keys 'name' (intent id), 'patterns' (list of example utterances) and 'responses' (list of example responses)."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Directory containing YAML intent definitions. Each file should be a dict
    # with keys: `name` (intent id), `patterns` (list of example utterances) and `responses` (list of example responses).
    INTENTS_DIR = BASE_DIR / "runtimeInfo" / "Intents"
    corpus = []
    for intent_file in list(INTENTS_DIR.glob("*.yaml")):
        with open(intent_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            # Quick sanity-check: each intent file must be a dict with required keys.
            if not isinstance(data, dict) or "name" not in data or "patterns" not in data or "responses" not in data:
                raise ValueError(
                    f"Incorrect Defintion: Unsupported format in {intent_file}: "
                    "Expected a dictionary with 'name', 'patterns' and 'responses' keys."
                )
            data_name = data.get("name")
            data_patterns = data.get("patterns", [])
            data_responses = data.get("responses", [])
            corpus.append({
                "name": data_name,
                "patterns": data_patterns,
                "responses": data_responses,
            })
    return corpus
