from pathlib import Path
import random
import yaml

BASEPATH = Path(__file__).resolve().parent.parent
ENCOURAGEMENT_ANY_FILEPATH = BASEPATH / "runtimeInfo" / "encouragement" / "encouragementsForAny.yaml"


def get_encouragement_on_path(filepath) -> str:
    if not filepath.exists():
        return "I can't find a encouragement; but keep going!"

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    encouragements = data.get("encouragements", [])
    if not encouragements:
        return "I can't find a encouragement; but keep going!"

    return random.choice(encouragements)


def encouragement_switch(state) -> str:
    if state == "any":
        return get_encouragement_on_path(ENCOURAGEMENT_ANY_FILEPATH)
    return "I don't have any encouragements for this situation, but keep going!"