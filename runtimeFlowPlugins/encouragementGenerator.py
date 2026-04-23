"""Encouragement text helper.

Loads encouragement messages from YAML and returns a random message for the
requested context (currently only the generic `any` state is implemented).
"""

from pathlib import Path
import random
import yaml

BASEPATH = Path(__file__).resolve().parent.parent
ENCOURAGEMENT_ANY_FILEPATH = BASEPATH / "runtimeInfo" / "encouragement" / "encouragementsForAny.yaml"


def get_encouragement_on_path(filepath,state, tag) -> str:
    """Load encouragement lines from a YAML file and return one random item."""
    if not filepath.exists():
        return "I can't find a encouragement; but keep going!"

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if state == "any":
        messages = data.get("any", [])
    else:
        category_dict = data.get("tiered", {})
        messages = category_dict.get(tag, [])

    if not messages:
        return "I can't find a encouragement; but keep going!"
    
    return random.choice(messages)

def encouragement_switch(state, tag="any") -> str:
    """Map a lightweight encouragement state key to the right data source."""
    
    # return "I don't have any encouragements for this situation, but keep going!"

    return get_encouragement_on_path(ENCOURAGEMENT_ANY_FILEPATH, state, tag)