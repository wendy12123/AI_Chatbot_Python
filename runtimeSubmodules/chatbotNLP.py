"""Refactored NLP utilities for the chatbot.

Purpose:
- Provide small, well-documented helpers used by the runtime chatbot: tokenization
  + lemmatization (`clean_up_sentence`), bag-of-words vector creation (`bow`),
  + and intent prediction (`predict_class`).

Notes / usage:
- This module currently loads a Keras model and supporting pickles at import time
  from `../runtimeModels/`. That makes these functions convenient to call
  immediately after `import`, but also means imports will fail if those files
  are missing. Consider converting to lazy loading or a bootstrap function if
  you need training and runtime to coexist without fragile import-order.
"""

from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import datetime
import time
import sys


# Ensure the tokenizer and lexical resources are available. These calls are
# idempotent; if resources are missing NLTK will attempt to download them.
nltk.download('punkt_tab')
nltk.download('wordnet')

# Project layout assumption: this file sits under Support Chatbot/runtimeSubmodules
# and the trained model + pickles are stored in Support Chatbot/runtimeModels
PATH = Path(__file__).parent.parent
runtimeModel_path = PATH / 'runtimeModels'


# Lemmatizer converts words to their base form (e.g. 'running' -> 'run').
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Model + runtime data loading
# ---------------------------------------------------------------------------
# IMPORTANT: these are loaded at import time. If the model or pickles do not
# exist this import will raise. This is convenient for a deployed runtime but
# inconvenient during training or tests where artifacts may not yet be present.
model = load_model(str(runtimeModel_path / 'intent_model.keras'))
with open(runtimeModel_path / 'words.pkl', 'rb') as f:
    words = pickle.load(f)
with open(runtimeModel_path / 'classes.pkl', 'rb') as f:
    classes = pickle.load(f)


def clean_up_sentence(sentence):
    """Tokenize and lemmatize an input sentence.

    Parameters:
    - sentence (str): raw input text from the user.

    Returns:
    - list[str]: a list of normalized tokens (lowercased, lemmatized).

    Implementation notes:
    - Uses NLTK's `word_tokenize` to split text into tokens (handles punctuation
      reasonably well for simple conversational text).
    - Lemmatization reduces inflected forms to a common base so the bag-of-words
      matches regardless of tense/pluralization.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    """Create a bag-of-words numpy vector for `sentence` against vocabulary `words`.

    Parameters:
    - sentence (str): input sentence to vectorize.
    - words (list[str]): vocabulary (ordered) used to build the vector.

    Returns:
    - numpy.ndarray: binary vector of the same length as `words` where 1 indicates
      presence of the vocabulary word in the input sentence.

    Notes:
    - This is a simple binary bag-of-words (presence/absence). It is fast and
      interpretable for small intent classifiers but does not capture order or
      frequency. For better performance consider TF-IDF or embeddings.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    """Predict intent classes for `sentence` using the loaded model.

    Returns a list of intent candidates ordered by probability. Each entry is a
    dict with keys `intent` and `probability` (stringified float from the model).

    Behavior:
    - Converts the sentence into a bag-of-words vector, runs the Keras model,
      filters results below an `ERROR_THRESHOLD`, sorts by confidence, and
      returns a compact list suitable for downstream response selection.
    """
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    # Keep only predictions above the threshold and sort by probability.
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })
    return return_list