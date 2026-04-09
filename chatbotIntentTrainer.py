"""
Train an intent classification model from YAML intent definition files.

This script loads intent definitions from `runtimeInfo/Intents/*.yaml`, tokenizes
and lemmatizes the patterns, builds a bag-of-words representation, trains a
simple feed-forward neural network (Dense layers) to predict intent classes,
and writes runtime artifacts (`words.pkl`, `classes.pkl`, `intent_model.keras`) to
`runtimeModels/`.

Why this design:
- Bag-of-words + one-hot labels: simple, explainable, and sufficient for small
    intent classification tasks where word order and deep context are less
    important.
- Lemmatization: reduces variants (e.g., "running" -> "run") so the model
    needs fewer distinct tokens.
- SGD with Nesterov: chosen for a stable baseline; Adam could be used for
    faster convergence but may require fewer epochs.

How to run:
    python chatbotIntentTrainer.py

Edge cases / notes:
- Intent YAML must contain `name` and `patterns` keys; malformed files raise
    ValueError to avoid silent failures.
- For larger datasets, consider switching to embeddings+RAG or sequence models.
"""

from pathlib import Path
import json
import pickle
import random

import nltk
import numpy as np
import yaml
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

from traintimeSubmodules.intentLoader import load_intents

# Ensure NLTK resources exist. These are quick and safe to call repeatedly.
nltk.download("punkt")
nltk.download("wordnet")

BASE_DIR = Path(__file__).resolve().parent
# Directory containing YAML intent definitions. Each file should be a dict
# with keys: `name` (intent id) and `patterns` (list of example utterances).
INTENTS_DIR = BASE_DIR / "runtimeInfo" / "Intents"
# Directory where trained model and runtime artifacts are stored.
RUNTIME_DIR = BASE_DIR / "runtimeModels"
RUNTIME_DIR.mkdir(exist_ok=True)

lemmatizer = WordNetLemmatizer()

# Load all intent definitions into a simple corpus list of dicts. We validate
# input early so the downstream preprocessing is simpler and errors are clear.
corpus = load_intents()

words = []
classes = []
documents = []
# Characters to ignore during preprocessing.
ignore_words = ["?", "!", ".", ","]

# Tokenize and collect words / document pairs. Each `documents` entry is a
# tuple: (tokenized_pattern_list, intent_name). We also gather the set of
# intent class names.
for intent in corpus:
    name = intent.get("name")
    if not name:
        raise ValueError(f"Incorrect Defintion: Missing 'name' key in intent: {intent}")

    for pattern in intent.get("patterns", []):
        tokenized = nltk.word_tokenize(pattern)
        words.extend(tokenized)
        documents.append((tokenized, name))
        if name not in classes:
            classes.append(name)

# Normalize tokens: lemmatize and lowercase; remove ignore words; build unique
# sorted vocabulary used as the bag-of-words features.
words = sorted(
    set(lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words)
)
classes = sorted(set(classes))

# Persist vocabulary and class list for runtime use by the chat runtime.
with open(RUNTIME_DIR / "words.pkl", "wb") as f:
    pickle.dump(words, f)

with open(RUNTIME_DIR / "classes.pkl", "wb") as f:
    pickle.dump(classes, f)

training = []
output_empty = [0] * len(classes)

# Build bag-of-words vectors for each pattern and the corresponding one-hot
# output vector for the intent class. `training` becomes a list of [bag, out].
for doc in documents:
    bag = []
    # Normalize pattern tokens before checking presence in vocabulary.
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Define a small feed-forward neural network. This is intentionally simple
# because bag-of-words input does not encode sequence information that RNNs
# or transformers are designed to capture.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Use SGD with Nesterov momentum as a stable baseline optimizer.
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model; for larger datasets increase `epochs` or switch optimizer.
hist = model.fit(np.array(train_x), np.array(train_y), epochs=20, batch_size=5, verbose=1)

# Save the trained model for runtime loading by the chatbot.
model.save(RUNTIME_DIR / "intent_model.keras")

ACC_KEY = "accuracy" if "accuracy" in hist.history else "acc"
epochs_range = range(1, len(hist.history[ACC_KEY]) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, hist.history[ACC_KEY], label="Training Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, hist.history["loss"], label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print("Intent model created in runtimeModels")
