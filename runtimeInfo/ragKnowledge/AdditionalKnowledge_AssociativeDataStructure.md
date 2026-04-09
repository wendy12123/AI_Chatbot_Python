# AI Knowledge Base: Associative Data Structures

## 1. Dictionaries (Hash Maps)

Dictionaries store data in **Key-Value pairs**. They are the most efficient way to look up information without searching through an entire dataset.

* **Logic:** Use a unique "Key" to instantly retrieve a "Value."
* **AI Application (NLP):** **Word Indexing**. Mapping unique words to integer IDs so they can be processed by a Neural Network.
* **AI Application (Reinforcement Learning):** **Parameter Tracking**. Storing the $\alpha$ (successes) and $\beta$ (failures) for different choices in Thompson Sampling.

**Python Implementation:**

```python
# NLP Example: Mapping words to IDs
word_index = {"artificial": 1, "intelligence": 2, "search": 3}
print(word_index["artificial"]) # Returns: 1

# Thompson Sampling Example: Tracking 'Arm' statistics
bandit_data = {
    "arm_0": {"alpha": 12, "beta": 5},
    "arm_1": {"alpha": 8, "beta": 14}
}
print(bandit_data["arm_0"]["alpha"]) # Returns: 12
```

---

## 2. Lists (Sequences)

Lists are ordered collections that maintain the sequence of data. In AI, the **order** of data is often as important as the data itself.

* **Logic:** Data is accessed by its numerical position (index), starting at 0.
* **AI Application (NLP):** **Sequencing**. Converting a sentence into a list of integers based on the word index.
* **AI Application (Search):** **Explored Sets**. Keeping a list of all states the agent has already visited to avoid infinite loops.

**Python Implementation:**

```python
# NLP Example: A sentence converted to a sequence
# Original: "Artificial intelligence search"
sequence = [1, 2, 3] 

# Search Example: Tracking explored coordinates in a maze
explored = [(0,0), (0,1), (1,1)]
if (1,2) not in explored:
    print("New state found!")
```

---

## 3. Data Transformation: From Unstructured to Structured

A common workflow in AI datasheets is the transformation of raw text into a format suitable for an Artificial Neural Network (ANN) using both structures:

1. **Raw Text:** `"the car hit the tank"`
2. **Dictionary Lookup:** The system uses a `word_index` dictionary to find IDs.
3. **List Creation:** The system produces a list: `[2, 14, 5, 2, 18]`.
4. **Padding:** If the model requires a length of 6, the list becomes `[2, 14, 5, 2, 18, 0]`.

---

## 4. Comparison Table: Lists vs. Dictionaries

| Feature | List (Sequence) | Dictionary (Mapping) |
| :--- | :--- | :--- |
| **Access Method** | By Index (e.g., `data[0]`) | By Key (e.g., `data["word"]`) |
| **Ordering** | Maintains strict order | Unordered (or ordered by insertion) |
| **Best For...** | Sentences, time-series, paths | Lookups, configurations, stats |
| **AI Context** | Tokens in a sentence, Maze paths | Vocabulary IDs, Agent beliefs |

---

## 5. Advanced Structure: Nested Dictionaries

In complex AI tasks like **Adversarial Search (Minimax)**, nested structures are used to represent the state of a game board or the utility values of different actions.

```python
# Representing a Tic-Tac-Toe state and its Minimax value
game_state = {
    "board": [["X", "O", "X"], [" ", "X", " "], ["O", " ", " "]],
    "utility": 1, # Maximizer is winning
    "best_move": (1, 2)
}
```
