# AI Knowledge Base: OOP with Python Implementation

## 1. Classes and Objects: The Building Blocks

A **Class** is the blueprint for a data structure, while an **Object** is a specific instance of that blueprint. In AI, classes are used to represent states or agents.

**Python Implementation:**

```python
class Node():
    def __init__(self, state, parent, action, cost=0):
        self.state = state     # e.g., (row, col) coordinates
        self.parent = parent   # Reference to the previous node
        self.action = action   # e.g., "Up", "Down", "Left", "Right"
        self.path_cost = cost  # The g(n) value used in A* search

# Creating specific instances (Objects)
start_node = Node(state=(0,0), parent=None, action=None)
next_node = Node(state=(0,1), parent=start_node, action="Right", cost=1)
```

---

## 2. Inheritance: Specialized Behavior

Inheritance allows a child class to take on the features of a parent class. This is used to create different versions of a search algorithm while keeping the core "Frontier" logic the same.

**Python Implementation:**

```python
class Frontier():
    def __init__(self):
        self.cells = []
    def add(self, node):
        self.cells.append(node)
    def contains_state(self, state):
        return any(node.state == state for node in self.cells)

# StackFrontier inherits from Frontier but adds LIFO behavior for DFS
class StackFrontier(Frontier):
    def remove(self):
        if len(self.cells) == 0:
            raise Exception("Empty frontier")
        else:
            node = self.cells[-1] # Take the last item (LIFO)
            self.cells = self.cells[:-1]
            return node

# QueueFrontier inherits from Frontier but adds FIFO behavior for BFS
class QueueFrontier(Frontier):
    def remove(self):
        if len(self.cells) == 0:
            raise Exception("Empty frontier")
        else:
            node = self.cells[0] # Take the first item (FIFO)
            self.cells = self.cells[1:]
            return node
```

---

## 3. Encapsulation: Protecting Model State

Encapsulation bundles data (attributes) and methods together. In Reinforcement Learning, it prevents external code from manually changing an agent's success counts without a proper trial.

**Python Implementation:**

```python
class ThompsonAgent:
    def __init__(self, num_arms):
        self.__alpha = [1] * num_arms # Private variable (Double underscore)
        self.__beta = [1] * num_arms  # Private variable

    def update_belief(self, arm_index, reward):
        if reward == 1:
            self.__alpha[arm_index] += 1
        else:
            self.__beta[arm_index] += 1

    def get_parameters(self, arm_index):
        return self.__alpha[arm_index], self.__beta[arm_index]
```

---

## 4. Polymorphism: Shared Interfaces

Polymorphism allows different classes to be accessed via the same method name. For example, in NLP, different models (Sentiment vs. Classification) might both have a `.predict()` method, even though their inner math is different.

**Python Implementation:**

```python
# Both models share the same "interface" (the .predict method)
class SentimentModel:
    def predict(self, text):
        return "Positive" if "happy" in text else "Negative"

class SpamModel:
    def predict(self, text):
        return "Spam" if "win money" in text else "Ham"

# A loop that doesn't care which model it is using
models = [SentimentModel(), SpamModel()]
for m in models:
    print(m.predict("I am happy to win money"))
```

---

## 5. Summary of Why We Use OOP in AI

1. **Complexity Management:** It is easier to write `model.add(Dense(10))` (Keras OOP) than to manually calculate the matrix multiplication and gradient descent updates.
2. **Scalability:** By using classes like `Node`, we can expand a search tree to millions of states by simply creating more objects.
3. **Standardization:** Using common OOP patterns (like the `role` and `content` structure in Chatbots) allows different AI systems to communicate with each other easily.
