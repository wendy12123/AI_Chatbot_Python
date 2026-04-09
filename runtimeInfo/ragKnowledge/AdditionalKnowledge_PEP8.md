This knowledge sheet serves as a technical reference for a Retrieval-Augmented Generation (RAG) system, focusing on **PEP 8**, the official Style Guide for Python Code. Adhering to these standards ensures that AI-generated code remains readable, maintainable, and consistent with professional academic and industry practices.

# AI Knowledge Base: PEP 8 Python Coding Standards

## 1. Why Use PEP 8?

PEP stands for **Python Enhancement Proposal**. PEP 8 is the specific document that provides guidelines and best practices on how to write Python code.

* **Readability:** Code is read much more often than it is written.
* **Consistency:** Allows different programmers (or AI agents) to collaborate on the same project (e.g., your `maze.py` or CNN models) without confusion.
* **Maintainability:** Standardized code is easier to debug and update over time.

---

## 2. Naming Conventions

Naming is one of the most critical aspects of PEP 8. The style changes depending on what is being named.

| Identifier | Convention | Example |
| :--- | :--- | :--- |
| **Class** | PascalCase (CapWords) | `class StackFrontier:` |
| **Function** | snake_case | `def get_weights():` |
| **Variable** | snake_case | `current_state = (0, 0)` |
| **Constant** | UPPER_SNAKE_CASE | `MAX_EPOCHS = 100` |
| **Private Internal** | Leading Underscore | `self._internal_value = 10` |

---

## 3. Code Layout & Indentation

Python uses whitespace to define logical blocks. PEP 8 strictly defines how this should be handled to prevent "IndentationErrors."

### Indentation

* **Rule:** Use **4 spaces** per indentation level.
* **Avoid Tabs:** Never mix tabs and spaces. Most modern IDEs (like Google Colab) automatically convert tabs to 4 spaces.

### Maximum Line Length

* **Rule:** Limit all lines to a maximum of **79 characters**.
* **Reason:** This allows multiple files to be open side-by-side and prevents horizontal scrolling.

### Blank Lines

* **Top-level functions and classes:** Surround with **two blank lines**.
* **Method definitions inside a class:** Surround with **one blank line**.

---

## 4. Whitespace in Expressions and Statements

Avoid extraneous whitespace in the following situations:

* **Immediately inside brackets:** * *Incorrect:* `spam( ham[ 1 ], { eggs: 2 } )`
  * *Correct:* `spam(ham[1], {eggs: 2})`
* **Before a comma or semicolon:**
  * *Incorrect:* `if x == 4 : print(x , y)`
  * *Correct:* `if x == 4: print(x, y)`
* **Assignment operators:** Always surround operators with a single space on either side.
  * *Correct:* `x = 1`, `y += 5`, `i = i + 1`

---

## 5. Programming Recommendations

### Comparisons to Singletons

When checking if a variable is `None`, always use `is` or `is not`, never the equality operators.

```python
# Correct
if node.parent is None:
    return path

# Incorrect
if node.parent == None:
    return path
```

### Imports

Imports should always be at the top of the file, grouped in the following order:

1. Standard library imports (e.g., `import os`, `import math`).
2. Related third-party imports (e.g., `import matplotlib.pyplot as plt`, `import numpy as np`).
3. Local application/library specific imports (e.g., `from frontier import StackFrontier`).

**Note:** Use absolute imports over relative imports where possible.

---

## 6. Implementation Example (PEP 8 Compliant)

Below is a snippet of a compliant class structure, similar to those found in your AI coursework.

```python
import numpy as np  # Third-party import


class ModelEvaluator:
    """Evaluates the performance of an AI model."""

    def __init__(self, target_accuracy=0.95):
        self.target_accuracy = target_accuracy
        self.results = []

    def log_result(self, epoch, accuracy):
        # 4-space indentation used here
        if accuracy >= self.target_accuracy:
            status = "PASSED"
        else:
            status = "FAILED"
            
        self.results.append((epoch, status))
```
