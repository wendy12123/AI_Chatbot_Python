# AI Knowledge Base: Data Structures & Algorithmic Logic

## 1. Linear Data Structures: LIFO vs. FIFO

In search algorithms, the way you store and retrieve the "Frontier" (the set of nodes to be explored) determines the search strategy.

### LIFO (Last-In, First-Out) - The Stack

* **Logic:** The most recently added item is the first one removed.
* **AI Application:** Used in **Depth-First Search (DFS)**. It explores as deep as possible down one branch before backtracking.
* **Python Implementation:**

    ```python
    # Using a list as a stack
    stack = []
    stack.append(node_A) # Push A
    stack.append(node_B) # Push B
    # State of stack: [node_A, node_B]
    
    node = stack.pop()   # Removes node_B (Last-In)
    ```

### FIFO (First-In, First-Out) - The Queue

* **Logic:** The oldest added item is the first one removed.
* **AI Application:** Used in **Breadth-First Search (BFS)**. It explores all neighbors at the current depth before moving deeper, ensuring the **optimal (shortest) path** in unweighted graphs.
* **Python Implementation:**

    ```python
    from collections import deque
    
    queue = deque([])
    queue.append(node_A) # Enqueue A
    queue.append(node_B) # Enqueue B
    # State of queue: [node_A, node_B]
    
    node = queue.popleft() # Removes node_A (First-In)
    ```

---

## 2. Associative Structures: Lists & Dictionaries

These are essential for mapping unstructured data (like words) into structured numerical formats that AI models can process.

### Dictionaries (Hash Maps)

* **Usage in NLP:** Mapping unique words to integer IDs (Word Indexing).
  * *Example:* `word_index = {"apple": 1, "banana": 2}`.
* **Usage in Thompson Sampling:** Storing the parameters for different "Arms" or actions.
  * *Example:* `bandit_records = {"ad_v1": {"alpha": 10, "beta": 2}}`.

### Lists (Arrays)

* **Usage in NLP:** Storing **Sequences**. After tokenization, a sentence becomes a list of integers (e.g., `[1, 5, 22, 9]`).
* **Padding:** Since Neural Networks require fixed-size inputs, lists are "padded" with zeros to reach a uniform length.

---

## 3. Multi-Dimensional Structures: Matrices

In Deep Learning, data is rarely a single number; it is organized into grids or tensors.

### 2D & 3D Matrices (Arrays)

* **Grayscale Images:** Represented as a 2D matrix where each element is a pixel value (0–255).
* **Color (RGB) Images:** Represented as a 3D matrix (Height x Width x 3 Channels).
* **AI Application (CNNs):** Convolutional filters (kernels) are small matrices that slide over the image matrix to perform element-wise multiplication and summation.

---

## 4. Logic Summary Table for AI Tasks

| Data Structure | Logic/Property | Specific AI Application |
| :--- | :--- | :--- |
| **Stack** | LIFO | **DFS**: Searching a maze by going deep into one path. |
| **Queue** | FIFO | **BFS**: Finding the shortest path in a maze. |
| **Dictionary** | Key-Value Pair | **NLP Tokenization**: Mapping words to unique IDs. |
| **NumPy Array** | Grid/Matrix | **CNN**: Representing image pixels and filter weights. |
| **Priority Queue** | Sorted by Value | **A\* Search**: Always picking the node with the lowest $f(n)$. |
