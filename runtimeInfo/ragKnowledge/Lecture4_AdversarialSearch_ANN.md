# AI Knowledge Base: Neural Networks & Adversarial Search

## 1. Artificial Neural Networks (ANNs)

Artificial Neural Networks are computational models inspired by biological brain structures, designed to recognize patterns and model complex, non-linear relationships.

### Core Components of a Neuron

* **Inputs ($x$):** Data features provided to the network.
* **Weights ($w$):** Parameters that determine the importance of each input.
* **Bias ($b$):** An adjustable value that allows the activation function to shift.
* **Activation Function:** Determines the output signal.
  * **ReLU (Rectifier):** $f(x) = \max(0, x)$; the most common function for hidden layers.
  * **Sigmoid:** Maps values to a (0, 1) range; used for probability.
  * **Softmax:** Used in the output layer for multi-class classification.

### Architecture and Learning

* **Layers:** Composed of an Input Layer, multiple Hidden Layers (Deep Learning), and an Output Layer.
* **Forward-Propagation:** The process of passing input data through the network to generate a prediction and calculate the **Loss Error (C)**.
* **Backward-Propagation:** The process of moving backward from the output to update weights and biases using **Gradient Descent** to minimize the error.

---

## 2. Adversarial Search

Adversarial search is used in competitive environments (games) where two or more agents have conflicting goals.

### The Minimax Algorithm

* **Definition:** A recursive algorithm used to choose the optimal move for a player, assuming the opponent is also playing optimally.
* **Players:**
  * **MAX:** Aims to maximize the score/utility.
  * **MIN:** Aims to minimize the score/utility (or maximize the loss for MAX).
* **Process:** The algorithm explores the entire game tree down to terminal states to determine the "value" of each possible move.

### Optimization Techniques

Because game trees can be exponentially large (e.g., Chess or Go), optimizations are required:

1. **Alpha-Beta Pruning:** * A search algorithm that decreases the number of nodes evaluated by the Minimax algorithm in its search tree.
    * It stops evaluating a move as soon as it finds evidence that the move is worse than a previously examined option.
    * **Alpha:** The best value that the Maximizer currently can guarantee.
    * **Beta:** The best value that the Minimizer currently can guarantee.

2. **Depth-Limited Minimax:**
    * The search stops at a predefined depth rather than reaching a terminal state.
    * **Evaluation Function:** Since the game isn't finished, an evaluation function estimates the "utility" or favorability of the current state.

---

## 3. Programming Implementation: maze.py

This section covers the practical application of search algorithms within a Python environment.

* **Node Class:** A data structure that keeps track of:
  * The current **State**.
  * The **Parent** node (to reconstruct the path).
  * The **Action** taken to reach the state.
* **Frontier Classes:**
  * **StackFrontier:** Implements LIFO (Last-In, First-Out) for **Depth-First Search (DFS)**.
  * **QueueFrontier:** Implements FIFO (First-In, First-Out) for **Breadth-First Search (BFS)**.
* **Execution Flow:**
    1. Load maze from a text file.
    2. Identify start and goal.
    3. Initialize the frontier with the starting state.
    4. Repeat: Expand nodes and add neighbors to the frontier until the goal is found or the frontier is empty.
