# Artificial Intelligence Search Algorithms Knowledge Base

## 1. Core Definitions and Terms

The following terms establish the environment and components for any search problem:

* **Agent**: An entity that perceives its environment and acts upon it.
* **Environment**: The conditions under which a problem exists (e.g., a map, maze, or game).
* **State**: A configuration of the agent and its environment.
  * **Initial State**: The starting configuration of the agent.
  * **Goal State**: The state in which the problem is solved.
  * **State Space**: The set of all reachable states from the initial state through any sequence of actions.
* **Actions**: Choices available to the agent in a given state.
* **Frontier**: The set of all available states that could be explored next which have not yet been visited.
* **Explored Set**: A set used to track nodes already visited to avoid redundant work and infinite loops.
* **Node**: A data structure tracking a specific state, its parent node, the action applied to reach it, and the path cost.
* **Path Cost**: A numerical value associated with a specific path.
* **Solution**: A sequence of actions leading from the initial state to the goal state.
  * **Optimal Solution**: A solution with the lowest path cost among all possible solutions.

---

## 2. Uninformed Search Algorithms

Uninformed search strategies do not use problem-specific knowledge to find solutions; they only rely on the available actions.

### Depth-First Search (DFS)

* **Definition**: Expands the deepest node in the frontier first.
* **Data Structure**: Uses a **Stack** ("Last-In, First-Out") for the frontier.
* **Performance**:
  * May not always find the optimal solution.
  * Only finds the optimal solution if the agent is "lucky".
  * Prone to wasting time in loops if an "explored set" is not maintained.

### Breadth-First Search (BFS)

* **Definition**: Expands the shallowest node in the frontier first.
* **Data Structure**: Uses a **Queue** ("First-In, First-Out") for the frontier.
* **Performance**: Always finds the optimal solution.

---

## 3. Informed Search Algorithms

Informed search uses problem-specific knowledge, often the distance to the goal, to find solutions more efficiently.

### Heuristic Functions

* **Heuristic ($h(n)$)**: An estimate of the cost from the current node to the goal.
* **Manhattan Distance**: A common heuristic for mazes, calculated by adding the vertical and horizontal distances between a cell and the goal, ignoring walls.

### Greedy Best-First Search (GBFS)

* **Definition**: Expands the node that it estimates is closest to the goal based solely on the heuristic function $h(n)$.
* **Limitation**: Can be inefficient if the heuristic erroneously marks a long path as the "best".

### A* Search

* **Definition**: Expands the node with the lowest value of $f(n)$, where **$f(n) = g(n) + h(n)$**.
  * **$g(n)$**: The cumulative cost to reach the current node.
  * **$h(n)$**: The estimated cost from the current node to the goal.
* **Mechanism**: If the total cost $f(n)$ exceeds a previous option, the algorithm discards the current path and returns to the previous option.
* **Optimality**: To be optimal, $h(n)$ must never overestimate the true cost to the goal.
* **Pros/Cons**: More efficient than GBFS because it accounts for the cost already incurred, though its performance is strictly tied to the quality of the heuristic used.
