# AI Knowledge Base: Effective Matplotlib Visualization

## 1. Environment & Setup

For RAG systems, it is essential to define the environment constraints. In your course, this is primarily **Google Colab**.

* **Inline Rendering:** Always use `%matplotlib inline` to ensure plots are stored within the notebook's cell output.
* **Backend Selection:** For non-interactive environments, use `matplotlib.use('Agg')` before importing `pyplot` to prevent errors related to display GUI requirements.

---

## 2. Standardized Plotting Workflows

A RAG system should be instructed to select plot types based on the **Objective Function** of the AI task:

### A. Training & Optimization (Line Plots)

Used for **Gradient Descent** (Lesson 5) and **ANN/CNN Training** (Lessons 4 & 6).

* **Data:** Epochs ($x$) vs. Loss ($y$).
* **Effective Use:** Plot both *Training Loss* and *Validation Loss* on the same axes to identify **Overfitting**.

### B. Probabilistic Beliefs (Area/Distribution Plots)

Used for **Thompson Sampling** (Lesson 3).

* **Data:** Probability density of the Beta Distribution.
* **Effective Use:** Use `plt.fill_between()` to highlight the area under the curve defined by $\alpha$ (successes) and $\beta$ (failures).

### C. Spatial Search (Heatmaps/Image Plots)

Used for **Maze Solving** (Lesson 2 & 4) and **Computer Vision** (Lesson 6).

* **Data:** 2D NumPy arrays representing grids or pixel intensities.
* **Effective Use:** Use `plt.imshow()` with a binary or grayscale colormap (`cmap='gray'`) to visualize the state space and the agent's path.

---

## 3. The "Datasheet Standard" for Readability

To ensure visualizations are "Effective" (not just present), the following code structure might help:

```python
import matplotlib.pyplot as plt

def create_plot(x, y, title, xlabel, ylabel, label):
    plt.figure(figsize=(10, 6)) # 1. Standardize Size
    plt.plot(x, y, label=label, linewidth=2, marker='o', markersize=4) # 2. Clear Markers
    plt.title(title, fontsize=14, fontweight='bold') # 3. Bold Titles
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7) # 4. Grid for data pinpointing
    plt.legend()
    plt.tight_layout() # 5. Prevent label clipping
    plt.show()
```

---

## 4. Multi-Model Comparisons (Subplots)

When a datasheet needs to compare two algorithms (e.g., **DFS vs. BFS** or **A* vs. GBFS**), use the `subplots` interface to keep the comparison in a single visual frame.

**Implementation Logic:**

* Use `fig, (ax1, ax2) = plt.subplots(1, 2)` for horizontal comparison.
* Share the y-axis (`sharey=True`) if comparing accuracy or loss values to ensure the scale is identical and fair.

---

## 5. Advanced RAG Instruction: "The Interpretive Layer"

An effective datasheet doesn't just show the code; it explains the **visual cues**:

* **The "Elbow":** In a loss plot, the point where the curve flattens indicates the model has converged.
* **Spikes:** Sudden increases in a loss plot suggest the **Learning Rate** is too high, causing the optimizer to "overshoot" the minimum.
* **Distribution Narrowness:** In Thompson Sampling, a "thin/tall" curve represents high confidence in a slot machine's payout, while a "flat/wide" curve represents the need for further **Exploration**.
