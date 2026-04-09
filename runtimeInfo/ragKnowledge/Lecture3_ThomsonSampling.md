# Reinforcement Learning: Multi-Armed Bandit & Thompson Sampling

## 1. The Multi-Armed Bandit Problem (MABP)

The Multi-Armed Bandit Problem is a classic framework in Reinforcement Learning that illustrates the fundamental trade-off between **exploration** and **exploitation**.

### The Environment

* **Analogy**: An agent is in a casino with multiple slot machines (one-armed bandits).
* **The Setup**: Each machine has an underlying, unknown probability of delivering a reward.
* **The Reward Structure**:
  * **Success (+1)**: The machine returns a win.
  * **Failure (-1 or 0)**: The machine takes the money without a win.
* **The Goal**: Maximize the total reward over a series of trials (rounds).

### Exploration vs. Exploitation

* **Exploration**: Trying different machines to gather more information about their payout probabilities.
* **Exploitation**: Repeatedly pulling the arm of the machine that has provided the highest rewards so far to maximize short-term gain.

---

## 2. Thompson Sampling

Thompson Sampling is a sophisticated algorithm used to solve the Multi-Armed Bandit Problem by using a probabilistic approach.

### Core Mechanism: The Beta Distribution

Thompson Sampling uses the **Beta Distribution** to represent the agent's belief about the probability of success for each machine.

* **Shape Parameters**:
  * $\alpha$ (Alpha): Represents the number of times a reward of **1** was received.
  * $\beta$ (Beta): Represents the number of times a reward of **0** was received.
* **Behavior**: As more trials are conducted, the distribution for each machine narrows, reflecting increased confidence in the true payout probability.

### The Algorithm Steps

1. **Sample**: For each machine $i$, pick a random number $\theta_i$ from its current Beta distribution.
2. **Select**: Choose the machine with the highest sampled value $\theta_i$.
3. **Observe**: Pull the arm and observe the actual reward ($0$ or $1$).
4. **Update**: Adjust the $\alpha$ or $\beta$ parameters for that specific machine based on the result:
    * If reward = 1: $N_i^1(n) = N_i^1(n) + 1$
    * If reward = 0: $N_i^0(n) = N_i^0(n) + 1$

---

## 3. Practical Application: Online Advertising

The Multi-Armed Bandit framework is widely used in digital marketing to optimize click-through rates (CTR).

* **Scenario**: A company has five different versions of an ad (the "arms").
* **Objective**: Determine which ad version is most effective.
* **Process**:
    1. Each time a user visits a site, Thompson Sampling selects an ad to display.
    2. If the user clicks (Reward = 1), the model updates the "success" count for that ad.
    3. If the user does not click (Reward = 0), the model updates the "failure" count.
* **Outcome**: Over time, the algorithm identifies the ad with the highest CTR and displays it more frequently, while still occasionally exploring others to ensure the data remains current.

---

## 4. Comparison: Thompson Sampling vs. Traditional A/B Testing

* **A/B Testing**: Usually requires a fixed period of exploration (showing all ads equally) before switching entirely to the winner. This results in "regret" (lost potential revenue) during the testing phase.
* **Thompson Sampling**: Updates in real-time. It shifts traffic toward the better-performing ads dynamically, reducing the cost of exploration and maximizing cumulative rewards much faster.
