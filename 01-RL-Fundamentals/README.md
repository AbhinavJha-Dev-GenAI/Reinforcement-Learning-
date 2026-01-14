# 01. Reinforcement Learning Fundamentals 🎮

Core concepts that define the framework of Reinforcement Learning.

## 1. Markov Decision Process (MDP) 🧠

An MDP provides a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

### Components:
*   **State ($S$):** A set of states the environment can be in.
*   **Action ($A$):** A set of actions the agent can take.
*   **Transition Probability ($P$):** The probability $P(s' | s, a)$ of moving to state $s'$ from state $s$ by taking action $a$.
*   **Reward ($R$):** The immediate reward $R(s, a, s')$ received after transitioning.
*   **Discount Factor ($\gamma$):** A value between 0 and 1 that determines the importance of future rewards (0 = shortsighted, 1 = farsighted).

---

## 2. Policy ($\pi$) and Value Functions 🛤️

### Policy ($\pi$)
The strategy that the agent uses to decide the next action based on the current state.
- **Deterministic**: $a = \pi(s)$
- **Stochastic**: $\pi(a | s) = P(A_t = a | S_t = s)$

### Value Functions
*   **State-Value Function $V_{\pi}(s)$:** Expected return starting from state $s$ and following policy $\pi$.
*   **Action-Value Function $Q_{\pi}(s, a)$:** Expected return starting from state $s$, taking action $a$, and then following policy $\pi$.

---

## 3. The Bellman Equation ⚖️

The fundamental recursive relationship for value functions.

### Bellman Expectation Equation:
$V_{\pi}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S, r \in R} p(s', r|s, a) [r + \gamma V_{\pi}(s')]$

> [!IMPORTANT]
> The goal of RL is to find the **Optimal Policy** $\pi^*$ that maximizes the expected return.

---

## 4. Exploration vs. Exploitation ⚖️

*   **Exploration**: Trying out new actions to discover more about the environment.
*   **Exploitation**: Using the current knowledge to take the best action and maximize immediate reward.

### Common Strategy: $\epsilon$-Greedy
With probability $\epsilon$, choose a random action (explore). With probability $1-\epsilon$, choose the best action (exploit).

---

## 🛠️ Essential Snippet (MDP logic)

```python
# Simplified MDP Transition
import numpy as np

def step(state, action, P, R):
    # P: Transition matrix [S, A, S']
    # R: Reward matrix [S, A]
    next_state = np.random.choice(len(P[state, action]), p=P[state, action])
    reward = R[state, action]
    return next_state, reward

# Simple Epsilon-Greedy
def select_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values)) # Explore
    else:
        return np.argmax(q_values) # Exploit
```
