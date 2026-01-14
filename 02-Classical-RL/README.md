# 02. Classical Reinforcement Learning 🏛️

Solving MDPs using tabular methods when the state and action spaces are small.

## 1. Dynamic Programming (DP) 🔄

Used when the model of the environment (Transitions and Rewards) is fully known.

*   **Policy Iteration**: Alternates between Policy Evaluation (calculate $V$ for fixed $\pi$) and Policy Improvement (make $\pi$ greedy w.r.t $V$).
*   **Value Iteration**: Directly updates $V$ using the Bellman Optimality Equation until convergence, then extracts the policy.

---

## 2. Monte Carlo (MC) Methods 🎲

Learning from complete episodes of experience. No model of the environment is required.
*   **Pros**: Simple, unbiased.
*   **Cons**: High variance, only works for episodic tasks.

---

## 3. Temporal Difference (TD) Learning ⏳

The most central idea in RL. Combines MC and DP. Learns from incomplete episodes by updating estimates based on other estimates (Bootstrapping).

### SARSA (On-Policy)
Updates $Q(s, a)$ based on the action $a'$ actually taken by the current policy.
$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$

### Q-Learning (Off-Policy)
Updates $Q(s, a)$ based on the *best possible* future action, regardless of the policy being followed.
$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

---

## 4. Multi-Armed Bandits 🎰

Simplest form of RL with a single state.
*   **Upper Confidence Bound (UCB)**: Selects actions based on their potential (uncertainty).
*   **Thompson Sampling**: A Bayesian approach that maintains a probability distribution for each arm's reward.

---

## 🛠️ Essential Snippet (Q-Learning)

```python
import numpy as np

# Q-Table initialization
q_table = np.zeros([state_space_size, action_space_size])

# Q-Learning update
def update_q_table(state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
```

---

## ⚖️ Comparison: SARSA vs Q-Learning

| Feature | SARSA | Q-Learning |
| :--- | :--- | :--- |
| **Policy Type** | On-Policy | Off-Policy |
| **Behavior** | Conservative (Avoids traps) | Optimistic (Takes shortcuts) |
| **Convergence** | Continuous exploration | To the optimal policy |
| **Best For** | Safety-critical tasks | Finding the shortest path |
