# 10. Reinforcement Learning Interview Preparation 🧠

Comprehensive Q&A and scenario-based questions for RL-focused roles.

## 1. Theoretical Fundamentals 📖

*   **Q: Difference between Off-policy and On-policy learning?**
    - *A:* **On-policy** (e.g., SARSA, PPO) improves the policy that is currently being used to make decisions. **Off-policy** (e.g., Q-Learning, DQN) improves a different policy (the optimal one) regardless of what the agent is currently doing.
*   **Q: What is the Bellman Equation?**
    - *A:* It's a recursive definition of the value function: $V(s) = R(s) + \gamma \sum P(s'|s,a) V(s')$. It breaks the value of a state into the immediate reward and the discounted value of the next state.
*   **Q: What is "Bootstrapping" in TD Learning?**
    - *A:* It means updating an estimate of a value based on other estimates (rather than waiting for the final actual return).

---

## 2. Advanced Deep RL 🧪

*   **Q: Why do we use a Target Network in DQN?**
    - *A:* To prevent the model from chasing a "moving target." If we use the same network to calculate the target and update the values, the training becomes highly unstable and can diverge.
*   **Q: Explain the "Deadly Triad."**
    - *A:* It's the unstable combination of:
        1. **Function Approximation** (Neural Networks).
        2. **Bootstrapping** (TD Learning).
        3. **Off-policy Learning** (Q-Learning).
*   **Q: What is the KL-Divergence penalty in RLHF?**
    - *A:* It ensures the fine-tuned model doesn't drift too far from the original model, preserving the basic language understanding and safety guardrails.

---

## 3. Practical Scenarios 🛠️

*   **Scenario: Your RL agent is getting high rewards but failing the actual task.**
    - *A:* This is likely **Reward Hacking**. The agent found a "shortcut" to get rewards without solving the task. Solution: Redesign the reward function or use Reward Shaping.
*   **Scenario: Your agent works perfectly in a simulator but fails in the real world.**
    - *A:* This is the **Reality Gap**. Simulators are often too smooth or lack noise. Solution: Domain Randomization (varying physics parameters during training).

---

## 🎯 Cheat Sheet: Which Algorithm to Use?
1. **Discrete Actions + High Dim Observation**: DQN / Rainbow.
2. **Continuous Actions + High Stability**: PPO.
3. **Continuous Actions + High Sample Efficiency**: SAC.
4. **Multi-Agent Cooperative**: MAPPO.
5. **Very Sparse Rewards**: Curiosity-driven RL or MCTS.
