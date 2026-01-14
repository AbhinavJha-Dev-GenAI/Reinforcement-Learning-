# 05. Model-Based Reinforcement Learning 🏗️

Model-based RL agents attempt to learn a model of the environment's dynamics (transitions and rewards) to plan their actions without interacting with the real world constantly.

## 1. Learning the World Model 🌍

A world model typically consists of:
*   **Transition Model**: $P(s' | s, a) \approx \hat{P}(s' | s, a)$.
*   **Reward Model**: $R(s, a) \approx \hat{R}(s, a)$.

### Why use Model-Based?
- **Sample Efficiency**: Learning from a model (hallucinating) is much faster than interacting with a slow or expensive physical environment (e.g., Robotics).

---

## 2. Dyna-Q Architecture 🔄

Dyna-Q integrates **learning** (from real experience), **modeling** (updating the system world model), and **planning** (using the model to update the value function).
- *Cycle:* Real Experience → Model Update → Background Planning → Policy Update.

---

## 3. Planning with Search 🔍

### Monte Carlo Tree Search (MCTS)
The algorithm behind **AlphaGo**. It builds a search tree by simulating many possible futures and following the paths that look most promising.
1.  **Selection**: Traverse down the tree using UCB.
2.  **Expansion**: Add a new child node.
3.  **Simulation (Rollout)**: Run a random policy to the end of the episode.
4.  **Backpropagation**: Update the value of nodes along the path.

---

## 4. Modern SOTA: Dreamer & World Models 🛌

*   **Dreamer (v1/v2/v3)**: Trains a latent dynamics model and learns behaviors purely by imagining trajectories within that model.
*   **MuZero**: A successor to AlphaZero that learns a model of the dynamics and then applies MCTS to that learned model, achieving superhuman performance without knowing the rules of the game.

---

## 🛠️ Essential Snippet (Simple Planning Loop)

```python
# Conceptual Dyna-Q Planning
def dyna_q_planning(model, q_table, n_steps):
    for _ in range(n_steps):
        # Sample a previously visited state and action
        s, a = sample_visited_history()
        
        # Consult the learned model
        s_next, r = model[s][action]
        
        # Update Q-table using the 'hallucinated' experience
        update_q_table(s, a, r, s_next)
```

---

## ⚖️ Model-Based vs Model-Free

| Feature | Model-Free (PPO/DQN) | Model-Based (MuZero/Dreamer) |
| :--- | :--- | :--- |
| **Complexity** | Simple to implement | High complexity |
| **Optimality** | Converges to optimal | Limited by model accuracy |
| **Sample Efficiency** | Low | High |
| **Computation** | Low at runtime | High during planning/search |
