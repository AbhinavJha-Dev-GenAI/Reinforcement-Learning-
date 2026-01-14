# 03. Deep Reinforcement Learning (Deep Q-Learning) 🧠

Deep RL uses Neural Networks as function approximators to solve RL problems with high-dimensional state spaces (like pixels of an Atari game).

## 1. Deep Q-Networks (DQN) 🎮

The model that started it all (DeepMind, 2013). It replaces the Q-table with a Neural Network $Q(s, a; \theta)$.

### Key Innovations:
1.  **Experience Replay**: Stores transitions $(s, a, r, s')$ in a buffer and samples them randomly to break correlations in the data sequence.
2.  **Target Network**: Uses a separate, slowly-updating network $\hat{Q}$ to calculate the TD-target, making the training more stable.

---

## 2. Standard DQN Improvements 🚀

### Double DQN (DDQN)
Reduces the "Overestimation Bias" by using the main network to select the action and the target network to evaluate its value.
$Target = r + \gamma \hat{Q}(s', \text{argmax}_{a'} Q(s', a'; \theta); \theta^{-})$

### Dueling DQN
Splits the Q-network into two streams: one for the **State Value $V(s)$** and one for the **Action Advantage $A(s, a)$**.
$Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a'))$

### Prioritized Experience Replay (PER)
Instead of sampling uniformly, sample transitions with higher TD-error more frequently, as they provide more "learning potential."

---

## 3. Rainbow DQN 🌈

A state-of-the-art agent that combines seven improvements:
- DQN + Double DQN + Dueling DQN + PER + Multi-step Learning + Distributional RL + Noisy Nets.

---

## 4. Challenges in Value-Based Deep RL 🧪
- **The Deadly Triad**: The instability caused by combining **Function Approximation**, **Bootstrapping**, and **Off-policy learning**.

---

## 🛠️ Essential Snippet (Simple DQN Structure)

```python
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)
```

---

## 📊 When to use DQN?
- Discrete action spaces.
- High-vividness state spaces (images/pixels).
- When a model of the environment is not available.
