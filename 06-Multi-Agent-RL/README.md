# 06. Multi-Agent Reinforcement Learning (MARL) 🤝😈

MARL involves multiple agents interacting within the same environment. Agents may cooperate, compete, or a mix of both.

## 1. Key Challenges in MARL ⚔️

*   **Non-Stationarity**: From the perspective of Agent A, the environment is changing because Agent B is also learning and changing its behavior. Bellman equations assume a stationary environment.
*   **Scalability**: The state-action space grows exponentially with the number of agents.
*   **Credit Assignment**: How to determine which agent contributed most to a shared reward?

---

## 2. Paradigms of MARL 🗺️

### Centralized Training, Decentralized Execution (CTDE)
Agents are trained using data from all other agents, but at execution time, they only use their local observations.
- *Standard:* **MAPPO** (Multi-Agent PPO), **MADDPG**.

### Independent Learning (IQL)
Every agent treats other agents as part of the environment (noise).
- **Pros**: Simple, scales well.
- **Cons**: Often fails due to non-stationarity.

---

## 3. Game Theory Concepts 🃏

*   **Nash Equilibrium**: A state where no agent can benefit by changing its strategy while others keep theirs unchanged.
*   **Zero-Sum Game**: One agent's gain is exactly another's loss (e.g., Chess, Poker).
*   **Cooperative Games**: Agents share a common reward and must coordinate (e.g., Team sports).

---

## 4. Communication in MARL 📡

Some architectures allow agents to send messages (vectors) to each other during training to coordinate complex strategies (e.g., **CommNet**, **DIAL**).

---

## 🛠️ Essential Snippet (Centralized Critic)

```python
# Conceptual Centralized Critic for Actor-Critic MARL
class CentralizedCritic(nn.Module):
    def __init__(self, all_agents_obs, all_agents_actions):
        super().__init__()
        # Input is the concatenation of EVERYONE'S observations and actions
        self.fc = nn.Linear(len(all_agents_obs) + len(all_agents_actions), 1)

    def forward(self, obs_list, action_list):
        x = torch.cat([obs_list, action_list], dim=-1)
        return self.fc(x) # Single value for the global state
```

---

## 📚 Tools for MARL
- **PettingZoo**: The Multi-agent equivalent of Gymnasium.
- **SMAC (StarCraft Multi-Agent Challenge)**: A popular benchmark for cooperative MARL.
- **Mava**: A library for distributed MARL.
