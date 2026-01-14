# 04. Policy Gradient Methods 🎯

Directly optimizing the policy $\pi(a|s)$ instead of learning a value function. This is essential for **Continuous Action Spaces**.

## 1. REINFORCE (Basic Policy Gradient) 📈

The policy is parameterized by $\theta$. We update $\theta$ to maximize the expected return using the Gradient Ascent.
$\nabla_{\theta} J(\theta) \approx \sum_{t} \nabla_{\theta} \log \pi(a_t | s_t) G_t$
- *Intuition:* Make "good" actions (those with high return $G_t$) more likely.

---

## 2. Actor-Critic Architectures (A2C / A3C) 🎭

Directly optimizing the policy can have high variance. Actor-Critic methods fix this:
*   **Actor**: Learns the policy $\pi(a|s)$.
*   **Critic**: Learns the value function $V(s)$ to evaluate the actor's actions.
*   **Advantage Function**: $A(s, a) = Q(s, a) - V(s)$. It tells us if an action was *better than average*.

---

## 3. PPO (Proximal Policy Optimization) 🛡️

The industry standard for RL (OpenAI's favorite). It solves the problem of "large policy updates" destroying performance.
*   **Clipping**: It clips the policy update to be within a small range (e.g., $[0.8, 1.2]$) ensuring the new policy doesn't deviate too much from the old one.

---

## 4. SAC (Soft Actor-Critic) 🏺

An off-policy actor-critic algorithm that uses **Entropy Regularization**.
*   **Objective**: Maximize Reward + Entropy.
*   **Goal**: Encourages the agent to explore more and find multiple ways to solve a task. Best for **Continuous Control** (Robotics).

---

## 🛠️ Essential Snippet (A2C Logic)

```python
import torch
import torch.nn.functional as F

# Loss computation for Actor-Critic
def compute_loss(probs, values, returns):
    # Advantage = Return - Baseline (Value)
    advantage = returns - values
    
    # Actor Loss: Negative Log Prob * Advantage
    actor_loss = -(torch.log(probs) * advantage.detach()).mean()
    
    # Critic Loss: MSE between Value and Return
    critic_loss = F.mse_loss(values, returns)
    
    return actor_loss + 0.5 * critic_loss
```

---

## ⚖️ Comparison: Policy-Based vs Value-Based

| Feature | Value-Based (DQN) | Policy-Based (PPO/SAC) |
| :--- | :--- | :--- |
| **Action Space** | Discrete Only | Discrete & Continuous |
| **Stability** | Potential for oscillate | More stable (with trust regions) |
| **Sample Efficiency** | Higher (Off-policy) | Lower (On-policy - PPO) |
| **Exploration** | Local (Epsilon-greedy) | Global (Stochastic Policy) |
