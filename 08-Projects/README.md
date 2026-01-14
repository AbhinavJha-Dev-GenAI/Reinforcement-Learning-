# 08. Reinforcement Learning Projects 🛠️

Hands-on projects to solidify your understanding of RL algorithms.

## Project 1: Classic Control (Taxi-v3) 🚕
*   **Goal**: Solve the Gymnasium "Taxi" environment where an agent must pick up and drop off passengers.
*   **Tech Stack**: Gymnasium, NumPy.
*   **Key Learning**:
    - Implementing Q-Learning and SARSA from scratch.
    - Handling discrete state-action spaces.
    - Visualizing the Q-table.

## Project 2: Atari Pong with DQN 🏓
*   **Goal**: Train an agent to beat the built-in AI in Atari Pong.
*   **Tech Stack**: PyTorch, Stable-Baselines3, Gym-Atari.
*   **Key Learning**:
    - Processing image frames (CNNs in RL).
    - Experience Replay and Target Networks.
    - Tuning hyperparameters like learning rate and epsilon decay.

## Project 3: Stock Trading Bot 💸
*   **Goal**: Maximize profit in a simulated stock market.
*   **Tech Stack**: FinRL, yfinance.
*   **Key Learning**:
    - Custom environment creation.
    - Handling continuous action spaces (Buy/Sell percentages).
    - Using SAC (Soft Actor-Critic) for complex financial control.

## Project 4: Mini-RLHF (Text Ranking) 📝
*   **Goal**: Fine-tune a small language model to generate "positive" sentiment text.
*   **Tech Stack**: Hugging Face TRL, PyTorch.
*   **Key Learning**:
    - Training a Reward Model (Sentiment Classifier).
    - Implementing the PPO loop for text generation.
    - Managing KL-Divergence penalties.

---

## 🚀 Getting Started
1.  **Start with Gymnasium**: Learn the `env.step()` and `env.reset()` API thoroughly.
2.  **Use Stable-Baselines3**: Don't reinvent the wheel for standard algorithms (PPO/DQN).
3.  **WandB (Weights & Biases)**: Essential for tracking RL training curves, which are notoriously noisy.
