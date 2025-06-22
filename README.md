# RL-Basketball
# 🏀 AI Basketball Game with Deep Reinforcement Learning

This project is a **self-play basketball simulation** where two AI agents learn to play 1v1 basketball using **Deep Q-Learning (DQN)** and **Pygame**.

Red and Blue agents train by hitting a ball and trying to score into each other’s hoops — learning entirely through **reinforcement** and feedback from their environment.

---

## 🚀 Features

- 🤖 **Two AI agents (Red & Blue)** trained using Deep Q-Networks (DQN)
- 🎮 **Live simulation** using Pygame
- 🧠 **Custom reward shaping**:
  - Negative reward for staying far from the ball
  - Positive reward for hitting the ball
  - Positive reward for pushing ball toward opponent’s goal
  - Large reward for scoring
- 📈 Model improvement across episodes
- 💾 Automatic saving/loading of agent weights and training state
- 🔁 Training resumes from previous best-performing models

---

## 🧠 Reinforcement Learning Logic

Agents observe their environment (positions, distances, and velocities) and decide on actions (move up/down/left/right or stay) to maximize long-term rewards.  
Each action updates the game state, and the agent receives a reward based on the outcome.

Reward Shaping Summary:
| Behavior                     | Reward (actual reward used in script varies) |
|-----------------------------|----------------|
| Distance from ball          | Negative        |
| Hitting the ball            | +2              |
| Moving ball toward goal     | + (velocity ⋅ goal direction) × 0.05 |
| Scoring a goal              | +500            |

---

## 🛠 Tech Stack

- **Python 3.12+**
- **Pygame** – for 2D simulation
- **TensorFlow / Keras** – DQN implementation
- **NumPy**, **random**, **json**, **os**

---

## 🔧 Setup Instructions

1. **Clone the repo**
2. **Install dependencies**
3. **Run the training**
4. 
This will open a Pygame window and start training two AI agents from scratch (or continue from saved weights if available).

---

## 📁 Project Structure

├── main.py # Game loop & environment visualization
├── train.py # Training logic and episode control
├── dqn_agent.py # Deep Q-Learning agent logic
├── env.py # Environment: physics, rewards, game state
├── red_agent.weights.h5 / blue_agent.weights.h5 # Saved weights
├── best_scores.json # Tracks best reward for each agent
├── README.md
└── requirements.txt

---

## 🏆 Results

After ~100+ episodes of self-play training, agents learn to:
- Chase the ball effectively
- Push it toward the opponent's goal
- Consistently score goals!

> Average rewards improve drastically over time, from random motion to strategic gameplay.

---

## 🎥 Demo

> Contains a gameplay video showing AI agents learning and scoring in real time! 🎥

---

## 🤝 Contributions

Open to improvements or pull requests! Feel free to:
- Tune the reward shaping
- Improve action/state representation
- Add multiplayer or advanced physics

---

## 📬 Contact

Made by [me](linkedin.com/in/ankit-anand-b293422a4/)

Let’s connect if you’re working on game AI, reinforcement learning, or cool projects!

---

