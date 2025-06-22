from game_env import BasketballEnv
from dqn_agent import DQNAgent
import time
import pygame
import sys
import os
import json

env = BasketballEnv()
state_size = 8  # x, y of red, blue, ball + vx, vy
action_size = 4

episodes = 1000
agent_red = DQNAgent("Red", state_size, action_size)
agent_blue = DQNAgent("Blue", state_size, action_size)

# Load latest saved weights & model/parameters if available
if os.path.exists("red_agent.weights.h5"):
    agent_red.model.load_weights("red_agent.weights.h5")
    agent_red.load_state("red_agent_state.json")
    print("✅ Loaded red agent weights.")
if os.path.exists("blue_agent.weights.h5"):
    agent_blue.model.load_weights("blue_agent.weights.h5")
    agent_blue.load_state("blue_agent_state.json")
    print("✅ Loaded blue agent weights.")

best_scores_file = "best_scores.json"
best_red_score = -float('inf')
best_blue_score = -float('inf')

# Load saved best scores safely
if os.path.exists(best_scores_file):
    try:
        with open(best_scores_file, "r") as f:
            scores = json.load(f)
            best_red_score = scores.get("best_red_score", -float('inf'))
            best_blue_score = scores.get("best_blue_score", -float('inf'))
            print(f"✅ Loaded best scores: Red={best_red_score:.2f}, Blue={best_blue_score:.2f}")
    except (json.JSONDecodeError, IOError):
        print("⚠️ best_scores.json is empty or corrupted. Using default scores.")



for ep in range(1, episodes + 1):
    state = env.reset()
    done = False
    total_red, total_blue = 0, 0
    start_time = time.time()

    while True:

         # handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Window closed. Saving and exiting...")
                #save weights
                agent_red.model.save_weights("red_agent.weights.h5")
                agent_blue.model.save_weights("blue_agent.weights.h5")
                #save model and other parameters like epsilon
                agent_red.save_state("red_agent_state.json")
                agent_blue.save_state("blue_agent_state.json")
                pygame.quit()
                sys.exit()

        action_red = agent_red.act(state)
        action_blue = agent_blue.act(state)

        next_state, reward_red, reward_blue, done = env.step(action_red, action_blue)

        agent_red.remember(state, action_red, reward_red, next_state, done)
        agent_blue.remember(state, action_blue, reward_blue, next_state, done)

        state = next_state
        total_red += reward_red
        total_blue += reward_blue

        env.render(episode=ep, score_red=round(total_red, 2), score_blue=round(total_blue, 2))

        # End episode after 30 seconds
        if time.time() - start_time > 15:
            break

    agent_red.replay()
    agent_blue.replay()

        # Update target models more frequently to enable learning across episodes
    agent_red.update_target()
    agent_blue.update_target()

    # Soft update every episode
    agent_red.soft_update_target(tau=0.01)
    agent_blue.soft_update_target(tau=0.01)

    if total_red > best_red_score:
        best_red_score = total_red
        agent_red.model.save_weights("best_red.weights.h5")
        print(f"🔴 New best Red model saved with reward {total_red:.2f}")

    if total_blue > best_blue_score:
        best_blue_score = total_blue
        agent_blue.model.save_weights("best_blue.weights.h5")
        print(f"🔵 New best Blue model saved with reward {total_blue:.2f}")

    # Save updated best scores to disk
    with open(best_scores_file, "w") as f:
        json.dump({
            "best_red_score": best_red_score,
            "best_blue_score": best_blue_score
        }, f)


# Save latest model weights (last but not best)
print("saving")
agent_red.model.save_weights("red_agent.weights.h5")
agent_blue.model.save_weights("blue_agent.weights.h5")

# Save training metrics
total_metrics = {
    "episode": ep,
    "red_reward": round(total_red, 2),
    "blue_reward": round(total_blue, 2)
}

import json
with open("training_metrics.json", "w") as f:
    json.dump(total_metrics, f, indent=2)
