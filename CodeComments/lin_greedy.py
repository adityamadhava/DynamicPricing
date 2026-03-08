import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time

# Add the current directory to path so we can import our own files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our shared feature extraction utility (BFS-based)
from Code.feature_utils import extract_features

# Import the ride sharing environment
from RideSharing import DynamicPricingEnv

# Fix random seed for reproducibility
np.random.seed(42)

# ─── Hyperparameters ────────────────────────────────────────────────────────

N_ACTIONS = 20          # Number of price bins we discretize [0,1] into
HORIZON = 720           # Steps per episode (1 hour at 5s per step = 720)
N_EPISODES = 200        # Total number of episodes to train for
EPSILON_START = 1.0     # Start fully random (100% exploration)
EPSILON_DECAY = 0.995   # Multiply epsilon by this after each episode
EPSILON_MIN = 0.05      # Never go below 5% exploration
WINDOW_SIZE = 2000      # How many recent steps to average for the reward curve
MAX_DRIVERS = 10        # Maximum number of nearby drivers (from environment)
RIDGE_REG = 1e-4        # Ridge regularization to keep matrix A invertible

# Total number of training steps across all episodes
TOTAL_STEPS = N_EPISODES * HORIZON  # 144,000


def load_map():
    # Build the full path to map_agent.png in the same folder as this script
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")

    # Open the image using PIL
    img = Image.open(path)

    # Convert image to numpy array
    arr = np.array(img)

    # If the image has multiple channels (RGB/RGBA), just keep the first channel
    if arr.ndim >= 3:
        arr = arr[:, :, 0]

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    # Values > 0.5 will be treated as white (road), rest as black (wall)
    return arr.astype(np.float64) / 255.0


def action_to_bin_index(action_continuous):
    # Clamp the continuous action to [0, 1] just in case
    a = np.clip(float(action_continuous), 0.0, 1.0)

    # Map the continuous value to a bin index (0 to N_ACTIONS-1)
    idx = int(a * N_ACTIONS)

    # Edge case: if action is exactly 1.0, idx would be N_ACTIONS which is out of range
    if idx >= N_ACTIONS:
        idx = N_ACTIONS - 1

    return idx


def bin_index_to_action(idx):
    # Clamp index to valid range
    idx = max(0, min(N_ACTIONS - 1, int(idx)))

    # Return the midpoint price of the bin
    # e.g. bin 0 → price 0.025, bin 1 → price 0.075, etc.
    return (idx + 0.5) / N_ACTIONS


class LinearEpsilonGreedyBandit:
    def __init__(self, feature_dim=7, n_actions=N_ACTIONS, ridge_reg=RIDGE_REG):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.ridge_reg = ridge_reg

        # A[a] is the gram matrix for action a, initialized as ridge_reg * I
        # This is the matrix we invert to get the ridge regression weights
        self.A = [np.eye(feature_dim) * ridge_reg for _ in range(n_actions)]

        # b[a] is the reward-weighted feature sum for action a, initialized to zeros
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]

        # Store inverse of A for each action to avoid recomputing every step
        # Initially A = ridge_reg * I, so A_inv = I / ridge_reg
        self.A_inv = [np.eye(feature_dim) / ridge_reg for _ in range(n_actions)]

    def predict(self, phi):
        # Flatten feature vector to 1D just in case
        phi = np.asarray(phi).ravel()

        # Array to store predicted reward for each action bin
        preds = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Ridge regression weights for action a: w = A_inv @ b
            w = self.A_inv[a] @ self.b[a]

            # Predicted reward = phi dot w (linear model)
            preds[a] = float(phi @ w)

        return preds

    def select_action(self, phi, epsilon):
        # With probability epsilon, pick a random action (exploration)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)

        # Otherwise pick the action with highest predicted reward (exploitation)
        preds = self.predict(phi)
        return int(np.argmax(preds))

    def update(self, action_idx, phi, reward):
        # Flatten feature vector
        phi = np.asarray(phi).ravel()

        # Clamp action index to valid range
        a = max(0, min(self.n_actions - 1, int(action_idx)))

        # Update gram matrix: A += phi * phi^T (outer product)
        self.A[a] += np.outer(phi, phi)

        # Update reward vector: b += reward * phi
        self.b[a] += reward * phi

        # Recompute the inverse of A after the update
        try:
            self.A_inv[a] = np.linalg.inv(self.A[a])
        except np.linalg.LinAlgError:
            # If matrix is somehow singular, skip the update (shouldn't happen with ridge reg)
            pass


def fmt_time(seconds):
    # Convert raw seconds into a human-readable h/m/s string for logging
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_training(env, map_img, bandit):
    # List to store every reward received across all episodes and steps
    all_rewards = []

    # Start epsilon at EPSILON_START (1.0 = fully random at the beginning)
    epsilon = EPSILON_START

    # Counter for total steps taken across all episodes
    global_step = 0

    # Record wall-clock time when training starts
    training_start = time.time()

    # Store time taken per episode for ETA estimation
    episode_times = []

    print(f"  Training: {N_EPISODES} episodes × {HORIZON} steps = {TOTAL_STEPS:,} total steps")

    for episode in range(N_EPISODES):
        # Record time at start of this episode
        ep_start = time.time()

        # Reset environment to get initial observation for this episode
        obs, _ = env.reset()

        # Accumulate reward within this episode for logging
        episode_reward = 0.0

        for t in range(HORIZON):
            context = obs

            # Extract 7-dimensional feature vector from the raw context using BFS
            phi = extract_features(context, map_img, max_drivers=MAX_DRIVERS)

            # Select a bin index using epsilon-greedy
            action_idx = bandit.select_action(phi, epsilon)

            # Convert bin index to a continuous price value (midpoint of the bin)
            action_continuous = bin_index_to_action(action_idx)

            # Take the action in the environment, get next observation and reward
            next_obs, reward, terminated, truncated, _ = env.step(float(action_continuous))

            # Update the bandit model with what we observed
            bandit.update(action_idx, phi, reward)

            # Store the reward for plotting later
            all_rewards.append(float(reward))
            episode_reward += float(reward)

            # Move to the next observation
            obs = next_obs
            global_step += 1

        # Record how long this episode took
        ep_elapsed = time.time() - ep_start
        episode_times.append(ep_elapsed)

        # Decay epsilon after each episode (shift from exploration toward exploitation)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # ── Logging ──────────────────────────────────────────────────────────
        avg_rew = episode_reward / HORIZON
        total_elapsed = time.time() - training_start
        avg_ep_time = np.mean(episode_times)
        eps_remaining = N_EPISODES - (episode + 1)
        eta = avg_ep_time * eps_remaining
        pct = (episode + 1) / N_EPISODES * 100

        print(
            f"[{episode+1:>3}/{N_EPISODES}] ({pct:5.1f}%)  "
            f"AvgRew: {avg_rew:+.5f}  "
            f"ε: {epsilon:.4f}  "
            f"EpTime: {ep_elapsed:.1f}s  "
            f"Elapsed: {fmt_time(total_elapsed)}  "
            f"ETA: {fmt_time(eta)}"
        )

        # Print a more detailed summary every 10 episodes
        if (episode + 1) % 10 == 0:
            recent = all_rewards[-HORIZON * 10:]
            print(f"Last 10 eps mean reward: {np.mean(recent):.5f}  "
                  f"std: {np.std(recent):.5f}  "
                  f"steps so far: {global_step:,}\n")

    total_time = time.time() - training_start
    print(f"Training complete in {fmt_time(total_time)}")
    print(f"Total steps: {global_step:,}  |  Mean reward: {np.mean(all_rewards):.5f}")

    # ── Receding Window Average ───────────────────────────────────────────────
    # For each step i, compute the average reward over the last WINDOW_SIZE steps
    # This smooths out noise and shows the trend in learning
    n = len(all_rewards)
    window_avg = []
    for i in range(n):
        start = max(0, i - WINDOW_SIZE + 1)
        window_avg.append(np.mean(all_rewards[start: i + 1]))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(window_avg, color="steelblue", linewidth=0.8)
    plt.title("Lin-Greedy: Receding Window Average Reward (window=2000)")
    plt.xlabel("Time slot (across all episodes)")
    plt.ylabel("Average reward")
    plt.grid(True)

    # Save the plot as a PNG in the same folder as this script
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_greedy_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curve to {out_path}")

    return all_rewards, window_avg


def main():
    print("Loading map and environment...")

    # Load map image as a normalized 2D numpy array
    map_img = load_map()

    # Create the ride sharing environment
    env = DynamicPricingEnv()

    # Create the bandit with 7 features and 20 action bins
    bandit = LinearEpsilonGreedyBandit(feature_dim=7, n_actions=N_ACTIONS)

    print("Starting Lin-Greedy training (200 episodes, 720 steps each) ")
    run_training(env, map_img, bandit)
    print("Done!")


# Only run main() if this script is executed directly (not imported)
if __name__ == "__main__":
    main()