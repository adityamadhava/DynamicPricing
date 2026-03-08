import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time

# Add current directory to path so we can import our own files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shared BFS-based feature extraction
from Code.feature_utils import extract_features

# Import the ride sharing environment
from RideSharing import DynamicPricingEnv

# Fix random seed for reproducibility
np.random.seed(42)

# ─── Hyperparameters ────────────────────────────────────────────────────────

N_ACTIONS = 20      # Number of price bins we discretize [0, 1] into
HORIZON = 720       # Steps per episode (1 hour at 5s per step = 720)
N_EPISODES = 200    # Total number of episodes to train for
ALPHA_REG = 1.0     # Regularization: initializes A_a = alpha_reg * I (stronger prior than lin_greedy)
ALPHA_UCB = 2.0     # Scales the exploration bonus — higher means more optimistic/exploratory
WINDOW_SIZE = 2000  # How many recent steps to average for the reward curve
MAX_DRIVERS = 10    # Maximum number of nearby drivers (from environment)

# Total number of training steps across all episodes
TOTAL_STEPS = N_EPISODES * HORIZON  # 144,000


def load_map():
    # Build full path to map_agent.png in the same folder as this script
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")

    # Raise an error early if the map file is missing
    if not os.path.isfile(path):
        raise FileNotFoundError("map_agent.png not found")

    # Open and convert image to numpy array
    img = Image.open(path)
    arr = np.array(img)

    # If image has multiple channels (RGB/RGBA), just keep the first channel
    if arr.ndim >= 3:
        arr = arr[:, :, 0]

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    return arr.astype(np.float64) / 255.0


def bin_index_to_action(idx):
    # Clamp index to valid range
    idx = max(0, min(N_ACTIONS - 1, int(idx)))

    # Return the midpoint price of the bin
    # e.g. bin 0 → 0.025, bin 19 → 0.975
    return (idx + 0.5) / N_ACTIONS


def fmt_time(seconds):
    # Convert raw seconds to a readable h/m/s string for logging
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class LinUCBBandit:
    def __init__(self, feature_dim=7, n_actions=N_ACTIONS, alpha_reg=ALPHA_REG, alpha_ucb=ALPHA_UCB):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.alpha_reg = alpha_reg
        self.alpha_ucb = alpha_ucb  # Controls how much we trust the uncertainty bonus

        # A[a] is the gram matrix for action a, initialized as alpha_reg * I
        # Higher alpha_reg means stronger regularization and larger initial uncertainty
        self.A = [np.eye(feature_dim) * alpha_reg for _ in range(n_actions)]

        # b[a] is the reward-weighted feature sum for action a, initialized to zeros
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]

        # Precompute inverse of A for each action to avoid recomputing every step
        # Initially A = alpha_reg * I, so A_inv = I / alpha_reg
        self.A_inv = [np.eye(feature_dim) / alpha_reg for _ in range(n_actions)]

    def ucb_scores(self, phi):
        # Flatten feature vector to 1D
        phi = np.asarray(phi).ravel()

        # Array to store UCB score for each action bin
        scores = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Compute ridge regression weights for this action: w = A_inv @ b
            w = self.A_inv[a] @ self.b[a]

            # Mean predicted reward: phi^T * w
            mean = float(phi @ w)

            # Uncertainty term: phi^T * A_inv * phi
            # This is large when the action hasn't been explored much (high posterior variance)
            var_term = float(phi @ self.A_inv[a] @ phi)

            # Clamp to 0 to avoid numerical issues giving negative variance
            var_term = max(0.0, var_term)

            # UCB score = mean reward + exploration bonus
            # alpha_ucb scales how optimistic we are about uncertain actions
            scores[a] = mean + self.alpha_ucb * np.sqrt(var_term)

        return scores

    def select_action(self, phi):
        # Compute UCB scores for all bins and pick the highest
        # No random exploration needed — uncertainty bonus handles it automatically
        scores = self.ucb_scores(phi)
        return int(np.argmax(scores))

    def update(self, action_idx, phi, reward):
        # Flatten feature vector
        phi = np.asarray(phi).ravel()

        # Clamp action index to valid range
        a = max(0, min(self.n_actions - 1, int(action_idx)))

        # Update gram matrix: A += phi * phi^T
        self.A[a] += np.outer(phi, phi)

        # Update reward vector: b += reward * phi
        self.b[a] += reward * phi

        # Recompute the inverse of A after the update
        try:
            self.A_inv[a] = np.linalg.inv(self.A[a])
        except np.linalg.LinAlgError:
            # Skip if matrix is somehow singular (shouldn't happen with regularization)
            pass


def run_training(env, map_img, bandit):
    # List to store every reward received across all episodes and steps
    all_rewards = []

    # Counter for total steps taken
    global_step = 0

    # Record wall-clock start time
    training_start = time.time()

    # Store time per episode for ETA estimation
    episode_times = []

    print(f"LinUCB Training: {N_EPISODES} episodes × {HORIZON} steps = {TOTAL_STEPS:,} total steps")
    print(f"alpha_reg={ALPHA_REG}  alpha_ucb={ALPHA_UCB}")

    for episode in range(N_EPISODES):
        # Record start time of this episode
        ep_start = time.time()

        # Reset environment to get initial observation
        obs, _ = env.reset()

        # Accumulate reward within this episode for logging
        episode_reward = 0.0

        for t in range(HORIZON):
            # Extract 7-dimensional feature vector from the raw context using BFS
            phi = extract_features(obs, map_img, max_drivers=MAX_DRIVERS)

            # Select action bin using UCB scores (no epsilon needed)
            action_idx = bandit.select_action(phi)

            # Convert bin index to continuous price value (midpoint of bin)
            action_continuous = bin_index_to_action(action_idx)

            # Take the action in the environment
            next_obs, reward, terminated, truncated, _ = env.step(float(action_continuous))

            # Update the bandit model with the observed reward
            bandit.update(action_idx, phi, reward)

            # Store reward for plotting
            all_rewards.append(float(reward))
            episode_reward += float(reward)

            # Move to next observation
            obs = next_obs
            global_step += 1

            # End episode early if environment signals done
            if terminated or truncated:
                break

        # Record episode time
        ep_elapsed = time.time() - ep_start
        episode_times.append(ep_elapsed)

        # ── Logging ──────────────────────────────────────────────────────────
        avg_rew = episode_reward / HORIZON
        total_elapsed = time.time() - training_start
        avg_ep_time = np.mean(episode_times)
        eta = avg_ep_time * (N_EPISODES - (episode + 1))
        pct = (episode + 1) / N_EPISODES * 100

        print(
            f"[{episode+1:>3}/{N_EPISODES}] ({pct:5.1f}%)"
            f"AvgRew: {avg_rew:+.5f}"
            f"EpTime: {ep_elapsed:.1f}s"
            f"Elapsed: {fmt_time(total_elapsed)}"
            f"ETA: {fmt_time(eta)}"
        )

        # Detailed summary every 10 episodes
        if (episode + 1) % 10 == 0:
            recent = all_rewards[-HORIZON * 10:]
            print(f"  └─ Last 10 eps mean reward: {np.mean(recent):.5f}"
                  f"std: {np.std(recent):.5f}"
                  f"steps so far: {global_step:,}\n")

    total_time = time.time() - training_start
    print(f"Training complete in {fmt_time(total_time)}")
    print(f"Total steps: {global_step:,}  |  Mean reward: {np.mean(all_rewards):.5f}")

    # ── Receding Window Average ───────────────────────────────────────────────
    # For each step i, compute the average reward over the last WINDOW_SIZE steps
    # Written as a list comprehension (same logic as lin_greedy, just more compact)
    n = len(all_rewards)
    window_avg = [
        np.mean(all_rewards[max(0, i - WINDOW_SIZE + 1): i + 1])
        for i in range(n)
    ]

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(window_avg, color="darkgreen", linewidth=0.8)
    plt.title("LinUCB: Receding Window Average Reward (window=2000)")
    plt.xlabel("Time slot (across all episodes)")
    plt.ylabel("Average reward")
    plt.grid(True)

    # Save plot to same folder as this script
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_ucb_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curve to {out_path}")

    return all_rewards, window_avg


def main():
    print("Loading map and environment...")

    # Load map image as normalized 2D numpy array
    map_img = load_map()

    # Create the ride sharing environment
    env = DynamicPricingEnv()

    # Create LinUCB bandit with 7 features and 20 action bins
    bandit = LinUCBBandit(feature_dim=7, n_actions=N_ACTIONS, alpha_reg=ALPHA_REG, alpha_ucb=ALPHA_UCB)

    print("Starting LinUCB training...")
    run_training(env, map_img, bandit)
    print("Done!")


# Only run main() if this script is executed directly (not imported)
if __name__ == "__main__":
    main()