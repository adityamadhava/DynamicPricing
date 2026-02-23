import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import extract_features
from RideSharing import DynamicPricingEnv

np.random.seed(42)

# Hyperparameters and constants
N_ACTIONS = 20         
HORIZON = 720
N_EPISODES = 200
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
WINDOW_SIZE = 2000
MAX_DRIVERS = 10
RIDGE_REG = 1e-4        


def load_map():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64) / 255.0  # normalize to [0,1] for white > 0.5


def action_to_bin_index(action_continuous):
    a = np.clip(float(action_continuous), 0.0, 1.0)
    idx = int(a * N_ACTIONS)
    if idx >= N_ACTIONS:
        idx = N_ACTIONS - 1
    return idx


def bin_index_to_action(idx):
    idx = max(0, min(N_ACTIONS - 1, int(idx)))
    return (idx + 0.5) / N_ACTIONS


class LinearEpsilonGreedyBandit:
    def __init__(self, feature_dim=7, n_actions=N_ACTIONS, ridge_reg=RIDGE_REG):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.ridge_reg = ridge_reg
        # Per-action: A_a (d x d), b_a (d,), w_a = A_a^{-1} b_a
        self.A = [np.eye(feature_dim) * ridge_reg for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.A_inv = [np.eye(feature_dim) / ridge_reg for _ in range(n_actions)]

    def predict(self, phi):
        # Predict expected reward for each action: reward_a = w_a^T phi.
        phi = np.asarray(phi).reshape(-1, 1)
        preds = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            w = self.A_inv[a] @ self.b[a]
            preds[a] = float(phi.T @ w)
        return preds

    def select_action(self, phi, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        preds = self.predict(phi)
        return int(np.argmax(preds))

    def update(self, action_idx, phi, reward):
        phi = np.asarray(phi).ravel()
        phi = phi.reshape(-1, 1)
        a = max(0, min(self.n_actions - 1, int(action_idx)))
        # A_a += phi phi^T
        self.A[a] += phi @ phi.T
        self.b[a] += reward * phi.ravel()
        # Recompute A_inv
        try:
            self.A_inv[a] = np.linalg.inv(self.A[a])
        except np.linalg.LinAlgError:
            pass  # keep previous A_inv
        return


def run_training(env, map_img, bandit):
    all_rewards = []
    epsilon = EPSILON_START

    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0.0
        for t in range(HORIZON):
            context = obs
            phi = extract_features(context, map_img, max_drivers=MAX_DRIVERS)
            action_idx = bandit.select_action(phi, epsilon)
            action_continuous = bin_index_to_action(action_idx)
            # env.step expects action in [0, 1]
            next_obs, reward, terminated, truncated, _ = env.step(float(action_continuous))
            bandit.update(action_idx, phi, reward)
            all_rewards.append(float(reward))
            episode_reward += float(reward)
            obs = next_obs

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (episode + 1) % 10 == 0:
            avg_rew = episode_reward / HORIZON
            print(f"Episode {episode + 1}/{N_EPISODES}  Avg reward: {avg_rew:.6f}  epsilon: {epsilon:.4f}")

    # Receding window average
    n = len(all_rewards)
    window_avg = []
    for i in range(n):
        start = max(0, i - WINDOW_SIZE + 1)
        window_avg.append(np.mean(all_rewards[start : i + 1]))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(window_avg, color="steelblue", linewidth=0.8)
    plt.title("Lin-Greedy: Receding Window Average Reward (window=2000)")
    plt.xlabel("Time slot (across all episodes)")
    plt.ylabel("Average reward")
    plt.grid(True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_greedy_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curve to {out_path}")
    return all_rewards, window_avg


def main():
    print("Loading map and environment ")
    map_img = load_map()
    env = DynamicPricingEnv()
    bandit = LinearEpsilonGreedyBandit(feature_dim=7, n_actions=N_ACTIONS)
    print("Starting Lin-Greedy training (200 episodes, 720 steps each) ")
    run_training(env, map_img, bandit)
    print("Done ")


if __name__ == "__main__":
    main()
