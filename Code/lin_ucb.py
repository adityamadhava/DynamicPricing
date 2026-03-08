import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Code.feature_utils import extract_features
from RideSharing import DynamicPricingEnv

np.random.seed(42)

# Hyperparameters

N_ACTIONS = 20
HORIZON = 720
N_EPISODES = 200
ALPHA_REG = 1.0        # Regularization: A_a = alpha_reg * I
ALPHA_UCB = 2.0        # Exploration parameter in UCB bonus
WINDOW_SIZE = 2000
MAX_DRIVERS = 10



TOTAL_STEPS = N_EPISODES * HORIZON  # 144,000


def load_map():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")
    if not os.path.isfile(path):
        raise FileNotFoundError("map_agent.png not found")
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64) / 255.0


def bin_index_to_action(idx):
    idx = max(0, min(N_ACTIONS - 1, int(idx)))
    return (idx + 0.5) / N_ACTIONS


def fmt_time(seconds):
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
        self.alpha_ucb = alpha_ucb
        # A_a = alpha_reg * I  (d x d),  b_a = 0  (d,)
        self.A = [np.eye(feature_dim) * alpha_reg for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.A_inv = [np.eye(feature_dim) / alpha_reg for _ in range(n_actions)]

    def ucb_scores(self, phi):
        phi = np.asarray(phi).ravel()          # shape (d,)  — keep 1D
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            w = self.A_inv[a] @ self.b[a]                  # (d,)
            mean = float(phi @ w)                           # scalar
            var_term = float(phi @ self.A_inv[a] @ phi)    # scalar
            var_term = max(0.0, var_term)
            scores[a] = mean + self.alpha_ucb * np.sqrt(var_term)
        return scores

    def select_action(self, phi):
        scores = self.ucb_scores(phi)
        return int(np.argmax(scores))

    def update(self, action_idx, phi, reward):
        phi = np.asarray(phi).ravel()          # shape (d,)
        a = max(0, min(self.n_actions - 1, int(action_idx)))
        self.A[a] += np.outer(phi, phi)        # A_a += phi phi^T
        self.b[a] += reward * phi              # b_a += r * phi
        try:
            self.A_inv[a] = np.linalg.inv(self.A[a])
        except np.linalg.LinAlgError:
            pass

def run_training(env, map_img, bandit):
    all_rewards = []
    global_step = 0
    training_start = time.time()
    episode_times = []


    print(f"LinUCB Training: {N_EPISODES} episodes × {HORIZON} steps = {TOTAL_STEPS:,} total steps")
    print(f"alpha_reg={ALPHA_REG}  alpha_ucb={ALPHA_UCB}")

    for episode in range(N_EPISODES):
        ep_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0.0

        for t in range(HORIZON):
            phi = extract_features(obs, map_img, max_drivers=MAX_DRIVERS)
            action_idx = bandit.select_action(phi)
            action_continuous = bin_index_to_action(action_idx)
            next_obs, reward, terminated, truncated, _ = env.step(float(action_continuous))
            bandit.update(action_idx, phi, reward)
            all_rewards.append(float(reward))
            episode_reward += float(reward)
            obs = next_obs
            global_step += 1
            if terminated or truncated:
                break

        ep_elapsed = time.time() - ep_start
        episode_times.append(ep_elapsed)

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

        if (episode + 1) % 10 == 0:
            recent = all_rewards[-HORIZON * 10:]
            print(f"  └─ Last 10 eps mean reward: {np.mean(recent):.5f}"
                  f"std: {np.std(recent):.5f}"
                  f"steps so far: {global_step:,}\n")

    total_time = time.time() - training_start

    print(f"Training complete in {fmt_time(total_time)}")
    print(f"Total steps: {global_step:,}  |  Mean reward: {np.mean(all_rewards):.5f}")

    # Receding window average
    n = len(all_rewards)
    window_avg = [
        np.mean(all_rewards[max(0, i - WINDOW_SIZE + 1): i + 1])
        for i in range(n)
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(window_avg, color="darkgreen", linewidth=0.8)
    plt.title("LinUCB: Receding Window Average Reward (window=2000)")
    plt.xlabel("Time slot (across all episodes)")
    plt.ylabel("Average reward")
    plt.grid(True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lin_ucb_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curve to {out_path}")
    return all_rewards, window_avg


def main():
    print("Loading map and environment...")
    map_img = load_map()
    env = DynamicPricingEnv()
    bandit = LinUCBBandit(feature_dim=7, n_actions=N_ACTIONS, alpha_reg=ALPHA_REG, alpha_ucb=ALPHA_UCB)
    print("Starting LinUCB training...")
    run_training(env, map_img, bandit)
    print("Done!")


if __name__ == "__main__":
    main()