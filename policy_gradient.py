import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import extract_features
from RideSharing import DynamicPricingEnv

np.random.seed(42)

# Hyperparameters

FEATURE_DIM = 7
HORIZON = 720
N_EPISODES = 200
STD_FIXED = 0.1          # Fixed std for Gaussian policy
LR = 0.001               # Adam learning rate
BASELINE_DECAY = 0.99    # EMA decay for reward baseline
WINDOW_SIZE = 2000
MAX_DRIVERS = 10


# Test Hyperparameters

# FEATURE_DIM = 7
# HORIZON = 720
# N_EPISODES = 5
# STD_FIXED = 0.1
# LR = 0.001
# BASELINE_DECAY = 0.99
# WINDOW_SIZE = 100
# MAX_DRIVERS = 10

TOTAL_STEPS = N_EPISODES * HORIZON  # 144,000

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found.")
    sys.exit(1)


def load_map():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64) / 255.0


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"



# Policy Network: 7 - 64 - 32 - 1 (sigmoid)
# Output is the mean price in [0, 1]

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # output in (0, 1)
        return x.squeeze(-1)


class PolicyGradientAgent:
    def __init__(self, lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY):
        self.policy = PolicyNet()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.std = std
        self.baseline_decay = baseline_decay
        self.baseline = 0.0

    def _phi_tensor(self, phi):
        return torch.as_tensor(
            np.asarray(phi).astype(np.float32)
        ).reshape(1, -1)

    def get_mean(self, phi):
        with torch.no_grad():
            return self.policy(self._phi_tensor(phi)).item()

    def select_action(self, phi, explore=True):
        x = self._phi_tensor(phi)
        mu = self.policy(x).squeeze()          # scalar tensor, in (0,1)
        mu_val = mu.item()

        if not explore:
            return float(np.clip(mu_val, 0.0, 1.0)), None

        dist = torch.distributions.Normal(mu, self.std)
        z = dist.sample()
        action = float(z.clamp(0.0, 1.0).item())

        log_prob_normal = dist.log_prob(z).item()

        # Jacobian correction for sigmoid output layer
        eps = 1e-6
        jacobian_log = float(np.log(mu_val * (1.0 - mu_val) + eps))
        log_prob_corrected = log_prob_normal - jacobian_log

        return action, log_prob_corrected

    def update_step(self, phi, action, reward):
        x = self._phi_tensor(phi)
        mu = self.policy(x).squeeze()          # scalar tensor
        mu_val = mu.item()

        dist = torch.distributions.Normal(mu, self.std)
        action_t = torch.tensor(action, dtype=torch.float32)
        log_p = dist.log_prob(action_t)

        # Jacobian correction (same as in select_action)
        eps = 1e-6
        jacobian_log = float(np.log(mu_val * (1.0 - mu_val) + eps))
        log_prob_corrected = log_p - jacobian_log

        # Update EMA baseline
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1.0 - self.baseline_decay) * reward
        )

        # Policy gradient loss (minimize negative expected reward)
        loss = -(reward - self.baseline) * log_prob_corrected
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_training(env, map_img, agent):
    all_rewards = []
    global_step = 0
    training_start = time.time()
    episode_times = []
    print(f"Policy Gradient Training: {N_EPISODES} episodes × {HORIZON} steps = {TOTAL_STEPS:,} total steps")
    print(f"lr={LR}  std={STD_FIXED}  baseline_decay={BASELINE_DECAY}")

    for episode in range(N_EPISODES):
        ep_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0.0

        for t in range(HORIZON):
            phi = extract_features(obs, map_img, max_drivers=MAX_DRIVERS)
            action, _ = agent.select_action(phi, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(float(action))
            agent.update_step(phi, action, float(reward))
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
            f"Baseline: {agent.baseline:.5f}"
            f"EpTime: {ep_elapsed:.1f}s"
            f"Elapsed: {fmt_time(total_elapsed)}"
            f"ETA: {fmt_time(eta)}"
        )

        if (episode + 1) % 10 == 0:
            recent = all_rewards[-HORIZON * 10:]
            print(f"Last 10 eps mean reward: {np.mean(recent):.5f}"
                  f"std: {np.std(recent):.5f}  "
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
    plt.plot(window_avg, color="purple", linewidth=0.8)
    plt.title("Policy Gradient: Receding Window Average Reward (window=2000)")
    plt.xlabel("Time slot (across all episodes)")
    plt.ylabel("Average reward")
    plt.grid(True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_gradient_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curve to {out_path}")
    return all_rewards, window_avg


def run_sanity_tests(env, agent):

    # Feature order from feature_utils.py:
    #   [0] trip_dist_norm
    #   [1] alpha_p_norm        passenger price sensitivity
    #   [2] n_drivers_norm
    #   [3] mean_driver_dist
    #   [4] min_driver_dist
    #   [5] mean_alpha_d         driver price sensitivity
    #   [6] road_density

    # Neutral baseline feature vector
    base_phi = np.array([0.3, 0.5, 0.5, 0.2, 0.1, 0.5, 0.5], dtype=np.float32)

    # Test 1: Price vs Passenger Sensitivity (alpha_p = index 1)
    # Expectation: higher passenger alpha -- agent should quote higher price
    alphas = np.linspace(0.05, 0.95, 15)
    prices_1 = []
    for alpha in alphas:
        phi = base_phi.copy()
        phi[1] = alpha          # index 1 = passenger alpha
        prices_1.append(agent.get_mean(phi))

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, prices_1, "o-", color="teal")
    plt.title("Sanity Test 1: Price vs Passenger Price Sensitivity")
    plt.xlabel("Passenger alpha (price sensitivity)")
    plt.ylabel("Quoted price (mean)")
    plt.grid(True)
    out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_test_1.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # Test 2: Price vs Trip Distance (trip_dist = index 0)
    # Expectation: longer trip --- higher price quoted
    trip_distances = np.linspace(0.05, 0.9, 15)
    prices_2 = []
    for td in trip_distances:
        phi = base_phi.copy()
        phi[0] = td             # index 0 = trip_dist_norm
        prices_2.append(agent.get_mean(phi))

    plt.figure(figsize=(6, 4))
    plt.plot(trip_distances, prices_2, "o-", color="coral")
    plt.title("Sanity Test 2: Price vs Trip Distance")
    plt.xlabel("Trip distance (normalized)")
    plt.ylabel("Quoted price (mean)")
    plt.grid(True)
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_test_2.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


def main():
    print("Loading map and environment...")
    map_img = load_map()
    env = DynamicPricingEnv()
    agent = PolicyGradientAgent(lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY)
    print("Starting Policy Gradient training...")
    run_training(env, map_img, agent)
    print("\nRunning sanity tests...")
    run_sanity_tests(env, agent)
    print("Done!")


if __name__ == "__main__":
    main()


