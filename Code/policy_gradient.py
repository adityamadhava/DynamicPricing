import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time
import torch
import torch.nn as nn
TORCH_AVAILABLE = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

TOTAL_STEPS = N_EPISODES * HORIZON


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def extract_features_fast(obs, max_drivers=10):

    # Fast feature extraction using Euclidean distance instead of BFS.
    # Avoids the 4-hour runtime caused by repeated BFS on the map image.
    #
    # Feature order:
    #   [0] trip_dist          - Euclidean distance from passenger origin to destination
    #   [1] min_driver_dist    - distance of closest driver to passenger
    #   [2] mean_driver_dist   - mean distance of all drivers to passenger
    #   [3] passenger_alpha    - passenger price sensitivity
    #   [4] mean_driver_alpha  - mean driver price sensitivity
    #   [5] min_driver_alpha   - min driver price sensitivity (most willing driver)
    #   [6] num_drivers_norm   - number of nearby drivers / max_drivers

    c_passenger, c_drivers = obs
    c_passenger = np.asarray(c_passenger).ravel()

    x_orig, y_orig = float(c_passenger[0]), float(c_passenger[1])
    x_dest, y_dest = float(c_passenger[2]), float(c_passenger[3])
    passenger_alpha = float(c_passenger[4])

    # Euclidean trip distance
    trip_dist = np.sqrt((x_dest - x_orig) ** 2 + (y_dest - y_orig) ** 2)

    driver_dists = []
    driver_alphas = []
    if c_drivers is not None and len(c_drivers) > 0:
        for d in c_drivers:
            d = np.asarray(d).ravel()
            if len(d) >= 3:
                x_d, y_d, alpha_d = float(d[0]), float(d[1]), float(d[2])
                dist = np.sqrt((x_d - x_orig) ** 2 + (y_d - y_orig) ** 2)
                driver_dists.append(dist)
                driver_alphas.append(alpha_d)

    min_driver_dist   = float(np.min(driver_dists))   if driver_dists  else 0.0
    mean_driver_dist  = float(np.mean(driver_dists))  if driver_dists  else 0.0
    mean_driver_alpha = float(np.mean(driver_alphas)) if driver_alphas else 0.0
    min_driver_alpha  = float(np.min(driver_alphas))  if driver_alphas else 0.0
    num_drivers_norm  = len(driver_alphas) / max(max_drivers, 1)

    passenger_alpha   = max(0.0, passenger_alpha)
    mean_driver_alpha = max(0.0, mean_driver_alpha)
    min_driver_alpha  = max(0.0, min_driver_alpha)

    return np.array([
        trip_dist,
        min_driver_dist,
        mean_driver_dist,
        passenger_alpha,
        mean_driver_alpha,
        min_driver_alpha,
        num_drivers_norm
    ], dtype=np.float64)


# Policy Network: 7 -> 64 -> 32 -> 1 (sigmoid)
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
        # Because output = sigmoid(pre_activation), we must account for the
        # change-of-variables: log p(a) = log p(z) - log |d(sigmoid)/d(z)|
        # d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z)) = mu * (1 - mu)
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

        # Policy gradient loss: minimize -(reward - baseline) * log_prob
        loss = -(reward - self.baseline) * log_prob_corrected
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_training(env, agent):
    all_rewards = []
    global_step = 0
    training_start = time.time()
    episode_times = []

    print(f"Policy Gradient Training: {N_EPISODES} episodes x {HORIZON} steps = {TOTAL_STEPS:,} total steps")
    print(f"lr={LR}  std={STD_FIXED}  baseline_decay={BASELINE_DECAY}")

    for episode in range(N_EPISODES):
        ep_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0.0

        for t in range(HORIZON):
            phi = extract_features_fast(obs, max_drivers=MAX_DRIVERS)
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
            f"[{episode+1:>3}/{N_EPISODES}] ({pct:5.1f}%)  "
            f"AvgRew: {avg_rew:+.5f}  "
            f"Baseline: {agent.baseline:.5f}  "
            f"EpTime: {ep_elapsed:.1f}s  "
            f"Elapsed: {fmt_time(total_elapsed)}  "
            f"ETA: {fmt_time(eta)}"
        )

        if (episode + 1) % 10 == 0:
            recent = all_rewards[-HORIZON * 10:]
            print(f"  Last 10 eps mean reward: {np.mean(recent):.5f}  "
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
    """
    Two sanity tests to verify the trained agent behaves logically.

    Feature order:
      [0] trip_dist          - Euclidean distance passenger origin -> destination
      [1] min_driver_dist    - distance of closest driver to passenger
      [2] mean_driver_dist   - mean distance of all drivers to passenger
      [3] passenger_alpha    - passenger price sensitivity
      [4] mean_driver_alpha  - mean driver price sensitivity
      [5] min_driver_alpha   - min driver price sensitivity
      [6] num_drivers_norm   - number of nearby drivers / max_drivers
    """

    # Neutral baseline feature vector
    base_phi = np.array([0.3, 0.2, 0.3, 0.5, 0.5, 0.3, 0.5], dtype=np.float32)

    # Test 1: Price vs Passenger Sensitivity (passenger_alpha = index 3)
    # Expectation: higher passenger alpha -> agent should quote higher price
    # (passenger is more willing to pay, so app can charge more)
    alphas = np.linspace(0.05, 0.95, 15)
    prices_1 = []
    for alpha in alphas:
        phi = base_phi.copy()
        phi[3] = alpha          # index 3 = passenger_alpha
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
    # Expectation: longer trip -> higher price quoted
    # (longer trip = more commission earned)
    trip_distances = np.linspace(0.05, 0.9, 15)
    prices_2 = []
    for td in trip_distances:
        phi = base_phi.copy()
        phi[0] = td             # index 0 = trip_dist
        prices_2.append(agent.get_mean(phi))

    plt.figure(figsize=(6, 4))
    plt.plot(trip_distances, prices_2, "o-", color="coral")
    plt.title("Sanity Test 2: Price vs Trip Distance")
    plt.xlabel("Trip distance (normalized Euclidean)")
    plt.ylabel("Quoted price (mean)")
    plt.grid(True)
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_test_2.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


def main():
    print("Initializing environment...")
    env = DynamicPricingEnv()
    agent = PolicyGradientAgent(lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY)

    print("Starting Policy Gradient training...")
    run_training(env, agent)

    print("\nRunning sanity tests...")
    run_sanity_tests(env, agent)

    print("Done!")


if __name__ == "__main__":
    main()

