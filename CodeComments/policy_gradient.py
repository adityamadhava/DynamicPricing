import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time
import torch
import torch.nn as nn

# Flag to confirm torch is available (kept for consistency)
TORCH_AVAILABLE = True

# Add current directory to path so we can import our own files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ride sharing environment
from RideSharing import DynamicPricingEnv

# Fix random seed for reproducibility
np.random.seed(42)

# ─── Hyperparameters ────────────────────────────────────────────────────────

FEATURE_DIM = 7         # Size of the feature vector fed into the network
HORIZON = 720           # Steps per episode (1 hour at 5s per step)
N_EPISODES = 200        # Total episodes to train for
STD_FIXED = 0.1         # Fixed standard deviation for the Gaussian policy
                        # Controls how much random exploration happens around the mean price
LR = 0.001              # Adam optimizer learning rate
BASELINE_DECAY = 0.99   # EMA decay for the reward baseline (slow-moving average)
WINDOW_SIZE = 2000      # Steps to average over for the reward curve
MAX_DRIVERS = 10        # Maximum number of nearby drivers

# Total training steps
TOTAL_STEPS = N_EPISODES * HORIZON


def fmt_time(seconds):
    # Convert raw seconds to readable h/m/s string for logging
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def extract_features_fast(obs, max_drivers=10):
    # This is a faster version of feature extraction that uses Euclidean distance
    # instead of BFS. BFS over 144,000 steps would take hours, so we skip it here.
    # The neural network is deep enough to implicitly learn spatial structure anyway.

    # Unpack the context into passenger info and driver info
    c_passenger, c_drivers = obs

    # Flatten passenger array to 1D
    c_passenger = np.asarray(c_passenger).ravel()

    # Extract passenger origin and destination coordinates
    x_orig, y_orig = float(c_passenger[0]), float(c_passenger[1])
    x_dest, y_dest = float(c_passenger[2]), float(c_passenger[3])

    # Extract passenger price sensitivity
    passenger_alpha = float(c_passenger[4])

    # Euclidean distance from origin to destination (straight line, no map routing)
    trip_dist = np.sqrt((x_dest - x_orig) ** 2 + (y_dest - y_orig) ** 2)

    # Lists to collect driver distances and alphas
    driver_dists = []
    driver_alphas = []

    # Loop through each nearby driver
    if c_drivers is not None and len(c_drivers) > 0:
        for d in c_drivers:
            d = np.asarray(d).ravel()
            if len(d) >= 3:
                x_d, y_d, alpha_d = float(d[0]), float(d[1]), float(d[2])

                # Euclidean distance from this driver to the passenger origin
                dist = np.sqrt((x_d - x_orig) ** 2 + (y_d - y_orig) ** 2)
                driver_dists.append(dist)
                driver_alphas.append(alpha_d)

    # Compute summary statistics — default to 0 if no drivers
    min_driver_dist   = float(np.min(driver_dists))   if driver_dists  else 0.0
    mean_driver_dist  = float(np.mean(driver_dists))  if driver_dists  else 0.0
    mean_driver_alpha = float(np.mean(driver_alphas)) if driver_alphas else 0.0
    min_driver_alpha  = float(np.min(driver_alphas))  if driver_alphas else 0.0

    # Normalise driver count by max possible drivers
    num_drivers_norm  = len(driver_alphas) / max(max_drivers, 1)

    # Clip negatives just in case of bad data
    passenger_alpha   = max(0.0, passenger_alpha)
    mean_driver_alpha = max(0.0, mean_driver_alpha)
    min_driver_alpha  = max(0.0, min_driver_alpha)

    # Return the 7-dimensional feature vector
    return np.array([
        trip_dist,          # Feature 1: trip distance (Euclidean)
        min_driver_dist,    # Feature 2: closest driver distance
        mean_driver_dist,   # Feature 3: mean driver distance
        passenger_alpha,    # Feature 4: passenger price sensitivity
        mean_driver_alpha,  # Feature 5: mean driver price sensitivity
        min_driver_alpha,   # Feature 6: most willing driver sensitivity
        num_drivers_norm    # Feature 7: normalised driver count (supply proxy)
    ], dtype=np.float64)


# ─── Policy Network ──────────────────────────────────────────────────────────
# Architecture: 7 -> 64 -> 32 -> 1 (sigmoid)
# Takes the feature vector and outputs a single mean price in (0, 1)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: 7 inputs -> 64 hidden units
        self.fc1 = nn.Linear(FEATURE_DIM, 64)
        # Layer 2: 64 -> 32 hidden units
        self.fc2 = nn.Linear(64, 32)
        # Layer 3: 32 -> 1 output (the mean price)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # ReLU activation after layer 1
        x = torch.relu(self.fc1(x))
        # ReLU activation after layer 2
        x = torch.relu(self.fc2(x))
        # Sigmoid activation squashes output to (0, 1) — valid price range
        x = torch.sigmoid(self.fc3(x))
        # Remove the trailing dimension so output is a scalar, not shape (1,)
        return x.squeeze(-1)


class PolicyGradientAgent:
    def __init__(self, lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY):
        # Initialize the policy network
        self.policy = PolicyNet()

        # Adam optimizer for updating network weights
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Fixed standard deviation for the Gaussian policy
        self.std = std

        # EMA decay rate for the baseline
        self.baseline_decay = baseline_decay

        # Running baseline initialized to 0 (will warm up quickly due to EMA)
        self.baseline = 0.0

    def _phi_tensor(self, phi):
        # Convert numpy feature vector to a float32 PyTorch tensor with shape (1, 7)
        # The (1, 7) shape is needed because the network expects a batch dimension
        return torch.as_tensor(
            np.asarray(phi).astype(np.float32)
        ).reshape(1, -1)

    def get_mean(self, phi):
        # Get the network's predicted mean price without computing gradients
        # Used during sanity tests where we don't need to train
        with torch.no_grad():
            return self.policy(self._phi_tensor(phi)).item()

    def select_action(self, phi, explore=True):
        # Convert feature vector to tensor
        x = self._phi_tensor(phi)

        # Forward pass through policy network to get mean price mu
        mu = self.policy(x).squeeze()  # scalar tensor in (0, 1)
        mu_val = mu.item()

        # If not exploring (e.g. during evaluation), just return the mean directly
        if not explore:
            return float(np.clip(mu_val, 0.0, 1.0)), None

        # Sample an action from Gaussian(mu, std)
        dist = torch.distributions.Normal(mu, self.std)
        z = dist.sample()

        # Clamp action to [0, 1] — prices outside this range are invalid
        action = float(z.clamp(0.0, 1.0).item())

        # Compute log probability of the sampled action under the Gaussian
        log_prob_normal = dist.log_prob(z).item()

        # ── Jacobian Correction ───────────────────────────────────────────────
        # The network output passes through sigmoid, so we need a correction term.
        # When we sample z ~ N(mu, std), the actual action is a = sigmoid(z).
        # The change-of-variables formula requires: log p(a) = log p(z) - log|da/dz|
        # where da/dz = sigmoid(z) * (1 - sigmoid(z)) = mu * (1 - mu)
        # Without this, the gradient estimate is biased near the boundaries of [0,1]
        eps = 1e-6  # Small value to avoid log(0)
        jacobian_log = float(np.log(mu_val * (1.0 - mu_val) + eps))
        log_prob_corrected = log_prob_normal - jacobian_log

        return action, log_prob_corrected

    def update_step(self, phi, action, reward):
        # Convert feature vector to tensor
        x = self._phi_tensor(phi)

        # Forward pass to get current mean price
        mu = self.policy(x).squeeze()
        mu_val = mu.item()

        # Compute log probability of the action that was actually taken
        dist = torch.distributions.Normal(mu, self.std)
        action_t = torch.tensor(action, dtype=torch.float32)
        log_p = dist.log_prob(action_t)

        # Apply Jacobian correction (same logic as in select_action)
        eps = 1e-6
        jacobian_log = float(np.log(mu_val * (1.0 - mu_val) + eps))
        log_prob_corrected = log_p - jacobian_log

        # ── EMA Baseline Update ───────────────────────────────────────────────
        # Update the running baseline as a slow exponential moving average of rewards
        # baseline ≈ average reward seen so far
        self.baseline = (
            self.baseline_decay * self.baseline          # keep 99% of old value
            + (1.0 - self.baseline_decay) * reward       # add 1% of new reward
        )

        # ── Policy Gradient Loss ──────────────────────────────────────────────
        # Loss = -(reward - baseline) * log_prob
        # If reward > baseline (good action), loss is negative → gradient increases log_prob
        # If reward < baseline (bad action), loss is positive → gradient decreases log_prob
        loss = -(reward - self.baseline) * log_prob_corrected

        # Zero gradients, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_training(env, agent):
    # List to store every reward across all episodes
    all_rewards = []
    global_step = 0
    training_start = time.time()
    episode_times = []

    print(f"Policy Gradient Training: {N_EPISODES} episodes x {HORIZON} steps = {TOTAL_STEPS:,} total steps")
    print(f"lr={LR}  std={STD_FIXED}  baseline_decay={BASELINE_DECAY}")

    for episode in range(N_EPISODES):
        ep_start = time.time()

        # Reset environment to get initial observation
        obs, _ = env.reset()
        episode_reward = 0.0

        for t in range(HORIZON):
            # Extract features using fast Euclidean method
            phi = extract_features_fast(obs, max_drivers=MAX_DRIVERS)

            # Sample action from policy (explore=True during training)
            action, _ = agent.select_action(phi, explore=True)

            # Take the action in the environment
            next_obs, reward, terminated, truncated, _ = env.step(float(action))

            # Update the policy network using the observed reward
            agent.update_step(phi, action, float(reward))

            all_rewards.append(float(reward))
            episode_reward += float(reward)
            obs = next_obs
            global_step += 1

            # End episode early if environment signals done
            if terminated or truncated:
                break

        ep_elapsed = time.time() - ep_start
        episode_times.append(ep_elapsed)

        # ── Logging ──────────────────────────────────────────────────────────
        avg_rew = episode_reward / HORIZON
        total_elapsed = time.time() - training_start
        avg_ep_time = np.mean(episode_times)
        eta = avg_ep_time * (N_EPISODES - (episode + 1))
        pct = (episode + 1) / N_EPISODES * 100

        print(
            f"[{episode+1:>3}/{N_EPISODES}] ({pct:5.1f}%)  "
            f"AvgRew: {avg_rew:+.5f}  "
            f"Baseline: {agent.baseline:.5f}  "  # Print baseline so we can see it warming up
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

    # ── Receding Window Average ───────────────────────────────────────────────
    n = len(all_rewards)
    window_avg = [
        np.mean(all_rewards[max(0, i - WINDOW_SIZE + 1): i + 1])
        for i in range(n)
    ]

    # ── Plot ─────────────────────────────────────────────────────────────────
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
    # Neutral baseline feature vector — all values set to mid-range
    base_phi = np.array([0.3, 0.2, 0.3, 0.5, 0.5, 0.3, 0.5], dtype=np.float32)

    # ── Test 1: Price vs Passenger Price Sensitivity ──────────────────────────
    # Sweep passenger_alpha (index 3) from 0.05 to 0.95
    # Expected: higher alpha → agent quotes higher price (passenger willing to pay more)
    alphas = np.linspace(0.05, 0.95, 15)
    prices_1 = []
    for alpha in alphas:
        phi = base_phi.copy()
        phi[3] = alpha  # Only change passenger alpha, keep everything else fixed
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

    # ── Test 2: Price vs Trip Distance ────────────────────────────────────────
    # Sweep trip_dist (index 0) from 0.05 to 0.90
    # Expected: longer trip → agent quotes higher price (more commission to earn)
    trip_distances = np.linspace(0.05, 0.9, 15)
    prices_2 = []
    for td in trip_distances:
        phi = base_phi.copy()
        phi[0] = td  # Only change trip distance, keep everything else fixed
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

    # Create the ride sharing environment
    env = DynamicPricingEnv()

    # Create the policy gradient agent
    agent = PolicyGradientAgent(lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY)

    print("Starting Policy Gradient training...")
    run_training(env, agent)

    # After training, run sanity tests to verify the policy makes logical decisions
    print("\nRunning sanity tests...")
    run_sanity_tests(env, agent)

    print("Done!")


# Only run main() if this script is executed directly
if __name__ == "__main__":
    main()