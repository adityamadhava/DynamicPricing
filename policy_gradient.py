import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import extract_features
from RideSharing import DynamicPricingEnv

np.random.seed(42)

# Hyperparameters
FEATURE_DIM = 7
HORIZON = 720
N_EPISODES = 150
STD_FIXED = 0.1
LR = 0.001
BASELINE_DECAY = 0.99
WINDOW_SIZE = 2000
MAX_DRIVERS = 10

# PyTorch policy
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_map():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_agent.png")
    if not os.path.isfile(path):
        raise FileNotFoundError("map_agent.png not found.")
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64) / 255.0


class PolicyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(-1)


class PolicyGradientAgent:

    def __init__(self, lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY):
        self.policy = PolicyNet()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.std = std
        self.baseline_decay = baseline_decay
        self.baseline = 0.0

    def get_mean(self, phi):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(phi).astype(np.float32)).reshape(1, -1)
            return self.policy(x).item()

    def select_action(self, phi, explore=True):
        x = torch.as_tensor(np.asarray(phi).astype(np.float32)).reshape(1, -1)
        mean = self.policy(x).squeeze()
        if not explore:
            return mean.clamp(0.0, 1.0).item(), None
        dist = torch.distributions.Normal(mean, self.std)
        z = dist.sample()
        action = z.clamp(0.0, 1.0).item()
        log_prob_normal = dist.log_prob(z).sum().item()
        # Change-of-variable correction: action = sigmoid(z) => d(action)/d(z) = action*(1-action)
        # log_prob_corrected = log_prob_normal - log|d(sigmoid)/dz| = log_prob_normal - log(action*(1-action))
        eps = 1e-6
        jacobian_log = np.log(action * (1 - action) + eps)
        log_prob_corrected = log_prob_normal - jacobian_log
        return action, log_prob_corrected

    def update_step(self, phi, action, reward):
        phi_t = torch.as_tensor(np.asarray(phi).astype(np.float32)).reshape(1, -1)
        mean = self.policy(phi_t).squeeze()
        dist = torch.distributions.Normal(mean, self.std)
        action_t = torch.tensor([action], dtype=torch.float32, device=mean.device)
        log_p = dist.log_prob(action_t).sum()
        eps = 1e-6
        jacobian_log = np.log(action * (1 - action) + eps)
        log_prob_corrected = log_p - jacobian_log
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        loss = -(reward - self.baseline) * log_prob_corrected
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_training(env, map_img, agent):
    all_rewards = []
    for episode in range(N_EPISODES):
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
        if (episode + 1) % 10 == 0:
            avg_rew = episode_reward / HORIZON
            print(f"Episode {episode + 1}/{N_EPISODES}  Avg reward: {avg_rew:.6f}  Baseline: {agent.baseline:.6f}")

    n = len(all_rewards)
    window_avg = [np.mean(all_rewards[max(0, i - WINDOW_SIZE + 1) : i + 1]) for i in range(n)]
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
    base_phi = np.array([0.3, 0.2, 0.25, 0.15, 0.5, 0.4, 0.5], dtype=np.float32)  # trip_dist, min_d, mean_d, pass_alpha, mean_d_alpha, min_d_alpha, n_drivers

    # Test 1: vary passenger alpha (index 3)
    alphas = np.linspace(0.1, 2.0, 10)
    prices_1 = []
    for alpha in alphas:
        phi = base_phi.copy()
        phi[3] = alpha
        price = agent.get_mean(phi)
        prices_1.append(price)
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, prices_1, "o-", color="teal")
    plt.title("Sanity Test 1: Price vs Passenger Sensitivity (alpha)")
    plt.xlabel("Passenger alpha")
    plt.ylabel("Quoted price")
    plt.grid(True)
    out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_test_1.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # Test 2: vary trip distance (index 0)
    trip_distances = np.linspace(0.05, 0.8, 10)
    prices_2 = []
    for td in trip_distances:
        phi = base_phi.copy()
        phi[0] = td
        price = agent.get_mean(phi)
        prices_2.append(price)
    plt.figure(figsize=(6, 4))
    plt.plot(trip_distances, prices_2, "o-", color="coral")
    plt.title("Sanity Test 2: Price vs Trip Distance")
    plt.xlabel("Trip distance (normalized)")
    plt.ylabel("Quoted price")
    plt.grid(True)
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_test_2.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


def main():
    print("Loading map and environment")
    map_img = load_map()
    env = DynamicPricingEnv()
    agent = PolicyGradientAgent(lr=LR, std=STD_FIXED, baseline_decay=BASELINE_DECAY)
    print("Starting Policy Gradient training (150 episodes)")
    run_training(env, map_img, agent)
    print("Running sanity tests")
    run_sanity_tests(env, agent)
    print("Done")


if __name__ == "__main__":
    main()


# ========== TensorFlow/Keras fallback (if PyTorch not installed) ==========
# Uncomment and use the block below when PyTorch is not available.
#
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
#
# class PolicyNetKeras(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.d1 = layers.Dense(64, activation="relu")
#         self.d2 = layers.Dense(32, activation="relu")
#         self.d3 = layers.Dense(1, activation="sigmoid")
#
#     def call(self, x):
#         x = self.d1(x)
#         x = self.d2(x)
#         return self.d3(x)
#
# class PolicyGradientAgentKeras:
#     def __init__(self, lr=0.001, std=0.1, baseline_decay=0.99):
#         self.policy = PolicyNetKeras()
#         self.optimizer = keras.optimizers.Adam(lr)
#         self.std = std
#         self.baseline = 0.0
#         self.baseline_decay = baseline_decay
#
#     def get_mean(self, phi):
#         x = tf.convert_to_tensor(np.asarray(phi).astype(np.float32).reshape(1, -1))
#         return float(self.policy(x).numpy().flat[0])
#
#     def select_action(self, phi, explore=True):
#         x = tf.convert_to_tensor(np.asarray(phi).astype(np.float32).reshape(1, -1))
#         mean = self.policy(x).numpy().flat[0]
#         if not explore:
#             return np.clip(mean, 0.0, 1.0), None
#         action = np.clip(np.random.normal(mean, self.std), 0.0, 1.0).item()
#         log_p = -0.5 * ((action - mean) / self.std) ** 2 - np.log(self.std * np.sqrt(2 * np.pi))
#         jacobian_log = np.log(action * (1 - action) + 1e-6)
#         log_prob_corrected = log_p - jacobian_log
#         return action, log_prob_corrected
#
#     def update_step(self, phi, action, reward):
#         with tf.GradientTape() as tape:
#             x = tf.convert_to_tensor(np.asarray(phi).astype(np.float32).reshape(1, -1))
#             mean = self.policy(x)
#             dist = tfp.distributions.Normal(mean, self.std)
#             log_p = dist.log_prob(tf.constant([[action]], dtype=tf.float32))
#             jacobian_log = np.log(action * (1 - action) + 1e-6)
#             log_prob_corrected = tf.reduce_sum(log_p) - jacobian_log
#             self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
#             loss = -(reward - self.baseline) * log_prob_corrected
#         grads = tape.gradient(loss, self.policy.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
#
# Then in main(), use PolicyGradientAgentKeras() and agent.update_step(phi, action, reward).
# Requires: pip install tensorflow tensorflow-probability
