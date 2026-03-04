(.venv) adityamadhava@adityamadhavas-MacBook-Pro DynamicPricing % python policy_gradient.py
Loading map and environment...
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:236: UserWarning: WARN: Box low's precision lowered by casting to float32, current low.dtype=float64
  gym.logger.warn(
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:306: UserWarning: WARN: Box high's precision lowered by casting to float32, current high.dtype=float64
  gym.logger.warn(
Starting Policy Gradient training...
Policy Gradient Training: 5 episodes × 720 steps = 3,600 total steps
lr=0.001  std=0.1  baseline_decay=0.99
[  1/5] ( 20.0%)AvgRew: +0.01474Baseline: 0.01173EpTime: 40.2sElapsed: 40sETA: 2m 40s
[  2/5] ( 40.0%)AvgRew: +0.01833Baseline: 0.01906EpTime: 38.9sElapsed: 1m 19sETA: 1m 58s
[  3/5] ( 60.0%)AvgRew: +0.01804Baseline: 0.01708EpTime: 37.9sElapsed: 1m 57sETA: 1m 18s
[  4/5] ( 80.0%)AvgRew: +0.01797Baseline: 0.01730EpTime: 40.5sElapsed: 2m 37sETA: 39s
[  5/5] (100.0%)AvgRew: +0.01830Baseline: 0.01771EpTime: 40.3sElapsed: 3m 17sETA: 0s
Training complete in 3m 17s
Total steps: 3,600  |  Mean reward: 0.01747
Saved reward curve to /Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/policy_gradient_reward.png

Running sanity tests...
Saved /Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/sanity_test_1.png
Saved /Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/sanity_test_2.png
Done!