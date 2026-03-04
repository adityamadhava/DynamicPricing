(.venv) adityamadhava@adityamadhavas-MacBook-Pro DynamicPricing % python lin_greedy.py
Loading map and environment...
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:236: UserWarning: WARN: Box low's precision lowered by casting to float32, current low.dtype=float64
  gym.logger.warn(
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:306: UserWarning: WARN: Box high's precision lowered by casting to float32, current high.dtype=float64
  gym.logger.warn(
Starting Lin-Greedy training (200 episodes, 720 steps each)...
  Training: 10 episodes × 720 steps = 7,200 total steps
[  1/10] ( 10.0%)  AvgRew: +0.01133  ε: 0.9950  EpTime: 39.0s  Elapsed: 39s  ETA: 5m 51s
[  2/10] ( 20.0%)  AvgRew: +0.01428  ε: 0.9900  EpTime: 39.0s  Elapsed: 1m 17s  ETA: 5m 11s
[  3/10] ( 30.0%)  AvgRew: +0.01149  ε: 0.9851  EpTime: 39.0s  Elapsed: 1m 56s  ETA: 4m 32s
[  4/10] ( 40.0%)  AvgRew: +0.01572  ε: 0.9801  EpTime: 38.6s  Elapsed: 2m 35s  ETA: 3m 53s
[  5/10] ( 50.0%)  AvgRew: +0.01271  ε: 0.9752  EpTime: 38.5s  Elapsed: 3m 13s  ETA: 3m 13s
[  6/10] ( 60.0%)  AvgRew: +0.01130  ε: 0.9704  EpTime: 39.1s  Elapsed: 3m 53s  ETA: 2m 35s
[  7/10] ( 70.0%)  AvgRew: +0.01270  ε: 0.9655  EpTime: 38.2s  Elapsed: 4m 31s  ETA: 1m 56s
[  8/10] ( 80.0%)  AvgRew: +0.01186  ε: 0.9607  EpTime: 37.8s  Elapsed: 5m 8s  ETA: 1m 17s
[  9/10] ( 90.0%)  AvgRew: +0.01513  ε: 0.9559  EpTime: 38.9s  Elapsed: 5m 47s  ETA: 38s
[ 10/10] (100.0%)  AvgRew: +0.01240  ε: 0.9511  EpTime: 38.7s  Elapsed: 6m 26s  ETA: 0s
  └─ Last 10 eps mean reward: 0.01289  std: 0.03539  steps so far: 7,200

  Training complete in 6m 26s
  Total steps: 7,200  |  Mean reward: 0.01289
Saved reward curve to /Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/lin_greedy_reward.png
Done!