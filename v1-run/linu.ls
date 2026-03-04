(.venv) adityamadhava@adityamadhavas-MacBook-Pro DynamicPricing % python lin_ucb.py
Loading map and environment...
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:236: UserWarning: WARN: Box low's precision lowered by casting to float32, current low.dtype=float64
  gym.logger.warn(
/Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:306: UserWarning: WARN: Box high's precision lowered by casting to float32, current high.dtype=float64
  gym.logger.warn(
Starting LinUCB training...
LinUCB Training: 10 episodes × 720 steps = 7,200 total steps
alpha_reg=1.0  alpha_ucb=2.0
[  1/10] ( 10.0%)AvgRew: +0.01139EpTime: 39.4sElapsed: 39sETA: 5m 54s
[  2/10] ( 20.0%)AvgRew: +0.01375EpTime: 38.5sElapsed: 1m 17sETA: 5m 11s
[  3/10] ( 30.0%)AvgRew: +0.01132EpTime: 37.1sElapsed: 1m 55sETA: 4m 28s
[  4/10] ( 40.0%)AvgRew: +0.01222EpTime: 38.2sElapsed: 2m 33sETA: 3m 49s
[  5/10] ( 50.0%)AvgRew: +0.01248EpTime: 39.5sElapsed: 3m 12sETA: 3m 12s
[  6/10] ( 60.0%)AvgRew: +0.01161EpTime: 39.7sElapsed: 3m 52sETA: 2m 35s
[  7/10] ( 70.0%)AvgRew: +0.01234EpTime: 39.3sElapsed: 4m 31sETA: 1m 56s
[  8/10] ( 80.0%)AvgRew: +0.01229EpTime: 40.2sElapsed: 5m 12sETA: 1m 18s
[  9/10] ( 90.0%)AvgRew: +0.01172EpTime: 39.6sElapsed: 5m 51sETA: 39s
[ 10/10] (100.0%)AvgRew: +0.01204EpTime: 39.9sElapsed: 6m 31sETA: 0s
  └─ Last 10 eps mean reward: 0.01212std: 0.03401steps so far: 7,200

Training complete in 6m 31s
Total steps: 7,200  |  Mean reward: 0.01212
Saved reward curve to /Users/adityamadhava/Downloads/SEM-6/RL/Assignments/DynamicPricing/lin_ucb_reward.png
Done.