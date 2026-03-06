import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# create environment
env = gym.make("CarRacing-v2", render_mode="rgb_array")
env = Monitor(env)

# create PPO model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# train model
model.learn(
    total_timesteps=50000,
    tb_log_name="ppo_carracing"
)

env.close()