
# modified run.py due to outdated gym

import gym
import gym_carla
import carla
import torch as th
import torch.nn as nn

from shimmy import GymV21CompatibilityV0
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CarlaMultiFrameCNN(BaseFeaturesExtractor):
    """
    Custom CNN for gym-carla's stacked observation space: (5, 256, 256, 3)
    Treats the 5 frames * 3 channels = 15 channels as input to a CNN.
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # (5, 256, 256, 3) -> merge frames+channels into one dim -> (15, 256, 256)
        n_frames, h, w, c = observation_space.shape
        in_channels = n_frames * c  # 5 * 3 = 15

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with th.no_grad():
            sample = th.zeros(1, in_channels, h, w)
            cnn_out = self.cnn(sample)
            cnn_out_dim = cnn_out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # obs shape: (batch, 5, 256, 256, 3)
        b, n_frames, h, w, c = obs.shape
        # Reshape to (batch, 15, 256, 256)
        x = obs.permute(0, 1, 4, 2, 3).reshape(b, n_frames * c, h, w)
        return self.linear(self.cnn(x))


def main():

    params = {
        "number_of_vehicles": 1,
        "number_of_walkers": 0,
        "display_size": 256,
        "max_past_step": 1,
        "dt": 0.1,
        "discrete": True,
        "discrete_acc": [-3.0, 0.0, 3.0],
        "discrete_steer": [-0.2, 0.0, 0.2],
        "continuous_accel_range": [-3.0, 3.0],
        "continuous_steer_range": [-0.3, 0.3],
        "ego_vehicle_filter": "vehicle.lincoln*",
        "port": 4000,
        "town": "Town03",
        "max_time_episode": 1000,
        "max_waypt": 12,
        "obs_range": 32,
        "lidar_bin": 0.125,
        "d_behind": 12,
        "out_lane_thres": 2.0,
        "desired_speed": 8,
        "max_ego_spawn_times": 200,
        "display_route": True,
    }

    raw_env = gym.make("carla-v0", params=params)
    env = GymV21CompatibilityV0(env=raw_env)
    env = Monitor(env)

    policy_kwargs = dict(
        features_extractor_class=CarlaMultiFrameCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_starts=1000,  # start learning after 1000 steps instead of 50000
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=2000)
    model.save("carla_dqn_model")
    print("Training finished")


if __name__ == "__main__":
    main()