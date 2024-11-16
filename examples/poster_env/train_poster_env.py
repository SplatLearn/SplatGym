#!/usr/bin/env python3

from pathlib import Path
import os
import sys

base_dir = Path(__file__).parent.parent.parent.resolve()
sys.path.append(f"{base_dir / 'src'}")
from nerfgym.NeRFEnv import *
import open3d as o3d
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

config_path = base_dir / "outputs/poster/splatfacto/2024-11-15_002955/config.yml"
ply_path = base_dir / "outputs/poster/point_cloud.ply"

# convert the point cloud .ply file to a .pcd file
pcd = o3d.io.read_point_cloud(str(ply_path))
pcd_path = ply_path.with_suffix(".pcd")
o3d.io.write_point_cloud(str(pcd_path), pcd)

def make_single_env():
    min_bound = np.array([-4, -4, -4])
    max_bound = np.array([4, 4, 4])
    bbox_sides = 2
    camera_box_size = [0.01, 0.01, 0.01]
    translate_step = 0.1
    goal_xyzrpy = [
        0.3179229490458535,
        0.044831075423480554,
        -0.3,
        90.0,
        0.0,
        80,
    ]

    env = NeRFFPSEnv(
        config_path,
        pcd_path,
        min_bound=min_bound,
        max_bound=max_bound,
        bbox_sides=bbox_sides,
        camera_box_size=camera_box_size,
    )
    env.goal_xyzrpy = goal_xyzrpy
    env.TRANSLATE_STEP = translate_step
    return env


def main():
    # Set the base directory to the root of the project due to some assumptions in the code
    os.chdir(base_dir)

    # create the environment with logging
    log_dir = "logs"
    info_keywords = ["steps_away_from_goal", "reached_goal_count", "trajectory"]
    env = make_vec_env(
        lambda render_mode: make_single_env(),
        n_envs=1,
        monitor_dir=str(Path(log_dir) / "monitor") if log_dir else None,
        monitor_kwargs={"info_keywords": tuple(info_keywords)},
        env_kwargs={"render_mode": "rgb_array"},
        vec_env_cls=SubprocVecEnv,
    )

    # train the control policy
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=300000, progress_bar=True)
    model.save("ppo_nerf")


if __name__ == "__main__":
    main()
