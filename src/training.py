import datetime
import json
import os
import pickle
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack)
from tqdm import tqdm

from nerfgym.NeRFEnv import *
from nerfgym.pybind_collision_detector import (CollisionDetector,
                                               CollisionDetectorOptions)
from nerfgym.VirtualCamera import VirtualCamera


def make_single_env(env_id, seed, log_dir=None, render_mode="rgb_array"):
    os.chdir("/home/liyouzhou/study/GSRL")
    config_path = Path("outputs/garden_objects/splatfacto/2024-07-24_185403/config.yml")
    pcd_path = Path("data/garden_objects/point_cloud.pcd")
    min_bound = [-14, -8, -0.6]
    max_bound = [20, 8, 5]
    bbox_sides = 4

    if env_id == "circle":
        env = NeRFCircleEnv(
            config_path,
            pcd_path=pcd_path,
        )
    elif env_id == "hemisphere":
        env = NeRFHemisphereEnv(
            config_path,
            pcd_path=pcd_path,
        )
    elif env_id == "plane":
        env = NeRFPolarPlaneEnv(
            config_path,
            pcd_path=pcd_path,
        )
    elif env_id == "free":
        env = NeRFEnv(
            config_path,
            pcd_path=pcd_path,
            min_bound=min_bound,
            max_bound=max_bound,
            bbox_sides=bbox_sides,
        )
    elif env_id == "fps":
        env = NeRFFPSEnv(
            config_path,
            pcd_path=pcd_path,
            min_bound=min_bound,
            max_bound=max_bound,
            bbox_sides=bbox_sides,
        )
    elif env_id == "apple":
        env = NeRFAppleEnv(
            config_path,
            pcd_path=pcd_path,
            min_bound=min_bound,
            max_bound=max_bound,
            bbox_sides=bbox_sides,
        )
    else:
        raise ValueError(f"Unknown environment {env_id}")

    env.reset()

    return env


def make_env(env_id, seed, log_dir=None, n_envs=1, n_stack=1):
    info_keywords = ["angle", "action"]
    if env_id in ["free", "fps", "apple"]:
        info_keywords += ["steps_away_from_goal", "reached_goal_count", "trajectory"]

    env = make_vec_env(
        lambda render_mode: make_single_env(
            env_id, seed, log_dir=log_dir, render_mode=render_mode
        ),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=Path(log_dir) / "monitor" if log_dir else None,
        monitor_kwargs={"info_keywords": tuple(info_keywords)},
        env_kwargs={"render_mode": "rgb_array"},
    )

    if n_stack > 1:
        env = VecFrameStack(env, n_stack=8)

    return env


def create_model(env, alg="ppo"):
    if alg == "ppo":
        return PPO("CnnPolicy", env, verbose=1)
    elif alg == "dqn":
        return DQN("CnnPolicy", env, verbose=1)
    elif alg == "a2c":
        return A2C("CnnPolicy", env, verbose=1)


def train(
    env_id="circle",
    num_training_steps=2048 * 40,
    alg="ppo",
    n_stacks=1,
):
    params = {"num_training_steps": num_training_steps, "env_id": env_id, "alg": alg}
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = Path(f"logs/{timestamp}")
    log_folder.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(log_folder / "params.json", "w"), indent=4)
    env = make_env(env_id, 1, log_dir=log_folder, n_stack=n_stacks)
    model = create_model(env, alg)
    model.learn(num_training_steps, progress_bar=True)

    policyFileName = str(log_folder / "policy.zip")
    print(f"Saving policy {policyFileName}")
    model.save(policyFileName)


def rollout(policy_file_name, steps=1000, decorate=True):
    policy_file_name = Path(policy_file_name)
    if policy_file_name.is_dir():
        policy_file_name = policy_file_name / "policy.zip"
    log_folder = Path(policy_file_name).parent
    video_folder = log_folder / "video"
    video_folder.mkdir(parents=True, exist_ok=True)

    with open(log_folder / "params.json") as f:
        params = json.load(f)

    env = make_single_env(params["env_id"], 1, log_dir=video_folder)
    model = PPO.load(policy_file_name)
    env.steps_away_from_goal = 20

    # rollout for a number of steps
    obs, info = env.reset()
    step_count = 0
    total_reward = 0
    for i in tqdm(range(steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        total_reward += reward
        image = env.render()
        img = image.copy()
        img = np.array((img * 255).astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if decorate:
            start_point = np.array((300, 300))
            angle = 0
            length = 100
            if action == 1:  # turn left
                angle = 180
            elif action == 2:  # move forward
                angle = 30
            elif action == 3:  # move backward
                angle = 210

            angle = angle * np.pi / 180
            end_point = (
                start_point + length * np.array([np.cos(angle), np.sin(angle)])
            ).astype(int)

            # cv2.arrowedLine(
            #     img,
            #     list(start_point),
            #     list(end_point),
            #     (255, 255, 255),
            #     2,
            # )
            for s, y in zip(
                [
                    # f"step: {step_count}, angles: {info['angle'][0]:0.2f} {info['angle'][1]:0.2f}",
                    f"step: {step_count}, r: {total_reward:0.2f}",
                    f"reward: {reward}, done: {done}",
                ],
                [30, 60],
            ):
                cv2.putText(
                    img,
                    s,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        if done or truncated:
            with open(video_folder / f"step_{i}_episode.json", "wb") as f:
                pickle.dump(info, f)
            obs, info = env.reset()
            print(
                f"Episode, step: {step_count}, reward: {total_reward}, steps_away_from_goal: {info['steps_away_from_goal']}, reached_goal_count: {info['reached_goal_count']}"
            )
            step_count = 0
            total_reward = 0

        cv2.imwrite(str(video_folder / f"step_{i}.png"), img)

    cmd = f"ffmpeg -framerate 5 -i '{video_folder}/step_%d.png' -c:v libx264 -preset slow -tune stillimage -crf 24 -vf format=yuv420p -movflags +faststart -y {log_folder}/output.mp4"
    os.system(cmd)


if __name__ == "__main__":
    import fire

    fire.Fire()
