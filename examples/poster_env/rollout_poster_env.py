#!/usr/bin/env python3

import tqdm
import cv2
import tempfile
from pathlib import Path
import os
import numpy as np
from stable_baselines3 import PPO
from train_poster_env import make_single_env

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent.resolve()
    os.chdir(base_dir)

    env = make_single_env()
    # set the number of steps away from the goal when randomly initializing
    env.steps_away_from_goal = 8
    env.STEP_LIMIT = 25
    model = PPO.load(f"{base_dir/'ppo_nerf.zip'}")

    # rollout for a number of episodes
    obs, info = env.reset()
    episodes = 10
    output_video = "output.mp4"

    with tempfile.TemporaryDirectory() as video_folder:
        step_count = 0
        for i in tqdm.tqdm(range(episodes)):
            total_reward = 0
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                total_reward += reward

                image = env.render()
                img = image.copy()
                img = np.array((img * 255).astype(np.uint8))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(Path(video_folder) / f"step_{step_count}.png"), img)

                if done or truncated:
                    obs, info = env.reset()
                    break

        cmd = f"ffmpeg -framerate 5 -i '{video_folder}/step_%d.png' -c:v libx264 -preset slow -tune stillimage -crf 24 -vf format=yuv420p -movflags +faststart -y {video_folder}/output.mp4"
        os.system(cmd)
        os.system(f"mv {video_folder}/output.mp4 {output_video}")
        print(f"Video saved to {output_video}")
