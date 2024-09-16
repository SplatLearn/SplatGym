from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import open3d as o3d
import tqdm
from scipy.spatial.transform import Rotation

from nerfgym.pybind_collision_detector import (CollisionDetector,
                                               CollisionDetectorOptions)
from nerfgym.VirtualCamera import VirtualCamera


class NeRFBaseEnv(gym.Env):
    def __init__(
        self,
        config_path: Path,
        pcd_path: Path,
        min_bound=[-14, -8, -0.6],
        max_bound=[20, 8, 5],
        bbox_sides=4,
    ):
        self.vc = VirtualCamera(config_path)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.vc.view().shape, dtype=np.uint8
        )

        # create collision detectors
        self.collision_detectors = []

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        vol = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        pcd = pcd.crop(vol)
        downpcd = pcd.voxel_down_sample(voxel_size=bbox_sides)

        for point in tqdm.tqdm(downpcd.points):
            point = np.array(point)
            opt = CollisionDetectorOptions()
            opt.x_max = bbox_sides / 2 + point[0]
            opt.x_min = -bbox_sides / 2 + point[0]
            opt.y_max = bbox_sides / 2 + point[1]
            opt.y_min = -bbox_sides / 2 + point[1]
            opt.z_max = bbox_sides / 2 + point[2]
            opt.z_min = -bbox_sides / 2 + point[2]

            c = CollisionDetector(str(pcd_path), opt)
            self.collision_detectors.append(c)

    def detect_collision(self, x, y, z, roll, pitch, yaw):
        for c in self.collision_detectors:
            if c.detectCollision(x, y, z, roll, pitch, yaw):
                return True
        return False

    def render(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.vc.view(resolution=(400, 600))


class NeRFEnv(NeRFBaseEnv):
    TRANSLATE_STEP = 0.2
    ANGLE_STEP = 45 / 2
    BBOX_SIZE = 2
    X_MIN = -BBOX_SIZE / 2
    Y_MIN = -BBOX_SIZE / 2
    Z_MIN = -0.6
    X_MAX = BBOX_SIZE / 2
    Y_MAX = BBOX_SIZE / 2
    Z_MAX = 1
    ACTION_POSITIVE_X = 0
    ACTION_NEGATIVE_X = 1
    ACTION_POSITIVE_Y = 2
    ACTION_NEGATIVE_Y = 3
    ACTION_POSITIVE_SPIN = 4
    ACTION_NEGATIVE_SPIN = 5
    ACTIONS = [
        ACTION_POSITIVE_X,
        ACTION_NEGATIVE_X,
        ACTION_POSITIVE_Y,
        ACTION_NEGATIVE_Y,
        ACTION_POSITIVE_SPIN,
        ACTION_NEGATIVE_SPIN,
    ]
    REACHED_GOAL_COUNT_THRESHOLD = 50
    STEPS_AWAY_FROM_GOAL_MAX = 15
    GOAL_REWARD = 50
    STEP_PENALTY = -0.2
    COLLISION_PENALTY = -10
    OUT_OF_BOUNDS_PENALTY = -10
    STEP_LIMIT = 200
    RANDOM_PHASE_STEPS_MAX = 11
    RANDOM_PHASE_ANGLES_MAX = 9
    COLLISION_DETECTION = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.origin = [0, 0, -0.3]
        self.goal_xyzrpy = [-0.8, 0, -0.3, 90, 0.0, -90]
        self.steps_away_from_goal = 1
        self.reached_goal_count = 0
        self.step_count = 0

        self.trajectory = []

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        self.step_count = 0
        reset_xyzrpy = self.goal_xyzrpy.copy()

        actions = [
            np.random.choice([0, 1]),
            np.random.choice([2, 3]),
            np.random.choice([4, 5]),
        ]

        repeats = [0, 0, 0]
        if self.steps_away_from_goal < self.STEPS_AWAY_FROM_GOAL_MAX:
            # progressive phase
            repeats[0] = np.random.randint(self.steps_away_from_goal + 1)
            repeats[1] = self.steps_away_from_goal - repeats[0]
            repeats[2] = np.random.randint(
                min(self.RANDOM_PHASE_ANGLES_MAX, self.steps_away_from_goal)
            )
        else:
            # random phase
            repeats[0] = np.random.randint(self.RANDOM_PHASE_STEPS_MAX)
            repeats[1] = np.random.randint(self.RANDOM_PHASE_STEPS_MAX)
            repeats[2] = np.random.randint(self.RANDOM_PHASE_ANGLES_MAX)

        for action, repeat in zip(actions, repeats):
            for i in range(repeat):
                if action == self.ACTION_POSITIVE_X:
                    reset_xyzrpy[0] += self.TRANSLATE_STEP
                elif action == self.ACTION_NEGATIVE_X:
                    reset_xyzrpy[0] -= self.TRANSLATE_STEP
                elif action == self.ACTION_POSITIVE_Y:
                    reset_xyzrpy[1] += self.TRANSLATE_STEP
                elif action == self.ACTION_NEGATIVE_Y:
                    reset_xyzrpy[1] -= self.TRANSLATE_STEP
                elif action == self.ACTION_POSITIVE_SPIN:
                    reset_xyzrpy[-1] += self.ANGLE_STEP
                elif action == self.ACTION_NEGATIVE_SPIN:
                    reset_xyzrpy[-1] -= self.ANGLE_STEP
                else:
                    raise ValueError("Invalid action")

        reset_xyzrpy[0] = np.clip(reset_xyzrpy[0], self.X_MIN, self.X_MAX)
        reset_xyzrpy[1] = np.clip(reset_xyzrpy[1], self.Y_MIN, self.Y_MAX)

        if self.COLLISION_DETECTION and self.detect_collision(*reset_xyzrpy):
            return self.reset(seed)

        return self.reset_to_xyzrpy(reset_xyzrpy)

    def reset_to_xyzrpy(self, xyzrpy):
        reset_pose = self.xyzrpy_to_pose(*xyzrpy)
        self.vc.reset_view(reset_pose)
        self.trajectory = [tuple(xyzrpy)]
        return self.vc.view(), {
            "angle": 0,
            "action": -1,
            "steps_away_from_goal": self.steps_away_from_goal,
            "reached_goal_count": self.reached_goal_count,
            "trajectory": self.trajectory,
        }

    def xyzrpy_to_pose(self, x, y, z, roll, pitch, yaw):
        # turn x, y, z, roll, pitch, yaw into a 4x4 matrix
        rotation_matrix = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_matrix()
        translation = np.array([x, y, z])
        pose = np.zeros([4, 4])
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = translation
        pose[3, 3] = 1

        return pose

    def angles_to_pose(self, angles, distance):
        x, y, z, roll, pitch, yaw = self.angles_to_xyzrpy(angles, distance)
        return self.xyzrpy_to_pose(x, y, z, roll, pitch, yaw)

    def angles_to_xyzrpy(self, angles, distance):
        azimuth = angles[0] / 180 * np.pi
        elevation = angles[1] / 180 * np.pi

        d = distance * np.cos(elevation)

        x = self.origin[0] + d * np.cos(azimuth)
        y = self.origin[1] + d * np.sin(azimuth)
        z = self.origin[2] + distance * np.sin(elevation)

        roll = 90 - elevation / np.pi * 180
        pitch = 0
        yaw = 90 + azimuth / np.pi * 180

        return x, y, z, roll, pitch, yaw

    def move_vc(self, action):
        x, y, z, roll, pitch, yaw = self.vc.xyzrpy()

        if action == self.ACTION_POSITIVE_X:
            if x <= self.X_MAX:
                self.vc.translate_x(self.TRANSLATE_STEP)
        elif action == self.ACTION_NEGATIVE_X:
            if x >= self.X_MIN:
                self.vc.translate_x(-self.TRANSLATE_STEP)
        elif action == self.ACTION_POSITIVE_Y:
            if y <= self.Y_MAX:
                self.vc.translate_y(self.TRANSLATE_STEP)
        elif action == self.ACTION_NEGATIVE_Y:
            if y >= self.Y_MIN:
                self.vc.translate_y(-self.TRANSLATE_STEP)
        elif action == self.ACTION_POSITIVE_SPIN:
            self.vc.rotate_y(self.ANGLE_STEP)
        elif action == self.ACTION_NEGATIVE_SPIN:
            self.vc.rotate_y(-self.ANGLE_STEP)
        else:
            raise ValueError("Invalid action")

    def step(self, action):
        self.step_count += 1

        self.move_vc(action)

        x, y, z, roll, pitch, yaw = self.vc.xyzrpy()
        r = self.detect_collision(x, y, z, roll, pitch, yaw)

        if r:
            # large penalty for collision
            reward = self.COLLISION_PENALTY
            terminated = True
        elif (
            x < self.X_MIN
            or x > self.X_MAX
            or y < self.Y_MIN
            or y > self.Y_MAX
            or z < self.Z_MIN
            or z > self.Z_MAX
        ):
            # large penalty for going out of bounds
            reward = self.OUT_OF_BOUNDS_PENALTY
            terminated = True
        elif np.allclose(
            np.array([x, y, z]), self.goal_xyzrpy[:3], atol=self.TRANSLATE_STEP
        ) and np.allclose(
            np.array([roll, pitch, yaw]), self.goal_xyzrpy[3:], atol=self.ANGLE_STEP
        ):
            self.reached_goal_count += 1

            if self.reached_goal_count >= self.REACHED_GOAL_COUNT_THRESHOLD:
                self.steps_away_from_goal += 1
                if self.steps_away_from_goal > self.STEPS_AWAY_FROM_GOAL_MAX:
                    self.steps_away_from_goal = self.STEPS_AWAY_FROM_GOAL_MAX
                self.reached_goal_count = 0

            reward = self.GOAL_REWARD
            terminated = True
        else:
            # negative reward for each step to discourage unnecessary movement
            reward = self.STEP_PENALTY
            terminated = False

        if self.step_count >= self.STEP_LIMIT:
            truncated = True
        else:
            truncated = False
        self.trajectory.append((x, y, z, roll, pitch, yaw))

        info = {
            "angle": 0,
            "action": action,
            "steps_away_from_goal": self.steps_away_from_goal,
            "reached_goal_count": self.reached_goal_count,
            "trajectory": self.trajectory,
        }

        return self.vc.view(), reward, terminated, truncated, info


class NeRFFPSEnv(NeRFEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(4)
        self.TRANSLATE_STEP = 0.1
        self.ANGLE_STEP = 10
        self.RANDOM_PHASE_STEPS_MAX = 15
        self.RANDOM_PHASE_ANGLES_MAX = 18
        self.STEPS_AWAY_FROM_GOAL_MAX = 26

    def move_vc(self, action):
        x, y, z, roll, pitch, yaw = self.vc.xyzrpy()
        yaw_rad = np.deg2rad(yaw)

        dx = 0
        dy = 0
        if action == 0:  # move forward
            dx = -self.TRANSLATE_STEP * np.sin(yaw_rad)
            dy = self.TRANSLATE_STEP * np.cos(yaw_rad)
        elif action == 1:  # move backwards
            dx = self.TRANSLATE_STEP * np.sin(yaw_rad)
            dy = -self.TRANSLATE_STEP * np.cos(yaw_rad)
        elif action == 2:  # Rotate left
            self.vc.rotate_y(self.ANGLE_STEP)
        elif action == 3:  # Rotate right
            self.vc.rotate_y(-self.ANGLE_STEP)
        else:
            raise ValueError("Invalid action")

        if (dx > 0 and x <= self.X_MAX) or (dx < 0 and x >= self.X_MIN):
            self.vc.translate_x(dx)
        if (dy > 0 and y <= self.Y_MAX) or (dy < 0 and y >= self.Y_MIN):
            self.vc.translate_y(dy)


class NeRFAppleEnv(NeRFFPSEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(4)

        self.apple = cv2.imread("src/apple.jpg")
        self.apple = np.clip(self.apple.astype(np.float32) / 255, 0, 1)
        self.apple = cv2.resize(self.apple, (100, 100))
        self.apple = cv2.cvtColor(self.apple, cv2.COLOR_BGR2RGB)

        cameras = self.vc.train_cameras
        fx = cameras.fx[0].to("cpu").numpy()[0]
        fy = cameras.fy[0].to("cpu").numpy()[0]
        cx = cameras.cx[0].to("cpu").numpy()[0]
        cy = cameras.cy[0].to("cpu").numpy()[0]
        self.inference_camera_width = cameras.width[0].to("cpu").numpy()[0]
        self.inference_camera_height = cameras.height[0].to("cpu").numpy()[0]
        distortion_params = cameras.distortion_params[0].to("cpu").numpy()

        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # construct a distortion matrix
        self.dist = np.array(distortion_params)

        self.apple_position = np.array([0, 0, -0.3])

        self.goal_xyzrpy = [
            self.apple_position[0],
            self.apple_position[1],
            self.apple_position[2],
            90,
            0.0,
            -90,
        ]
        self.COLLISION_DETECTION = False

    def step(self, action):
        self.step_count += 1

        self.move_vc(action)
        img = self.vc.view()
        try:
            img = self.render_apple_in_scene(img)
        except Exception as e:
            with open("error.log", "a") as f:
                f.write(str(e) + "\n")
            print(e)
        average_red = np.sum(img[:, :, 0]) / (img.shape[0] * img.shape[1])

        x, y, z, roll, pitch, yaw = self.vc.xyzrpy()

        # print(
        #     np.linalg.norm(np.array([x, y, z]) - self.apple_position),
        #     self.TRANSLATE_STEP,
        #     average_red,
        # )

        if (
            x < self.X_MIN
            or x > self.X_MAX
            or y < self.Y_MIN
            or y > self.Y_MAX
            or z < self.Z_MIN
            or z > self.Z_MAX
        ):
            # large penalty for going out of bounds
            reward = self.OUT_OF_BOUNDS_PENALTY
            terminated = True
        elif (
            np.linalg.norm(np.array([x, y, z]) - self.apple_position)
            < self.TRANSLATE_STEP * 1.1
        ) and (average_red > 0.55):
            self.reached_goal_count += 1

            if self.reached_goal_count >= self.REACHED_GOAL_COUNT_THRESHOLD:
                self.steps_away_from_goal += 1
                if self.steps_away_from_goal > self.STEPS_AWAY_FROM_GOAL_MAX:
                    self.steps_away_from_goal = self.STEPS_AWAY_FROM_GOAL_MAX
                self.reached_goal_count = 0

            reward = self.GOAL_REWARD
            terminated = True
        else:
            # negative reward for each step to discourage unnecessary movement
            reward = self.STEP_PENALTY
            terminated = False

        if self.step_count >= self.STEP_LIMIT:
            truncated = True
        else:
            truncated = False
        self.trajectory.append((x, y, z, roll, pitch, yaw))

        info = {
            "angle": 0,
            "action": action,
            "steps_away_from_goal": self.steps_away_from_goal,
            "reached_goal_count": self.reached_goal_count,
            "trajectory": self.trajectory,
        }

        return (
            img,
            reward,
            terminated,
            truncated,
            info,
        )

    def render_apple_in_scene(self, img):
        # project the centre point of apple from world frame into camera frame
        point = np.array([0, 0, 0, 1.0])
        point[:3] = self.apple_position
        # print(point)
        pose = self.vc.get_pose()
        inv_pose = np.linalg.inv(pose)
        point = inv_pose @ point

        if point[2] < 0:  # if the apple is in front of the camera
            # find a vector parallel to the image plane
            vec_up = np.array([0, 1, 0])
            vec_orth = np.cross(vec_up, point[:3])
            vec_orth = vec_orth / np.linalg.norm(vec_orth)

            # assume apple sprite is 0.08 in diameter and parallel to the image plane
            edge_point = point[:3] + vec_orth * 0.08

            points = np.array([point[:3], edge_point])
            zero_rot = Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
            d_point = cv2.projectPoints(
                points, zero_rot, np.array([0.0, 0, 0]), self.K, self.dist[:4]
            )

            w_scale = self.inference_camera_width / img.shape[1]
            h_scale = self.inference_camera_height / img.shape[0]
            scale = np.array([w_scale, h_scale])

            apple_centre = d_point[0][0][0] / scale
            apple_edge = d_point[0][1][0] / scale
            apple_radius = int(np.linalg.norm((apple_edge - apple_centre)))

            apple_centre = np.round(apple_centre).astype(int)
            apple_centre[0] = img.shape[1] - apple_centre[0]
            apple_centre[1] = img.shape[0] - apple_centre[1]

            min_x = apple_centre[0] - apple_radius
            min_y = apple_centre[1] - apple_radius
            max_x = apple_centre[0] + apple_radius
            max_y = apple_centre[1] + apple_radius

            apple_x_start = max(0, -min_x)
            apple_y_start = max(0, -min_y)

            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(img.shape[1], max_x)
            max_y = min(img.shape[0], max_y)

            # print(points, d_point[0])
            # print(apple_centre, apple_radius, min_x, min_y, max_x, max_y, img.shape)

            if apple_radius > 0:
                apple_scale = apple_radius * 2 / self.apple.shape[1]
                # print(apple_scale, apple_copy.shape)
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        apple_x = x - min_x + apple_x_start
                        apple_y = y - min_y + apple_y_start

                        orig_apple_x = int(apple_x / apple_scale)
                        orig_apple_y = int(apple_y / apple_scale)

                        color = self.apple[orig_apple_y, orig_apple_x]

                        if np.linalg.norm(color - self.apple[0, 0]) > 0.2:
                            img[y, x] = color

        return img

    def render(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.render_apple_in_scene(super().render(seed=seed, options=options))


class NeRFHemisphereEnv(NeRFBaseEnv):
    ANGLE_STEP = 10
    DISTANCE_STEP = 0.1

    def __init__(
        self,
        config_path: Path,
        origin: Tuple[float, float, float] = (0, 0, -0.3),
        goal_distance: float = 0.8,
        goal_angles: list[float] = [180, 0],
        pcd_path: Path = Path("point_cloud.pcd"),
    ):
        super().__init__(config_path, pcd_path)
        self.action_space = gym.spaces.Discrete(
            6
        )  # left right up down forwards backwards
        self.origin = origin
        self.goal_distance = goal_distance
        self.distance = self.goal_distance
        self.goal_angles = goal_angles
        self.angles = self.goal_angles

        self.reset()

    def reset(self):
        self.angles = [np.random.uniform(-180, 180), np.random.uniform(0, 90)]
        self.distance = np.random.uniform(0.5, 1.5)
        self.vc.reset_view(pose=self.angles_to_pose(self.angles, self.distance))
        info = {"angle": self.angles, "action": -1}
        return self.vc.view(), info

    def angles_to_xyzrpy(self, angles, distance):
        azimuth = angles[0] / 180 * np.pi
        elevation = angles[1] / 180 * np.pi

        d = distance * np.cos(elevation)

        x = self.origin[0] + d * np.cos(azimuth)
        y = self.origin[1] + d * np.sin(azimuth)
        z = self.origin[2] + distance * np.sin(elevation)

        roll = 90 - elevation / np.pi * 180
        pitch = 0
        yaw = 90 + azimuth / np.pi * 180

        return x, y, z, roll, pitch, yaw

    def angles_to_pose(self, angles, distance):
        x, y, z, roll, pitch, yaw = self.angles_to_xyzrpy(angles, distance)

        # turn x, y, z, roll, pitch, yaw into a 4x4 matrix
        rotation_matrix = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_matrix()
        translation = np.array([x, y, z])
        pose = np.zeros([4, 4])
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = translation
        pose[3, 3] = 1

        return pose

    def step(self, action):
        # negative reward for each step to discourage unnecessary movement
        reward = -0.2

        if action == 0:
            self.angles[0] += self.ANGLE_STEP
        elif action == 1:
            self.angles[0] -= self.ANGLE_STEP
        elif action == 2:
            self.angles[1] += self.ANGLE_STEP
        elif action == 3:
            self.angles[1] -= self.ANGLE_STEP
        elif action == 4:
            self.distance += self.DISTANCE_STEP
        elif action == 5:
            self.distance -= self.DISTANCE_STEP
        else:
            raise ValueError("Invalid action")

        self.angles[1] = np.clip(self.angles[1], 0, 90)

        while np.abs(self.angles[0]) > 180:
            if self.angles[0] > 180:
                self.angles[0] -= 360
            elif self.angles[0] < -180:
                self.angles[0] += 360

        done = False
        terminate = False

        if self.distance < 0.5 or self.distance > 1.5:
            # large penalty for going out of bounds
            reward = -1
            done = True
        self.distance = np.clip(self.distance, 0.5, 1.5)
        pose = self.angles_to_pose(self.angles, self.distance)
        self.vc.reset_view(pose=pose)

        # give a reward if close to goal reward
        if (
            np.abs(self.angles[0] - self.goal_angles[0]) < self.ANGLE_STEP
            and np.abs(self.angles[1] - self.goal_angles[1]) < self.ANGLE_STEP
            and np.abs(self.distance - self.goal_distance) < self.DISTANCE_STEP
        ):
            reward = 1
            done = True

        # detect collision
        x, y, z, roll, pitch, yaw = self.angles_to_xyzrpy(self.angles, self.distance)
        r = self.detect_collision(x, y, z, roll, pitch, yaw)

        if r:
            # large penalty for collision
            reward = -1
            done = True

        info = {"angle": self.angles, "action": action}
        terminate = done
        return self.vc.view(), reward, done, terminate, info


class NeRFCircleEnv(NeRFHemisphereEnv):
    def __init__(
        self,
        config_path: Path,
        origin: Tuple[float, float, float] = (0, 0, -0.3),
        distance: float = 0.8,
        goal_angle: float = 180,
        pcd_path: Path = Path("point_cloud.pcd"),
    ):
        super().__init__(config_path, origin, distance, [goal_angle, 0], pcd_path)
        self.action_space = gym.spaces.Discrete(2)  # left right
        self.reset()

    def reset(self, seed=None):
        self.angles = [np.random.uniform(-180, 180), 0]
        self.vc.reset_view(pose=self.angles_to_pose(self.angles, self.distance))
        info = {"angle": self.angles, "action": -1}
        return self.vc.view(), info


class NeRFPolarPlaneEnv(NeRFHemisphereEnv):
    def __init__(
        self,
        config_path: Path,
        origin: Tuple[float, float, float] = (0, 0, -0.3),
        distance: float = 0.8,
        goal_angle: float = 180,
        pcd_path: Path = Path("point_cloud.pcd"),
    ):
        super().__init__(config_path, origin, distance, [goal_angle, 0], pcd_path)
        self.action_space = gym.spaces.Discrete(4)  # left right forward backward
        self.reset()

    def reset(self, seed=None):
        self.angles = [np.random.uniform(-180, 180), 0]
        self.distance = np.random.uniform(0.5, 1.5)
        self.vc.reset_view(pose=self.angles_to_pose(self.angles, self.distance))
        info = {"angle": self.angles, "action": -1}
        return self.vc.view(), info

    def step(self, action):
        if action in [0, 1]:
            return super().step(action)
        elif action in [2, 3]:
            return super().step(action + 2)
        else:
            raise ValueError("Invalid action")


if __name__ == "__main__":
    import os
    from pathlib import Path

    from matplotlib import pyplot as plt
    from VirtualCamera import VirtualCamera

    os.chdir("/home/liyouzhou/study/GSRL")
    config_path = Path("outputs/garden_objects/splatfacto/2024-07-24_185403/config.yml")
    env = NeRFCircleEnv(config_path)
    env.reset()

    import time

    # create 10x10 subplots
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    axs = axs.ravel()

    start = time.time()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, truncated, info, done = env.step(action)
        print(f"step {i}, reward: {reward}, done: {done}")
        axs[i].imshow(obs)
    print(f"Time: {time.time() - start}")

    plt.show()
