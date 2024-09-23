#! /usr/bin/env python3

import time
from pathlib import Path

import fire
import numpy as np
import viser
import viser.transforms as tf
from scipy.spatial.transform import Rotation

from training import make_single_env


def qvec2rotmat(qvec):
    return Rotation.from_quat(qvec).as_matrix()


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[0, 3] = -camera.position[1]
    c2w[1, 3] = camera.position[0]
    c2w[2, 3] = camera.position[2]
    return c2w


def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


class ViserGUI:
    def __init__(
        self,
        config_path: Path = Path(
            "outputs/garden_objects/splatfacto/2024-07-24_185403/config.yml"
        ),
        pcd_path: Path = Path("data/garden_objects/point_cloud.pcd"),
    ):
        self.server = viser.ViserServer()
        self.server.configure_theme(dark_mode=True)

        self.position_text = self.server.add_gui_text(
            label="Position",
            initial_value="0",
            disabled=True,
            hint="Camera position",
        )
        self.pose_text = self.server.add_gui_text(
            label="Pose",
            initial_value="0",
            disabled=True,
            hint="Camera pose",
        )
        self.reset_position_text = self.server.add_gui_text(
            label="Reset Position",
            initial_value="0",
            disabled=True,
            hint="Camera position",
        )
        self.reset_pose_text = self.server.add_gui_text(
            label="Reset Pose",
            initial_value="0",
            disabled=True,
            hint="Camera pose",
        )
        self.reward_text = self.server.add_gui_text(
            label="Reward",
            initial_value="0",
            disabled=True,
            hint="Reward",
        )
        self.reset_button = self.server.add_gui_button("Reset Env")
        self.reset_button.on_click(self.reset_env)

        self.env = make_single_env(
            "snake", 1, config_path=config_path, pcd_path=pcd_path
        )

        self.fwd_button = self.server.add_gui_button("forward")
        self.bwd_button = self.server.add_gui_button("backward")
        self.left_button = self.server.add_gui_button("left")
        self.right_button = self.server.add_gui_button("right")

        self.fwd_button.on_click(lambda _: self.navigate("Forward"))
        self.bwd_button.on_click(lambda _: self.navigate("Backward"))
        self.left_button.on_click(lambda _: self.navigate("Left"))
        self.right_button.on_click(lambda _: self.navigate("Right"))

        self.speed_slider = self.server.add_gui_slider(
            label="Speed", min=1, max=10, step=1, initial_value=3
        )

        self.need_update = True

        self.snake_button = self.server.add_gui_checkbox("Snake", initial_value=False)
        self.snake_button.on_update(self.toggle_snake)
        self.snake_step_time = 0

        self.reward = 0

    def toggle_snake(self, _):
        pass

    def navigate(self, move):
        action = -1
        if move == "Forward":
            action = 0
        elif move == "Backward":
            action = 1
        elif move == "Left":
            action = 2
        elif move == "Right":
            action = 3

        img, reward, terminated, truncated, info = self.env.step(action)
        self.reward += reward

    def reset_env(self, _):
        self.env.reset()
        self.need_update = True
        x, y, z, roll, pitch, yaw = self.env.vc.xyzrpy()
        self.reset_pose_text.value = f"{roll:.2f}, {pitch:.2f}, {yaw:.2f}"
        self.reset_position_text.value = f"{x:.2f}, {y:.2f}, {z:.2f}"
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = np.array([x, y, z])
                client.camera.wxyz = Rotation.from_euler(
                    "xyz", np.array([roll, pitch, yaw]), degrees=True
                ).as_quat(False)

    def update(self):
        if self.need_update:
            for client in self.server.get_clients().values():
                camera = client.camera
                # convert quaternion to rotation matrix
                x, y, z, roll, pitch, yaw = self.env.vc.xyzrpy()
                with client.atomic():
                    client.camera.position = np.array([-y, x, z])
                    client.camera.wxyz = Rotation.from_euler(
                        "xyz", np.array([roll, yaw, pitch]), degrees=True
                    ).as_quat(False)

                # self.env.vc.reset_view(pose=c2w)
                img = self.env.render()
                client.set_background_image(img, format="jpeg")

                c2w = get_c2w(camera)
                rotmat = c2w[:3, :3]
                rot = Rotation.from_matrix(rotmat).as_euler("xyz")
                np.rad2deg(rot, out=rot)
                x, y, z = c2w[:3, 3]

                self.position_text.value = f"{x:.2f}, {y:.2f}, {z:.2f}"
                self.pose_text.value = f"{rot[0]:.2f}, {rot[1]:.2f}, {rot[2]:.2f}"

                now = time.time()
                if (
                    self.snake_button.value
                    and now - self.snake_step_time > 1 / self.speed_slider.value
                ):
                    self.snake_step_time = now
                    img, reward, terminated, truncated, info = self.env.step(0)
                    self.reward += reward

                self.reward_text.value = f"{self.reward:.2f}"

        return


def main(
    config_path: Path = Path(
        "outputs/garden_objects/splatfacto/2024-07-24_185403/config.yml"
    ),
    pcd_path: Path = Path("data/garden_objects/point_cloud.pcd"),
):
    gui = ViserGUI(config_path=config_path, pcd_path=pcd_path)

    while True:
        gui.update()


def play_connect_4(server: viser.ViserServer) -> None:
    return


def play_tic_tac_toe(server: viser.ViserServer) -> None:
    return


if __name__ == "__main__":
    fire.Fire(main)
