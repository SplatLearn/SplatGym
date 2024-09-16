from pathlib import Path

import numpy as np
import torch
import torchvision
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation
from torch import Tensor


class VirtualCamera:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            for line in f:
                if "num_rays_per_chunk" in line:
                    eval_num_rays_per_chunk = int(line.split(":")[-1].strip())
        _, pipeline, _, _ = eval_setup(
            config_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
            test_mode="inference",
        )
        self.pipeline = pipeline
        self.train_cameras = pipeline.datamanager.train_dataset.cameras
        self.reset_view()

        self.high_res_image = None

    def reset_view(self, pose=None):
        if pose is not None:
            self.pose = pose
            self.pose[3, 3] = 1
        else:
            idx = len(self.train_cameras.camera_to_worlds) // 3
            self.pose = np.zeros([4, 4])
            self.pose[:3, :] = (
                self.train_cameras.camera_to_worlds[idx].to("cpu").numpy()
            )
            self.pose[3, 3] = 1

    def get_pose(self):
        return self.pose

    def translation(self):
        return self.pose[:3, 3]

    def rotation(self):
        return Rotation.from_matrix(self.pose[:3, :3])

    @torch.no_grad()
    def view(self, resolution=(64, 64)):
        pose = Tensor(np.array([self.pose[:3, :]]))
        idx = 0
        new_camera = Cameras(
            camera_to_worlds=pose,
            fx=self.train_cameras.fx[idx : idx + 1],
            fy=self.train_cameras.fy[idx : idx + 1],
            cx=self.train_cameras.cx[idx : idx + 1],
            cy=self.train_cameras.cy[idx : idx + 1],
            width=self.train_cameras.width[idx : idx + 1],
            height=self.train_cameras.height[idx : idx + 1],
            distortion_params=self.train_cameras.distortion_params[idx : idx + 1],
            camera_type=self.train_cameras.camera_type[idx : idx + 1],
        )
        outputs = self.pipeline.model.get_outputs_for_camera(new_camera)
        outputs_img = outputs["rgb"]
        outputs_img = outputs_img.permute(2, 0, 1)
        outputs_img = torchvision.transforms.Resize(resolution)(outputs_img)
        outputs_img = outputs_img.permute(1, 2, 0)
        img = outputs_img.to("cpu").numpy()

        return img

    def rotate(self, axis: str, angle: float):
        if axis == "x":
            new_r = Rotation.from_euler("xyz", [angle, 0, 0], degrees=True)
        elif axis == "y":
            new_r = Rotation.from_euler("xyz", [0, angle, 0], degrees=True)
        elif axis == "z":
            new_r = Rotation.from_euler("xyz", [0, 0, angle], degrees=True)
        else:
            raise ValueError("Invalid axis")

        mat = np.zeros([4, 4])
        mat[:3, :3] = new_r.as_matrix()
        mat[3, 3] = 1

        # apply rotation to the pose matrix
        self.pose = self.pose @ mat

    def rotate_x(self, angle: float):
        return self.rotate("x", angle)

    def rotate_y(self, angle: float):
        return self.rotate("y", angle)

    def rotate_z(self, angle: float):
        return self.rotate("z", angle)

    def translate(self, axis: str, distance: float):
        zero_r = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
        translate_vec = np.zeros([3, 1])
        if axis == "x":
            translate_vec[0] = distance
        elif axis == "y":
            translate_vec[1] = distance
        elif axis == "z":
            translate_vec[2] = distance
        else:
            raise ValueError("Invalid axis")

        mat = np.zeros([4, 4])
        mat[:3, :3] = zero_r.as_matrix()
        mat[:3, 3] = translate_vec.flatten()
        mat[3, 3] = 1

        # apply translation to the pose matrix
        self.pose = mat @ self.pose

    def translate_x(self, distance: float):
        return self.translate("x", distance)

    def translate_y(self, distance: float):
        return self.translate("y", distance)

    def translate_z(self, distance: float):
        return self.translate("z", distance)

    def xyzrpy(self):
        """
        Convert the pose matrix to x, y, z, roll, pitch, yaw
        """
        translation = self.translation()
        rotation = self.rotation()
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)

        return translation[0], translation[1], translation[2], roll, pitch, yaw
