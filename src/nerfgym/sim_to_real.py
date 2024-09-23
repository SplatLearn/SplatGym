from nerfgym.NeRFEnv import *


class Sim2RealEnv(NeRFFPSEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, camera_box_size=[0.01] * 3, **kwargs)
        self.goal_xyzrpy = [-0.025, -0.047, -0.32, 90, 0, -40]
        self.X_MIN = -2
        self.X_MAX = 2
        self.Y_MIN = -2
        self.Y_MAX = 2
        self.TRANSLATE_STEP = 0.05
        self.STEP_LIMIT = 100
        self.RANDOM_PHASE_STEPS_MAX = 20

    def step(self, action):
        self.step_count += 1

        self.move_vc(action)

        x, y, z, roll, pitch, yaw = self.vc.xyzrpy()

        # regularization by randomly perturb the camera
        x_noise = 0
        y_noise = 0
        z_noise = np.random.uniform(-0.02, 0.02)
        roll_noise = np.random.uniform(-5, 5)
        pitch_noise = np.random.uniform(-5, 5)
        yaw_noise = np.random.uniform(0, 0)

        self.vc.reset_view(
            pose=self.xyzrpy_to_pose(
                x + x_noise,
                y + y_noise,
                z + z_noise,
                roll + roll_noise,
                pitch + pitch_noise,
                yaw + yaw_noise,
            )
        )

        # generate observation image with the perturbed camera
        img = self.vc.view()
        average_red = np.sum(img[:, :, 0]) / (img.shape[0] * img.shape[1])

        # reset the camera to the original unperturbed position
        self.vc.reset_view(
            pose=self.xyzrpy_to_pose(
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
            )
        )

        # determine reward and termination
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
            np.linalg.norm(np.array([x, y, z]) - np.array(self.goal_xyzrpy[:3]))
            < self.TRANSLATE_STEP * 2
        ) and (average_red > 0.6):
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
