# Splat Gym

Splat gym is a simulator for reinforcement learning free space navigation policies in Gaussian splat environments.

It has the following main features:

- Novel View Synthesis - Generate photo-realistic images of the scene from any arbitrary camera pose.
- Collision Detection - Detect collision between the camera and the underlying scene objects.

## Usage

First, install dependencies:

```sh
> pip install -r src/requirements.txt
```

To use the gym environment, a scene must be trained using nerfstudio

```sh
> ns-train splatfacto --data <data_folder>
```

Point cloud must be obtained for the scene

```sh
> ns-train nerfacto --data <data_folder> --pipeline.model.predict-normals=True
> ns-export pointcloud --load-config <nerfacto config> --output-dir <output_dir> 
```

Then the gym environment can be constructed from python:

```python
from nerfgym.NeRFEnv import NeRFEnv

config_path = Path("...")
pcd_path = Path("...")
nerf_env = NeRFEnv(config_path, pcd_path)
```

The environment follows the [Gymnasium Env API](https://gymnasium.farama.org/api/env/) and can be used seamlessly with existing reinforcement learning libraries.

A training script has been written to use PPO to solve the free space navigation problem. This can be invoked:

```sh
> PYTHONPATH=<repo_dir>/src  python3 src/training.py train --num_training_steps 300000 --env_id free
```
