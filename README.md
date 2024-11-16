<p align="center">
    <h1 style="text-align:center">SplatGym</h1>
</p>

<p align="center">
  Open Source Neural Simulator for Robot Learning.
</p>

<p align="center">
  <br />
    <picture>
      <img src="docs/figures/logo.jpeg" width="200px">
    </picture>
  </a>
</p>

## Introduction

<p align="left">
  <br />
    <picture>
      <img src="docs/figures/splatgym_simulator_overview.drawio.png" width="600px">
    </picture>
  </a>
</p>

SplatGym is a simulator for reinforcement learning free space navigation policies in Gaussian splat environments.

It has the following main features:

- Novel View Synthesis - Generate photo-realistic images of the scene from any arbitrary camera pose.
- Collision Detection - Detect collision between the camera and the underlying scene objects.

### Software Stack

<p align="left">
  <br />
    <picture>
      <img src="docs/figures/software_stack.drawio.png" width="600px">
    </picture>
  </a>
</p>

## Installation

### Docker Container

To run a pre-built image:

```bash
docker run --gpus all \
            -u $(id -u):$(id -u) \
            -v /folder/of/your/data:/workspace/ \
            -v /home/$USER/.cache/:/home/user/.cache/ \
            -p 7007:7007 \
            --rm \
            -it \
            --shm-size=12gb \
            docker.io/liyouzhou/splat_gym:latest
```

To build image from scratch, run the following command at the root of this repository:

```bash
docker build .
```

### Native Installation

First, install nerfstudio and its dependencies by following this [guide](https://docs.nerf.studio/quickstart/installation.html). This project uses nerfstudio `1.1.3`. Then run the following command:

```bash
> sudo apt-get install swig
> pip install -r src/requirements.txt
```

Refer to [collision_detector](https://github.com/SplatLearn/collision_detector) to build the collision detector for your architecture. In your build folder, your should now have a `pybind_collision_detector.cpython-<version>-<arch>-<os>.so` file.

Now set PYTHONPATH to allow python to find all splat gym modules

```bash
> export PYTHONPATH="<path to collision_detector>/build:<path to SplatGym>/src":$PYTHONPATH
```

## Usage

To use the gym environment, a scene must be trained using nerfstudio

```bash
> ns-train splatfacto --data <data_folder>
```

Point cloud must be obtained for the scene

```bash
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
> python3 src/training.py train --num_training_steps 300000 --env_id free
```

## Citation

You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2410.19564).

If you use this library or find the documentation useful for your research, please consider citing:

```
@misc{zhou2024roboticlearningbackyardneural,
      title={Robotic Learning in your Backyard: A Neural Simulator from Open Source Components}, 
      author={Liyou Zhou and Oleg Sinavski and Athanasios Polydoros},
      year={2024},
      eprint={2410.19564},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.19564}, 
}
```
