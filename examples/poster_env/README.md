# Finding the Poster

In this example we show how to train a robot navigation policy to find a poster in an indoor scene.

## Data Preparation

Download the dataset

```console
foo@bar:~/SplatGym$ ns-download-data nerfstudio --capture-name=poster
```

Train both nerfacto and splatfacto models

```console
foo@bar:~/SplatGym$ ns-train splatfacto --data data/nerfstudio/poster
ns-train nerfacto --data data/nerfstudio/poster --pipeline.model.predict-normals=True
```

You should have now under the folder `outputs/poster` a splatfacto model and a nerfacto model. Next, extract point cloud from the nerfacto models

```console
foo@bar:~/SplatGym$ ns-export pointcloud --load-config outputs/poster/nerfacto/<session_id>/config.yml --output-dir outputs/poster --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False
```

This will create `outputs/poster/point_cloud.ply`

## Setting up parameters of the navigation problem

Run the jupyter notebook `explore_poster_env`. Follow the cells one by one. You will be guided to visualize the scene and decide on cropping parameters, step sizes etc.

## Training a model

`train_poster_env.py` contains a simple script to train a navigation policy. It outputs logs in the `logs/monitor` folder and saves the trained model as `ppo_nerf.zip`

```console
foo@bar:~/SplatGym$ cd examples/poster_env
foo@bar:~/SplatGym/examples/poster_env$ ./train_poster_env.py
```

## Test the model

`rollout_poster_env.py` is a simple script to roll out the trained policy a few episodes and produce a video.


```console
foo@bar:~/SplatGym/examples/poster_env$ ./rollout_poster_env.py
```

You will get a video as below in `output.mp4`. It shows the trained policy navigating to a position in front of poster regardless of starting position.

<video src="../../docs/videos/example_poster_env_policy.mp4" width="320" height="240" controls></video>


## Tips

- Take care to obtain a good video covering a diverse range of viewpoints to be able to train a clean NeRF environment.
- Training for large number of iterations will yield more robust policy.
- Have a look at `NeRFEnv.py` to see how you can inherit the base class and tweak the behaviour of the environment.
- More description of the project can be found in the accompanying [paper](https://arxiv.org/abs/2410.19564).
