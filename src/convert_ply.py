#! /usr/bin/env python3

import fire
import open3d as o3d


def main(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace(".ply", ".pcd")
    pcd = o3d.io.read_point_cloud(input_file)
    o3d.io.write_point_cloud(output_file, pcd)


if __name__ == "__main__":
    fire.Fire(main)
