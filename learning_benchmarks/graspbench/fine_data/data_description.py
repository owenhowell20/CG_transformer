import numpy as np
import open3d as o3d
import os
import sys
import torch
from transform import Rotation, Transform


def to_o3d_pcd(xyz, normals=None):
    """
    Convert array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def line_as_cylinder(start, end, radius=0.04, resolution=20, color=[0.2, 0.8, 0.2]):
    """
    Create a cylinder mesh between two 3D points.

    Args:
        start (np.array): 3D coordinates of the start point.
        end (np.array): 3D coordinates of the end point.
        radius (float): Radius (thickness) of the cylinder.
        resolution (int): Cylinder resolution.
        color (list): RGB list for the cylinder color.

    Returns:
        o3d.geometry.TriangleMesh: A cylinder mesh between start and end.
    """
    # Create unit cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=1.0, resolution=resolution
    )
    cylinder.paint_uniform_color(color)

    # Direction and length
    start = np.array(start)
    end = np.array(end)
    vec = end - start
    height = np.linalg.norm(vec)

    # Align cylinder with direction vector
    cylinder.scale(height, center=(0, 0, 0))
    midpoint = (start + end) / 2
    direction = vec / height

    # Default cylinder axis is [0, 0, 1]
    default_dir = np.array([0, 0, 1])
    if not np.allclose(direction, default_dir):
        rot_axis = np.cross(default_dir, direction)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm < 1e-6:
            # Opposite direction
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * default_dir)
        else:
            rot_axis /= rot_axis_norm
            angle = np.arccos(np.clip(np.dot(default_dir, direction), -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_axis)
        cylinder.rotate(R, center=(0, 0, 0))

    cylinder.translate(midpoint)
    return cylinder


for i in range(100):
    data = np.load(os.path.join("./pts_1024", "scene_{}.npz".format(i)))
    print("data keys:", data.files)
    print("pcd points:", data["pos"].shape)
    print("pcd normals:", data["normals"].shape)
    print("idx:", data["idx"])
    print("contact:", data["contact"])
    print("depth_projection:", data["depth_projection"])
    print("quat:", data["quat"])
    print("rotation:", data["rotation"])

    # visualize the pcd and the gripper pose
    pos = data["pos"]
    normals = data["normals"]
    pcd = to_o3d_pcd(pos, normals=normals)
    colors = np.zeros_like(pos)
    colors[:, :] = np.asarray([255, 165, 0]) / 255
    colors[data["idx"]] = np.asarray([255, 0, 0]) / 255
    colors[data["contact"]] = np.asarray([0, 255, 0]) / 255
    rotation = Rotation.from_quat(data["quat"])
    translation = 0.065 - data["depth_projection"]
    translation = -translation * rotation.as_matrix()[:, -1] + data["pos"][data["idx"]]

    cylinder1 = line_as_cylinder(
        translation, translation - rotation.as_matrix()[:, -1] * 0.03
    )
    cylinder2 = line_as_cylinder(
        translation, translation + rotation.as_matrix()[:, 1] * 0.04
    )
    cylinder3 = line_as_cylinder(
        translation, translation - rotation.as_matrix()[:, 1] * 0.04
    )
    cylinder4 = line_as_cylinder(
        translation + rotation.as_matrix()[:, 1] * 0.04,
        translation
        + rotation.as_matrix()[:, 1] * 0.04
        + rotation.as_matrix()[:, -1] * 0.06,
    )
    cylinder5 = line_as_cylinder(
        translation - rotation.as_matrix()[:, 1] * 0.04,
        translation
        - rotation.as_matrix()[:, 1] * 0.04
        + rotation.as_matrix()[:, -1] * 0.06,
    )

    # print(translation)
    trans_m = Transform(
        rotation=Rotation.from_quat(data["quat"]), translation=translation
    )
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.06, origin=[0, 0, 0]
    )
    mesh_frame.transform(trans_m.as_matrix())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd, cylinder1, cylinder2, cylinder3, cylinder4, cylinder5]
    )
