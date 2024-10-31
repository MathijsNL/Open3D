# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np


def test_in_memory_xyz():
    # Reading/Writing bytes from bytes object
    pcb0 = b"1.0000000000 2.0000000000 3.0000000000\n4.0000000000 5.0000000000 6.0000000000\n7.0000000000 8.0000000000 9.0000000000\n"
    pc0 = o3d.io.read_point_cloud_from_bytes(pcb0, "mem::xyz")
    assert len(pc0.points) == 3
    pcb1 = o3d.io.write_point_cloud_to_bytes(pc0, "mem::xyz")
    assert len(pcb1) == len(pcb0)
    pc1 = o3d.io.read_point_cloud_from_bytes(pcb1, "mem::xyz")
    assert len(pc1.points) == 3
    # Reading/Writing bytes from PointCloud
    pc2 = o3d.geometry.PointCloud()
    pc2_points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    pc2.points = o3d.utility.Vector3dVector(pc2_points)
    pcb2 = o3d.io.write_point_cloud_to_bytes(pc2, "mem::xyz")
    assert len(pcb2) == len(pcb0)
    pc3 = o3d.io.read_point_cloud_from_bytes(pcb2, "mem::xyz")
    assert len(pc3.points) == 3
    np.testing.assert_allclose(np.asarray(pc3.points), pc2_points)


def test_in_memory_xyzn():
    # Reading/Writing bytes from bytes object
    pcb0 = b"1.0000000000 2.0000000000 3.0000000000 0.1000000000 0.2000000000 0.3000000000\n4.0000000000 5.0000000000 6.0000000000 0.4000000000 0.5000000000 0.6000000000\n7.0000000000 8.0000000000 9.0000000000 0.7000000000 0.8000000000 0.9000000000\n"
    pc0 = o3d.io.read_point_cloud_from_bytes(pcb0, "mem::xyzn")
    assert len(pc0.points) == 3
    assert len(pc0.normals) == 3
    pcb1 = o3d.io.write_point_cloud_to_bytes(pc0, "mem::xyzn")
    print(f"pcb1: {pcb1}")
    print(f"pcb0: {pcb0}")
    assert len(pcb1) == len(pcb0)
    pc1 = o3d.io.read_point_cloud_from_bytes(pcb1, "mem::xyzn")
    assert len(pc1.points) == 3
    assert len(pc1.normals) == 3
    # Reading/Writing bytes from PointCloud
    pc2 = o3d.geometry.PointCloud()
    pc2_points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    pc2_normals = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    pc2.points = o3d.utility.Vector3dVector(pc2_points)
    pc2.normals = o3d.utility.Vector3dVector(pc2_normals)
    pcb2 = o3d.io.write_point_cloud_to_bytes(pc2, "mem::xyzn")
    assert len(pcb2) == len(pcb0)
    pc3 = o3d.io.read_point_cloud_from_bytes(pcb2, "mem::xyzn")
    assert len(pc3.points) == 3
    assert len(pc3.normals) == 3
    np.testing.assert_allclose(np.asarray(pc3.points), pc2_points)
    np.testing.assert_allclose(np.asarray(pc3.normals), pc2_normals)


def test_in_memory_xyzrgb():
    # Reading/Writing bytes from bytes object  
    pcb0 = b"1.0000000000 2.0000000000 3.0000000000 255 0 0\n4.0000000000 5.0000000000 6.0000000000 0 255 0\n7.0000000000 8.0000000000 9.0000000000 0 0 255\n"
    pc0 = o3d.io.read_point_cloud_from_bytes(pcb0, "mem::xyzrgb")
    assert len(pc0.points) == 3
    assert len(pc0.colors) == 3
    pcb1 = o3d.io.write_point_cloud_to_bytes(pc0, "mem::xyzrgb")
    assert len(pcb1) == len(pcb0)
    pc1 = o3d.io.read_point_cloud_from_bytes(pcb1, "mem::xyzrgb")
    assert len(pc1.points) == 3
    assert len(pc1.colors) == 3
    # Reading/Writing bytes from PointCloud
    pc2 = o3d.geometry.PointCloud()
    pc2_points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    pc2_colors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    pc2.points = o3d.utility.Vector3dVector(pc2_points)
    pc2.colors = o3d.utility.Vector3dVector(pc2_colors)
    pcb2 = o3d.io.write_point_cloud_to_bytes(pc2, "mem::xyzrgb")
    assert len(pcb2) == len(pcb0)
    pc3 = o3d.io.read_point_cloud_from_bytes(pcb2, "mem::xyzrgb")
    assert len(pc3.points) == 3
    assert len(pc3.colors) == 3
    np.testing.assert_allclose(np.asarray(pc3.points), pc2_points)
    np.testing.assert_allclose(np.asarray(pc3.colors), pc2_colors)
