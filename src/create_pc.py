import numpy as np
import os
import glob
import torch
from sklearn import preprocessing
import pymeshlab as ml
import open3d as o3d
import copy
import random
from utils import simple_clean, meshlab_to_open3d, move_to_center, rotate_data_3d, area_and_normal, sample_points_with_normal_features, farthest_point_sample

def process_file(file_path, saved_folder, degrees=None, n_pts=10000, scale=10.0):
    file_id = "tmp"
    saved_path = os.path.join(str(saved_folder), f"{file_id}.pth")
    ms = simple_clean(file_path)
    ms.apply_filter("compute_matrix_from_scaling_or_normalization",
                    uniformflag=True, axisx=1. / scale, axisy=1. / scale, axisz=1. / scale, scalecenter="origin")
    ms.apply_filter("apply_matrix_freeze")

    data = meshlab_to_open3d(ms)
    data.orient_triangles()
    data.remove_unreferenced_vertices()
    # center and rotate
    data = move_to_center(data, middle=True, in_place=True)
    if degrees:
        for d in degrees:
            data = rotate_data_3d(data, degrees=d, in_place=True)
    data = move_to_center(data, middle=True, in_place=True)

    vertices = np.array(data.vertices)
    faces = np.array(data.triangles)
    data.compute_vertex_normals()
    normals = np.array(data.vertex_normals)
    data.compute_triangle_normals()
    face_normals = np.array(data.triangle_normals)
    surface_area = data.get_surface_area()
    _, face_areas = area_and_normal(vertices, faces)


    N = vertices.shape[0]
    original_index = np.arange(N)
    if N < n_pts:
        sampled_points, sampled_face_normals = sample_points_with_normal_features(vertices, faces, face_normals, face_areas, n_points=n_pts - N)
        coord = np.concatenate((vertices, sampled_points), axis=0)
        norm = np.concatenate((normals, sampled_face_normals), axis=0)

    else:
        coord = vertices
        norm = normals
    # fps
    fps_index = farthest_point_sample(coord, n_pts)
    norm = preprocessing.normalize(norm, norm='l2')
    res = dict(coord=coord, norm=norm, fps_index=fps_index, original_index=original_index, area=surface_area)
    torch.save(res, saved_path)

if __name__ == '__main__':
    from open3d_utils import show_mesh
    # file = "./pc_operations/data/SPRING0002.obj"
    # mesh = o3d.io.read_triangle_mesh(file)
    # degrees = [(0, 0, np.pi * 5.3 / 18), (-np.pi / 2, 0, 0)]
    # data = move_to_center(mesh, middle=True, in_place=True)
    # for d in degrees:
    #     data = rotate_data_3d(data, degrees=d, in_place=True)
    # data = move_to_center(data, middle=True, in_place=True)
    # # show_mesh(data)
    # o3d.io.write_triangle_mesh("./pc_operations/data/tmp.ply", data)

    file = "./pc_operations/data/tmp.ply"
    process_file(file, saved_folder="./pc_operations/data/", degrees=None, n_pts=25000, scale=1.0)
    #
    # # o3d_mesh = o3d.io.read_triangle_mesh(file)
    # # vertices = np.array(o3d_mesh.vertices)
    # # faces = np.array(o3d_mesh.triangles)
    # # print(vertices.shape, faces.shape)
    # #
    # from open3d_utils import show_pc
    # file = "./pc_operations/data/tmp.pth"
    # data_dict = torch.load(file, weights_only=False)
    # coord = data_dict["coord"]
    # show_pc(coord)
