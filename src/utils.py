import open3d as o3d
import numpy as np
import pymeshlab as ml
import copy
import random


def simple_clean(file_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(file_path)
    for f in ['meshing_remove_duplicate_vertices',
              'meshing_remove_duplicate_faces',
              'meshing_remove_unreferenced_vertices',
              'meshing_remove_folded_faces',
              'meshing_remove_null_faces',
              'meshing_repair_non_manifold_edges']:
        ms.apply_filter(f)
    return ms


def meshlab_to_open3d(ms):
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return o3d_mesh

def move_to_center(data_3d, middle=True, target_point=None, in_place=True):
    if target_point:
        center_coord = target_point
    else:
        if middle:
            max_coord = data_3d.get_max_bound()
            min_coord = data_3d.get_min_bound()
            center_coord = (max_coord + min_coord) / 2
        else:
            center_coord = data_3d.get_center()

    if in_place:
        data_3d.translate(-center_coord, relative=True)
    else:
        data_3d = copy.deepcopy(data_3d).translate(-center_coord, relative=True)

    return data_3d

def rotate_data_3d(data_3d, degrees=(-np.pi / 2, np.pi, 0), center=(0, 0, 0), in_place=True):
    rotation_matrix = data_3d.get_rotation_matrix_from_axis_angle(degrees)

    if in_place:
        data_3d.rotate(rotation_matrix, center=center)
    else:
        data_3d = copy.deepcopy(data_3d).rotate(rotation_matrix)
    return data_3d

def area_and_normal(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-face normals and areas for a triangular mesh.

    Given mesh vertices and triangular faces, this function computes the
    outward face normals (unit vectors) and the corresponding face areas.
    Faces with zero area (degenerate triangles) receive a zero normal.

    Args:
        vertices: np.ndarray
            Array of shape (N, 3) containing vertex coordinates (XYZ).
        faces: np.ndarray
            Array of shape (M, 3) containing vertex indices for each
            triangular face. Each row is a triplet of integer indices into ``vertices``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - face_normals: Array of shape (M, 3) with unit normal vectors
              for each face. Degenerate faces have a zero vector.
            - face_areas: Array of shape (M,) with the area of each face.
    """
    cross_product = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                             vertices[faces[:, 2]] - vertices[faces[:, 1]])   # [M, 3]
    cross_product_normal = np.sqrt(np.sum(cross_product ** 2, axis=1))        # [M, ]
    cross_product_normal_broadcast = cross_product_normal[:, np.newaxis]      # [M, 1]
    # if cross product normal is 0, the result is zero
    face_normals = np.divide(cross_product, cross_product_normal_broadcast,
                             out=np.zeros_like(cross_product), where=cross_product_normal_broadcast != 0)   # [M ,3]
    face_areas = cross_product_normal * 0.5   # [M, ]
    return face_normals, face_areas


def sample_points_with_normal_features(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray,
                                       face_areas: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on a mesh surface with associated face normals.

        Points are sampled on the triangular mesh defined by ``vertices`` and
        ``faces``. Triangles are chosen with probability proportional to their
        surface area, and points are sampled uniformly within each selected
        triangle using random barycentric coordinates. The function returns
        both the sampled 3D points and the corresponding per-point normals,
        taken from the face normals.

        Args:
            vertices: Array of shape (V, 3) containing the mesh vertex
                coordinates (XYZ).
            faces: Array of shape (F, 3) containing vertex indices for each
                triangular face. Each row is a triplet of integer indices into
                ``vertices``.
            face_normals: Array of shape (F, 3) with the normal vector for
                each face, typically unit-length.
            face_areas: Array of shape (F,) with the area of each face.
            n_points: Number of points to sample on the mesh surface.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - sampled_points: Array of shape (n_points, 3) with the sampled
                  XYZ coordinates on the mesh surface.
                - sampled_normals: Array of shape (n_points, 3) with the
                  corresponding normal vectors (one per sampled point), copied
                  from the selected face normals.
        """
    prob = face_areas / np.sum(face_areas)
    index = np.random.choice(faces.shape[0], size=n_points, replace=True, p=prob)
    sampled_faces = faces[index]
    sampled_face_normals = face_normals[index]  # [n_points, 3]

    sampled_points = []
    for sampled_face in sampled_faces:
        v1_idx, v2_idx, v3_idx = sampled_face
        v1, v2, v3 = vertices[v1_idx], vertices[v2_idx], vertices[v3_idx]
        s, t = sorted([random.random(), random.random()])
        f_v = lambda i: s * v1[i] + (t - s) * v2[i] + (1 - t) * v3[i]

        sampled_points.append([f_v(0), f_v(1), f_v(2)])
    sampled_points = np.array(sampled_points)
    return sampled_points, sampled_face_normals

def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """Select points using Farthest Point Sampling (FPS).

    This function selects `npoint` indices from an input point cloud such that
    each newly selected point is as far as possible (in Euclidean distance)
    from the already selected set. Only the first three dimensions of
    `point` are used as XYZ coordinates.

    Args:
        point: np.ndarray
            Point cloud array of shape (N, D). Only the first three columns
            are interpreted as XYZ coordinates, so D must be at least 3.
        npoint: np.ndarray
            Number of points to sample. Must satisfy 1 <= npoint <= N.

    Returns:
        np.ndarray:
            Array of shape (npoint,) containing the indices of the
            sampled points (dtype int32).

    Raises:
        AssertionError: If `npoint` is greater than the number of input points N.
    """
    N, D = point.shape
    assert N >= npoint
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids.astype(np.int32)