import open3d as o3d
import numpy as np
import torch
from open3d.visualization import gui
from open3d.visualization import rendering
from matplotlib import colormaps
import copy


def show_pc(coord, color=None, normal=None, saved=False, filename="tmp_.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([pcd])

    if saved:
        o3d.io.write_point_cloud(filename, pcd)

def normalized_coord(coord):
    centroid = np.mean(coord, axis=0)
    coord -= centroid
    # m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
    m = np.sqrt(np.max(np.sum(coord ** 2, axis=1)))
    coord = coord / m
    return coord

def move_positive(coord):
    centroid = np.min(coord, axis=0)
    coord -= centroid
    return coord


def show_data(data_dict):
    print(data_dict.keys())
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v))
        else:
            print(k, v)


def rotate_data_3d(data_3d, degrees=(-np.pi / 2, np.pi, 0), center=(0, 0, 0)):
    rotation_matrix = data_3d.get_rotation_matrix_from_axis_angle(degrees)
    data_3d.rotate(rotation_matrix, center=center)
    return data_3d

def show_mesh(mesh):
    o3d.visualization.draw_geometries([mesh])


def make_demo_sphere_pcd():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(5000)
    return pcd


def attach_label_layout(win, scene_widget, label):
    """
    Reusable layout function:
    - Scene fills entire content area.
    - Label is a horizontal bar near the top that stretches with window size.
    """

    def on_layout(ctx):
        r = win.content_rect
        # Scene fills the window
        scene_widget.frame = r

        # Size label to its preferred size and pin to top-left with some margin
        pref = label.calc_preferred_size(
            ctx, gui.Widget.Constraints()  # constraints are required by API
        )
        margin = int(8 * win.scaling)
        label.frame = gui.Rect(r.x + margin, r.y + margin, pref.width, pref.height)

    win.set_on_layout(on_layout)

def create_pcd_from_np(coord):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    return pcd

def create_arrow(length, shaft_ratio=0.7, color=(1.0, 0, 0)):
    base_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length*0.02,
        cone_radius=length*0.04,
        cylinder_height=length*shaft_ratio,
        cone_height=length*(1-shaft_ratio)
    )
    base_arrow.compute_vertex_normals()
    base_arrow.paint_uniform_color(color)
    return base_arrow


def rotation_matrix_from_vectors(z_dir: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that takes z_dir -> target_dir.
    Both should be 3D unit vectors.
    """
    z_dir = z_dir / np.linalg.norm(z_dir)
    target_dir = target_dir / np.linalg.norm(target_dir)

    v = np.cross(z_dir, target_dir)
    c = np.dot(z_dir, target_dir)
    if c > 0.9999:
        # almost same direction
        return np.eye(3)
    if c < -0.9999:
        # opposite: rotate 180 around any perpendicular axis, e.g. x-axis
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.float64)

    s = np.linalg.norm(v)
    v = v / s
    # Rodrigues' formula
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + s * K + ((1 - c) * (K @ K))
    return R


def create_normal_arrows_from_pcd(pcd, normal, num_normal=200, indices=None, length=None):
    coord = np.asarray(pcd.points)

    if indices is None:
        if num_normal < len(coord):
            indices = np.random.choice(np.arange(len(coord)), size=num_normal, replace=False)
        else:
            indices = np.arange(len(coord))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    if length is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        length = diag * 0.05  # 5% of diagonal, tweak as needed

    base_arrow = create_arrow(length)
    arrows = o3d.geometry.TriangleMesh()
    # the direction the arrow is currently pointing in its local coordinates.
    z_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for idx in indices:
        position = coord[idx]
        direction = normal[idx]
        norm_direction = np.linalg.norm(direction)
        if norm_direction < 1e-8:
            continue
        R = rotation_matrix_from_vectors(z_dir, direction)
        arrow_i = copy.deepcopy(base_arrow)
        arrow_i.rotate(R, center=np.zeros(3))
        arrow_i.translate(position)  # base at the point

        arrows += arrow_i

    return arrows, indices, length

def apply_default_color(o3d_obj):
    if isinstance(o3d_obj, o3d.geometry.TriangleMesh):
        if not o3d_obj.has_vertex_colors():
            o3d_obj.paint_uniform_color([0.6, 0.6, 0.6])
    elif isinstance(o3d_obj, o3d.geometry.PointCloud):
        if not o3d_obj.has_colors():
            pts = np.asarray(o3d_obj.points)
            assert pts.shape[0] > 0
            z = pts[:, 2]
            z_min, z_max = z.min(), z.max()
            z_norm = (z - z_min) / (z_max - z_min + 1e-9)

            colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
            o3d_obj.colors = o3d.utility.Vector3dVector(colors)

    return o3d_obj

def apply_normal(o3d_obj):
    if isinstance(o3d_obj, o3d.geometry.TriangleMesh):
        o3d_obj.compute_vertex_normals()

    return o3d_obj


def get_information(o3d_obj):
    def format_vec3(vec, width=10, prec=5):
        return np.array2string(vec, formatter={'float_kind': lambda x: f"{x:>{width}.{prec}f}"})

    bounds = o3d_obj.get_axis_aligned_bounding_box()
    str_info = ""
    if isinstance(o3d_obj, o3d.geometry.TriangleMesh):
        str_info = f"Vertices: {np.asarray(o3d_obj.vertices).shape[0]} \nFaces: {np.asarray(o3d_obj.triangles).shape[0]}\n"
    elif isinstance(o3d_obj, o3d.geometry.PointCloud):
        str_info = f"Points: {np.asarray(o3d_obj.points).shape[0]}\n"

    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    center = bounds.get_center()
    str_bound = f"min      :  {format_vec3(min_bound)} \n" \
                f"max     :  {format_vec3(max_bound)} \n" \
                f"center :  {format_vec3(center)} \n"

    return str_info + str_bound


def show_two_windows(o3d_obj1: o3d.geometry.Geometry,
                     o3d_obj2: o3d.geometry.Geometry,
                     label1_text: str = "",
                     label2_text: str = "",
                     arrows1=None,
                     arrows2=None,
                     show_axis=False):
    apply_default_color(o3d_obj1)
    apply_default_color(o3d_obj2)
    apply_normal(o3d_obj1)
    apply_normal(o3d_obj2)

    w1_size = (1500, 2000)
    w1_location = (50, 50)
    w2_size = w1_size
    w2_location = (w1_location[0] + w1_size[0] + 200, w1_location[1])

    label1_color = gui.Color(1.0, 1.0, 1.0, 1.0)
    label1_background_color = gui.Color(0, 0, 0, 0.45)
    label2_color = label1_color
    label2_background_color = label1_background_color

    field_of_view = 60.0
    label_padding = 8
    sun_dir = np.array([0.577, -0.577, -0.577], dtype=np.float32)

    app = gui.Application.instance
    app.initialize()

    # ---------- MATERIALS ----------
    # Point cloud: unlit + point size
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 5.0

    # Mesh: USE LIT SHADER so normals & lighting work
    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"
    mat_mesh.base_color = [0.6, 0.6, 0.6, 1.0]
    mat_mesh.base_roughness = 0.6
    mat_mesh.base_metallic = 0.0

    # ---------- WINDOW 1 (mesh or obj1) ----------
    w1 = app.create_window("Window 1", *w1_size, *w1_location)
    scene1 = gui.SceneWidget()
    scene1.scene = rendering.Open3DScene(w1.renderer)
    if isinstance(o3d_obj1, o3d.geometry.TriangleMesh):
        scene1.scene.add_geometry("obj1", o3d_obj1, mat_mesh)
        scene1.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, sun_dir)
    else:
        scene1.scene.add_geometry("obj1", o3d_obj1, mat_pcd)
        scene1.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, sun_dir)

    if arrows1:
        scene1.scene.add_geometry("obj1-arrows", arrows1, mat_mesh)


    scene1.scene.show_axes(show_axis)
    # legacy background
    scene1.scene.set_background(np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))
    bounds1 = o3d_obj1.get_axis_aligned_bounding_box()
    scene1.setup_camera(field_of_view, bounds1, bounds1.get_center())

    label1 = gui.Label(get_information(o3d_obj1) + label1_text)
    label1.text_color = label1_color
    label1_block = gui.Vert(0, gui.Margins(label_padding, label_padding, label_padding, label_padding))
    label1_block.background_color = label1_background_color
    label1_block.add_child(label1)

    w1.add_child(scene1)
    w1.add_child(label1_block)
    attach_label_layout(w1, scene1, label1_block)

    # ---------- WINDOW 2 (pcd or obj2) ----------
    w2 = app.create_window("Window 2", *w2_size, *w2_location)
    scene2 = gui.SceneWidget()
    scene2.scene = rendering.Open3DScene(w2.renderer)
    if isinstance(o3d_obj2, o3d.geometry.TriangleMesh):
        scene1.scene.add_geometry("obj2", o3d_obj2, mat_mesh)
        scene1.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, sun_dir)
    else:
        scene2.scene.add_geometry("obj2", o3d_obj2, mat_pcd)
        scene2.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, sun_dir)
    if arrows2:
        scene2.scene.add_geometry("obj2-arrows", arrows2, mat_mesh)

    scene2.scene.show_axes(show_axis)
    scene2.scene.set_background(np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))
    bounds2 = o3d_obj2.get_axis_aligned_bounding_box()
    scene2.setup_camera(field_of_view, bounds2, bounds2.get_center())

    label2 = gui.Label(get_information(o3d_obj2) + label2_text)
    label2.text_color = label2_color
    label2_block = gui.Vert(0, gui.Margins(label_padding, label_padding, label_padding, label_padding))
    label2_block.background_color = label2_background_color
    label2_block.add_child(label2)

    w2.add_child(scene2)
    w2.add_child(label2_block)
    attach_label_layout(w2, scene2, label2_block)

    app.run()


if __name__ == '__main__':
    pass
    # mesh_file = "75_1_1_rotated.ply"
    # o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    #
    # pc_file = "75_1_1.pth"
    # pcd = o3d.geometry.PointCloud()
    # data = torch.load(pc_file, weights_only=False)
    # coord = data["coord"]
    # fps = data["fps_index"]
    # pcd.points = o3d.utility.Vector3dVector(coord[fps])
    #
    # show_two_windows(o3d_mesh, pcd, label1_text="mesh", label2_text="point cloud")