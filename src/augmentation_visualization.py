import copy

import torch
import numpy as np
from augmentation_class import (NormalizeCoord, NormalizeNormal, PositiveShift, CenterShift, RandomShift, RandomRotate, \
    RandomScale, RandomTranslate, RandomJitter, RandomFlip, RandomDropout, ShufflePoint, NormalizeColor, PointClip, \
    ClipGaussianJitter, ChromaticAutoContrast, ChromaticAutoContrastPercent, ChromaticTranslation, ChromaticJitter,
                                RandomColorGrayScale, RandomColorJitter, HueSaturationTranslation, RandomColorAugment,
                                ElasticDistortion, SphereCrop, Sampling, SamplingDynamic, GridSample, AddBackgroundNoise,
                                AddOutlier, AddNoise)

from open3d_utils import show_two_windows, create_pcd_from_np, create_normal_arrows_from_pcd
import open3d as o3d
from matplotlib import colormaps


def CreatePC_test():
    mesh_file = '../data/tmp.ply'
    pc_file = '../data/tmp.pth'

    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    pcd = o3d.geometry.PointCloud()
    data = torch.load(pc_file, weights_only=False)
    coord = data["coord"]
    fps = data["fps_index"]
    pcd.points = o3d.utility.Vector3dVector(coord[fps])

    arrows, indices, length = create_normal_arrows_from_pcd(pcd, normal=data["norm"],
                                                            num_normal=20)
    show_two_windows(o3d_mesh, pcd, label1_text="mesh", label2_text="point cloud", arrows2=arrows)

def NormalizeCoord_test():
    file = '../data/tmp.pth'
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = NormalizeCoord()(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def NormalizeNormal_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = NormalizeNormal()(aug_data_dict)

    normal = data_dict["norm"]
    aug_normal = aug_data_dict["norm"]

    ori_lengths = np.linalg.norm(normal, axis=1)
    aug_lengths = np.linalg.norm(aug_normal, axis=1)
    tol = 1e-3  # you can tighten or loosen this

    print("min ori_length:", ori_lengths.min())
    print("max ori_length:", ori_lengths.max())
    print("mean ori_length:", ori_lengths.mean())
    print("std ori_length:", ori_lengths.std())
    print(f"All normals unit length within {tol}? {np.all(np.abs(ori_lengths - 1.0) < tol)}")
    print("*" * 50)
    print("min aug_length:", aug_lengths.min())
    print("max aug_length:", aug_lengths.max())
    print("mean aug_length:", aug_lengths.mean())
    print("std aug_length:", aug_lengths.std())
    print(f"All normals unit length within {tol}? {np.all(np.abs(aug_lengths - 1.0) < tol)}")

def PositiveShift_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = PositiveShift()(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def CenterShift_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = CenterShift(mean=True, apply_z=False)(aug_data_dict)
    # aug_data_dict = CenterShift(mean=False, apply_z=False)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomShift_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomShift(apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomRotate_test():
    # Normal vector should rotate along with the rotation matrix
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomRotate(angle=(30, 31), axis="y", apply_p=1.0)(aug_data_dict)
    aug_data_dict = RandomRotate(angle=(30, 31), axis="x", apply_p=1.0)(aug_data_dict)
    aug_data_dict = RandomRotate(angle=(30, 31), axis="z", apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)
    arrows1, indices, length = create_normal_arrows_from_pcd(pcd_ori, normal=data_dict["norm"], num_normal=100)
    arrows2, _, _ = create_normal_arrows_from_pcd(pcd_aug, normal=aug_data_dict["norm"], indices=indices, length=length)
    show_two_windows(pcd_ori, pcd_aug, arrows1=arrows1, arrows2=arrows2)

def RandomScale_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomScale(scale=(0.5, 1.5), apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomTranslate_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomTranslate(translate_range=(-0.2, 0.2), apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomJitter_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomJitter(sigma=0.01, clip=0.05, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomFlip_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomFlip(flip_axis=(0, 1, 2), apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def RandomDropout_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomDropout(max_dropout_ratio=1.0, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def ShufflePoint_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    values = np.arange(len(coord)) / len(coord)
    colors = colormaps['Reds'](values)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ShufflePoint(apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])


    print(f"Shape same? {coord.shape} vs {aug_coord.shape}: {coord.shape == aug_coord.shape}")
    print(f"Order changed? {not np.allclose(coord, aug_coord, atol=1e-7)}")

    def sort_rows_lex(x):
        keys = [x[:, i] for i in reversed(range(x.shape[1]))]  # keys: [3, n_pts] -> col0 - col2: z, y, x
        # np.lexsort is NumPy’s function for doing a lexicographic sort — i.e.,
        # “sort by column A; if ties, sort by column B; if still ties, sort by column C; …”.
        # z, y, x -> primary is x, then y, then z
        idx = np.lexsort(keys)
        return x[idx]

    same_set = np.allclose(sort_rows_lex(coord), sort_rows_lex(aug_coord), atol=1e-7)
    print(f"Same value set? {same_set}")

    # show_two_windows(pcd_ori, pcd_aug)

def NormalizeColor_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    values = np.arange(len(coord)) / len(coord)
    color = colormaps['Reds'](values)[:, :3]  # RGBA -> RGB
    data_dict["color"] = color

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = NormalizeColor(low=-1.0, high=1.0, range255=False)(aug_data_dict)
    aug_color = aug_data_dict["color"]

    print("min color:", color.min())
    print("max color:", color.max())
    print("mean color:", color.mean())
    print("std color:", color.std())
    print("*" * 50)
    print("min aug_color:", aug_color.min())
    print("max aug_color:", aug_color.max())
    print("mean aug_color:", aug_color.mean())
    print("std aug_color:", aug_color.std())

def PointClip_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = PointClip(use_sphere=True, radius=1.0, apply_p=1.0)(aug_data_dict)
    # aug_data_dict = PointClip(use_sphere=False, box_range=(0.3, 0.5, 0.2))(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def ClipGaussianJitter_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ClipGaussianJitter(scalar=0.02, quantile=1.96, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def ChromaticAutoContrast_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['summer'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ChromaticAutoContrast(blend_factor=0.8, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def ChromaticAutoContrastPercent_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['summer'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ChromaticAutoContrastPercent(blend_factor=0.8, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def ChromaticTranslation_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['summer'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ChromaticTranslation(ratio=0.2, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def ChromaticJitter_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['summer'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ChromaticJitter(std=0.2, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def RandomColorGrayScale_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomColorGrayScale(apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def RandomColorJitter_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, range255=False, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def HueSaturationTranslation_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = HueSaturationTranslation(hue_max=0.2, saturation_max=0.2, range255=False, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def RandomColorAugment_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)
    coord = data_dict["coord"]

    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = RandomColorAugment(color_augment=5, range255=False, apply_p=1.0)(
        aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)

def ElasticDistortion_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = ElasticDistortion(apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)

def SphereCrop_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = SphereCrop(point_max=20000)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)


def Sampling_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = Sampling(n_pts=4096, method="fps")(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)


def SamplingDynamic_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = SamplingDynamic(key="area", pts_ratio=8192/1.8)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    pcd_ori = create_pcd_from_np(coord)
    pcd_aug = create_pcd_from_np(aug_coord)

    show_two_windows(pcd_ori, pcd_aug)


def GridSample_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    coord = data_dict["coord"]
    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    # np.random.shuffle(z_norm)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    # aug_data_dict = GridSample(grid_size=0.02, grid_number=None, sampling=True, method="random", return_relative=False,
    #                            mode="train")(aug_data_dict)
    # aug_data_dict = GridSample(grid_size=0.02, grid_number=(32, 64, 16), sampling=True, method="random", return_relative=False,
    #                            mode="train")(aug_data_dict)
    aug_data_dict = GridSample(grid_size=0.02, grid_number=None, sampling=True, method="mean", return_relative=False,
                               mode="train")(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    print(data_dict["coord"].shape, aug_data_dict["coord"].shape)
    print(data_dict["norm"].shape, aug_data_dict["norm"].shape)
    print(data_dict["color"].shape, aug_data_dict["color"].shape)

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    show_two_windows(pcd_ori, pcd_aug)


def AddBackgroundNoise_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    coord = data_dict["coord"]
    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['Blues'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = AddBackgroundNoise(max_regions=8, region_max_k=128, region_size=0.2, fixed=True, mode="inside", apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    print(data_dict["coord"].shape, aug_data_dict["coord"].shape)
    print(data_dict["norm"].shape, aug_data_dict["norm"].shape)
    print(data_dict["color"].shape, aug_data_dict["color"].shape)

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    noise_mask = aug_data_dict["noise_index"].astype(bool)
    noise_pcd = create_pcd_from_np(aug_coord[noise_mask])
    arrows, indices, length = create_normal_arrows_from_pcd(noise_pcd, normal=aug_data_dict["norm"][noise_mask], num_normal=20)

    show_two_windows(pcd_ori, pcd_aug, arrows2=arrows)


def AddOutlier_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    coord = data_dict["coord"]
    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['Blues'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = AddOutlier(max_ratio=0.01, radius_min=0.5, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    print(data_dict["coord"].shape, aug_data_dict["coord"].shape)
    print(data_dict["norm"].shape, aug_data_dict["norm"].shape)
    print(data_dict["color"].shape, aug_data_dict["color"].shape)

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    noise_mask = aug_data_dict["noise_index"].astype(bool)
    noise_pcd = create_pcd_from_np(aug_coord[noise_mask])
    arrows, indices, length = create_normal_arrows_from_pcd(noise_pcd, normal=aug_data_dict["norm"][noise_mask],
                                                            num_normal=20)

    show_two_windows(pcd_ori, pcd_aug, arrows2=arrows)

def AddNoise_test():
    file = "../data/tmp.pth"
    data_dict = torch.load(file, weights_only=False)

    coord = data_dict["coord"]
    z = coord[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)
    colors = colormaps['jet'](z_norm)[:, :3]  # RGBA -> RGB
    data_dict["color"] = colors

    aug_data_dict = copy.deepcopy(data_dict)
    aug_data_dict = AddNoise(noise_size_ratio=1./1024, noise_max_k=16, fixed=True, method="custom", boundary="sphere",
                             low=-0.1, high=0.1, ball_r=0.03, apply_p=1.0)(aug_data_dict)

    coord = data_dict["coord"]
    aug_coord = aug_data_dict["coord"]

    print(data_dict["coord"].shape, aug_data_dict["coord"].shape)
    print(data_dict["norm"].shape, aug_data_dict["norm"].shape)
    print(data_dict["color"].shape, aug_data_dict["color"].shape)

    pcd_ori = create_pcd_from_np(coord)
    pcd_ori.colors = o3d.utility.Vector3dVector(data_dict["color"])
    pcd_aug = create_pcd_from_np(aug_coord)
    pcd_aug.colors = o3d.utility.Vector3dVector(aug_data_dict["color"])

    noise_mask = aug_data_dict["noise_index"].astype(bool)
    noise_pcd = create_pcd_from_np(aug_coord[noise_mask])
    arrows, indices, length = create_normal_arrows_from_pcd(noise_pcd, normal=aug_data_dict["norm"][noise_mask],
                                                            num_normal=20)

    show_two_windows(pcd_ori, pcd_aug, arrows2=arrows)


if __name__ == '__main__':
    # CreatePC_test()
    # NormalizeCoord_test()
    # NormalizeNormal_test()
    # PositiveShift_test()
    # CenterShift_test()
    # RandomShift_test()
    # RandomRotate_test()
    # RandomScale_test()
    # RandomTranslate_test()
    # RandomJitter_test()
    # RandomFlip_test()
    # RandomDropout_test()
    # ShufflePoint_test()
    # NormalizeColor_test()
    # PointClip_test()
    # ClipGaussianJitter_test()
    # ChromaticAutoContrast_test()
    # ChromaticAutoContrastPercent_test()
    # ChromaticTranslation_test()
    # ChromaticJitter_test()
    # RandomColorGrayScale_test()
    # RandomColorJitter_test()
    # HueSaturationTranslation_test()
    # RandomColorAugment_test()
    # ElasticDistortion_test()
    # SphereCrop_test()
    # Sampling_test()
    # SamplingDynamic_test()
    # GridSample_test()
    # AddBackgroundNoise_test()
    # AddOutlier_test()
    AddNoise_test()