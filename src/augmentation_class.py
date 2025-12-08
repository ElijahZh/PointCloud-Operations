import numpy as np
import random
from Registry import Registry
import sklearn.preprocessing as preprocessing
from collections.abc import Sequence, Mapping
import torch
import scipy
from typing import Any

TRANSFORMS = Registry("transforms")

@TRANSFORMS.register()
class NormalizeCoord:
    """Normalizes the point cloud into a unit sphere, where the center is the mean of the point set.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    """
    def __call__(self, data_dict: dict) -> dict:
        """Normalizes point cloud coordinates into unit sphere.

        Args:
            data_dict (dict): Input dictionary that contains a "coord" key with a NumPy array of shape (N, 3) representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` normalized into a unit sphere.
        """
        if "coord" in data_dict.keys():
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            # m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            m = np.sqrt(np.max(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m

        return data_dict

@TRANSFORMS.register()
class NormalizeNormal:
    """Normalize normal vectors to unit length for a point cloud.

    This transform expects a dictionary containing:
    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    """
    def __call__(self, data_dict: dict) -> dict:
        """Normalizes normal vectors to unit length.
        Args:
            data_dict (dict): Input dictionary that must contain a "norm" key
                        with a NumPy array of shape (N, 3) representing normal vectors.

        Returns:
            dict: The same dictionary with "norm" normalized to unit length.
        """
        if "norm" in data_dict.keys():
            # preprocess.normalize handles zero divisor
            data_dict["norm"] = preprocessing.normalize(data_dict["norm"], norm='l2')

        return data_dict


@TRANSFORMS.register()
class PositiveShift:
    """Shift point coordinates so all values are non-negative.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    """
    def __call__(self, data_dict: dict) -> dict:
        """Moves points so that all coordinate values become non-negative.

        Args:
            data_dict (dict): Input dictionary that must contain a "coord" key
                with a NumPy array of shape (N, 3) representing point
                coordinates.

        Returns:
            dict:
                The same dictionary with "coord" shifted so all values are greater than or equal to zero.
        """
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], axis=0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register()
class CenterShift:
    """Translate point coordinates so they are centered around a reference point.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    It computes a shift vector and subtracts it from all coordinates in place.
    There are two ways to define the shift:

    * Mean-based centering (``mean=True``):
      - The shift is the mean (centroid) of all points along each axis.
      - If ``apply_z`` is False, the z-component of the shift is replaced by the minimum z value of the points, so:
        - x and y are centered by their mean.
        - z is shifted so that the lowest point lies at z = 0.

    * Bounding-box centering (``mean=False``):
      - The shift is the center of the axis-aligned bounding box (AABB), i.e., the midpoint between min and max along each axis.
      - If ``apply_z`` is False, the z-component of the shift is set to the minimum z value of the points, so the bottom of the bounding box is at z = 0.

    Args:
        mean (bool, optional): If True, use the mean of the coordinates as the
            shift (centroid). If False, use the center of the bounding box.
            Defaults to False.
        apply_z (bool, optional): If True, apply the same centering logic to
            the z-axis as x and y. If False, the z shift is always set to the
            minimum z value, so the lowest point (or bottom of the bounding
            box) sits at z = 0.
            Defaults to True.
    """
    def __init__(self, mean: bool = False, apply_z: bool = True):
        self.mean = mean
        self.apply_z = apply_z

    def __call__(self, data_dict: dict) -> dict:
        """Center the point cloud coordinates in place.

            Args:
                data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3) representing point coordinates.

            Returns:
                dict: The same dictionary with `"coord"` translated according to the centering strategy.
        """
        if "coord" in data_dict.keys():
            if self.mean:
                shift = np.mean(data_dict["coord"], axis=0)
                if not self.apply_z:
                    shift[2] = data_dict["coord"].min(axis=0)[2]
            else:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                if self.apply_z:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
                else:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register()
class RandomShift:
    """Randomly translate point coordinates along the x, y, and z axes.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    It samples a random shift for each axis from the corresponding interval in `shift` and adds it to all points in place.

    Args:
        shift (tuple[tuple[float, float], tuple[float, float], tuple[float, float]], optional):
            A tuple of three `(min, max)` pairs controlling the uniform sampling
            range of the shift per axis:

            * `shift[0]` → (x_min, x_max) for the x-axis shift.
            * `shift[1]` → (y_min, y_max) for the y-axis shift.
            * `shift[2]` → (z_min, z_max) for the z-axis shift.

            Each shift value is sampled from a uniform distribution:
            `np.random.uniform(min, max)`.

            With the default configuration:

            * x ~ U(-0.02, 0.02)
            * y ~ U(-0.02, 0.02)
            * z ~ U( 0.02, 0.02)

            Defaults to `((-0.02, 0.02), (-0.02, 0.02), (0.02, 0.02))`.
        apply_p (float, optional): Probability of applying the random shift.
            Defaults to 1.0.
    """
    def __init__(self, shift: tuple[tuple[float, float]] = ((-0.02, 0.02), (-0.02, 0.02), (0.02, 0.02)), apply_p: float = 1.0):
        self.shift = shift
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a random global shift to the point coordinates.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key
                with a NumPy array of shape (N, 3) representing point
                coordinates.

        Returns:
            dict: The same dictionary with `"coord"` translated by a random shift vector, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register()
class RandomRotate:
    """Randomly rotate 3D points (and optionally normals) around a given axis.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally `"norm"`: NumPy array of shape (N, 3) with normals associated with each point.

    The transform samples a rotation angle (in degrees) from `angle`, builds a rotation matrix around the specified axis,
    and applies it to the coordinates (and normals, if present). The rotation is applied around a center point:

    * If `center` is ``None``, the rotation center is taken as the center of the axis-aligned bounding box (AABB) of the coordinates.
    * If `center` is provided, it is used directly as the rotation center.

    Args:
        angle (tuple[float, float] | None, optional):
            A `(min_deg, max_deg)` pair specifying the range of rotation angles in degrees. The actual angle is sampled
            uniformly from this interval and converted to radians internally. If ``None`` (default), it is set to
            `(-180, 180)`. For example, `angle=(-10, 10)` means a random
            rotation between -10° and +10°.
            Defaults to None.
        center (tuple[float, float, float] | np.ndarray | None, optional):
            Rotation center in 3D, given as a 3-element tuple or NumPy array
            `(cx, cy, cz)`. If ``None`` (default), the center of the bounding
            box of `data_dict["coord"]` is used:
            `center = ((x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2)`.
            Defaults to None.
        axis (str, optional):
            Axis (or axes) around which the rotation is applied. One of `"x"`, `"y"`, `"z"`, or `"xyz"`.

            * `"x"`: single rotation around the x-axis.
            * `"y"`: single rotation around the y-axis.
            * `"z"`: single rotation around the z-axis.
            * `"xyz"`: three independent random rotations are sampled
              (one for x, one for y, one for z), and the final rotation
              matrix is computed as `R = R_z @ R_y @ R_x`.

            In all cases, angles are sampled (in degrees) from the same
            `angle` range.
            Defaults to `"y"`.
        apply_p (float, optional): Probability of applying the rotation.
            Defaults to 1.0.
        """
    def __init__(self,
                 angle: tuple = None,
                 center: tuple | np.ndarray = None,
                 axis: str = 'y',
                 apply_p: float = 1.0,
                 ) -> None:

        self.angle = (-180, 180) if angle is None else angle
        self.center = np.array(center) if center is not None else center
        self.axis = axis
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a random rotation to coordinates (and normals, if present).

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates. Optionally may contain `"norm"` with a NumPy array of shape (N, 3)
                representing normal vectors.

        Returns:
            dict: The same dictionary with `"coord"` rotated around the chosen center, and `"norm"` rotated if present.
        """
        if random.random() > self.apply_p:
            return data_dict

        # angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        angle = np.random.uniform(self.angle[0], self.angle[1])
        angle = np.deg2rad(angle)
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        elif self.axis == "xyz":
            angle = np.random.uniform(self.angle[0], self.angle[1])
            angle = np.deg2rad(angle)
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            rot_x = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])

            angle = np.random.uniform(self.angle[0], self.angle[1])
            angle = np.deg2rad(angle)
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            rot_y = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])

            angle = np.random.uniform(self.angle[0], self.angle[1])
            angle = np.deg2rad(angle)
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            rot_z = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            rot_t = rot_z @ rot_y @ rot_x
        else:
            raise NotImplementedError

        if "coord" in data_dict.keys():
            if self.center is None:
                # rotate by the center point
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center

        if "norm" in data_dict.keys():
            data_dict["norm"] = np.dot(data_dict["norm"], np.transpose(rot_t))

        return data_dict


@TRANSFORMS.register()
class RandomScale:
    """Randomly scale 3D coordinates uniformly or per-axis.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    It samples a scale factor (or factors) from `scale` and multiplies the coordinates in place.

    Args:
        scale (list[float, float] | tuple[float, float], optional):
            A `(min_scale, max_scale)` pair used as the uniform sampling range for the scale factor(s). Values are drawn from
            `np.random.uniform(min_scale, max_scale, size=...)`.

            Examples:
                * `scale=(0.95, 1.05)` → small random resize around 1.0.
                * `scale=(0.5, 1.5)` → more aggressive zoom in/out.

            Defaults to `(0.95, 1.05)`.
        anisotropic (bool, optional): Controls whether scaling is uniform or
            per-axis.

            * `False`: Sample a single scalar `s` and apply `coord *= s`.
            * `True`: Sample a 3D vector `[sx, sy, sz]` and apply
              `coord *= [sx, sy, sz]`.

            Defaults to `False`.
        apply_p (float, optional):
            Probability of applying the random scaling.
            Defaults to 1.0.
        """
    def __init__(self, scale: list | tuple = (0.95, 1.05), anisotropic: bool = False, apply_p: float = 1.0) -> None:
        self.scale = scale
        self.anisotropic = anisotropic  # create separate scale parameters or only one parameter
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a random scaling to the point coordinates.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` scaled by a random factor (uniform or per-axis), if applied.
        """
        if random.random() > self.apply_p:
            return data_dict
        if "coord" in data_dict.keys():
            scale = np.random.uniform(self.scale[0], self.scale[1], size=3 if self.anisotropic else 1)
            # print(scale)
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register()
class RandomTranslate:
    """Randomly translate 3D coordinates by the same offset vector along x, y, z.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    It samples a translation vector `[tx, ty, tz]` from the given range and adds it to all coordinates in place.

    Args:
        translate_range (tuple[float, float], optional):
            A `(min_translate, max_translate)` pair specifying the uniform sampling range for each axis.
            The translation vector is drawn as::

            translate = np.random.uniform(min_translate, max_translate, size=3)

            That is:

            * `tx ~ U(min_translate, max_translate)`
            * `ty ~ U(min_translate, max_translate)`
            * `tz ~ U(min_translate, max_translate)`

            Defaults to `(-0.2, 0.2)`.
        apply_p (float, optional):
            Probability of applying the translation.
            Defaults to 1.0.
    """
    def __init__(self, translate_range: tuple = (-0.2, 0.2), apply_p: float = 1.0):
        self.translate_range = translate_range
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a random global translation to the point coordinates.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` translated by a random offset vector, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            translate = np.random.uniform(self.translate_range[0], self.translate_range[1], size=3)
            data_dict["coord"] += translate
        return data_dict


@TRANSFORMS.register()
class RandomJitter:
    """Add small Gaussian noise to 3D coordinates (point-wise jitter).

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    It samples Gaussian noise for each point and each axis, scales it by `sigma`, clips it to `[-clip, clip]`, and
    adds it to the coordinates in place.

    Args:
        sigma (float, optional):
            Standard deviation of the Gaussian noise before clipping. Noise is drawn as:

                jitter_raw ~ N(0, sigma^2)

            per coordinate.
            Defaults to 0.01.
        clip (float, optional):
            Maximum absolute value for the jitter. After sampling, the noise is clipped to the range `[-clip, clip]`:

                jitter = np.clip(jitter_raw, -clip, clip)

            Must be positive.
            Defaults to 0.05.
        apply_p (float, optional):
            Probability of applying the jitter.
            Defaults to 1.0.
    """
    def __init__(self, sigma: float = 0.01, clip: float = 0.05, apply_p: float = 1.0):
        assert (clip > 0)
        self.sigma = sigma
        self.clip = clip
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply point-wise Gaussian jitter to the point coordinates.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` perturbed by clipped Gaussian noise, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            jitter = np.clip(self.sigma * np.random.randn(data_dict["coord"].shape[0], 3), -self.clip, self.clip)
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register()
class RandomFlip:
    """Randomly flip point coordinates (and normals) by sign along selected axes.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally `"norm"`: NumPy array of shape (N, 3) with normals
      associated with each point.

    Given the axes in `flip_axis`, each axis may be flipped by multiplying
    the corresponding coordinate (and normal, if present) by -1.

    Args:
        flip_axis (tuple[int, ...], optional):
            Indices of axes to consider for flipping. Each element must be in `{0, 1, 2}`:

            * `0` → x-axis
            * `1` → y-axis
            * `2` → z-axis

            For each axis in this tuple, a random decision is made (with
            probability `apply_p`) whether to flip that axis.

            Examples:
                * `flip_axis=(0,)` → only possible flip is x-axis.
                * `flip_axis=(1, 2)` → y and z axes may be flipped
                  independently.

            Defaults to `(0, 2)`.
        apply_p (float, optional):
            Probability of flipping **each** axis.
            Defaults to 1.0.
    """
    def __init__(self, flip_axis: tuple = (0, 2), apply_p: float = 1.0) -> None:
        self.flip_axis = flip_axis
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply random sign flips along selected axes to coords (and normals).

        Args:
            data_dict (dict): Input dictionary that should contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates. Optionally may contain a `"norm"` key with a NumPy array of shape (N, 3)
                representing normal vectors.

        Returns:
            dict: The same dictionary with `"coord"` (and `"norm"` if present) potentially flipped by sign along the
                specified axes.
        """
        for axis in self.flip_axis:
            if np.random.rand() < self.apply_p:
                if "coord" in data_dict.keys():
                    data_dict["coord"][:, axis] = -data_dict["coord"][:, axis]
                if "norm" in data_dict.keys():
                    data_dict["norm"][:, axis] = -data_dict["norm"][:, axis]

        return data_dict


@TRANSFORMS.register()
class RandomDropout:
    """Randomly drop a subset of points (and aligned per-point attributes).

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`)
      that have length N along the first dimension.

    Args:
        max_dropout_ratio (float, optional):
            Maximum fraction of points that may be dropped. The actual dropout ratio is drawn from:

                ratio ~ U(0, max_dropout_ratio)

            For example, if `max_dropout_ratio = 0.2`, then up to 20% of
            points can be removed in any application of this transform.
            Defaults to 0.2.
        apply_p (float, optional):
            Probability of applying the dropout.
            Defaults to 1.0.
    """
    def __init__(self, max_dropout_ratio: float = 0.2, apply_p: float = 1.0):
        self.max_dropout_ratio = max_dropout_ratio
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply random point dropout to coords and aligned attributes.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3).
                Any other entry whose value is a NumPy array or `Sequence` of length N and whose key does not contain
                `"origin"` will also be subsampled.

        Returns:
            dict: The same dictionary with a subset of points (and aligned per-point attributes) kept, if dropout is applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            n = len(data_dict["coord"])
            ratio = np.random.uniform(0, self.max_dropout_ratio)
            size = int(n * (1 - ratio))
            assert size > 0
            idx = np.random.choice(n, size, replace=False)
            for key, value in data_dict.items():
                if isinstance(value, (np.ndarray, Sequence)) and len(value) == n and "origin" not in key:
                    data_dict[key] = value[idx]

        return data_dict


@TRANSFORMS.register()
class ShufflePoint:
    """Randomly permute the order of points (and aligned per-point attributes).

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`) that have length N along the first dimension.

    All per-point arrays of matching length are shuffled with the same
    permutation, preserving correspondence between them.

    Args:
        apply_p (float, optional):
            Probability of applying the shuffling.
            Defaults to 1.0.
    """
    def __init__(self, apply_p: float = 1.0):
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Shuffle the order of points and aligned per-point attributes.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3).
                Any other entry whose value is a NumPy array or `Sequence` of length N and whose key does not contain
                `"origin"` will be permuted with the same shuffle indices.

        Returns:
            dict: The same dictionary with `"coord"` and aligned per-point attributes shuffled in order, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            shuffle_index = np.arange(data_dict["coord"].shape[0])
            np.random.shuffle(shuffle_index)
            n_pts = len(shuffle_index)
            # print(data_dict["noise_index"])
            for key, val in data_dict.items():
                if isinstance(val, (np.ndarray, Sequence)) and len(val) == n_pts and "origin" not in key:
                    data_dict[key] = val[shuffle_index]
            # print(data_dict["noise_index"])
        return data_dict

@TRANSFORMS.register()
class PointClip:
    """Randomly clip a local region around a randomly chosen point.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`)
      that have length N along the first dimension.

    A random point index is selected and its coordinate is used as the center:

        center = coord[center_idx]

    Then it builds either:

    * a spherical region of radius `radius` around `center` if `use_sphere=True`, or
    * an axis-aligned box centered at `center` with half-extent `box_range`
      if `use_sphere=False`.

    Only points inside this region are kept; all others are dropped. All aligned per-point attributes are filtered with
    the same mask.

    Args:
        use_sphere (bool, optional):
            If True, use a spherical region. For each point `p`, compute squared distance:

                dist2 = ||p - center||^2

            and keep points with `dist2 <= radius^2`. If False, use an axis-aligned box instead.
            Defaults to True.
        radius (float, optional): Radius of the sphere used when `use_sphere=True`. The clipped region is:

                { p : ||p - center|| <= radius }

            Defaults to 1.0.
        box_range (tuple[float, float, float], optional): Half-extent of the axis-aligned box along each axis,
            used when `use_sphere=False`. Interpreted as `(rx, ry, rz)`. The box is defined as:

            * `x_min, y_min, z_min = center - box_range`
            * `x_max, y_max, z_max = center + box_range`

            A point `p = (x, y, z)` is kept if:

                x_min <= x <= x_max
                y_min <= y <= y_max
                z_min <= z <= z_max

            Defaults to `(0.0, 0.0, 0.0)`.
        apply_p (float, optional):
            Probability of applying the clipping.
            Defaults to 1.0.
    """
    def __init__(self, use_sphere: bool = True, radius: float = 1.0, box_range: tuple = (0.0, 0.0, 0.0),
                 apply_p: float = 1.0):
        self.use_sphere = use_sphere
        self.radius = radius
        self.box_range = np.array(box_range)
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a local region crop (sphere or box) around a random center.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3).
                Any other entry whose value is a NumPy array or `Sequence` of length N and whose key does not contain
                `"origin"` will also be masked.

        Returns:
            dict: The same dictionary with `"coord"` and aligned per-point attributes cropped to a local region, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            coord = data_dict["coord"]
            n = len(coord)
            center_idx = np.random.randint(low=0, high=n)
            center = coord[center_idx]
            if self.use_sphere:
                diff = coord - center[np.newaxis, :]
                dist2 = np.sum(diff * diff, axis=1)
                mask = dist2 <= self.radius ** 2
            else:
                x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
                # x_min, x_max, y_min, y_max, z_min, z_max = self.box_range + np.repeat(center, 2)
                x_min, y_min, z_min = -self.box_range + center
                x_max, y_max, z_max = self.box_range + center
                # print(x_min, x_max, y_min, y_max, z_min, z_max)
                mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)

            for key, value in data_dict.items():
                if isinstance(value, (np.ndarray, Sequence)) and len(value) == n and "origin" not in key:
                    data_dict[key] = value[mask]
        return data_dict


@TRANSFORMS.register()
class ClipGaussianJitter:
    """Add clipped multivariate Gaussian noise to 3D coordinates.

    This transform expects a dictionary containing:
    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    Unlike a simple per-axis jitter (RandomJitter) with independent 1D Gaussians, this transform uses a
    **multivariate normal distribution**, allowing you to encode correlations between axes via the covariance matrix.
    It samples 3D Gaussian noise from a multivariate normal, normalizes and clips it using a `quantile` parameter, scales
    it by `scalar`, and adds it to the coordinates in place.

    In the default setting:

    * `mean = [0.0, 0.0, 0.0]`
    * `cov = I_3` (3×3 identity matrix → isotropic Gaussian)

    A raw sample is drawn as:

        jitter_raw ~ N(mean, cov)

    Then it is transformed as:

        jitter = scalar * clip(jitter_raw / quantile, -1, 1)

    Intuition:

    * For a standard normal, most values lie within ±`quantile`
      (e.g., 1.96 ≈ 97.5% quantile).
    * Dividing by `quantile` and clipping to [-1, 1] effectively bounds
      each component before scaling, so typical magnitudes are on the
      order of `±scalar`.

    Args:
        quantile (float, optional):
            Normalization factor used before clipping. Noise is divided by `quantile` and then clipped to [-1, 1].
            For `quantile=1.96`, about 95–97.5% of standard normal samples fall in [-1.96, 1.96], so after dividing most
            samples lie in [-1, 1] before clipping.
            Increasing `quantile` makes the effective jitter slightly smaller; decreasing it makes it larger (and more aggressively clipped).
            Defaults to 1.96.
        scalar (float, optional):
            Overall scale factor for the jitter after clipping. Roughly controls the maximum perturbation per coordinate
            (since final values are typically in approximately`[-scalar, scalar]`).
            Defaults to 0.02.
        apply_p (float, optional):
            Probability of applying the jitter.
            Defaults to 1.0.
    """
    def __init__(self, quantile: float = 1.96, scalar: float = 0.02, apply_p: float = 1.0):
        self.mean = [0.0, 0.0, 0.0]
        self.conv = np.identity(3)
        self.quantile = quantile
        self.scalar = scalar
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply clipped multivariate Gaussian jitter to the point coordinates.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` perturbed by clipped multivariate Gaussian noise, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys():
            # [data_dict["coord"].shape[0], len(self.mean)]
            jitter = np.random.multivariate_normal(self.mean, self.conv, data_dict["coord"].shape[0])
            jitter = self.scalar * np.clip(jitter / self.quantile, -1, 1)
            data_dict["coord"] += jitter

        return data_dict


@TRANSFORMS.register()
class NormalizeColor:
    """Normalize point-wise color features into a target value range ``[low, high]``.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    If `range255` is True, input colors are assumed to be in `[0, 255]` and are
    first scaled to `[0, 1]` by dividing by 255. Otherwise, they are assumed
    to already be in `[0, 1]`.

    The normalized `[0, 1]` values are then mapped linearly to `[low, high]`
    such that:

    * 0 → low
    * 1 → high

    Args:
        low (float, optional):
            Lower bound of the target color range.
            Defaults to -1.0.
        high (float, optional):
            Upper bound of the target color range. Must be greater than `low`.
            Defaults to 1.0.
        range255 (bool, optional):
            Whether the input color values are in `[0, 255]`. If True, values are divided by 255.0 before mapping.
            If False, values are used as-is (assumed to be in `[0, 1]`).
            Defaults to False.
    """
    def __init__(self, low: float = -1., high: float = 1., range255=False):
        assert high > low
        self.low = low
        self.high = high
        self.range255 = range255

    def __call__(self, data_dict: dict) -> dict:
        """Normalize the `"color"` entry in `data_dict` into `[low, high]`.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` normalized into `[low, high]`, if present.
        """
        if "color" in data_dict.keys():
            # data_dict["color"] = data_dict["color"] / 127.5 - 1
            normalized_color = data_dict["color"] / 255. if self.range255 else data_dict["color"]  # [0, 1]
            tmp = (self.high - self.low)
            data_dict["color"] = (normalized_color - 1.0) * tmp + self.high
        return data_dict


@TRANSFORMS.register()
class ChromaticAutoContrast:
    """Apply chromatic auto-contrast to point colors with optional blending.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    The transform computes per-channel minimum and maximum over the whole point set,
    linearly stretches each channel to the target range, and then blends this auto-contrasted
    result with the original colors using a `blend_factor`.

    Args:
        blend_factor (float | None, optional):
            Weight used to blend the original colors with the auto-contrasted colors:

                out = (1 - blend_factor) * original + blend_factor * contrasted

            If a float in [0, 1], the same value is used for every call. If ``None``, a new random
            value in [0, 1] is sampled on each call.
            Defaults to ``None``.
        range255 (bool, optional):
            Whether the input color values are in `[0, 255]`. If True, the auto-contrast maps into `[0, 255]`. If
            False, it maps into `[0, 1]`.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the auto-contrast.
            Defaults to 1.0.
    """
    def __init__(self, blend_factor: float = None, range255=False, apply_p: float = 1.0):
        self.blend_factor = blend_factor
        self.range255 = range255
        self.apply_p = apply_p


    def __call__(self, data_dict: dict) -> dict:
        """Apply chromatic auto-contrast to the first 3 color channels.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """

        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            dtype = data_dict["color"].dtype
            if self.range255:
                target_range = 255.0
            else:
                target_range = 1.0

            # choose blend for this call
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor

            # Apply autocontrast calculation
            lo = np.min(data_dict["color"], axis=0, keepdims=True)
            hi = np.max(data_dict["color"], axis=0, keepdims=True)

            # Prevent division by zero if hi == lo
            scale = target_range / (hi - lo + 1e-6)  # Add small epsilon for stability

            # Apply contrast adjustment to the first 3 color channels
            contrast_feat = (data_dict["color"][:, :3] - lo[:, :3]) * scale[:, :3]

            # Blend the original with the contrasted feature
            # Ensure the blending maintains the data type
            blended_color = (1 - blend_factor) * data_dict["color"][:, :3] + blend_factor * contrast_feat

            # Clip values to ensure they stay within the target range [0, target_range]
            # and convert to the appropriate data type
            data_dict["color"][:, :3] = np.clip(blended_color, 0, target_range).astype(dtype)

        return data_dict


@TRANSFORMS.register()
class ChromaticAutoContrastPercent:
    """Apply percentile-based chromatic auto-contrast with optional blending.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    This is similar to ``ChromaticAutoContrast``, but instead of using the absolute min/max per channel,
    it uses the 1st and 99th percentiles as low/high boundaries. This makes the transform effective even when:

    * The input already spans the full range (e.g., `lo=0.0`, `hi=1.0`), or
    * There are outliers that would otherwise dominate the min/max.


    Args:
        blend_factor (float | None, optional):
            Blending weight between the original and auto-contrasted colors. If a float in [0, 1], the
            same value is used for every call. If ``None``, a new random value in [0, 1) is sampled on each call.
            Defaults to ``None``.
        range255 (bool, optional):
            Whether the input color values are in `[0, 255]`. If True, the auto-contrast maps into `[0, 255]`.
            If False, it maps into `[0, 1]`.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the auto-contrast.
            Defaults to 1.0.
    """
    def __init__(self, blend_factor: float = None, range255=False, apply_p: float = 1.0):
        self.blend_factor = blend_factor
        self.range255 = range255
        self.apply_p = apply_p


    def __call__(self, data_dict: dict) -> dict:
        """Apply percentile-based chromatic auto-contrast to the point color.
        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            dtype = data_dict["color"].dtype
            if self.range255:
                target_range = 255.0
            else:
                target_range = 1.0
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor

            # Apply autocontrast calculation, calculate 1% and 99% of the value among each channels
            low_p, high_p = 1, 99
            lo = np.percentile(data_dict["color"][:, :3], low_p, axis=0, keepdims=True)
            hi = np.percentile(data_dict["color"][:, :3], high_p, axis=0, keepdims=True)

            # Prevent division by zero if hi == lo
            scale = target_range / (hi - lo + 1e-6)  # Add small epsilon for stability

            # Apply contrast adjustment to the first 3 color channels
            contrast_feat = (data_dict["color"][:, :3] - lo[:, :3]) * scale[:, :3]

            # Blend the original with the contrasted feature
            # Ensure the blending maintains the data type
            blended_color = (1 - blend_factor) * data_dict["color"][:, :3] + blend_factor * contrast_feat

            # Clip values to ensure they stay within the target range [0, target_range]
            # and convert to the appropriate data type
            data_dict["color"][:, :3] = np.clip(blended_color, 0, target_range).astype(dtype)

        return data_dict

@TRANSFORMS.register()
class ChromaticTranslation:
    """Apply random global color translation to the point color.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    For each call (with some probability), it samples a random translation
    vector `tr` in a bounded range and adds it to all color values:

        tr ∈ [-target_range * ratio, target_range * ratio]^3

    where `target_range` is 255.0 if `range255=True`, else 1.0. The result is
    then clipped back to `[0, target_range]` and cast to the original dtype.

    Args:
        ratio (float, optional):
            Maximum relative translation magnitude as a fraction of the full range. For `range255=True`, each channel
            offset lies in:

                [-255 * ratio, 255 * ratio]

            For `range255=False`, each channel offset lies in:

                [-1.0 * ratio, 1.0 * ratio]

            Defaults to 0.05.
        range255 (bool, optional):
            Whether the input color values are in `[0, 255]`. If True, translations and clipping are done in that
            range. If False, they are done in `[0, 1]`.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the chromatic translation.
            Defaults to 1.0.
    """
    def __init__(self, ratio: float = 0.05, range255=False, apply_p: float = 1.0):
        self.ratio = ratio
        self.range255 = range255
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply a random color translation to the first 3 channels.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            dtype = data_dict["color"].dtype
            if self.range255:
                target_range = 255.0
            else:
                target_range = 1.0

            tr = (np.random.rand(1, 3) - 0.5) * target_range * 2 * self.ratio  # [-255 * ratio, 255 * ratio]
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, target_range).astype(dtype)

        return data_dict

@TRANSFORMS.register()
class ChromaticJitter:
    """Add Gaussian noise (jitter) to point colors.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    It samples per-point, per-channel Gaussian noise and adds it to the channels:

        noise ~ N(0, (std * target_range)^2)

    where `target_range` is 255.0 if `range255=True`, else 1.0. The result is then clipped back to `[0, target_range]`
    and cast to the original dtype.

    Args:
        std (float, optional):
            Standard deviation of the jitter, expressed as a fraction of the target range. The actual noise standard
            deviation per channel is:

                sigma_noise = std * target_range

            For example, with `std=0.005` and `range255=True`, the noise standard deviation is `0.005 * 255 ≈ 1.275`.
            Defaults to 0.005.
        range255 (bool, optional):
            Whether the input color values are in `[0, 255]`. If True, noise and clipping are done in that range.
            If False, they are done in `[0, 1]`.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the chromatic jitter.
            Defaults to 1.0.
    """
    def __init__(self, std: float = 0.005, range255=False, apply_p: float = 1.0):
        self.std = std
        self.range255 = range255
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply Gaussian color jitter to the first 3 channels.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` jittered in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            dtype = data_dict["color"].dtype
            if self.range255:
                target_range = 255.0
            else:
                target_range = 1.0

            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise = noise * self.std * target_range
            data_dict["color"][:, :3] = np.clip(noise + data_dict["color"][:, :3], 0, target_range).astype(dtype)

        return data_dict

@TRANSFORMS.register()
class RandomColorGrayScale:
    """Randomly convert point colors to grayscale.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    It converts the `"color"` from RGB to grayscale using an NTSC-style luminance formula, and returns
    a 3-channel grayscale image (gray copied into R, G, B).

    Args:
        apply_p (float, optional):
            Probability of converting colors to grayscale.
            Defaults to 1.0.
    """
    def __init__(self, apply_p: float = 1.0):
        self.apply_p = apply_p

    @staticmethod
    def rgb_to_grayscale(color: np.ndarray, num_output_channels=1) -> np.ndarray:
        """Convert RGB colors to grayscale using an NTSC-style formula.

        Uses the luminance computation:

            gray = 0.2999 * R + 0.587 * G + 0.114 * B

        Args:
            color (np.ndarray):
                Input color array with shape (..., C), where C ≥ 3 and the last dimension contains RGB channels
                in positions 0, 1, 2.
            num_output_channels (int, optional):
                Number of channels in the output. Must be 1 or 3.

                * 1 → returns a single-channel grayscale array.
                * 3 → returns a 3-channel array with the grayscale value broadcast to R, G, B.

                Defaults to 1.

        Returns:
            gray (np.ndarray):
                Grayscale array with the same dtype as `color` and either 1 or 3 channels in the last dimension.
        """
        assert color.shape[-1] >= 3
        assert num_output_channels in (1, 3)
        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        # NTSC formula
        gray = (0.2999 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)
        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict: dict) -> dict:
        """Randomly convert `"color"` to grayscale in-place.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` converted to 3-channel grayscale, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register()
class RandomColorJitter:
    """Random color jitter for 3D point cloud colors (similar to torchvision).

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    At each call (with probability ``apply_p``), it:

    1. Converts `"color"` to float in the range [0, 1].
       - If ``range255=True``, it divides by 255.
       - Otherwise it assumes values are already in [0, 1] (or compatible).
    2. Randomly samples brightness, contrast, saturation, and hue factors within the ranges specified at initialization.
    3. Applies a random ordering of these adjustments (brightness, contrast, saturation, hue) to the colors.
    4. Clips the result to [0, 1].
    5. Converts back to the original dtype (and multiplies by 255 if ``range255=True``).

    The argument conventions follow torchvision's ``ColorJitter``:

    Args:
        brightness (float | tuple[float, float], optional):
            How much to jitter brightness.

                * If a single non-negative float ``b`` is given, the brightness
                  factor is chosen uniformly from ``[max(0, 1 - b), 1 + b]``.
                * If a tuple ``(b_min, b_max)`` is given, the brightness factor is
                  chosen uniformly from ``[b_min, b_max]``.
                * If set to 0 or (1.0, 1.0), no brightness change is applied.

            Defaults to 0.
        contrast (float | tuple[float, float], optional):
            How much to jitter contrast. Same semantics as ``brightness`` (centered at 1.0).
            If set to 0 or (1.0, 1.0), no contrast change is applied.
            Defaults to 0.
        saturation (float | tuple[float, float], optional):
            How much to jitter saturation. Same semantics as ``brightness`` (centered at 1.0).
            If set to 0 or (1.0, 1.0), no saturation change is applied.
            Defaults to 0.
        hue (float | tuple[float, float], optional):
            How much to jitter hue.

            * If a single float ``h`` is given, the hue factor is chosen
              uniformly from ``[-h, h]``.
            * If a tuple ``(h_min, h_max)`` is given, it is chosen uniformly
              from ``[h_min, h_max]``.

            Values must be in ``[-0.5, 0.5]``. If set to 0 or (0.0, 0.0), no hue change is applied.
            Defaults to 0.
        range255 (bool, optional):
            Whether the input color values are in ``[0, 255]``. If True, values are converted to float in [0, 1]
            before jittering and converted back to [0, 255] afterwards. If False, values are treated as floats in [0, 1].
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the color jitter.
            Defaults to 1.0.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, range255=False, apply_p=1.0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.range255 = range255
        self.apply_p = apply_p

    @staticmethod
    def _check_input(value: int | float | tuple | list, name: str, center: float = 1, bound: tuple = (0, float("inf")),
                     clip_first_on_zero: bool = True):
        """Normalize jitter argument into a (min, max) interval or None.

        This helper follows torchvision's ``ColorJitter._check_input`` logic.

        Args:
            value (int | float | tuple | list):
                Jitter specification.
                    * Single number: interpreted as symmetric range around ``center``, i.e.
                    ``[center - value, center + value]``.
                    * Tuple/list of length 2: interpreted as explicit ``(min, max)``.
            name (str):
                Name of the parameter (for error messages).
            center (float, optional):
                Center value. Usually 1 for brightness/contrast/saturation, 0 for hue.
                Defaults to 1.
            bound (tuple[float, float], optional):
                Allowed bounds for the resulting interval.
                Defaults to ``(0, inf)``.
            clip_first_on_zero (bool, optional):
                If True and value is scalar, the lower bound is clipped at 0.0 (for cases like brightness or contrast).
                For hue, this is False.
                Defaults to True.

        Returns:
            value (list[float] | None):
                A 2-element list ``[min, max]`` if the range is non-trivial, or ``None`` if no change should be applied
                for this parameter.
        """
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        """Blend two color tensors with a given ratio.

        Computes:

            out = ratio * color1 + (1 - ratio) * color2

        and clips the result to [0, 1].

        Args:
            color1 (np.ndarray):
                First color array in [0, 1].
            color2 (np.ndarray):
                Second color array, broadcastable to color1.
            ratio (float):
                Blend ratio. 1.0 means all ``color1``, 0.0 means all ``color2``.

        Returns:
            np.ndarray:
                Blended color array with same dtype as ``color1``.
        """
        ratio = float(ratio)
        return (ratio * color1 + (1.0 - ratio) * color2).clip(0, 1.0).astype(color1.dtype)

    @staticmethod
    def rgb2hsv(rgb):
        """Convert RGB colors in [0, 1] to HSV.

        Args:
            rgb (np.ndarray):
            Array of shape (..., 3) with RGB channels in the last dimension.

        Returns:
            np.ndarray:
            Array of shape (..., 3) with HSV channels in the last dimension.
        """
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        """Convert HSV colors in [0, 1] to RGB.

        Args:
            hsv (np.ndarray):
                Array of shape (..., 3) with HSV channels in the last dimension.

        Returns:
            np.ndarray:
                Array of shape (..., 3) with RGB channels in the last dimension.
        """
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        """Adjust brightness of a color tensor.

        Args:
            color (np.ndarray):
                Color array in [0, 1].
            brightness_factor (float):
                Non-negative brightness scaling factor. 0 means all black, 1 means no change, >1 makes it brighter.

        Returns:
            np.ndarray:
                Brightness-adjusted color array in [0, 1].
        """
        if brightness_factor < 0:
            raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")
        # color and zeros are in [0,1] float
        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        """Adjust contrast of a color tensor.

        Args:
            color (np.ndarray):
                Color array in [0, 1].
            contrast_factor (float):
                Non-negative contrast scaling factor. 0 means the colors at mean intensity (from grayscale value),
                1 means no change, >1 increases contrast.

        Returns:
            np.ndarray:
                Contrast-adjusted color array in [0, 1].
        """
        if contrast_factor < 0:
            raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        """Adjust saturation of a color tensor.

        Args:
            color (np.ndarray):
                Color array in [0, 1].
            saturation_factor (float):
                Non-negative saturation scaling factor. 0 means grayscale, 1 means no change, >1 increases saturation.

        Returns:
            np.ndarray:
                Saturation-adjusted color array in [0, 1].
        """
        if saturation_factor < 0:
            raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        """Adjust hue of a color tensor.

        Args:
            color (np.ndarray):
                Color array in [0, 1].
            hue_factor (float):
                Hue shift factor in [-0.5, 0.5]. The hue channel (in HSV) is shifted by this amount modulo 1.

        Returns:
            np.ndarray:
                Hue-adjusted color array in [0, 1].
        """
        # color in (0, 1.0) range
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")
        hsv = self.rgb2hsv(color)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        return self.hsv2rgb(hsv)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Sample random jitter parameters and a random order of operations.

        Args:
            brightness: Parsed brightness range from ``_check_input`` or ``None``.
            contrast: Parsed contrast range from ``_check_input`` or ``None``.
            saturation: Parsed saturation range from ``_check_input`` or ``None``.
            hue: Parsed hue range from ``_check_input`` or ``None``.

        Returns:
            tuple:
                ``(fn_idx, b, c, s, h)`` where:

                * ``fn_idx`` is a permutation of [0, 1, 2, 3] indicating the
                  order in which brightness/contrast/saturation/hue will be
                  applied.
                * ``b, c, s, h`` are the sampled scalar factors (or ``None`` if
                  the corresponding transform is disabled).
        """
        fn_idx = np.arange(4)
        np.random.shuffle(fn_idx)
        b = (None if brightness is None else np.random.uniform(brightness[0], brightness[1]))
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (None if saturation is None else np.random.uniform(saturation[0], saturation[1]))
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict: dict) -> dict:
        """Apply random color jitter to the `"color"` entry in `data_dict`.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            # make sure color range is [0, 1.0]
            dtype = data_dict["color"].dtype
            if self.range255:
                data_dict["color"] = data_dict["color"].astype(np.float32) / 255.
            else:
                data_dict["color"] = data_dict["color"].astype(np.float32)


            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness,
                                                                                                        self.contrast,
                                                                                                        self.saturation,
                                                                                                        self.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    data_dict["color"] = self.adjust_brightness(data_dict["color"], brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    data_dict["color"] = self.adjust_contrast(data_dict["color"], contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    data_dict["color"] = self.adjust_saturation(data_dict["color"], saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)


            data_dict["color"] = np.clip(data_dict["color"], 0.0, 1.0)
            # convert back to original dtype / range
            if self.range255:
                data_dict["color"] = (data_dict["color"] * 255.).round().astype(dtype)
            else:
                data_dict["color"] = data_dict["color"].astype(dtype)

        return data_dict

@TRANSFORMS.register()
class HueSaturationTranslation:
    """Randomly shift hue and scale saturation of point colors.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    For each call (with probability ``apply_p``), it:

    1. Converts `"color"` to float in the range [0, 1].
       - If ``range255=True``, it divides by 255.
       - Otherwise it assumes values are already in [0, 1] (or compatible).
    2. Converts the color from RGB to HSV.
    3. Samples:
       - a hue shift ``h ~ U(-hue_max, hue_max)`` and applies it modulo 1.0,
       - a saturation scale ``s = 1 + U(-saturation_max, saturation_max)`` and
         multiplies the saturation channel by ``s`` (clipped to [0, 1]).
    4. Converts HSV back to RGB, clips to [0, 1], and restores the original dtype (multiplying by 255 if needed).

    Args:
        hue_max (float, optional):
            Maximum absolute hue shift. The actual hue offset is sampled uniformly from ``[-hue_max, hue_max]`` and added
            to the hue channel modulo 1.0.
            Defaults to 0.5.
        saturation_max (float, optional):
            Maximum relative change in saturation. The saturation scale factor is sampled as:

                s = 1 + U(-saturation_max, saturation_max)

            so saturation can be slightly decreased or increased.
            Defaults to 0.2.
        range255 (bool, optional):
            Whether the input color values are in ``[0, 255]``. If True, values are converted to float in [0, 1]
            before modification and converted back to [0, 255] afterwards. If False, values are treated as floats in [0, 1].
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the hue/saturation translation.
            Defaults to 1.0.
    """
    def __init__(self, hue_max=0.5, saturation_max=0.2, range255=False, apply_p=1.0):
        self.hue_max = hue_max
        self.saturation_max = saturation_max
        self.range255 = range255
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply random hue and saturation translation to `"color"`.
        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():

            # make sure color range is [0, 1.0]
            dtype = data_dict["color"].dtype
            if self.range255:
                data_dict["color"] = data_dict["color"].astype(np.float32) / 255.
            else:
                data_dict["color"] = data_dict["color"].astype(np.float32)

            hsv = RandomColorJitter.rgb2hsv(data_dict["color"][:, :3])
            hue_val = np.random.uniform(-self.hue_max, self.hue_max)
            sat_ratio = 1 + np.random.uniform(-self.saturation_max, self.saturation_max)
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)

            data_dict["color"][:, :3] = np.clip(RandomColorJitter.hsv2rgb(hsv), 0, 1.0)
            if self.range255:
                data_dict["color"] = (data_dict["color"] * 255.).round().astype(dtype)
            else:
                data_dict["color"] = data_dict["color"].astype(dtype)

        return data_dict

@TRANSFORMS.register()
class RandomColorAugment:
    """Apply a simple global color scaling to point colors.

    This transform expects a dictionary containing:

    * `"color"`: NumPy array of shape (N, 3) with point color values.

    It multiplies the color values by ``color_augment`` and clips them to a valid range:

    * If ``range255=True`` → values are clipped to ``[0, 255]``.
    * If ``range255=False`` → values are clipped to ``[0, 1]``.

    Args:
        color_augment (float, optional):
            Multiplicative scaling factor applied to all color channels (e.g., 1.1 to slightly brighten, 0.9 to slightly darken).
            Defaults to 1.1.
        range255 (bool, optional):
            Whether the input color values are in ``[0, 255]``. If True, clipping is done in that range. If False,
            clipping is done in ``[0, 1]``.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the color scaling.
            Defaults to 1.0.
    """
    def __init__(self, color_augment: float = 1.1, range255=False, apply_p: float = 1.0):
        self.color_augment = color_augment
        self.range255 = range255
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply global color scaling to the `"color"` entry in `data_dict`.

        Args:
            data_dict (dict): Input dictionary that must contain a `"color"` key with a NumPy array of shape (N, 3)
                representing point color values.

        Returns:
            dict: The same dictionary with `"color"` updated in-place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "color" in data_dict.keys():
            dtype = data_dict["color"].dtype
            if self.range255:
                target_range = 255.0
            else:
                target_range = 1.0

            data_dict["color"] = np.clip(data_dict["color"] * self.color_augment, 0, target_range).astype(dtype)

        return data_dict

@TRANSFORMS.register()
class ElasticDistortion:
    """Apply elastic distortion to 3D point coordinates.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.

    The distortion is implemented by:

    1. Creating a coarse 3D grid of Gaussian noise with resolution determined
       by `granularity`.
    2. Smoothing the noise with separable 3D convolutions.
    3. Trilinearly interpolating the smoothed noise at each input coordinate.
    4. Adding the interpolated noise (scaled by `magnitude`) to the original
       coordinates.

    Multiple `(granularity, magnitude)` pairs can be applied sequentially to
    produce multi-scale elastic deformations.

    Args:
        distortion_params (list[list[float]] | list[tuple[float, float]] | None, optional):
            List of `(granularity, magnitude)` pairs controlling the elastic fields to apply. Each pair is:

            * `granularity` (float):
                Size of the noise grid in the same units as the coordinates (e.g., meters or centimeters).
                Larger values → smoother, more global distortions.
            * `magnitude` (float):
                Amplitude of the noise displacement added to the coordinates.

            If ``None``, a default two-scale configuration is used: ``[[0.2, 0.4], [0.8, 1.6]]``.
            Defaults to ``None``.
        apply_p (float, optional):
            Probability of applying the elastic
            Defaults to 1.0.
    """
    def __init__(self, distortion_params=None, apply_p: float = 1.0):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        self.apply_p = apply_p

    @staticmethod
    def elastic_distortion(coord, granularity, magnitude):
        """
        Apply a single elastic distortion field to coordinates.

        Args:
            coord (np.ndarray):
                Array of shape (N, D) with point coordinates. The first 3 dimensions are treated as spatial coordinates.
            granularity (float):
                Size of the noise grid in the same units as `coord` (e.g., meters or centimeters).Controls the spatial
                smoothness of the distortion.
            magnitude (float):
                Noise multiplier that scales the interpolated noise displacement added to `coord`.

        Returns:
            np.ndarray:
            The same coordinate array, distorted in place and also returned for convenience.
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coord.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coord - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [np.linspace(d_min, d_max, d) for d_min, d_max, d in zip(coords_min - granularity,
                                                                      coords_min + granularity * (noise_dim - 2),
                                                                      noise_dim)
              ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coord += interp(coord) * magnitude
        return coord

    def __call__(self, data_dict: dict) -> dict:
        """Apply elastic distortion(s) to `"coord"` in `data_dict`.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3)
                representing point coordinates.

        Returns:
            dict: The same dictionary with `"coord"` distorted in place, if applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" in data_dict.keys() and self.distortion_params is not None:
            for granularity, magnitude in self.distortion_params:
                data_dict["coord"] = self.elastic_distortion(data_dict["coord"], granularity, magnitude)
        return data_dict

@TRANSFORMS.register()
class SphereCrop:
    """Crop a point cloud to a fixed number of points using a spherical region.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`) that have length N along the first dimension.

    The target number of points can be controlled either by an absolute cap (`point_max`) or a relative rate (`sample_rate`):

    * If `sample_rate` is not ``None``, the effective maximum is `point_max_eff = int(sample_rate * N)`.
    * Otherwise, the fixed `point_max` value is used.

    The center of the crop is chosen according to `mode`:

    * `"random"`: use a randomly selected point as center.
    * `"center"`: use the point at index `N // 2` as center (e.g., middle in
      the current ordering).

    If `N <= point_max_eff`, no cropping is applied.

    Args:
        point_max (int, optional):
            Maximum number of points to keep if`sample_rate` is ``None``.
            Defaults to 80,000.
        sample_rate (float | None, optional):
            If provided, the effective maximum number of points is computed as:

                point_max_eff = int(sample_rate * N)

            where `N` is the current number of points. This allows dataset-dependent cropping. If ``None``, the fixed `point_max`
            is used instead.
            Defaults to ``None``.
        mode (str, optional):
            Strategy to select the crop center. Must be `"random"` or `"center"`.

            * `"random"`: center is a random point from `"coord"`.
            * `"center"`: center is `coord[N // 2]`.

            Defaults to `"random"`.
    """
    def __init__(self, point_max: int = 80000, sample_rate: float = None, mode: str = "random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center"]
        self.mode = mode

    def __call__(self, data_dict: dict) -> dict:
        """Apply spherical cropping to the `"coord"` entry in `data_dict`.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3).
                Any other entry whose value is a NumPy array or `Sequence` of length N and whose key does not contain
                `"origin"` will be permuted with the same shuffle indices.

        Returns:
            dict: The same dictionary with `"coord"` and aligned per-point attributes cropped to at most `point_max_eff`
            points, if applied.
        """
        if "coord" in data_dict.keys():
            n = len(data_dict["coord"])
            if self.sample_rate is not None:
                point_max = int(self.sample_rate * n)
            else:
                point_max = self.point_max

            # mode is "random" or "center"
            if n > point_max:
                if self.mode == "random":
                    center = data_dict["coord"][np.random.randint(n)]
                elif self.mode == "center":
                    center = data_dict["coord"][n // 2]
                else:
                    raise NotImplementedError

                idx = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[: point_max]
                for key, value in data_dict.items():
                    if isinstance(value, (np.ndarray, Sequence)) and len(value) == n and "origin" not in key:
                        data_dict[key] = value[idx]

        return data_dict

@TRANSFORMS.register()
class Sampling:
    """Subsample points (and aligned attributes) using random or FPS-based indices.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`) that have length N along the first dimension.
    * For FPS-based methods (`"random_fps"` and `"fps"`), a key `"fps_index"` is also required, containing a 1D array of
      precomputed farthest-point-sampling indices into `"coord"`.

    If the requested number of points `n_pts` is **less** than the current number of points `N`, it selects a subset of
    indices and applies the same index selection to all aligned per-point fields (except keys containing `"origin"`).
    If `n_pts >= N`, no subsampling is applied.

    The sampling strategy is controlled by `method`:

    * `"random"`:
      - Uniformly sample `n_pts` indices from `[0, N)` without replacement.
    * `"random_fps"`:
      - Sample `n_pts` indices from `fps_index` without replacement, then map the index to get the final subset.
      (it is for better augmentation instead of fixed fps index each time despite not guarantee perfect uniform coverage)
    * `"fps"`:
      - Directly use the first `n_pts` entries from `data_dict["fps_index"]`.

    Args:
        n_pts (int, optional):
            Target number of points after sampling. If `n_pts >= N`, no sampling is applied. Defaults to 1024.
        method (str, optional):
            Sampling strategy. One of:

            * `"random"`: uniform random sampling.
            * `"random_fps"`: random subset of precomputed FPS indices.
            * `"fps"`: first `n_pts` precomputed FPS indices.

            Defaults to `"fps"`.
    """
    def __init__(self, n_pts=1024, method="fps"):
        self.n_pts = n_pts
        self.method = method
        assert method in ["random", "random_fps", "fps"]

    def __call__(self, data_dict: dict) -> dict:
        """Subsample points and aligned fields according to the chosen method.

        Args:
            data_dict (dict): Input dictionary that must contain a `"coord"` key with a NumPy array of shape (N, 3).
                Any other entry whose value is a NumPy array or `Sequence` of length N and whose key does not contain
                `"origin"` will be permuted with the same shuffle indices.

        Returns:
            dict: The same dictionary with `"coord"` and aligned per-point attributes subsampled to at most `n_pts` points, if applied.
        """
        # print(data_dict.keys())
        if "coord" in data_dict.keys():
            N = len(data_dict["coord"])
            if self.n_pts < N :
                if self.method == "random":
                    idx = np.random.choice(N, self.n_pts, replace=False)
                elif self.method == "random_fps":
                    assert self.n_pts <= len(data_dict["fps_index"])
                    idx = np.random.choice(len(data_dict["fps_index"]), self.n_pts, replace=False)
                    idx = data_dict["fps_index"][idx]
                elif self.method == "fps":
                    assert self.n_pts <= len(data_dict["fps_index"])
                    idx = data_dict["fps_index"][:self.n_pts]
                else:
                    raise NotImplementedError(f"method {self.method} is not supported.")

                for key, val in data_dict.items():
                    if isinstance(val, (np.ndarray, Sequence)) and len(val) == N and "origin" not in key:
                        data_dict[key] = val[idx]

        return data_dict

@TRANSFORMS.register()
class SamplingDynamic:
    """Dynamically choose the number of sampled points based on a scalar attribute.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally other per-point arrays (e.g., `"norm"`, `"color"`, `"label"`) that have length N along the first dimension.
    * `"fps_index"`: 1D NumPy array of precomputed farthest-point-sampling indices into `"coord"`.
    * A scalar entry `key` (default `"area"`) used to determine how many points to sample.

    The number of target points is computed as:

        pts = int(data_dict[key] * pts_ratio)

    Then, from `"fps_index"` it selects:

    * The first `pts` indices if `len(fps_index) > pts`, or
    * All of `fps_index` otherwise.

    All per-point arrays of length `N` (except those whose key contains `"origin"`) are then indexed with this subset.

    Args:
        key (str, optional):
            Name of the scalar field in `data_dict` used to determine the dynamic number of points. Common choices might
            include `"area"`, `"volume"`, etc.
            Defaults to `"area"`.
        pts_ratio (float, optional):
            Multiplicative factor that maps the scalar value `data_dict[key]` to a target number of points:

                pts = int(data_dict[key] * pts_ratio)

            For example, if `data_dict["area"] = 1.8` and `pts_ratio` is `8192 / 1.8`, then `pts ≈ 8192`.
            Defaults to `8192 / 1.8`.
    """
    def __init__(self, key="area", pts_ratio=8192/1.8):
        self.pts_ratio = pts_ratio
        self.key = key

    def __call__(self, data_dict: dict) -> dict:
        """Apply dynamic FPS-based subsampling to `"coord"` and aligned fields.

        Args:
            data_dict (dict): Input dictionary containing at least `"coord"`, `self.key`, and `"fps_index"`.

        Returns:
            dict: The same dictionary with `"coord"` and aligned per-point attributes subsampled according to the
            dynamically chosen number of points.
        """
        if "coord" in data_dict.keys():
            N = len(data_dict["coord"])
            assert self.key in data_dict
            assert "fps_index" in data_dict

            pts = int(data_dict[self.key] * self.pts_ratio)
            if len(data_dict["fps_index"]) > pts:
                idx = data_dict["fps_index"][:pts]
            else:
                idx = data_dict["fps_index"]

            for key, val in data_dict.items():
                if isinstance(val, (np.ndarray, Sequence)) and len(val) == N and "origin" not in key:
                    data_dict[key] = val[idx]

        return data_dict

@TRANSFORMS.register()
class GridSample:
    """Grid-based voxelization and optional per-voxel sampling/pooling.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally per-point attributes (e.g. `"norm"`, `"color"`, `"label"`) that have length N along the first dimension.

    It first normalizes coordinates by subtracting the global minimum so that all coordinates are non-negative,
    then assigns each point to a 3D grid cell (voxel) either by:

    * a fixed cell size (`grid_size`), or
    * a fixed number of cells per axis (`grid_number`).

    The per-point integer voxel index is stored in `"grid_coord"`.

    Optionally, it can:

    * Store per-point relative coordinates within each voxel in `"relative_coord"`, normalized to approximately `[-1, 1]` per axis.

    * Perform voxel-level sampling/aggregation:

      - `method="random"`:
        * `mode="train"`: pick one random point per voxel.
        * `mode="test"`: generate up to `max_count` “views”, each view containing one point per voxel
        (cycling through the points in each voxel). Returns a list of dictionaries.

      - `method="mean"`:
        * Average features per voxel and replace `"coord"`, `"norm"`, `"color"` accordingly.
        * For `"label"`:
          - If integer dtype → take majority vote per voxel.
          - Else → average values per voxel.

    Args:
        grid_size (float, optional):
            Fixed grid cell size along each axis when `grid_number` is ``None``. The
            same size is used for x, y, z. Coordinates are voxelized as:

                grid_coord = floor((coord - coord_min) / grid_size)

            Defaults to 0.02.
        grid_number (tuple[int] | None, optional):
            If not ``None``, use a fixed number of cells per axis instead of a fixed size. In that case, the effective
            grid size is computed as:

                grid_size = coord_norm.max(axis=0) / grid_number

            and voxel indices are clipped to `[0, grid_number-1]`.
            Defaults to ``None``.
        sampling (bool, optional):
            Whether to perform per-voxel sampling or pooling. If False, the transform only computes `"grid_coord"`
            (and `"relative_coord"` if `return_relative=True`) and does not change the number of points.
            Defaults to True.
        method (str, optional):
            Sampling/pooling strategy when `sampling=True`. Supported values:

            * `"random"`: per-voxel random selection.
            * `"mean"`: per-voxel averaging of features (and label fusion).

            Defaults to `"random"`.
        return_relative (bool, optional):
            If True, adds `"relative_coord"` to `data_dict`, containing per-point offsets relative to the voxel center,
            normalized to approximately `[-1, 1]` per axis.
            Defaults to False.
        mode (str, optional):
            Behavior mode for `method="random"`.
            * `"train"`: returns a single dictionary, selecting one random point per voxel.
            * `"test"`: returns a list of dictionaries, each containing one point per voxel, cycling through all points within each voxel.

            Defaults to `"train"`.
    """

    def __init__(self, grid_size: float = 0.02, grid_number: tuple[int] = None, sampling: bool = True,
                 method: str = "random", return_relative: bool = False, mode: str = "train"):
        self.grid_number = grid_number
        self.grid_size = grid_size
        self.sampling = sampling
        assert method in ["random", "mean"]
        self.method = method
        self.return_relative = return_relative
        assert mode in ["train", "test"]
        self.mode = mode

    @staticmethod
    def _ravel_3d(arr_grid):
        """Convert 3D integer grid coordinates to a 1D key via row-major flattening.

        Given integer grid coordinates of shape (N, 3), this computes a unique 1D key for each voxel coordinate
        such that points in the same voxel share the same key. This is used for grouping points by voxel
        (e.g. for random voxel-wise sampling).

        Args:
            arr_grid (np.ndarray): Integer grid coordinates of shape (N, 3).

        Returns:
            np.ndarray: 1D array of length N containing raveled voxel keys.
        """
        assert arr_grid.ndim == 2 and arr_grid.shape[-1] == 3
        max_grid = np.max(arr_grid, axis=0)
        keys = arr_grid[:, 0] + arr_grid[:, 1] * (max_grid[0] + 1) + arr_grid[:, 2] * (max_grid[0] + 1) * (
                max_grid[1] + 1)
        return keys

    def __call__(self, data_dict: dict) -> dict | list:
        """Apply grid voxelization and optional voxel-wise sampling/pooling.


        Args:
            data_dict (dict): Input dictionary that must contain `"coord"`.
                Optionally `"norm"`, `"color"`, `"label"`, etc.

        Returns:
            dict | list[dict]:
                * If `sampling=False` or `method="mean"` or `mode="train"`:
                  returns a single modified `data_dict`.
                * If `method="random"` and `mode="test"`: returns a list of dictionaries, each representing a
                  different per-voxel sampling pass.
        """
        if "coord" in data_dict.keys():
            coord = data_dict["coord"]
            coord_min = coord.min(axis=0)
            coord_norm = coord - coord_min
            # kill tiny negative noise due to FP
            coord_norm = np.clip(coord_norm, 0.0, None)  # everything < 0 → 0.0

            if self.grid_number is None:
                # fixed grid cell size
                grid_size = self.grid_size
                grid_coord = np.floor(coord_norm / grid_size).astype(int)
            else:
                # fixed number of cells per axis
                grid_number = np.array(self.grid_number, dtype=np.int32)
                grid_size = coord_norm.max(axis=0) / grid_number
                # avoid division by zero in degenerate dims
                grid_size[grid_size == 0] = 1e-6

                grid_coord = np.floor(coord_norm / grid_size).astype(np.int32)
                # floor() would give integers in [0, grid_number-1] except when coord_norm == max, you get exactly grid_number.
                grid_coord = np.clip(grid_coord, a_min=0, a_max=grid_number - 1)

            data_dict["grid_coord"] = grid_coord

            if self.return_relative:
                grid_center = (grid_coord + 0.5) * grid_size
                # normalized to -1, 1 range
                data_dict["relative_coord"] = (coord_norm - grid_center) * (2.0 / grid_size)

            if not self.sampling:
                return data_dict

            N = len(data_dict["coord"])
            if self.method == "random":
                key = self._ravel_3d(grid_coord)
                idx_sort = np.argsort(key)  # [min_key_index, ... max_key_index]
                key_sort = key[idx_sort]
                unique_keys, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

                if self.mode != "test":
                    # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(),
                    #                                                                          count.size) % count

                    # TRAIN: pick 1 random point per voxel
                    offsets = np.cumsum(np.insert(count, 0, 0))[:-1]
                    # one random offset in [0, count_i) per voxel
                    rand_offsets = np.random.randint(0, count.max(), size=count.size) % count
                    idx_select = offsets + rand_offsets  # indices into sorted list

                    idx_unique = idx_sort[idx_select]
                    for key, value in data_dict.items():
                        if isinstance(value, (np.ndarray, Sequence)) and len(value) == N and "origin" not in key:
                            data_dict[key] = value[idx_unique]
                else:
                    data_part_list = []
                    max_count = int(count.max())
                    offsets = np.cumsum(np.insert(count, 0, 0))[:-1]

                    for i in range(max_count):
                        idx_select = offsets + (i % count)
                        idx_unique = idx_sort[idx_select]

                        data_part = dict(origin_index=idx_unique)
                        for key, value in data_dict.items():
                            if isinstance(value, (np.ndarray, Sequence)) and len(value) == N and "origin" not in key:
                                data_part[key] = value[idx_unique]
                            else:
                                data_part[key] = data_dict[key]
                        data_part_list.append(data_part)
                    return data_part_list


            elif self.method == "mean":
                unique_keys, inverse, count = np.unique(grid_coord, return_inverse=True, return_counts=True, axis=0)
                unique_sizes = len(unique_keys)

                # ----- combine features along feature dimension -----
                feat_parts = []
                feat_slices = {}   # record where each feature lives in combined vector
                start = 0
                for name in ["coord", "norm", "color"]:
                    if name in data_dict:
                        arr = np.array(data_dict[name])
                        dim = arr.shape[1]
                        feat_parts.append(arr)
                        feat_slices[name] = slice(start, start + dim)
                        start += dim
                if feat_parts:
                    combined_feat = np.concat(feat_parts, axis=1)

                    sum_feat = np.zeros((unique_sizes, combined_feat.shape[1]), dtype=combined_feat.dtype)
                    np.add.at(sum_feat, inverse, combined_feat)
                    avg_feat = sum_feat / count[:, np.newaxis]


                    data_dict["grid_coord"] = unique_keys
                    # ----- write back per feature -----
                    for key in feat_slices.keys():
                        data_dict[key] = avg_feat[:, feat_slices[key]]

                if "label" in data_dict and len(data_dict["label"]) == N:
                    labels = np.array(data_dict["label"])
                    if np.issubdtype(labels.dtype, np.integer):
                        # treat as classification
                        num_classes_tmp = int(labels.max()) + 1
                        hist = np.zeros((unique_sizes, num_classes_tmp), dtype=np.int32)
                        np.add.at(hist, (inverse, labels), 1)   # (inverse, labels) is the 2D index for the hist
                        data_dict["label"] = hist.argmax(axis=1)
                    else:
                        # treat as regression
                        if labels.ndim == 1:
                            labels = labels[:, np.newaxis]
                        sum_label = np.zeros((unique_sizes, labels.shape[1]), dtype=np.float32)
                        np.add.at(sum_label, inverse, labels)
                        avg_label = sum_label / count[:, None]
                        if labels.ndim == 1:
                            avg_label = avg_label[:, 0]
                        data_dict["label"] = avg_label

        return data_dict

@TRANSFORMS.register()
class AddOutlier:
    """Add synthetic outlier points (and attributes) around a point cloud.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally `"norm"`: per-point normals of shape (N, 3).
    * Optionally `"color"`: per-point colors of shape (N, C).
    * Optionally `"noise_index"`: 1D array of length N indicating noise labels (existing noise index).

    It augments the point cloud by sampling additional points in a spherical shell around the existing point cloud
    and appends them (and their attributes) to the existing arrays.

    The N number of outliers is:

        N = int(max_ratio * n_pts)

    where ``n_pts`` is the original number of points. If this is 0, the transform
    does nothing.

    Behavior of ``fixed``:

    * If ``fixed=False``:
      - Use **all** `N` generated outliers.
    * If ``fixed=True``:
      - Randomly choose a subset of the outliers with size:

            N_used ∼ Uniform{ N // 2, ..., N }

        and only append this subset.

    Outliers are sampled inside a spherical shell defined from the centroid
    and bounding radius:

    1. Compute the centroid:

           center = mean(coord, axis=0)

    2. Compute the maximum radius from the centroid:

           r_max = max(||coord[i] - center||)

    3. Define a spherical shell with inner radius:

           r_min = radius_min * r_max

       and outer radius:

           r_max_shell = r_max

    4. Sample directions uniformly on the sphere and radii uniformly in **volume** within `[r_min, r_max_shell]`,
       then map back to world coordinates.

    For each outlier point, a random unit normal is generated (if `"norm"` exists), and random colors are generated
    (if `"color"` exists).

    The `"noise_index"` field is updated as follows:

    * If `"noise_index"` does not exist:
      - A new array of length `N + N_used` is created, filled with 0 for original points and 3
      (indication of outlier type noise) for new outliers.
    * If `"noise_index"` exists:
      - It is extended to length `N + N_used`, keeping existing values and setting all new entries to 3.

    Args:
        max_ratio (float, optional):
            Maximum ratio of outliers to original points. The N number of generated outliers is:

                N = int(max_ratio * n_pts)

            where n_pts is the original point count.
            Defaults to 0.2.
        fixed (bool, optional):
            Controls whether to subsample the generated outliers:

            * False → use all generated outliers (`N`).
            * True → randomly select a subset of size between `N // 2` and `N` (inclusive).
            Defaults to False.
        radius_min (float, optional):
            Fraction of the maximum radius used as the inner radius of the sampling shell. The shell is defined as
            `[radius_min * r_max, r_max]`.
            Defaults to 0.5.
        range255 (bool, optional): Whether `"color"` values are in `[0, 255]`. If True, outlier colors are sampled as
            integers in `[0, 255]`. If False, they are sampled as floats in `[0, 1]`.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the adding outliers.
            Defaults to 1.0.
    """
    def __init__(self, max_ratio=0.2, fixed=False, radius_min=0.5, range255=False, apply_p=1.0):
        self.max_ratio = max_ratio
        self.fixed = fixed
        self.radius_min = radius_min
        self.range255 = range255
        self.apply_p = apply_p


    def __call__(self, data_dict: dict) -> dict:
        """Add outlier points and update aligned per-point attributes.

        Args:
            data_dict (dict): Input dictionary containing `"coord"` and optionally `"norm"`, `"color"`, and `"noise_index"`.

        Returns:
            dict: The same dictionary with additional outlier points and updated attributes, if adding outliers is applied.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" not in data_dict.keys():
            return data_dict

        coord = data_dict["coord"]
        if "noise_index" in data_dict:
            coord = coord[np.logical_not(data_dict["noise_index"])]
        n_pts = len(coord)

        # how many outliers to add
        N = int(self.max_ratio * n_pts)
        if N <= 0:
            return data_dict

        # --- define a bounding sphere from current coords ---
        center = coord.mean(axis=0, keepdims=True)  # (1,3)
        rel = coord - center  # (N,3)
        r_max = np.linalg.norm(rel, axis=1).max() + 1e-6  # avoid zero

        r_min = self.radius_min * r_max
        r_max_shell = r_max  # you can change to > r_max if you want truly outside
        # --- sample directions uniformly on sphere ---
        noise_dir = np.random.normal(size=(N, 3))  # Gaussian
        noise_dir /= np.linalg.norm(noise_dir, axis=-1, keepdims=True)  # normalize
        # --- sample radii uniformly in the volume shell [r_min, r_max_shell] ---
        u = np.random.uniform(low=0.0, high=1.0, size=N)
        # uniform in [r_min^3, r_max^3], then cube root → uniform in volume
        r = (r_min ** 3 + u * (r_max_shell ** 3 - r_min ** 3)) ** (1.0 / 3.0)
        noise = noise_dir * r[:, None]  # (N,3) in bounding shell
        noise_coord = center + noise  # map back to world coords

        # add normal
        rand_dir = np.random.normal(size=noise_coord.shape)  # gaussian noise
        # normalize to unit vectors
        norm_len = np.linalg.norm(rand_dir, axis=-1, keepdims=True) + 1e-8
        noise_normal = (rand_dir / norm_len).astype(np.float32)
        # noise_normal = noise_dir              # or use direction as "normal"


        if not self.fixed and N > 1:
            # choose subset size between N // 2 and N (inclusive)
            k_min = max(1, N // 2)
            k = np.random.randint(k_min, N + 1)
            idx_sel = np.random.choice(N, size=k, replace=False)
            noise_coord = noise_coord[idx_sel]
            noise_normal = noise_normal[idx_sel]
            N = k

        # ----- add coords -----
        data_dict["coord"] = np.concatenate([data_dict["coord"], noise_coord], axis=0)
        # ----- add normals (if present) -----
        if "norm" in data_dict:
            data_dict["norm"] = np.concatenate([data_dict["norm"], noise_normal], axis=0)
        # ----- add color for outliers (if present) -----
        if "color" in data_dict:
            orig_color = data_dict["color"]
            C = orig_color.shape[1]
            dtype = orig_color.dtype
            if self.range255:
                noise_color = np.random.randint(0, 256, size=(N, C)).astype(dtype)
            else:
                noise_color = np.random.rand(N, C).astype(dtype)
            data_dict["color"] = np.concatenate([data_dict["color"], noise_color], axis=0)

        # ----- update noise_index -----
        if "noise_index" not in data_dict:
            noise_index = np.zeros(len(data_dict["coord"]), dtype=int)
            noise_index[-N:] = 3  # mark new outliers as 3
            data_dict["noise_index"] = noise_index
        else:
            old_len = len(data_dict["noise_index"])
            new_len = len(data_dict["coord"])
            tmp = np.ones(new_len, dtype=int) * 3
            tmp[:old_len] = data_dict["noise_index"]
            data_dict["noise_index"] = tmp

        return data_dict

@TRANSFORMS.register()
class AddBackgroundNoise:
    """Add structured background noise patches around a point cloud.

    This transform expects a dictionary containing:

    * `"coord"`: NumPy array of shape (N, 3) with point coordinates.
    * Optionally `"norm"`: per-point normals of shape (N, 3).
    * Optionally `"color"`: per-point colors of shape (N, C).
    * Optionally `"noise_index"`: 1D array of length N indicating noise labels (existing noise index).

    It generates several small 3D regions (up to ``max_regions``), each containing up to ``region_max_k`` random points.
    These regions are anchored to the bounding box of the (non-noise) point cloud and then added as background clutter:

        1. Compute the axis-aligned bounding box (AABB) from points with ``noise_index == 0`` (or all points if ``noise_index`` is absent).
        2. Sample up to ``max_regions`` random centers inside the box (with behavior controlled by ``mode``).
        3. For each region, sample points uniformly in a cube of side length ``region_size`` around the center.
        4. Snap one coordinate of each region to one face of the bounding box, so that noise patches lie on or around the box surface.
        5. Generate random unit normals for these noise points.
        6. Optionally subsample a random subset of regions/points when ``fixed=False``.
        7. Concatenate noise coordinates, normals, and colors (if present) to the existing arrays, and update ``noise_index`` to mark them asbackground noise (label 2).

    Args:
        max_regions (int, optional):
            Maximum number of noise regions to generate. Each region is a local cube of sampled noise points.
            Defaults to 8.
        region_max_k (int, optional):
            Maximum number of points per region before optional subsampling. Total initial noise points are
            roughly ``max_regions * region_max_k`` before any reduction.
            Defaults to 128.
        region_size (float, optional):
            Side length of each cubic noise region. Points are sampled uniformly in a cube of side ``region_size``
            centered at each region center.
            Defaults to 0.5.
        fixed (bool, optional):
            Controls randomness and subsampling of the generated noise:
            * If ``False``:
              - Randomly choose a number of regions between 1 and ``max_regions``.
              - Flatten all noise points from those regions.
              - Randomly select a subset of points, with size between roughly 1/8 of all region points and the full set.
            * If ``True``:
              - Use all ``max_regions * region_max_k`` points (no region or point subsampling).

            Defaults to False.
        range255 (bool, optional):
            Whether color values are in the range ``[0, 255]``. If True, noise colors are sampled as integers in
            ``[0, 255]``. If False, noise colors are sampled as floats in ``[0, 1]``.
            Defaults to False.
        mode ({"outside", "inside"}, optional): Controls how the region
            centers are chosen relative to the bounding box:

            * ``"inside"``:
              - Region centers are chosen so that the entire noise cube (of side ``region_size``) lies strictly inside the bounding
                box, as much as possible. This ensures all noise points stay on/within the box volume.
            * ``"outside"``:
              - Region centers are sampled anywhere inside the bounding box, and one coordinate of each region is snapped to a box face.
                Some noise points may lie slightly outside the box, mimicking more realistic background clutter.

            Defaults to ``"outside"``.
        apply_p (float, optional):
            Probability of applying the background
            Defaults to 1.0.
    """
    def __init__(self, max_regions: int =8, region_max_k: int =128, region_size: float=0.5, fixed:bool=False, range255: bool=False, mode: str="outside", apply_p: float=1.0):
        self.max_regions = max_regions
        self.region_max_k = region_max_k
        self.fixed = fixed
        self.region_size = region_size
        self.range255 = range255
        self.apply_p = apply_p
        assert mode in ["outside", "inside"]
        # "inside" is to make sure all noise are on the bounding box, "outside" only guarantee the center is on the
        # bounding box, some noise might be outside the box, "inside" is for nice for segmentation / denoising. But
        # "outside" is more like real background clutter.
        self.mode = mode

    def __call__(self, data_dict: dict) -> dict:
        """Add background noise regions and update aligned per-point attributes.

        Args:
            data_dict (dict): Input dictionary containing at least `"coord"`. May also contain `"norm"`, `"color"`, and `"noise_index"`.
                Any existing `"noise_index" > 0` points are excluded when computing the bounding box used to place new regions.

        Returns:
            dict: The same dictionary with added background noise points (coords, normals, colors if present) and an updated
            `"noise_index"` marking these new points with label 2.
        """
        if random.random() > self.apply_p:
            return data_dict

        if "coord" not in data_dict.keys():
            return data_dict

        coord = data_dict["coord"]
        if "noise_index" in data_dict:
            coord = coord[np.logical_not(data_dict["noise_index"])]
        max_value, min_value = np.max(coord, axis=0), np.min(coord, axis=0)
        bounding_box = np.stack((min_value, max_value), axis=1)

        half = self.region_size / 2.0
        if self.mode == "inside":
            span = max_value - min_value
            inner_min = min_value + np.minimum(half, np.maximum(span / 2.0 - 1e-6, 0.0))
            inner_max = max_value - np.minimum(half, np.maximum(span / 2.0 - 1e-6, 0.0))
        else:
            inner_min = min_value
            inner_max = max_value

        random_x = np.random.uniform(inner_min[0], inner_max[0], self.max_regions)  # [max_regions]
        random_y = np.random.uniform(inner_min[1], inner_max[1], self.max_regions)  # [max_regions]
        random_z = np.random.uniform(inner_min[2], inner_max[2], self.max_regions)  # [max_regions]
        random_point_center = np.stack((random_x, random_y, random_z), axis=1)  # [max_regions, 3]

        # [max_regions, region_max_k, 3]
        noise = np.random.uniform(low=-half, high=half, size=(len(random_point_center), self.region_max_k, 3))
        random_point = random_point_center[:, np.newaxis, :] + noise  # [max_regions, 1, 3] + [max_regions, region_max_k, 3]
        # random_point_normal = np.zeros_like(random_point)  # [max_regions, region_max_k, 3]
        # tmp_normal = np.stack((np.eye(3), np.eye(3) * -1), axis=1)  # [3, 2, 3]

        random_face = np.random.choice(6, self.max_regions, replace=True)
        for i, val in enumerate(random_face):
            divide = val // 2
            mod = val % 2
            random_point[i, :, divide] = bounding_box[divide][mod]
            # random_point_normal[i, :] = tmp_normal[divide][mod]

        # ----- RANDOM NORMALS INSTEAD OF FACE NORMALS -----
        # shape (R, K, 3)
        rand_dir = np.random.normal(size=random_point.shape)  # gaussian noise
        # normalize to unit vectors
        norm_len = np.linalg.norm(rand_dir, axis=-1, keepdims=True) + 1e-8
        random_point_normal = (rand_dir / norm_len).astype(np.float32)

        if not self.fixed:
            region_size = np.random.randint(1, self.max_regions + 1)
            random_point = random_point[:region_size].reshape(-1, 3)
            random_point_normal = random_point_normal[:region_size].reshape(-1, 3)
            total_size = np.random.randint(max(1, len(random_point) // 8), len(random_point) + 1)

            shuffle_idx = np.arange(len(random_point))
            np.random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:total_size]
            random_point = random_point[shuffle_idx]
            random_point_normal = random_point_normal[shuffle_idx]
        else:
            random_point = random_point.reshape(-1, 3)
            random_point_normal = random_point_normal.reshape(-1, 3)

        # ----- ADD COLOR FOR NOISE -----
        if "color" in data_dict:
            orig_color = data_dict["color"]
            C = orig_color.shape[1]
            dtype = orig_color.dtype
            if self.range255:
                noise_color = np.random.randint(0, 256, size=(random_point.shape[0], C)).astype(dtype)
            else:
                noise_color = np.random.rand(random_point.shape[0], C).astype(dtype)

        # ----- CONCATENATE -----
        data_dict["coord"] = np.concatenate((data_dict["coord"], random_point), axis=0)
        if "norm" in data_dict:
            data_dict["norm"] = np.concatenate((data_dict["norm"], random_point_normal), axis=0)

        if "color" in data_dict:
            data_dict["color"] = np.concatenate((data_dict["color"], noise_color), axis=0)

        # ----- UPDATE noise_index -----
        num_new = random_point.shape[0]
        if "noise_index" not in data_dict:
            noise_index = np.zeros(len(data_dict["coord"]), dtype=int)
            noise_index[-num_new:] = 2  # mark new as noise
            data_dict["noise_index"] = noise_index
        else:
            old_len = len(data_dict["noise_index"])
            new_len = len(data_dict["coord"])
            tmp = np.ones(new_len, dtype=int) * 2
            tmp[:old_len] = data_dict["noise_index"]
            data_dict["noise_index"] = tmp

        return data_dict

@TRANSFORMS.register()
class AddNoise:
    """Add local noise clusters around randomly chosen points.

    This transform expects a dictionary containing:

    * ``"coord"``: NumPy array of shape (N, 3) with point coordinates.
    * Optionally ``"norm"``: per-point normals of shape (N, 3), required if
      ``method="custom"``.
    * Optionally ``"color"``: per-point colors of shape (N, C).
    * Optionally ``"noise_index"``: 1D array of length N indicating existing noise labels.

    The transform works by:

    1. Selecting a subset of points as **noise centers**.
       The number of centers is:

           noise_center_count = int(N * noise_size_ratio)

       (clamped to ``[1, N]``). If this is 0, no noise is added.
    2. For each center, generating up to ``noise_max_k`` noise points in its
       local neighborhood (cluster), according to the chosen ``method`` and ``boundary``.
    3. Optionally subsampling the generated noise when ``fixed=False``.
    4. Mapping the local offsets to world coordinates and appending them (and their normals/colors) to the existing arrays.
    5. Updating ``noise_index`` to mark the new points as local noise (value 1).

    Noise generation is controlled by two knobs:

    * ``method``: defines *how* offsets are sampled:
        - ``"uniform"``:
          * If ``boundary="sphere"``: sample directions uniformly and radii uniformly in volume within a ball of radius ``ball_r``.
          * If ``boundary="cube"``: sample coordinates uniformly in ``[low, high]^3``.
        - ``"gaussian"``:
          * If ``boundary="sphere"``: Gaussian around the center, clamped to lie within a ball of radius ``ball_r``.
          * If ``boundary="cube"``: Gaussian in a cube, then clipped to the interval ``[low, high]`` per axis.
        - ``"custom"``:
          * Requires a ``"norm"`` field.
          * For each center, uses its normal as a base direction and applies directional + radial jitter (using ``ball_r`` as scale) to create
            offsets in a “tube-like” pattern around the surface.
    * ``boundary``: defines the *shape* of the local region:
        - ``"sphere"``: offsets lie in or near a ball of radius ``ball_r``.
        - ``"cube"``: offsets lie in or near a cube with side approximately ``high - low`` (for uniform/gaussian).

    Colors for noise points are derived from the center colors:

    * For ``"uniform"`` and ``"gaussian"`` methods:
      - Noise points inherit exactly the color of their center.
    * For ``"custom"``:
      - Optionally add Gaussian jitter in color space (scale depends on whether ``range255`` is True or False),
      then clip to valid range.

    The number of **final** noise points is controlled by ``fixed``:

    * If ``fixed=True``:
      - Use all generated noise: ``noise_center_count * noise_max_k`` points.
    * If ``fixed=False``:
      - Randomly shuffle all noise points and keep a random subset of size
        between approximately one-eighth and the full generated count.

    Args:
        noise_size_ratio (float, optional):
            Fraction of non-noise points to use as noise centers. The number of centers is:

                noise_center_count = int(N * noise_size_ratio)

            Defaults to 1./64.
        noise_max_k (int, optional):
            Maximum number of noise points generated per center before optional subsampling.
            Defaults to 16.
        fixed (bool, optional):
            Whether to keep all generated noise points or randomly subsample them:

            * True  → keep all ``noise_center_count * noise_max_k`` noise points.
            * False → shuffle and randomly keep a subset of those points.

            Defaults to True.
        method ({"uniform", "gaussian", "custom"}, optional):
            Noise sampling strategy. See above for details. ``"custom"`` requires
            ``"norm"`` in ``data_dict``.
            Defaults to ``"uniform"``.
        boundary ({"sphere", "cube"}, optional):
            Shape of the local region in which noise is sampled. Interacts with ``method`` as described above.
            Defaults to ``"sphere"``.
        low (float, optional):
            Lower bound for cube-based offsets (used for ``boundary="cube"`` in ``"uniform"`` and ``"gaussian"``).
            Defaults to -0.1.
        high (float, optional):
            Upper bound for cube-based offsets.
            Defaults to 0.1.
        ball_r (float, optional):
            Radius of the spherical neighborhood for ``boundary="sphere"`` methods; also used as a scale for radial
            jitter in the ``"custom"`` method.
            Defaults to 0.1.
        range255 (bool, optional):
            Whether colors are stored in ``[0, 255]`` (integer-like). If True, color jitter in the ``"custom"`` method
            is done in that range. If False, colors are assumed in ``[0, 1]``.
            Defaults to False.
        apply_p (float, optional):
            Probability of applying the noise
            Defaults to 1.0.

    """
    def __init__(self, noise_size_ratio:float=1./64, noise_max_k:int=16, fixed:bool=True, method:str="uniform", boundary:str="sphere",
                 low:float=-0.1, high:float=0.1, ball_r:float=0.1, range255:bool=False, apply_p:float=1.0):
        self.noise_size_ratio = noise_size_ratio
        self.noise_max_k = noise_max_k
        self.fixed = fixed
        assert method in ["uniform", "gaussian", "custom"]
        self.method = method
        assert boundary in ["sphere", "cube"]
        self.boundary = boundary
        self.low, self.high = low, high
        self.ball_r = ball_r
        self.range255 = range255
        self.apply_p = apply_p

    def __call__(self, data_dict: dict) -> dict:
        """Apply local noise cluster augmentation to a point cloud dictionary.

        Args:
            data_dict (dict): Input dictionary containing at least ``"coord"`` (NumPy array of shape (N, 3)).
                May also contain ``"norm"``, ``"color"``, and ``"noise_index"``. For ``method="custom"``, ``"norm"`` must be present.

        Returns:
            dict: The same dictionary with additional noise points appended to the relevant per-point fields and an updated ``"noise_index"`` marking newly added noise points with label 1.
        """
        if random.random() > self.apply_p:
            return data_dict
        if "coord" not in data_dict.keys():
            return data_dict

        coord = data_dict["coord"]
        if "noise_index" in data_dict:
            coord = coord[np.logical_not(data_dict["noise_index"])]
        N = coord.shape[0]

        # how many centers
        noise_center_count = int(N * self.noise_size_ratio)
        if noise_center_count <= 0:
            return data_dict
        noise_center_count = min(noise_center_count, N)

        # choose centers
        center_idx = np.random.choice(N, noise_center_count, replace=False)
        centers = coord[center_idx]  # (C, 3)
        C = noise_center_count
        K = self.noise_max_k

        # ----------------- generate offsets (noise in local coords) -----------------
        noise_offsets = None  # (C, K, 3)
        noise_normals = None  # (C, K, 3) or None

        # helper: random unit directions
        def random_unit_vectors(shape):
            v = np.random.normal(size=shape)
            norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
            return v / norm

        if self.method == "uniform":
            if self.boundary == "sphere":
                # uniform in ball of radius ball_r
                dirs = random_unit_vectors((C, K, 3))
                u = np.random.rand(C, K)
                radii = (u ** (1.0 / 3.0)) * self.ball_r  # uniform in volume
                noise_offsets = dirs * radii[..., None]
                noise_normals = dirs  # can use direction as "normal"
            elif self.boundary == "cube":
                noise_offsets = np.random.uniform(low=self.low, high=self.high, size=(C, K, 3))
                noise_normals = random_unit_vectors((C, K, 3))

        elif self.method == "gaussian":
            if self.boundary == "sphere":
                # sample Gaussian around center
                offsets = np.random.normal(loc=0.0, scale=self.ball_r / 3.0, size=(C, K, 3))
                # radii of those offsets
                radii = np.linalg.norm(offsets, axis=-1, keepdims=True) + 1e-8  # (C, K, 1)
                # clamp radius to ball_r by scaling each vector
                # scale = 1 if inside, ball_r / r if outside
                scale = np.minimum(1.0, self.ball_r / radii)  # (C, K, 1), broadcasts over last dim
                offsets = offsets * scale  # (C, K, 3)
                noise_offsets = offsets
                # recompute or reuse direction as normals
                new_radii = np.linalg.norm(offsets, axis=-1, keepdims=True) + 1e-8
                noise_normals = offsets / new_radii
            elif self.boundary == "cube":
                # Gaussian in a cube, then clipped to [low, high]
                scale = (self.high - self.low) / 6.0  # ~99.7% in [low, high]
                offsets = np.random.normal(loc=0.0, scale=scale, size=(C, K, 3))
                offsets = np.clip(offsets, self.low, self.high)
                noise_offsets = offsets
                noise_normals = random_unit_vectors((C, K, 3))

        elif self.method == "custom":
            assert "norm" in data_dict, "custom method requires 'norm' in data_dict."

            noise_size = len(center_idx)  # N' centers
            K = self.noise_max_k

            # ---- center normals ----
            normal_centers = np.asarray(data_dict["norm"], dtype=np.float32)[center_idx]  # (N', 3)
            # make sure they are unit vectors
            normal_centers /= (np.linalg.norm(normal_centers, axis=-1, keepdims=True) + 1e-8)

            # [N', K, 3] repeat normals per patch
            repeat_normals = np.repeat(normal_centers[:, np.newaxis, :], K, axis=1)  # (N', K, 3)

            # =====================================================
            # 1) JITTER FOR NORMAL DIRECTION (directional jitter)
            # =====================================================
            # small Gaussian noise added to the normal, then renormalized
            dir_sigma = 2  # strength of direction jitter (tune if needed)
            dir_jitter = np.random.normal(
                loc=0.0,
                scale=dir_sigma,
                size=(noise_size, K, 3)
            )

            perturbed_dir = repeat_normals + dir_jitter  # (N', K, 3)
            dir_norm = np.linalg.norm(perturbed_dir, axis=-1, keepdims=True) + 1e-8
            dir_unit = perturbed_dir / dir_norm  # (N', K, 3), unit direction per noise point

            # =====================================================
            # 2) JITTER FOR DISTANCE TO CENTER (radial jitter)
            # =====================================================
            # base radius in [0, ball_r], biased toward outer region (your original idea)
            u = np.random.rand(noise_size, K)  # (N', K)
            base_radius = (u ** (1.0 / 3.0)) * self.ball_r  # (N', K), >= 0

            # additive jitter around base_radius
            dist_sigma = 1.  # in *absolute* units of ball_r (tune this)
            jitter = np.random.normal(
                loc=0.0,
                scale=dist_sigma * self.ball_r,
                size=(noise_size, K)
            )  # (N', K)

            r = base_radius + jitter  # (N', K)
            r = np.clip(r, self.ball_r / 4.0, self.ball_r)  # keep in [0, ball_r]
            r = r[:, :, None]  # (N', K, 1)

            # =====================================================
            # FINAL OFFSETS & NORMALS (LOCAL COORDS)
            # =====================================================
            # offset from center = jittered direction * jittered distance
            noise_offsets = dir_unit * r  # (N', K, 3)
            noise_normals = dir_unit  # (N', K, 3)
        else:
            return data_dict

        # ----------------- COLOR: center-based -----------------
        noise_colors = None
        if "color" in data_dict:
            orig_color = np.asarray(data_dict["color"])
            center_colors = orig_color[center_idx]  # (C, Cc)
            base_colors = np.repeat(center_colors[:, None, :], K, axis=1)  # (C, K, Cc)

            if self.method in ("uniform", "gaussian"):
                # exactly same color as center
                noise_colors = base_colors.astype(orig_color.dtype)

            elif self.method == "custom":
                if self.range255:
                    color_jitter = np.random.normal(loc=0.0, scale=10.0, size=base_colors.shape)
                    noise_colors = base_colors + color_jitter
                    noise_colors = np.clip(noise_colors, 0, 255).round().astype(orig_color.dtype)
                else:
                    color_jitter = np.random.normal(loc=0.0, scale=0.05, size=base_colors.shape)
                    noise_colors = base_colors + color_jitter
                    noise_colors = np.clip(noise_colors, 0.0, 1.0).astype(orig_color.dtype)

        # ----------------- map to world coords -----------------
        noise_points = noise_offsets + centers[:, None, :]  # (C, K, 3)

        # flatten
        noise_points = noise_points.reshape(-1, 3)
        if noise_normals is not None:
            noise_normals = noise_normals.reshape(-1, 3)
        if noise_colors is not None:
            noise_colors = noise_colors.reshape(-1, noise_colors.shape[-1])

        # ----------------- optional subsampling -----------------
        if not self.fixed:
            idx = np.arange(len(noise_points))
            np.random.shuffle(idx)
            total = np.random.randint(max(1, len(noise_points) // 8), len(noise_points) + 1)
            idx = idx[:total]

            noise_points = noise_points[idx]
            if noise_normals is not None:
                noise_normals = noise_normals[idx]
            if noise_colors is not None:
                noise_colors = noise_colors[idx]

        # ----------------- append to data_dict -----------------
        data_dict["coord"] = np.concatenate([data_dict["coord"], noise_points], axis=0)

        if "norm" in data_dict and noise_normals is not None:
            data_dict["norm"] = np.concatenate([data_dict["norm"], noise_normals], axis=0)

        if "color" in data_dict and noise_colors is not None:
            data_dict["color"] = np.concatenate([data_dict["color"], noise_colors], axis=0)

        # ----------------- noise_index -----------------
        num_noise = noise_points.shape[0]
        if "noise_index" not in data_dict:
            ni = np.zeros(len(data_dict["coord"]), dtype=int)
            ni[-num_noise:] = 1
            data_dict["noise_index"] = ni
        else:
            old_len = len(data_dict["noise_index"])
            new_len = len(data_dict["coord"])
            ni = np.ones(new_len, dtype=int) * 1
            ni[:old_len] = data_dict["noise_index"]
            data_dict["noise_index"] = ni

        return data_dict

@TRANSFORMS.register()
class ToTensor:
    """Recursively convert NumPy arrays, scalars, and container structures to PyTorch tensors.

    This transform is designed to work on:

    * Individual values:
      - ``torch.Tensor``: returned as-is.
      - ``int`` → ``LongTensor([value])``.
      - ``float`` → ``FloatTensor([value])``.
      - ``str``: returned as-is (strings are *not* converted).
      - ``np.ndarray`` with:
        - boolean dtype → ``torch.from_numpy(arr)`` (bool tensor),
        - integer dtype → ``torch.from_numpy(arr).long()``,
        - float dtype → ``torch.from_numpy(arr).float()``.
    * Containers:
      - ``Mapping`` (e.g. ``dict``): converts each value recursively, preserving keys.
      - ``Sequence`` (e.g. ``list``, ``tuple``): converts each element recursively and
        returns a Python ``list`` of tensors/converted items.

    Any unsupported type will raise a ``TypeError``.

    Typical usage is at the end of a preprocessing pipeline, to convert a nested
    sample dictionary (coords, features, labels) from NumPy to tensors.
    """
    def __call__(self, data: Any) -> Any:
        """Convert input data (and nested contents) to PyTorch tensors.
        Args:
            data: Arbitrary input to convert. Can be a scalar, NumPy array,
                tensor, mapping (e.g. dict), or sequence (e.g. list/tuple).

        Returns:
            Converted object where all supported leaves are PyTorch tensors,
            and the original container structure (dict/list) is preserved.

        Raises:
            TypeError: If ``data`` (or some nested leaf) has a type that cannot be converted to a tensor.

        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register()
class FinalFeatures:
    """Assemble a final feature tensor and manage bookkeeping fields in a sample dict.

    This transform is typically used at the **end** of a preprocessing pipeline to:

    1. Build a unified feature array under the key ``"feat"`` by selecting and
       concatenating one or more existing fields from ``data_dict``.
    2. Optionally remove some intermediate fields (e.g., ``"norm"``,
       auxiliary features) to save memory.
    3. Optionally add offset fields (``"offset"``, ``"fps_offset"``) that are
       useful when batching variable-length point clouds.

    Behavior:

    * Feature construction:
      - If ``feat`` is a string, ``data_dict["feat"]`` is set to ``data_dict[feat]``.
      - If ``feat`` is a list/tuple of strings, the corresponding arrays are concatenated along the last dimension:

            feat = np.concatenate([data_dict[name] for name in feat], axis=-1)

      All specified feature names must exist in ``data_dict``.

    * Field removal:
      - If ``remove`` is a string, that key is deleted from ``data_dict``.
      - If ``remove`` is a list/tuple of strings, each corresponding key is deleted.
      - All specified keys must exist in ``data_dict``.

    * Offsets:
      - If ``add_offset`` is True and ``"offset"`` is not already present, then:

            data_dict["offset"] = len(data_dict["coord"])

        This is often interpreted as the number of points in this sample.
      - If ``add_fps_offset`` is True and ``"fps_offset"`` is not present but ``"fps_index"`` exists, then:

            data_dict["fps_offset"] = len(data_dict["fps_index"])

        This is typically the number of FPS (farthest-point sampling) indices for this sample.

    Args:
        feat (str or sequence of str, optional):
            Name(s) of fields in ``data_dict`` to be used as final features. If a sequence, the corresponding arrays are
            concatenated along the last dimension and stored in ``data_dict["feat"]``. If ``None``, no feature tensor is constructed.
            Defaults to ``"coord"``.
        remove (str or sequence of str, optional):
            Name(s) of fields to delete from ``data_dict`` after (optional) feature construction. This is useful to drop
            intermediate fields (e.g., ``"norm"``) that are no longer needed. If ``None``, no keys are removed.
            Defaults to ``"norm"``.
        add_offset (bool, optional):
            If True and ``"offset"`` is not already present, add an integer offset equal to ``len(data_dict["coord"])``.
            Defaults to True.
        add_fps_offset (bool, optional):
            If True, ``"fps_offset"`` is not present, and ``"fps_index"`` exists, add an integer offset equal to
            ``len(data_dict["fps_index"])``.
            Defaults to True.
    """
    def __init__(self, feat: str | tuple[str] ="coord", remove: str | tuple[str]="norm",
                 add_offset: bool=True, add_fps_offset: bool=True):
        self.feat = feat
        self.remove = remove
        self.add_offset = add_offset
        self.add_fps_offset = add_fps_offset

    def __call__(self, data_dict: dict) -> dict:
        """Construct the final feature array and update bookkeeping fields.

       Args:
           data_dict (dict): Sample dictionary containing at least the keys referenced by ``self.feat`` (if not ``None``),
               plus ``"coord"`` (used for ``offset``) and optionally ``"fps_index"`` (used for ``fps_offset``).

       Returns:
           dict: The same dictionary, with:
               * a new ``"feat"`` field (if requested),
               * selected keys removed,
               * and optional ``"offset"`` / ``"fps_offset"`` fields added.
       """
        if self.feat is not None:
            if isinstance(self.feat, (list, tuple)):
                tmp = []
                for val in self.feat:
                    assert val in data_dict
                    tmp.append(data_dict[val])
                tmp = np.concatenate(tmp, axis=-1)
            else:
                assert self.feat in data_dict
                tmp = data_dict[self.feat]
            data_dict["feat"] = tmp

        if self.remove is not None:
            if isinstance(self.remove, (list, tuple)):
                for val in self.remove:
                    assert val in data_dict
                    del data_dict[val]
            else:
                assert self.remove in data_dict
                del data_dict[self.remove]

        if self.add_offset:
            if "offset" not in data_dict:
                data_dict["offset"] = len(data_dict["coord"])

        if self.add_fps_offset:
            if "fps_offset" not in data_dict and "fps_index" in data_dict:
                data_dict["fps_offset"] = len(data_dict["fps_index"])

        return data_dict


def show_data(data_dict):
    print(data_dict.keys())
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)


if __name__ == '__main__':
    import glob
    files = glob.glob(r"ShapeNet/02691156/*")
    file = files[1]
    print(len(files), file)
    data = torch.load(file, weights_only=False)
    sampling_class = Sampling(n_pts=2048, method="fps")
    normalize_coord_class = NormalizeCoord()
    normalize_normal_class = NormalizeNormal()

    data = sampling_class(data)
    data = normalize_normal_class(normalize_coord_class(data))

    random_rotate = RandomRotate()
    random_scale = RandomScale(apply_p=0.5)
    random_jitter = RandomJitter(apply_p=0.5)
    random_flip = RandomFlip(apply_p=0.5)
    data = random_flip(random_jitter(random_scale(random_rotate(data))))
    show_data(data)

    add_background = AddBackgroundNoise(apply_p=1.0)
    data = add_background(data)



    # add_noise_class = AddNoise(noise_size_ratio=1/64, noise_max_k=16, fixed=True, method="custom")
    # data = add_noise_class(data)
    # print(data["noise_index"])

    # shuffle_class = ShufflePoint(apply_p=1.0)
    # data = normalize_normal_class(normalize_coord_class(shuffle_class(data)))
    #
    # show_data(data)
    # torch.save(data, "tmp.pth")
