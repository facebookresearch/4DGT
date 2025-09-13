# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

from enum import auto, Enum
from typing import Union

import cv2
import numpy as np
import torch

from scipy import interpolate
from scipy.spatial.transform import Rotation

from .chunk_utils import multi_gather

from .data_utils import as_numpy_func


def compute_camera_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # c2ws = affine_inverse(w2cs)  # N, L, 3, 4
    # src_exts = affine_padding(w2cs)  # N, L, 4, 4

    # tar_c2ws = c2ws
    # src_c2ws = affine_inverse(src_exts)
    centers_target = tar_c2ws[..., :3, 3]  # N, L, 3
    centers_source = src_c2ws[..., :3, 3]  # N, L, 3

    # Using distance between centers for camera selection
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(
        dim=-1
    )  # N, N, L,

    # Source view index and there similarity
    src_sims, src_inds = sims.sort(
        dim=1, descending=True
    )  # similarity to source views # Target, Source, Latent
    return src_sims, src_inds  # N, N, L


def compute_camera_zigzag_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # Get the camera centers
    centers_target = tar_c2ws[..., :3, 3]  # (Vt, F, 3)
    centers_source = src_c2ws[..., :3, 3]  # (Vs, F, 3)

    # Compute the distance between the centers
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(
        dim=-1
    )  # (Vt, Vs, F)
    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # (Vt, Vs, F), (Vt, Vs, F)

    # Select the closest source view as the reference view for each target view
    ref_view = multi_gather(
        centers_source.permute(1, 0, 2), src_inds.permute(2, 0, 1)[..., 0]
    ).permute(1, 0, 2)  # (Vt, F, 3)

    # Compute the cross product between the reference view and target view, and the cross product between the source views and the target view
    ref_cross = torch.cross(ref_view, centers_target, dim=-1)  # (Vt, F, 3)
    src_cross = torch.cross(
        centers_source[None], centers_target[:, None], dim=-1
    )  # (Vt, Vs, F, 3)

    # Compute the inner product between the cross products to determine the zigzag placing
    zigzag = (ref_cross[:, None] * src_cross).sum(dim=-1)  # (Vt, Vs, F)

    zigzag_src_sims, zigzag_src_inds = src_sims.clone(), src_inds.clone()
    # Re-indexing the similarity and indices
    for v in range(len(zigzag)):
        # Get the sorted zig and zag similarity and indices respectively
        zig_msk = torch.sum(
            torch.eq(
                torch.arange(len(centers_source))[zigzag[v, :, 0] > 0][:, None],
                src_inds[v, :, 0],
            ),
            dim=0,
        ).bool()
        zig_src_sims, zig_src_inds = (
            src_sims[v][zig_msk],
            src_inds[v][zig_msk],
        )  # (L, F), (L, F)
        zag_msk = torch.sum(
            torch.eq(
                torch.arange(len(centers_source))[zigzag[v, :, 0] < 0][:, None],
                src_inds[v, :, 0],
            ),
            dim=0,
        ).bool()
        zag_src_sims, zag_src_inds = (
            src_sims[v][zag_msk],
            src_inds[v][zag_msk],
        )  # (R, F), (R, F)

        # Concatenate the zig and zag similarity and indices in order zig-zag-zig-zag-...
        size = min(len(zig_src_sims), len(zag_src_sims))
        zigzag_src_sims[v, 0 : size * 2 : 2], zigzag_src_sims[v, 1 : size * 2 : 2] = (
            zig_src_sims[:size],
            zag_src_sims[:size],
        )  # (S*2, F), (S*2, F)
        zigzag_src_inds[v, 0 : size * 2 : 2], zigzag_src_inds[v, 1 : size * 2 : 2] = (
            zig_src_inds[:size],
            zag_src_inds[:size],
        )  # (S*2, F), (S*2, F)

        # Concatenate the remaining similarity and indices
        if len(zig_src_sims) > len(zag_src_sims):
            zigzag_src_sims[v, size * 2 :], zigzag_src_inds[v, size * 2 :] = (
                zig_src_sims[size:],
                zig_src_inds[size:],
            )
        else:
            zigzag_src_sims[v, size * 2 :], zigzag_src_inds[v, size * 2 :] = (
                zag_src_sims[size:],
                zag_src_inds[size:],
            )

    # Return the zigzag similarity and indices
    return zigzag_src_sims, zigzag_src_inds


class Sourcing(Enum):
    # Type of source indexing
    DISTANCE = auto()  # the default source indexing
    ZIGZAG = auto()  # will index the source view in zigzag order


class Interpolation(Enum):
    # Type of interpolation to use
    SMOOTH = auto()  # the default interpolation
    FOCUS = auto()  # the default interpolation
    CUBIC = auto()  # the default interpolation
    ORBIT = auto()  # will find a full circle around the cameras, the default orbit path
    SLERP = auto()  # will find a full circle around the cameras, the default orbit path
    SPIRAL = auto()  # will perform spiral motion around the cameras
    SECTOR = auto()  # will find a circular sector around the cameras
    NONE = auto()  # used as is


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-13)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


# From https://github.com/NVLabs/instant-ngp


def compute_center_of_attention(
    c2ws: np.ndarray,
    # fix_flipped_z: bool = True
) -> np.ndarray:
    # Extract the relevant parts of the transformation matrices
    c2ws = c2ws[..., :3, :4]
    # Get the origins and directions
    origins = c2ws[:, :, 3]
    directions = c2ws[:, :, 2]
    # Normalize directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # Compute cross products and denominators
    cross_products = np.cross(directions[:, None, :], directions[None, :, :])
    denom = np.linalg.norm(cross_products, axis=2) ** 2
    # Compute the differences in origins
    t = origins[None, :, :] - origins[:, None, :]
    # Prepare matrices for determinant calculation
    det_matrices_db = np.stack(
        [t, np.broadcast_to(directions[None, :, :], t.shape), cross_products], axis=-1
    )
    det_matrices_da = np.stack(
        [t, np.broadcast_to(directions[:, None, :], t.shape), cross_products], axis=-1
    )
    # Compute determinants
    det_t_db_c = np.linalg.det(det_matrices_db)
    det_t_da_c = np.linalg.det(det_matrices_da)
    # Compute ta and tb
    ta = det_t_db_c / (denom + 1e-8)
    tb = det_t_da_c / (denom + 1e-8)
    # Compute closest points
    closest_points = (
        origins[:, None, :]
        + ta[..., None] * directions[:, None, :]
        + origins[None, :, :]
        + tb[..., None] * directions[None, :, :]
    ) * 0.5
    # Compute weights
    weights = denom
    # Filter out small weights
    totp = np.sum(closest_points * weights[..., None], axis=(0, 1))
    totw = np.sum(weights)
    # Compute the weighted average
    totp /= totw + 1e-8

    # if fix_flipped_z:
    #     center = c2ws[..., 3].mean(0)  # (3)
    #     z = totp - center
    #     z_mean = c2ws[..., 2].mean(0)  # (3)
    #     if np.dot(z_mean, z) < 0:
    #         totp = center + normalize(z) * 1.0  # FIXME: 1.0 is a hard-coded look at
    return totp


# From: https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/ray_utils.py


def average_c2ws(
    c2ws: np.ndarray,
    align_cameras: bool = True,
    look_at_center: bool = True,
    fix_flipped_z: bool = True,
) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    c2ws = c2ws[..., :3, :4]
    if align_cameras:
        # 1. Compute the center
        center = compute_center_of_attention(c2ws)  # (3)
        # 2. Compute the z axis
        z = -normalize(c2ws[..., 1].mean(0))  # (3) # FIXME: WHY?
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 2].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    else:
        # 1. Compute the center
        center = c2ws[..., 3].mean(0)  # (3)
        # 2. Compute the z axis
        if look_at_center:
            look = compute_center_of_attention(c2ws)  # (3)
            z = normalize(look - center)
            if fix_flipped_z:
                z_mean = normalize(c2ws[..., 2].mean(0))  # (3)
                if np.dot(z, z_mean) < 0:
                    z = z_mean  # do not try to look at center if this isn't reasonable
        else:
            z = normalize(c2ws[..., 2].mean(0))  # (3)
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 1].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    c2w_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return c2w_avg


def align_c2ws(c2ws: np.ndarray, c2w_avg: Union[np.ndarray, None] = None) -> np.ndarray:
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    c2ws = c2ws[..., :3, :4]
    c2w_avg = c2w_avg if c2w_avg is not None else average_c2ws(c2ws)  # (3, 4)
    c2w_avg_homo = np.eye(4, dtype=c2ws.dtype)
    c2w_avg_homo[:3] = (
        c2w_avg  # convert to homogeneous coordinate for faster computation
    )

    last_row = np.tile(
        np.asarray([0, 0, 0, 1], dtype=np.float32), (len(c2ws), 1, 1)
    )  # (N_images, 1, 4)
    c2ws_homo = np.concatenate(
        [c2ws, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    c2ws_centered = np.linalg.inv(c2w_avg_homo) @ c2ws_homo  # (N_images, 4, 4)
    c2ws_centered = c2ws_centered[:, :3]  # (N_images, 3, 4)

    return c2ws_centered


def average_w2cs(w2cs: np.ndarray) -> np.ndarray:
    # Transform the world2camera extrinsic from matrix representation to vector representation
    rvecs = np.array(
        [cv2.Rodrigues(w2c[:3, :3])[0] for w2c in w2cs], dtype=np.float32
    )  # (V, 3, 1)
    tvecs = w2cs[:, :3, 3:]  # (V, 3, 1)

    # Compute the average view direction and center in vector mode
    rvec_avg = rvecs.mean(axis=0)  # (3, 1)
    tvec_avg = tvecs.mean(axis=0)  # (3, 1)

    # Back to matrix representation
    w2c_avg = np.concatenate([cv2.Rodrigues(rvec_avg)[0], tvec_avg], axis=1)
    return w2c_avg


def gen_cam_interp_func_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    center_t, center_u, front_t, front_u, up_t, up_u = gen_cam_interp_params_bspline(
        c2ws, smoothing_term, per
    )

    def f(us: np.ndarray):
        if isinstance(us, int) or isinstance(us, float):
            us = np.asarray([us])
        if isinstance(us, list):
            us = np.asarray(us)

        # The interpolation t
        center = np.asarray(interpolate.splev(us, center_t)).T.astype(c2ws.dtype)
        v_front = np.asarray(interpolate.splev(us, front_t)).T.astype(c2ws.dtype)
        v_up = np.asarray(interpolate.splev(us, up_t)).T.astype(c2ws.dtype)

        # Normalization
        v_front = normalize(v_front)
        v_up = normalize(v_up)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)

        # Combination
        render_c2ws = np.stack([v_right, v_down, v_front, center], axis=-1)
        return render_c2ws

    return f


def gen_cam_interp_params_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    """Return B-spline interpolation parameters for the camera # MARK: Quite easy to error out
    Actually this should be implemented as a general interpolation function
    Reference get_camera_up_front_center for the definition of worldup, front, center
    Args:
        smoothing_term(float): degree of smoothing to apply on the camera path interpolation
    """
    centers = c2ws[..., :3, 3]
    fronts = c2ws[..., :3, 2]
    ups = -c2ws[..., :3, 1]

    center_t, center_u = interpolate.splprep(
        centers.T, s=smoothing_term, per=per
    )  # array of u corresponds to parameters of specific camera points
    front_t, front_u = interpolate.splprep(
        fronts.T, s=smoothing_term, per=per
    )  # array of u corresponds to parameters of specific camera points
    up_t, up_u = interpolate.splprep(
        ups.T, s=smoothing_term, per=per
    )  # array of u corresponds to parameters of specific camera points
    return center_t, center_u, front_t, front_u, up_t, up_u


def cubic_spline_weird_impl(us: np.ndarray, N: int):
    # FIXME: This doesn't look like a cubic spline, more a bezier curve
    if isinstance(us, (int, float)):
        us = np.asarray([us])
    if isinstance(us, list):
        us = np.asarray(us)

    # Normalize us to [0, 1] range
    t = (N - 1) * us  # expanded to the length of the sequence
    i1 = np.floor(t).astype(np.int32)
    i0 = i1 - 1
    i2 = i1 + 1
    i3 = i1 + 2

    # Handle boundary conditions
    i0 = np.clip(i0, 0, N - 1)
    i1 = np.clip(i1, 0, N - 1)
    i2 = np.clip(i2, 0, N - 1)
    i3 = np.clip(i3, 0, N - 1)

    # Re-normalize t to [0, 1] within each interval
    t = t - i1
    t = t.astype(np.float32)

    # Compute cubic spline coefficients
    a = (1 - t) ** 3 / 6.0
    b = (3 * t**3 - 6 * t**2 + 4) / 6.0
    c = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6.0
    d = t**3 / 6.0

    # # Compute cubic spline coefficients
    # a = -1 / 6 * (t**3) + 1 / 2 * (t**2) - 1 / 2 * t + 1 / 6
    # b = 1 / 2 * (t**3) - (t**2) + 2 / 3
    # c = -1 / 2 * (t**3) + 1 / 2 * (t**2) + 1 / 2 * t + 1 / 6
    # d = 1 / 6 * (t**3)

    return t, (i0, i1, i2, i3), (a, b, c, d)


class InterpolatingExtrinsics:
    def __init__(self, c2w: np.ndarray) -> None:
        self.Q = Rotation.from_matrix(c2w[..., :3, :3]).as_quat()
        self.T = c2w[..., :3, 3]

    def __add__(self, rhs: InterpolatingExtrinsics):  # FIXME: Dangerous
        Ql, Qr = self.Q, rhs.Q
        Qr = np.where((Ql * Qr).sum(axis=-1, keepdims=True) < 0, -Qr, Qr)
        self.Q = Ql + Qr
        self.T = self.T + rhs.T
        return self

    def __radd__(self, lhs: InterpolatingExtrinsics):
        return self.__add__(lhs)

    def __mul__(self, rhs: np.ndarray):
        self.Q = rhs[..., None] * self.Q
        self.T = rhs[..., None] * self.T
        return self  # inplace modification

    def __rmul__(self, lhs: np.ndarray):
        return self.__mul__(lhs)

    def numpy(self):
        return np.concatenate(
            [Rotation.from_quat(self.Q).as_matrix(), self.T[..., None]], axis=-1
        ).astype(np.float32)


def gen_cubic_spline_interp_func(
    c2ws: np.ndarray, smoothing_term: int = 1, *args, **kwargs
):
    # Split interpolation
    N = len(c2ws)
    assert N > 3, "Cubic Spline interpolation requires at least four inputs"
    if smoothing_term == 0:
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = N - 1 + 4 - 2  # should remove last one segment, please just work...
        c2ws = np.concatenate([c2ws[-2:], c2ws, c2ws[:2]])

    def lf(us: np.ndarray):
        N = len(c2ws)  # should this be recomputed?
        t, (i0, i1, i2, i3), (a, b, c, d) = cubic_spline_weird_impl(us, N)

        # Extra inter target
        c0, c1, c2, c3 = (
            InterpolatingExtrinsics(c2ws[i0]),
            InterpolatingExtrinsics(c2ws[i1]),
            InterpolatingExtrinsics(c2ws[i2]),
            InterpolatingExtrinsics(c2ws[i3]),
        )
        c = c0 * a + c1 * b + c2 * c + c3 * d  # to utilize operator overloading
        c = c.numpy()  # from InterpExt to numpy
        if isinstance(us, int) or isinstance(us, float):
            c = c[0]  # remove extra dim
        return c

    if smoothing_term == 0:

        def pf(us):
            return lf(
                (us * N - low) / (high - low)
            )  # periodic function will call the linear function

        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f


def slerp(
    q0: torch.Tensor, q1: torch.Tensor, qs: torch.Tensor, shortest_arc: bool = True
):
    from easyvolcap.utils.math_utils import normalize

    cos_omega = (q0 * q1).sum(dim=-1)

    # Flip some quaternions to perform shortest arc interpolation.
    if shortest_arc:
        q1 = q1.clone()
        q1[cos_omega < 0, :] *= -1
        cos_omega = cos_omega.abs()

    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - qs) * omega)
    beta = torch.sin(qs * omega)

    # Use linear interpolation for nearby quaternions
    alpha[nearby_quaternions] = 1 - qs[nearby_quaternions]
    beta[nearby_quaternions] = qs[nearby_quaternions]

    # Interpolation
    q = alpha[..., None] * q0 + beta[..., None] * q1
    q = normalize(q)

    return q


def gen_splerp_interp_func(
    c2ws: Union[torch.Tensor, np.ndarray],
    smoothing_term: float = 10.0,
    device: str = "cuda",
    shortest_arc: bool = True,
    *args,
    **kwargs,
):
    c2ws = (
        torch.as_tensor(c2ws).to(device, non_blocking=True)
        if not isinstance(c2ws, torch.Tensor)
        else c2ws
    )

    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    q = matrix_to_quaternion(c2ws[..., :3, :3])  # N, 4
    t = c2ws[..., :3, 3]  # N, 3

    x = t
    N, C = x.shape
    t = torch.linspace(0, 1, N, device=c2ws.device)
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

    coeffs = natural_cubic_spline_coeffs(t, x)
    spline = NaturalCubicSpline(coeffs)

    def f(us: Union[np.ndarray, torch.Tensor]):
        is_numpy = isinstance(us, np.ndarray)
        us = (
            torch.as_tensor(us, device=device)
            if not isinstance(us, torch.Tensor)
            else c2ws
        )
        us = us.to(c2ws, non_blocking=True)

        t = spline.evaluate(us)  # N, 3

        nonlocal q
        qs = us * (N - 1)
        i0 = qs.floor().int()
        i0 = i0.clip(0, N - 2)
        i1 = i0 + 1
        qs = qs - i0
        q0 = q[i0]
        q1 = q[i1]
        q = slerp(q0, q1, qs)
        r = quaternion_to_matrix(q)
        c = torch.cat([r, t[..., None]], dim=-1)

        if is_numpy:
            c = c.detach().cpu().numpy()
        return c

    return f


def gen_linear_interp_func(
    lins: np.ndarray, smoothing_term=10.0
):  # smoothing_term <= will loop the interpolation
    if smoothing_term == 0:
        n = len(lins)
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = n - 1 + 4 - 2  # should remove last one segment, please just work...
        lins = np.concatenate([lins[-2:], lins, lins[:2]])

    # cf = interpolate.interp1d(np.linspace(0, 1, len(lins), dtype=np.float32), lins, axis=-2, kind='cubic' if len(lins) > 3 else 'linear')  # repeat
    cf = interpolate.interp1d(
        np.linspace(0, 1, len(lins), dtype=np.float32), lins, axis=-2, kind="linear"
    )  # repeat

    if smoothing_term == 0:

        def pf(us):
            return cf(
                (us * n - low) / (high - low)
            )  # periodic function will call the linear function

        f = pf  # periodic function
    else:
        f = cf  # linear function
    return f


def generate_smooth_path(
    c2ws: np.ndarray, n_render_views=50, smoothing_term: float = 3.0, **kwargs
):
    # Compute the center of attension
    # Interpolate the center of the camera using Cubic Spline connecting all dots
    # Using SLERP to interpolate between the lookat vector and the actual orientation
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    f_slerp = gen_splerp_interp_func(c2ws, smoothing_term)
    f_cubic = gen_cubic_spline_interp_func(c2ws, smoothing_term)
    c_slerp = f_slerp(us)  # V, 3, 4
    c_cubic = f_cubic(us)  # V, 3, 4

    alpha = (1 - np.abs(us - 0.5) * 2) ** (1 / smoothing_term)

    q0 = Rotation.from_matrix(c_slerp[..., :3, :3]).as_quat().astype(np.float32)  # V, 4
    q1 = Rotation.from_matrix(c_cubic[..., :3, :3]).as_quat().astype(np.float32)  # V, 4
    q = as_numpy_func(slerp)(q0, q1, alpha)  # V, 4
    r = Rotation.from_quat(q).as_matrix().astype(np.float32)  # V, 3, 3

    t = (
        c_slerp[..., :3, 3] * (1 - alpha[..., None])
        + c_cubic[..., :3, 3] * alpha[..., None]
    )  # V, 3
    c = np.concatenate([r, t[..., None]], axis=-1)  # V, 3, 4
    return c


def generate_focus_path(
    c2ws: np.ndarray, n_render_views=50, smoothing_term: float = 3.0, **kwargs
):
    # Compute the center of attension
    # Interpolate the center of the camera using Cubic Spline connecting all dots
    # Using SLERP to interpolate between the lookat vector and the actual orientation
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    center = compute_center_of_attention(c2ws)  # 3,
    f = gen_splerp_interp_func(c2ws)
    c = f(us)  # V, 3, 4
    t = c[..., :3, 3]
    right = c[..., :3, 0]  # V, 3

    front = normalize(t - center)  # V, 3
    alpha = (1 - np.abs(us - 0.5) * 2) ** (1 / smoothing_term)
    down = np.cross(front, right)  # V, 3
    right = np.cross(down, front)  # V, 3
    r = np.stack([right, down, front], axis=-1)  # V, 3, 3

    q0 = Rotation.from_matrix(c[..., :3, :3]).as_quat().astype(np.float32)  # V, 4
    q1 = Rotation.from_matrix(r).as_quat().astype(np.float32)  # V, 4
    q = as_numpy_func(slerp)(q0, q1, alpha)  # V, 4

    r = Rotation.from_quat(q).as_matrix().astype(np.float32)  # V, 3, 3
    c = np.concatenate([r, t[..., None]], axis=-1)  # V, 3, 4
    return c


def generate_slerp_path(
    c2ws: np.ndarray, n_render_views=50, smoothing_term: float = 1.0, **kwargs
):
    # Store interpolation parameters
    f = gen_splerp_interp_func(c2ws, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    return f(us)


def interpolate_camera_path(
    c2ws: np.ndarray, n_render_views=50, smoothing_term: float = 1.0, **kwargs
):
    # Store interpolation parameters
    f = gen_cubic_spline_interp_func(c2ws, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    return f(us)


def interpolate_camera_lins(
    lins: np.ndarray, n_render_views=50, smoothing_term: float = 1.0, **kwargs
):
    # Store interpolation parameters
    f = gen_linear_interp_func(lins, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=lins.dtype)
    return f(us)


def generate_spiral_path(
    c2ws: np.ndarray,
    n_render_views=300,
    n_rots=2,
    zrate=0.5,
    percentile=70,
    focal_offset=0.0,
    radius_ratio=1.0,
    xyz_ratio=(1.0, 1.0, 0.25),
    xyz_offset=(0.0, 0.0, 0.0),
    radii_overwrite=None,
    min_focal=0.5,
    c2w_avg_overwrite=None,
    **kwargs,
) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.
    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py
    """
    # Prepare input data
    c2ws = c2ws[..., :3, :4]

    # Center pose
    if c2w_avg_overwrite is None:
        c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)  # [3, 4]
    else:
        c2w_avg = c2w_avg_overwrite
        c2w_avg = c2w_avg[..., :3, :4]

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    focal = focal_offset + np.linalg.norm(
        compute_center_of_attention(c2ws) - c2w_avg[..., 3]
    )  # (3)
    focal = max(min_focal, focal)
    if radii_overwrite is None:
        # Get radii for spiral path using 70th percentile of camera origins.
        radii = (
            np.percentile(np.abs(c2ws[:, :3, 3] - c2w_avg[..., 3]), percentile, 0)
            * radius_ratio
        )  # N, 3
        radii = np.concatenate([xyz_ratio * radii, [1.0]])  # 4,
    else:
        radii = np.concatenate([np.asarray(radii_overwrite), [1.0]])  # 4,

    # Generate c2ws for spiral path.
    render_c2ws = []
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_render_views, endpoint=False):
        t = radii * [
            np.cos(theta),
            np.sin(theta),
            np.sin(theta * zrate),
            1.0,
        ] + np.concatenate([xyz_offset, [0.0]])

        center = c2w_avg @ t
        center = center.astype(c2ws.dtype)
        lookat = c2w_avg @ np.array([0, 0, focal, 1.0], dtype=c2ws.dtype)

        v_front = -normalize(center - lookat)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws


def generate_hemispherical_orbit(
    c2ws: np.ndarray,
    n_render_views=50,
    orbit_height=0.0,
    orbit_radius=-1,
    radius_ratio=1.0,
    **kwargs,
):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    TODO: Implement this for non-centered camera paths
    """
    # Center pose
    c2w_avg = average_c2ws(c2ws)  # [3, 4]

    # Find the origin and radius for the orbit
    origins = c2ws[:, :3, 3]
    radius = (
        (np.sqrt(np.mean(np.sum(origins**2, axis=-1))) * radius_ratio)
        if orbit_radius <= 0
        else orbit_radius
    )

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Assume that z-axis points up towards approximate camera hemispherical
    sin_phi = np.mean(origins[:, 2], axis=0) / radius
    cos_phi = np.sqrt(1 - sin_phi**2)
    render_c2ws = []

    for theta in np.linspace(
        0.0, 2.0 * np.pi, n_render_views, endpoint=False, dtype=c2ws.dtype
    ):
        center = radius * np.asarray(
            [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi],
            dtype=c2ws.dtype,
        )
        center[2] += orbit_height
        v_front = -normalize(center)
        center += c2w_avg[..., :3, -1]  # last dim, center of avg
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws
