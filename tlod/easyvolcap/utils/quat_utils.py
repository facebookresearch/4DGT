# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Handles 4D rotation and two-sided quaternions along with the eigen decomposition of the 4D covariance matrix.
This essentially serves as the backward function of the compute_cov_4d routine
"""

import numpy as np
import torch
from torch.nn import functional as F

from .console_utils import warn_once
from .math_utils import normalize


def unitquat_to_rotmat(quat):
    r"""
    Converts unit quaternion into rotation matrix representation.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = -x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = -x2 - y2 + z2 + w2
    return matrix


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle axis to a quaternion.
    Args:
        angle_axis (Tensor): Tensor of Nx3 representing the rotations
    Returns:
        quaternions (Tensor): Tensor of Nx4 representing the quaternions
    References:
        https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py

    Equations
    qx = ax * sin(angle/2)
    qy = ay * sin(angle/2)
    qz = az * sin(angle/2)
    qw = cos(angle/2)

    where:

    the axis is normalised so: ax*ax + ay*ay + az*az = 1
    the quaternion is also normalised so cos(angle/2)2 + ax*ax * sin(angle/2)2 + ay*ay * sin(angle/2)2+ az*az * sin(angle/2)2 = 1
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)  # N, 1
    half_angle = 0.5 * angle
    eps = 1e-6
    small_angle = angle.data.abs() < eps
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)
    # for small angle, use taylor series
    sin_half_angle = torch.where(
        small_angle, half_angle - 0.5 * half_angle**3, sin_half_angle
    )
    cos_half_angle = torch.where(small_angle, 1 - 0.5 * half_angle**2, cos_half_angle)
    quaternions = torch.cat(
        [cos_half_angle, sin_half_angle * normalize(angle_axis)], dim=-1
    )
    return quaternions


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def gen_coeffs():
    keys = [
        "ap",
        "aq",
        "ar",
        "as",
        "bp",
        "bq",
        "br",
        "bs",
        "cp",
        "cq",
        "cr",
        "cs",
        "dp",
        "dq",
        "dr",
        "ds",
    ]

    M_l = np.asarray(
        [
            " a",
            "-b",
            "-c",
            "-d",
            " b",
            " a",
            "-d",
            " c",
            " c",
            " d",
            " a",
            "-b",
            " d",
            "-c",
            " b",
            " a",
        ]
    ).reshape(4, 4)
    M_r = np.asarray(
        [
            " p",
            " q",
            " r",
            " s",
            "-q",
            " p",
            "-s",
            " r",
            "-r",
            " s",
            " p",
            "-q",
            "-s",
            "-r",
            " q",
            " p",
        ]
    ).reshape(4, 4)

    all_coeffs = []

    for i in range(4):
        for j in range(4):
            # Print the coefficients for the R[i, j] entry

            coeffs = np.zeros(16)
            for k in range(4):
                key = M_l[i, k][-1] + M_r[k, j][-1]
                idx = keys.index(key)
                sign = M_l[i, k][0] == M_r[k, j][0]
                coeffs[idx] = 1 if sign else -1

            all_coeffs.append(coeffs)

    # print(np.asarray(all_coeffs))
    all_coeffs = np.stack(all_coeffs)
    return all_coeffs


def compute_normal(cov: torch.Tensor):
    cov = solve_3d_interpretation(cov)  # reassignment
    s, V = solve_scaling_rotation(cov)  # agnostic to dimentionality, N, 3; N, 3, 3
    # min_scale_index = s.min(dim=-1).indices  # N,
    # min_scale_index = min_scale_index[..., None, None].expand(*V.shape[:-1], 1)  # N, 3, 1
    # norm = multi_gather(V, min_scale_index, dim=-1)[..., 0]  # N, 3
    # NOTE: Assuming the smallest scale is the last one
    norm = V[..., -1]
    return norm


def align_normal_to_view(norm: torch.Tensor, R: torch.Tensor):
    # Sanitize input
    # if isinstance(C, torch.Tensor): C = C.to(norm, non_blocking=True)
    # else: C = torch.as_tensor(C).to(norm, non_blocking=True)
    if isinstance(R, torch.Tensor):
        R = R.to(norm, non_blocking=True)
    else:
        R = torch.as_tensor(R).to(norm, non_blocking=True)

    # Convert normal to camera view
    norm = norm @ R.mT

    # Normalize normal for display
    norm = norm * norm[..., 2:3].sign()
    norm[..., 0] *= -1
    return norm


def solve_4d_params(cov: torch.Tensor, ms: torch.Tensor, cov_t: torch.Tensor):
    cov = solve_4d_interpretation(cov, ms, cov_t)
    s, ql, qr = solve_covariance(cov)
    return s, ql, qr


def solve_3d_interpretation(cov: torch.Tensor):
    if cov.shape[-1] == 6:
        cov_11 = cov.new_zeros(cov.shape[:-1] + (3, 3))
        # inds = torch.triu_indices(3, 3, device=cov.device)  # 2, 6
        # cov_11[..., inds[0], inds[1]] = cov
        # cov = cov_11

        cov_11[..., 0, 0] = cov[..., 0]
        cov_11[..., 0, 1] = cov[..., 1]
        cov_11[..., 0, 2] = cov[..., 2]
        cov_11[..., 1, 1] = cov[..., 3]
        cov_11[..., 1, 2] = cov[..., 4]
        cov_11[..., 2, 2] = cov[..., 5]

        # cov_11[..., 0, 0] = cov[..., 0]
        cov_11[..., 1, 0] = cov[..., 1]
        cov_11[..., 2, 0] = cov[..., 2]
        # cov_11[..., 1, 1] = cov[..., 3]
        cov_11[..., 2, 1] = cov[..., 4]
        # cov_11[..., 2, 2] = cov[..., 5]
        cov = cov_11
    return cov


def solve_4d_interpretation(cov: torch.Tensor, ms: torch.Tensor, cov_t: torch.Tensor):
    cov = solve_3d_interpretation(cov)
    if ms.ndim == cov.ndim - 1:
        ms = ms[..., None]
    if cov_t.ndim == cov.ndim - 1:
        cov_t = cov_t[..., None]

    cov_11 = cov
    cov = cov_11.new_zeros(cov_11.shape[:-2] + (4, 4))
    cov_12 = ms * cov_t  # N, 3, 1
    cov_11 = cov_11 + cov_12 @ cov_12.mT / cov_t  # N, 3, 3

    cov[..., :3, :3] = cov_11
    cov[..., 3:, 3:] = cov_t
    cov[..., :3, 3:] = cov_12
    cov[..., 3:, :3] = cov_12.mT

    return cov


def solve_scaling_rotation(A: torch.Tensor):
    """
    The simple SVD decomposition result sometimes isn't a real 4D rotation matrix
    Need to manually remove the reflection from these
    """
    if not hasattr(solve_scaling_rotation, "svd"):
        try:
            from torch_batch_svd import svd

            def svd_check_cuda(A):
                if A.device.type == "cuda":
                    U, S, V = svd(A)
                    return U, S, V.mH
                else:
                    warn_once(
                        "torch_batch_svd requires cuda tensor, will use torch.linalg.svd instead"
                    )
                    return torch.linalg.svd(A)

            solve_scaling_rotation.svd = svd_check_cuda
        except ImportError as e:  # noqa: F841
            warn_once("torch_batch_svd not found, using torch.linalg.svd instead")
            solve_scaling_rotation.svd = torch.linalg.svd

    U, S, Vh = solve_scaling_rotation.svd(A)  # N, 4, 4
    # U1, S1, Vh1 = torch.linalg.svd(A)
    # breakpoint()
    s = S.sqrt()
    V = Vh.mT

    # Make sure V is a rotation matrix by computing its determinant and flip the mirroring operation manually
    det = torch.linalg.det(V)

    s_out = s.clone()
    V_out = V.clone()
    s_out[..., 0], s_out[..., 1] = (
        torch.where(det > 0, s[..., 0], s[..., 1]),
        torch.where(det > 0, s[..., 1], s[..., 0]),
    )
    V_out[..., 0], V_out[..., 1] = (
        torch.where((det > 0)[..., None], V[..., 0], V[..., 1]),
        torch.where((det > 0)[..., None], V[..., 1], V[..., 0]),
    )

    return s_out, V_out
    # L, V = torch.linalg.eigh(A)
    # s = L.sqrt()  # N, 4
    # return s, V


def solve_covariance(A: torch.Tensor):
    """
    Given a 4x4 covariance matrix, solve for its scaling and rotation part
    """
    s, V = solve_scaling_rotation(A)
    ql, qr = solve_rotation(V)
    return s, ql, qr


def solve_rotation(R: torch.Tensor):
    """
    a_{00}+a_{11}+a_{22}+a_{33} & +a_{10}-a_{01}-a_{32}+a_{23} & +a_{20}+a_{31}-a_{02}-a_{13} & +a_{30}-a_{21}+a_{12}-a_{03}
    a_{10}-a_{01}+a_{32}-a_{23} & -a_{00}-a_{11}+a_{22}+a_{33} & +a_{30}-a_{21}-a_{12}+a_{03} & -a_{20}-a_{31}-a_{02}-a_{13}
    a_{20}-a_{31}-a_{02}+a_{13} & -a_{30}-a_{21}-a_{12}-a_{03} & -a_{00}+a_{11}-a_{22}+a_{33} & +a_{10}+a_{01}-a_{32}-a_{23}
    a_{30}+a_{21}-a_{12}-a_{03} & +a_{20}-a_{31}+a_{02}-a_{13} & -a_{10}-a_{01}-a_{32}-a_{23} & -a_{00}+a_{11}+a_{22}-a_{33}

    R[0,0]+R[1,1]+R[2,2]+R[3,3] & +R[1,0]-R[0,1]-R[3,2]+R[2,3] & +R[2,0]+R[3,1]-R[0,2]-R[1,3] & +R[3,0]-R[2,1]+R[1,2]-R[0,3]
    R[1,0]-R[0,1]+R[3,2]-R[2,3] & -R[0,0]-R[1,1]+R[2,2]+R[3,3] & +R[3,0]-R[2,1]-R[1,2]+R[0,3] & -R[2,0]-R[3,1]-R[0,2]-R[1,3]
    R[2,0]-R[3,1]-R[0,2]+R[1,3] & -R[3,0]-R[2,1]-R[1,2]-R[0,3] & -R[0,0]+R[1,1]-R[2,2]+R[3,3] & +R[1,0]+R[0,1]-R[3,2]-R[2,3]
    R[3,0]+R[2,1]-R[1,2]-R[0,3] & +R[2,0]-R[3,1]+R[0,2]-R[1,3] & -R[1,0]-R[0,1]-R[3,2]-R[2,3] & -R[0,0]+R[1,1]+R[2,2]-R[3,3]

    ap**2 + aq**2 + ar**2 + sa**2 + bp**2 + bq**2 + br**2 + bs**2 + cp**2 + cq**2 + cr**2 + cs**2 + dp**2 + dq**2 + dr**2 + ds**2
    """
    ap = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] + R[..., 3, 3]) / 4  # noqa: F841
    aq = -(+R[..., 1, 0] - R[..., 0, 1] - R[..., 3, 2] + R[..., 2, 3]) / 4  # noqa: F841
    ar = -(+R[..., 2, 0] + R[..., 3, 1] - R[..., 0, 2] - R[..., 1, 3]) / 4  # noqa: F841
    sa = -(+R[..., 3, 0] - R[..., 2, 1] + R[..., 1, 2] - R[..., 0, 3]) / 4  # noqa: F841
    bp = (R[..., 1, 0] - R[..., 0, 1] + R[..., 3, 2] - R[..., 2, 3]) / 4  # noqa: F841
    bq = -(-R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2] + R[..., 3, 3]) / 4  # noqa: F841
    br = -(+R[..., 3, 0] - R[..., 2, 1] - R[..., 1, 2] + R[..., 0, 3]) / 4  # noqa: F841
    bs = -(-R[..., 2, 0] - R[..., 3, 1] - R[..., 0, 2] - R[..., 1, 3]) / 4  # noqa: F841
    cp = (R[..., 2, 0] - R[..., 3, 1] - R[..., 0, 2] + R[..., 1, 3]) / 4  # noqa: F841
    cq = -(-R[..., 3, 0] - R[..., 2, 1] - R[..., 1, 2] - R[..., 0, 3]) / 4  # noqa: F841
    cr = -(-R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2] + R[..., 3, 3]) / 4  # noqa: F841
    cs = -(+R[..., 1, 0] + R[..., 0, 1] - R[..., 3, 2] - R[..., 2, 3]) / 4  # noqa: F841
    dp = (R[..., 3, 0] + R[..., 2, 1] - R[..., 1, 2] - R[..., 0, 3]) / 4  # noqa: F841
    dq = -(+R[..., 2, 0] - R[..., 3, 1] + R[..., 0, 2] - R[..., 1, 3]) / 4  # noqa: F841
    dr = -(-R[..., 1, 0] - R[..., 0, 1] - R[..., 3, 2] - R[..., 2, 3]) / 4  # noqa: F841
    ds = -(-R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - R[..., 3, 3]) / 4  # noqa: F841

    # M = torch.stack(
    #     [
    #         ap, aq, ar, sa,
    #         bp, bq, br, bs,
    #         cp, cq, cr, cs,
    #         dp, dq, dr, ds,
    #     ], dim=-1
    # ).view(*R.shape[:-2], 4, 4)

    # torch.linalg.matrix_rank(M)

    # breakpoint()

    p, q, r, s = ap, aq, ar, sa
    a, b, c, d = torch.ones_like(p), bp / p, cp / p, dp / p

    ql = torch.stack([a, b, c, d], dim=-1)
    qr = torch.stack([p, q, r, s], dim=-1)

    ql = F.normalize(ql, dim=-1)
    qr = F.normalize(qr, dim=-1)

    return ql, qr


def solve_rotation_lstsq(R: torch.Tensor):
    """
    Given a 4D rotation matrix, we aim to get back to its original 2-quaternion form
    1. Solve for ap, aq, ar, as... totalling at 16 parameters with 16 equations, all linear
    2. Solve for a, b, c, d, p, q, r, s with the 16 quadratic equation, substituting a with 1 and normalize later
    """

    if not hasattr(solve_rotation, "L"):
        # ap, aq, ar, as, bp, bq, br, bs, cp, cq, cr, cs, dp, dq, dr, ds
        a, b, c, d = 1, 1, 1, 1
        # fmt: off
        A = \
            np.asarray([[ 1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1],
                        [ 0,  1,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  1,  0],
                        [ 0,  0,  1,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0, -1,  0,  0],
                        [ 0,  0,  0,  1,  0,  0, -1,  0,  0,  1,  0,  0, -1,  0,  0,  0],
                        [ 0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  1,  0],
                        [ 1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1],
                        [ 0,  0,  0, -1,  0,  0,  1,  0,  0,  1,  0,  0, -1,  0,  0,  0],
                        [ 0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0],
                        [ 0,  0, -1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0, -1,  0,  0],
                        [ 0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0],
                        [ 1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1],
                        [ 0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0],
                        [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  1,  0,  0,  1,  0,  0,  0],
                        [ 0,  0, -1,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  0,  0],
                        [ 0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0],
                        [ 1,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  1]])
        # fmt: on
        At = A.T
        solve_rotation.L = np.linalg.inv(At @ A) @ At  # 16, 16
        solve_rotation.A = A

    b = torch.stack(
        [
            R[..., 0, 0],
            R[..., 0, 1],
            R[..., 0, 2],
            R[..., 0, 3],
            R[..., 1, 0],
            R[..., 1, 1],
            R[..., 1, 2],
            R[..., 1, 3],
            R[..., 2, 0],
            R[..., 2, 1],
            R[..., 2, 2],
            R[..., 2, 3],
            R[..., 3, 0],
            R[..., 3, 1],
            R[..., 3, 2],
            R[..., 3, 3],
        ],
        dim=-1,
    )  # N, 16

    L = torch.as_tensor(solve_rotation.L).to(R, non_blocking=True)  # 16, 16
    x = b @ L.mT  # N, 16

    # A = torch.as_tensor(solve_rotation.A).to(R, non_blocking=True)  # 16, 16
    # for s in R.shape[:-2]:
    #     A = A[None].expand(s, *A.shape)
    # x = torch.linalg.lstsq(A, b).solution  # N, 16

    # ap, aq, ar, ass, bq, bp, bs, br, cr, cs, cp, cq, ds, dr, dq, dp = x.unbind(-1)
    ap, aq, ar, sa, bp, bq, br, bs, cp, cq, cr, cs, dp, dq, dr, ds = x.unbind(-1)

    # FIXME: Only half of the results are correct: when values of the quat > 0
    p, q, r, s = ap, aq, ar, sa
    a, b, c, d = torch.ones_like(p), bp / p, cp / p, dp / p
    # breakpoint()

    # p, q, r, s = torch.where(ap > 0, ap, -ap), torch.where(ap > 0, aq, -aq), torch.where(ap > 0, ar, -ar), torch.where(ap > 0, ass, -ass)
    # a, b, c, d = torch.where(ap > 0, torch.ones_like(p), -torch.ones_like(p)), bp / p, cp / p, dp / p
    # breakpoint()

    ql = torch.stack([a, b, c, d], dim=-1)
    qr = torch.stack([p, q, r, s], dim=-1)

    ql = F.normalize(ql, dim=-1)
    qr = F.normalize(qr, dim=-1)

    return ql, qr
