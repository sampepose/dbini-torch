import time
from typing import Optional, Tuple

import torch
import torch.sparse as sparse
from tqdm import tqdm

from .solver import conjugate_gradient_solver


def move_mask(mask: torch.Tensor, direction: str) -> torch.Tensor:
    """
    PyTorch version of move_mask that exactly matches NumPy implementation.
    Args:
        mask: [H,W] tensor in HW format
        direction: one of "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2D tensor [H,W], got shape {mask.shape}")

    H, W = mask.shape
    if direction == "left":
        result = torch.zeros_like(mask)
        result[:, :-1] = mask[:, 1:]
    elif direction == "right":
        result = torch.zeros_like(mask)
        result[:, 1:] = mask[:, :-1]
    elif direction == "top":
        result = torch.zeros_like(mask)
        result[:-1, :] = mask[1:, :]
    elif direction == "bottom":
        result = torch.zeros_like(mask)
        result[1:, :] = mask[:-1, :]
    elif direction == "top_left":
        result = torch.zeros_like(mask)
        result[:-1, :-1] = mask[1:, 1:]
    elif direction == "top_right":
        result = torch.zeros_like(mask)
        result[:-1, 1:] = mask[1:, :-1]
    elif direction == "bottom_left":
        result = torch.zeros_like(mask)
        result[1:, :-1] = mask[:-1, 1:]
    elif direction == "bottom_right":
        result = torch.zeros_like(mask)
        result[1:, 1:] = mask[:-1, :-1]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return result


def create_sparse_derivative(
    nz: torch.Tensor,
    neighbor_mask: torch.Tensor,
    pixel_idx: torch.Tensor,
    mask: torch.Tensor,
    num_pixel: int,
    step_size: float,
    device: torch.device,
) -> torch.sparse.Tensor:
    """Creates sparse derivative matrix.
    Args:
        nz: [N] tensor of z components
        neighbor_mask: [H,W] tensor
        pixel_idx: [H,W] tensor of indices
        mask: [H,W] tensor
    """
    # Get flattened indices where mask is True
    flat_indices = torch.where(neighbor_mask)
    y_indices = flat_indices[0]
    x_indices = flat_indices[1]

    # Sort by pixel index
    pixel_values = pixel_idx[neighbor_mask]
    sorted_order = torch.argsort(pixel_values)
    y_indices = y_indices[sorted_order]
    x_indices = x_indices[sorted_order]
    nz_sorted = nz[sorted_order]

    valid_pairs = []
    pair_values = []

    # For each pixel
    for i in range(len(pixel_values)):
        current_idx = pixel_values[sorted_order[i]].item()
        current_nz = nz_sorted[i].item()

        # First add self connection
        valid_pairs.append((current_idx, current_idx))
        pair_values.append(current_nz / step_size)

        # Then find lowest available index
        lowest_idx = None
        for idx in range(num_pixel):
            if idx < current_idx and idx in pixel_values:
                lowest_idx = idx
                break

        if lowest_idx is not None:
            valid_pairs.append((current_idx, lowest_idx))
            pair_values.append(-current_nz / step_size)

    # Convert to tensor format
    indices = torch.tensor(valid_pairs, device=device).t()
    values = torch.tensor(pair_values, device=device)

    return torch.sparse_coo_tensor(
        indices=indices, values=values, size=(num_pixel, num_pixel), device=device
    ).coalesce()


def generate_dx_dy(
    mask: torch.Tensor,
    nz_horizontal: torch.Tensor,
    nz_vertical: torch.Tensor,
    step_size: float = 1,
):
    """
    Generate sparse derivative matrices for bilateral normal integration.

    Args:
        mask: [H,W] tensor in CHW format
        nz_horizontal: [N] tensor of horizontal components
        nz_vertical: [N] tensor of vertical components
        step_size: Pixel size in world coordinates (default: 1)

    Returns:
        Tuple of sparse CSR matrices (D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg)
    """
    # Input validation
    if mask.dim() != 2:
        raise ValueError(f"mask must be 3D tensor [H,W], got shape {mask.shape}")

    num_pixel = mask.sum().item()
    device = mask.device

    # Verify nz arrays match number of True values
    if len(nz_horizontal) != num_pixel:
        raise ValueError(
            f"nz_horizontal length {len(nz_horizontal)} doesn't match number of True pixels {num_pixel}"
        )
    if len(nz_vertical) != num_pixel:
        raise ValueError(
            f"nz_vertical length {len(nz_vertical)} doesn't match number of True pixels {num_pixel}"
        )

    # Generate pixel indices
    pixel_idx = torch.zeros_like(mask, dtype=torch.long)
    pixel_idx[mask] = torch.arange(num_pixel, device=device)

    # Create neighbor masks
    has_left = torch.logical_and(move_mask(mask, "right"), mask)
    has_right = torch.logical_and(move_mask(mask, "left"), mask)
    has_bottom = torch.logical_and(move_mask(mask, "top"), mask)
    has_top = torch.logical_and(move_mask(mask, "bottom"), mask)

    # Extract components for valid neighbors
    nz_left = nz_horizontal[has_left[mask]]
    nz_right = nz_horizontal[has_right[mask]]
    nz_top = nz_vertical[has_top[mask]]
    nz_bottom = nz_vertical[has_bottom[mask]]

    def create_derivative_matrix(
        nz_values: torch.Tensor,
        has_mask: torch.Tensor,
        move_fn,
        reverse_order: bool = False,
    ) -> torch.sparse.Tensor:
        """Helper to create derivative matrix in CSR format"""
        data = torch.stack([-nz_values / step_size, nz_values / step_size], -1).flatten()

        if reverse_order:
            indices = torch.stack(
                [pixel_idx[move_fn(has_mask)].flatten(), pixel_idx[has_mask].flatten()],
                -1,
            ).flatten()
        else:
            indices = torch.stack(
                [pixel_idx[has_mask].flatten(), pixel_idx[move_fn(has_mask)].flatten()],
                -1,
            ).flatten()

        indptr = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(has_mask[mask].to(torch.long) * 2, dim=0),
            ]
        )

        return torch.sparse_csr_tensor(
            crow_indices=indptr,
            col_indices=indices,
            values=data,
            size=(num_pixel, num_pixel),
            device=device,
        )

    # Create all four matrices
    D_horizontal_neg = create_derivative_matrix(
        nz_left, has_left, lambda x: move_mask(x, "left"), reverse_order=True
    )
    D_horizontal_pos = create_derivative_matrix(
        nz_right, has_right, lambda x: move_mask(x, "right")
    )
    D_vertical_pos = create_derivative_matrix(nz_top, has_top, lambda x: move_mask(x, "top"))
    D_vertical_neg = create_derivative_matrix(
        nz_bottom, has_bottom, lambda x: move_mask(x, "bottom"), reverse_order=True
    )

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask: torch.Tensor) -> torch.Tensor:
    """
    Constructs quadrilateral facets from a binary mask for mesh generation.

    Each facet is defined by four vertices in counter-clockwise order:
        v1 -- v4
        |     |
        v2 -- v3

    The function identifies valid facets where all four corners are True in the mask.
    Each facet is represented as [4, idx_v1, idx_v2, idx_v3, idx_v4] where:
    - 4 indicates it's a quadrilateral
    - idx_vN is the flattened index of vertex N in the point cloud

    Args:
        mask: Binary mask [H,W] indicating valid vertices

    Returns:
        facets: Tensor of shape [N,5] containing N valid facets
               Each row is [4, top_left, bottom_left, bottom_right, top_right]
    """
    device = mask.device
    idx = torch.zeros_like(mask, dtype=torch.long, device=device)
    idx[mask] = torch.arange(mask.sum().item(), device=device)

    # Create facet vertex masks [H,W]
    facet_top_left = (
        move_mask(mask, "top") & move_mask(mask, "left") & move_mask(mask, "top_left") & mask
    )

    facet_top_right = move_mask(facet_top_left, "right")
    facet_bottom_left = move_mask(facet_top_left, "bottom")
    facet_bottom_right = move_mask(facet_top_left, "bottom_right")

    # Create facets [N,5]
    num_facets = facet_top_left.sum().item()
    facets = torch.zeros((num_facets, 5), dtype=torch.long, device=device)
    facets[:, 0] = 4  # Number of vertices per facet
    facets[:, 1] = idx[facet_top_left]
    facets[:, 2] = idx[facet_bottom_left]
    facets[:, 3] = idx[facet_bottom_right]
    facets[:, 4] = idx[facet_top_right]

    return facets


def map_depth_map_to_point_clouds(
    depth_map: torch.Tensor,
    mask: torch.Tensor,
    K: Optional[torch.Tensor] = None,
    step_size: float = 1,
) -> torch.Tensor:
    """
    Maps depth values to 3D point coordinates in either orthographic or perspective projection.

    Coordinate system:
        y
        |  z
        | /
        |/
        o ---x

    Args:
        depth_map: Depth values [H,W]
        mask: Binary mask indicating valid depth values [H,W]
        K: Optional camera intrinsic matrix for perspective projection [3,3]
        step_size: Pixel size in world coordinates for orthographic projection (default: 1)
                  Only used when K is None

    Returns:
        vertices: 3D point coordinates [N,3] where N is number of True values in mask
    """
    device = depth_map.device
    H, W = mask.shape

    # Create coordinate grid
    xx = torch.arange(H - 1, -1, -1, device=device)[:, None].expand(H, W)
    yy = torch.arange(W, device=device)[None, :].expand(H, W)

    if K is None:
        vertices = torch.zeros((3, H, W), device=device)  # [3,H,W]
        vertices[0] = xx * step_size  # X coordinates
        vertices[1] = yy * step_size  # Y coordinates
        vertices[2] = depth_map  # Z coordinates
        vertices = vertices.permute(1, 2, 0)[mask]  # [H,W,3] then [N,3]
    else:
        u = torch.zeros((3, H, W), device=device, dtype=depth_map.dtype)  # [3,H,W]
        u[0] = xx  # X pixel coordinates
        u[1] = yy  # Y pixel coordinates
        u[2] = 1  # Homogeneous coordinates
        u = u.reshape(3, -1)  # [3,H*W]
        u = u[:, mask.reshape(-1)]  # [3,N]
        K_inv = torch.linalg.inv_ex(K.type(depth_map.dtype))[0].to(device)
        vertices = (K_inv @ u).t() * depth_map[mask].unsqueeze(-1)  # [N,3]

    return vertices


def bilateral_normal_integration(
    normal_map: torch.Tensor,
    normal_mask: torch.Tensor,
    k: float = 2,
    depth_map: Optional[torch.Tensor] = None,
    depth_mask: Optional[torch.Tensor] = None,
    lambda1: float = 0,
    K: Optional[torch.Tensor] = None,
    step_size: float = 1,
    max_iter: int = 150,
    tol: float = 1e-4,
    cg_max_iter: int = 5000,
    cg_rtol: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs bilateral normal integration to reconstruct a depth map from surface normals.

    Args:
        normal_map: Normal vectors in CHW format [3,H,W]
        normal_mask: Binary mask indicating valid normal vectors [H,W]
        k: Bilateral weight parameter (default: 2)
        depth_map: Optional initial depth values [H,W]
        depth_mask: Optional mask for depth values [H,W]
        lambda1: Weight for depth regularization (default: 0)
        K: Optional camera intrinsic matrix for perspective projection
        step_size: Pixel size in world coordinates (default: 1)
        max_iter: Maximum number of iterations (default: 150)
        tol: Convergence tolerance for relative energy (default: 1e-4)
        cg_max_iter: Maximum conjugate gradient iterations (default: 5000)
        cg_rtol: Relative tolerance for conjugate gradient (default: 1e-3)

    Returns:
        Tuple containing:
        - depth_map: Reconstructed depth values [H,W]
        - surface: Tuple of (vertices [N,3], faces [M,5])
        - wu_map: Horizontal bilateral weights [H,W]
        - wv_map: Vertical bilateral weights [H,W]
        - energy_list: List of energy values per iteration
    """
    device = normal_map.device
    if device.type == "cpu" and not torch.backends.mkl.is_available():
        raise RuntimeError(
            "Bilateral normal integration requires either CUDA or MKL support for efficient "
            "sparse matrix operations. Your PyTorch installation doesn't have MKL support "
            "and you're running on CPU. Please either:\n"
            "1. Install PyTorch with MKL support\n"
            "2. Use a CUDA-capable GPU\n"
        )

    # Normals input validation
    if normal_map.dim() != 3:
        raise ValueError(f"normal_map must be 3D tensor [3,H,W], got shape {normal_map.shape}")
    if normal_map.shape[0] != 3:
        raise ValueError(
            f"normal_map must have 3 channels first (CHW format), got {normal_map.shape}"
        )
    if normal_mask.dim() != 2:
        raise ValueError(f"normal_mask must be 2D tensor [H,W], got shape {normal_mask.shape}")

    # Mask input validation
    H, W = normal_mask.shape
    if normal_map.shape[1:] != (H, W):
        raise ValueError(
            f"normal_map spatial dimensions {normal_map.shape[1:]} "
            f"must match normal_mask dimensions {normal_mask.shape}"
        )
    if not normal_mask.dtype == torch.bool:
        raise ValueError(f"normal_mask must be boolean tensor, got {normal_mask.dtype}")

    num_normals = normal_mask.sum().item()
    device = normal_map.device

    projection = "orthographic" if K is None else "perspective"
    print(
        f"Running bilateral normal integration with k={k} in the {projection} case. \n"
        f"The number of normal vectors is {num_normals}."
    )

    # Extract normal components from CHW format
    nx = normal_map[1][normal_mask]  # x-channel
    ny = normal_map[0][normal_mask]  # y-channel
    nz = -normal_map[2][normal_mask]  # z-channel

    if K is not None:
        # Create coordinate grid
        H, W = normal_mask.shape
        xx = torch.arange(H - 1, -1, -1, device=device)[:, None].expand(H, W)
        yy = torch.arange(W, device=device)[None, :].expand(H, W)

        cx, cy = K[0, 2].item(), K[1, 2].item()
        fx, fy = K[0, 0].item(), K[1, 1].item()

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
    else:
        nz_u = nz.clone()
        nz_v = nz.clone()

    # Generate sparse matrices
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_v, nz_u, step_size)

    # Logic copied directly from original cupy implementation
    pixel_idx = torch.zeros_like(normal_mask, dtype=int, device=normal_mask.device)
    pixel_idx[normal_mask] = torch.arange(num_normals, device=normal_mask.device)
    pixel_idx_flat = torch.arange(num_normals, device=normal_mask.device)
    pixel_idx_flat_indptr = torch.arange(num_normals + 1, device=normal_mask.device)

    has_left_mask = torch.logical_and(move_mask(normal_mask, "right"), normal_mask)
    has_left_mask_left = move_mask(has_left_mask, "left")
    has_right_mask = torch.logical_and(move_mask(normal_mask, "left"), normal_mask)
    has_right_mask_right = move_mask(has_right_mask, "right")
    has_bottom_mask = torch.logical_and(move_mask(normal_mask, "top"), normal_mask)
    has_bottom_mask_bottom = move_mask(has_bottom_mask, "bottom")
    has_top_mask = torch.logical_and(move_mask(normal_mask, "bottom"), normal_mask)
    has_top_mask_top = move_mask(has_top_mask, "top")

    has_left_mask_flat = has_left_mask[normal_mask]
    has_right_mask_flat = has_right_mask[normal_mask]
    has_bottom_mask_flat = has_bottom_mask[normal_mask]
    has_top_mask_flat = has_top_mask[normal_mask]

    has_left_mask_left_flat = has_left_mask_left[normal_mask]
    has_right_mask_right_flat = has_right_mask_right[normal_mask]
    has_bottom_mask_bottom_flat = has_bottom_mask_bottom[normal_mask]
    has_top_mask_top_flat = has_top_mask_top[normal_mask]

    nz_left_square = nz_v[has_left_mask_flat] ** 2
    nz_right_square = nz_v[has_right_mask_flat] ** 2
    nz_top_square = nz_u[has_top_mask_flat] ** 2
    nz_bottom_square = nz_u[has_bottom_mask_flat] ** 2

    pixel_idx_left_center = pixel_idx[has_left_mask]
    pixel_idx_right_right = pixel_idx[has_right_mask_right]
    pixel_idx_top_center = pixel_idx[has_top_mask]
    pixel_idx_bottom_bottom = pixel_idx[has_bottom_mask_bottom]

    pixel_idx_left_left_indptr = torch.concatenate(
        [
            torch.tensor([0], device=normal_map.device),
            torch.cumsum(has_left_mask_left_flat, 0),
        ]
    )
    pixel_idx_right_center_indptr = torch.concatenate(
        [
            torch.tensor([0], device=normal_map.device),
            torch.cumsum(has_right_mask_flat, 0),
        ]
    )
    pixel_idx_top_top_indptr = torch.concatenate(
        [
            torch.tensor([0], device=normal_map.device),
            torch.cumsum(has_top_mask_top_flat, 0),
        ]
    )
    pixel_idx_bottom_center_indptr = torch.concatenate(
        [
            torch.tensor([0], device=normal_map.device),
            torch.cumsum(has_bottom_mask_flat, 0),
        ]
    )

    wu = 0.5 * torch.ones(num_normals, dtype=normal_map.dtype, device=normal_mask.device)
    wv = 0.5 * torch.ones(num_normals, dtype=normal_map.dtype, device=normal_mask.device)
    z = torch.zeros(num_normals, dtype=normal_map.dtype, device=normal_mask.device)
    energy = (
        torch.sum(wu * (torch.matmul(A1, z) + nx) ** 2)
        + torch.sum((1 - wu) * (torch.matmul(A2, z) + nx) ** 2)
        + torch.sum(wv * (torch.matmul(A3, z) + ny) ** 2)
        + torch.sum((1 - wv) * (torch.matmul(A4, z) + ny) ** 2)
    )

    if depth_map is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)
        z_prior = torch.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]
        z_prior[~depth_mask_flat] = 0
    # /copied logic

    tic = time.time()

    energy_list = []
    pbar = tqdm(range(max_iter))
    for i in pbar:
        data_term_top = wu[has_top_mask_flat] * nz_top_square
        data_term_bottom = (1 - wu[has_bottom_mask_flat]) * nz_bottom_square
        data_term_left = (1 - wv[has_left_mask_flat]) * nz_left_square
        data_term_right = wv[has_right_mask_flat] * nz_right_square

        diagonal_data_term = torch.zeros(
            num_normals, device=normal_mask.device, dtype=normal_map.dtype
        )
        diagonal_data_term[has_left_mask_flat] += data_term_left
        diagonal_data_term[has_left_mask_left_flat] += data_term_left
        diagonal_data_term[has_right_mask_flat] += data_term_right
        diagonal_data_term[has_right_mask_right_flat] += data_term_right
        diagonal_data_term[has_top_mask_flat] += data_term_top
        diagonal_data_term[has_top_mask_top_flat] += data_term_top
        diagonal_data_term[has_bottom_mask_flat] += data_term_bottom
        diagonal_data_term[has_bottom_mask_bottom_flat] += data_term_bottom
        if depth_map is not None:
            diagonal_data_term[depth_mask_flat] += lambda1

        A_mat_d = torch.sparse_csr_tensor(
            crow_indices=pixel_idx_flat_indptr,
            col_indices=pixel_idx_flat,
            values=diagonal_data_term,
            size=(num_normals, num_normals),
        )
        A_mat_left_odu = torch.sparse_csr_tensor(
            crow_indices=pixel_idx_left_left_indptr,
            col_indices=pixel_idx_left_center,
            values=-data_term_left,
            size=(num_normals, num_normals),
        )
        A_mat_right_odu = torch.sparse_csr_tensor(
            crow_indices=pixel_idx_right_center_indptr,
            col_indices=pixel_idx_right_right,
            values=-data_term_right,
            size=(num_normals, num_normals),
        )
        A_mat_top_odu = torch.sparse_csr_tensor(
            crow_indices=pixel_idx_top_top_indptr,
            col_indices=pixel_idx_top_center,
            values=-data_term_top,
            size=(num_normals, num_normals),
        )
        A_mat_bottom_odu = torch.sparse_csr_tensor(
            crow_indices=pixel_idx_bottom_center_indptr,
            col_indices=pixel_idx_bottom_bottom,
            values=-data_term_bottom,
            size=(num_normals, num_normals),
        )
        A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu
        A_mat = A_mat_d + A_mat_odu
        A_mat = (
            A_mat + A_mat_odu.t().to_sparse_csr()
        )  # diagnol + upper triangle + lower triangle matrix

        D = (
            # TODO: This is a bottleneck, see https://github.com/pytorch/pytorch/issues/104981
            sparse.spdiags(
                1 / torch.clamp(diagonal_data_term, min=1e-5).cpu(),
                torch.tensor([0], device="cpu"),
                (num_normals, num_normals),
            )
            .to(normal_mask.device)
            .coalesce()
        )
        b_vec = (
            A1.t() @ (wu * (-nx))
            + A2.t() @ ((1 - wu) * (-nx))
            + A3.t() @ (wv * (-ny))
            + A4.t() @ ((1 - wv) * (-ny))
        )

        if depth_map is not None:
            b_vec += lambda1 * z_prior
            offset = torch.mean((z_prior - z)[depth_mask_flat])
            z = z + offset

        z, _ = conjugate_gradient_solver(A_mat, b_vec, cg_max_iter, z, D, rtol=cg_rtol)
        del A_mat, b_vec, wu, wv

        # Update weights
        wu = torch.sigmoid(k * ((A2 @ z) ** 2 - (A1 @ z) ** 2))
        wv = torch.sigmoid(k * ((A4 @ z) ** 2 - (A3 @ z) ** 2))
        W = (
            # TODO: This is a bottleneck, see https://github.com/pytorch/pytorch/issues/104981
            sparse.spdiags(
                torch.cat((wu, 1 - wu, wv, 1 - wv)).cpu(),
                torch.tensor([0], device="cpu"),
                (4 * num_normals, 4 * num_normals),
            )
            .to(normal_mask.device)
            .coalesce()
        )

        energy_old = energy
        energy = (
            torch.sum(wu * (torch.matmul(A1, z) + nx) ** 2)
            + torch.sum((1 - wu) * (torch.matmul(A2, z) + nx) ** 2)
            + torch.sum(wv * (torch.matmul(A3, z) + ny) ** 2)
            + torch.sum((1 - wv) * (torch.matmul(A4, z) + ny) ** 2)
        )
        energy_list.append(energy)

        relative_energy = torch.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}"
        )
        if relative_energy < tol:
            break
    del A1, A2, A3, A4, nx, ny

    toc = time.time()
    print(f"Total time: {toc - tic} seconds")

    # Reconstruct outputs in [H,W] format
    depth_map_out = torch.full_like(normal_mask, float("nan"), dtype=normal_map.dtype)
    depth_map_out[normal_mask] = z

    if K is not None:
        depth_map_out = torch.exp(depth_map_out)
        vertices = map_depth_map_to_point_clouds(depth_map_out, normal_mask, K=K)
    else:
        vertices = map_depth_map_to_point_clouds(
            depth_map_out, normal_mask, K=None, step_size=step_size
        )

    facets = construct_facets_from(normal_mask)
    if nz.mean() > 0:
        facets = facets[:, [0, 1, 4, 3, 2]]

    # Weight maps in [H,W] format
    wu_map = torch.full_like(normal_mask, float("nan"), dtype=normal_map.dtype)
    wu_map[normal_mask] = wv

    wv_map = torch.full_like(normal_mask, float("nan"), dtype=normal_map.dtype)
    wv_map[normal_mask] = wu

    return depth_map_out, (vertices, facets), wu_map, wv_map, energy_list
