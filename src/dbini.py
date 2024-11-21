from typing import Dict, Optional, Tuple

import torch
import torch.sparse as sparse
from torch.sparse import spdiags, vstack

from .solver import conjugate_gradient_solver


class CUDAStreams:
    """
    Context manager for handling CUDA streams and synchronization.
    Manages stream creation, event recording, and synchronization automatically.

    Usage:
        with CUDAStreams() as streams:
            with streams.stream1():
                # computations on stream1
            with streams.stream2():
                # parallel computations on stream2
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CUDAStreams, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.s1 = torch.cuda.Stream()
            self.s2 = torch.cuda.Stream()
            self.events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
            self._initialized = True

    def __enter__(self):
        # Record the current stream to restore it later
        self.previous_stream = torch.cuda.current_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Wait for both streams to complete
        if exc_type is None:  # Only synchronize if no exception occurred
            self.events[0].synchronize()
            self.events[1].synchronize()

        # Switch back to the previous stream
        torch.cuda.current_stream().wait_stream(self.s1)
        torch.cuda.current_stream().wait_stream(self.s2)
        self.previous_stream.synchronize()

    @contextmanager
    def stream1(self):
        """Context manager for stream1"""
        try:
            torch.cuda.stream(self.s1).__enter__()
            yield self.s1
        finally:
            torch.cuda.stream(self.s1).__exit__(None, None, None)
            self.events[0].record()

    @contextmanager
    def stream2(self):
        """Context manager for stream2"""
        try:
            torch.cuda.stream(self.s2).__enter__()
            yield self.s2
        finally:
            torch.cuda.stream(self.s2).__exit__(None, None, None)
            self.events[1].record()


def get_device_buffers(
    num_normals: int, device: torch.device, depth_map: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Initializes pre-allocated buffers for optimization computations.

    Args:
        num_normals: Number of normal vectors
        device: Target device for tensors
        depth_map: Optional depth map tensor for depth-based computations

    Returns:
        Dictionary containing pre-allocated tensors
    """
    # Pin memory if using CPU tensors for faster GPU transfer
    pin_memory = device.type == "cuda"

    buffers = {
        "wu": torch.empty(num_normals, device=device, pin_memory=pin_memory),
        "wv": torch.empty(num_normals, device=device, pin_memory=pin_memory),
        "W_values": torch.empty(4 * num_normals, device=device, pin_memory=pin_memory),
        "temp_mul1": torch.empty(num_normals, device=device, pin_memory=pin_memory),
        "temp_mul2": torch.empty(num_normals, device=device, pin_memory=pin_memory),
        "p1": torch.empty(num_normals, device=device, pin_memory=pin_memory),
        "p2": torch.empty(num_normals, device=device, pin_memory=pin_memory),
    }

    if depth_map is not None:
        buffers["depth_diff"] = torch.empty(num_normals, device=device, pin_memory=pin_memory)

    return buffers


@torch.jit.script
def compute_weights_component(
    z: torch.Tensor,
    A1: torch.sparse.Tensor,
    A2: torch.sparse.Tensor,
    buffers: Dict[str, torch.Tensor],
    buf_key: str,
    k: float,
) -> None:
    """
    Computes weights for a single component (wu or wv).

    Args:
        z: Input depth values
        A1, A2: Sparse derivative matrices
        buffers: Pre-allocated tensor buffers
        buf_key: Key for storing result in buffers
        k: Stiffness parameter
        device_type: 'cuda' or 'cpu'
    """
    # Non-blocking matrix multiplications
    buffers["p1"].copy_(torch.sparse.mm(A1, z.reshape(-1, 1)).squeeze(1))
    buffers["p2"].copy_(torch.sparse.mm(A2, z.reshape(-1, 1)).squeeze(1))

    # Fused operations for better performance
    torch.addcmul(
        torch.zeros_like(buffers[buf_key]),
        buffers["p2"],
        buffers["p2"],
        value=1.0,
        out=buffers[buf_key],
    )
    torch.addcmul_(buffers[buf_key], buffers["p1"], buffers["p1"], value=-1.0)

    # Apply stiffness and sigmoid
    buffers[buf_key].mul_(k)
    torch.sigmoid_(buffers[buf_key])


def compute_weights(
    z: torch.Tensor,
    A1: torch.sparse.Tensor,
    A2: torch.sparse.Tensor,
    A3: torch.sparse.Tensor,
    A4: torch.sparse.Tensor,
    buffers: Dict[str, torch.Tensor],
    k: float,
) -> torch.sparse.Tensor:
    """
    Computes weights for both components in parallel on GPU.

    Args:
        z: Input depth values
        A1-A4: Sparse derivative matrices
        buffers: Pre-allocated tensor buffers
        k: Stiffness parameter

    Returns:
        Sparse diagonal weight matrix
    """
    if z.device.type == "cuda":
        with CUDAStreams() as streams:
            with streams.stream1():
                compute_weights_component(z, A1, A2, buffers, "wu", k)
            with streams.stream2():
                compute_weights_component(z, A3, A4, buffers, "wv", k)
    else:
        compute_weights_component(z, A1, A2, buffers, "wu", k)
        compute_weights_component(z, A3, A4, buffers, "wv", k)

    # Concatenate weights efficiently
    torch.cat(
        [buffers["wu"], 1 - buffers["wu"], buffers["wv"], 1 - buffers["wv"]],
        out=buffers["W_values"],
    )

    return sparse.spdiags(buffers["W_values"], 0, 4 * len(z), 4 * len(z))


@torch.jit.script
def generate_mask_indices(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """JIT-compiled mask index generation for better performance"""
    H, W = mask.shape
    device = mask.device

    # Generate base indices
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )

    # Create pixel index map
    pixel_idx = torch.zeros_like(mask, dtype=torch.long)
    pixel_idx[mask] = torch.arange(mask.sum().item(), device=device)

    return {"xx": xx, "yy": yy, "pixel_idx": pixel_idx}


def move_mask(mask: torch.Tensor, direction: str) -> torch.Tensor:
    """
    Shifts a binary mask in the specified direction.
    Assumes mask is [H,W] in CHW format.
    """
    valid_directions = {
        "left",
        "right",
        "top",
        "bottom",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }
    if direction not in valid_directions:
        raise ValueError(f"Unknown direction: {direction}. Must be one of {valid_directions}")

    H, W = mask.shape
    result = torch.zeros_like(mask)

    # Moving in height dimension (H)
    if direction == "top":
        result[:-1, :] = mask[1:, :]
    elif direction == "bottom":
        result[1:, :] = mask[:-1, :]
    # Moving in width dimension (W)
    elif direction == "left":
        result[:, :-1] = mask[:, 1:]
    elif direction == "right":
        result[:, 1:] = mask[:, :-1]
    # Diagonal movements
    elif direction == "top_left":
        result[:-1, :-1] = mask[1:, 1:]
    elif direction == "top_right":
        result[:-1, 1:] = mask[1:, :-1]
    elif direction == "bottom_left":
        result[1:, :-1] = mask[:-1, 1:]
    elif direction == "bottom_right":
        result[1:, 1:] = mask[:-1, :-1]

    return result


@torch.jit.script
def map_depth_map_to_point_clouds(
    depth_map: torch.Tensor,
    mask: torch.Tensor,
    K: Optional[torch.Tensor] = None,
    step_size: float = 1,
) -> torch.Tensor:
    """
    Maps depth values to 3D point coordinates.
    Assumes inputs:
        depth_map: [H,W]
        mask: [H,W]
    Returns:
        vertices: [N,3] where N is number of valid points
    """
    device = depth_map.device
    H, W = mask.shape

    # Create coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),  # H dimension first
        torch.arange(W, device=device),  # W dimension second
        indexing="ij",  # Important: use 'ij' for matrix indexing
    )

    if K is None:
        vertices = torch.zeros((3, H, W), device=device)  # [3,H,W]
        vertices[0] = xx * step_size  # X coordinates
        vertices[1] = yy * step_size  # Y coordinates
        vertices[2] = depth_map  # Z coordinates
        vertices = vertices[:, mask].t()  # [N,3]
    else:
        u = torch.zeros((3, H, W), device=device)  # [3,H,W]
        u[0] = xx  # X pixel coordinates
        u[1] = yy  # Y pixel coordinates
        u[2] = 1  # Homogeneous coordinates
        u = u[:, mask]  # [3,N]
        K_inv = torch.inverse(K).to(device)
        vertices = (K_inv @ u).t() * depth_map[mask, None]  # [N,3]

    return vertices


def generate_dx_dy(
    mask: torch.Tensor, nz_horizontal: torch.Tensor, nz_vertical: torch.Tensor, step_size: float = 1
) -> Tuple[torch.sparse.Tensor, torch.sparse.Tensor, torch.sparse.Tensor, torch.sparse.Tensor]:
    """
    Generates sparse derivative matrices.
    Input formats:
        mask: [H,W] binary mask
        nz_horizontal, nz_vertical: [N] flattened normal components where N is number of valid pixels
    """
    device = mask.device
    num_pixel = mask.sum().item()

    # Generate indices using JIT
    indices = generate_mask_indices(mask)
    pixel_idx = indices["pixel_idx"]

    # Generate neighbor masks efficiently
    directions = ["right", "left", "bottom", "top"]
    neighbor_masks = {
        direction: torch.logical_and(move_mask(mask, direction), mask) for direction in directions
    }

    # Extract normal components
    nz_comps = {
        "left": nz_horizontal[neighbor_masks["left"][mask]],
        "right": nz_horizontal[neighbor_masks["right"][mask]],
        "top": nz_vertical[neighbor_masks["top"][mask]],
        "bottom": nz_vertical[neighbor_masks["bottom"][mask]],
    }

    def create_sparse_derivative(
        values: torch.Tensor, indices: torch.Tensor
    ) -> torch.sparse.Tensor:
        return (
            torch.sparse_coo_tensor(
                indices=indices, values=values, size=(num_pixel, num_pixel), device=device
            )
            .coalesce()
            .to_sparse_csr()
        )

    # Create sparse matrices efficiently
    matrices = []
    for direction, (mask_key, nz) in zip(
        ["left", "right", "top", "bottom"],
        [
            ("left", nz_comps["left"]),
            ("right", nz_comps["right"]),
            ("top", nz_comps["top"]),
            ("bottom", nz_comps["bottom"]),
        ],
    ):
        move_fn = lambda m: move_mask(m, direction)
        indices = torch.stack(
            [pixel_idx[move_fn(neighbor_masks[mask_key])], pixel_idx[neighbor_masks[mask_key]]]
        ).reshape(2, -1)

        values = torch.stack([-nz / step_size, nz / step_size]).flatten()
        matrices.append(create_sparse_derivative(values, indices))

    return tuple(matrices)


@torch.jit.script
def construct_facets_from(mask: torch.Tensor) -> torch.Tensor:
    """
    Constructs mesh facets from a binary mask.

    Args:
        mask: [H,W] binary mask
    Returns:
        [N,5] facet indices where N is number of valid facets
    """
    device = mask.device
    idx = torch.zeros_like(mask, dtype=torch.long, device=device)
    idx[mask] = torch.arange(mask.sum().item(), device=device)

    # Create facet vertex masks [H,W]
    facet_top_left = torch.logical_and.reduce(
        (move_mask(mask, "top"), move_mask(mask, "left"), move_mask(mask, "top_left"), mask)
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
    cg_max_iter=5000,
    cg_rtol=1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs bilateral normal integration.

    Input formats:
        normal_map: [3,H,W] - CHW format
        normal_mask: [H,W]
        depth_map: [H,W] if provided
        depth_mask: [H,W] if provided
    """
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

    device = normal_map.device
    device_type = device.type
    num_normals = normal_mask.sum().item()
    buffers = get_device_buffers(num_normals, device, depth_map)

    # Extract normal components from CHW format
    nx = normal_map[0][normal_mask]  # x-channel
    ny = normal_map[1][normal_mask]  # y-channel
    nz = -normal_map[2][normal_mask]  # z-channel

    if K is not None:
        H, W = normal_mask.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )

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
    A = vstack([A1, A2, A3, A4]).coalesce()
    b = torch.cat([-nx, -nx, -ny, -ny])

    # Initialize optimization
    W = spdiags(
        0.5 * torch.ones(4 * num_normals, device=device), 0, 4 * num_normals, 4 * num_normals
    )
    z = torch.zeros(num_normals, device=device)
    energy = (A @ z - b).T @ W @ (A @ z - b)

    if depth_map is not None:
        m = depth_mask[normal_mask].float()
        M = spdiags(m, 0, num_normals, num_normals)
        z_prior = torch.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    # Optimization loop
    for i in range(max_iter):
        A_mat = (A.t() @ W @ A).coalesce()
        b_vec = A.t() @ W @ b

        if depth_map is not None:
            if buffers["depth_diff"] is not None:
                torch.sub(z_prior, z, out=buffers["depth_diff"])
                depth_diff = M @ buffers["depth_diff"]
                depth_diff[depth_diff == 0] = float("nan")
                offset = torch.nanmean(depth_diff)
                z.add_(offset)
                A_mat = (A_mat + lambda1 * M).coalesce()
                b_vec += lambda1 * M @ z_prior

        D = spdiags(1 / torch.clamp(A_mat.diagonal(), min=1e-5), 0, num_normals, num_normals)
        z = conjugate_gradient_solver(A_mat, b_vec, cg_max_iter, z, D, rtol=cg_rtol)

        W = compute_weights(z, A1, A2, A3, A4, buffers, k, device_type)

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        if torch.abs(energy - energy_old) / energy_old < tol:
            break

    # Reconstruct outputs in [H,W] format
    depth_map_out = torch.full_like(normal_mask, float("nan"), dtype=torch.float32)
    depth_map_out[normal_mask] = z

    if K is not None:
        depth_map_out = torch.exp(depth_map_out)
        vertices = map_depth_map_to_point_clouds(depth_map_out, normal_mask, K=K)
    else:
        vertices = map_depth_map_to_point_clouds(
            depth_map_out, normal_mask, K=None, step_size=step_size
        )

    facets = construct_facets_from(normal_mask)
    if normal_map[2].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]

    # Weight maps in [H,W] format
    wu_map = torch.full_like(normal_mask, float("nan"), dtype=torch.float32)
    wu_map[normal_mask] = buffers["wv"]

    wv_map = torch.full_like(normal_mask, float("nan"), dtype=torch.float32)
    wv_map[normal_mask] = buffers["wu"]

    return depth_map_out, vertices, facets, wu_map, wv_map, energy
