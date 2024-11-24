"""
Tests to verify that PyTorch implementations in dbini_torch match the reference NumPy implementations.
Each test compares outputs between the two implementations to ensure exact parity.
"""

import numpy as np
import pytest
import torch
from xucao_bilateral_normal_integration.bilateral_normal_integration_numpy import (
    generate_dx_dy as np_generate_dx_dy,
)
from xucao_bilateral_normal_integration.bilateral_normal_integration_numpy import (
    move_bottom,
    move_left,
    move_right,
    move_top,
)

from dbini_torch.dbini import generate_dx_dy, move_mask


@pytest.mark.parametrize(
    "direction,np_func",
    [("left", move_left), ("right", move_right), ("top", move_top), ("bottom", move_bottom)],
)
def test_move_mask_parity(direction, np_func):
    """Test move_mask against NumPy reference implementation"""
    test_cases = [
        # Simple 3x3 mask
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        # Empty mask
        np.zeros((4, 4), dtype=bool),
        # Full mask
        np.ones((3, 3), dtype=bool),
        # Random mask
        np.random.randint(0, 2, (5, 5), dtype=bool),
        # Single pixel mask
        np.array([[1]], dtype=bool),
        # Single row mask
        np.array([[1, 0, 1]], dtype=bool),
        # Single column mask
        np.array([[1], [0], [1]], dtype=bool),
        # Checkerboard pattern
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool),
        # U-shaped mask
        np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        # Large random mask
        np.random.randint(0, 2, (10, 10), dtype=bool),
        # Large sparse mask
        np.random.choice([0, 1], size=(10, 10), p=[0.9, 0.1]).astype(bool),
        # Large dense mask
        np.random.choice([0, 1], size=(10, 10), p=[0.1, 0.9]).astype(bool),
        # Edge case: Alternating stripes
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
        # Edge case: Border only
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        # Edge case: Diagonal pattern
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        # Edge case: All True except one False
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),  # Redundant with "Border only"
        # Edge case: All False except one True
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
        # Edge case: Non-square mask
        np.array([[1, 1, 1, 1]], dtype=bool),  # 1x4
        np.array([[1], [1], [1], [1]], dtype=bool),  # 4x1
        # Edge case: Asymmetric mask
        np.array([[1, 0], [0, 0], [1, 1]], dtype=bool),  # 3x2
    ]

    for i, np_mask in enumerate(test_cases):
        pt_mask = torch.from_numpy(np_mask)
        np_result = np_func(np_mask)
        pt_result = move_mask(pt_mask, direction).numpy()

        assert np.array_equal(np_result, pt_result), f"""
        Test case {i} failed for direction {direction}:
        Input: {np_mask}
        NumPy result: {np_result}
        PyTorch result: {pt_result}
        """


def test_generate_dx_dy_parity():
    """Test generate_dx_dy against NumPy reference implementation"""
    test_cases = [
        # Simple 3x3 mask with matching nz values
        (
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([0.5, 0.3, 0.7, 0.2, 0.8]),
            np.array([0.1, 0.4, 0.6, 0.3, 0.9]),
            1.0,
        ),
        # Single pixel mask
        (np.array([[1]], dtype=bool), np.array([1.0]), np.array([1.0]), 1.0),
        # Disconnected regions
        (
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=bool),
            np.array([0.5, 0.3, 0.7, 0.2]),
            np.array([0.1, 0.4, 0.6, 0.3]),
            1.0,
        ),
        # Test different step sizes
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([0.5, 0.3, 0.7, 0.2]),
            np.array([0.1, 0.4, 0.6, 0.3]),
            0.5,
        ),
        # L-shaped pattern
        (np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool), None, None, 1.0),
        # Thin line horizontal
        (np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool), None, None, 1.0),
        # Thin line vertical
        (np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), None, None, 1.0),
        # Checkerboard
        (np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool), None, None, 1.0),
        # Spiral pattern
        (
            np.array([[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]], dtype=bool),
            None,
            None,
            1.0,
        ),
        # Random case
        (np.random.randint(0, 2, (5, 5), dtype=bool), None, None, 1.0),
        # Edge case: Very large step size
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            1000.0,
        ),
        # Edge case: Very small step size
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            1e-6,
        ),
        # Edge case: All zeros in nz_horizontal/vertical
        (np.array([[1, 1], [1, 1]], dtype=bool), np.zeros(4), np.zeros(4), 1.0),
        # Edge case: All ones in nz_horizontal/vertical
        (np.array([[1, 1], [1, 1]], dtype=bool), np.ones(4), np.ones(4), 1.0),
        # Edge case: Non-square mask
        (
            np.array([[1, 1, 1]], dtype=bool),  # 1x3
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            1.0,
        ),
        # Edge case: Mixed positive/negative nz values
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([-0.5, 0.5, -0.5, 0.5]),
            np.array([0.5, -0.5, 0.5, -0.5]),
            1.0,
        ),
    ]

    for i, (np_mask, nz_h, nz_v, step_size) in enumerate(test_cases):
        # Fill in random values for None entries
        if nz_h is None or nz_v is None:
            num_true = np.sum(np_mask)
            nz_h = np.random.rand(num_true)
            nz_v = np.random.rand(num_true)

        # Convert to PyTorch tensors
        pt_mask = torch.from_numpy(np_mask)
        pt_nz_h = torch.from_numpy(nz_h)
        pt_nz_v = torch.from_numpy(nz_v)

        # Get results from both implementations
        np_result = np_generate_dx_dy(np_mask, nz_h, nz_v, step_size)
        pt_result = generate_dx_dy(pt_mask, pt_nz_h, pt_nz_v, step_size)

        # Compare results
        for np_mat, pt_mat in zip(np_result, pt_result):
            assert np.allclose(np_mat.todense(), pt_mat.to_dense().numpy(), rtol=1e-5), f"""
            Test case {i} failed:
            NumPy result:
            {np_mat.todense()}
            PyTorch result:
            {pt_mat.to_dense().numpy()}
            """


def array_equal_nan_inf(a, b):
    """Compare arrays that may contain NaN values"""
    if a.shape != b.shape:
        return False

    equal = np.equal(a, b) | (np.isnan(a) & np.isnan(b))
    return np.all(equal)


@pytest.mark.parametrize(
    "test_case",
    [
        # Zero-sized arrays
        (np.zeros((0, 0), dtype=bool), np.array([]), np.array([])),
        (np.zeros((0, 1), dtype=bool), np.array([]), np.array([])),
        (np.zeros((1, 0), dtype=bool), np.array([]), np.array([])),
        # NaN/Inf values in nz arrays
        (
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([np.nan, np.inf, -np.inf, 1.0]),
            np.array([1.0, np.nan, np.inf, -np.inf]),
        ),
    ],
)
def test_generate_dx_dy_special_values(test_case):
    """Test both implementations handle special values identically"""
    mask, nz_h, nz_v = test_case

    # NumPy implementation
    np_result = np_generate_dx_dy(mask, nz_h, nz_v)

    # PyTorch implementation
    pt_mask = torch.from_numpy(mask)
    pt_nz_h = torch.from_numpy(nz_h)
    pt_nz_v = torch.from_numpy(nz_v)
    pt_result = generate_dx_dy(pt_mask, pt_nz_h, pt_nz_v)

    # Compare results exactly, handling NaN values
    for np_mat, pt_mat in zip(np_result, pt_result):
        assert array_equal_nan_inf(np_mat.todense(), pt_mat.to_dense().numpy()), f"""
        Special values test failed:
        Input mask: {mask}
        Input nz_h: {nz_h}
        Input nz_v: {nz_v}
        NumPy result:
        {np_mat.todense()}
        PyTorch result:
        {pt_mat.to_dense().numpy()}
        """

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bilateral_normal_integration_parity(device):
    """Test bilateral_normal_integration against NumPy reference implementation"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    import os

    import cv2
    from xucao_bilateral_normal_integration.bilateral_normal_integration_numpy import (
        bilateral_normal_integration as np_bni,
    )

    from dbini_torch.dbini import bilateral_normal_integration as pt_bni

    # Test parameters
    k = 2.0
    max_iter = 150
    tol = 1e-4

    # Load test data
    data_dir = os.path.join(os.path.dirname(__file__), "data/person")

    # Load inputs
    normal_map = cv2.cvtColor(
        cv2.imread(os.path.join(data_dir, "normal_map.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_RGB2BGR,
    )
    if normal_map.dtype == np.dtype(np.uint16):
        normal_map = normal_map / 65535 * 2 - 1
    else:
        normal_map = normal_map / 255 * 2 - 1

    mask = cv2.imread(os.path.join(data_dir, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)

    # Convert inputs to PyTorch tensors
    pt_normal_map = torch.from_numpy(normal_map).permute(2, 0, 1).to(device)
    pt_mask = torch.from_numpy(mask).to(device)

    # Run PyTorch implementation
    pt_depth_map, pt_surface, pt_wu_map, pt_wv_map, pt_energy_list = pt_bni(
        normal_map=pt_normal_map, normal_mask=pt_mask, k=k, max_iter=max_iter, tol=tol
    )

    # Run NumPy implementation
    np_depth_map, np_surface, np_wu_map, np_wv_map, np_energy_list = np_bni(
        normal_map=normal_map, normal_mask=mask, k=k, max_iter=max_iter, tol=tol
    )

    # Convert PyTorch outputs to NumPy for comparison
    pt_depth_map = pt_depth_map.cpu().squeeze().numpy()
    pt_wu_map = pt_wu_map.cpu().squeeze().numpy()
    pt_wv_map = pt_wv_map.cpu().squeeze().numpy()
    pt_energy_list = np.array(pt_energy_list)

    # Compare results
    assert np.allclose(np_depth_map, pt_depth_map, rtol=1e-5, equal_nan=True), "Depth map mismatch"
    assert np.allclose(np_energy_list, pt_energy_list, rtol=1e-5, equal_nan=True), "Energy list mismatch"
    assert np.allclose(np_wu_map, pt_wu_map, rtol=1e-5, equal_nan=True), "Wu map mismatch"
    assert np.allclose(np_wv_map, pt_wv_map, rtol=1e-5, equal_nan=True), "Wv map mismatch"
    assert np.allclose(np_surface[0], pt_surface[0].cpu().numpy(), rtol=1e-5, equal_nan=True), "vertices mismatch"
    assert np.allclose(np_surface[1], pt_surface[1].cpu().numpy(), rtol=1e-5, equal_nan=True), "faces mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
