import pytest
import torch


def test_move_mask():
    """Test move_mask with various mask patterns"""
    from dbini_torch.dbini import move_mask

    # Test case: Simple 3x3 mask
    mask = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool)

    expected = {
        "left": torch.tensor([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.bool),
        "right": torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.bool),
        "top": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool),
        "bottom": torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool),
        "top_left": torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.bool),
        "top_right": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.bool),
        "bottom_left": torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool),
        "bottom_right": torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool),
    }

    for direction, expected_result in expected.items():
        result = move_mask(mask, direction)
        assert torch.equal(result, expected_result), f"Failed on direction {direction}"

    # Test case: Empty mask
    empty_mask = torch.zeros((4, 4), dtype=torch.bool)
    for direction in ["left", "right", "top", "bottom"]:
        result = move_mask(empty_mask, direction)
        assert torch.equal(result, torch.zeros((4, 4), dtype=torch.bool))

    # Test case: Full mask
    full_mask = torch.ones((3, 3), dtype=torch.bool)
    expected_full = {
        "left": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool),
        "right": torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=torch.bool),
        "top": torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=torch.bool),
        "bottom": torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.bool),
        "top_left": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        "top_right": torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=torch.bool),
        "bottom_left": torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "bottom_right": torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool),
    }

    for direction, expected_result in expected_full.items():
        result = move_mask(full_mask, direction)
        assert torch.equal(result, expected_result), f"Failed on direction {direction}"

    # Test case: Single pixel
    single_pixel = torch.zeros((3, 3), dtype=torch.bool)
    single_pixel[1, 1] = True

    expected_single = {
        "left": torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool),
        "right": torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool),
        "top": torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        "bottom": torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.bool),
        "top_left": torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        "top_right": torch.tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        "bottom_left": torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.bool),
        "bottom_right": torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.bool),
    }

    for direction, expected_result in expected_single.items():
        result = move_mask(single_pixel, direction)
        assert torch.equal(result, expected_result), f"Failed on single pixel {direction}"


def test_move_mask_invalid_inputs():
    """Test move_mask error handling"""
    from dbini_torch.dbini import move_mask

    # Test invalid dimension
    with pytest.raises(ValueError):
        invalid_mask = torch.ones(3)  # Missing channel dimension
        move_mask(invalid_mask, "left")

    # Test invalid direction
    with pytest.raises(ValueError):
        valid_mask = torch.ones(3, 3)
        move_mask(valid_mask, "invalid_direction")


def test_generate_dx_dy():
    """Test generate_dx_dy with various patterns"""
    from dbini_torch.dbini import generate_dx_dy

    # Test case: 2x2 mask with known values
    mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    nz_h = torch.tensor([1.0, 1.0, 1.0, 1.0])
    nz_v = torch.tensor([1.0, 1.0, 1.0, 1.0])

    h_pos, h_neg, v_pos, v_neg = generate_dx_dy(mask, nz_h, nz_v)

    # For a 2x2 mask with all ones, we expect specific sparsity patterns
    assert h_pos.shape == (4, 4)
    assert h_neg.shape == (4, 4)
    assert v_pos.shape == (4, 4)
    assert v_neg.shape == (4, 4)

    # Verify matrices are sparse CSR format
    assert h_pos.layout == torch.sparse_csr
    assert h_neg.layout == torch.sparse_csr
    assert v_pos.layout == torch.sparse_csr
    assert v_neg.layout == torch.sparse_csr

    # Test case: Single pixel
    single_mask = torch.zeros((3, 3), dtype=torch.bool)
    single_mask[1, 1] = True
    single_nz = torch.tensor([1.0])

    results = generate_dx_dy(single_mask, single_nz, single_nz)
    for mat in results:
        assert mat.shape == (1, 1)
        assert mat._nnz() == 0  # No neighbors, so empty matrices
        assert mat.layout == torch.sparse_csr


def test_generate_dx_dy_step_size():
    """Test generate_dx_dy with different step sizes"""
    from dbini_torch.dbini import generate_dx_dy

    mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    nz_h = torch.ones(4)
    nz_v = torch.ones(4)

    # Get baseline with step_size = 1.0
    baseline = generate_dx_dy(mask, nz_h, nz_v, step_size=1.0)

    # Test with step_size = 0.5
    half_step = generate_dx_dy(mask, nz_h, nz_v, step_size=0.5)
    for base, half in zip(baseline, half_step):
        # Values should scale inversely with step size (1/step_size)
        assert torch.allclose(base.values() * 2.0, half.values())

    # Test with step_size = 2.0
    double_step = generate_dx_dy(mask, nz_h, nz_v, step_size=2.0)
    for base, double in zip(baseline, double_step):
        # Values should scale inversely with step size (1/step_size)
        assert torch.allclose(base.values() * 0.5, double.values())


def test_special_patterns():
    """Test generate_dx_dy with special patterns"""
    from dbini_torch.dbini import generate_dx_dy

    patterns = {
        "L-shape": torch.tensor([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool),
        "horizontal_line": torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool),
        "vertical_line": torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.bool),
        "checkerboard": torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
    }

    for name, mask in patterns.items():
        num_true = mask.sum().item()
        nz = torch.ones(num_true)

        h_pos, h_neg, v_pos, v_neg = generate_dx_dy(mask, nz, nz)

        # Verify basic properties
        assert h_pos.shape == (num_true, num_true)
        assert h_neg.shape == (num_true, num_true)
        assert v_pos.shape == (num_true, num_true)
        assert v_neg.shape == (num_true, num_true)

        # Verify no NaN or Inf values
        for mat in [h_pos, h_neg, v_pos, v_neg]:
            assert not torch.any(torch.isnan(mat.to_dense()))
            assert not torch.any(torch.isinf(mat.to_dense()))


def test_create_sparse_derivative():
    """Test create_sparse_derivative with various inputs"""
    from dbini_torch.dbini import create_sparse_derivative

    # Test case: 2x2 mask
    mask = torch.ones(2, 2, dtype=torch.bool)
    nz = torch.ones(4)
    neighbor_mask = mask.clone()
    pixel_idx = torch.zeros_like(mask, dtype=torch.long)
    pixel_idx[mask] = torch.arange(4)

    result = create_sparse_derivative(
        nz, neighbor_mask, pixel_idx, mask, num_pixel=4, step_size=1.0, device=mask.device
    )

    assert result.shape == (4, 4)
    assert not torch.any(torch.isnan(result.to_dense()))
    assert not torch.any(torch.isinf(result.to_dense()))

    # Test different step sizes
    step_sizes = [0.5, 2.0]
    baseline = create_sparse_derivative(
        nz, neighbor_mask, pixel_idx, mask, num_pixel=4, step_size=1.0, device=mask.device
    )

    for step_size in step_sizes:
        result = create_sparse_derivative(
            nz, neighbor_mask, pixel_idx, mask, num_pixel=4, step_size=step_size, device=mask.device
        )
        # Values should scale inversely with step size (1/step_size)
        expected = baseline.values() / step_size
        assert torch.allclose(result.values(), expected)


def test_move_mask_additional_patterns():
    """Test move_mask with additional patterns"""
    from dbini_torch.dbini import move_mask

    patterns = {
        "single_row": torch.tensor([[1, 0, 1]], dtype=torch.bool),
        "single_column": torch.tensor([[1], [0], [1]], dtype=torch.bool),
        "u_shape": torch.tensor([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool),
        # Large sparse/dense patterns
        "large_sparse": torch.bernoulli(torch.ones(10, 10) * 0.1),
        "large_dense": torch.bernoulli(torch.ones(10, 10) * 0.9),
    }

    for name, mask in patterns.items():
        for direction in ["left", "right", "top", "bottom"]:
            result = move_mask(mask, direction)
            assert result.shape == mask.shape
            assert not torch.any(torch.isnan(result))


def test_generate_dx_dy_additional_patterns():
    """Test generate_dx_dy with additional connectivity patterns"""
    from dbini_torch.dbini import generate_dx_dy

    patterns = {
        "isolated_blocks": torch.tensor(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool
        ),
        "border": torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool),
        "spiral": torch.tensor(
            [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.bool
        ),
    }

    for name, mask in patterns.items():
        num_true = mask.sum().item()
        nz = torch.ones(num_true)

        h_pos, h_neg, v_pos, v_neg = generate_dx_dy(mask, nz, nz)

        # Verify basic properties
        for mat in [h_pos, h_neg, v_pos, v_neg]:
            assert mat.shape == (num_true, num_true)
            assert mat.layout == torch.sparse_csr
            assert not torch.any(torch.isnan(mat.to_dense()))
            assert not torch.any(torch.isinf(mat.to_dense()))


if __name__ == "__main__":
    pytest.main([__file__])
