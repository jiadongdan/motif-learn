import pytest
import numpy as np
from ._zmoments import nm2j

def test_scalar_inputs():
    """Test function with valid scalar inputs."""
    assert nm2j(0, 0) == 0
    assert nm2j(1, -1) == 1
    assert nm2j(2, 0) == 4
    assert nm2j(3, 1) == 7
    assert nm2j(4, -4) == 8
    assert nm2j(5, 3) == 13

def test_array_inputs():
    """Test function with valid array inputs."""
    n_values = [0, 1, 2, 2, 3]
    m_values = [0, -1, 0, 2, 3]
    expected_j = [0, 1, 4, 5, 9]
    np.testing.assert_array_equal(nm2j(n_values, m_values), expected_j)

def test_edge_cases():
    """Test function with edge case inputs."""
    # n = 0, m = 0
    assert nm2j(0, 0) == 0
    # n = large value, m = n
    large_n = 1000
    expected_j = ((large_n + 2) * large_n + large_n) // 2
    assert nm2j(large_n, large_n) == expected_j

def test_invalid_n_negative():
    """Test function with negative n, should raise ValueError."""
    with pytest.raises(ValueError, match="Radial order `n` must be non-negative."):
        nm2j(-1, 0)

def test_invalid_m_out_of_range():
    """Test function where abs(m) > n, should raise ValueError."""
    with pytest.raises(ValueError, match="Azimuthal frequency `m` must satisfy \\|m\\| ≤ n."):
        nm2j(2, 3)
    with pytest.raises(ValueError):
        nm2j(0, 1)

def test_invalid_n_minus_m_even():
    """Test function where n - |m| is not even, should raise ValueError."""
    with pytest.raises(ValueError, match="`n - \\|m\\|` must be even."):
        nm2j(1, 0)
    with pytest.raises(ValueError):
        nm2j(3, 2)

def test_mismatched_shapes():
    """Test function with mismatched input shapes, should raise ValueError."""
    with pytest.raises(ValueError, match="`n` and `m` must have the same shape."):
        nm2j([1, 2], [0])

def test_non_integer_inputs():
    """Test function with non-integer inputs."""
    # Float inputs that are equivalent to integers
    assert nm2j(2.0, 0.0) == 4
    # Float inputs that are not integers, should raise ValueError
    with pytest.raises(ValueError):
        nm2j(2.5, 0)
    with pytest.raises(ValueError):
        nm2j(2, 0.5)

def test_large_array_inputs():
    """Test function with large array inputs."""
    n_values = np.arange(0, 100)
    m_values = np.where(n_values % 2 == 0, 0, 1)
    # Ensure that n - |m| is even
    n_values = n_values * 2
    m_values = m_values * 2
    j_values = nm2j(n_values, m_values)
    assert len(j_values) == 100
    # Test a few known values
    assert j_values[0] == 0
    expected_j = ((n_values[1] + 2) * n_values[1] + m_values[1]) // 2
    assert j_values[1] == expected_j

def test_output_type():
    """Test that the output is of the correct type."""
    assert isinstance(nm2j(2, 0), int)
    assert isinstance(nm2j([2], [0]), np.ndarray)

