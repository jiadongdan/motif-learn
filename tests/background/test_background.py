import numpy as np
import pytest

import mtflearn.background as background_pkg
from mtflearn.background import estimate_background_baseline
from mtflearn.background import estimate_characteristic_spacing
from mtflearn.background import estimate_background_opening
from mtflearn.background import estimate_background_rolling_ball
from mtflearn.background import remove_background_baseline
from mtflearn.background import remove_background_opening
from mtflearn.background import remove_background_rolling_ball
from mtflearn.background import select_background_parameter
from mtflearn.background import suggest_background_parameters


@pytest.fixture
def synthetic_image():
    image = np.zeros((32, 32), dtype=float)
    image += np.linspace(0.0, 0.5, 32)[None, :]
    image[10, 10] = 2.0
    image[21, 18] = 1.5
    return image


def test_background_subpackage_exports_public_api():
    expected = {
        "estimate_background_opening",
        "remove_background_opening",
        "estimate_background_rolling_ball",
        "remove_background_rolling_ball",
        "estimate_background_baseline",
        "remove_background_baseline",
        "estimate_characteristic_spacing",
        "suggest_background_parameters",
        "select_background_parameter",
    }
    assert expected.issubset(set(dir(background_pkg)))


def test_estimate_background_opening_rejects_non_2d_input():
    with pytest.raises(ValueError, match="image must be a 2D array."):
        estimate_background_opening(np.ones((4, 4, 2)), size=3)


def test_estimate_background_opening_rejects_invalid_size(synthetic_image):
    with pytest.raises(ValueError, match="size must be a positive integer."):
        estimate_background_opening(synthetic_image, size=0)
    with pytest.raises(ValueError, match="size must be a positive int or a length-2 tuple."):
        estimate_background_opening(synthetic_image, size=(3, 0))


def test_remove_background_opening_returns_nonnegative_residual(synthetic_image):
    residual, background = remove_background_opening(synthetic_image, size=7, clip=True)
    assert residual.shape == synthetic_image.shape
    assert background.shape == synthetic_image.shape
    assert np.all(residual >= 0)
    assert np.all(background <= synthetic_image)


def test_estimate_background_rolling_ball_rejects_invalid_radius(synthetic_image):
    with pytest.raises(ValueError, match="radius must be positive."):
        estimate_background_rolling_ball(synthetic_image, radius=0)


def test_remove_background_rolling_ball_returns_nonnegative_residual(synthetic_image):
    residual, background = remove_background_rolling_ball(synthetic_image, radius=6, clip=True)
    assert residual.shape == synthetic_image.shape
    assert background.shape == synthetic_image.shape
    assert np.all(residual >= 0)


def test_estimate_background_baseline_rejects_invalid_input(synthetic_image):
    with pytest.raises(ValueError, match="image must be a 2D array."):
        estimate_background_baseline(np.ones((4, 4, 2)), sigma=3, num_iters=2)
    with pytest.raises(ValueError, match="num_iters must be positive."):
        estimate_background_baseline(synthetic_image, sigma=3, num_iters=0)


def test_remove_background_baseline_returns_nonnegative_residual(synthetic_image):
    residual, background = remove_background_baseline(synthetic_image, sigma=3, num_iters=4, clip=True)
    assert residual.shape == synthetic_image.shape
    assert background.shape == synthetic_image.shape
    assert np.all(residual >= 0)
    assert np.all(background <= synthetic_image + 1e-12)


def test_background_methods_preserve_constant_background():
    image = np.full((16, 16), 0.25, dtype=float)

    residual_open, background_open = remove_background_opening(image, size=5)
    residual_rb, background_rb = remove_background_rolling_ball(image, radius=4)
    residual_base, background_base = remove_background_baseline(image, sigma=2, num_iters=3)

    np.testing.assert_allclose(background_open, image)
    np.testing.assert_allclose(background_rb, image)
    np.testing.assert_allclose(background_base, image)
    np.testing.assert_allclose(residual_open, np.zeros_like(image))
    np.testing.assert_allclose(residual_rb, np.zeros_like(image))
    np.testing.assert_allclose(residual_base, np.zeros_like(image))


def test_suggest_background_parameters_uses_explicit_spacing(synthetic_image):
    params = suggest_background_parameters(synthetic_image, spacing=20)
    assert params["spacing"] == 20.0
    assert params["opening_size"] == 41
    assert params["rolling_ball_radius"] == 60
    assert params["baseline_sigma"] == 30.0


def test_select_background_parameter_returns_method_specific_value(synthetic_image):
    assert select_background_parameter("opening", synthetic_image, spacing=20) == 41
    assert select_background_parameter("rolling_ball", synthetic_image, spacing=20) == 60
    assert select_background_parameter("baseline", synthetic_image, spacing=20) == 30.0


def test_select_background_parameter_rejects_unknown_method(synthetic_image):
    with pytest.raises(ValueError, match="method must be one of"):
        select_background_parameter("unknown", synthetic_image, spacing=20)


def test_estimate_characteristic_spacing_rejects_non_2d_input():
    with pytest.raises(ValueError, match="image must be a 2D array."):
        estimate_characteristic_spacing(np.ones((4, 4, 2)))
