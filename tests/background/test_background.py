import numpy as np
import pytest

import mtflearn.background as background_pkg
from mtflearn.background import estimate_background_baseline
from mtflearn.background import estimate_characteristic_spacing
from mtflearn.background import estimate_background_opening
from mtflearn.background import estimate_background_rolling_ball
from mtflearn.background import local_variance
from mtflearn.background import remove_background_baseline
from mtflearn.background import remove_background_opening
from mtflearn.background import remove_background_rolling_ball
from mtflearn.background import score_opening_background_local_variance
from mtflearn.background import select_background_parameter
from mtflearn.background import select_opening_size_local_variance
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
        "local_variance",
        "score_opening_background_local_variance",
        "select_opening_size_local_variance",
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


def test_estimate_background_opening_requires_exactly_one_shape_argument(synthetic_image):
    with pytest.raises(ValueError, match="Provide exactly one of size or footprint."):
        estimate_background_opening(synthetic_image)
    with pytest.raises(ValueError, match="Provide exactly one of size or footprint."):
        estimate_background_opening(synthetic_image, size=3, footprint=np.ones((3, 3), dtype=bool))


def test_estimate_background_opening_supports_footprint(synthetic_image):
    footprint = np.ones((3, 3), dtype=bool)
    background = estimate_background_opening(synthetic_image, footprint=footprint)
    assert background.shape == synthetic_image.shape
    assert np.all(background <= synthetic_image)


def test_estimate_background_opening_rejects_invalid_footprint(synthetic_image):
    with pytest.raises(ValueError, match="footprint must be a 2D array."):
        estimate_background_opening(synthetic_image, footprint=np.ones((3, 3, 1), dtype=bool))
    with pytest.raises(ValueError, match="footprint must contain at least one nonzero entry."):
        estimate_background_opening(synthetic_image, footprint=np.zeros((3, 3), dtype=bool))


def test_estimate_background_opening_supports_explicit_square_shape(synthetic_image):
    background = estimate_background_opening(synthetic_image, size=5, shape="square")
    assert background.shape == synthetic_image.shape
    assert np.all(background <= synthetic_image)


def test_estimate_background_opening_supports_explicit_rectangle_shape(synthetic_image):
    background = estimate_background_opening(synthetic_image, size=(3, 5), shape="rectangle")
    assert background.shape == synthetic_image.shape
    assert np.all(background <= synthetic_image)


def test_estimate_background_opening_rejects_invalid_shape_usage(synthetic_image):
    with pytest.raises(ValueError, match="shape must be one of"):
        estimate_background_opening(synthetic_image, size=5, shape="triangle")
    with pytest.raises(ValueError, match="tuple size requires shape='rectangle'"):
        estimate_background_opening(synthetic_image, size=(3, 5), shape="disk")


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


def test_local_variance_rejects_invalid_window_size(synthetic_image):
    with pytest.raises(ValueError, match="window_size must be a positive integer."):
        local_variance(synthetic_image, window_size=0)


def test_local_variance_returns_nonnegative_map(synthetic_image):
    lv = local_variance(synthetic_image, window_size=5)
    assert lv.shape == synthetic_image.shape
    assert np.all(lv >= 0)


def test_score_opening_background_local_variance_returns_scalar(synthetic_image):
    score = score_opening_background_local_variance(
        synthetic_image,
        opening_size=5,
        spacing=6,
        shape="disk",
    )
    assert isinstance(score, float)
    assert score >= 0


def test_select_opening_size_local_variance_uses_smallest_size_below_cutoff(synthetic_image):
    result = select_opening_size_local_variance(
        synthetic_image,
        sizes=[3, 5, 7],
        spacing=6,
        shape="disk",
        relative_threshold=1.0,
    )
    assert result["selected_size"] == 3
    assert result["sizes"] == [3, 5, 7]
    assert len(result["scores"]) == 3


def test_select_opening_size_local_variance_rejects_invalid_sizes(synthetic_image):
    with pytest.raises(ValueError, match="sizes must contain at least one candidate."):
        select_opening_size_local_variance(synthetic_image, sizes=[], spacing=6)
    with pytest.raises(ValueError, match="sizes must contain only positive integers."):
        select_opening_size_local_variance(synthetic_image, sizes=[3, 0], spacing=6)
